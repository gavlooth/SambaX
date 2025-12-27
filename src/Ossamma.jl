module Ossamma

"""
Ossamma -> Oscillatory State Space Attention Masked Mixer Architecture

Architecture:
- Input → LayerNorm (time-conditioned with scale, shift, α_bias)
- Two parallel branches:
  1. Global-Spectral GLU: Dense(dim→2*dim) → split → LinearAttn(content) ⊙ sigmoid(OscSSM(gate)) → Dense
  2. Local-Sharp: Windowed Softmax Attention (SWAttention)
- Mix: α·GLU + (1-α)·Local where α = σ(f(x) + α_bias(t))
- Residual + FFN
"""

include("Dlinoss.jl")
include("DlinossParallel.jl")
include("Attention.jl")
include("linearAttention.jl")
include("ossm.jl")

using .Attention: SWAttention
using .Dlinoss: DLinOSS
using .DlinossParallel: DLinOSSParallel
using .ossm: OSSMLayer as OscSSM

# Import struct from LinearAttention module
using .LinearAttention: LinearAttentionLayer

using Lux
using Random
using NNlib
using Statistics: mean

const LuxLayer =
    isdefined(Lux, :AbstractExplicitLayer) ? Lux.AbstractExplicitLayer :
    Lux.AbstractLuxLayer

# ============================================================================
# Time-Conditioned LayerNorm
# ============================================================================
struct TimeConditionedLayerNorm <: LuxLayer
    embedding_dimension::Int
    time_dimension::Int
    LayerNorm::Lux.LayerNorm
    ScaleProjection::Lux.Dense    # t → scale
    ShiftProjection::Lux.Dense    # t → shift
    AlphaBiasProjection::Lux.Dense # t → α_bias for mixing
end

function TimeConditionedLayerNorm(embedding_dimension::Int, time_dimension::Int)
    return TimeConditionedLayerNorm(
        embedding_dimension,
        time_dimension,
        Lux.LayerNorm((embedding_dimension,)),
        Lux.Dense(time_dimension => embedding_dimension),
        Lux.Dense(time_dimension => embedding_dimension),
        Lux.Dense(time_dimension => 1),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::TimeConditionedLayerNorm)
    return (
        LayerNorm = Lux.initialparameters(rng, layer.LayerNorm),
        ScaleProjection = Lux.initialparameters(rng, layer.ScaleProjection),
        ShiftProjection = Lux.initialparameters(rng, layer.ShiftProjection),
        AlphaBiasProjection = Lux.initialparameters(rng, layer.AlphaBiasProjection),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::TimeConditionedLayerNorm)
    return (
        LayerNorm = Lux.initialstates(rng, layer.LayerNorm),
        ScaleProjection = Lux.initialstates(rng, layer.ScaleProjection),
        ShiftProjection = Lux.initialstates(rng, layer.ShiftProjection),
        AlphaBiasProjection = Lux.initialstates(rng, layer.AlphaBiasProjection),
    )
end

function (layer::TimeConditionedLayerNorm)(input_tensor, time_input, params, state)
    # input_tensor: (embedding_dim, seq_len, batch) or (embedding_dim, seq_len)
    # time_input: (time_dim, batch) or (time_dim,)

    # Apply base LayerNorm - flatten to 2D first, then reshape back
    is_batched = ndims(input_tensor) == 3
    original_size = size(input_tensor)
    input_flattened = reshape(input_tensor, layer.embedding_dimension, :)
    normalized_flat, ln_state = layer.LayerNorm(input_flattened, params.LayerNorm, state.LayerNorm)
    normalized = reshape(normalized_flat, original_size)

    # Compute time-conditioned scale and shift
    scale_raw, scale_state = layer.ScaleProjection(time_input, params.ScaleProjection, state.ScaleProjection)
    shift, shift_state = layer.ShiftProjection(time_input, params.ShiftProjection, state.ShiftProjection)
    alpha_bias, alpha_state = layer.AlphaBiasProjection(time_input, params.AlphaBiasProjection, state.AlphaBiasProjection)

    scale = 1.0f0 .+ scale_raw  # Center around 1

    # Broadcast scale and shift: (embedding_dim, batch) → (embedding_dim, 1, batch)
    if is_batched
        scale_broadcast = reshape(scale, size(scale, 1), 1, size(scale, 2))
        shift_broadcast = reshape(shift, size(shift, 1), 1, size(shift, 2))
    else
        scale_broadcast = reshape(scale, :, 1)
        shift_broadcast = reshape(shift, :, 1)
    end

    output = normalized .* scale_broadcast .+ shift_broadcast

    new_state = (
        LayerNorm = ln_state,
        ScaleProjection = scale_state,
        ShiftProjection = shift_state,
        AlphaBiasProjection = alpha_state,
    )

    return output, alpha_bias, new_state
end

# ============================================================================
# SwiGLU Feed-Forward Network
# ============================================================================

"""
    SwiGLU - Swish-Gated Linear Unit FFN layer.

Implements the SwiGLU variant from "GLU Variants Improve Transformer" (Shazeer, 2020).

Structure: Dense(d → 2h) → split → SiLU(a) ⊙ b → Dense(h → d)

Where h = expansion_factor × d (default 3/2).

# Arguments
- `dim::Int`: Input/output dimension
- `expansion_factor::Float32`: Hidden dimension multiplier (default 3/2)
"""
struct SwiGLU <: LuxLayer
    in_dim::Int
    hidden_dim::Int
    Expand::Lux.Dense
    Contract::Lux.Dense
end

function SwiGLU(dim::Int; expansion_factor::Float32 = 3f0 / 2f0)
    hidden = round(Int, dim * expansion_factor)
    # Ensure hidden is even for clean split
    hidden = hidden + (hidden % 2)

    SwiGLU(
        dim,
        hidden,
        Lux.Dense(dim => hidden),           # d → 3d/2
        Lux.Dense(hidden ÷ 2 => dim)        # 3d/4 → d
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, ffn::SwiGLU)
    return (
        Expand = Lux.initialparameters(rng, ffn.Expand),
        Contract = Lux.initialparameters(rng, ffn.Contract),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, ffn::SwiGLU)
    return (
        Expand = Lux.initialstates(rng, ffn.Expand),
        Contract = Lux.initialstates(rng, ffn.Contract),
    )
end

function (ffn::SwiGLU)(x, ps, st)
    # Expand: d → 2h
    expanded, st_expand = ffn.Expand(x, ps.Expand, st.Expand)

    # Split into two halves
    half = ffn.hidden_dim ÷ 2
    a = selectdim(expanded, 1, 1:half)
    b = selectdim(expanded, 1, (half+1):ffn.hidden_dim)

    # SwiGLU: Swish(a) ⊙ b  (swish = x * sigmoid(x) = SiLU)
    gated = NNlib.swish.(a) .* b

    # Contract: h → d
    output, st_contract = ffn.Contract(gated, ps.Contract, st.Contract)

    return output, (Expand = st_expand, Contract = st_contract)
end

# ============================================================================
# Main Ossamma Block
# ============================================================================
struct OssammaBlock{OSC <: Lux.AbstractLuxLayer} <: LuxLayer
    embedding_dimension::Int
    sequence_length::Int
    number_of_heads::Int
    time_dimension::Int
    state_dimension::Int
    dropout_rate::Float32
    use_ffn::Bool
    use_glu_output_projection::Bool    # Optional d→d Dense after GLU (disabled by default)
    use_parallel_scan::Bool            # GPU optimization: parallel associative scan

    # Time-conditioned normalization
    InputNorm::TimeConditionedLayerNorm

    # GLU branch: Dense → split → LinearAttn(content) ⊙ sigmoid(OscSSM(gate))
    GluProjection::Lux.Dense           # dim → 2*dim
    LinearAttention::LinearAttentionLayer
    OscillatorLayer::OSC               # DLinOSS or DLinOSSParallel
    GluOutputProjection::Union{Lux.Dense, Nothing}  # Optional d→d (ablation)

    # Local branch: InputGate + Windowed Softmax Attention
    InputGate::Lux.Dense               # dim → dim (no bias, sigmoid activation)
    SlidingWindowAttention::SWAttention

    # Mixing: α projection from input
    AlphaProjection::Lux.Dense         # dim → 1

    # Dropout after mixing
    AttentionDropout::LuxLayer

    # SwiGLU FFN (expressiveness) + Output normalization
    FFN::Union{SwiGLU, Nothing}
    OutputNorm::Lux.LayerNorm
end

function OssammaBlock(
    embedding_dimension::Int,
    sequence_length::Int,
    number_of_heads::Int,
    time_dimension::Int;
    state_dimension::Int = embedding_dimension,
    window_size::Int = 256,
    min_frequency::Float32 = 0.1f0,
    max_frequency::Float32 = 10.0f0,
    default_time_step::Float32 = 0.1f0,
    dropout_rate::Float32 = 0.1f0,
    use_ffn::Bool = true,
    ffn_expansion::Float32 = 3f0 / 2f0,
    use_glu_output_projection::Bool = false,  # Optional d→d Dense after GLU
    use_parallel_scan::Bool = false,          # GPU optimization
    parallel_chunk_size::Int = 64,
)
    # Choose oscillator implementation based on parallelization setting
    oscillator_layer = if use_parallel_scan
        DLinOSSParallel(
            embedding_dimension, state_dimension, embedding_dimension,
            min_frequency, max_frequency, default_time_step;
            chunk_size = parallel_chunk_size
        )
    else
        DLinOSS(embedding_dimension, state_dimension, embedding_dimension,
                min_frequency, max_frequency, default_time_step)
    end

    return OssammaBlock(
        embedding_dimension,
        sequence_length,
        number_of_heads,
        time_dimension,
        state_dimension,
        dropout_rate,
        use_ffn,
        use_glu_output_projection,
        use_parallel_scan,
        # Time-conditioned LayerNorm
        TimeConditionedLayerNorm(embedding_dimension, time_dimension),
        # GLU branch
        Lux.Dense(embedding_dimension => 2 * embedding_dimension),
        LinearAttentionLayer(embedding_dimension, sequence_length, number_of_heads, time_dimension),
        oscillator_layer,
        use_glu_output_projection ? Lux.Dense(embedding_dimension => embedding_dimension) : nothing,
        # Local branch: InputGate + SlidingWindowAttention
        Lux.Dense(embedding_dimension => embedding_dimension; use_bias = false),
        SWAttention(sequence_length, embedding_dimension, number_of_heads; window_size = window_size),
        # Alpha mixing projection
        Lux.Dense(embedding_dimension => 1),
        # Dropout after mixing
        Lux.Dropout(dropout_rate),
        # SwiGLU FFN + Output normalization
        use_ffn ? SwiGLU(embedding_dimension; expansion_factor = ffn_expansion) : nothing,
        Lux.LayerNorm((embedding_dimension,)),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::OssammaBlock)
    ffn_params = if layer.use_ffn && layer.FFN !== nothing
        Lux.initialparameters(rng, layer.FFN)
    else
        NamedTuple()
    end

    glu_out_params = if layer.use_glu_output_projection && layer.GluOutputProjection !== nothing
        Lux.initialparameters(rng, layer.GluOutputProjection)
    else
        NamedTuple()
    end

    # Initialize gate weights with small values (std=0.02) for gates starting near 0.5
    input_gate_params = Lux.initialparameters(rng, layer.InputGate)
    if haskey(input_gate_params, :weight)
        input_gate_params = (weight = input_gate_params.weight .* 0.02f0,)
    end

    return (
        InputNorm = Lux.initialparameters(rng, layer.InputNorm),
        GluProjection = Lux.initialparameters(rng, layer.GluProjection),
        LinearAttention = Lux.initialparameters(rng, layer.LinearAttention),
        OscillatorLayer = Lux.initialparameters(rng, layer.OscillatorLayer),
        GluOutputProjection = glu_out_params,
        InputGate = input_gate_params,
        SlidingWindowAttention = Lux.initialparameters(rng, layer.SlidingWindowAttention),
        AlphaProjection = Lux.initialparameters(rng, layer.AlphaProjection),
        AttentionDropout = Lux.initialparameters(rng, layer.AttentionDropout),
        FFN = ffn_params,
        OutputNorm = Lux.initialparameters(rng, layer.OutputNorm),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::OssammaBlock)
    ffn_state = if layer.use_ffn && layer.FFN !== nothing
        Lux.initialstates(rng, layer.FFN)
    else
        NamedTuple()
    end

    glu_out_state = if layer.use_glu_output_projection && layer.GluOutputProjection !== nothing
        Lux.initialstates(rng, layer.GluOutputProjection)
    else
        NamedTuple()
    end

    return (
        InputNorm = Lux.initialstates(rng, layer.InputNorm),
        GluProjection = Lux.initialstates(rng, layer.GluProjection),
        LinearAttention = Lux.initialstates(rng, layer.LinearAttention),
        OscillatorLayer = Lux.initialstates(rng, layer.OscillatorLayer),
        GluOutputProjection = glu_out_state,
        InputGate = Lux.initialstates(rng, layer.InputGate),
        SlidingWindowAttention = Lux.initialstates(rng, layer.SlidingWindowAttention),
        AlphaProjection = Lux.initialstates(rng, layer.AlphaProjection),
        AttentionDropout = Lux.initialstates(rng, layer.AttentionDropout),
        FFN = ffn_state,
        OutputNorm = Lux.initialstates(rng, layer.OutputNorm),
    )
end

function (block::OssammaBlock)(inputs::Tuple, params, state)
    input_tensor, time_input = inputs
    # input_tensor: (embedding_dim, seq_len, batch) or (embedding_dim, seq_len)
    # time_input: (time_dim, batch) or (time_dim,)

    residual = input_tensor

    # =========================================================================
    # 1. Time-Conditioned LayerNorm
    # =========================================================================
    normalized, alpha_bias, norm_state = block.InputNorm(
        input_tensor, time_input, params.InputNorm, state.InputNorm
    )

    # =========================================================================
    # 2. Global-Spectral GLU Branch
    # =========================================================================
    # Project to 2*dim and split
    glu_projected, glu_proj_state = block.GluProjection(
        normalized, params.GluProjection, state.GluProjection
    )

    # Split into content and gate halves (use copy to avoid GPU scalar indexing)
    dim = block.embedding_dimension
    content_half = copy(selectdim(glu_projected, 1, 1:dim))
    gate_half = copy(selectdim(glu_projected, 1, (dim+1):size(glu_projected, 1)))

    # Content → Linear Attention
    content_output, lin_attn_state = block.LinearAttention(
        (content_half, time_input), params.LinearAttention, state.LinearAttention
    )

    # Gate → Oscillator SSM → sigmoid
    gate_output, osc_state = block.OscillatorLayer(
        gate_half, params.OscillatorLayer, state.OscillatorLayer
    )
    gate_activated = NNlib.sigmoid.(gate_output)

    # GLU: content ⊙ gate
    glu_output = content_output .* gate_activated

    # Optional: d→d projection after GLU (ablation experiment)
    glu_output, glu_out_state = if block.use_glu_output_projection && block.GluOutputProjection !== nothing
        block.GluOutputProjection(glu_output, params.GluOutputProjection, state.GluOutputProjection)
    else
        glu_output, NamedTuple()
    end

    # =========================================================================
    # 3. Local-Sharp Branch with Input Gating
    # =========================================================================
    # Step 3a: Input Gate - GLU controls what features Local should attend to
    input_gate_logits, input_gate_state = block.InputGate(
        glu_output, params.InputGate, state.InputGate
    )
    input_gate = NNlib.sigmoid.(input_gate_logits)

    # gated_x = x_norm * input_gate (suppresses irrelevant features)
    gated_x = normalized .* input_gate

    # Step 3b: Sliding Window Attention on gated input
    local_output, sw_attn_state = block.SlidingWindowAttention(
        gated_x, params.SlidingWindowAttention, state.SlidingWindowAttention
    )

    # =========================================================================
    # 4. Adaptive Mixing: α·GLU + (1-α)·Local
    # =========================================================================
    # Compute α from input (mean-pooled over sequence)
    is_batched = ndims(input_tensor) == 3
    seq_dim = 2

    # Mean pool over sequence dimension
    input_pooled = dropdims(mean(normalized, dims = seq_dim), dims = seq_dim)

    alpha_logits, alpha_state = block.AlphaProjection(
        input_pooled, params.AlphaProjection, state.AlphaProjection
    )

    # Add time-conditioned bias and apply sigmoid
    alpha = NNlib.sigmoid.(alpha_logits .+ alpha_bias)

    # Broadcast alpha for mixing: (1, batch) → (1, 1, batch)
    if is_batched
        alpha_broadcast = reshape(alpha, 1, 1, size(alpha, 2))
    else
        alpha_broadcast = reshape(alpha, 1, 1)
    end

    # Mix outputs
    mixed_output = alpha_broadcast .* glu_output .+ (1.0f0 .- alpha_broadcast) .* local_output

    # =========================================================================
    # 5. Dropout
    # =========================================================================
    mixed_output, attn_dropout_state = block.AttentionDropout(
        mixed_output, params.AttentionDropout, state.AttentionDropout
    )

    # =========================================================================
    # 6. SwiGLU FFN (expressiveness / non-linearity)
    # =========================================================================
    mixed_output, ffn_state = if block.use_ffn && block.FFN !== nothing
        block.FFN(mixed_output, params.FFN, state.FFN)
    else
        mixed_output, NamedTuple()
    end

    # =========================================================================
    # 7. Residual + Output LayerNorm
    # =========================================================================
    output_pre_norm = residual .+ mixed_output

    output_flat = reshape(output_pre_norm, block.embedding_dimension, :)
    output_norm_flat, output_norm_state = block.OutputNorm(
        output_flat, params.OutputNorm, state.OutputNorm
    )
    output = reshape(output_norm_flat, size(output_pre_norm))

    # =========================================================================
    # 8. Update State
    # =========================================================================
    new_state = (
        InputNorm = norm_state,
        GluProjection = glu_proj_state,
        LinearAttention = lin_attn_state,
        OscillatorLayer = osc_state,
        GluOutputProjection = glu_out_state,
        InputGate = input_gate_state,
        SlidingWindowAttention = sw_attn_state,
        AlphaProjection = alpha_state,
        AttentionDropout = attn_dropout_state,
        FFN = ffn_state,
        OutputNorm = output_norm_state,
    )

    return output, new_state
end

# ============================================================================
# OssammaNERBlock - Alias for backward compatibility
# ============================================================================
"""
OssammaNERBlock is now an alias for OssammaBlock.
Both blocks have identical architecture - the only difference in usage
is the classification head added by the NER model.
"""
const OssammaNERBlock = OssammaBlock

# ============================================================================
# LLaDA Text Diffusion Model
# ============================================================================
include("LLaDA.jl")
using .LLaDA: LLaDAModel, TimeMLPEmbedding, SinusoidalTimeEmbedding
using .LLaDA: LLaDAConfig, load_config, save_config, config_from_dict
using .LLaDA: default_config, small_config, base_config, large_config, production_config
using .LLaDA: apply_mask, sample_mask_ratio, unmask_step, generate

# ============================================================================
# Training Utilities
# ============================================================================
include("Training.jl")
using .Training: masked_cross_entropy, diffusion_loss
using .Training: TrainState, create_train_state, train_step!
using .Training: warmup_cosine_schedule, evaluate, compute_accuracy
using .Training: TrainingConfig, load_training_config, train!
using .Training: load_checkpoint, save_checkpoint

# ============================================================================
# Classification Model
# ============================================================================
include("Classification.jl")
using .Classification: OssammaClassifier, ClassifierConfig
using .Classification: SequencePooling, FixedTimeEmbedding
using .Classification: tiny_classifier, small_classifier, base_classifier
using .Classification: load_pretrained_encoder

# ============================================================================
# NER Model (Token-level classification)
# ============================================================================
include("NER.jl")
using .NER: OssammaNER, NERConfig
using .NER: tiny_ner, small_ner, base_ner
using .NER: ner_cross_entropy, predict_labels, extract_entities
using .NER: RAG_LABELS, ENTITY_TYPES, LABEL_TO_ID, ID_TO_LABEL, NUM_LABELS
using .NER: load_ner_config, estimate_parameters, print_config_summary
using .NER: load_training_config as load_ner_training_config

# ============================================================================
# CRF Layer (Conditional Random Field for sequence labeling)
# ============================================================================
include("CRF.jl")
using .CRF: LinearChainCRF, CRFTagger
using .CRF: crf_loss, viterbi_decode
using .CRF: is_valid_transition, build_transition_mask

# ============================================================================
# Drafter Model (TiDAR-style diffusion drafter for LLM generation)
# ============================================================================
include("Drafter.jl")
using .Drafter: OssammaDrafterBlock, OssammaDrafter
using .Drafter: GRANITE_VOCAB_SIZE, QWEN3_VOCAB_SIZE, LLAMA3_VOCAB_SIZE
using .Drafter: DrafterConfig, load_drafter_config, default_granite_config, default_qwen3_config

# ============================================================================
# Drafter Training Utilities
# ============================================================================
include("DrafterTraining.jl")
using .DrafterTraining: drafter_mlm_loss, distillation_loss, combined_drafter_loss
using .DrafterTraining: save_drafter_checkpoint, load_drafter_checkpoint
using .DrafterTraining: DrafterTrainingConfig, create_drafter_training_state
using .DrafterTraining: apply_random_mask, apply_block_mask, apply_suffix_mask

# ============================================================================
# Deep Scaling Strategies (48-96+ layers)
# ============================================================================
include("DeepScaling.jl")
using .DeepScaling: HierarchicalFrequencyConfig, compute_layer_frequencies, frequency_summary
using .DeepScaling: LayerScaleConfig, apply_layer_scale, init_layer_scale
using .DeepScaling: StochasticDepthConfig, should_drop_layer, layer_drop_rate, drop_path
using .DeepScaling: CheckpointConfig, should_checkpoint
using .DeepScaling: OssammaBlockDeep
using .DeepScaling: DeepModelConfig, create_deep_blocks, print_model_summary
using .DeepScaling: deep_48L_config, ultra_96L_config, long_context_config
using .DeepScaling: BlockTypeSchedule, UNIFORM, PROGRESSIVE, SANDWICH, ALTERNATING, get_block_type

# ============================================================================
# TiDAR - Speculative Decoding with Granite Models
# ============================================================================
include("TiDAR.jl")
using .TiDAR: OssammaDrafterDeep, OssammaDrafterBlockDeep
using .TiDAR: TiDARConfig
using .TiDAR: granite_2b_drafter_config, granite_3b_drafter_config, granite_4_3b_drafter_config, granite_8b_drafter_config
using .TiDAR: granite_drafter_deep_config, granite_3b_drafter_deep_config
using .TiDAR: GRANITE_VOCAB_SIZE, GRANITE_MASK_TOKEN_ID
using .TiDAR: GRANITE_3B_VOCAB_SIZE, GRANITE_3B_MASK_TOKEN_ID  # Legacy
using .TiDAR: draft_tokens, verify_and_accept, tidar_generate_step, tidar_generate_step_cached
using .TiDAR: estimate_drafter_params, print_tidar_config

# ============================================================================
# Logic-Gated Routing (MoE scaffolding)
# ============================================================================
include("LogicGated.jl")
using .LogicGated: TokenRouter, top1_expert, build_spans
using .LogicGated: heuristic_labels, heuristic_labels_batch
using .LogicGated: router_supervision_loss, router_accuracy
using .LogicGated: EXPERT_LOGIC, EXPERT_LANGUAGE, EXPERT_MATH, EXPERT_MEMORY, EXPERT_NAMES

# ============================================================================
# MoE Transformer (Logic-gated)
# ============================================================================
include("MoET.jl")
using .MoET: MoETConfig, ExpertTower, MoETModel

# ============================================================================
# HuggingFace Tokenizer (optional - requires PyCall)
# ============================================================================
include("HFTokenizer.jl")
# Note: HFTokenizer requires PyCall and transformers Python package
# Usage: using Ossamma.HFTokenizer

# ============================================================================
# Data Processing Modules
# ============================================================================
include("data/NERDataset.jl")
include("data/Tokenizer.jl")
include("data/Augmentation.jl")

# ============================================================================
# Evaluation Metrics
# ============================================================================
include("evaluation/NERMetrics.jl")

# ============================================================================
# Serving Modules
# ============================================================================
include("serve/Monitoring.jl")
include("serve/InferenceServer.jl")

# ============================================================================
# Exports
# ============================================================================

# Re-export submodules for callers who want direct access.
export Dlinoss, DlinossParallel, Attention, LinearAttention, ossm, LLaDA, Classification, NER
export CRF, NERDataset, Tokenizer, Augmentation, NERMetrics
export Monitoring, InferenceServer

# Main block
export OssammaBlock, OssammaNERBlock, TimeConditionedLayerNorm

# LLaDA model and utilities
export LLaDAModel, TimeMLPEmbedding, SinusoidalTimeEmbedding
export LLaDAConfig, load_config, save_config, config_from_dict
export default_config, small_config, base_config, large_config, production_config
export apply_mask, sample_mask_ratio, unmask_step, generate

# Training utilities
export masked_cross_entropy, diffusion_loss
export TrainState, create_train_state, train_step!
export warmup_cosine_schedule, evaluate, compute_accuracy
export TrainingConfig, load_training_config, train!
export load_checkpoint, save_checkpoint

# Provide conventional aliases for the main layer types.
export DLinOSS, DLinOSSParallel, SWAttention, OscSSM

# Classification model
export OssammaClassifier, ClassifierConfig
export SequencePooling, FixedTimeEmbedding
export tiny_classifier, small_classifier, base_classifier
export load_pretrained_encoder

# NER model
export OssammaNER, NERConfig
export tiny_ner, small_ner, base_ner
export ner_cross_entropy, predict_labels, extract_entities
export RAG_LABELS, ENTITY_TYPES, LABEL_TO_ID, ID_TO_LABEL, NUM_LABELS
export load_ner_config, estimate_parameters, print_config_summary
export load_ner_training_config

# CRF layer
export LinearChainCRF, CRFTagger
export crf_loss, viterbi_decode

# Drafter model (TiDAR-style)
export Drafter
export OssammaDrafterBlock, OssammaDrafter
export GRANITE_VOCAB_SIZE, QWEN3_VOCAB_SIZE, LLAMA3_VOCAB_SIZE
export DrafterConfig, load_drafter_config, default_granite_config, default_qwen3_config

# Drafter training utilities
export DrafterTraining, HFTokenizer
export drafter_mlm_loss, distillation_loss, combined_drafter_loss
export save_drafter_checkpoint, load_drafter_checkpoint
export DrafterTrainingConfig, create_drafter_training_state
export apply_random_mask, apply_block_mask, apply_suffix_mask

# CRF utilities
export is_valid_transition, build_transition_mask

# Deep Scaling (48-96+ layers)
export DeepScaling
export HierarchicalFrequencyConfig, compute_layer_frequencies, frequency_summary
export LayerScaleConfig, apply_layer_scale, init_layer_scale
export StochasticDepthConfig, should_drop_layer, layer_drop_rate, drop_path
export CheckpointConfig, should_checkpoint
export OssammaBlockDeep
export DeepModelConfig, create_deep_blocks, print_model_summary
export deep_48L_config, ultra_96L_config, long_context_config
export BlockTypeSchedule, UNIFORM, PROGRESSIVE, SANDWICH, ALTERNATING, get_block_type

# TiDAR (Speculative Decoding with Granite Models)
export TiDAR
export OssammaDrafterDeep, OssammaDrafterBlockDeep
export TiDARConfig
export granite_2b_drafter_config, granite_3b_drafter_config, granite_4_3b_drafter_config, granite_8b_drafter_config
export granite_drafter_deep_config, granite_3b_drafter_deep_config
export GRANITE_VOCAB_SIZE, GRANITE_MASK_TOKEN_ID
export GRANITE_3B_VOCAB_SIZE, GRANITE_3B_MASK_TOKEN_ID  # Legacy
export draft_tokens, verify_and_accept, tidar_generate_step, tidar_generate_step_cached
export estimate_drafter_params, print_tidar_config

# Logic-gated routing exports
export LogicGated
export TokenRouter, top1_expert, build_spans
export heuristic_labels, heuristic_labels_batch
export router_supervision_loss, router_accuracy
export EXPERT_LOGIC, EXPERT_LANGUAGE, EXPERT_MATH, EXPERT_MEMORY, EXPERT_NAMES

# MoE Transformer exports
export MoET
export MoETConfig, ExpertTower, MoETModel

end
