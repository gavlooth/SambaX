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
include("Attention.jl")
include("linearAttention.jl")
include("ossm.jl")

using .Attention: SWAttention
using .Dlinoss: DLinOSS
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
# Main Ossamma Block
# ============================================================================
struct OssammaBlock <: LuxLayer
    embedding_dimension::Int
    sequence_length::Int
    number_of_heads::Int
    time_dimension::Int
    state_dimension::Int

    # Time-conditioned normalization
    InputNorm::TimeConditionedLayerNorm

    # GLU branch: Dense → split → LinearAttn(content) ⊙ sigmoid(OscSSM(gate)) → Dense
    GluProjection::Lux.Dense           # dim → 2*dim
    LinearAttention::LinearAttentionLayer
    OscillatorLayer::DLinOSS
    GluOutputProjection::Lux.Dense     # dim → dim

    # Local branch: Windowed Softmax Attention
    SlidingWindowAttention::SWAttention

    # Mixing: α projection from input
    AlphaProjection::Lux.Dense         # dim → 1
end

function OssammaBlock(
    embedding_dimension::Int,
    sequence_length::Int,
    number_of_heads::Int,
    time_dimension::Int;
    state_dimension::Int = embedding_dimension,
    window_size::Int = 5,
    min_frequency::Float32 = 0.1f0,
    max_frequency::Float32 = 10.0f0,
    default_time_step::Float32 = 0.1f0,
)
    return OssammaBlock(
        embedding_dimension,
        sequence_length,
        number_of_heads,
        time_dimension,
        state_dimension,
        # Time-conditioned LayerNorm
        TimeConditionedLayerNorm(embedding_dimension, time_dimension),
        # GLU branch
        Lux.Dense(embedding_dimension => 2 * embedding_dimension),
        LinearAttentionLayer(embedding_dimension, sequence_length, number_of_heads, time_dimension),
        DLinOSS(embedding_dimension, state_dimension, embedding_dimension, min_frequency, max_frequency, default_time_step),
        Lux.Dense(embedding_dimension => embedding_dimension),
        # Local branch
        SWAttention(sequence_length, embedding_dimension, number_of_heads; window_size = window_size),
        # Alpha mixing projection
        Lux.Dense(embedding_dimension => 1),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::OssammaBlock)
    return (
        InputNorm = Lux.initialparameters(rng, layer.InputNorm),
        GluProjection = Lux.initialparameters(rng, layer.GluProjection),
        LinearAttention = Lux.initialparameters(rng, layer.LinearAttention),
        OscillatorLayer = Lux.initialparameters(rng, layer.OscillatorLayer),
        GluOutputProjection = Lux.initialparameters(rng, layer.GluOutputProjection),
        SlidingWindowAttention = Lux.initialparameters(rng, layer.SlidingWindowAttention),
        AlphaProjection = Lux.initialparameters(rng, layer.AlphaProjection),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::OssammaBlock)
    return (
        InputNorm = Lux.initialstates(rng, layer.InputNorm),
        GluProjection = Lux.initialstates(rng, layer.GluProjection),
        LinearAttention = Lux.initialstates(rng, layer.LinearAttention),
        OscillatorLayer = Lux.initialstates(rng, layer.OscillatorLayer),
        GluOutputProjection = Lux.initialstates(rng, layer.GluOutputProjection),
        SlidingWindowAttention = Lux.initialstates(rng, layer.SlidingWindowAttention),
        AlphaProjection = Lux.initialstates(rng, layer.AlphaProjection),
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

    # Split into content and gate halves
    dim = block.embedding_dimension
    content_half = selectdim(glu_projected, 1, 1:dim)
    gate_half = selectdim(glu_projected, 1, (dim+1):size(glu_projected, 1))

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
    glu_combined = content_output .* gate_activated

    # Output projection
    glu_output, glu_out_state = block.GluOutputProjection(
        glu_combined, params.GluOutputProjection, state.GluOutputProjection
    )

    # =========================================================================
    # 3. Local-Sharp Branch (Windowed Softmax Attention)
    # =========================================================================
    local_output, sw_attn_state = block.SlidingWindowAttention(
        normalized, params.SlidingWindowAttention, state.SlidingWindowAttention
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
    # 5. Residual Connection → Output
    # =========================================================================
    output = residual .+ mixed_output

    # =========================================================================
    # 6. Update State
    # =========================================================================
    new_state = (
        InputNorm = norm_state,
        GluProjection = glu_proj_state,
        LinearAttention = lin_attn_state,
        OscillatorLayer = osc_state,
        GluOutputProjection = glu_out_state,
        SlidingWindowAttention = sw_attn_state,
        AlphaProjection = alpha_state,
    )

    return output, new_state
end

# ============================================================================
# OssammaNERBlock (with dual gating from GLU to Local branch)
# ============================================================================

"""
OssammaNERBlock - OSSAMMA block with dual gating for NER tasks.

Key architecture (per the NER v2 specification):
- Time-Conditioned LayerNorm
- GLU-Global Branch (processes first):
  Dense(dim→2*dim) → split → LinearAttn(path_a) ⊙ sigmoid(DLinOSS(path_b)) → Dense
- Local-Sharp Branch with DUAL GATING from GLU:
  1. Input Gate: gated_x = x * sigmoid(W_input_gate @ GLU_out) - before attention
  2. Sliding Window Attention on gated input
  3. Output Gate: Local_final = local_out + sigmoid(W_output_gate @ GLU_out) * GLU_out
- Adaptive Mixing: α·GLU_out + (1-α)·Local_final
- Residual + LayerNorm

The dual gating allows GLU to guide what Local attends to (input gate)
and inject global context where needed (output gate).
"""
struct OssammaNERBlock <: LuxLayer
    embedding_dimension::Int
    sequence_length::Int
    number_of_heads::Int
    time_dimension::Int
    state_dimension::Int
    dropout_rate::Float32

    # Time-conditioned normalization
    InputNorm::TimeConditionedLayerNorm

    # GLU branch: Dense → split → LinearAttn(content) ⊙ sigmoid(OscSSM(gate)) → Dense
    GluProjection::Lux.Dense           # dim → 2*dim
    LinearAttention::LinearAttentionLayer
    OscillatorLayer::DLinOSS
    GluOutputProjection::Lux.Dense     # dim → dim

    # Local branch: Windowed Softmax Attention
    SlidingWindowAttention::SWAttention

    # DUAL GATING: GLU guides Local branch
    InputGate::Lux.Dense               # dim → dim (no bias recommended)
    OutputGate::Lux.Dense              # dim → dim (no bias recommended)

    # Mixing: α projection from input + time bias
    AlphaProjection::Lux.Dense         # dim → 1

    # Dropout
    AttentionDropout::Lux.Dropout

    # Output LayerNorm
    OutputNorm::Lux.LayerNorm
end

function OssammaNERBlock(
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
)
    return OssammaNERBlock(
        embedding_dimension,
        sequence_length,
        number_of_heads,
        time_dimension,
        state_dimension,
        dropout_rate,
        # Time-conditioned LayerNorm
        TimeConditionedLayerNorm(embedding_dimension, time_dimension),
        # GLU branch
        Lux.Dense(embedding_dimension => 2 * embedding_dimension),
        LinearAttentionLayer(embedding_dimension, sequence_length, number_of_heads, time_dimension),
        DLinOSS(embedding_dimension, state_dimension, embedding_dimension, min_frequency, max_frequency, default_time_step),
        Lux.Dense(embedding_dimension => embedding_dimension),
        # Local branch
        SWAttention(sequence_length, embedding_dimension, number_of_heads; window_size = window_size),
        # Dual gating projections (no bias for cleaner gradients)
        Lux.Dense(embedding_dimension => embedding_dimension; use_bias = false),
        Lux.Dense(embedding_dimension => embedding_dimension; use_bias = false),
        # Alpha mixing projection
        Lux.Dense(embedding_dimension => 1),
        # Dropout
        Lux.Dropout(dropout_rate),
        # Output LayerNorm
        Lux.LayerNorm((embedding_dimension,)),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, block::OssammaNERBlock)
    # Initialize gate weights with small values (std=0.02) for gates starting near 0.5
    input_gate_params = Lux.initialparameters(rng, block.InputGate)
    output_gate_params = Lux.initialparameters(rng, block.OutputGate)

    # Scale down gate weights for softer initialization
    if haskey(input_gate_params, :weight)
        input_gate_params = (weight = input_gate_params.weight .* 0.02f0,)
    end
    if haskey(output_gate_params, :weight)
        output_gate_params = (weight = output_gate_params.weight .* 0.02f0,)
    end

    return (
        InputNorm = Lux.initialparameters(rng, block.InputNorm),
        GluProjection = Lux.initialparameters(rng, block.GluProjection),
        LinearAttention = Lux.initialparameters(rng, block.LinearAttention),
        OscillatorLayer = Lux.initialparameters(rng, block.OscillatorLayer),
        GluOutputProjection = Lux.initialparameters(rng, block.GluOutputProjection),
        SlidingWindowAttention = Lux.initialparameters(rng, block.SlidingWindowAttention),
        InputGate = input_gate_params,
        OutputGate = output_gate_params,
        AlphaProjection = Lux.initialparameters(rng, block.AlphaProjection),
        AttentionDropout = Lux.initialparameters(rng, block.AttentionDropout),
        OutputNorm = Lux.initialparameters(rng, block.OutputNorm),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, block::OssammaNERBlock)
    return (
        InputNorm = Lux.initialstates(rng, block.InputNorm),
        GluProjection = Lux.initialstates(rng, block.GluProjection),
        LinearAttention = Lux.initialstates(rng, block.LinearAttention),
        OscillatorLayer = Lux.initialstates(rng, block.OscillatorLayer),
        GluOutputProjection = Lux.initialstates(rng, block.GluOutputProjection),
        SlidingWindowAttention = Lux.initialstates(rng, block.SlidingWindowAttention),
        InputGate = Lux.initialstates(rng, block.InputGate),
        OutputGate = Lux.initialstates(rng, block.OutputGate),
        AlphaProjection = Lux.initialstates(rng, block.AlphaProjection),
        AttentionDropout = Lux.initialstates(rng, block.AttentionDropout),
        OutputNorm = Lux.initialstates(rng, block.OutputNorm),
    )
end

function (block::OssammaNERBlock)(inputs::Tuple, params, state)
    input_tensor, time_input = inputs
    # input_tensor: (embedding_dim, seq_len, batch) or (embedding_dim, seq_len)
    # time_input: (time_dim, batch) or (time_dim,)

    residual = input_tensor
    is_batched = ndims(input_tensor) == 3

    # =========================================================================
    # 1. Time-Conditioned LayerNorm
    # =========================================================================
    normalized, alpha_bias, norm_state = block.InputNorm(
        input_tensor, time_input, params.InputNorm, state.InputNorm
    )

    # =========================================================================
    # 2. GLU-Global Branch (MUST complete before Local starts)
    # =========================================================================
    # Project to 2*dim and split
    glu_projected, glu_proj_state = block.GluProjection(
        normalized, params.GluProjection, state.GluProjection
    )

    # Split into path_a (LinearAttention) and path_b (Oscillator)
    dim = block.embedding_dimension
    path_a = selectdim(glu_projected, 1, 1:dim)
    path_b = selectdim(glu_projected, 1, (dim+1):size(glu_projected, 1))

    # path_a → Linear Attention
    attn_out, lin_attn_state = block.LinearAttention(
        (path_a, time_input), params.LinearAttention, state.LinearAttention
    )

    # path_b → Oscillator SSM → sigmoid
    osc_out, osc_state = block.OscillatorLayer(
        path_b, params.OscillatorLayer, state.OscillatorLayer
    )

    # GLU gating: attn_out * sigmoid(osc_out)
    gated = attn_out .* NNlib.sigmoid.(osc_out)

    # Output projection
    glu_out, glu_out_state = block.GluOutputProjection(
        gated, params.GluOutputProjection, state.GluOutputProjection
    )

    # =========================================================================
    # 3. Local-Sharp Branch with DUAL GATING
    # =========================================================================

    # Step 3a: Input Gate - GLU controls what features Local should attend to
    # input_gate = sigmoid(W_input_gate @ GLU_out)
    input_gate_logits, input_gate_state = block.InputGate(
        glu_out, params.InputGate, state.InputGate
    )
    input_gate = NNlib.sigmoid.(input_gate_logits)

    # gated_x = x_norm * input_gate (suppresses irrelevant features)
    gated_x = normalized .* input_gate

    # Step 3b: Sliding Window Attention on gated input
    local_out, sw_attn_state = block.SlidingWindowAttention(
        gated_x, params.SlidingWindowAttention, state.SlidingWindowAttention
    )

    # Step 3c: Output Gate - GLU injects global context where Local needs help
    # output_gate = sigmoid(W_output_gate @ GLU_out)
    output_gate_logits, output_gate_state = block.OutputGate(
        glu_out, params.OutputGate, state.OutputGate
    )
    output_gate = NNlib.sigmoid.(output_gate_logits)

    # Local_final = local_out + output_gate * GLU_out
    # (adds global information at positions where it's needed)
    local_final = local_out .+ (output_gate .* glu_out)

    # =========================================================================
    # 4. Adaptive Mixing: α·GLU_out + (1-α)·Local_final
    # =========================================================================
    seq_dim = 2

    # Mean pool over sequence dimension for alpha computation
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
    mixed_output = alpha_broadcast .* glu_out .+ (1.0f0 .- alpha_broadcast) .* local_final

    # =========================================================================
    # 5. Dropout
    # =========================================================================
    mixed_output, attn_dropout_state = block.AttentionDropout(
        mixed_output, params.AttentionDropout, state.AttentionDropout
    )

    # =========================================================================
    # 6. Residual + Output LayerNorm
    # =========================================================================
    output_pre_norm = residual .+ mixed_output

    # Apply output LayerNorm
    output_flat = reshape(output_pre_norm, block.embedding_dimension, :)
    output_norm_flat, output_norm_state = block.OutputNorm(output_flat, params.OutputNorm, state.OutputNorm)
    output = reshape(output_norm_flat, size(output_pre_norm))

    # =========================================================================
    # 7. Update State
    # =========================================================================
    new_state = (
        InputNorm = norm_state,
        GluProjection = glu_proj_state,
        LinearAttention = lin_attn_state,
        OscillatorLayer = osc_state,
        GluOutputProjection = glu_out_state,
        SlidingWindowAttention = sw_attn_state,
        InputGate = input_gate_state,
        OutputGate = output_gate_state,
        AlphaProjection = alpha_state,
        AttentionDropout = attn_dropout_state,
        OutputNorm = output_norm_state,
    )

    return output, new_state
end

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
export Dlinoss, Attention, LinearAttention, ossm, LLaDA, Classification, NER
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
export DLinOSS, SWAttention, OscSSM

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
export is_valid_transition, build_transition_mask

end
