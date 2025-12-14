module Ossamma

"""
Ossamma -> Oscillatory State Space Attention Masked Morphed Architecture

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
using .LinearAttention: LinearAttention
using .Dlinoss: DLinOSS
using .ossm: ossm as OscSSM

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

    # Apply base LayerNorm
    normalized, ln_state = layer.LayerNorm(input_tensor, params.LayerNorm, state.LayerNorm)

    # Compute time-conditioned scale and shift
    scale_raw, scale_state = layer.ScaleProjection(time_input, params.ScaleProjection, state.ScaleProjection)
    shift, shift_state = layer.ShiftProjection(time_input, params.ShiftProjection, state.ShiftProjection)
    alpha_bias, alpha_state = layer.AlphaBiasProjection(time_input, params.AlphaBiasProjection, state.AlphaBiasProjection)

    scale = 1.0f0 .+ scale_raw  # Center around 1

    # Broadcast scale and shift: (embedding_dim, batch) → (embedding_dim, 1, batch)
    is_batched = ndims(input_tensor) == 3
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
    LinearAttentionLayer::LinearAttention
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
        LinearAttention(embedding_dimension, sequence_length, number_of_heads, time_dimension),
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
        LinearAttentionLayer = Lux.initialparameters(rng, layer.LinearAttentionLayer),
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
        LinearAttentionLayer = Lux.initialstates(rng, layer.LinearAttentionLayer),
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
    content_half = glu_projected[1:dim, ..]
    gate_half = glu_projected[(dim+1):end, ..]

    # Content → Linear Attention
    content_output, lin_attn_state = block.LinearAttentionLayer(
        (content_half, time_input), params.LinearAttentionLayer, state.LinearAttentionLayer
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
        LinearAttentionLayer = lin_attn_state,
        OscillatorLayer = osc_state,
        GluOutputProjection = glu_out_state,
        SlidingWindowAttention = sw_attn_state,
        AlphaProjection = alpha_state,
    )

    return output, new_state
end

# ============================================================================
# LLaDA Text Diffusion Model
# ============================================================================
include("LLaDA.jl")
using .LLaDA: LLaDAModel, TimeMLPEmbedding, SinusoidalTimeEmbedding
using .LLaDA: LLaDAConfig, load_config, save_config, config_from_dict
using .LLaDA: default_config, small_config, base_config, large_config
using .LLaDA: apply_mask, sample_mask_ratio, unmask_step, generate

# ============================================================================
# Training Utilities
# ============================================================================
include("Training.jl")
using .Training: masked_cross_entropy, diffusion_loss
using .Training: TrainState, create_train_state, train_step!
using .Training: warmup_cosine_schedule, evaluate, compute_accuracy
using .Training: TrainingConfig, load_training_config, train!

# ============================================================================
# Exports
# ============================================================================

# Re-export submodules for callers who want direct access.
export Dlinoss, Attention, LinearAttention, ossm, LLaDA

# Main block
export OssammaBlock, TimeConditionedLayerNorm

# LLaDA model and utilities
export LLaDAModel, TimeMLPEmbedding, SinusoidalTimeEmbedding
export LLaDAConfig, load_config, save_config, config_from_dict
export default_config, small_config, base_config, large_config
export apply_mask, sample_mask_ratio, unmask_step, generate

# Training utilities
export masked_cross_entropy, diffusion_loss
export TrainState, create_train_state, train_step!
export warmup_cosine_schedule, evaluate, compute_accuracy
export TrainingConfig, load_training_config, train!

# Provide conventional aliases for the main layer types.
export DLinOSS, SWAttention, OscSSM

end
