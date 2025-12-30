module DeepScaling

"""
DeepScaling - Strategies for training very deep Ossamma models (48-96+ layers).

This module implements:
1. Hierarchical frequency ranges - different oscillator frequencies per layer
2. Layer scale initialization - stabilize deep networks
3. Stochastic depth - regularization via random layer dropping
4. Gradient checkpointing - memory-efficient training
5. OssammaBlockDeep - deep-optimized block variant
6. Mixed attention builders - allocate attention budget wisely
7. Deep model configurations

Key insight: Ossamma's O(T) complexity (vs Transformer's O(T²)) allows
4-8× more layers for the same compute budget.

Reference: docs/DEPTH_SCALING_STRATEGY.md
"""

using Lux
using LuxCore
using Random
using NNlib
using Statistics: mean

# Import parent module types (will be set when included)
import ..TimeConditionedLayerNorm
import ..LinearAttentionLayer
import ..DLinOSS
import ..DLinOSSParallel
import ..SwiGLU
import ..SWAttention
import ..RMSNorm

export HierarchicalFrequencyConfig, compute_layer_frequencies
export LayerScaleConfig, apply_layer_scale
export StochasticDepthConfig, should_drop_layer
export OssammaBlockDeep
export DeepModelConfig, create_deep_blocks
export deep_48L_config, ultra_96L_config, long_context_config

# =============================================================================
# 1. HIERARCHICAL FREQUENCY RANGES
# =============================================================================

"""
    HierarchicalFrequencyConfig

Configuration for layer-dependent oscillator frequency ranges.

The intuition:
- Early layers (high freq): Capture local syntax, adjacent word patterns
- Late layers (low freq): Capture document-level semantics, long-range deps

# Fields
- `base_min_freq::Float32`: Minimum frequency at the deepest layer
- `base_max_freq::Float32`: Maximum frequency at the first layer
- `decay_rate::Float32`: Exponential decay rate (higher = faster transition to low freq)
- `scaling_type::Symbol`: :exponential, :linear, or :logarithmic
"""
Base.@kwdef struct HierarchicalFrequencyConfig
    base_min_freq::Float32 = 0.01f0
    base_max_freq::Float32 = 100.0f0
    decay_rate::Float32 = 3.0f0
    scaling_type::Symbol = :exponential
end

"""
    compute_layer_frequencies(layer_idx, total_layers, config)

Compute the (min_freq, max_freq) range for a specific layer.

# Arguments
- `layer_idx::Int`: Current layer index (1-indexed)
- `total_layers::Int`: Total number of layers in the model
- `config::HierarchicalFrequencyConfig`: Frequency configuration

# Returns
- `(min_freq::Float32, max_freq::Float32)`: Frequency range for this layer

# Example
```julia
config = HierarchicalFrequencyConfig()
for i in 1:48
    min_f, max_f = compute_layer_frequencies(i, 48, config)
    println("Layer \$i: [\$min_f, \$max_f]")
end
```
"""
function compute_layer_frequencies(
    layer_idx::Int,
    total_layers::Int,
    config::HierarchicalFrequencyConfig = HierarchicalFrequencyConfig()
)
    progress = (layer_idx - 1) / max(total_layers - 1, 1)  # 0.0 → 1.0

    if config.scaling_type == :exponential
        # Exponential decay: early = high freq, late = low freq
        scale = exp(-config.decay_rate * progress)
        max_freq = config.base_max_freq * scale
        min_freq = config.base_min_freq + (1.0f0 - scale) * 0.1f0

    elseif config.scaling_type == :linear
        # Linear interpolation
        max_freq = config.base_max_freq * (1.0f0 - 0.9f0 * progress)
        min_freq = config.base_min_freq * (1.0f0 + 9.0f0 * progress)

    elseif config.scaling_type == :logarithmic
        # Logarithmic spacing (good for very deep networks)
        log_progress = log(1.0f0 + progress * (exp(1.0f0) - 1.0f0))
        max_freq = config.base_max_freq * exp(-config.decay_rate * log_progress)
        min_freq = config.base_min_freq * exp(log_progress)

    else
        error("Unknown scaling_type: $(config.scaling_type). Use :exponential, :linear, or :logarithmic")
    end

    # Ensure min < max
    min_freq = min(min_freq, max_freq * 0.1f0)

    return (Float32(min_freq), Float32(max_freq))
end

"""
    frequency_summary(total_layers, config)

Print a summary of frequencies across all layers.
"""
function frequency_summary(total_layers::Int, config::HierarchicalFrequencyConfig = HierarchicalFrequencyConfig())
    println("Hierarchical Frequency Summary ($(total_layers) layers, $(config.scaling_type)):")
    println("=" ^ 60)

    checkpoints = [1, total_layers ÷ 4, total_layers ÷ 2, 3 * total_layers ÷ 4, total_layers]

    for i in checkpoints
        min_f, max_f = compute_layer_frequencies(i, total_layers, config)
        role = if i <= total_layers ÷ 4
            "local syntax"
        elseif i <= total_layers ÷ 2
            "phrase/clause"
        elseif i <= 3 * total_layers ÷ 4
            "paragraph"
        else
            "document-level"
        end
        println("  Layer $i: freq ∈ [$(round(min_f, digits=4)), $(round(max_f, digits=2))] - $role")
    end
    println()
end

# =============================================================================
# 2. LAYER SCALE INITIALIZATION
# =============================================================================

"""
    LayerScaleConfig

Configuration for layer scale initialization (from CaiT, DeepViT papers).

Layer scale helps stabilize training of very deep networks by scaling
the residual contribution of each layer by a small learnable value.

# Fields
- `init_value::Float32`: Initial scale value (typically 0.1 or 1e-4 for very deep)
- `learnable::Bool`: Whether the scale is learnable or fixed
- `per_channel::Bool`: Per-channel scale (true) or scalar (false)
"""
Base.@kwdef struct LayerScaleConfig
    init_value::Float32 = 0.1f0
    learnable::Bool = true
    per_channel::Bool = true
end

"""
    init_layer_scale(rng, dim, config, layer_idx, total_layers)

Initialize layer scale parameters.

For very deep networks, scales earlier layers more and later layers less.
"""
function init_layer_scale(
    rng::Random.AbstractRNG,
    dim::Int,
    config::LayerScaleConfig,
    layer_idx::Int = 1,
    total_layers::Int = 1
)
    # Scale init value based on layer depth (deeper = smaller init)
    depth_factor = sqrt(layer_idx / total_layers)
    adjusted_init = config.init_value * depth_factor

    if config.per_channel
        return fill(Float32(adjusted_init), dim)
    else
        return Float32[adjusted_init]
    end
end

"""
    apply_layer_scale(x, scale)

Apply layer scale to tensor x.
"""
function apply_layer_scale(x::AbstractArray, scale::AbstractVector)
    # Broadcast scale across sequence and batch dimensions
    if ndims(x) == 2
        # (dim, seq)
        return x .* reshape(scale, :, 1)
    elseif ndims(x) == 3
        # (dim, seq, batch)
        return x .* reshape(scale, :, 1, 1)
    else
        return x .* scale
    end
end

# =============================================================================
# 3. STOCHASTIC DEPTH
# =============================================================================

"""
    StochasticDepthConfig

Configuration for stochastic depth (drop path) regularization.

During training, randomly skips layers with increasing probability
for deeper layers. This regularizes the model and speeds up training.

# Fields
- `drop_rate::Float32`: Maximum drop rate (for deepest layer)
- `mode::Symbol`: :linear (linearly increase) or :uniform (same for all)
"""
Base.@kwdef struct StochasticDepthConfig
    drop_rate::Float32 = 0.1f0
    mode::Symbol = :linear
end

"""
    layer_drop_rate(layer_idx, total_layers, config)

Compute the drop probability for a specific layer.
"""
function layer_drop_rate(
    layer_idx::Int,
    total_layers::Int,
    config::StochasticDepthConfig
)
    if config.mode == :linear
        # Linear increase: first layer = 0, last layer = drop_rate
        return config.drop_rate * (layer_idx - 1) / max(total_layers - 1, 1)
    elseif config.mode == :uniform
        return config.drop_rate
    else
        error("Unknown stochastic depth mode: $(config.mode)")
    end
end

"""
    should_drop_layer(layer_idx, total_layers, config; training=true)

Determine if a layer should be dropped during this forward pass.
Only drops during training.
"""
function should_drop_layer(
    layer_idx::Int,
    total_layers::Int,
    config::StochasticDepthConfig;
    training::Bool = true
)
    if !training
        return false
    end

    drop_prob = layer_drop_rate(layer_idx, total_layers, config)
    return rand() < drop_prob
end

"""
    drop_path(x, drop_prob; training=true)

Apply drop path (stochastic depth) to input x.

During training, with probability `drop_prob`, returns zeros.
During inference, always returns x unchanged.
"""
function drop_path(x::AbstractArray, drop_prob::Float32; training::Bool = true)
    if !training || drop_prob == 0.0f0
        return x
    end

    keep_prob = 1.0f0 - drop_prob

    # Sample a binary mask (same for all elements in a batch item)
    if ndims(x) == 3
        # (dim, seq, batch)
        batch_size = size(x, 3)
        mask = rand(Float32, 1, 1, batch_size) .< keep_prob
        return x .* mask ./ keep_prob
    else
        mask = rand() < keep_prob
        return mask ? x / keep_prob : zero(x)
    end
end

# =============================================================================
# 4. GRADIENT CHECKPOINTING
# =============================================================================

"""
    CheckpointConfig

Configuration for gradient checkpointing.

Gradient checkpointing trades compute for memory by not storing
intermediate activations, instead recomputing them during backward pass.

# Fields
- `checkpoint_every::Int`: Checkpoint every N layers
- `enabled::Bool`: Whether checkpointing is enabled
"""
Base.@kwdef struct CheckpointConfig
    checkpoint_every::Int = 4
    enabled::Bool = true
end

"""
    should_checkpoint(layer_idx, config)

Determine if this layer should be checkpointed.
"""
function should_checkpoint(layer_idx::Int, config::CheckpointConfig)
    return config.enabled && (layer_idx % config.checkpoint_every == 0)
end

# Note: Actual checkpointing implementation requires Zygote.checkpointed
# which is integrated at the training loop level, not here.

# =============================================================================
# 5. DEEP OSSAMMA BLOCK
# =============================================================================

"""
    OssammaBlockDeep

A deep-network optimized variant of OssammaBlock with:
- Layer scale initialization
- Stochastic depth support
- Hierarchical frequency configuration
- Optional block type variants (local-only, global-only, full)

# Architecture variants
- `:full`: Full OssammaBlock (LinearAttn + DLinOSS + SWAttention)
- `:global_only`: LinearAttention + DLinOSS only (like OssammaDrafterBlock)
- `:local_only`: SWAttention only (lightweight local processing)
"""
struct OssammaBlockDeep <: LuxCore.AbstractLuxLayer
    # Core dimensions
    embedding_dimension::Int
    sequence_length::Int
    number_of_heads::Int
    time_dimension::Int
    state_dimension::Int

    # Deep network config
    layer_idx::Int
    total_layers::Int
    block_type::Symbol  # :full, :global_only, :local_only

    # Feature flags
    dropout_rate::Float32
    use_ffn::Bool
    use_parallel_scan::Bool
    use_layer_scale::Bool
    use_stochastic_depth::Bool
    layer_scale_init::Float32
    stochastic_depth_rate::Float32

    # Frequency range (computed from hierarchical config)
    min_frequency::Float32
    max_frequency::Float32

    # Layers (some may be nothing depending on block_type)
    InputNorm::TimeConditionedLayerNorm
    GluProjection::Union{Lux.Dense, Nothing}
    LinearAttention::Union{LinearAttentionLayer, Nothing}
    OscillatorLayer::Union{DLinOSS, DLinOSSParallel, Nothing}
    LinearAttnNorm::Union{RMSNorm, Nothing}    # RMSNorm after linear attention
    OscillatorNorm::Union{RMSNorm, Nothing}    # RMSNorm after oscillator
    InputGate::Union{Lux.Dense, Nothing}
    SlidingWindowAttention::Union{SWAttention, Nothing}
    AlphaProjection::Union{Lux.Dense, Nothing}
    Dropout::Lux.Dropout
    FFN::Union{SwiGLU, Nothing}
    OutputNorm::Lux.LayerNorm
end

"""
    OssammaBlockDeep(embedding_dim, seq_len, heads, time_dim; kwargs...)

Create a deep-optimized Ossamma block.

# Arguments
- `embedding_dimension::Int`: Model dimension
- `sequence_length::Int`: Maximum sequence length
- `number_of_heads::Int`: Number of attention heads
- `time_dimension::Int`: Time embedding dimension

# Keyword Arguments
- `layer_idx::Int = 1`: This layer's index (1-indexed)
- `total_layers::Int = 1`: Total layers in model
- `block_type::Symbol = :full`: Block variant (:full, :global_only, :local_only)
- `freq_config::HierarchicalFrequencyConfig`: Frequency configuration
- `use_layer_scale::Bool = true`: Enable layer scale
- `layer_scale_init::Float32 = 0.1`: Layer scale init value
- `use_stochastic_depth::Bool = true`: Enable stochastic depth
- `stochastic_depth_rate::Float32 = 0.1`: Max drop rate
- `use_parallel_scan::Bool = true`: Use parallel oscillator scan
- `window_size::Int = 64`: SWAttention window size
"""
function OssammaBlockDeep(
    embedding_dimension::Int,
    sequence_length::Int,
    number_of_heads::Int,
    time_dimension::Int;
    layer_idx::Int = 1,
    total_layers::Int = 1,
    block_type::Symbol = :full,
    state_dimension::Int = embedding_dimension,
    freq_config::HierarchicalFrequencyConfig = HierarchicalFrequencyConfig(),
    dropout_rate::Float32 = 0.1f0,
    use_ffn::Bool = true,
    ffn_expansion::Float32 = 1.5f0,
    use_parallel_scan::Bool = true,
    parallel_chunk_size::Int = 64,
    use_layer_scale::Bool = true,
    layer_scale_init::Float32 = 0.1f0,
    use_stochastic_depth::Bool = true,
    stochastic_depth_rate::Float32 = 0.1f0,
    window_size::Int = 64,
)
    # Compute layer-specific frequency range
    min_freq, max_freq = compute_layer_frequencies(layer_idx, total_layers, freq_config)

    # Build layers based on block_type
    needs_global = block_type in (:full, :global_only)
    needs_local = block_type in (:full, :local_only)
    needs_mixing = block_type == :full

    # Oscillator layer (for global branch)
    oscillator_layer = if needs_global
        if use_parallel_scan
            DLinOSSParallel(
                embedding_dimension, state_dimension, embedding_dimension,
                min_freq, max_freq, 0.1f0;
                chunk_size = parallel_chunk_size
            )
        else
            DLinOSS(
                embedding_dimension, state_dimension, embedding_dimension,
                min_freq, max_freq, 0.1f0
            )
        end
    else
        nothing
    end

    # Linear attention (for global branch)
    linear_attn = needs_global ?
        LinearAttentionLayer(embedding_dimension, sequence_length, number_of_heads, time_dimension) :
        nothing

    # GLU projection (for global branch)
    glu_proj = needs_global ?
        Lux.Dense(embedding_dimension => 2 * embedding_dimension) :
        nothing

    # SWAttention (for local branch)
    sw_attn = needs_local ?
        SWAttention(sequence_length, embedding_dimension, number_of_heads; window_size = window_size) :
        nothing

    # Input gate (for local branch when using full block)
    input_gate = needs_mixing ?
        Lux.Dense(embedding_dimension => embedding_dimension; use_bias = false) :
        nothing

    # Alpha projection (for mixing)
    alpha_proj = needs_mixing ?
        Lux.Dense(embedding_dimension => 1) :
        nothing

    # RMSNorm layers for GLU gating (only for global branch)
    lin_attn_norm = needs_global ? RMSNorm(embedding_dimension) : nothing
    osc_norm = needs_global ? RMSNorm(embedding_dimension) : nothing

    return OssammaBlockDeep(
        embedding_dimension,
        sequence_length,
        number_of_heads,
        time_dimension,
        state_dimension,
        layer_idx,
        total_layers,
        block_type,
        dropout_rate,
        use_ffn,
        use_parallel_scan,
        use_layer_scale,
        use_stochastic_depth,
        layer_scale_init,
        stochastic_depth_rate,
        min_freq,
        max_freq,
        # Layers
        TimeConditionedLayerNorm(embedding_dimension, time_dimension),
        glu_proj,
        linear_attn,
        oscillator_layer,
        lin_attn_norm,
        osc_norm,
        input_gate,
        sw_attn,
        alpha_proj,
        Lux.Dropout(dropout_rate),
        use_ffn ? SwiGLU(embedding_dimension; expansion_factor = ffn_expansion) : nothing,
        Lux.LayerNorm((embedding_dimension,)),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, block::OssammaBlockDeep)
    params = Dict{Symbol, Any}()

    params[:InputNorm] = Lux.initialparameters(rng, block.InputNorm)
    params[:Dropout] = Lux.initialparameters(rng, block.Dropout)
    params[:OutputNorm] = Lux.initialparameters(rng, block.OutputNorm)

    # Global branch params
    if block.GluProjection !== nothing
        params[:GluProjection] = Lux.initialparameters(rng, block.GluProjection)
    end
    if block.LinearAttention !== nothing
        params[:LinearAttention] = Lux.initialparameters(rng, block.LinearAttention)
    end
    if block.OscillatorLayer !== nothing
        params[:OscillatorLayer] = Lux.initialparameters(rng, block.OscillatorLayer)
    end
    # RMSNorm for GLU gating
    if block.LinearAttnNorm !== nothing
        params[:LinearAttnNorm] = Lux.initialparameters(rng, block.LinearAttnNorm)
    end
    if block.OscillatorNorm !== nothing
        params[:OscillatorNorm] = Lux.initialparameters(rng, block.OscillatorNorm)
    end

    # Local branch params
    if block.InputGate !== nothing
        gate_params = Lux.initialparameters(rng, block.InputGate)
        if haskey(gate_params, :weight)
            gate_params = (weight = gate_params.weight .* 0.02f0,)
        end
        params[:InputGate] = gate_params
    end
    if block.SlidingWindowAttention !== nothing
        params[:SlidingWindowAttention] = Lux.initialparameters(rng, block.SlidingWindowAttention)
    end

    # Mixing params
    if block.AlphaProjection !== nothing
        params[:AlphaProjection] = Lux.initialparameters(rng, block.AlphaProjection)
    end

    # FFN params
    if block.use_ffn && block.FFN !== nothing
        params[:FFN] = Lux.initialparameters(rng, block.FFN)
    end

    # Layer scale (learnable)
    if block.use_layer_scale
        scale_config = LayerScaleConfig(init_value = block.layer_scale_init)
        params[:layer_scale] = init_layer_scale(
            rng, block.embedding_dimension, scale_config,
            block.layer_idx, block.total_layers
        )
    end

    return NamedTuple(params)
end

function Lux.initialstates(rng::Random.AbstractRNG, block::OssammaBlockDeep)
    states = Dict{Symbol, Any}()

    states[:InputNorm] = Lux.initialstates(rng, block.InputNorm)
    states[:Dropout] = Lux.initialstates(rng, block.Dropout)
    states[:OutputNorm] = Lux.initialstates(rng, block.OutputNorm)

    if block.GluProjection !== nothing
        states[:GluProjection] = Lux.initialstates(rng, block.GluProjection)
    end
    if block.LinearAttention !== nothing
        states[:LinearAttention] = Lux.initialstates(rng, block.LinearAttention)
    end
    if block.OscillatorLayer !== nothing
        states[:OscillatorLayer] = Lux.initialstates(rng, block.OscillatorLayer)
    end
    # RMSNorm for GLU gating
    if block.LinearAttnNorm !== nothing
        states[:LinearAttnNorm] = Lux.initialstates(rng, block.LinearAttnNorm)
    end
    if block.OscillatorNorm !== nothing
        states[:OscillatorNorm] = Lux.initialstates(rng, block.OscillatorNorm)
    end
    if block.InputGate !== nothing
        states[:InputGate] = Lux.initialstates(rng, block.InputGate)
    end
    if block.SlidingWindowAttention !== nothing
        states[:SlidingWindowAttention] = Lux.initialstates(rng, block.SlidingWindowAttention)
    end
    if block.AlphaProjection !== nothing
        states[:AlphaProjection] = Lux.initialstates(rng, block.AlphaProjection)
    end
    if block.use_ffn && block.FFN !== nothing
        states[:FFN] = Lux.initialstates(rng, block.FFN)
    end

    # Training flag for stochastic depth
    states[:training] = true

    return NamedTuple(states)
end

function (block::OssammaBlockDeep)(inputs::Tuple, params, state)
    input_tensor, time_input = inputs

    # Check for stochastic depth skip
    training = get(state, :training, true)
    if block.use_stochastic_depth && training
        drop_prob = layer_drop_rate(block.layer_idx, block.total_layers,
            StochasticDepthConfig(drop_rate = block.stochastic_depth_rate))
        if rand() < drop_prob
            # Skip this layer entirely
            return input_tensor, state
        end
    end

    residual = input_tensor

    # 1. Time-Conditioned LayerNorm
    normalized, alpha_bias, norm_state = block.InputNorm(
        input_tensor, time_input, params.InputNorm, state.InputNorm
    )

    new_states = Dict{Symbol, Any}(:InputNorm => norm_state)

    # 2. Process based on block type
    if block.block_type == :global_only
        output = forward_global_only(block, normalized, time_input, params, state, new_states)
    elseif block.block_type == :local_only
        output = forward_local_only(block, normalized, params, state, new_states)
    else  # :full
        output = forward_full(block, normalized, time_input, alpha_bias, params, state, new_states)
    end

    # 3. Dropout
    output, dropout_state = block.Dropout(output, params.Dropout, state.Dropout)
    new_states[:Dropout] = dropout_state

    # 4. FFN
    if block.use_ffn && block.FFN !== nothing
        output, ffn_state = block.FFN(output, params.FFN, state.FFN)
        new_states[:FFN] = ffn_state
    end

    # 5. Layer scale
    if block.use_layer_scale && haskey(params, :layer_scale)
        output = apply_layer_scale(output, params.layer_scale)
    end

    # 6. Residual + Output LayerNorm
    output_pre_norm = residual .+ output
    output_flat = reshape(output_pre_norm, block.embedding_dimension, :)
    output_norm_flat, output_norm_state = block.OutputNorm(
        output_flat, params.OutputNorm, state.OutputNorm
    )
    output = reshape(output_norm_flat, size(output_pre_norm))
    new_states[:OutputNorm] = output_norm_state

    # Preserve training flag
    new_states[:training] = training

    return output, NamedTuple(new_states)
end

# Helper: Global-only forward (LinearAttn + DLinOSS)
function forward_global_only(block, normalized, time_input, params, state, new_states)
    dim = block.embedding_dimension

    # GLU projection
    glu_projected, glu_state = block.GluProjection(normalized, params.GluProjection, state.GluProjection)
    new_states[:GluProjection] = glu_state

    # Split
    content_half = copy(selectdim(glu_projected, 1, 1:dim))
    gate_half = copy(selectdim(glu_projected, 1, (dim+1):size(glu_projected, 1)))

    # Content → Linear Attention → RMSNorm
    content_output, lin_attn_state = block.LinearAttention(
        (content_half, time_input), params.LinearAttention, state.LinearAttention
    )
    new_states[:LinearAttention] = lin_attn_state
    content_output, lin_attn_norm_state = block.LinearAttnNorm(
        content_output, params.LinearAttnNorm, state.LinearAttnNorm
    )
    new_states[:LinearAttnNorm] = lin_attn_norm_state

    # Gate → Oscillator → RMSNorm → sigmoid
    gate_output, osc_state = block.OscillatorLayer(gate_half, params.OscillatorLayer, state.OscillatorLayer)
    new_states[:OscillatorLayer] = osc_state
    gate_output, osc_norm_state = block.OscillatorNorm(
        gate_output, params.OscillatorNorm, state.OscillatorNorm
    )
    new_states[:OscillatorNorm] = osc_norm_state

    # GLU gating: RMSNorm(content) ⊙ sigmoid(RMSNorm(gate))
    return content_output .* NNlib.sigmoid.(gate_output)
end

# Helper: Local-only forward (SWAttention)
function forward_local_only(block, normalized, params, state, new_states)
    output, sw_state = block.SlidingWindowAttention(
        normalized, params.SlidingWindowAttention, state.SlidingWindowAttention
    )
    new_states[:SlidingWindowAttention] = sw_state
    return output
end

# Helper: Full forward (both branches + mixing)
function forward_full(block, normalized, time_input, alpha_bias, params, state, new_states)
    dim = block.embedding_dimension

    # Global branch
    glu_projected, glu_state = block.GluProjection(normalized, params.GluProjection, state.GluProjection)
    new_states[:GluProjection] = glu_state

    content_half = copy(selectdim(glu_projected, 1, 1:dim))
    gate_half = copy(selectdim(glu_projected, 1, (dim+1):size(glu_projected, 1)))

    # Content → Linear Attention → RMSNorm
    content_output, lin_attn_state = block.LinearAttention(
        (content_half, time_input), params.LinearAttention, state.LinearAttention
    )
    new_states[:LinearAttention] = lin_attn_state
    content_output, lin_attn_norm_state = block.LinearAttnNorm(
        content_output, params.LinearAttnNorm, state.LinearAttnNorm
    )
    new_states[:LinearAttnNorm] = lin_attn_norm_state

    # Gate → Oscillator → RMSNorm → sigmoid
    gate_output, osc_state = block.OscillatorLayer(gate_half, params.OscillatorLayer, state.OscillatorLayer)
    new_states[:OscillatorLayer] = osc_state
    gate_output, osc_norm_state = block.OscillatorNorm(
        gate_output, params.OscillatorNorm, state.OscillatorNorm
    )
    new_states[:OscillatorNorm] = osc_norm_state

    # GLU gating: RMSNorm(content) ⊙ sigmoid(RMSNorm(gate))
    glu_output = content_output .* NNlib.sigmoid.(gate_output)

    # Local branch with input gating
    input_gate_logits, input_gate_state = block.InputGate(glu_output, params.InputGate, state.InputGate)
    new_states[:InputGate] = input_gate_state
    input_gate = NNlib.sigmoid.(input_gate_logits)
    gated_x = normalized .* input_gate

    local_output, sw_state = block.SlidingWindowAttention(
        gated_x, params.SlidingWindowAttention, state.SlidingWindowAttention
    )
    new_states[:SlidingWindowAttention] = sw_state

    # Adaptive mixing (token-wise): α_t = σ(Wα · h_t + bα)
    is_batched = ndims(normalized) == 3
    original_size = size(normalized)
    normalized_flat = reshape(normalized, block.embedding_dimension, :)  # (dim, seq*batch)

    alpha_logits_flat, alpha_state = block.AlphaProjection(
        normalized_flat, params.AlphaProjection, state.AlphaProjection
    )
    new_states[:AlphaProjection] = alpha_state

    # Reshape back to (1, seq, batch) or (1, seq)
    if is_batched
        alpha_logits = reshape(alpha_logits_flat, 1, original_size[2], original_size[3])
        # Broadcast alpha_bias: (1, batch) → (1, 1, batch)
        alpha_bias_broadcast = reshape(alpha_bias, 1, 1, size(alpha_bias, 2))
    else
        alpha_logits = reshape(alpha_logits_flat, 1, original_size[2])
        alpha_bias_broadcast = reshape(alpha_bias, 1, 1)
    end

    # Token-wise mixing
    alpha = NNlib.sigmoid.(alpha_logits .+ alpha_bias_broadcast)

    return alpha .* glu_output .+ (1.0f0 .- alpha) .* local_output
end

# =============================================================================
# 6. MIXED ATTENTION LAYER BUILDER
# =============================================================================

"""
    BlockTypeSchedule

Determines which block type to use at each layer.

# Types
- `:uniform`: Same block type for all layers
- `:progressive`: local → global → full
- `:sandwich`: full at edges, lightweight in middle
- `:alternating`: Alternate between types
"""
@enum BlockTypeSchedule begin
    UNIFORM
    PROGRESSIVE
    SANDWICH
    ALTERNATING
end

"""
    get_block_type(layer_idx, total_layers, schedule)

Get the block type for a specific layer.
"""
function get_block_type(layer_idx::Int, total_layers::Int, schedule::BlockTypeSchedule)
    progress = layer_idx / total_layers

    if schedule == UNIFORM
        return :full

    elseif schedule == PROGRESSIVE
        # Early: local, Mid: global, Late: full
        if progress <= 0.25
            return :local_only
        elseif progress <= 0.6
            return :global_only
        else
            return :full
        end

    elseif schedule == SANDWICH
        # Full at edges, global in middle
        if progress <= 0.15 || progress >= 0.85
            return :full
        else
            return :global_only
        end

    elseif schedule == ALTERNATING
        # Alternate local/global, with full every 4th layer
        if layer_idx % 4 == 0
            return :full
        elseif layer_idx % 2 == 0
            return :global_only
        else
            return :local_only
        end
    end
end

# =============================================================================
# 7. DEEP MODEL CONFIGURATIONS
# =============================================================================

"""
    DeepModelConfig

Configuration for deep Ossamma models.
"""
Base.@kwdef struct DeepModelConfig
    # Core architecture
    vocab_size::Int = 32000
    max_sequence_length::Int = 4096
    embedding_dimension::Int = 384
    number_of_heads::Int = 6
    number_of_layers::Int = 48
    time_dimension::Int = 64

    # Deep network optimizations
    use_parallel_scan::Bool = true
    use_layer_scale::Bool = true
    layer_scale_init::Float32 = 0.1f0
    use_stochastic_depth::Bool = true
    stochastic_depth_rate::Float32 = 0.1f0

    # Hierarchical frequencies
    freq_config::HierarchicalFrequencyConfig = HierarchicalFrequencyConfig()

    # Block type schedule
    block_schedule::BlockTypeSchedule = PROGRESSIVE

    # Attention config
    window_size::Int = 64

    # FFN config
    use_ffn::Bool = true
    ffn_expansion::Float32 = 1.5f0
    dropout_rate::Float32 = 0.1f0

    # Checkpointing
    checkpoint_every::Int = 4
end

"""
    create_deep_blocks(config)

Create a vector of OssammaBlockDeep instances based on config.
"""
function create_deep_blocks(config::DeepModelConfig)
    blocks = OssammaBlockDeep[]

    for i in 1:config.number_of_layers
        block_type = get_block_type(i, config.number_of_layers, config.block_schedule)

        block = OssammaBlockDeep(
            config.embedding_dimension,
            config.max_sequence_length,
            config.number_of_heads,
            config.time_dimension;
            layer_idx = i,
            total_layers = config.number_of_layers,
            block_type = block_type,
            freq_config = config.freq_config,
            dropout_rate = config.dropout_rate,
            use_ffn = config.use_ffn,
            ffn_expansion = config.ffn_expansion,
            use_parallel_scan = config.use_parallel_scan,
            use_layer_scale = config.use_layer_scale,
            layer_scale_init = config.layer_scale_init,
            use_stochastic_depth = config.use_stochastic_depth,
            stochastic_depth_rate = config.stochastic_depth_rate,
            window_size = config.window_size,
        )

        push!(blocks, block)
    end

    return blocks
end

"""
    print_model_summary(config)

Print a summary of the deep model configuration.
"""
function print_model_summary(config::DeepModelConfig)
    println("=" ^ 70)
    println("Deep Ossamma Model Configuration")
    println("=" ^ 70)
    println()
    println("Architecture:")
    println("  Layers:          $(config.number_of_layers)")
    println("  Dimension:       $(config.embedding_dimension)")
    println("  Heads:           $(config.number_of_heads)")
    println("  Sequence Length: $(config.max_sequence_length)")
    println("  Vocab Size:      $(config.vocab_size)")
    println()
    println("Deep Network Optimizations:")
    println("  Parallel Scan:      $(config.use_parallel_scan)")
    println("  Layer Scale:        $(config.use_layer_scale) (init=$(config.layer_scale_init))")
    println("  Stochastic Depth:   $(config.use_stochastic_depth) (rate=$(config.stochastic_depth_rate))")
    println("  Checkpoint Every:   $(config.checkpoint_every) layers")
    println()
    println("Block Schedule: $(config.block_schedule)")

    # Show block type distribution
    type_counts = Dict(:full => 0, :global_only => 0, :local_only => 0)
    for i in 1:config.number_of_layers
        bt = get_block_type(i, config.number_of_layers, config.block_schedule)
        type_counts[bt] += 1
    end
    println("  :full blocks:       $(type_counts[:full])")
    println("  :global_only:       $(type_counts[:global_only])")
    println("  :local_only:        $(type_counts[:local_only])")
    println()

    # Frequency summary
    frequency_summary(config.number_of_layers, config.freq_config)

    # Estimate parameters (rough)
    d = config.embedding_dimension
    L = config.number_of_layers
    V = config.vocab_size

    # Per block estimate
    full_block_params = 4 * d * d + 2 * d * d + d * d + round(Int, d * config.ffn_expansion * 2) * d
    global_block_params = 4 * d * d + d * d + round(Int, d * config.ffn_expansion * 2) * d
    local_block_params = 2 * d * d + round(Int, d * config.ffn_expansion * 2) * d

    total_block_params = (
        type_counts[:full] * full_block_params +
        type_counts[:global_only] * global_block_params +
        type_counts[:local_only] * local_block_params
    )

    embedding_params = V * d + config.max_sequence_length * d
    total_params = total_block_params + embedding_params

    println("Estimated Parameters: ~$(round(total_params / 1e6, digits=1))M")
    println("=" ^ 70)
end

# =============================================================================
# 8. PRESET CONFIGURATIONS
# =============================================================================

"""
    deep_48L_config(; kwargs...)

Create a 48-layer deep configuration (~120M params).
Recommended starting point for deep experiments.
"""
function deep_48L_config(; kwargs...)
    return DeepModelConfig(;
        embedding_dimension = 384,
        number_of_layers = 48,
        number_of_heads = 6,
        max_sequence_length = 4096,
        use_parallel_scan = true,
        use_layer_scale = true,
        layer_scale_init = 0.1f0,
        use_stochastic_depth = true,
        stochastic_depth_rate = 0.1f0,
        block_schedule = PROGRESSIVE,
        freq_config = HierarchicalFrequencyConfig(
            base_min_freq = 0.01f0,
            base_max_freq = 100.0f0,
            scaling_type = :exponential
        ),
        kwargs...
    )
end

"""
    ultra_96L_config(; kwargs...)

Create a 96-layer ultra-deep configuration (~100M params).
Very narrow but extremely deep for research.
"""
function ultra_96L_config(; kwargs...)
    return DeepModelConfig(;
        embedding_dimension = 256,
        number_of_layers = 96,
        number_of_heads = 4,
        max_sequence_length = 8192,
        use_parallel_scan = true,
        use_layer_scale = true,
        layer_scale_init = 0.01f0,  # Smaller for very deep
        use_stochastic_depth = true,
        stochastic_depth_rate = 0.2f0,  # Higher for very deep
        block_schedule = SANDWICH,  # Full at edges
        checkpoint_every = 6,
        freq_config = HierarchicalFrequencyConfig(
            base_min_freq = 0.001f0,  # Very low for 96L
            base_max_freq = 100.0f0,
            scaling_type = :logarithmic
        ),
        kwargs...
    )
end

"""
    long_context_config(; kwargs...)

Create a configuration optimized for very long sequences (16K+).
"""
function long_context_config(; kwargs...)
    return DeepModelConfig(;
        embedding_dimension = 512,
        number_of_layers = 32,
        number_of_heads = 8,
        max_sequence_length = 16384,
        use_parallel_scan = true,  # CRITICAL for 16K
        use_layer_scale = true,
        layer_scale_init = 0.1f0,
        use_stochastic_depth = true,
        stochastic_depth_rate = 0.1f0,
        block_schedule = PROGRESSIVE,
        window_size = 128,  # Larger windows for long context
        checkpoint_every = 4,
        freq_config = HierarchicalFrequencyConfig(
            base_min_freq = 0.001f0,  # Very low for long-range
            base_max_freq = 50.0f0,
            scaling_type = :exponential
        ),
        kwargs...
    )
end

end # module DeepScaling
