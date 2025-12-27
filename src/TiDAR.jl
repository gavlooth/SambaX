module TiDAR

"""
TiDAR - Token-level Iterative Drafting with AR Refinement

This module implements speculative decoding with OssammaDrafter and
an AR verifier (Granite 3B, Qwen3, Llama3).

Architecture:
1. Drafter: Deep Ossamma model predicts K tokens in parallel (diffusion-style)
2. Verifier: AR model (Granite 3B) validates/rejects predictions
3. Accept: Tokens that match verifier's top prediction
4. Reject: Re-draft from first rejection point

Key insight: Drafter uses O(T) complexity with parallel scan, enabling
deep (48-96L) narrow models that run fast on GPU.

Reference: docs/DEPTH_SCALING_STRATEGY.md
"""

using Lux
using LuxCore
using NNlib
using Random
using Statistics: mean
import TOML

# Import from parent module
import ..TimeConditionedLayerNorm
import ..LinearAttentionLayer
import ..DLinOSS
import ..DLinOSSParallel
import ..SwiGLU

# Import deep scaling utilities
import ..DeepScaling: HierarchicalFrequencyConfig, compute_layer_frequencies
import ..DeepScaling: LayerScaleConfig, init_layer_scale, apply_layer_scale
import ..DeepScaling: StochasticDepthConfig, layer_drop_rate

# =============================================================================
# Granite Model Constants (from HuggingFace config.json)
# =============================================================================
# Granite 3.1 uses vocab_size = 49155 (verified from ibm-granite/granite-3.1-2b-instruct)
# Granite 4.0 uses vocab_size = 49160 (granite-4.0-tiny-preview)
# Special tokens: bos=0, eos=0, pad=0 in HF; Julia uses 1-based indices (+1)

const GRANITE_VOCAB_SIZE = 49155
const GRANITE_4_VOCAB_SIZE = 49160
const GRANITE_BOS_TOKEN_ID = 1
const GRANITE_EOS_TOKEN_ID = 1
const GRANITE_PAD_TOKEN_ID = 1

# Granite model dimensions for reference:
# - Granite 2B: hidden=2048, layers=40, heads=32, kv_heads=8
# - Granite 3B MoE: hidden=1536, layers=32, heads=24, active_params=800M
# - Granite 8B: hidden=4096, layers=32, heads=32

# For TiDAR we use a [MASK] token - this needs to be added to tokenizer
# or we can use an unused token. Using last vocab position as mask (1-based).
const GRANITE_MASK_TOKEN_ID = GRANITE_VOCAB_SIZE
const GRANITE_4_MASK_TOKEN_ID = GRANITE_4_VOCAB_SIZE

# Legacy aliases for backward compatibility
const GRANITE_3B_VOCAB_SIZE = GRANITE_VOCAB_SIZE
const GRANITE_3B_MASK_TOKEN_ID = GRANITE_MASK_TOKEN_ID
const GRANITE_3B_PAD_TOKEN_ID = GRANITE_PAD_TOKEN_ID
const GRANITE_3B_EOS_TOKEN_ID = GRANITE_EOS_TOKEN_ID

export OssammaDrafterDeep, TiDARConfig
export granite_2b_drafter_config, granite_3b_drafter_config, granite_4_3b_drafter_config, granite_8b_drafter_config
export granite_drafter_deep_config
export GRANITE_VOCAB_SIZE, GRANITE_MASK_TOKEN_ID, GRANITE_4_VOCAB_SIZE, GRANITE_4_MASK_TOKEN_ID
export GRANITE_3B_VOCAB_SIZE, GRANITE_3B_MASK_TOKEN_ID  # Legacy

# =============================================================================
# Deep Drafter Block with All Optimizations
# =============================================================================

"""
    OssammaDrafterBlockDeep

Deep-optimized drafter block with:
- Hierarchical frequency ranges
- Layer scale initialization
- Stochastic depth
- Parallel scan (mandatory for speed)

This is the building block for TiDAR drafters.
"""
struct OssammaDrafterBlockDeep <: LuxCore.AbstractLuxLayer
    # Dimensions
    embedding_dimension::Int
    sequence_length::Int
    number_of_heads::Int
    time_dimension::Int
    state_dimension::Int

    # Deep scaling config
    layer_idx::Int
    total_layers::Int
    min_frequency::Float32
    max_frequency::Float32

    # Features
    dropout_rate::Float32
    use_ffn::Bool
    use_layer_scale::Bool
    layer_scale_init::Float32
    use_stochastic_depth::Bool
    stochastic_depth_rate::Float32

    # Layers
    InputNorm::TimeConditionedLayerNorm
    GluProjection::Lux.Dense
    LinearAttention::LinearAttentionLayer
    OscillatorLayer::DLinOSSParallel  # Always parallel for TiDAR
    Dropout::LuxCore.AbstractLuxLayer
    FFN::Union{SwiGLU, Nothing}
    OutputNorm::Lux.LayerNorm
end

function OssammaDrafterBlockDeep(
    embedding_dimension::Int,
    sequence_length::Int,
    number_of_heads::Int,
    time_dimension::Int;
    layer_idx::Int = 1,
    total_layers::Int = 1,
    state_dimension::Int = embedding_dimension,
    freq_config::HierarchicalFrequencyConfig = HierarchicalFrequencyConfig(),
    dropout_rate::Float32 = 0.1f0,
    use_ffn::Bool = true,
    ffn_expansion::Float32 = 1.5f0,
    use_layer_scale::Bool = true,
    layer_scale_init::Float32 = 0.1f0,
    use_stochastic_depth::Bool = true,
    stochastic_depth_rate::Float32 = 0.1f0,
    parallel_chunk_size::Int = 64,
)
    # Compute layer-specific frequencies
    min_freq, max_freq = compute_layer_frequencies(layer_idx, total_layers, freq_config)

    # Always use parallel scan for TiDAR (speed is critical)
    oscillator = DLinOSSParallel(
        embedding_dimension, state_dimension, embedding_dimension,
        min_freq, max_freq, 0.1f0;
        chunk_size = parallel_chunk_size
    )

    return OssammaDrafterBlockDeep(
        embedding_dimension,
        sequence_length,
        number_of_heads,
        time_dimension,
        state_dimension,
        layer_idx,
        total_layers,
        min_freq,
        max_freq,
        dropout_rate,
        use_ffn,
        use_layer_scale,
        layer_scale_init,
        use_stochastic_depth,
        stochastic_depth_rate,
        # Layers
        TimeConditionedLayerNorm(embedding_dimension, time_dimension),
        Lux.Dense(embedding_dimension => 2 * embedding_dimension),
        LinearAttentionLayer(embedding_dimension, sequence_length, number_of_heads, time_dimension),
        oscillator,
        Lux.Dropout(dropout_rate),
        use_ffn ? SwiGLU(embedding_dimension; expansion_factor = ffn_expansion) : nothing,
        Lux.LayerNorm((embedding_dimension,)),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, block::OssammaDrafterBlockDeep)
    ffn_params = if block.use_ffn && block.FFN !== nothing
        Lux.initialparameters(rng, block.FFN)
    else
        NamedTuple()
    end

    params = (
        InputNorm = Lux.initialparameters(rng, block.InputNorm),
        GluProjection = Lux.initialparameters(rng, block.GluProjection),
        LinearAttention = Lux.initialparameters(rng, block.LinearAttention),
        OscillatorLayer = Lux.initialparameters(rng, block.OscillatorLayer),
        Dropout = Lux.initialparameters(rng, block.Dropout),
        FFN = ffn_params,
        OutputNorm = Lux.initialparameters(rng, block.OutputNorm),
    )

    # Add layer scale if enabled
    if block.use_layer_scale
        scale_config = LayerScaleConfig(init_value = block.layer_scale_init)
        layer_scale = init_layer_scale(
            rng, block.embedding_dimension, scale_config,
            block.layer_idx, block.total_layers
        )
        params = merge(params, (layer_scale = layer_scale,))
    end

    return params
end

function Lux.initialstates(rng::Random.AbstractRNG, block::OssammaDrafterBlockDeep)
    ffn_state = if block.use_ffn && block.FFN !== nothing
        Lux.initialstates(rng, block.FFN)
    else
        NamedTuple()
    end

    return (
        InputNorm = Lux.initialstates(rng, block.InputNorm),
        GluProjection = Lux.initialstates(rng, block.GluProjection),
        LinearAttention = Lux.initialstates(rng, block.LinearAttention),
        OscillatorLayer = Lux.initialstates(rng, block.OscillatorLayer),
        Dropout = Lux.initialstates(rng, block.Dropout),
        FFN = ffn_state,
        OutputNorm = Lux.initialstates(rng, block.OutputNorm),
        training = true,
    )
end

function (block::OssammaDrafterBlockDeep)(inputs::Tuple, params, state)
    input_tensor, time_input = inputs
    training = get(state, :training, true)

    # Stochastic depth: skip layer during training
    if block.use_stochastic_depth && training
        drop_prob = layer_drop_rate(
            block.layer_idx, block.total_layers,
            StochasticDepthConfig(drop_rate = block.stochastic_depth_rate)
        )
        if rand() < drop_prob
            return input_tensor, state
        end
    end

    residual = input_tensor

    # 1. Time-Conditioned LayerNorm
    normalized, _alpha_unused, norm_state = block.InputNorm(
        input_tensor, time_input, params.InputNorm, state.InputNorm
    )

    # 2. GLU-Global Branch
    glu_projected, glu_proj_state = block.GluProjection(
        normalized, params.GluProjection, state.GluProjection
    )

    dim = block.embedding_dimension
    path_a = copy(selectdim(glu_projected, 1, 1:dim))
    path_b = copy(selectdim(glu_projected, 1, (dim+1):size(glu_projected, 1)))

    # LinearAttention (global context)
    attn_out, lin_attn_state = block.LinearAttention(
        (path_a, time_input), params.LinearAttention, state.LinearAttention
    )

    # Oscillator SSM (temporal memory)
    osc_out, osc_state = block.OscillatorLayer(
        path_b, params.OscillatorLayer, state.OscillatorLayer
    )

    # GLU gating
    output = attn_out .* NNlib.sigmoid.(osc_out)

    # 3. Dropout
    output, dropout_state = block.Dropout(output, params.Dropout, state.Dropout)

    # 4. SwiGLU FFN
    output, ffn_state = if block.use_ffn && block.FFN !== nothing
        block.FFN(output, params.FFN, state.FFN)
    else
        output, NamedTuple()
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

    new_state = (
        InputNorm = norm_state,
        GluProjection = glu_proj_state,
        LinearAttention = lin_attn_state,
        OscillatorLayer = osc_state,
        Dropout = dropout_state,
        FFN = ffn_state,
        OutputNorm = output_norm_state,
        training = training,
    )

    return output, new_state
end

# =============================================================================
# Sinusoidal Time Embedding
# =============================================================================

struct SinusoidalTimeEmbedding <: LuxCore.AbstractLuxLayer
    time_dimension::Int
    max_period::Float32
end

function SinusoidalTimeEmbedding(time_dimension::Int; max_period::Float32 = 10000.0f0)
    @assert time_dimension % 2 == 0 "time_dimension must be even"
    return SinusoidalTimeEmbedding(time_dimension, max_period)
end

Lux.initialparameters(::Random.AbstractRNG, ::SinusoidalTimeEmbedding) = (;)
Lux.initialstates(::Random.AbstractRNG, ::SinusoidalTimeEmbedding) = (;)

function (layer::SinusoidalTimeEmbedding)(t, params, state)
    half_dim = layer.time_dimension ÷ 2
    freqs = exp.(-log(layer.max_period) .* collect(Float32, 0:half_dim-1) ./ half_dim)

    if t isa Number
        angles = Float32(t) .* freqs
        embedding = vcat(sin.(angles), cos.(angles))
    else
        t_col = reshape(Float32.(t), 1, :)
        freqs_col = reshape(freqs, :, 1)
        angles = t_col .* freqs_col
        embedding = vcat(sin.(angles), cos.(angles))
    end

    return embedding, state
end

# =============================================================================
# Time MLP Embedding
# =============================================================================

struct TimeMLPEmbedding <: LuxCore.AbstractLuxLayer
    time_dimension::Int
    embedding_dimension::Int
    SinusoidalEmbed::SinusoidalTimeEmbedding
    MLP1::Lux.Dense
    MLP2::Lux.Dense
end

function TimeMLPEmbedding(time_dimension::Int, embedding_dimension::Int)
    return TimeMLPEmbedding(
        time_dimension,
        embedding_dimension,
        SinusoidalTimeEmbedding(time_dimension),
        Lux.Dense(time_dimension => embedding_dimension, NNlib.gelu),
        Lux.Dense(embedding_dimension => embedding_dimension),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::TimeMLPEmbedding)
    return (
        SinusoidalEmbed = Lux.initialparameters(rng, layer.SinusoidalEmbed),
        MLP1 = Lux.initialparameters(rng, layer.MLP1),
        MLP2 = Lux.initialparameters(rng, layer.MLP2),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::TimeMLPEmbedding)
    return (
        SinusoidalEmbed = Lux.initialstates(rng, layer.SinusoidalEmbed),
        MLP1 = Lux.initialstates(rng, layer.MLP1),
        MLP2 = Lux.initialstates(rng, layer.MLP2),
    )
end

function (layer::TimeMLPEmbedding)(t, params, state)
    sinusoidal, sin_state = layer.SinusoidalEmbed(t, params.SinusoidalEmbed, state.SinusoidalEmbed)
    hidden, mlp1_state = layer.MLP1(sinusoidal, params.MLP1, state.MLP1)
    output, mlp2_state = layer.MLP2(hidden, params.MLP2, state.MLP2)

    new_state = (
        SinusoidalEmbed = sin_state,
        MLP1 = mlp1_state,
        MLP2 = mlp2_state,
    )
    return output, new_state
end

# =============================================================================
# OssammaDrafterDeep - Full Deep Drafter Model for TiDAR
# =============================================================================

"""
    OssammaDrafterDeep

Deep Ossamma drafter model optimized for TiDAR speculative decoding.

Features:
- 48-96 layers (leveraging O(T) complexity)
- Hierarchical oscillator frequencies
- Layer scale + stochastic depth
- Parallel scan (mandatory)
- Matches Granite 3B vocabulary

Architecture:
```
token_ids + diffusion_time_t
    ↓
TokenEmbedding + PositionEmbedding
    ↓
TimeEmbedding(t) → time_emb
    ↓
N × OssammaDrafterBlockDeep(hidden, time_emb)
    ↓
Final LayerNorm
    ↓
LM Head (d → vocab_size)
    ↓
logits
```
"""
struct OssammaDrafterDeep{E,P,T,N,L} <: LuxCore.AbstractLuxLayer
    # Configuration
    vocab_size::Int
    max_sequence_length::Int
    embedding_dimension::Int
    number_of_heads::Int
    number_of_layers::Int
    time_dimension::Int
    mask_token_id::Int

    # Layers
    TokenEmbedding::E
    PositionEmbedding::P
    TimeEmbedding::T
    Blocks::Vector{OssammaDrafterBlockDeep}
    FinalNorm::N
    LMHead::L
end

# =============================================================================
# TiDAR Configuration
# =============================================================================

"""
    TiDARConfig

Configuration for TiDAR drafter model.

The drafter vocabulary MUST match the AR verifier.
Default Granite config uses vocab_size = 49155; override for other verifiers.
"""
Base.@kwdef struct TiDARConfig
    # AR verifier identification
    ar_model::String = "granite_3b"

    # Vocabulary (MUST match Granite: 49155 tokens)
    vocab_size::Int = GRANITE_VOCAB_SIZE
    mask_token_id::Int = GRANITE_MASK_TOKEN_ID  # Can be > vocab_size to reserve an extra token

    # Drafter architecture
    max_sequence_length::Int = 4096
    embedding_dimension::Int = 384
    number_of_heads::Int = 6
    number_of_layers::Int = 48
    time_dimension::Int = 64

    # Deep scaling optimizations
    use_layer_scale::Bool = true
    layer_scale_init::Float32 = 0.1f0
    use_stochastic_depth::Bool = true
    stochastic_depth_rate::Float32 = 0.1f0
    freq_config::HierarchicalFrequencyConfig = HierarchicalFrequencyConfig()

    # FFN configuration
    use_ffn::Bool = true
    ffn_expansion::Float32 = 1.5f0
    dropout_rate::Float32 = 0.1f0

    # TiDAR inference settings
    draft_length::Int = 8          # Tokens to draft per step
    confidence_threshold::Float32 = 0.8f0
    temperature::Float32 = 0.9f0
end

# =============================================================================
# Granite Model-Specific Drafter Configurations
# =============================================================================
# All drafters match their verifier vocab but have different sizes
# based on the target verifier model.

"""
    granite_2b_drafter_config(; kwargs...)

Drafter config for Granite 2B verifier.
- Verifier: 2B params, hidden=2048, layers=40
- Drafter: ~40M params (24L × 384d)
"""
function granite_2b_drafter_config(; kwargs...)
    return TiDARConfig(;
        ar_model = "granite_2b",
        vocab_size = GRANITE_VOCAB_SIZE,
        mask_token_id = GRANITE_MASK_TOKEN_ID,
        embedding_dimension = 384,
        number_of_layers = 24,
        number_of_heads = 6,
        max_sequence_length = 4096,
        use_layer_scale = true,
        layer_scale_init = 0.1f0,
        use_stochastic_depth = true,
        stochastic_depth_rate = 0.05f0,
        draft_length = 8,
        kwargs...
    )
end

"""
    granite_3b_drafter_config(; kwargs...)

Drafter config for Granite 3B MoE verifier.
- Verifier: 3B params (800M active), hidden=1536, layers=32
- Drafter: ~60M params (32L × 384d)
"""
function granite_3b_drafter_config(; kwargs...)
    return TiDARConfig(;
        ar_model = "granite_3b",
        vocab_size = GRANITE_VOCAB_SIZE,
        mask_token_id = GRANITE_MASK_TOKEN_ID,
        embedding_dimension = 384,
        number_of_layers = 32,
        number_of_heads = 6,
        max_sequence_length = 4096,
        use_layer_scale = true,
        layer_scale_init = 0.1f0,
        use_stochastic_depth = true,
        stochastic_depth_rate = 0.05f0,
        freq_config = HierarchicalFrequencyConfig(
            base_min_freq = 0.01f0,
            base_max_freq = 100.0f0,
            scaling_type = :exponential
        ),
        draft_length = 8,
        kwargs...
    )
end

"""
    granite_4_3b_drafter_config(; kwargs...)

Drafter config for Granite 4.0 3B verifier.
- Verifier: Granite 4.0 3B
- Drafter: ~60M params (32L × 384d)
"""
function granite_4_3b_drafter_config(; kwargs...)
    return TiDARConfig(;
        ar_model = "granite4_3b",
        vocab_size = GRANITE_4_VOCAB_SIZE,
        mask_token_id = GRANITE_4_MASK_TOKEN_ID,
        embedding_dimension = 384,
        number_of_layers = 32,
        number_of_heads = 6,
        max_sequence_length = 4096,
        use_layer_scale = true,
        layer_scale_init = 0.1f0,
        use_stochastic_depth = true,
        stochastic_depth_rate = 0.05f0,
        freq_config = HierarchicalFrequencyConfig(
            base_min_freq = 0.01f0,
            base_max_freq = 100.0f0,
            scaling_type = :exponential
        ),
        draft_length = 8,
        kwargs...
    )
end

"""
    granite_8b_drafter_config(; kwargs...)

Drafter config for Granite 8B verifier.
- Verifier: 8B params, hidden=4096, layers=32
- Drafter: ~80M params (48L × 384d)
"""
function granite_8b_drafter_config(; kwargs...)
    return TiDARConfig(;
        ar_model = "granite_8b",
        vocab_size = GRANITE_VOCAB_SIZE,
        mask_token_id = GRANITE_MASK_TOKEN_ID,
        embedding_dimension = 384,
        number_of_layers = 48,
        number_of_heads = 6,
        max_sequence_length = 8192,
        use_layer_scale = true,
        layer_scale_init = 0.1f0,
        use_stochastic_depth = true,
        stochastic_depth_rate = 0.1f0,
        freq_config = HierarchicalFrequencyConfig(
            base_min_freq = 0.01f0,
            base_max_freq = 100.0f0,
            scaling_type = :exponential
        ),
        draft_length = 12,
        kwargs...
    )
end

"""
    granite_drafter_deep_config(; kwargs...)

Deep drafter config (48-96 layers) leveraging O(T) complexity.
Recommended for maximum acceptance rate.

- Drafter: ~80-100M params (48-64L × 384d)
- Uses all deep scaling strategies
"""
function granite_drafter_deep_config(; kwargs...)
    return TiDARConfig(;
        ar_model = "granite_deep",
        vocab_size = GRANITE_VOCAB_SIZE,
        mask_token_id = GRANITE_MASK_TOKEN_ID,
        embedding_dimension = 384,
        number_of_layers = 48,
        number_of_heads = 6,
        max_sequence_length = 8192,
        use_layer_scale = true,
        layer_scale_init = 0.1f0,
        use_stochastic_depth = true,
        stochastic_depth_rate = 0.1f0,
        freq_config = HierarchicalFrequencyConfig(
            base_min_freq = 0.01f0,
            base_max_freq = 100.0f0,
            decay_rate = 3.0f0,
            scaling_type = :exponential
        ),
        draft_length = 12,
        temperature = 0.9f0,
        kwargs...
    )
end

# Legacy alias
granite_3b_drafter_deep_config(; kwargs...) = granite_drafter_deep_config(; ar_model="granite_3b", kwargs...)

"""
    OssammaDrafterDeep(config::TiDARConfig)

Create an OssammaDrafterDeep from a TiDARConfig.
"""
function OssammaDrafterDeep(config::TiDARConfig)
    # Create deep blocks
    blocks = [
        OssammaDrafterBlockDeep(
            config.embedding_dimension,
            config.max_sequence_length,
            config.number_of_heads,
            config.time_dimension;
            layer_idx = i,
            total_layers = config.number_of_layers,
            freq_config = config.freq_config,
            dropout_rate = config.dropout_rate,
            use_ffn = config.use_ffn,
            ffn_expansion = config.ffn_expansion,
            use_layer_scale = config.use_layer_scale,
            layer_scale_init = config.layer_scale_init,
            use_stochastic_depth = config.use_stochastic_depth,
            stochastic_depth_rate = config.stochastic_depth_rate,
        )
        for i in 1:config.number_of_layers
    ]

    actual_vocab_size = max(config.vocab_size, config.mask_token_id)

    return OssammaDrafterDeep(
        actual_vocab_size,
        config.max_sequence_length,
        config.embedding_dimension,
        config.number_of_heads,
        config.number_of_layers,
        config.time_dimension,
        config.mask_token_id,
        # Layers
        Lux.Embedding(actual_vocab_size => config.embedding_dimension),
        Lux.Embedding(config.max_sequence_length => config.embedding_dimension),
        TimeMLPEmbedding(config.time_dimension, config.embedding_dimension),
        blocks,
        Lux.LayerNorm((config.embedding_dimension,)),
        Lux.Dense(config.embedding_dimension => actual_vocab_size),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, model::OssammaDrafterDeep)
    block_params = [Lux.initialparameters(rng, block) for block in model.Blocks]

    return (
        TokenEmbedding = Lux.initialparameters(rng, model.TokenEmbedding),
        PositionEmbedding = Lux.initialparameters(rng, model.PositionEmbedding),
        TimeEmbedding = Lux.initialparameters(rng, model.TimeEmbedding),
        Blocks = block_params,
        FinalNorm = Lux.initialparameters(rng, model.FinalNorm),
        LMHead = Lux.initialparameters(rng, model.LMHead),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, model::OssammaDrafterDeep)
    block_states = [Lux.initialstates(rng, block) for block in model.Blocks]

    return (
        TokenEmbedding = Lux.initialstates(rng, model.TokenEmbedding),
        PositionEmbedding = Lux.initialstates(rng, model.PositionEmbedding),
        TimeEmbedding = Lux.initialstates(rng, model.TimeEmbedding),
        Blocks = block_states,
        FinalNorm = Lux.initialstates(rng, model.FinalNorm),
        LMHead = Lux.initialstates(rng, model.LMHead),
    )
end

"""
    (model::OssammaDrafterDeep)(token_ids, t, params, state)

Forward pass of the deep drafter.

# Arguments
- `token_ids`: (seq_len, batch) or (seq_len,) - token IDs (may include [MASK])
- `t`: scalar or (batch,) - diffusion timestep in [0, 1]
- `params`: Model parameters
- `state`: Model state

# Returns
- `logits`: (vocab_size, seq_len, batch) - predictions for all positions
- `new_state`: Updated state
"""
function (model::OssammaDrafterDeep)(token_ids, t, params, state)
    # Handle input dimensions
    if ndims(token_ids) == 1
        token_ids = reshape(token_ids, :, 1)
        was_unbatched = true
    else
        was_unbatched = false
    end

    seq_len, batch_size = size(token_ids)

    # 1. Token Embeddings
    token_flat = vec(token_ids)
    token_emb_flat, tok_state = model.TokenEmbedding(
        token_flat, params.TokenEmbedding, state.TokenEmbedding
    )
    token_emb = reshape(token_emb_flat, model.embedding_dimension, seq_len, batch_size)

    # 2. Position Embeddings
    position_indices = collect(1:seq_len)
    pos_emb_raw, pos_state = model.PositionEmbedding(
        position_indices, params.PositionEmbedding, state.PositionEmbedding
    )
    pos_emb = reshape(pos_emb_raw, model.embedding_dimension, seq_len, 1)

    # 3. Time Embedding
    t_input = t isa Number ? fill(Float32(t), batch_size) : Float32.(t)
    sinusoidal_emb, sin_state = model.TimeEmbedding.SinusoidalEmbed(
        t_input, params.TimeEmbedding.SinusoidalEmbed, state.TimeEmbedding.SinusoidalEmbed
    )
    time_emb, time_state = model.TimeEmbedding(
        t_input, params.TimeEmbedding, state.TimeEmbedding
    )

    # 4. Combine Embeddings
    hidden = token_emb .+ pos_emb

    # 5. Apply Deep Blocks
    block_states = []
    for (i, block) in enumerate(model.Blocks)
        hidden, blk_state = block((hidden, sinusoidal_emb), params.Blocks[i], state.Blocks[i])
        push!(block_states, blk_state)
    end

    # 6. Final LayerNorm
    hidden_flat = reshape(hidden, model.embedding_dimension, :)
    hidden_norm, norm_state = model.FinalNorm(
        hidden_flat, params.FinalNorm, state.FinalNorm
    )
    hidden = reshape(hidden_norm, model.embedding_dimension, seq_len, batch_size)

    # 7. LM Head
    hidden_flat = reshape(hidden, model.embedding_dimension, :)
    logits_flat, lm_state = model.LMHead(
        hidden_flat, params.LMHead, state.LMHead
    )
    logits = reshape(logits_flat, model.vocab_size, seq_len, batch_size)

    new_state = (
        TokenEmbedding = tok_state,
        PositionEmbedding = pos_state,
        TimeEmbedding = time_state,
        Blocks = block_states,
        FinalNorm = norm_state,
        LMHead = lm_state,
    )

    if was_unbatched
        logits = dropdims(logits, dims=3)
    end

    return logits, new_state
end

# =============================================================================
# TiDAR Inference Utilities
# =============================================================================

export draft_tokens, verify_and_accept, tidar_generate_step, tidar_generate_step_cached
export estimate_drafter_params, print_tidar_config

"""
    draft_tokens(model, prefix_ids, draft_length, params, state; temperature=0.9)

Draft `draft_length` tokens starting from `prefix_ids`.

Uses diffusion-style parallel prediction:
1. Append [MASK] tokens to prefix
2. Forward pass at t=0 (fully masked)
3. Sample from logits at mask positions

# Returns
- `drafted_ids`: (prefix_len + draft_length,) - full sequence with drafts
- `draft_logits`: (vocab, draft_length) - logits for drafted tokens
"""
function draft_tokens(
    model::OssammaDrafterDeep,
    prefix_ids::AbstractVector{<:Integer},
    draft_length::Int,
    params,
    state;
    temperature::Float32 = 0.9f0,
    mask_token_id::Int = GRANITE_MASK_TOKEN_ID,
)
    prefix_len = length(prefix_ids)
    total_len = prefix_len + draft_length

    # Create input with [MASK] tokens for draft positions
    input_ids = vcat(prefix_ids, fill(mask_token_id, draft_length))

    # Forward pass at t=0 (predict all masks)
    logits, new_state = model(input_ids, 0.0f0, params, state)

    # Extract logits for draft positions only
    draft_logits = logits[:, (prefix_len+1):total_len]  # (vocab, draft_length)

    # Sample from logits with temperature
    if temperature > 0
        probs = NNlib.softmax(draft_logits ./ temperature, dims=1)
        drafted_tokens = [sample_categorical(probs[:, i]) for i in 1:draft_length]
    else
        # Greedy
        drafted_tokens = [argmax(draft_logits[:, i]) for i in 1:draft_length]
    end

    drafted_ids = vcat(prefix_ids, drafted_tokens)

    return drafted_ids, draft_logits, new_state
end

"""
    sample_categorical(probs; rng=Random.default_rng())

Sample from a categorical distribution.
"""
function sample_categorical(probs::AbstractVector; rng = Random.default_rng())
    cumsum_probs = cumsum(probs)
    r = rand(rng, Float32)
    for (i, cp) in enumerate(cumsum_probs)
        if r <= cp
            return i
        end
    end
    return length(probs)
end

"""
    sample_from_probs(probs; top_p=1.0, rng=Random.default_rng())

Sample from a probability vector with optional top-p (nucleus) truncation.
"""
function sample_from_probs(
    probs::AbstractVector;
    top_p::Float32 = 1.0f0,
    rng = Random.default_rng()
)
    if top_p <= 0f0
        return argmax(probs)
    end
    if top_p >= 1f0
        return sample_categorical(probs; rng = rng)
    end

    sorted_indices = sortperm(probs, rev = true)
    cumsum_probs = cumsum(probs[sorted_indices])
    cutoff_idx = findfirst(x -> x > top_p, cumsum_probs)
    if cutoff_idx === nothing
        cutoff_idx = length(sorted_indices)
    end

    nucleus_mask = falses(length(probs))
    for i in 1:cutoff_idx
        nucleus_mask[sorted_indices[i]] = true
    end

    masked_probs = probs .* nucleus_mask
    masked_probs ./= sum(masked_probs)
    return sample_categorical(masked_probs; rng = rng)
end

"""
    verify_and_accept(drafted_ids, prefix_len, verifier_logits; kwargs...)

Verify drafted tokens against the AR verifier.

# Arguments
- `drafted_ids`: Full sequence including drafted tokens (1-based token IDs)
- `prefix_len`: Length of the prefix (known tokens)
- `verifier_logits`: (vocab, seq_len) from AR verifier

# Keyword Arguments
- `draft_logits`: (vocab, draft_length) logits from the drafter (required for :rejection)
- `temperature`: Softmax temperature
- `mode`: `:argmax` (greedy) or `:rejection` (rejection sampling)
- `top_p`: Top-p threshold for replacement sampling (rejection mode)
- `verifier_offset`: Extra token offset in verifier input (e.g., 1 if a BOS token is prepended)
- `rng`: Random number generator

# Returns
- `accepted_length`: Number of draft tokens accepted (0 to draft_length)
- `first_rejection_idx`: Position of first rejection (or nothing if all accepted)
- `replacement_token`: Token ID from verifier to use at rejection (or nothing if all accepted)
"""
function verify_and_accept(
    drafted_ids::AbstractVector{<:Integer},
    prefix_len::Int,
    verifier_logits::AbstractMatrix;
    draft_logits::Union{Nothing, AbstractMatrix} = nothing,
    temperature::Float32 = 1.0f0,
    mode::Symbol = :rejection,
    top_p::Float32 = 1.0f0,
    verifier_offset::Int = 0,
    rng = Random.default_rng(),
)
    draft_length = length(drafted_ids) - prefix_len
    accepted = 0

    if mode == :rejection && draft_logits === nothing
        error("verify_and_accept(mode=:rejection) requires draft_logits.")
    end
    if mode == :rejection && temperature <= 0f0
        error("verify_and_accept(mode=:rejection) requires temperature > 0. Use mode=:argmax for greedy verification.")
    end

    verifier_probs = mode == :rejection ? NNlib.softmax(verifier_logits ./ temperature, dims = 1) : nothing
    draft_probs = mode == :rejection ? NNlib.softmax(draft_logits ./ temperature, dims = 1) : nothing

    for i in 1:draft_length
        pos = prefix_len + i
        drafted_token = drafted_ids[pos]

        # Verifier predicts token at position `pos` given prefix[:pos-1]
        # So we look at logits[:, pos-1] for the prediction of token at pos
        verifier_idx = pos - 1 + verifier_offset
        if verifier_idx < 1
            error("verifier_logits must include predictions aligned to input tokens; provide a BOS token or set verifier_offset.")
        elseif verifier_idx > size(verifier_logits, 2)
            error("verifier_logits is too short for position $pos (index $verifier_idx).")
        end

        if mode == :argmax
            verifier_pred = argmax(verifier_logits[:, verifier_idx])
            if drafted_token == verifier_pred
                accepted += 1
            else
                return accepted, pos, verifier_pred
            end
        elseif mode == :rejection
            ar_probs = verifier_probs[:, verifier_idx]
            draft_p = draft_probs[drafted_token, i]
            ar_p = ar_probs[drafted_token]
            acceptance_prob = min(1.0f0, ar_p / (draft_p + 1f-10))

            if rand(rng) < acceptance_prob
                accepted += 1
            else
                replacement = sample_from_probs(ar_probs; top_p = top_p, rng = rng)
                return accepted, pos, replacement
            end
        else
            error("Unknown verification mode: $mode. Use :argmax or :rejection.")
        end
    end

    return accepted, nothing, nothing  # All accepted
end

"""
    tidar_generate_step(drafter, drafter_params, drafter_state,
                        verifier_fn, prefix_ids, draft_length; kwargs...)

One step of TiDAR generation.

# Arguments
- `drafter`: OssammaDrafterDeep model
- `drafter_params`: Drafter parameters
- `drafter_state`: Drafter state
- `verifier_fn`: Function (token_ids[, verifier_state]) -> logits or (logits, verifier_state)
- `prefix_ids`: Current prefix sequence
- `draft_length`: Number of tokens to draft

# Keyword Arguments
- `temperature`: Sampling temperature (used for draft + verification)
- `verify_mode`: `:argmax` or `:rejection` (default)
- `top_p`: Top-p threshold for replacement sampling (rejection mode)
- `add_bos_to_verifier`: Prepend BOS token to verifier input for alignment
- `bos_token_id`: BOS token ID (1-based)
- `rng`: Random number generator

# Returns
- `new_prefix`: Extended prefix with accepted tokens
- `tokens_accepted`: Number of tokens accepted this step
- `new_drafter_state`: Updated drafter state
"""
function tidar_generate_step(
    drafter::OssammaDrafterDeep,
    drafter_params,
    drafter_state,
    verifier_fn::Function,
    prefix_ids::AbstractVector{<:Integer},
    draft_length::Int;
    temperature::Float32 = 0.9f0,
    verify_mode::Symbol = :rejection,
    top_p::Float32 = 1.0f0,
    add_bos_to_verifier::Bool = true,
    bos_token_id::Int = GRANITE_BOS_TOKEN_ID,
    rng = Random.default_rng(),
)
    # 1. Draft K tokens
    drafted_ids, draft_logits, new_drafter_state = draft_tokens(
        drafter, prefix_ids, draft_length, drafter_params, drafter_state;
        temperature = temperature,
        mask_token_id = drafter.mask_token_id,
    )

    # 2. Get verifier's predictions for the full sequence
    verifier_input = add_bos_to_verifier ? vcat(bos_token_id, drafted_ids) : drafted_ids
    verifier_logits = verifier_fn(verifier_input)
    verifier_offset = add_bos_to_verifier ? 1 : 0

    # 3. Verify and find acceptance point
    prefix_len = length(prefix_ids)
    accepted, rejection_idx, replacement = verify_and_accept(
        drafted_ids, prefix_len, verifier_logits;
        draft_logits = draft_logits,
        temperature = temperature,
        mode = verify_mode,
        top_p = top_p,
        verifier_offset = verifier_offset,
        rng = rng
    )

    # 4. Build new prefix
    if accepted == draft_length
        # All accepted
        new_prefix = drafted_ids
    elseif rejection_idx !== nothing
        # Partial acceptance - keep up to rejection, use verifier's prediction
        new_prefix = drafted_ids[1:rejection_idx-1]
        # Add verifier's token at rejection point (replacement)
        verifier_token = replacement === nothing ?
            argmax(verifier_logits[:, rejection_idx - 1 + verifier_offset]) :
            replacement
        push!(new_prefix, verifier_token)
    else
        # No tokens accepted - use verifier's first prediction
        verifier_token = replacement === nothing ?
            argmax(verifier_logits[:, prefix_len + verifier_offset]) :
            replacement
        new_prefix = vcat(prefix_ids, [verifier_token])
    end

    return new_prefix, accepted, new_drafter_state
end

"""
    tidar_generate_step_cached(drafter, drafter_params, drafter_state,
                               verifier_fn, verifier_state, prefix_ids, draft_length; kwargs...)

One step of TiDAR generation with a verifier cache/state.

# Arguments
- `drafter`: OssammaDrafterDeep model
- `drafter_params`: Drafter parameters
- `drafter_state`: Drafter state
- `verifier_fn`: Function (token_ids[, verifier_state]) -> logits or (logits, verifier_state)
- `verifier_state`: Verifier cache/state (opaque)
- `prefix_ids`: Current prefix sequence
- `draft_length`: Number of tokens to draft

# Keyword Arguments
- `temperature`: Sampling temperature (used for draft + verification)
- `verify_mode`: `:argmax` or `:rejection` (default)
- `top_p`: Top-p threshold for replacement sampling (rejection mode)
- `add_bos_to_verifier`: Prepend BOS token to verifier input for alignment
- `bos_token_id`: BOS token ID (1-based)
- `rng`: Random number generator

# Returns
- `new_prefix`: Extended prefix with accepted tokens
- `tokens_accepted`: Number of tokens accepted this step
- `new_drafter_state`: Updated drafter state
- `new_verifier_state`: Updated verifier cache/state
"""
function tidar_generate_step_cached(
    drafter::OssammaDrafterDeep,
    drafter_params,
    drafter_state,
    verifier_fn::Function,
    verifier_state,
    prefix_ids::AbstractVector{<:Integer},
    draft_length::Int;
    temperature::Float32 = 0.9f0,
    verify_mode::Symbol = :rejection,
    top_p::Float32 = 1.0f0,
    add_bos_to_verifier::Bool = true,
    bos_token_id::Int = GRANITE_BOS_TOKEN_ID,
    rng = Random.default_rng(),
)
    # 1. Draft K tokens
    drafted_ids, draft_logits, new_drafter_state = draft_tokens(
        drafter, prefix_ids, draft_length, drafter_params, drafter_state;
        temperature = temperature,
        mask_token_id = drafter.mask_token_id,
    )

    # 2. Get verifier's predictions for the full sequence (with optional BOS)
    verifier_input = add_bos_to_verifier ? vcat(bos_token_id, drafted_ids) : drafted_ids

    verifier_out = if applicable(verifier_fn, verifier_input, verifier_state)
        verifier_fn(verifier_input, verifier_state)
    else
        verifier_fn(verifier_input)
    end

    verifier_logits, new_verifier_state = verifier_out isa Tuple ? verifier_out : (verifier_out, verifier_state)
    verifier_offset = add_bos_to_verifier ? 1 : 0

    # 3. Verify and find acceptance point
    prefix_len = length(prefix_ids)
    accepted, rejection_idx, replacement = verify_and_accept(
        drafted_ids, prefix_len, verifier_logits;
        draft_logits = draft_logits,
        temperature = temperature,
        mode = verify_mode,
        top_p = top_p,
        verifier_offset = verifier_offset,
        rng = rng
    )

    # 4. Build new prefix
    if accepted == draft_length
        new_prefix = drafted_ids
    elseif rejection_idx !== nothing
        new_prefix = drafted_ids[1:rejection_idx-1]
        verifier_token = replacement === nothing ?
            argmax(verifier_logits[:, rejection_idx - 1 + verifier_offset]) :
            replacement
        push!(new_prefix, verifier_token)
    else
        verifier_token = replacement === nothing ?
            argmax(verifier_logits[:, prefix_len + verifier_offset]) :
            replacement
        new_prefix = vcat(prefix_ids, [verifier_token])
    end

    return new_prefix, accepted, new_drafter_state, new_verifier_state
end

"""
    estimate_drafter_params(config::TiDARConfig)

Estimate parameter count for a TiDAR drafter.
"""
function estimate_drafter_params(config::TiDARConfig)
    d = config.embedding_dimension
    L = config.number_of_layers
    V = max(config.vocab_size, config.mask_token_id)
    S = config.max_sequence_length
    t_dim = config.time_dimension

    # Embeddings
    tok_emb = V * d
    pos_emb = S * d
    time_emb = t_dim * d + d * d + d * d

    # Per block (GLU + LinearAttn + DLinOSS + FFN)
    per_block = 2 * d * d +  # GLU projection
                4 * d * d +  # LinearAttn (Q, K, V, O)
                d * d +      # DLinOSS projections
                round(Int, d * config.ffn_expansion * 2) * d  # SwiGLU

    blocks_total = per_block * L

    # LM Head
    lm_head = d * V

    total = tok_emb + pos_emb + time_emb + blocks_total + lm_head

    return (
        token_embedding = tok_emb,
        position_embedding = pos_emb,
        time_embedding = time_emb,
        blocks = blocks_total,
        lm_head = lm_head,
        total = total,
    )
end

"""
    print_tidar_config(config::TiDARConfig)

Print a summary of the TiDAR configuration.
"""
function print_tidar_config(config::TiDARConfig)
    params = estimate_drafter_params(config)

    println("=" ^ 70)
    println("TiDAR Drafter Configuration")
    println("=" ^ 70)
    println()
    println("AR Verifier: $(config.ar_model)")
    actual_vocab = max(config.vocab_size, config.mask_token_id)
    println("Vocab Size:  $(config.vocab_size) (effective=$(actual_vocab))")
    println("Mask Token:  $(config.mask_token_id)")
    println()
    println("Architecture:")
    println("  Layers:          $(config.number_of_layers)")
    println("  Dimension:       $(config.embedding_dimension)")
    println("  Heads:           $(config.number_of_heads)")
    println("  Sequence Length: $(config.max_sequence_length)")
    println()
    println("Deep Scaling:")
    println("  Layer Scale:       $(config.use_layer_scale) (init=$(config.layer_scale_init))")
    println("  Stochastic Depth:  $(config.use_stochastic_depth) (rate=$(config.stochastic_depth_rate))")
    println("  Hierarchical Freq: $(config.freq_config.scaling_type)")
    println()
    println("TiDAR Settings:")
    println("  Draft Length:       $(config.draft_length) tokens")
    println("  Confidence Thresh:  $(config.confidence_threshold)")
    println("  Temperature:        $(config.temperature)")
    println()
    println("Estimated Parameters:")
    println("  Embeddings:    $(round(params.token_embedding / 1e6, digits=1))M")
    println("  Blocks:        $(round(params.blocks / 1e6, digits=1))M")
    println("  LM Head:       $(round(params.lm_head / 1e6, digits=1))M")
    println("  Total:         $(round(params.total / 1e6, digits=1))M")
    println()

    # Complexity comparison
    T = config.max_sequence_length
    d = config.embedding_dimension
    println("Complexity Advantage (vs Transformer):")
    println("  Sequence length T = $(T)")
    println("  Transformer: O(T² × d) = O($(T^2 * d / 1e9) B)")
    println("  Ossamma:     O(T × d²) = O($(T * d^2 / 1e9) B)")
    println("  Ratio: $(round(T / d, digits=1))× more layers possible")
    println("=" ^ 70)
end

end # module TiDAR
