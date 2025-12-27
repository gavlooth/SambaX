# Drafter.jl - Simplified Ossamma block for TiDAR-style drafting
#
# This is Option C: LinearAttention + DLinOSS only, no SWAttention.
# The hypothesis: AR verifier handles grammar, drafter needs global semantics.

module Drafter

using Lux
using LuxCore
using NNlib
using Random
using Statistics: mean
import TOML

# Import from parent module (will be included after Ossamma.jl)
import ..TimeConditionedLayerNorm
import ..LinearAttentionLayer
import ..DLinOSS
import ..DLinOSSParallel
import ..SwiGLU

export OssammaDrafterBlock

# =============================================================================
# OssammaDrafterBlock - Simplified block for language model drafting
# =============================================================================
#
# Architecture (Transformer-style residuals):
#
#   Input ─────────────────────────────────┐
#       ↓                                  │
#   Time-Conditioned LayerNorm             │
#       ↓                                  │
#   Dense(d → 2d) → split                  │
#       ↓              ↓                   │
#   LinearAttn      DLinOSS                │
#       ↓              ↓                   │
#       └─── ⊙ sigmoid ─┘                  │
#              ↓                           │
#          Dropout                         │
#              ↓                           │
#        SwiGLU FFN                        │
#              ↓                           │
#           + ←────────────────────────────┘  (standard residual)
#              ↓
#         LayerNorm
#              ↓
#           Output
#
# Key differences from OssammaNERBlock:
#   - No SlidingWindowAttention (AR verifier handles local patterns)
#   - No α-mixing gating - uses standard residual like transformers
#   - Simpler: drafter proposes, AR verifier decides

struct OssammaDrafterBlock <: LuxCore.AbstractLuxLayer
    # Dimensions
    embedding_dimension::Int
    sequence_length::Int
    number_of_heads::Int
    time_dimension::Int
    state_dimension::Int
    dropout_rate::Float32
    use_ffn::Bool
    use_parallel_scan::Bool

    # Layers
    InputNorm::TimeConditionedLayerNorm
    GluProjection::Lux.Dense           # d → 2d
    LinearAttention::LinearAttentionLayer
    OscillatorLayer::Union{DLinOSS, DLinOSSParallel}
    Dropout::Lux.Dropout
    FFN::Union{SwiGLU, Nothing}
    OutputNorm::Lux.LayerNorm
end

"""
    OssammaDrafterBlock(embedding_dimension, sequence_length, number_of_heads, time_dimension; kwargs...)

Create a simplified Ossamma block for TiDAR-style drafting.

This block uses only the GLU-Global branch (LinearAttention + DLinOSS),
removing the Local-Sharp branch (SWAttention) and α-mixing.

# Arguments
- `embedding_dimension::Int`: Model dimension
- `sequence_length::Int`: Maximum sequence length
- `number_of_heads::Int`: Number of attention heads
- `time_dimension::Int`: Dimension for time embeddings

# Keyword Arguments
- `state_dimension::Int`: Oscillator state dimension (default: embedding_dimension)
- `min_frequency::Float32`: Minimum oscillator frequency (default: 0.1)
- `max_frequency::Float32`: Maximum oscillator frequency (default: 10.0)
- `default_time_step::Float32`: Default Δt for oscillators (default: 0.1)
- `dropout_rate::Float32`: Dropout rate (default: 0.1)
- `use_ffn::Bool`: Enable SwiGLU FFN (default: true)
- `ffn_expansion::Float32`: FFN expansion factor (default: 1.5)
- `use_parallel_scan::Bool`: Use parallel scan for oscillators (default: false)
- `parallel_chunk_size::Int`: Chunk size for parallel scan (default: 64)
"""
function OssammaDrafterBlock(
    embedding_dimension::Int,
    sequence_length::Int,
    number_of_heads::Int,
    time_dimension::Int;
    state_dimension::Int = embedding_dimension,
    min_frequency::Float32 = 0.1f0,
    max_frequency::Float32 = 10.0f0,
    default_time_step::Float32 = 0.1f0,
    dropout_rate::Float32 = 0.1f0,
    use_ffn::Bool = true,
    ffn_expansion::Float32 = 3f0 / 2f0,
    use_parallel_scan::Bool = false,
    parallel_chunk_size::Int = 64,
)
    # Choose oscillator implementation
    oscillator_layer = if use_parallel_scan
        DLinOSSParallel(
            embedding_dimension, state_dimension, embedding_dimension,
            min_frequency, max_frequency, default_time_step;
            chunk_size = parallel_chunk_size
        )
    else
        DLinOSS(
            embedding_dimension, state_dimension, embedding_dimension,
            min_frequency, max_frequency, default_time_step
        )
    end

    return OssammaDrafterBlock(
        embedding_dimension,
        sequence_length,
        number_of_heads,
        time_dimension,
        state_dimension,
        dropout_rate,
        use_ffn,
        use_parallel_scan,
        # Layers
        TimeConditionedLayerNorm(embedding_dimension, time_dimension),
        Lux.Dense(embedding_dimension => 2 * embedding_dimension),
        LinearAttentionLayer(embedding_dimension, sequence_length, number_of_heads, time_dimension),
        oscillator_layer,
        Lux.Dropout(dropout_rate),
        use_ffn ? SwiGLU(embedding_dimension; expansion_factor = ffn_expansion) : nothing,
        Lux.LayerNorm((embedding_dimension,)),
    )
end

# =============================================================================
# Parameter and State Initialization
# =============================================================================

function Lux.initialparameters(rng::Random.AbstractRNG, block::OssammaDrafterBlock)
    ffn_params = if block.use_ffn && block.FFN !== nothing
        Lux.initialparameters(rng, block.FFN)
    else
        NamedTuple()
    end

    return (
        InputNorm = Lux.initialparameters(rng, block.InputNorm),
        GluProjection = Lux.initialparameters(rng, block.GluProjection),
        LinearAttention = Lux.initialparameters(rng, block.LinearAttention),
        OscillatorLayer = Lux.initialparameters(rng, block.OscillatorLayer),
        Dropout = Lux.initialparameters(rng, block.Dropout),
        FFN = ffn_params,
        OutputNorm = Lux.initialparameters(rng, block.OutputNorm),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, block::OssammaDrafterBlock)
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
    )
end

# =============================================================================
# Forward Pass
# =============================================================================

function (block::OssammaDrafterBlock)(inputs::Tuple, params, state)
    input_tensor, time_input = inputs
    # input_tensor: (embedding_dim, seq_len, batch) or (embedding_dim, seq_len)
    # time_input: (time_dim, batch) or (time_dim,)

    residual = input_tensor

    # =========================================================================
    # 1. Time-Conditioned LayerNorm
    # =========================================================================
    # TimeConditionedLayerNorm returns (output, alpha_bias, state)
    # We discard alpha_bias - standard residual used instead
    normalized, _alpha_unused, norm_state = block.InputNorm(
        input_tensor, time_input, params.InputNorm, state.InputNorm
    )

    # =========================================================================
    # 2. GLU-Global Branch
    # =========================================================================
    # Project to 2*dim and split
    glu_projected, glu_proj_state = block.GluProjection(
        normalized, params.GluProjection, state.GluProjection
    )

    # Split into path_a (LinearAttention) and path_b (Oscillator)
    dim = block.embedding_dimension
    path_a = copy(selectdim(glu_projected, 1, 1:dim))
    path_b = copy(selectdim(glu_projected, 1, (dim+1):size(glu_projected, 1)))

    # path_a → Linear Attention (global O(n) context)
    attn_out, lin_attn_state = block.LinearAttention(
        (path_a, time_input), params.LinearAttention, state.LinearAttention
    )

    # path_b → Oscillator SSM (sequential memory)
    osc_out, osc_state = block.OscillatorLayer(
        path_b, params.OscillatorLayer, state.OscillatorLayer
    )

    # GLU gating: attn_out ⊙ sigmoid(osc_out)
    # No output projection needed - SwiGLU FFN will transform
    output = attn_out .* NNlib.sigmoid.(osc_out)

    # =========================================================================
    # 3. Dropout
    # =========================================================================
    output, dropout_state = block.Dropout(
        output, params.Dropout, state.Dropout
    )

    # =========================================================================
    # 4. SwiGLU FFN
    # =========================================================================
    output, ffn_state = if block.use_ffn && block.FFN !== nothing
        block.FFN(output, params.FFN, state.FFN)
    else
        output, NamedTuple()
    end

    # =========================================================================
    # 5. Standard Residual + Output LayerNorm (transformer-style)
    # =========================================================================
    output_pre_norm = residual .+ output  # y = x + f(x)

    # Apply output LayerNorm
    output_flat = reshape(output_pre_norm, block.embedding_dimension, :)
    output_norm_flat, output_norm_state = block.OutputNorm(
        output_flat, params.OutputNorm, state.OutputNorm
    )
    output = reshape(output_norm_flat, size(output_pre_norm))

    # =========================================================================
    # 6. Collect new states
    # =========================================================================
    new_state = (
        InputNorm = norm_state,
        GluProjection = glu_proj_state,
        LinearAttention = lin_attn_state,
        OscillatorLayer = osc_state,
        Dropout = dropout_state,
        FFN = ffn_state,
        OutputNorm = output_norm_state,
    )

    return output, new_state
end

# =============================================================================
# OssammaDrafter - Full language model drafter for TiDAR-style generation
# =============================================================================
#
# Architecture:
#   token_ids + diffusion_time_t
#       ↓
#   TokenEmbedding + PositionEmbedding
#       ↓
#   TimeMLPEmbedding(t) → time_emb
#       ↓
#   N × OssammaDrafterBlock(hidden, time_emb)
#       ↓
#   Final LayerNorm
#       ↓
#   LM Head (d → vocab_size)
#       ↓
#   logits
#
# For TiDAR:
#   - Drafter predicts [MASK] tokens in parallel (diffusion)
#   - AR verifier validates via rejection sampling

export OssammaDrafter
export GRANITE_VOCAB_SIZE, QWEN3_VOCAB_SIZE, LLAMA3_VOCAB_SIZE
export DrafterConfig, load_drafter_config, default_granite_config, default_qwen3_config

# =============================================================================
# Vocabulary Constants for Supported AR Models
# =============================================================================
# Verified vocab sizes from HuggingFace config.json files:
const GRANITE_VOCAB_SIZE = 49155    # IBM Granite 3.1 (granite-3.1-2b-instruct)
const GRANITE_4_VOCAB_SIZE = 49160  # IBM Granite 4.0 (granite-4.0-tiny-preview)
const QWEN3_VOCAB_SIZE = 151936     # Alibaba Qwen3
const LLAMA3_VOCAB_SIZE = 128256    # Meta Llama 3.x

# Mask token IDs for TiDAR (Julia 1-based indices)
# Using last vocab position since these models don't have built-in [MASK] tokens
const GRANITE_MASK_TOKEN_ID = GRANITE_VOCAB_SIZE      # Last token for Granite 3.1 (1-based)
const GRANITE_4_MASK_TOKEN_ID = GRANITE_4_VOCAB_SIZE  # Last token for Granite 4.0 (1-based)
const QWEN3_MASK_TOKEN_ID = QWEN3_VOCAB_SIZE          # Last token for Qwen3 (1-based)
const LLAMA3_MASK_TOKEN_ID = LLAMA3_VOCAB_SIZE        # Last token for Llama 3.x (1-based)

# -----------------------------------------------------------------------------
# Sinusoidal Time Embedding (for diffusion timestep)
# -----------------------------------------------------------------------------
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
    # t: scalar or (batch,) array of timesteps in [0, 1]
    half_dim = layer.time_dimension ÷ 2

    # Frequency bands: exp(-log(max_period) * i / half_dim)
    freqs = exp.(-log(layer.max_period) .* collect(0:half_dim-1) ./ half_dim)

    # Handle batched or scalar input
    if t isa Number
        angles = t .* freqs
        embedding = vcat(sin.(angles), cos.(angles))
    else
        # t is (batch,) → output is (time_dim, batch)
        t_col = reshape(t, 1, :)  # (1, batch)
        freqs_col = reshape(freqs, :, 1)  # (half_dim, 1)
        angles = t_col .* freqs_col  # (half_dim, batch)
        embedding = vcat(sin.(angles), cos.(angles))  # (time_dim, batch)
    end

    return embedding, state
end

# -----------------------------------------------------------------------------
# Time MLP Embedding (sinusoidal → MLP → embedding_dim)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# OssammaDrafter Model
# -----------------------------------------------------------------------------
struct OssammaDrafter{E,P,T,B,N,L} <: LuxCore.AbstractLuxLayer
    # Configuration
    vocab_size::Int
    max_sequence_length::Int
    embedding_dimension::Int
    number_of_heads::Int
    number_of_layers::Int
    time_dimension::Int
    mask_token_id::Int              # Token ID for [MASK]

    # Layers
    TokenEmbedding::E               # vocab → d
    PositionEmbedding::P            # max_seq → d
    TimeEmbedding::T                # t → d (for diffusion step)
    Blocks::B                       # N × DrafterBlock
    FinalNorm::N                    # LayerNorm before LM head
    LMHead::L                       # d → vocab (language model head)
end

# =============================================================================
# DrafterConfig - Configuration struct for easy setup
# =============================================================================
"""
    DrafterConfig

Configuration for OssammaDrafter model.

# Fields
- `ar_model::String`: AR verifier model name ("granite", "granite4", "qwen3", "llama3")
- `vocab_size::Int`: Vocabulary size (auto-set based on ar_model)
- `mask_token_id::Int`: Token ID for [MASK] (auto-set based on ar_model; can be > vocab_size to reserve an extra token)
- `max_sequence_length::Int`: Maximum sequence length
- `embedding_dimension::Int`: Model hidden dimension
- `number_of_heads::Int`: Number of attention heads
- `number_of_layers::Int`: Number of DrafterBlock layers
- `time_dimension::Int`: Dimension for time embeddings
- `dropout_rate::Float32`: Dropout rate
- `use_ffn::Bool`: Enable SwiGLU FFN in blocks
- `ffn_expansion::Float32`: FFN expansion factor
- `use_parallel_scan::Bool`: Use parallel scan for oscillators
"""
Base.@kwdef struct DrafterConfig
    # AR model selection
    ar_model::String = "granite"

    # Vocabulary (auto-determined from ar_model if not specified)
    vocab_size::Int = GRANITE_VOCAB_SIZE
    mask_token_id::Int = GRANITE_MASK_TOKEN_ID

    # Architecture
    max_sequence_length::Int = 2048
    embedding_dimension::Int = 512
    number_of_heads::Int = 8
    number_of_layers::Int = 6
    time_dimension::Int = 64

    # Regularization
    dropout_rate::Float32 = 0.1f0

    # FFN
    use_ffn::Bool = true
    ffn_expansion::Float32 = 1.5f0

    # Optimization
    use_parallel_scan::Bool = false
end

"""
    default_granite_config(; kwargs...)

Create a DrafterConfig optimized for Granite 4.0 as AR verifier.
"""
function default_granite_config(;
    embedding_dimension::Int = 512,
    number_of_layers::Int = 6,
    number_of_heads::Int = 8,
    kwargs...
)
    return DrafterConfig(;
        ar_model = "granite4",
        vocab_size = GRANITE_4_VOCAB_SIZE,
        mask_token_id = GRANITE_4_MASK_TOKEN_ID,
        embedding_dimension = embedding_dimension,
        number_of_layers = number_of_layers,
        number_of_heads = number_of_heads,
        kwargs...
    )
end

"""
    default_qwen3_config(; kwargs...)

Create a DrafterConfig optimized for Qwen3 as AR verifier.
"""
function default_qwen3_config(;
    embedding_dimension::Int = 512,
    number_of_layers::Int = 6,
    number_of_heads::Int = 8,
    kwargs...
)
    return DrafterConfig(;
        ar_model = "qwen3",
        vocab_size = QWEN3_VOCAB_SIZE,
        mask_token_id = QWEN3_MASK_TOKEN_ID,
        embedding_dimension = embedding_dimension,
        number_of_layers = number_of_layers,
        number_of_heads = number_of_heads,
        kwargs...
    )
end

"""
    load_drafter_config(path::String) -> DrafterConfig

Load DrafterConfig from a TOML file.
"""
function load_drafter_config(path::String)
    config_dict = TOML.parsefile(path)

    # Determine vocab/mask from ar_model if not explicitly set
    ar_model = get(config_dict, "ar_model", "granite")
    if !haskey(config_dict, "vocab_size")
        config_dict["vocab_size"] = if ar_model == "granite"
            GRANITE_VOCAB_SIZE
        elseif ar_model == "granite4" || ar_model == "granite_4"
            GRANITE_4_VOCAB_SIZE
        elseif ar_model == "qwen3"
            QWEN3_VOCAB_SIZE
        elseif ar_model == "llama3"
            LLAMA3_VOCAB_SIZE
        else
            error("Unknown ar_model: $ar_model. Use 'granite', 'granite4', 'qwen3', or 'llama3'.")
        end
    end
    if !haskey(config_dict, "mask_token_id")
        config_dict["mask_token_id"] = if ar_model == "granite"
            GRANITE_MASK_TOKEN_ID
        elseif ar_model == "granite4" || ar_model == "granite_4"
            GRANITE_4_MASK_TOKEN_ID
        elseif ar_model == "qwen3"
            QWEN3_MASK_TOKEN_ID
        elseif ar_model == "llama3"
            LLAMA3_MASK_TOKEN_ID
        else
            0
        end
    end

    return DrafterConfig(;
        ar_model = ar_model,
        vocab_size = config_dict["vocab_size"],
        mask_token_id = config_dict["mask_token_id"],
        max_sequence_length = get(config_dict, "max_sequence_length", 2048),
        embedding_dimension = get(config_dict, "embedding_dimension", 512),
        number_of_heads = get(config_dict, "number_of_heads", 8),
        number_of_layers = get(config_dict, "number_of_layers", 6),
        time_dimension = get(config_dict, "time_dimension", 64),
        dropout_rate = Float32(get(config_dict, "dropout_rate", 0.1)),
        use_ffn = get(config_dict, "use_ffn", true),
        ffn_expansion = Float32(get(config_dict, "ffn_expansion", 1.5)),
        use_parallel_scan = get(config_dict, "use_parallel_scan", false),
    )
end

"""
    OssammaDrafter(config::DrafterConfig)

Create an OssammaDrafter from a DrafterConfig.
"""
function OssammaDrafter(config::DrafterConfig)
    return OssammaDrafter(;
        vocab_size = config.vocab_size,
        max_sequence_length = config.max_sequence_length,
        embedding_dimension = config.embedding_dimension,
        number_of_heads = config.number_of_heads,
        number_of_layers = config.number_of_layers,
        time_dimension = config.time_dimension,
        mask_token_id = config.mask_token_id,
        dropout_rate = config.dropout_rate,
        use_ffn = config.use_ffn,
        ffn_expansion = config.ffn_expansion,
        use_parallel_scan = config.use_parallel_scan,
    )
end

"""
    OssammaDrafter(; kwargs...)

Create a full Ossamma drafter model for TiDAR-style text generation.

# Keyword Arguments
- `vocab_size::Int = 100352`: Vocabulary size (default: Granite 4.0)
- `max_sequence_length::Int = 2048`: Maximum sequence length
- `embedding_dimension::Int = 512`: Model dimension
- `number_of_heads::Int = 8`: Number of attention heads
- `number_of_layers::Int = 6`: Number of DrafterBlock layers
- `time_dimension::Int = 64`: Dimension for time embeddings
- `mask_token_id::Int`: Token ID for [MASK] (default: Granite)
- `dropout_rate::Float32 = 0.1`: Dropout rate
- `use_ffn::Bool = true`: Enable SwiGLU FFN in blocks
- `ffn_expansion::Float32 = 1.5`: FFN expansion factor
- `use_parallel_scan::Bool = false`: Use parallel scan for oscillators
"""
function OssammaDrafter(;
    vocab_size::Int = GRANITE_VOCAB_SIZE,
    max_sequence_length::Int = 2048,
    embedding_dimension::Int = 512,
    number_of_heads::Int = 8,
    number_of_layers::Int = 6,
    time_dimension::Int = 64,
    mask_token_id::Int = GRANITE_MASK_TOKEN_ID,
    dropout_rate::Float32 = 0.1f0,
    use_ffn::Bool = true,
    ffn_expansion::Float32 = 3f0 / 2f0,
    use_parallel_scan::Bool = false,
)
    # Create blocks
    blocks = [
        OssammaDrafterBlock(
            embedding_dimension,
            max_sequence_length,
            number_of_heads,
            time_dimension;
            dropout_rate = dropout_rate,
            use_ffn = use_ffn,
            ffn_expansion = ffn_expansion,
            use_parallel_scan = use_parallel_scan,
        )
        for _ in 1:number_of_layers
    ]

    actual_vocab_size = max(vocab_size, mask_token_id)

    return OssammaDrafter(
        actual_vocab_size,
        max_sequence_length,
        embedding_dimension,
        number_of_heads,
        number_of_layers,
        time_dimension,
        mask_token_id,
        # Layers
        Lux.Embedding(actual_vocab_size => embedding_dimension),
        Lux.Embedding(max_sequence_length => embedding_dimension),
        TimeMLPEmbedding(time_dimension, embedding_dimension),
        blocks,
        Lux.LayerNorm((embedding_dimension,)),
        Lux.Dense(embedding_dimension => actual_vocab_size),  # LM Head
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, model::OssammaDrafter)
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

function Lux.initialstates(rng::Random.AbstractRNG, model::OssammaDrafter)
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
    (model::OssammaDrafter)(token_ids, t, params, state)

Forward pass of the drafter.

# Arguments
- `token_ids`: (seq_len, batch) or (seq_len,) - token IDs (may include [MASK])
- `t`: scalar or (batch,) - diffusion timestep in [0, 1]
- `params`: Model parameters
- `state`: Model state

# Returns
- `logits`: (vocab_size, seq_len, batch) - predictions for all positions
- `new_state`: Updated state
"""
function (model::OssammaDrafter)(token_ids, t, params, state)
    # Handle input dimensions
    if ndims(token_ids) == 1
        token_ids = reshape(token_ids, :, 1)  # (seq_len,) → (seq_len, 1)
        was_unbatched = true
    else
        was_unbatched = false
    end

    seq_len, batch_size = size(token_ids)

    # =========================================================================
    # 1. Token Embeddings
    # =========================================================================
    # Flatten for embedding lookup: (seq_len, batch) → (seq_len * batch,)
    token_flat = vec(token_ids)
    token_emb_flat, tok_state = model.TokenEmbedding(
        token_flat, params.TokenEmbedding, state.TokenEmbedding
    )
    # Reshape: (d, seq*batch) → (d, seq, batch)
    token_emb = reshape(token_emb_flat, model.embedding_dimension, seq_len, batch_size)

    # =========================================================================
    # 2. Position Embeddings
    # =========================================================================
    position_indices = collect(1:seq_len)
    pos_emb_raw, pos_state = model.PositionEmbedding(
        position_indices, params.PositionEmbedding, state.PositionEmbedding
    )
    # Broadcast: (d, seq) → (d, seq, batch)
    pos_emb = reshape(pos_emb_raw, model.embedding_dimension, seq_len, 1)

    # =========================================================================
    # 3. Time Embedding (diffusion timestep)
    # =========================================================================
    # Ensure t is (batch,) for batched processing
    t_input = t isa Number ? fill(Float32(t), batch_size) : Float32.(t)

    # Get sinusoidal embedding for blocks (time_dim, batch)
    # TimeConditionedLayerNorm in blocks expects (time_dim, batch)
    sinusoidal_emb, sin_state = model.TimeEmbedding.SinusoidalEmbed(
        t_input, params.TimeEmbedding.SinusoidalEmbed, state.TimeEmbedding.SinusoidalEmbed
    )
    # sinusoidal_emb: (time_dim, batch) - for blocks

    # Full time embedding (for potential future use)
    time_emb, time_state = model.TimeEmbedding(
        t_input, params.TimeEmbedding, state.TimeEmbedding
    )
    # time_emb: (embedding_dim, batch)

    # =========================================================================
    # 4. Combine Embeddings
    # =========================================================================
    hidden = token_emb .+ pos_emb  # (d, seq, batch)

    # =========================================================================
    # 5. Apply DrafterBlocks
    # =========================================================================
    block_states = []
    for (i, block) in enumerate(model.Blocks)
        # Pass sinusoidal_emb (time_dim, batch) to blocks, not the MLP-projected one
        hidden, blk_state = block((hidden, sinusoidal_emb), params.Blocks[i], state.Blocks[i])
        push!(block_states, blk_state)
    end

    # =========================================================================
    # 6. Final LayerNorm
    # =========================================================================
    hidden_flat = reshape(hidden, model.embedding_dimension, :)
    hidden_norm, norm_state = model.FinalNorm(
        hidden_flat, params.FinalNorm, state.FinalNorm
    )
    hidden = reshape(hidden_norm, model.embedding_dimension, seq_len, batch_size)

    # =========================================================================
    # 7. LM Head → logits
    # =========================================================================
    # Reshape for Dense: (d, seq, batch) → (d, seq*batch)
    hidden_flat = reshape(hidden, model.embedding_dimension, :)
    logits_flat, lm_state = model.LMHead(
        hidden_flat, params.LMHead, state.LMHead
    )
    # Reshape back: (vocab, seq*batch) → (vocab, seq, batch)
    logits = reshape(logits_flat, model.vocab_size, seq_len, batch_size)

    # =========================================================================
    # 8. Collect new state
    # =========================================================================
    new_state = (
        TokenEmbedding = tok_state,
        PositionEmbedding = pos_state,
        TimeEmbedding = time_state,
        Blocks = block_states,
        FinalNorm = norm_state,
        LMHead = lm_state,
    )

    # Remove batch dimension if input was unbatched
    if was_unbatched
        logits = dropdims(logits, dims=3)  # (vocab, seq, 1) → (vocab, seq)
    end

    return logits, new_state
end

# -----------------------------------------------------------------------------
# Utility: Count parameters
# -----------------------------------------------------------------------------
function count_parameters(model::OssammaDrafter)
    d = model.embedding_dimension
    V = model.vocab_size
    L = model.number_of_layers
    S = model.max_sequence_length
    t_dim = model.time_dimension

    # Embeddings
    tok_emb = V * d
    pos_emb = S * d
    time_emb = t_dim * d + d * d + d * d  # sinusoidal + MLP1 + MLP2 (approx)

    # Blocks (rough estimate per block)
    # GLU: d→2d, LinearAttn, DLinOSS, d→d, SwiGLU, LayerNorm
    per_block = 2d*d + d*d + d*d + d*d + Int(round(d * 1.5 * 2)) * d + d  # rough
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

end # module Drafter
