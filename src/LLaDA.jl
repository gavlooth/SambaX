module LLaDA

"""
LLaDA-style Text Diffusion Model using Ossamma architecture.

Forward (masking):   "The cat sat on mat" → "The [M] sat [M] mat" → "[M] [M] [M] [M] [M]"
Reverse (denoising): "[M] [M] [M] [M] [M]" → "The [M] sat [M] mat" → "The cat sat on mat"

Each step, model predicts masked tokens, we unmask some based on confidence.
"""

using Lux
using Random
using NNlib
using TOML

# Import parent module components
using ..Ossamma: OssammaBlock, LuxLayer

# ============================================================================
# Configuration
# ============================================================================

"""
Configuration struct for LLaDA model, loadable from TOML.
"""
Base.@kwdef struct LLaDAConfig
    # Model architecture
    vocab_size::Int = 32000
    max_sequence_length::Int = 512
    embedding_dimension::Int = 256
    number_of_heads::Int = 4
    number_of_layers::Int = 6
    mask_token_id::Int = -1  # -1 means use vocab_size (append [MASK])

    # Internal dimensions
    time_dimension::Int = 128
    state_dimension::Int = -1  # -1 means use embedding_dimension

    # Attention
    window_size::Int = 5

    # Oscillator SSM
    min_frequency::Float32 = 0.1f0
    max_frequency::Float32 = 10.0f0
    default_time_step::Float32 = 0.1f0

    # Generation defaults
    default_num_steps::Int = 10
    default_temperature::Float32 = 1.0f0

    # Training defaults
    mask_schedule::Symbol = :uniform  # :uniform, :cosine
end

"""
    load_config(path::String) -> LLaDAConfig

Load model configuration from a TOML file.
"""
function load_config(path::String)
    data = TOML.parsefile(path)
    return config_from_dict(data)
end

"""
    load_config(io::IO) -> LLaDAConfig

Load model configuration from an IO stream.
"""
function load_config(io::IO)
    data = TOML.parse(io)
    return config_from_dict(data)
end

"""
    config_from_dict(data::Dict) -> LLaDAConfig

Create config from a parsed TOML dictionary.
"""
function config_from_dict(data::Dict)
    # Flatten nested sections
    flat = Dict{Symbol, Any}()

    # Handle nested [model], [training], [generation] sections
    for (section, values) in data
        if values isa Dict
            for (key, val) in values
                flat[Symbol(key)] = val
            end
        else
            flat[Symbol(section)] = values
        end
    end

    # Convert mask_schedule string to Symbol
    if haskey(flat, :mask_schedule) && flat[:mask_schedule] isa String
        flat[:mask_schedule] = Symbol(flat[:mask_schedule])
    end

    # Build config with defaults for missing keys
    return LLaDAConfig(;
        vocab_size = get(flat, :vocab_size, 32000),
        max_sequence_length = get(flat, :max_sequence_length, 512),
        embedding_dimension = get(flat, :embedding_dimension, 256),
        number_of_heads = get(flat, :number_of_heads, 4),
        number_of_layers = get(flat, :number_of_layers, 6),
        mask_token_id = get(flat, :mask_token_id, -1),
        time_dimension = get(flat, :time_dimension, 128),
        state_dimension = get(flat, :state_dimension, -1),
        window_size = get(flat, :window_size, 5),
        min_frequency = Float32(get(flat, :min_frequency, 0.1)),
        max_frequency = Float32(get(flat, :max_frequency, 10.0)),
        default_time_step = Float32(get(flat, :default_time_step, 0.1)),
        default_num_steps = get(flat, :default_num_steps, 10),
        default_temperature = Float32(get(flat, :default_temperature, 1.0)),
        mask_schedule = get(flat, :mask_schedule, :uniform),
    )
end

"""
    save_config(config::LLaDAConfig, path::String)

Save configuration to a TOML file.
"""
function save_config(config::LLaDAConfig, path::String)
    open(path, "w") do io
        save_config(config, io)
    end
end

"""
    save_config(config::LLaDAConfig, io::IO)

Write configuration to an IO stream in TOML format.
"""
function save_config(config::LLaDAConfig, io::IO)
    println(io, "# LLaDA Model Configuration")
    println(io, "# Generated automatically\n")

    println(io, "[model]")
    println(io, "vocab_size = ", config.vocab_size)
    println(io, "max_sequence_length = ", config.max_sequence_length)
    println(io, "embedding_dimension = ", config.embedding_dimension)
    println(io, "number_of_heads = ", config.number_of_heads)
    println(io, "number_of_layers = ", config.number_of_layers)
    println(io, "mask_token_id = ", config.mask_token_id)
    println(io)

    println(io, "[model.dimensions]")
    println(io, "time_dimension = ", config.time_dimension)
    println(io, "state_dimension = ", config.state_dimension)
    println(io)

    println(io, "[model.attention]")
    println(io, "window_size = ", config.window_size)
    println(io)

    println(io, "[model.oscillator]")
    println(io, "min_frequency = ", config.min_frequency)
    println(io, "max_frequency = ", config.max_frequency)
    println(io, "default_time_step = ", config.default_time_step)
    println(io)

    println(io, "[generation]")
    println(io, "default_num_steps = ", config.default_num_steps)
    println(io, "default_temperature = ", config.default_temperature)
    println(io)

    println(io, "[training]")
    println(io, "mask_schedule = \"", config.mask_schedule, "\"")
end

"""
    default_config() -> LLaDAConfig

Return default configuration.
"""
default_config() = LLaDAConfig()

"""
    small_config() -> LLaDAConfig

Return configuration for a small model (for testing/debugging).
"""
function small_config()
    LLaDAConfig(
        vocab_size = 1000,
        max_sequence_length = 64,
        embedding_dimension = 64,
        number_of_heads = 2,
        number_of_layers = 2,
        time_dimension = 32,
        state_dimension = 64,
    )
end

"""
    base_config() -> LLaDAConfig

Return configuration for a base-sized model.
"""
function base_config()
    LLaDAConfig(
        vocab_size = 32000,
        max_sequence_length = 512,
        embedding_dimension = 512,
        number_of_heads = 8,
        number_of_layers = 12,
        time_dimension = 128,
        state_dimension = 512,
    )
end

"""
    large_config() -> LLaDAConfig

Return configuration for a large model.
"""
function large_config()
    LLaDAConfig(
        vocab_size = 32000,
        max_sequence_length = 1024,
        embedding_dimension = 1024,
        number_of_heads = 16,
        number_of_layers = 24,
        time_dimension = 256,
        state_dimension = 1024,
    )
end

# ============================================================================
# Sinusoidal Time Embedding (for mask ratio t ∈ [0,1])
# ============================================================================
struct SinusoidalTimeEmbedding <: LuxLayer
    time_dimension::Int
    max_period::Float32
end

function SinusoidalTimeEmbedding(time_dimension::Int; max_period::Float32 = 10000.0f0)
    @assert time_dimension % 2 == 0 "time_dimension must be even"
    return SinusoidalTimeEmbedding(time_dimension, max_period)
end

function Lux.initialparameters(::Random.AbstractRNG, ::SinusoidalTimeEmbedding)
    return (;)  # No learnable parameters
end

function Lux.initialstates(::Random.AbstractRNG, ::SinusoidalTimeEmbedding)
    return (;)
end

function (layer::SinusoidalTimeEmbedding)(t, params, state)
    # t: scalar or (1, batch) - mask ratio in [0, 1]
    half_dim = layer.time_dimension ÷ 2

    # Frequency bands
    freqs = exp.(-(log(layer.max_period)) .* collect(0:half_dim-1) ./ half_dim)

    # Handle batched input
    if ndims(t) == 0 || (ndims(t) == 2 && size(t, 1) == 1)
        t_flat = ndims(t) == 0 ? [t] : vec(t)
        # (half_dim,) * (batch,)' → (half_dim, batch)
        args = freqs * t_flat'
    else
        args = freqs .* t
    end

    # Sinusoidal embedding: [sin; cos]
    embedding = vcat(sin.(args), cos.(args))  # (time_dim, batch)

    return embedding, state
end

# ============================================================================
# Time MLP: Projects sinusoidal embedding to model dimension
# ============================================================================
struct TimeMLPEmbedding <: LuxLayer
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

# ============================================================================
# LLaDA Model: Full text diffusion architecture
# ============================================================================
struct LLaDAModel{E,P,T,B,N,O} <: LuxLayer
    vocab_size::Int
    max_sequence_length::Int
    embedding_dimension::Int
    number_of_heads::Int
    number_of_layers::Int
    mask_token_id::Int

    # Embeddings
    TokenEmbedding::E
    PositionEmbedding::P
    TimeEmbedding::T

    # Transformer blocks
    Blocks::B

    # Output
    FinalNorm::N
    OutputHead::O
end

"""
    LLaDAModel(config::LLaDAConfig)

Create a LLaDA model from a configuration struct.
"""
function LLaDAModel(config::LLaDAConfig)
    # Resolve -1 sentinel values
    mask_token_id = config.mask_token_id == -1 ? config.vocab_size : config.mask_token_id
    state_dimension = config.state_dimension == -1 ? config.embedding_dimension : config.state_dimension

    return LLaDAModel(;
        vocab_size = config.vocab_size,
        max_sequence_length = config.max_sequence_length,
        embedding_dimension = config.embedding_dimension,
        number_of_heads = config.number_of_heads,
        number_of_layers = config.number_of_layers,
        mask_token_id = mask_token_id,
        time_dimension = config.time_dimension,
        state_dimension = state_dimension,
        window_size = config.window_size,
        min_frequency = config.min_frequency,
        max_frequency = config.max_frequency,
        default_time_step = config.default_time_step,
    )
end

"""
    LLaDAModel(config_path::String)

Create a LLaDA model from a TOML configuration file.
"""
function LLaDAModel(config_path::String)
    config = load_config(config_path)
    return LLaDAModel(config)
end

function LLaDAModel(;
    vocab_size::Int,
    max_sequence_length::Int,
    embedding_dimension::Int,
    number_of_heads::Int,
    number_of_layers::Int,
    mask_token_id::Int = vocab_size,  # Default: last token is [MASK]
    time_dimension::Int = 128,
    state_dimension::Int = embedding_dimension,
    window_size::Int = 5,
    min_frequency::Float32 = 0.1f0,
    max_frequency::Float32 = 10.0f0,
    default_time_step::Float32 = 0.1f0,
)
    # Token embedding includes mask token
    actual_vocab_size = mask_token_id < vocab_size ? vocab_size : vocab_size + 1

    # Build stack of OssammaBlocks
    blocks = Tuple([
        OssammaBlock(
            embedding_dimension,
            max_sequence_length,
            number_of_heads,
            time_dimension;
            state_dimension = state_dimension,
            window_size = window_size,
            min_frequency = min_frequency,
            max_frequency = max_frequency,
            default_time_step = default_time_step,
        )
        for _ in 1:number_of_layers
    ])

    return LLaDAModel(
        actual_vocab_size,
        max_sequence_length,
        embedding_dimension,
        number_of_heads,
        number_of_layers,
        mask_token_id,
        # Embeddings
        Lux.Embedding(actual_vocab_size => embedding_dimension),
        Lux.Embedding(max_sequence_length => embedding_dimension),
        TimeMLPEmbedding(time_dimension, time_dimension),
        # Blocks
        blocks,
        # Output
        Lux.LayerNorm((embedding_dimension,)),
        Lux.Dense(embedding_dimension => actual_vocab_size; use_bias = false),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, model::LLaDAModel)
    block_params = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(
        Tuple(Lux.initialparameters(rng, block) for block in model.Blocks)
    )

    return (
        TokenEmbedding = Lux.initialparameters(rng, model.TokenEmbedding),
        PositionEmbedding = Lux.initialparameters(rng, model.PositionEmbedding),
        TimeEmbedding = Lux.initialparameters(rng, model.TimeEmbedding),
        Blocks = block_params,
        FinalNorm = Lux.initialparameters(rng, model.FinalNorm),
        OutputHead = Lux.initialparameters(rng, model.OutputHead),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, model::LLaDAModel)
    block_states = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(
        Tuple(Lux.initialstates(rng, block) for block in model.Blocks)
    )

    return (
        TokenEmbedding = Lux.initialstates(rng, model.TokenEmbedding),
        PositionEmbedding = Lux.initialstates(rng, model.PositionEmbedding),
        TimeEmbedding = Lux.initialstates(rng, model.TimeEmbedding),
        Blocks = block_states,
        FinalNorm = Lux.initialstates(rng, model.FinalNorm),
        OutputHead = Lux.initialstates(rng, model.OutputHead),
    )
end

function (model::LLaDAModel)(inputs::NamedTuple, params, state)
    # inputs: (token_ids = (seq_len, batch), mask_ratio = (1, batch) or scalar)
    (; token_ids, mask_ratio) = inputs

    seq_len = size(token_ids, 1)
    is_batched = ndims(token_ids) == 2
    batch_size = is_batched ? size(token_ids, 2) : 1

    # Standardize to batched format
    token_ids_batched = is_batched ? token_ids : reshape(token_ids, :, 1)

    # =========================================================================
    # 1. Token Embedding
    # =========================================================================
    # Flatten for embedding lookup, then reshape back
    token_flat = vec(token_ids_batched)  # (seq_len * batch,)
    token_emb_flat, tok_state = model.TokenEmbedding(token_flat, params.TokenEmbedding, state.TokenEmbedding)
    # token_emb_flat: (embedding_dim, seq_len * batch)
    token_emb = reshape(token_emb_flat, model.embedding_dimension, seq_len, batch_size)

    # =========================================================================
    # 2. Position Embedding
    # =========================================================================
    position_indices = collect(1:seq_len)
    pos_emb_raw, pos_state = model.PositionEmbedding(position_indices, params.PositionEmbedding, state.PositionEmbedding)
    # pos_emb_raw: (embedding_dim, seq_len)
    pos_emb = reshape(pos_emb_raw, model.embedding_dimension, seq_len, 1)  # Broadcast over batch

    # =========================================================================
    # 3. Combine Embeddings
    # =========================================================================
    hidden = token_emb .+ pos_emb

    # =========================================================================
    # 4. Time Embedding (mask ratio conditioning)
    # =========================================================================
    # Ensure mask_ratio is (1, batch) shape
    t_input = if ndims(mask_ratio) == 0
        fill(mask_ratio, 1, batch_size)
    elseif ndims(mask_ratio) == 1
        reshape(mask_ratio, 1, :)
    else
        mask_ratio
    end

    time_emb, time_state = model.TimeEmbedding(t_input, params.TimeEmbedding, state.TimeEmbedding)
    # time_emb: (time_dim, batch)

    # =========================================================================
    # 5. Process through OssammaBlocks
    # =========================================================================
    block_states = []
    for (i, block) in enumerate(model.Blocks)
        block_key = Symbol("Block_$i")
        block_params = params.Blocks[block_key]
        block_state = state.Blocks[block_key]

        hidden, new_block_state = block((hidden, time_emb), block_params, block_state)
        push!(block_states, new_block_state)
    end

    # =========================================================================
    # 6. Final Normalization
    # =========================================================================
    normalized, norm_state = model.FinalNorm(hidden, params.FinalNorm, state.FinalNorm)

    # =========================================================================
    # 7. Output Head → Logits
    # =========================================================================
    logits, out_state = model.OutputHead(normalized, params.OutputHead, state.OutputHead)
    # logits: (vocab_size, seq_len, batch)

    # Remove batch dim if input wasn't batched
    final_logits = is_batched ? logits : dropdims(logits, dims = 3)

    # =========================================================================
    # 8. Update State
    # =========================================================================
    new_block_states = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(
        Tuple(block_states)
    )

    new_state = (
        TokenEmbedding = tok_state,
        PositionEmbedding = pos_state,
        TimeEmbedding = time_state,
        Blocks = new_block_states,
        FinalNorm = norm_state,
        OutputHead = out_state,
    )

    return final_logits, new_state
end

# ============================================================================
# Diffusion Utilities
# ============================================================================

"""
    apply_mask(token_ids, mask_ratio, mask_token_id; rng)

Apply random masking to token_ids based on mask_ratio.
Returns (masked_ids, mask) where mask is true for masked positions.
"""
function apply_mask(token_ids::AbstractArray, mask_ratio::Real, mask_token_id::Int; rng = Random.default_rng())
    mask = rand(rng, Float32, size(token_ids)) .< mask_ratio
    masked_ids = ifelse.(mask, mask_token_id, token_ids)
    return masked_ids, mask
end

"""
    sample_mask_ratio(rng; schedule = :uniform)

Sample a mask ratio t ∈ [0, 1] for training.
"""
function sample_mask_ratio(rng::Random.AbstractRNG; schedule::Symbol = :uniform)
    if schedule == :uniform
        return rand(rng, Float32)
    elseif schedule == :cosine
        # Cosine schedule: more samples near t=0 and t=1
        u = rand(rng, Float32)
        return (1 - cos(π * u)) / 2
    else
        return rand(rng, Float32)
    end
end

"""
    unmask_step(logits, current_ids, mask, num_to_unmask, mask_token_id)

Select positions to unmask based on model confidence.
Returns new token_ids with some positions unmasked.
"""
function unmask_step(
    logits::AbstractArray,
    current_ids::AbstractArray,
    mask::AbstractArray{Bool},
    num_to_unmask::Int,
    mask_token_id::Int,
)
    # Get predictions and confidence
    predictions = argmax(logits, dims = 1)
    predictions = dropdims(predictions, dims = 1)  # (seq_len,) or (seq_len, batch)

    # Get confidence (max probability)
    probs = NNlib.softmax(logits, dims = 1)
    confidence = maximum(probs, dims = 1)
    confidence = dropdims(confidence, dims = 1)

    # Only consider masked positions
    confidence_masked = ifelse.(mask, confidence, -Inf32)

    # Find top-k most confident masked positions
    if ndims(current_ids) == 1
        # Unbatched case
        sorted_indices = sortperm(vec(confidence_masked), rev = true)
        to_unmask = sorted_indices[1:min(num_to_unmask, sum(mask))]

        new_ids = copy(current_ids)
        new_mask = copy(mask)
        for idx in to_unmask
            new_ids[idx] = predictions[idx]
            new_mask[idx] = false
        end
    else
        # Batched case - process each batch independently
        new_ids = copy(current_ids)
        new_mask = copy(mask)
        for b in axes(current_ids, 2)
            conf_b = confidence_masked[:, b]
            sorted_indices = sortperm(vec(conf_b), rev = true)
            n_masked = sum(mask[:, b])
            to_unmask = sorted_indices[1:min(num_to_unmask, n_masked)]

            for idx in to_unmask
                new_ids[idx, b] = predictions[idx, b]
                new_mask[idx, b] = false
            end
        end
    end

    return new_ids, new_mask
end

"""
    generate(model, params, state, seq_len, mask_token_id;
             num_steps, batch_size, rng)

Generate text by iterative denoising from fully masked sequence.
"""
function generate(
    model::LLaDAModel,
    params,
    state,
    seq_len::Int;
    num_steps::Int = 10,
    batch_size::Int = 1,
    rng::Random.AbstractRNG = Random.default_rng(),
)
    # Start with fully masked sequence
    current_ids = fill(model.mask_token_id, seq_len, batch_size)
    mask = trues(seq_len, batch_size)

    # Number of tokens to unmask per step
    tokens_per_step = ceil(Int, seq_len / num_steps)

    for step in 1:num_steps
        # Current mask ratio (decreasing)
        mask_ratio = 1.0f0 - (step - 1) / num_steps

        # Forward pass
        inputs = (token_ids = current_ids, mask_ratio = mask_ratio)
        logits, state = model(inputs, params, state)

        # Unmask most confident positions
        remaining_masked = sum(mask)
        num_to_unmask = min(tokens_per_step, remaining_masked)

        if num_to_unmask > 0
            current_ids, mask = unmask_step(
                logits, current_ids, mask, num_to_unmask, model.mask_token_id
            )
        end

        # Early exit if fully unmasked
        if !any(mask)
            break
        end
    end

    # Final pass with mask_ratio = 0 to refine any remaining
    if any(mask)
        inputs = (token_ids = current_ids, mask_ratio = 0.0f0)
        logits, state = model(inputs, params, state)
        predictions = dropdims(argmax(logits, dims = 1), dims = 1)
        current_ids = ifelse.(mask, predictions, current_ids)
    end

    return batch_size == 1 ? vec(current_ids) : current_ids
end

# ============================================================================
# Exports
# ============================================================================

# Model
export LLaDAModel, TimeMLPEmbedding, SinusoidalTimeEmbedding

# Configuration
export LLaDAConfig, load_config, save_config, config_from_dict
export default_config, small_config, base_config, large_config

# Diffusion utilities
export apply_mask, sample_mask_ratio, unmask_step, generate

end # module
