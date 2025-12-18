module Classification

"""
Classification model using Ossamma architecture.

Adapts OssammaBlock for sequence classification tasks by:
1. Using a fixed time embedding (no diffusion conditioning)
2. Pooling sequence representations to a single vector
3. Projecting to class logits

Supports multiple pooling strategies:
- :mean  - Average over sequence dimension
- :cls   - Use first token (requires prepending CLS token)
- :last  - Use last token representation
- :max   - Max pooling over sequence
"""

using Lux
using Random
using NNlib
using Statistics: mean

# Import parent module components (assumes we're included from Ossamma module)
import ..OssammaBlock
import ..TimeConditionedLayerNorm

const LuxLayer = Lux.AbstractLuxLayer

# ============================================================================
# Configuration
# ============================================================================

"""
Configuration for classification model.
"""
Base.@kwdef struct ClassifierConfig
    # Architecture
    vocab_size::Int = 32000
    max_sequence_length::Int = 512
    embedding_dimension::Int = 256
    number_of_heads::Int = 4
    number_of_layers::Int = 6
    num_classes::Int = 2

    # Internal dimensions
    time_dimension::Int = 128
    state_dimension::Int = -1  # -1 means use embedding_dimension

    # Attention
    window_size::Int = 5

    # Oscillator SSM
    min_frequency::Float32 = 0.1f0
    max_frequency::Float32 = 10.0f0
    default_time_step::Float32 = 0.1f0

    # Classification
    pooling::Symbol = :mean  # :mean, :cls, :last, :max
    dropout_rate::Float32 = 0.1f0
    use_cls_token::Bool = false  # Whether to prepend CLS token
end

# ============================================================================
# Pooling Layer
# ============================================================================

struct SequencePooling <: LuxLayer
    strategy::Symbol

    function SequencePooling(strategy::Symbol = :mean)
        @assert strategy in (:mean, :cls, :last, :max) "Pooling strategy must be :mean, :cls, :last, or :max"
        return new(strategy)
    end
end

Lux.initialparameters(::Random.AbstractRNG, ::SequencePooling) = (;)
Lux.initialstates(::Random.AbstractRNG, ::SequencePooling) = (;)

function (layer::SequencePooling)(x, params, state)
    # x: (embedding_dim, seq_len, batch) or (embedding_dim, seq_len)
    is_batched = ndims(x) == 3
    seq_dim = 2

    pooled = if layer.strategy == :mean
        dropdims(mean(x, dims=seq_dim), dims=seq_dim)
    elseif layer.strategy == :cls
        # First token
        is_batched ? x[:, 1, :] : x[:, 1]
    elseif layer.strategy == :last
        # Last token
        is_batched ? x[:, end, :] : x[:, end]
    elseif layer.strategy == :max
        dropdims(maximum(x, dims=seq_dim), dims=seq_dim)
    end

    return pooled, state
end

# ============================================================================
# Fixed Time Embedding (constant, no diffusion)
# ============================================================================

struct FixedTimeEmbedding <: LuxLayer
    time_dimension::Int
    fixed_value::Float32
end

function FixedTimeEmbedding(time_dimension::Int; fixed_value::Float32 = 0.5f0)
    return FixedTimeEmbedding(time_dimension, fixed_value)
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::FixedTimeEmbedding)
    # Learnable time embedding vector (initialized from fixed sinusoidal)
    half_dim = layer.time_dimension ÷ 2
    freqs = exp.(-(log(10000.0f0)) .* collect(0:half_dim-1) ./ half_dim)
    args = freqs .* layer.fixed_value
    init_embedding = vcat(sin.(args), cos.(args))
    return (embedding = init_embedding,)
end

Lux.initialstates(::Random.AbstractRNG, ::FixedTimeEmbedding) = (;)

function (layer::FixedTimeEmbedding)(batch_size::Int, params, state)
    # Return fixed embedding repeated for batch
    # Output: (time_dim, batch)
    embedding = repeat(reshape(params.embedding, :, 1), 1, batch_size)
    return embedding, state
end

# ============================================================================
# Ossamma Classifier
# ============================================================================

struct OssammaClassifier{E, P, T, B, PO, D, C} <: LuxLayer
    vocab_size::Int
    max_sequence_length::Int
    embedding_dimension::Int
    number_of_heads::Int
    number_of_layers::Int
    num_classes::Int
    pooling_strategy::Symbol
    use_cls_token::Bool
    dropout_rate::Float32

    # Embeddings
    TokenEmbedding::E
    PositionEmbedding::P
    TimeEmbedding::T

    # Encoder blocks
    Blocks::B

    # Classification head
    Pooling::PO
    Dropout::D
    Classifier::C
end

"""
    OssammaClassifier(config::ClassifierConfig)

Create a classifier from configuration.
"""
function OssammaClassifier(config::ClassifierConfig)
    state_dimension = config.state_dimension == -1 ? config.embedding_dimension : config.state_dimension

    return OssammaClassifier(;
        vocab_size = config.vocab_size,
        max_sequence_length = config.max_sequence_length,
        embedding_dimension = config.embedding_dimension,
        number_of_heads = config.number_of_heads,
        number_of_layers = config.number_of_layers,
        num_classes = config.num_classes,
        time_dimension = config.time_dimension,
        state_dimension = state_dimension,
        window_size = config.window_size,
        min_frequency = config.min_frequency,
        max_frequency = config.max_frequency,
        default_time_step = config.default_time_step,
        pooling = config.pooling,
        use_cls_token = config.use_cls_token,
        dropout_rate = config.dropout_rate,
    )
end

function OssammaClassifier(;
    vocab_size::Int,
    max_sequence_length::Int,
    embedding_dimension::Int,
    number_of_heads::Int,
    number_of_layers::Int,
    num_classes::Int,
    time_dimension::Int = 128,
    state_dimension::Int = embedding_dimension,
    window_size::Int = 5,
    min_frequency::Float32 = 0.1f0,
    max_frequency::Float32 = 10.0f0,
    default_time_step::Float32 = 0.1f0,
    pooling::Symbol = :mean,
    use_cls_token::Bool = false,
    dropout_rate::Float32 = 0.1f0,
)
    # Add CLS token to vocab if needed
    actual_vocab_size = use_cls_token ? vocab_size + 1 : vocab_size

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

    return OssammaClassifier(
        actual_vocab_size,
        max_sequence_length,
        embedding_dimension,
        number_of_heads,
        number_of_layers,
        num_classes,
        pooling,
        use_cls_token,
        dropout_rate,
        # Embeddings
        Lux.Embedding(actual_vocab_size => embedding_dimension),
        Lux.Embedding(max_sequence_length => embedding_dimension),
        FixedTimeEmbedding(time_dimension),
        # Encoder blocks
        blocks,
        # Classification head
        SequencePooling(pooling),
        Lux.Dropout(dropout_rate),
        Lux.Chain(
            Lux.LayerNorm((embedding_dimension,)),
            Lux.Dense(embedding_dimension => embedding_dimension, NNlib.gelu),
            Lux.Dropout(dropout_rate),
            Lux.Dense(embedding_dimension => num_classes),
        ),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, model::OssammaClassifier)
    block_params = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(
        Tuple(Lux.initialparameters(rng, block) for block in model.Blocks)
    )

    return (
        TokenEmbedding = Lux.initialparameters(rng, model.TokenEmbedding),
        PositionEmbedding = Lux.initialparameters(rng, model.PositionEmbedding),
        TimeEmbedding = Lux.initialparameters(rng, model.TimeEmbedding),
        Blocks = block_params,
        Pooling = Lux.initialparameters(rng, model.Pooling),
        Dropout = Lux.initialparameters(rng, model.Dropout),
        Classifier = Lux.initialparameters(rng, model.Classifier),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, model::OssammaClassifier)
    block_states = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(
        Tuple(Lux.initialstates(rng, block) for block in model.Blocks)
    )

    # Cache position indices to avoid allocation in forward pass
    # +1 to account for potential CLS token prepending
    max_positions = model.use_cls_token ? model.max_sequence_length + 1 : model.max_sequence_length
    position_indices = collect(1:max_positions)

    return (
        TokenEmbedding = Lux.initialstates(rng, model.TokenEmbedding),
        PositionEmbedding = Lux.initialstates(rng, model.PositionEmbedding),
        TimeEmbedding = Lux.initialstates(rng, model.TimeEmbedding),
        Blocks = block_states,
        Pooling = Lux.initialstates(rng, model.Pooling),
        Dropout = Lux.initialstates(rng, model.Dropout),
        Classifier = Lux.initialstates(rng, model.Classifier),
        position_indices = position_indices,
    )
end

function (model::OssammaClassifier)(token_ids::AbstractArray, params, state)
    # token_ids: (seq_len,) or (seq_len, batch)

    is_batched = ndims(token_ids) == 2
    seq_len = size(token_ids, 1)
    batch_size = is_batched ? size(token_ids, 2) : 1

    # Standardize to batched format
    token_ids_batched = is_batched ? token_ids : reshape(token_ids, :, 1)

    # =========================================================================
    # 0. Auto-prepend CLS token if enabled
    # =========================================================================
    if model.use_cls_token
        # CLS token ID is the last token in expanded vocab
        cls_token_id = model.vocab_size
        # Create CLS row with same element type as input
        cls_row = fill(eltype(token_ids_batched)(cls_token_id), 1, batch_size)
        token_ids_batched = vcat(cls_row, token_ids_batched)
        seq_len = seq_len + 1
    end

    # =========================================================================
    # 1. Token Embedding
    # =========================================================================
    token_flat = vec(token_ids_batched)
    token_emb_flat, tok_state = model.TokenEmbedding(token_flat, params.TokenEmbedding, state.TokenEmbedding)
    token_emb = reshape(token_emb_flat, model.embedding_dimension, seq_len, batch_size)

    # =========================================================================
    # 2. Position Embedding (using cached indices)
    # =========================================================================
    position_indices = @view state.position_indices[1:seq_len]
    pos_emb_raw, pos_state = model.PositionEmbedding(position_indices, params.PositionEmbedding, state.PositionEmbedding)
    pos_emb = reshape(pos_emb_raw, model.embedding_dimension, seq_len, 1)

    # =========================================================================
    # 3. Combine Embeddings
    # =========================================================================
    hidden = token_emb .+ pos_emb

    # =========================================================================
    # 4. Fixed Time Embedding (no mask ratio conditioning)
    # =========================================================================
    time_emb, time_state = model.TimeEmbedding(batch_size, params.TimeEmbedding, state.TimeEmbedding)

    # =========================================================================
    # 5. Process through OssammaBlocks
    # =========================================================================
    (hidden, block_states) = foldl(
        enumerate(model.Blocks);
        init = (hidden, ())
    ) do (h, states), (i, block)
        block_key = Symbol("Block_$i")
        block_params = params.Blocks[block_key]
        block_state = state.Blocks[block_key]

        new_h, new_block_state = block((h, time_emb), block_params, block_state)
        (new_h, (states..., new_block_state))
    end

    # =========================================================================
    # 6. Pool sequence to single vector
    # =========================================================================
    pooled, pool_state = model.Pooling(hidden, params.Pooling, state.Pooling)
    # pooled: (embedding_dim, batch)

    # =========================================================================
    # 7. Apply dropout after pooling
    # =========================================================================
    pooled, dropout_state = model.Dropout(pooled, params.Dropout, state.Dropout)

    # =========================================================================
    # 8. Classification head
    # =========================================================================
    logits, classifier_state = model.Classifier(pooled, params.Classifier, state.Classifier)
    # logits: (num_classes, batch)

    # Remove batch dim if input wasn't batched
    final_logits = is_batched ? logits : dropdims(logits, dims=2)

    # =========================================================================
    # 9. Update State
    # =========================================================================
    new_block_states = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(
        block_states
    )

    new_state = (
        TokenEmbedding = tok_state,
        PositionEmbedding = pos_state,
        TimeEmbedding = time_state,
        Blocks = new_block_states,
        Pooling = pool_state,
        Dropout = dropout_state,
        Classifier = classifier_state,
        position_indices = state.position_indices,
    )

    return final_logits, new_state
end

# ============================================================================
# Convenience constructors for common configurations
# ============================================================================

"""
    tiny_classifier(; num_classes, vocab_size, kwargs...)

Tiny classifier for debugging and quick tests.
"""
function tiny_classifier(;
    num_classes::Int = 2,
    vocab_size::Int = 1000,
    max_sequence_length::Int = 64,
    kwargs...
)
    config = ClassifierConfig(;
        vocab_size = vocab_size,
        max_sequence_length = max_sequence_length,
        embedding_dimension = 64,
        number_of_heads = 2,
        number_of_layers = 2,
        num_classes = num_classes,
        time_dimension = 32,
        kwargs...
    )
    return OssammaClassifier(config)
end

"""
    small_classifier(; num_classes, vocab_size, kwargs...)

Small classifier suitable for fine-tuning experiments.
"""
function small_classifier(;
    num_classes::Int = 2,
    vocab_size::Int = 32000,
    max_sequence_length::Int = 256,
    kwargs...
)
    config = ClassifierConfig(;
        vocab_size = vocab_size,
        max_sequence_length = max_sequence_length,
        embedding_dimension = 256,
        number_of_heads = 4,
        number_of_layers = 4,
        num_classes = num_classes,
        time_dimension = 64,
        kwargs...
    )
    return OssammaClassifier(config)
end

"""
    base_classifier(; num_classes, vocab_size, kwargs...)

Base-sized classifier for production use.
"""
function base_classifier(;
    num_classes::Int = 2,
    vocab_size::Int = 32000,
    max_sequence_length::Int = 512,
    kwargs...
)
    config = ClassifierConfig(;
        vocab_size = vocab_size,
        max_sequence_length = max_sequence_length,
        embedding_dimension = 512,
        number_of_heads = 8,
        number_of_layers = 8,
        num_classes = num_classes,
        time_dimension = 128,
        kwargs...
    )
    return OssammaClassifier(config)
end


# ============================================================================
# Pretrained Weight Loading
# ============================================================================

using Serialization

"""
    load_pretrained_encoder(classifier, checkpoint_path; rng=Random.default_rng())

Load encoder weights from a LLaDA checkpoint into a classifier.

Maps the following weights from LLaDA to classifier:
- `TokenEmbedding` (handles vocab size mismatch by copying available tokens)
- `PositionEmbedding` (handles sequence length mismatch)
- `Blocks.Block_N.*` (copies matching components from OssammaBlock)

Does NOT load:
- `TimeEmbedding` (LLaDA uses TimeMLPEmbedding, classifier uses FixedTimeEmbedding)
- `FinalNorm` (LLaDA specific)
- `OutputHead` (LLaDA specific, classifier doesn't have this)
- `Classifier` head (must be trained fresh for the classification task)

# Arguments
- `classifier`: An OssammaClassifier instance
- `checkpoint_path`: Path to a LLaDA checkpoint (.jls file)
- `rng`: Random number generator for initializing new parameters

# Returns
- `(params, state)`: Initialized parameters with pretrained encoder weights and fresh state

# Example
```julia
classifier = small_classifier(num_classes=5, vocab_size=32000)
params, state = load_pretrained_encoder(classifier, "checkpoints/llada_best.jls")
```
"""
function load_pretrained_encoder(
    classifier::OssammaClassifier,
    checkpoint_path::String;
    rng::Random.AbstractRNG = Random.default_rng(),
)
    # Load LLaDA checkpoint
    checkpoint_data = deserialize(checkpoint_path)
    llada_params = checkpoint_data[:params]

    # Initialize fresh classifier params and state
    classifier_params = Lux.initialparameters(rng, classifier)
    classifier_state = Lux.initialstates(rng, classifier)

    # Copy encoder weights
    new_params = _copy_encoder_weights(classifier_params, llada_params, classifier)

    return new_params, classifier_state
end

"""
Internal function to copy encoder weights from LLaDA to classifier.
"""
function _copy_encoder_weights(classifier_params, llada_params, classifier)
    # Deep copy to avoid mutation
    new_params = deepcopy(classifier_params)

    # 1. Token Embedding - handle vocab size mismatch
    if haskey(llada_params, :TokenEmbedding) && haskey(llada_params.TokenEmbedding, :weight)
        llada_tok = llada_params.TokenEmbedding.weight
        classifier_tok = new_params.TokenEmbedding.weight

        # Copy as many tokens as possible
        copy_cols = min(size(llada_tok, 2), size(classifier_tok, 2))
        @views classifier_tok[:, 1:copy_cols] .= llada_tok[:, 1:copy_cols]

        # Update in new_params (need to reconstruct the NamedTuple)
        new_params = merge(new_params, (TokenEmbedding = (weight = classifier_tok,),))
    end

    # 2. Position Embedding - handle sequence length mismatch
    if haskey(llada_params, :PositionEmbedding) && haskey(llada_params.PositionEmbedding, :weight)
        llada_pos = llada_params.PositionEmbedding.weight
        classifier_pos = new_params.PositionEmbedding.weight

        # Copy as many positions as possible
        copy_cols = min(size(llada_pos, 2), size(classifier_pos, 2))
        @views classifier_pos[:, 1:copy_cols] .= llada_pos[:, 1:copy_cols]

        new_params = merge(new_params, (PositionEmbedding = (weight = classifier_pos,),))
    end

    # 3. Blocks - copy matching layers
    if haskey(llada_params, :Blocks)
        blocks_params = new_params.Blocks
        new_blocks = Dict{Symbol, Any}()

        for i in 1:classifier.number_of_layers
            block_key = Symbol("Block_$i")

            if haskey(llada_params.Blocks, block_key)
                llada_block = llada_params.Blocks[block_key]
                classifier_block = blocks_params[block_key]

                # Copy matching sub-components
                new_block = _copy_block_params(classifier_block, llada_block)
                new_blocks[block_key] = new_block
            else
                new_blocks[block_key] = blocks_params[block_key]
            end
        end

        # Reconstruct blocks NamedTuple
        block_keys = ntuple(i -> Symbol("Block_$i"), classifier.number_of_layers)
        new_blocks_tuple = NamedTuple{block_keys}(Tuple(new_blocks[k] for k in block_keys))
        new_params = merge(new_params, (Blocks = new_blocks_tuple,))
    end

    return new_params
end

"""
Copy matching parameters from LLaDA block to classifier block.
"""
function _copy_block_params(classifier_block, llada_block)
    result = Dict{Symbol, Any}()

    for field in keys(classifier_block)
        if haskey(llada_block, field)
            # Try to copy recursively
            result[field] = _copy_params_recursive(classifier_block[field], llada_block[field])
        else
            # Keep classifier's initialization
            result[field] = classifier_block[field]
        end
    end

    return NamedTuple{Tuple(keys(result))}(values(result))
end

"""
Recursively copy parameters where shapes match.
"""
function _copy_params_recursive(dest, src)
    if src isa AbstractArray && dest isa AbstractArray
        if size(src) == size(dest)
            return copy(src)
        else
            # Shape mismatch, keep dest
            return dest
        end
    elseif src isa NamedTuple && dest isa NamedTuple
        result = Dict{Symbol, Any}()
        for k in keys(dest)
            if haskey(src, k)
                result[k] = _copy_params_recursive(dest[k], src[k])
            else
                result[k] = dest[k]
            end
        end
        return NamedTuple{Tuple(keys(result))}(values(result))
    else
        # Unknown type, keep dest
        return dest
    end
end

# ============================================================================
# Exports
# ============================================================================

export OssammaClassifier, ClassifierConfig
export SequencePooling, FixedTimeEmbedding
export tiny_classifier, small_classifier, base_classifier
export load_pretrained_encoder

end # module
