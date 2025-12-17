module NER

"""
Token-level NER using Ossamma architecture.

Unlike OssammaClassifier (sequence → single label), OssammaNER outputs
a label for each token position (sequence → sequence of labels).

RAG-optimized 9-label schema:
    PERSON, AGENCY, PLACE, ORGANISM, EVENT,
    INSTRUMENT, CREATIVE_WORK, DOMAIN, MEASURE
"""

using Lux
using Random
using NNlib
using Statistics: mean

# Import parent module components
import ..OssammaBlock
import ..TimeConditionedLayerNorm

const LuxLayer = Lux.AbstractLuxLayer

# =============================================================================
# NER Label Schema
# =============================================================================

const RAG_LABELS = [
    "O",           # Outside any entity
    "B-PERSON", "I-PERSON",
    "B-AGENCY", "I-AGENCY",
    "B-PLACE", "I-PLACE",
    "B-ORGANISM", "I-ORGANISM",
    "B-EVENT", "I-EVENT",
    "B-INSTRUMENT", "I-INSTRUMENT",
    "B-WORK", "I-WORK",
    "B-DOMAIN", "I-DOMAIN",
    "B-MEASURE", "I-MEASURE",
]

const NUM_LABELS = length(RAG_LABELS)  # 19 (O + 9 entity types × 2 for B/I)

const LABEL_TO_ID = Dict(label => i for (i, label) in enumerate(RAG_LABELS))
const ID_TO_LABEL = Dict(i => label for (i, label) in enumerate(RAG_LABELS))

# Entity types without B/I prefix
const ENTITY_TYPES = [
    "PERSON", "AGENCY", "PLACE", "ORGANISM", "EVENT",
    "INSTRUMENT", "WORK", "DOMAIN", "MEASURE"
]

# =============================================================================
# Configuration
# =============================================================================

Base.@kwdef struct NERConfig
    # Architecture
    vocab_size::Int = 32000
    max_sequence_length::Int = 512
    embedding_dimension::Int = 256
    number_of_heads::Int = 4
    number_of_layers::Int = 4
    num_labels::Int = NUM_LABELS  # 19 for BIO tagging

    # Internal dimensions
    time_dimension::Int = 64
    state_dimension::Int = -1  # -1 means use embedding_dimension

    # Attention
    window_size::Int = 5

    # Oscillator SSM
    min_frequency::Float32 = 0.1f0
    max_frequency::Float32 = 10.0f0
    default_time_step::Float32 = 0.1f0

    # Training
    dropout_rate::Float32 = 0.1f0
    label_smoothing::Float32 = 0.0f0
end

# =============================================================================
# Fixed Time Embedding (for NER, no diffusion)
# =============================================================================

struct FixedTimeEmbedding <: LuxLayer
    time_dimension::Int
    fixed_value::Float32
end

function FixedTimeEmbedding(time_dimension::Int; fixed_value::Float32 = 0.5f0)
    return FixedTimeEmbedding(time_dimension, fixed_value)
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::FixedTimeEmbedding)
    half_dim = layer.time_dimension ÷ 2
    freqs = exp.(-(log(10000.0f0)) .* collect(Float32, 0:half_dim-1) ./ half_dim)
    args = freqs .* layer.fixed_value
    init_embedding = vcat(sin.(args), cos.(args))
    return (embedding = init_embedding,)
end

Lux.initialstates(::Random.AbstractRNG, ::FixedTimeEmbedding) = (;)

function (layer::FixedTimeEmbedding)(batch_size::Int, params, state)
    embedding = repeat(reshape(params.embedding, :, 1), 1, batch_size)
    return embedding, state
end

# =============================================================================
# OssammaNER Model
# =============================================================================

struct OssammaNER{E, P, T, B, D, H} <: LuxLayer
    vocab_size::Int
    max_sequence_length::Int
    embedding_dimension::Int
    number_of_heads::Int
    number_of_layers::Int
    num_labels::Int
    dropout_rate::Float32

    # Embeddings
    TokenEmbedding::E
    PositionEmbedding::P
    TimeEmbedding::T

    # Encoder blocks
    Blocks::B

    # Dropout before classification
    Dropout::D

    # Token classification head (per-token output)
    ClassificationHead::H
end

"""
    OssammaNER(config::NERConfig)

Create NER model from configuration.
"""
function OssammaNER(config::NERConfig)
    state_dimension = config.state_dimension == -1 ? config.embedding_dimension : config.state_dimension

    return OssammaNER(;
        vocab_size = config.vocab_size,
        max_sequence_length = config.max_sequence_length,
        embedding_dimension = config.embedding_dimension,
        number_of_heads = config.number_of_heads,
        number_of_layers = config.number_of_layers,
        num_labels = config.num_labels,
        time_dimension = config.time_dimension,
        state_dimension = state_dimension,
        window_size = config.window_size,
        min_frequency = config.min_frequency,
        max_frequency = config.max_frequency,
        default_time_step = config.default_time_step,
        dropout_rate = config.dropout_rate,
    )
end

function OssammaNER(;
    vocab_size::Int,
    max_sequence_length::Int,
    embedding_dimension::Int,
    number_of_heads::Int,
    number_of_layers::Int,
    num_labels::Int = NUM_LABELS,
    time_dimension::Int = 64,
    state_dimension::Int = embedding_dimension,
    window_size::Int = 5,
    min_frequency::Float32 = 0.1f0,
    max_frequency::Float32 = 10.0f0,
    default_time_step::Float32 = 0.1f0,
    dropout_rate::Float32 = 0.1f0,
)
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

    return OssammaNER(
        vocab_size,
        max_sequence_length,
        embedding_dimension,
        number_of_heads,
        number_of_layers,
        num_labels,
        dropout_rate,
        # Embeddings
        Lux.Embedding(vocab_size => embedding_dimension),
        Lux.Embedding(max_sequence_length => embedding_dimension),
        FixedTimeEmbedding(time_dimension),
        # Encoder blocks
        blocks,
        # Dropout before classification
        Lux.Dropout(dropout_rate),
        # Per-token classification: LayerNorm → Dropout → Dense → Labels
        Lux.Chain(
            Lux.LayerNorm((embedding_dimension,)),
            Lux.Dropout(dropout_rate),
            Lux.Dense(embedding_dimension => num_labels),
        ),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, model::OssammaNER)
    block_params = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(
        Tuple(Lux.initialparameters(rng, block) for block in model.Blocks)
    )

    return (
        TokenEmbedding = Lux.initialparameters(rng, model.TokenEmbedding),
        PositionEmbedding = Lux.initialparameters(rng, model.PositionEmbedding),
        TimeEmbedding = Lux.initialparameters(rng, model.TimeEmbedding),
        Blocks = block_params,
        Dropout = Lux.initialparameters(rng, model.Dropout),
        ClassificationHead = Lux.initialparameters(rng, model.ClassificationHead),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, model::OssammaNER)
    block_states = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(
        Tuple(Lux.initialstates(rng, block) for block in model.Blocks)
    )

    # Cache position indices to avoid allocation in forward pass
    position_indices = collect(1:model.max_sequence_length)

    return (
        TokenEmbedding = Lux.initialstates(rng, model.TokenEmbedding),
        PositionEmbedding = Lux.initialstates(rng, model.PositionEmbedding),
        TimeEmbedding = Lux.initialstates(rng, model.TimeEmbedding),
        Blocks = block_states,
        Dropout = Lux.initialstates(rng, model.Dropout),
        ClassificationHead = Lux.initialstates(rng, model.ClassificationHead),
        position_indices = position_indices,
    )
end

function (model::OssammaNER)(token_ids::AbstractArray, params, state)
    # token_ids: (seq_len,) or (seq_len, batch)
    # Output: (num_labels, seq_len) or (num_labels, seq_len, batch)

    is_batched = ndims(token_ids) == 2
    seq_len = size(token_ids, 1)
    batch_size = is_batched ? size(token_ids, 2) : 1

    # Standardize to batched format
    token_ids_batched = is_batched ? token_ids : reshape(token_ids, :, 1)

    # =========================================================================
    # 1. Token Embedding
    # =========================================================================
    token_flat = vec(token_ids_batched)
    token_emb_flat, tok_state = model.TokenEmbedding(token_flat, params.TokenEmbedding, state.TokenEmbedding)
    # (embedding_dim, seq_len * batch)
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
    hidden = token_emb .+ pos_emb  # (embedding_dim, seq_len, batch)

    # =========================================================================
    # 4. Fixed Time Embedding
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
    # 6. Apply dropout before classification
    # =========================================================================
    # hidden: (embedding_dim, seq_len, batch)
    # Flatten for dropout
    hidden_flat = reshape(hidden, model.embedding_dimension, :)
    hidden_flat, dropout_state = model.Dropout(hidden_flat, params.Dropout, state.Dropout)

    # =========================================================================
    # 7. Per-token Classification
    # =========================================================================
    logits_flat, head_state = model.ClassificationHead(hidden_flat, params.ClassificationHead, state.ClassificationHead)
    # logits_flat: (num_labels, seq_len * batch)

    logits = reshape(logits_flat, model.num_labels, seq_len, batch_size)

    # Remove batch dim if input wasn't batched
    final_logits = is_batched ? logits : dropdims(logits, dims=3)

    # =========================================================================
    # 8. Update State
    # =========================================================================
    new_block_states = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(
        block_states
    )

    new_state = (
        TokenEmbedding = tok_state,
        PositionEmbedding = pos_state,
        TimeEmbedding = time_state,
        Blocks = new_block_states,
        Dropout = dropout_state,
        ClassificationHead = head_state,
        position_indices = state.position_indices,
    )

    return final_logits, new_state
end

# =============================================================================
# Convenience Constructors
# =============================================================================

"""
    tiny_ner(; vocab_size, kwargs...)

Tiny NER model for debugging.
"""
function tiny_ner(; vocab_size::Int = 1000, max_sequence_length::Int = 64, kwargs...)
    config = NERConfig(;
        vocab_size = vocab_size,
        max_sequence_length = max_sequence_length,
        embedding_dimension = 64,
        number_of_heads = 2,
        number_of_layers = 2,
        time_dimension = 32,
        kwargs...
    )
    return OssammaNER(config)
end

"""
    small_ner(; vocab_size, kwargs...)

Small NER model for experiments.
"""
function small_ner(; vocab_size::Int = 32000, max_sequence_length::Int = 256, kwargs...)
    config = NERConfig(;
        vocab_size = vocab_size,
        max_sequence_length = max_sequence_length,
        embedding_dimension = 256,
        number_of_heads = 4,
        number_of_layers = 4,
        time_dimension = 64,
        kwargs...
    )
    return OssammaNER(config)
end

"""
    base_ner(; vocab_size, kwargs...)

Base NER model for production.
"""
function base_ner(; vocab_size::Int = 32000, max_sequence_length::Int = 512, kwargs...)
    config = NERConfig(;
        vocab_size = vocab_size,
        max_sequence_length = max_sequence_length,
        embedding_dimension = 512,
        number_of_heads = 8,
        number_of_layers = 6,
        time_dimension = 128,
        kwargs...
    )
    return OssammaNER(config)
end

# =============================================================================
# Loss Functions
# =============================================================================

"""
    ner_cross_entropy(logits, labels; ignore_index=-100)

Cross-entropy loss for NER, ignoring padding tokens.

- logits: (num_labels, seq_len, batch)
- labels: (seq_len, batch) integer labels
"""
function ner_cross_entropy(logits, labels; ignore_index::Int = -100)
    num_labels, seq_len, batch_size = size(logits)

    # Flatten
    logits_flat = reshape(logits, num_labels, :)  # (num_labels, seq_len * batch)
    labels_flat = vec(labels)  # (seq_len * batch,)

    # Create mask for valid positions
    mask = labels_flat .!= ignore_index
    valid_count = sum(mask)

    if valid_count == 0
        return 0.0f0
    end

    # Compute softmax cross entropy
    log_probs = NNlib.logsoftmax(logits_flat, dims=1)

    # Gather log probs at label positions
    # Only compute for valid positions
    total_loss = 0.0f0
    for i in 1:length(labels_flat)
        if mask[i]
            label = labels_flat[i]
            total_loss -= log_probs[label, i]
        end
    end

    return total_loss / valid_count
end

# =============================================================================
# Inference Utilities
# =============================================================================

"""
    predict_labels(model, params, state, token_ids) -> Vector{String}

Predict NER labels for a sequence.
"""
function predict_labels(model::OssammaNER, params, state, token_ids)
    logits, _ = model(token_ids, params, state)

    # Get argmax predictions
    predictions = vec(mapslices(argmax, logits, dims=1))

    # Convert to label strings
    return [ID_TO_LABEL[p] for p in predictions]
end

"""
    extract_entities(tokens, labels) -> Vector{NamedTuple}

Extract entities from tokens and BIO labels.
"""
function extract_entities(tokens::Vector{String}, labels::Vector{String})
    entities = NamedTuple{(:text, :label, :start, :end_), Tuple{String, String, Int, Int}}[]

    current_entity = nothing
    current_tokens = String[]
    current_start = 0

    for (i, (token, label)) in enumerate(zip(tokens, labels))
        if startswith(label, "B-")
            # Save previous entity if exists
            if current_entity !== nothing
                push!(entities, (
                    text = join(current_tokens, " "),
                    label = current_entity,
                    start = current_start,
                    end_ = i - 1,
                ))
            end

            # Start new entity
            current_entity = label[3:end]  # Remove "B-"
            current_tokens = [token]
            current_start = i

        elseif startswith(label, "I-") && current_entity == label[3:end]
            # Continue current entity
            push!(current_tokens, token)

        else
            # End current entity
            if current_entity !== nothing
                push!(entities, (
                    text = join(current_tokens, " "),
                    label = current_entity,
                    start = current_start,
                    end_ = i - 1,
                ))
                current_entity = nothing
                current_tokens = String[]
            end
        end
    end

    # Don't forget last entity
    if current_entity !== nothing
        push!(entities, (
            text = join(current_tokens, " "),
            label = current_entity,
            start = current_start,
            end_ = length(tokens),
        ))
    end

    return entities
end

# =============================================================================
# Exports
# =============================================================================

export OssammaNER, NERConfig
export tiny_ner, small_ner, base_ner
export ner_cross_entropy, predict_labels, extract_entities
export RAG_LABELS, ENTITY_TYPES, LABEL_TO_ID, ID_TO_LABEL, NUM_LABELS

end # module
