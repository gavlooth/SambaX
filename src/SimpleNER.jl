"""
SimpleNER - A minimal NER model using only attention (no OSSM) for fast training.
"""
module SimpleNER

using Lux
using Random
using NNlib

export SimpleNERModel, SimpleNERConfig

"""
Configuration for SimpleNER model.
"""
Base.@kwdef struct SimpleNERConfig
    vocab_size::Int = 32000
    max_sequence_length::Int = 512
    embedding_dimension::Int = 256
    number_of_heads::Int = 4
    number_of_layers::Int = 4
    num_labels::Int = 19
    dropout_rate::Float32 = 0.1f0
end

"""
Simple transformer block using standard multi-head attention.
"""
struct SimpleBlock <: Lux.AbstractLuxLayer
    embedding_dimension::Int
    number_of_heads::Int

    # Layers
    Attention::Lux.MultiHeadAttention
    FFN::Lux.Chain
    Norm1::Lux.LayerNorm
    Norm2::Lux.LayerNorm
    Dropout::Lux.Dropout
end

function SimpleBlock(dim::Int, heads::Int, dropout::Float32)
    head_dim = div(dim, heads)
    # MHA format: (embed_dim, num_heads)
    return SimpleBlock(
        dim,
        heads,
        Lux.MultiHeadAttention((dim, dim, dim) => (dim, dim); nheads=heads, attention_dropout_probability=dropout),
        Lux.Chain(
            Lux.Dense(dim => dim * 4, NNlib.gelu),
            Lux.Dense(dim * 4 => dim),
            Lux.Dropout(dropout)
        ),
        Lux.LayerNorm((dim, 1, 1)),  # For (dim, seq, batch) input
        Lux.LayerNorm((dim, 1, 1)),  # For (dim, seq, batch) input
        Lux.Dropout(dropout)
    )
end

function Lux.initialparameters(rng::AbstractRNG, block::SimpleBlock)
    return (
        Attention = Lux.initialparameters(rng, block.Attention),
        FFN = Lux.initialparameters(rng, block.FFN),
        Norm1 = Lux.initialparameters(rng, block.Norm1),
        Norm2 = Lux.initialparameters(rng, block.Norm2),
        Dropout = Lux.initialparameters(rng, block.Dropout),
    )
end

function Lux.initialstates(rng::AbstractRNG, block::SimpleBlock)
    return (
        Attention = Lux.initialstates(rng, block.Attention),
        FFN = Lux.initialstates(rng, block.FFN),
        Norm1 = Lux.initialstates(rng, block.Norm1),
        Norm2 = Lux.initialstates(rng, block.Norm2),
        Dropout = Lux.initialstates(rng, block.Dropout),
    )
end

function (block::SimpleBlock)(x, params, state)
    # x: (dim, seq, batch) - transpose for MHA which expects (seq, batch, dim) or similar
    # Lux MultiHeadAttention expects (features, seq_len, batch)

    # Pre-norm attention
    x_norm, norm1_state = block.Norm1(x, params.Norm1, state.Norm1)

    # Self attention - MHA expects (features, length, batch)
    attn_out, attn_state = block.Attention(x_norm, params.Attention, state.Attention)

    # Dropout + residual
    attn_dropped, drop_state = block.Dropout(attn_out, params.Dropout, state.Dropout)
    x = x .+ attn_dropped

    # Pre-norm FFN
    x_norm2, norm2_state = block.Norm2(x, params.Norm2, state.Norm2)
    ffn_out, ffn_state = block.FFN(x_norm2, params.FFN, state.FFN)
    x = x .+ ffn_out

    new_state = (
        Attention = attn_state,
        FFN = ffn_state,
        Norm1 = norm1_state,
        Norm2 = norm2_state,
        Dropout = drop_state,
    )

    return x, new_state
end

"""
Simple NER model using standard transformer architecture.
"""
struct SimpleNERModel <: Lux.AbstractLuxLayer
    config::SimpleNERConfig

    TokenEmbedding::Lux.Embedding
    PositionEmbedding::Lux.Embedding
    Blocks::Vector{SimpleBlock}
    FinalNorm::Lux.LayerNorm
    Classifier::Lux.Dense
    Dropout::Lux.Dropout
end

function SimpleNERModel(config::SimpleNERConfig)
    blocks = [SimpleBlock(config.embedding_dimension, config.number_of_heads, config.dropout_rate)
              for _ in 1:config.number_of_layers]

    return SimpleNERModel(
        config,
        Lux.Embedding(config.vocab_size => config.embedding_dimension),
        Lux.Embedding(config.max_sequence_length => config.embedding_dimension),
        blocks,
        Lux.LayerNorm((config.embedding_dimension, 1, 1)),  # For 3D input
        Lux.Dense(config.embedding_dimension => config.num_labels),
        Lux.Dropout(config.dropout_rate)
    )
end

function Lux.initialparameters(rng::AbstractRNG, model::SimpleNERModel)
    block_params = NamedTuple{Tuple(Symbol.("Block_" .* string.(1:length(model.Blocks))))}(
        Tuple(Lux.initialparameters(rng, b) for b in model.Blocks)
    )
    return (
        TokenEmbedding = Lux.initialparameters(rng, model.TokenEmbedding),
        PositionEmbedding = Lux.initialparameters(rng, model.PositionEmbedding),
        Blocks = block_params,
        FinalNorm = Lux.initialparameters(rng, model.FinalNorm),
        Classifier = Lux.initialparameters(rng, model.Classifier),
        Dropout = Lux.initialparameters(rng, model.Dropout),
    )
end

function Lux.initialstates(rng::AbstractRNG, model::SimpleNERModel)
    block_states = NamedTuple{Tuple(Symbol.("Block_" .* string.(1:length(model.Blocks))))}(
        Tuple(Lux.initialstates(rng, b) for b in model.Blocks)
    )
    return (
        TokenEmbedding = Lux.initialstates(rng, model.TokenEmbedding),
        PositionEmbedding = Lux.initialstates(rng, model.PositionEmbedding),
        Blocks = block_states,
        FinalNorm = Lux.initialstates(rng, model.FinalNorm),
        Classifier = Lux.initialstates(rng, model.Classifier),
        Dropout = Lux.initialstates(rng, model.Dropout),
    )
end

function (model::SimpleNERModel)(token_ids, params, state)
    # token_ids: (seq_len, batch) or (seq_len,)
    is_batched = ndims(token_ids) == 2
    if !is_batched
        token_ids = reshape(token_ids, :, 1)
    end

    seq_len, batch_size = size(token_ids)

    # Token embeddings
    token_emb, tok_state = model.TokenEmbedding(token_ids, params.TokenEmbedding, state.TokenEmbedding)
    # token_emb: (dim, seq, batch)

    # Position embeddings
    positions = collect(1:seq_len)
    pos_emb, pos_state = model.PositionEmbedding(positions, params.PositionEmbedding, state.PositionEmbedding)
    # pos_emb: (dim, seq)

    # Combine embeddings
    hidden = token_emb .+ pos_emb

    # Process through blocks
    block_states = []
    for (i, block) in enumerate(model.Blocks)
        block_key = Symbol("Block_$i")
        hidden, new_block_state = block(hidden, params.Blocks[block_key], state.Blocks[block_key])
        push!(block_states, new_block_state)
    end

    # Final norm
    hidden, norm_state = model.FinalNorm(hidden, params.FinalNorm, state.FinalNorm)

    # Dropout
    hidden, drop_state = model.Dropout(hidden, params.Dropout, state.Dropout)

    # Classification
    logits, class_state = model.Classifier(hidden, params.Classifier, state.Classifier)
    # logits: (num_labels, seq, batch)

    if !is_batched
        logits = dropdims(logits, dims=3)
    end

    # Build new state
    new_block_states = NamedTuple{Tuple(Symbol.("Block_" .* string.(1:length(model.Blocks))))}(
        Tuple(block_states)
    )

    new_state = (
        TokenEmbedding = tok_state,
        PositionEmbedding = pos_state,
        Blocks = new_block_states,
        FinalNorm = norm_state,
        Classifier = class_state,
        Dropout = drop_state,
    )

    return logits, new_state
end

# Loss function
function simple_ner_cross_entropy(logits, labels; ignore_index=-100)
    # logits: (num_labels, seq_len, batch) or (num_labels, seq_len)
    # labels: (seq_len, batch) or (seq_len,)

    num_labels = size(logits, 1)

    # Flatten
    logits_flat = reshape(logits, num_labels, :)
    labels_flat = vec(labels)

    # Create mask
    mask = labels_flat .!= ignore_index

    # Replace ignored labels with 1 (valid index)
    labels_safe = ifelse.(mask, labels_flat, 1)

    # Compute log softmax
    log_probs = NNlib.logsoftmax(logits_flat; dims=1)

    # Gather losses for each position
    n_positions = length(labels_safe)
    losses = [-log_probs[labels_safe[i], i] for i in 1:n_positions]

    # Apply mask and compute mean
    masked_losses = losses .* mask
    total_loss = sum(masked_losses)
    n_valid = sum(mask)

    return n_valid > 0 ? total_loss / n_valid : 0.0f0
end

end # module
