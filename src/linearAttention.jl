module LinearAttention

using Lux
using Random
using NNlib

const LuxAttentionSupertype = isdefined(Lux, :AbstractExplicitLayer) ?
                              Lux.AbstractExplicitLayer :
                              Lux.AbstractLuxLayer

struct LinearAttention{Q,K,V,O,QC,KC,TP,PC,PS,LN} <: LuxAttentionSupertype
    sequence_length::Int
    embedding_dimension::Int
    number_of_heads::Int
    head_dimension::Int

    # Projections
    QueryProjection::Q
    KeyProjection::K
    ValueProjection::V
    OutputProjection::O

    # Feature Maps: Now just linear projections (No activation yet!)
    QueryFeatureLinear::QC
    KeyFeatureLinear::KC

    # Position Embeddings
    PositionEmbeddingCosine::PC
    PositionEmbeddingSine::PS

    # Time Conditioning
    TimeProjection::TP

    # STABILITY: LayerNorm for the features before Attn
    FeatureNorm::LN
end

function LinearAttention(embedding_dimension::Int, sequence_length::Int, number_of_heads::Int, time_dimension::Int)
    head_dimension = div(embedding_dimension, number_of_heads)

    # 1. Linear Only for the feature map (we activate later)
    # We expand to 2x dim to give the model "workspace", then project back
    make_linear_map() = Chain(
        Dense(head_dimension => head_dimension * 2, gelu),
        Dense(head_dimension * 2 => head_dimension), # No activation here!
    )

    return LinearAttention(
        sequence_length,
        embedding_dimension,
        number_of_heads,
        head_dimension,
        Dense(embedding_dimension => embedding_dimension),
        Dense(embedding_dimension => embedding_dimension),
        Dense(embedding_dimension => embedding_dimension),
        Dense(embedding_dimension => embedding_dimension),
        make_linear_map(),
        make_linear_map(),
        Embedding(sequence_length => head_dimension),
        Embedding(sequence_length => head_dimension),
        Dense(time_dimension => head_dimension),

        # Stability: Normalizes the Q/K streams to prevent explosion
        LayerNorm(head_dimension),
    )
end

# PARAMETER INITIALIZATION
function Lux.initialparameters(rng::Random.AbstractRNG, layer::LinearAttention)
    return (
        QueryProjection = Lux.initialparameters(rng, layer.QueryProjection),
        KeyProjection = Lux.initialparameters(rng, layer.KeyProjection),
        ValueProjection = Lux.initialparameters(rng, layer.ValueProjection),
        OutputProjection = Lux.initialparameters(rng, layer.OutputProjection),
        QueryFeatureLinear = Lux.initialparameters(rng, layer.QueryFeatureLinear),
        KeyFeatureLinear = Lux.initialparameters(rng, layer.KeyFeatureLinear),
        PositionEmbeddingCosine = Lux.initialparameters(rng, layer.PositionEmbeddingCosine),
        PositionEmbeddingSine = Lux.initialparameters(rng, layer.PositionEmbeddingSine),
        TimeProjection = Lux.initialparameters(rng, layer.TimeProjection),
        FeatureNorm = Lux.initialparameters(rng, layer.FeatureNorm),
    )
end

# STATE INITIALIZATION
function Lux.initialstates(rng::Random.AbstractRNG, layer::LinearAttention)
    return (
        QueryProjection = Lux.initialstates(rng, layer.QueryProjection),
        KeyProjection = Lux.initialstates(rng, layer.KeyProjection),
        ValueProjection = Lux.initialstates(rng, layer.ValueProjection),
        OutputProjection = Lux.initialstates(rng, layer.OutputProjection),
        QueryFeatureLinear = Lux.initialstates(rng, layer.QueryFeatureLinear),
        KeyFeatureLinear = Lux.initialstates(rng, layer.KeyFeatureLinear),
        PositionEmbeddingCosine = Lux.initialstates(rng, layer.PositionEmbeddingCosine),
        PositionEmbeddingSine = Lux.initialstates(rng, layer.PositionEmbeddingSine),
        TimeProjection = Lux.initialstates(rng, layer.TimeProjection),
        FeatureNorm = Lux.initialstates(rng, layer.FeatureNorm),
    )
end

function (layer::LinearAttention)(inputs::Tuple, parameters, state)
    input_tensor, time_input = inputs

    # 1. Projections
    query_tensor, query_state = layer.QueryProjection(input_tensor, parameters.QueryProjection, state.QueryProjection)
    key_tensor, key_state = layer.KeyProjection(input_tensor, parameters.KeyProjection, state.KeyProjection)
    value_tensor, value_state = layer.ValueProjection(input_tensor, parameters.ValueProjection, state.ValueProjection)

    # 2. Time & Reshape
    time_embedding, time_state = layer.TimeProjection(time_input, parameters.TimeProjection, state.TimeProjection)
    time_broadcast = reshape(time_embedding, layer.head_dimension, 1, 1, size(time_input, 2))

    reshape_to_heads(tensor) = reshape(tensor, layer.head_dimension, layer.number_of_heads, layer.sequence_length, :)
    query_heads = reshape_to_heads(query_tensor)
    key_heads = reshape_to_heads(key_tensor)
    value_heads = reshape_to_heads(value_tensor)

    # 3. Content Map (Linear Mix)
    # Add time BEFORE the dense layer
    query_with_time = query_heads .+ time_broadcast
    key_with_time = key_heads .+ time_broadcast

    collapse_dimensions(tensor) = reshape(tensor, layer.head_dimension, :)
    query_features_raw, query_feature_state =
        layer.QueryFeatureLinear(collapse_dimensions(query_with_time), parameters.QueryFeatureLinear, state.QueryFeatureLinear)
    key_features_raw, key_feature_state =
        layer.KeyFeatureLinear(collapse_dimensions(key_with_time), parameters.KeyFeatureLinear, state.KeyFeatureLinear)

    # Reshape back to (D, Heads, Seq, Batch) for broadcasting
    query_features_linear = reshape(query_features_raw, layer.head_dimension, layer.number_of_heads, layer.sequence_length, :)
    key_features_linear = reshape(key_features_raw, layer.head_dimension, layer.number_of_heads, layer.sequence_length, :)

    # 4. Position Injection & STABILIZED Activation
    position_indices = 1:layer.sequence_length
    position_cosine_raw, position_cosine_state = layer.PositionEmbeddingCosine(position_indices, parameters.PositionEmbeddingCosine, state.PositionEmbeddingCosine)
    position_sine_raw, position_sine_state = layer.PositionEmbeddingSine(position_indices, parameters.PositionEmbeddingSine, state.PositionEmbeddingSine)

    position_cosine = reshape(position_cosine_raw, layer.head_dimension, 1, layer.sequence_length, 1)
    position_sine = reshape(position_sine_raw, layer.head_dimension, 1, layer.sequence_length, 1)

    # --- The Fix: Multiply First, Softplus Last ---
    # This allows negative embeddings to flip the signal,
    # but ensures the final feature map is positive (Safe for Attention)

    # Stream 1
    query_stream_cosine = softplus.(query_features_linear .* position_cosine)
    key_stream_cosine = softplus.(key_features_linear .* position_cosine)

    # Stream 2
    query_stream_sine = softplus.(query_features_linear .* position_sine)
    key_stream_sine = softplus.(key_features_linear .* position_sine)

    # 5. Extra Stability: LayerNorm the Features
    # Linear Attention is sensitive to magnitude. Normalizing features fixes this.
    # (Applying LN per token vector)
    # Flatten to 2D for LayerNorm, then reshape back to 4D
    flatten_for_norm(tensor) = reshape(tensor, layer.head_dimension, :)
    unflatten_from_norm(tensor) = reshape(tensor, layer.head_dimension, layer.number_of_heads, layer.sequence_length, :)

    query_stream_cosine_flat, feature_norm_state_one = layer.FeatureNorm(flatten_for_norm(query_stream_cosine), parameters.FeatureNorm, state.FeatureNorm)
    query_stream_cosine = unflatten_from_norm(query_stream_cosine_flat)

    key_stream_cosine_flat, feature_norm_state_two = layer.FeatureNorm(flatten_for_norm(key_stream_cosine), parameters.FeatureNorm, state.FeatureNorm)
    key_stream_cosine = unflatten_from_norm(key_stream_cosine_flat)

    query_stream_sine_flat, feature_norm_state_three = layer.FeatureNorm(flatten_for_norm(query_stream_sine), parameters.FeatureNorm, state.FeatureNorm)
    query_stream_sine = unflatten_from_norm(query_stream_sine_flat)

    key_stream_sine_flat, feature_norm_state_four = layer.FeatureNorm(flatten_for_norm(key_stream_sine), parameters.FeatureNorm, state.FeatureNorm)
    key_stream_sine = unflatten_from_norm(key_stream_sine_flat)

    # 6. Linear Attention (Standard)
    merge_batch_dimensions(tensor) = reshape(tensor, layer.head_dimension, layer.sequence_length, :)
    value_sequence = merge_batch_dimensions(value_heads)

    function run_attention_stream(query_stream, key_stream)
        query_sequence = merge_batch_dimensions(query_stream)
        key_sequence = merge_batch_dimensions(key_stream)
        context_matrix = NNlib.batched_mul(value_sequence, permutedims(key_sequence, (2, 1, 3)))
        return NNlib.batched_mul(context_matrix, query_sequence)
    end

    attention_output = run_attention_stream(query_stream_cosine, key_stream_cosine) .+ run_attention_stream(query_stream_sine, key_stream_sine)

    # 7. Final Projection
    output_reshaped = reshape(attention_output, layer.embedding_dimension, layer.sequence_length, :)
    final_output, output_state = layer.OutputProjection(output_reshaped, parameters.OutputProjection, state.OutputProjection)

    # Update State
    new_state = (
        QueryProjection = query_state,
        KeyProjection = key_state,
        ValueProjection = value_state,
        OutputProjection = output_state,
        QueryFeatureLinear = query_feature_state,
        KeyFeatureLinear = key_feature_state,
        PositionEmbeddingCosine = position_cosine_state,
        PositionEmbeddingSine = position_sine_state,
        TimeProjection = time_state,
        FeatureNorm = feature_norm_state_four,
    )

    return final_output, new_state
end
end # module
