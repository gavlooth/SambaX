module LinearizedAttention

using Lux
using Random
using NNlib

const LuxAttentionSupertype =
    isdefined(Lux, :AbstractExplicitLayer) ? Lux.AbstractExplicitLayer :
    Lux.AbstractLuxLayer

struct LAttention <: LuxAttentionSupertype
    sequence_length::Int
    embedding_dimension::Int
    number_of_heads::Int
    head_dimension::Int    # Pre-calculated dimension per head

    # Layers for projections
    QueryProjection::Lux.Dense
    KeyProjection::Lux.Dense
    ValueProjection::Lux.Dense
    OutputProjection::Lux.Dense
end


@inline function sigsoftmax(logits; dims = 1)
    transformed_logits = logits .+ NNlib.logsigmoid.(logits)
    return NNlib.softmax(transformed_logits; dims = dims)
end

# 1. CONSTRUCTOR
function LAttention(
    sequence_length::Int,
    embedding_dimension::Int,
    number_of_heads::Int;
    window_size::Int = 5,
)
    @assert embedding_dimension % number_of_heads == 0 "Embedding dimension must be divisible by the number of heads."

    calculated_head_dimension = div(embedding_dimension, number_of_heads)

    return SWAttention(
        sequence_length,
        embedding_dimension,
        number_of_heads,
        window_size,
        calculated_head_dimension,
        Lux.Dense(embedding_dimension => embedding_dimension), # Query
        Lux.Dense(embedding_dimension => embedding_dimension), # Key
        Lux.Dense(embedding_dimension => embedding_dimension), # Value
        Lux.Dense(embedding_dimension => embedding_dimension),  # Output
    )
end

# 2. PARAMETER INITIALIZATION
# Explicitly define this to ensure params are structured correctly and accessible by field name
function Lux.initialparameters(rng::Random.AbstractRNG, layer::SWAttention)
    return (
        QueryProjection = Lux.initialparameters(rng, layer.QueryProjection),
        KeyProjection = Lux.initialparameters(rng, layer.KeyProjection),
        ValueProjection = Lux.initialparameters(rng, layer.ValueProjection),
        OutputProjection = Lux.initialparameters(rng, layer.OutputProjection),
    )
end

# 3. STATE INITIALIZATION
# We need a custom initialstates to include both child states and our mask
function Lux.initialstates(rng::Random.AbstractRNG, layer::SWAttention)
    # Initialize mask based on the hint sequence_length
    mask = build_sliding_window_mask(layer.sequence_length, layer.window_size)

    # recursively initialize child states
    child_states = (
        QueryProjection = Lux.initialstates(rng, layer.QueryProjection),
        KeyProjection = Lux.initialstates(rng, layer.KeyProjection),
        ValueProjection = Lux.initialstates(rng, layer.ValueProjection),
        OutputProjection = Lux.initialstates(rng, layer.OutputProjection),
    )

    return merge(child_states, (; window_mask = mask))
end

# 4. FORWARD PASS
function (layer::SWAttention)(input_tensor::AbstractArray, params, state)
    # ---------------------------------------------------------
    # A. Handle Dimensions & Masking
    # ---------------------------------------------------------
    # Expected Input: (Features, Time) or (Features, Time, Batch)
    is_input_batched = ndims(input_tensor) == 3

    # Get current sequence length T
    current_T = is_input_batched ? size(input_tensor, 2) : size(input_tensor, 2)

    # Dynamic Masking Check
    cached_mask = state.window_mask
    # Check if cached mask is valid for current input T. 
    # Assuming square mask (T, T).
    active_mask = if size(cached_mask, 1) == current_T
        cached_mask
    else
        build_sliding_window_mask(current_T, layer.window_size)
    end

    # Standardize to 3D Tensor: (Features, Time, Batch)
    input_3d_tensor =
        is_input_batched ? input_tensor :
        reshape(input_tensor, size(input_tensor, 1), size(input_tensor, 2), 1)

    (feature_dimension, sequence_length, batch_size) = size(input_3d_tensor)

    # ---------------------------------------------------------
    # B. Projections (Q, K, V)
    # ---------------------------------------------------------
    # Reshape to 2D (Features, Time * Batch) for efficient Dense Layer processing
    input_flattened_for_projection = reshape(input_3d_tensor, feature_dimension, :)

    # Apply Dense Layers with correct params/state
    # Params are automatically structured by Lux as (QueryProjection=..., etc.)
    # State is structured by us in initialstates

    q_flat, q_st = layer.QueryProjection(
        input_flattened_for_projection,
        params.QueryProjection,
        state.QueryProjection,
    )
    k_flat, k_st = layer.KeyProjection(
        input_flattened_for_projection,
        params.KeyProjection,
        state.KeyProjection,
    )
    v_flat, v_st = layer.ValueProjection(
        input_flattened_for_projection,
        params.ValueProjection,
        state.ValueProjection,
    )

    # Reshape back to 3D: (Features, Time, Batch)
    query_tensor = reshape(q_flat, feature_dimension, sequence_length, batch_size)
    key_tensor = reshape(k_flat, feature_dimension, sequence_length, batch_size)
    value_tensor = reshape(v_flat, feature_dimension, sequence_length, batch_size)

    # ---------------------------------------------------------
    # C. Multi-Head Splitting
    # ---------------------------------------------------------
    # Target Shape: (Head_Dim, Heads, Time, Batch) -> (Head_Dim, Time, Heads, Batch)

    reshape_and_permute =
        x -> begin
            x_r = reshape(
                x,
                layer.head_dimension,
                layer.number_of_heads,
                sequence_length,
                batch_size,
            )
            permutedims(x_r, (1, 3, 2, 4))
        end

    query_permuted = reshape_and_permute(query_tensor)
    key_permuted = reshape_and_permute(key_tensor)
    value_permuted = reshape_and_permute(value_tensor)

    # ---------------------------------------------------------
    # D. Attention Score Calculation
    # ---------------------------------------------------------
    # K^T * Q
    # Key Transposed: (Time, Head_Dim, Heads, Batch)
    key_transposed_for_score = permutedims(key_permuted, (2, 1, 3, 4))

    # Compute raw scores scaled by sqrt(d_k)
    scaling_factor = sqrt(Float32(layer.head_dimension))
    attention_scores_raw =
        NNlib.batched_mul(key_transposed_for_score, query_permuted) ./ scaling_factor

    # ---------------------------------------------------------
    # E. Sliding Window Masking
    # ---------------------------------------------------------
    masked_attention_scores = apply_sliding_window_mask(attention_scores_raw, active_mask)

    # ---------------------------------------------------------
    # F. Normalization (SigSoftmax Attention)
    # ---------------------------------------------------------
    normalized_attention_weights = sigsoftmax(masked_attention_scores; dims = 1)

    # ---------------------------------------------------------
    # G. Weighted Aggregation
    # ---------------------------------------------------------
    # Value * Weights
    weighted_values = NNlib.batched_mul(value_permuted, normalized_attention_weights)

    # ---------------------------------------------------------
    # H. Output Projection
    # ---------------------------------------------------------
    # Permute back: (Head_Dim, Heads, Time, Batch)
    weighted_values_permuted = permutedims(weighted_values, (1, 3, 2, 4))

    # Merge Heads: (Feature_Dim, Time, Batch)
    output_merged_heads =
        reshape(weighted_values_permuted, feature_dimension, sequence_length, batch_size)

    # Flatten for Dense Layer
    output_flattened_for_projection = reshape(output_merged_heads, feature_dimension, :)

    final_output_flat, o_st = layer.OutputProjection(
        output_flattened_for_projection,
        params.OutputProjection,
        state.OutputProjection,
    )

    # Restore 3D Shape
    final_output_3d =
        reshape(final_output_flat, feature_dimension, sequence_length, batch_size)

    # Handle Batch Dimension
    final_output = is_input_batched ? final_output_3d : dropdims(final_output_3d, dims = 3)

    # Update State
    new_state = (
        QueryProjection = q_st,
        KeyProjection = k_st,
        ValueProjection = v_st,
        OutputProjection = o_st,
        window_mask = active_mask,
    )

    return final_output, new_state
end

end # module
