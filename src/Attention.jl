module Attention

import LuxCore, Lux, NNlib, Random

cl = 1024
dim = 128

struct SWAttention <: Lux.AbstractLuxLayer
    sequence_length::Int
    dimension::Int
    number_of_heads::Int
    Q::Lux.Dense
    K::Lux.Dense
    V::Lux.Dense
    OUTPUT::Lux.Dense
end

function Lux.initialparameters(rng::Random.AbstractRNG, block::SWAttention)
    # Initialize parameters for Q, K, V, OUTPUT projection layers
    # Each Dense layer has weight: (output_dim, input_dim) and bias: (output_dim,)
    # All projections are (dimension, dimension)
    q = Lux.initialparameters(rng, block.Q)       # Q projection params
    k = Lux.initialparameters(rng, block.K)       # K projection params
    v = Lux.initialparameters(rng, block.V)       # V projection params
    output = Lux.initialparameters(rng, block.OUTPUT)  # output projection params


    return (q = q, k = k, v = v, output = output)
end

function SWAttention(sequence_length::Int, dimension::Int, number_of_heads::Int)
    # sequence_length: maximum sequence length (stored but not enforced)
    # dimension: embedding dimension (must be divisible by number_of_heads)
    # number_of_heads: number of attention heads (dimension is split across heads)
    @assert dimension % number_of_heads == 0 "dimension must be divisible by number_of_heads" # we want perfect division
    # Each head processes d_k = dimension / number_of_heads dimensions
    Q = Lux.Dense(dimension => dimension)  # Query projection
    K = Lux.Dense(dimension => dimension)  # Key projection
    V = Lux.Dense(dimension => dimension)  # Value projection
    OUTPUT = Lux.Dense(dimension => dimension)  # Output projection
    return SWAttention(sequence_length, dimension, number_of_heads, Q, K, V, OUTPUT)
end


function Lux.initialstates(rng::Random.AbstractRNG, block::SWAttention)
    # SWAttention is stateless - returns empty NamedTuple
    return (;)
end

@inline function normalized_sigmoids(seq; τ = 1.0, eps = 1e-12)
    # seq: vector of attention scores (length T)
    # τ: temperature parameter (default 1.0)
    # eps: small constant to prevent division by zero
    # Returns: normalized attention weights that sum to ~1.0 (length T)
    sigmoids = NNlib.sigmoid.(seq ./ τ)  # apply sigmoid with temperature scaling
    s = sum(sigmoids) + eps  # sum for normalization
    map!(x -> x / s, sigmoids, sigmoids)  # in-place normalization
    return sigmoids
end

function (block::SWAttention)(x, params::NamedTuple, _state::NamedTuple)
    # x: (dimension, T) where T is sequence length
    state = (;)
    q, _ = block.Q(x, params.q, state)  # (dimension, T)
    k, _ = block.K(x, params.k, state)  # (dimension, T)
    v, _ = block.V(x, params.v, state)  # (dimension, T)


    d_k = div(size(q, 1), block.number_of_heads)  # dimension per head

    T = size(q, 2)
    # Split into heads: (dimension, T) -> (d_k, num_heads, T) -> iterator of (d_k, T) slices
    Q_heads = reshape(q, d_k, block.number_of_heads, T) |> x -> eachslice(x; dims = 2)
    K_heads = reshape(k, d_k, block.number_of_heads, T) |> x -> eachslice(x; dims = 2)
    V_heads = reshape(v, d_k, block.number_of_heads, T) |> x -> eachslice(x; dims = 2)

    head_outputs = map(Q_heads, K_heads, V_heads) do q_row, k_row, v_row
        # q_row, k_row, v_row: each (d_k, T)
        Yh =
            q_row' * k_row / √d_k |>              # (T, d_k) * (d_k, T) -> (T, T) attention scores
            (x -> eachslice(x; dims = 1)) |>      # split into T rows of length T
            (row -> map(normalized_sigmoids, row)) |>  # normalize each row -> attention weights
            (x -> (v_row * reduce(hcat, x)))      # (d_k, T) * (T, T) -> (d_k, T) weighted values
        return Yh  # (d_k, T)
    end


    # Concatenate heads: num_heads × (d_k, T) -> (dimension, T)
    out, _ = reduce(vcat, head_outputs) |> Y -> block.OUTPUT(Y, params.output, state)  # (dimension, T)
    return out, (;)
end

end
