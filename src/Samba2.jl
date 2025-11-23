module Samba2

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
    q = Lux.initialparameters(rng, block.Q)
    k = Lux.initialparameters(rng, block.K)
    v = Lux.initialparameters(rng, block.V)
    output = Lux.initialparameters(rng, block.OUTPUT)


    return (q = q, k = k, v = v, output = output)
end

function SWAttention(sequence_length::Int, dimension::Int, number_of_heads::Int)
    @assert dimension % number_of_heads == 0 "dimension must be divisible by number_of_heads" # we want perfect division
    # sub_dimension = div(dimension, number_of_heads)
    Q = Lux.Dense(dimension => dimension)
    K = Lux.Dense(dimension => dimension)
    V = Lux.Dense(dimension => dimension)
    OUTPUT = Lux.Dense(dimension => dimension)
    return SWAttention(sequence_length, dimension, number_of_heads, Q, K, V, OUTPUT)
end


function Lux.initialstates(rng::Random.AbstractRNG, block::SWAttention)
    return (;)
end

@inline function normalized_sigmoids(seq; τ = 1.0, eps = 1e-12)
    sigmoids = NNlib.sigmoid.(seq ./ τ)
    s = sum(sigmoids) + eps
    map!(x -> x / s, sigmoids, sigmoids)
    return sigmoids
end

function (block::SWAttention)(x, params::NamedTuple, _state::NamedTuple)
    state = (;)
    q, _ = block.Q(x, params.q, state)
    k, _ = block.K(x, params.k, state)
    v, _ = block.V(x, params.v, state)


    d_k = div(size(q, 1), block.number_of_heads)

    T = size(q, 2)
    # weights = Matrix(size(q, 1),)
    Q_heads = reshape(q, d_k, block.number_of_heads, T) |> x -> eachslice(x; dims = 2)
    K_heads = reshape(k, d_k, block.number_of_heads, T) |> x -> eachslice(x; dims = 2)
    V_heads = reshape(v, d_k, block.number_of_heads, T) |> x -> eachslice(x; dims = 2)

    head_outputs = map(Q_heads, K_heads, V_heads) do q_row, k_row, v_row
        Yh =
            q_row' * k_row / √d_k |>
            (x -> eachslice(x; dims = 1)) |>
            (row -> map(normalized_sigmoids, row)) |>
            (x -> (v_row * reduce(hcat, x)))
        return Yh
    end


    out, _ = reduce(vcat, head_outputs) |> Y -> block.OUTPUT(Y, params.output, state)
    return out, (;)
end

end
