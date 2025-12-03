module DLinoss

import LuxCore, Lux, Random, NNlib


struct DLinOSS <: Lux.AbstractLuxLayer
    input_dimensions::Int
    state_dimensions::Int  # oscillaotr count
    output_dimensions::Int
    min_frequency::Float32  # Store the CONFIG
    max_frequency::Float32
    Δt::Float32
end

#  C shape: (in_dims, state_dims) (Note: D-LinOSS usually sets output dim = input dim).

function Lux.initialparameters(rng, layer::DLinOSS)

    freqs = range(
        log(layer.min_frequency),
        log(layer.max_frequency),
        length = layer.state_dimensions,
    )

    log_Δt = randn(rng, Float32, layer.state_dimensions) .* layer.Δt
    log_A = collect(
        range(
            log(layer.min_frequency),
            log(layer.max_frequency),
            length = layer.state_dimensions,
        ),
    )
    log_G = zeros(rng, Float32, layer.state_dimensions) .* 0.01f0

    B =
        0.01f0 .* randn(rng, Float32, layer.state_dimensions, layer.input_dimensions) .*
        log(0.01f0)
    C =
        0.01f0 .* randn(rng, Float32, layer.input_dimensions, layer.output_dimensions) .*
        log(0.01f0)

    return (log_Δt = log_Δt, log_A = log_A, log_G = log_G, B = B, C = C)

end

function Lux.initialstates(_rng::Random.AbstractRNG, layer::DLinOSS)
    (oscillator_state = zeros(Float32, 2, layer.state_dimensions),)

end



function (layer::DLinOSS)(token_sequence, params, state)
    #shape of  token_sequence    dimension x sequence_length
    (; log_Δt, log_A, log_G, B, C) = params
    A = exp.(log_A)
    Δt = exp.(log_Δt)
    G = exp.(log_G)
    sequence_length = size(token_sequence, 2)

    state = state.oscillator_state

    velocity = @view state[1, :]
    position = @view state[2, :]

    D_inverse = 1.0f0 ./ (1.0f0 .+ Δt .* G)
    mzz = D_inverse
    mzy = -Δt .* A .* D_inverse

    fz = Δt .* D_inverse
    fy = Δt .^ 2 .* D_inverse

    #shape of B state_dimensions x input_dimensions 
    sequence_state_projection = B * token_sequence   # shape: (state_dimensions, sequence_length)

    states_projections = eachslice(sequence_state_projection, dims = 2)

    step =
        (state, state_projection) -> begin
            previous_velocity, previous_position = state
            next_velocity =
                (mzz .* previous_velocity) .+ (mzy .* previous_position) .+
                (fz .* state_projection)
            next_position = previous_position .+ (Δt .* next_velocity)
            (next_velocity, next_position)
        end

    init_state = (velocity, position)
    history = accumulate(step, states_projections; init = init_state)
    positions = map(last, history) #or proadcast last.(history)
    output = C * stack(positions, dims = 2)  # make this a column vector for the update
    next_state = (oscillator_state = stack(history[end], dims = 1),) #last state

    return (output, next_state)
end


end
