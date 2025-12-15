module ossm

import LuxCore, Lux, Random, NNlib

struct OSSMLayer <: Lux.AbstractLuxLayer
    input_dimension::Int
    output_dimension::Int
    oscillator_count::Int
end

@inline function ossm_state_dimension(block::OSSMLayer)
    # state_dimension = 2 * oscillator_count (two coordinates per oscillator)
    2 * block.oscillator_count
end

function apply_oscillation(block, current_state, decay_factor, rotation_angle)
    oscillator_count = block.oscillator_count
    @assert size(current_state) == (2 * oscillator_count, 1)             # Convention A: state is a (2H,1) column
    @assert length(decay_factor) == oscillator_count && length(rotation_angle) == oscillator_count

    # View state as oscillator columns, each a 2-vector
    state_view = reshape(current_state, 2, oscillator_count)              # (2, H)
    slices = eachslice(state_view; dims = 2)   # iterate columns => one oscillator state xi ∈ ℝ^2

    # Apply per-oscillator damped rotation: xi ↦ ρi * R(θi) * xi
    columns = [
        decay * [cos(angle) -sin(angle); sin(angle) cos(angle)] * state_slice
        for (decay, angle, state_slice) in zip(decay_factor, rotation_angle, slices)
    ]                                      # length-H collection of 2-vectors

    next_state_matrix = reduce(hcat, columns)            # (2, H)
    return reshape(next_state_matrix, 2 * oscillator_count, 1)          # back to (2H, 1)
end

function Lux.initialparameters(rng::Random.AbstractRNG, block::OSSMLayer)
    state_dimension = ossm_state_dimension(block)
    oscillator_count = block.oscillator_count

    return (
        # Continuous-time-ish params (learned)
        frequency = randn(rng, oscillator_count),                 # frequencies per oscillator (unconstrained)
        damping_raw = randn(rng, oscillator_count),                 # raw damping params (we'll softplus in forward to force α>0)

        # Input/output mixing
        input_matrix = randn(rng, state_dimension, block.input_dimension),
        output_matrix = randn(rng, block.output_dimension, state_dimension),
        feedthrough_matrix = zeros(Float32, block.output_dimension, block.input_dimension),

        # Selective step: time_step = softplus(TimeStepWeight * input_token + TimeStepBias)
        time_step_weight = randn(rng, oscillator_count, block.input_dimension),  # (H, input_dimension)
        time_step_bias = randn(rng, oscillator_count),                # (H,)
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, block::OSSMLayer)
    # Convention A: keep state as a (2H,1) column
    (; oscillation_state = zeros(Float32, ossm_state_dimension(block), 1))
end

function oscillator_step(block, parameters, current_state, input_token)
    # current_state :: (2H,1), input_token :: (input_dimension,1)
    (; frequency, damping_raw, input_matrix, output_matrix, feedthrough_matrix, time_step_weight, time_step_bias) = parameters

    @assert size(current_state) == (ossm_state_dimension(block), 1)
    @assert size(input_token) == (block.input_dimension, 1)

    # --- Selective step size time_step ---
    # time_step_weight * input_token => (H,1); time_step_bias is length-H, so reshape to (H,1) before adding.
    time_step = NNlib.softplus.(time_step_weight * input_token .+ reshape(time_step_bias, :, 1))  # (H,1)
    time_step = vec(time_step)                                        # (H,) so it zips nicely later

    # --- Stable damping and phase advance ---
    # Force damping positive so decay_factor = exp(-damping * time_step) stays in (0,1]
    damping_positive = NNlib.softplus.(damping_raw)            # (H,)
    decay_factor = exp.(-(damping_positive .* time_step))              # (H,)
    rotation_angle = frequency .* time_step                          # (H,)

    # State update: next_state = A(decay_factor, rotation_angle) * current_state + input_matrix * input_token
    next_state = apply_oscillation(block, current_state, decay_factor, rotation_angle) + input_matrix * input_token   # (2H,1)

    # Output: output = output_matrix * current_state + feedthrough_matrix * input_token  (you can switch to next_state if you prefer)
    output = output_matrix * current_state + feedthrough_matrix * input_token                                     # (output_dimension,1)

    return (output, (; oscillation_state = next_state))
end

function (block::OSSMLayer)(input_sequence, parameters, state)
    # input_sequence :: (input_dimension, T), columns are tokens
    initial_state = state.oscillation_state
    @assert size(initial_state) == (ossm_state_dimension(block), 1)

    sequence_length = size(input_sequence, 2)
    output_sequence = similar(input_sequence, block.output_dimension, sequence_length)  # preallocate (output_dimension, T)

    iterator = enumerate(eachslice(input_sequence; dims = 2))  # (t, slice) where slice is a length-input_dimension vector
    final_state = foldl(iterator; init = initial_state) do current_state, (timestep, slice)
        input_token = reshape(slice, block.input_dimension, 1)          # Convention A: (input_dimension, 1)
        output, new_state = oscillator_step(block, parameters, current_state, input_token)

        @views output_sequence[:, timestep] .= vec(output)                      # store output (output_dimension,1) into column t
        new_state.oscillation_state                           # return next state as fold accumulator
    end

    return (output_sequence, (; oscillation_state = final_state))
end
end
