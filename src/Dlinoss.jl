module Dlinoss

using Lux
using Random
using NNlib

struct DLinOSS <: Lux.AbstractLuxLayer
    input_dimensions::Int
    state_dimensions::Int
    output_dimensions::Int
    min_frequency::Float32
    max_frequency::Float32
    dt::Float32
end

# 1. PARAMETER INITIALIZATION
function Lux.initialparameters(rng::Random.AbstractRNG, layer::DLinOSS)

    log_stiffness_coefficients = collect(
        range(
            log(layer.min_frequency),
            log(layer.max_frequency),
            length = layer.state_dimensions,
        ),
    )

    log_time_step = ones(Float32, layer.state_dimensions) .* log(layer.dt)

    log_damping_coefficients = ones(Float32, layer.state_dimensions) .* log(0.01f0)

    input_projection =
        randn(rng, Float32, layer.state_dimensions, layer.input_dimensions) .* 0.02f0

    output_projection =
        randn(rng, Float32, layer.output_dimensions, layer.state_dimensions) .* 0.02f0

    return (
        log_time_step = log_time_step,
        log_stiffness_coefficients = log_stiffness_coefficients,
        log_damping_coefficients = log_damping_coefficients,
        input_projection = input_projection,
        output_projection = output_projection,
    )
end

# 2. STATE INITIALIZATION
function Lux.initialstates(_rng::Random.AbstractRNG, layer::DLinOSS)
    # State is (2, N). Row 1 = Velocity, Row 2 = Position.
    (oscillator_state = zeros(Float32, 2, layer.state_dimensions),)
end

# 3. FORWARD PASS
function (layer::DLinOSS)(token_sequence::AbstractArray, params, state)

    # ---------------------------------------------------------
    # A. Handle Batch Dimensions
    # ---------------------------------------------------------
    is_batched = ndims(token_sequence) == 3

    # Reshape to (Features, Time, Batch) if needed
    standardized_input_tensor =
        is_batched ? token_sequence :
        reshape(token_sequence, size(token_sequence, 1), size(token_sequence, 2), 1)

    (number_of_features, number_of_timesteps, number_of_batches) =
        size(standardized_input_tensor)

    # ---------------------------------------------------------
    # B. Unpack Parameters
    # ---------------------------------------------------------
    (;
        log_time_step,
        log_stiffness_coefficients,
        log_damping_coefficients,
        input_projection,
        output_projection,
    ) = params

    stiffness_coefficients = exp.(log_stiffness_coefficients)
    time_step = exp.(log_time_step)
    damping_coefficients = exp.(log_damping_coefficients)

    # ---------------------------------------------------------
    # C. Physics Operators
    # ---------------------------------------------------------
    implicit_damping_factor = 1.0f0 ./ (1.0f0 .+ time_step .* damping_coefficients)
    velocity_retention_rate = implicit_damping_factor
    spring_stiffness_coupling =
        -time_step .* stiffness_coefficients .* implicit_damping_factor
    input_force_gain = time_step .* implicit_damping_factor

    # ---------------------------------------------------------
    # D. Input Projection (The Fold)
    # ---------------------------------------------------------
    input_flattened_time_batch = reshape(standardized_input_tensor, number_of_features, :)
    projected_input_flattened = input_projection * input_flattened_time_batch

    # Unfold to (State, Time, Batch)
    projected_input_3d_tensor = reshape(
        projected_input_flattened,
        layer.state_dimensions,
        number_of_timesteps,
        number_of_batches,
    )
    input_sequence_iterator = eachslice(projected_input_3d_tensor, dims = 2)

    # ---------------------------------------------------------
    # E. Execution (Stacked Matrix Scan)
    # ---------------------------------------------------------

    # Step function now takes a Single Matrix 'current_state' (2*N, Batch)
    # Top Half = Velocity, Bottom Half = Position
    evolve_state =
        (current_stacked_state, current_input_matrix) -> begin

            # Slicing is Zygote-safe
            previous_velocity = current_stacked_state[1:layer.state_dimensions, :]
            previous_position = current_stacked_state[(layer.state_dimensions+1):end, :]

            # Physics Update
            next_velocity =
                (velocity_retention_rate .* previous_velocity) .+
                (spring_stiffness_coupling .* previous_position) .+
                (input_force_gain .* current_input_matrix)

            next_position = previous_position .+ (time_step .* next_velocity)

            # Stack vertically: Velocity on Top, Position on Bottom
            return vcat(next_velocity, next_position)
        end

    # Initialize Batch State (Stacked)
    initial_velocity_view = @view state.oscillator_state[1, :]
    initial_position_view = @view state.oscillator_state[2, :]

    # Vertically stack initial vectors -> (2*N, )
    initial_stacked_vector = vcat(initial_velocity_view, initial_position_view)

    # Repeat across batch -> (2*N, Batch)
    initial_stacked_batch = repeat(initial_stacked_vector, 1, number_of_batches)

    # Run Scan (History is Vector of Matrices)
    state_history =
        accumulate(evolve_state, input_sequence_iterator; init = initial_stacked_batch)

    # ---------------------------------------------------------
    # F. Output Formatting
    # ---------------------------------------------------------

    # 1. Flatten History: Reduce Vector of Matrices -> Single Matrix (2N, Batch * Time)
    flattened_history = reduce(hcat, state_history)

    # 2. Extract ONLY Positions (Bottom Half)
    # We slice rows [N+1 : 2N, :]
    flattened_positions = flattened_history[(layer.state_dimensions+1):end, :]

    # 3. Project Output: (Out, State) * (State, Batch * Time) -> (Out, Batch * Time)
    output_flattened = output_projection * flattened_positions

    # 4. Reshape to (Out, Batch, Time) -- note the Batch-Major order from hcat
    output_batch_major = reshape(
        output_flattened,
        layer.output_dimensions,
        number_of_batches,
        number_of_timesteps,
    )

    # 5. Permute to (Out, Time, Batch)
    output_3d_tensor = permutedims(output_batch_major, (1, 3, 2))

    final_output = is_batched ? output_3d_tensor : dropdims(output_3d_tensor, dims = 3)

    # ---------------------------------------------------------
    # G. Prepare Next State
    # ---------------------------------------------------------
    last_stacked_state = state_history[end] # (2N, Batch)

    # Extract first batch item -> (2N,)
    last_stacked_vector = last_stacked_state[:, 1]

    # Reshape back to (2, N) to match Lux state format
    # Row 1: Velocity (1:N), Row 2: Position (N+1:end)
    # We construct it manually to avoid reshape mutation issues in Zygote
    next_velocity_row = transpose(last_stacked_vector[1:layer.state_dimensions])
    next_position_row = transpose(last_stacked_vector[(layer.state_dimensions+1):end])

    next_state_matrix = vcat(next_velocity_row, next_position_row)

    return (final_output, (oscillator_state = next_state_matrix,))
end

end # module
