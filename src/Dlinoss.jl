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

    # A (Stiffness/Frequency)
    log_stiffness_coefficients = collect(
        range(
            log(layer.min_frequency),
            log(layer.max_frequency),
            length = layer.state_dimensions,
        ),
    )

    # dt (Time Step)
    log_time_step = ones(Float32, layer.state_dimensions) .* log(layer.dt)

    # G (Damping)
    log_damping_coefficients = ones(Float32, layer.state_dimensions) .* log(0.01f0)

    # B (Input -> State)
    input_projection =
        randn(rng, Float32, layer.state_dimensions, layer.input_dimensions) .* 0.02f0

    # C (State -> Output)
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
    # Default State: (2, N) for a single item. 
    # This will be broadcasted to the batch size dynamically in the forward pass.
    (oscillator_state = zeros(Float32, 2, layer.state_dimensions),)
end

# 3. FORWARD PASS
function (layer::DLinOSS)(token_sequence::AbstractArray, params, state)

    # ---------------------------------------------------------
    # A. Handle Batch Dimensions
    # ---------------------------------------------------------
    # We standardize input to 3D: (Features, Time, Batch)
    is_batched = ndims(token_sequence) == 3

    # If not batched, reshape to (F, T, 1) for consistent math
    input_3d =
        is_batched ? token_sequence :
        reshape(token_sequence, size(token_sequence, 1), size(token_sequence, 2), 1)

    (number_of_features, number_of_timesteps, number_of_batches) = size(input_3d)

    # ---------------------------------------------------------
    # B. Unpack and constrain parameters
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
    # C. Pre-calculate IMEX Physics Operators
    # ---------------------------------------------------------

    # The term (1 + dt * G)^-1
    implicit_damping_factor = 1.0f0 ./ (1.0f0 .+ time_step .* damping_coefficients)

    # How much velocity survives the friction per step
    velocity_retention_rate = implicit_damping_factor

    # How much the spring pulls the velocity back (Negative Feedback)
    spring_stiffness_coupling =
        -time_step .* stiffness_coefficients .* implicit_damping_factor

    # How much the input signal adds to the velocity
    input_force_gain = time_step .* implicit_damping_factor

    # ---------------------------------------------------------
    # D. Execution (The Scan)
    # ---------------------------------------------------------

    # 1. Project Input: Fold Batch into Time to use Matrix Multiplication
    # Reshape (Features, Time, Batch) -> (Features, Time * Batch)
    input_flattened = reshape(input_3d, number_of_features, :)

    # Matrix Multiply: (State, Features) * (Features, Time * Batch) -> (State, Time * Batch)
    projected_input_flattened = input_projection * input_flattened

    # Unfold back to 3D: (State, Time, Batch)
    projected_input_3d = reshape(
        projected_input_flattened,
        layer.state_dimensions,
        number_of_timesteps,
        number_of_batches,
    )

    # Create Iterator: Each slice is a (State, Batch) matrix for a single time step
    input_sequence_iterator = eachslice(projected_input_3d, dims = 2)

    # 2. Define the Single-Step Physics Logic
    evolve_state =
        (current_state_tuple, current_input_matrix) -> begin

            # Unpack tuple of (N, B) matrices
            previous_velocity, previous_position = current_state_tuple

            # Physics Update (Broadcasting (N,) vs (N, B) happens automatically)
            next_velocity =
                (velocity_retention_rate .* previous_velocity) .+
                (spring_stiffness_coupling .* previous_position) .+
                (input_force_gain .* current_input_matrix)

            # Symplectic Position Update (Use next_velocity for stability)
            next_position = previous_position .+ (time_step .* next_velocity)

            return (next_velocity, next_position)
        end

    # 3. Initialize Batch State
    # We must repeat the initial state vector to match the batch size
    initial_velocity_vector = @view state.oscillator_state[1, :]
    initial_position_vector = @view state.oscillator_state[2, :]

    # Repeat: (N,) -> (N, Batch)
    initial_velocity_batch = repeat(initial_velocity_vector, 1, number_of_batches)
    initial_position_batch = repeat(initial_position_vector, 1, number_of_batches)

    initial_state_tuple = (initial_velocity_batch, initial_position_batch)

    # 4. Run Scan
    # Returns Vector of ((N, B), (N, B)) tuples
    state_history =
        accumulate(evolve_state, input_sequence_iterator; init = initial_state_tuple)

    # ---------------------------------------------------------
    # E. Output Formatting
    # ---------------------------------------------------------

    # Extract positions: Vector of (N, B)
    position_history = map(last, state_history)

    # Stack into 3D Tensor: (N, Time, Batch)
    hidden_states_3d = stack(position_history, dims = 2)

    # Fold: Reshape for Output Projection: (N, Time, Batch) -> (N, Time * Batch)
    hidden_states_flattened = reshape(hidden_states_3d, layer.state_dimensions, :)

    # Project: (Out, N) * (N, Time * Batch) -> (Out, Time * Batch)
    output_flattened = output_projection * hidden_states_flattened

    # Unfold: Reshape back: (Out, Time * Batch) -> (Out, Time, Batch)
    output_3d = reshape(
        output_flattened,
        layer.output_dimensions,
        number_of_timesteps,
        number_of_batches,
    )

    # If input wasn't batched, drop the dummy batch dimension to be polite
    final_output = is_batched ? output_3d : dropdims(output_3d, dims = 3)

    # ---------------------------------------------------------
    # F. Prepare Next State
    # ---------------------------------------------------------
    # Extract the last state (N, B) from history
    (last_velocity_batch, last_position_batch) = state_history[end]

    # For Lux state carry-over, we take the first batch item to keep shape (2, N)
    next_state_matrix =
        stack((last_velocity_batch[:, 1], last_position_batch[:, 1]), dims = 1)

    return (final_output, (oscillator_state = next_state_matrix,))
end

end # module
