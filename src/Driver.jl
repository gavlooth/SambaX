Module Driver


try
    include("src/Dlinoss.jl")
catch e
    @warn"You didn't load from project folder"
    @warn "Loading local module directly..."
    include("Dlinoss.jl")
end


using .Dlinoss
using Lux
using Random
using Optimisers
using Zygote
using LinearAlgebra
using Statistics

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
const SEQUENCE_LENGTH = 50
const BATCH_SIZE = 16
const STATE_DIMENSION = 64
const HIDDEN_DIMENSION = 16
const LEARNING_RATE = 0.005
const EPOCHS = 200 # Reduced slightly for quick testing

const INPUT_DIMENSION = 1
const OUTPUT_DIMENSION = 1

# ==============================================================================
# 2. DATA GENERATION
# ==============================================================================
function generate_sine_data(rng, batch_size, sequence_length)
    time_values = range(0, 4π, length = sequence_length)
    data = zeros(Float32, 1, sequence_length, batch_size)
    targets = zeros(Float32, 1, sequence_length, batch_size)

    for batch_index = 1:batch_size
        frequency = 1.0f0 + rand(rng, Float32)
        phase = rand(rng, Float32) * 2π
        signal = sin.(frequency .* time_values .+ phase)

        data[1, :, batch_index] = signal
        targets[1, 1:end-1, batch_index] = signal[2:end]
        targets[1, end, batch_index] = signal[1]
    end
    return Float32.(data), Float32.(targets)
end

# ==============================================================================
# 3. MODEL
# ==============================================================================
function build_model()
    return Chain(
        Dense(INPUT_DIMENSION => HIDDEN_DIMENSION),
        Dlinoss.DLinOSS(HIDDEN_DIMENSION, STATE_DIMENSION, HIDDEN_DIMENSION, 0.1f0, 5.0f0, 0.05f0),
        Lux.gelu,
        Dense(HIDDEN_DIMENSION => OUTPUT_DIMENSION),
    )
end

# ==============================================================================
# 4. TRAINING
# ==============================================================================
function train()
    rng = Random.default_rng()
    Random.seed!(rng, 42)

    model = build_model()
    parameters, state = Lux.setup(rng, model)

    optimizer = Optimisers.Adam(LEARNING_RATE)
    optimizer_state = Optimisers.setup(optimizer, parameters)

    println(">>> Model built. Layers: $(length(model))")
    println(">>> Starting Training...")

    for epoch = 1:EPOCHS
        input_batch, target_output = generate_sine_data(rng, BATCH_SIZE, SEQUENCE_LENGTH)

        loss, gradients = Zygote.withgradient(parameters) do params
            predicted_output, _ = Lux.apply(model, input_batch, params, state)
            mean(abs2, predicted_output .- target_output)
        end

        optimizer_state, parameters = Optimisers.update(optimizer_state, parameters, gradients[1])

        if epoch % 20 == 0 || epoch == 1
            println("    Epoch $epoch | Loss: $(round(loss, digits=6))")
        end
    end

    println("\n✅ Training Complete!")
end

train()

end
