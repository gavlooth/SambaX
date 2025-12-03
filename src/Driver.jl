module Driver


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
const STATE_DIM = 64
const HIDDEN_DIM = 16
const LEARNING_RATE = 0.005
const EPOCHS = 200 # Reduced slightly for quick testing

const INPUT_DIM = 1
const OUTPUT_DIM = 1

# ==============================================================================
# 2. DATA GENERATION
# ==============================================================================
function generate_sine_data(rng, batch_size, seq_len)
    t = range(0, 4π, length = seq_len)
    data = zeros(Float32, 1, seq_len, batch_size)
    targets = zeros(Float32, 1, seq_len, batch_size)

    for b = 1:batch_size
        freq = 1.0f0 + rand(rng, Float32)
        phase = rand(rng, Float32) * 2π
        signal = sin.(freq .* t .+ phase)

        data[1, :, b] = signal
        targets[1, 1:end-1, b] = signal[2:end]
        targets[1, end, b] = signal[1]
    end
    return Float32.(data), Float32.(targets)
end

# ==============================================================================
# 3. MODEL
# ==============================================================================
function build_model()
    return Chain(
        Dense(INPUT_DIM => HIDDEN_DIM),
        Dlinoss.DLinOSS(HIDDEN_DIM, STATE_DIM, HIDDEN_DIM, 0.1f0, 5.0f0, 0.05f0),
        Lux.gelu,
        Dense(HIDDEN_DIM => OUTPUT_DIM),
    )
end

# ==============================================================================
# 4. TRAINING
# ==============================================================================
function train()
    rng = Random.default_rng()
    Random.seed!(rng, 42)

    model = build_model()
    ps, st = Lux.setup(rng, model)

    opt = Optimisers.Adam(LEARNING_RATE)
    opt_state = Optimisers.setup(opt, ps)

    println(">>> Model built. Layers: $(length(model))")
    println(">>> Starting Training...")

    for epoch = 1:EPOCHS
        x_batch, y_true = generate_sine_data(rng, BATCH_SIZE, SEQUENCE_LENGTH)

        loss, grads = Zygote.withgradient(ps) do params
            y_pred, _ = Lux.apply(model, x_batch, params, st)
            mean(abs2, y_pred .- y_true)
        end

        opt_state, ps = Optimisers.update(opt_state, ps, grads[1])

        if epoch % 20 == 0 || epoch == 1
            println("    Epoch $epoch | Loss: $(round(loss, digits=6))")
        end
    end

    println("\n✅ Training Complete!")
end

train()

end # module
