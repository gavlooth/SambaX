module Driver


try
    include("src/Dlinoss.jl")
catch e
    @warn"You didn't load from project folder"
    @info "re-trying from file's folder"
    include("Dlinoss.jl")  #depends where we call the repl from
end

using Logging
using .Dlinoss
using Lux
using Random
using Optimisers
using Zygote
using LinearAlgebra
using Statistics


# ==============================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
const SEQUENCE_LENGTH = 50
const BATCH_SIZE      = 16
const STATE_DIM       = 64    # Number of oscillators
const HIDDEN_DIM      = 16    # Width of the sandwich layers
const LEARNING_RATE   = 0.005
const EPOCHS          = 500

# We will learn to predict a sine wave (Simple 1D input/output)
const INPUT_DIM       = 1
const OUTPUT_DIM      = 1

# ==============================================================================
# 2. DATA GENERATION
# ==============================================================================
"""
Generates a batch of sine waves with random phases and frequencies.
Shape: (1, Sequence_Length, Batch_Size)
"""
function generate_sine_data(rng, batch_size, seq_len)
    # Time axis
    t = range(0, 4π, length=seq_len)
    
    # Pre-allocate batch: (Feature, Time, Batch)
    data = zeros(Float32, 1, seq_len, batch_size)
    targets = zeros(Float32, 1, seq_len, batch_size)
    
    for b in 1:batch_size
        # Random frequency between 1.0 and 2.0
        freq = 1.0f0 + rand(rng, Float32) 
        phase = rand(rng, Float32) * 2π
        
        # Input: sin(ωt + φ)
        signal = sin.(freq .* t .+ phase)
        
        # Target: The same signal, shifted by 1 step (Next-token prediction)
        # We try to predict the immediate future
        data[1, :, b] = signal
        targets[1, 1:end-1, b] = signal[2:end]
        targets[1, end, b] = signal[1] # Wrap around (simplification)
    end
    
    return Float32.(data), Float32.(targets)
end

# ==============================================================================
# 3. MODEL ARCHITECTURE
# ==============================================================================
function build_model()
    return Chain(
        # 1. Encoder: Project 1D input -> 16 dimensions
        Dense(INPUT_DIM => HIDDEN_DIM),
        
        # 2. The D-LinOSS Core
        # Note: We must project Hidden -> Hidden (16 -> 16)
        Dlinoss.DLinOSS(
            HIDDEN_DIM,   # Input to this layer
            STATE_DIM,    # Internal oscillators (64)
            HIDDEN_DIM,   # Output from this layer
            0.1f0,        # Min Freq
            5.0f0,        # Max Freq
            0.05f0        # dt
        ),
        
        # 3. Non-Linearity
        Lux.gelu,
        
        # 4. Decoder: Project 16 dimensions -> 1D output
        Dense(HIDDEN_DIM => OUTPUT_DIM)
    )
end

# ==============================================================================
# 4. TRAINING LOOP
# ==============================================================================
function train()
    rng = Random.default_rng()
    Random.seed!(rng, 42)

    # --- Initialization ---
    model = build_model()
    ps, st = Lux.setup(rng, model)
    
    # Optimizer (AdamW is usually best for SSMs)
    opt_rule = Optimisers.Adam(LEARNING_RATE)
    opt_state = Optimisers.setup(opt_rule, ps)

    println(">>> Model Architecture:")
    println(model)
    println("\n>>> Starting Training on Sine Wave Task...")

    # --- Epoch Loop ---
    for epoch in 1:EPOCHS
        
        # Generate fresh data every epoch
        x_batch, y_true = generate_sine_data(rng, BATCH_SIZE, SEQUENCE_LENGTH)
        
        # Calculate Gradients
        # Zygote automatically handles the differentiation through your 'accumulate'
        loss, grads = Zygote.withgradient(ps) do params
            
            # Forward Pass
            y_pred, st_any = Lux.apply(model, x_batch, params, st)
            
            # MSE Loss
            diff = y_pred .- y_true
            return mean(abs2, diff)
        end

        # Update Parameters
        opt_state, ps = Optimisers.update(opt_state, ps, grads[1])

        # Logging
        if epoch % 50 == 0 || epoch == 1
            println("    Epoch $epoch | Loss: $(round(loss, digits=6))")
        end
    end

    println("\n✅ Training Complete!")
    return ps, st
end

# ==============================================================================
# 5. EXECUTION
# ==============================================================================
final_ps, final_st = train()
