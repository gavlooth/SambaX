#!/usr/bin/env julia
"""
Quickstart example for LLaDA text diffusion model.

Demonstrates:
1. Loading a model from config
2. Forward pass
3. Masking and unmasking
4. Generation
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using Lux

# Load module
include(joinpath(@__DIR__, "..", "src", "Ossamma.jl"))
using .Ossamma

println("LLaDA Quickstart Example")
println("=" ^ 40)

# ============================================================================
# 1. Create Model from Config
# ============================================================================
println("\n1. Creating model...")

# Option A: From preset config
config = small_config()

# Option B: From TOML file
# config = load_config("configs/small.toml")

# Option C: Custom config
# config = LLaDAConfig(
#     vocab_size = 500,
#     max_sequence_length = 32,
#     embedding_dimension = 64,
#     number_of_heads = 2,
#     number_of_layers = 2,
# )

model = LLaDAModel(config)
println("   Model created with $(config.number_of_layers) layers")

# Initialize parameters
rng = Random.default_rng()
Random.seed!(rng, 42)
params, state = Lux.setup(rng, model)
println("   Parameters initialized")

# ============================================================================
# 2. Forward Pass
# ============================================================================
println("\n2. Forward pass...")

# Create dummy input
seq_len = 32
batch_size = 2
token_ids = rand(rng, 1:config.vocab_size, seq_len, batch_size)
mask_ratio = 0.5f0  # 50% masked

println("   Input shape: (seq=$seq_len, batch=$batch_size)")
println("   Mask ratio: $mask_ratio")

# Apply masking
mask_token_id = config.vocab_size  # Last token is [MASK]
masked_ids, mask = apply_mask(token_ids, mask_ratio, mask_token_id; rng=rng)
println("   Tokens masked: $(sum(mask)) / $(length(mask))")

# Forward pass
inputs = (token_ids = masked_ids, mask_ratio = mask_ratio)
logits, new_state = model(inputs, params, state)
println("   Output logits shape: $(size(logits))")

# ============================================================================
# 3. Compute Loss (for training)
# ============================================================================
println("\n3. Computing loss...")

loss, _ = diffusion_loss(model, params, state, token_ids, mask_token_id; rng=rng)
println("   Diffusion loss: $(round(loss, digits=4))")

# ============================================================================
# 4. Generation (Iterative Denoising)
# ============================================================================
println("\n4. Generation...")

generated = generate(
    model, params, state, 16;  # Generate 16 tokens
    num_steps = 5,
    batch_size = 1,
    rng = rng,
)

println("   Generated $(length(generated)) tokens:")
println("   $generated")

# ============================================================================
# 5. Step-by-step Unmasking (Manual Control)
# ============================================================================
println("\n5. Manual unmasking demo...")

# Start fully masked
current_ids = fill(mask_token_id, 8, 1)
current_mask = trues(8, 1)
println("   Start: $([current_ids[i, 1] == mask_token_id ? "[M]" : string(current_ids[i, 1]) for i in 1:8])")

# Unmask in steps
for step in 1:4
    t = 1.0f0 - (step - 1) / 4
    inputs = (token_ids = current_ids, mask_ratio = t)
    logits, state = model(inputs, params, state)

    # Unmask 2 tokens per step
    current_ids, current_mask = unmask_step(logits, current_ids, current_mask, 2, mask_token_id)

    display_ids = [current_ids[i, 1] == mask_token_id ? "[M]" : string(current_ids[i, 1]) for i in 1:8]
    println("   Step $step: $display_ids")
end

# ============================================================================
# 6. Save/Load Config
# ============================================================================
println("\n6. Config save/load...")

# Save config to file
config_path = joinpath(@__DIR__, "..", "configs", "test_output.toml")
save_config(config, config_path)
println("   Saved config to: $config_path")

# Load it back
loaded_config = load_config(config_path)
println("   Loaded config: vocab=$(loaded_config.vocab_size), layers=$(loaded_config.number_of_layers)")

# Cleanup
rm(config_path)

println("\n" * "=" ^ 40)
println("Quickstart complete!")
