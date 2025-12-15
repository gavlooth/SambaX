#!/usr/bin/env julia
"""
Colab-ready training script for LLaDA model.

Run in Google Colab:
```
# Install Julia
!wget -q https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.2-linux-x86_64.tar.gz
!tar -xzf julia-1.10.2-linux-x86_64.tar.gz
!mv julia-1.10.2 /usr/local/julia
import os
os.environ['PATH'] = '/usr/local/julia/bin:' + os.environ['PATH']

# Clone repo and run
!git clone https://github.com/gavlooth/Ossamma.git
!cd Ossamma && julia --project=. -e 'using Pkg; Pkg.instantiate()'
!cd Ossamma && julia --project=. scripts/train_colab.jl
```
"""

using Pkg
Pkg.activate(@__DIR__ * "/..")
Pkg.instantiate()

# ============================================================================
# Configuration - Edit these for your run!
# ============================================================================

const CONFIG = (
    # Dataset
    dataset = :tinystories,  # :tinystories, :wikitext, or :custom
    custom_dataset_name = "",  # Only if dataset = :custom
    num_train_rows = 50000,
    num_val_rows = 5000,

    # Model
    model_size = :small,  # :small, :base, or :large
    vocab_size = 10000,   # Will be adjusted based on tokenizer

    # Training
    seq_length = 128,
    batch_size = 16,
    learning_rate = 1e-4f0,
    warmup_steps = 500,
    total_steps = 5000,
    eval_every = 500,
    log_every = 50,

    # Output
    save_checkpoints = true,
    checkpoint_dir = "checkpoints",
)

# ============================================================================
# Setup
# ============================================================================

println("=" ^ 60)
println("LLaDA Training Script")
println("=" ^ 60)
println()

# Load modules
include(joinpath(@__DIR__, "..", "src", "DataLoader.jl"))
include(joinpath(@__DIR__, "..", "src", "LLaDA.jl"))
include(joinpath(@__DIR__, "..", "src", "Training.jl"))

using .DataLoader
using .LLaDA
using .Training
using Random
using Optimisers

# Set random seed for reproducibility
rng = Random.MersenneTwister(42)

# ============================================================================
# Load Data
# ============================================================================

println("\n[1/4] Loading dataset...")

train_loader, val_loader, tokenizer = if CONFIG.dataset == :tinystories
    prepare_tinystories(;
        num_train_rows = CONFIG.num_train_rows,
        num_val_rows = CONFIG.num_val_rows,
        seq_length = CONFIG.seq_length,
        batch_size = CONFIG.batch_size,
        max_vocab_size = CONFIG.vocab_size,
        rng = rng,
    )
elseif CONFIG.dataset == :wikitext
    prepare_wikitext(;
        num_train_rows = CONFIG.num_train_rows,
        num_val_rows = CONFIG.num_val_rows,
        seq_length = CONFIG.seq_length,
        batch_size = CONFIG.batch_size,
        max_vocab_size = CONFIG.vocab_size,
        rng = rng,
    )
else
    prepare_custom_dataset(
        CONFIG.custom_dataset_name;
        num_train_rows = CONFIG.num_train_rows,
        num_val_rows = CONFIG.num_val_rows,
        seq_length = CONFIG.seq_length,
        batch_size = CONFIG.batch_size,
        max_vocab_size = CONFIG.vocab_size,
        rng = rng,
    )
end

actual_vocab_size = get_vocab_size(tokenizer)
mask_token_id = get_mask_token_id(tokenizer) + 1  # +1 for Julia 1-indexing

println("\nDataset loaded!")
println("  Vocab size: $actual_vocab_size")
println("  Mask token ID: $mask_token_id")
println("  Train batches: $(length(train_loader))")
println("  Val batches: $(length(val_loader))")

# ============================================================================
# Create Model
# ============================================================================

println("\n[2/4] Creating model...")

# Get base config and override vocab size
base_config = if CONFIG.model_size == :small
    small_config()
elseif CONFIG.model_size == :base
    base_config()
else
    large_config()
end

# Update config with actual vocab size
model_config = LLaDAConfig(
    vocab_size = actual_vocab_size,
    max_sequence_length = CONFIG.seq_length,
    embedding_dimension = base_config.embedding_dimension,
    number_of_heads = base_config.number_of_heads,
    number_of_layers = base_config.number_of_layers,
    time_embedding_dimension = base_config.time_embedding_dimension,
    state_dimension = base_config.state_dimension,
    mask_token_id = mask_token_id,
)

model = LLaDAModel(model_config)

# Count parameters
params, state = Lux.setup(rng, model)
param_count = sum(length, Lux.parameterlength(model))
println("Model created!")
println("  Size: $(CONFIG.model_size)")
println("  Embedding dim: $(model_config.embedding_dimension)")
println("  Layers: $(model_config.number_of_layers)")
println("  Heads: $(model_config.number_of_heads)")

# ============================================================================
# Setup Training
# ============================================================================

println("\n[3/4] Setting up training...")

optimizer = Optimisers.Adam(CONFIG.learning_rate)
train_state = create_train_state(model, optimizer; rng=rng)

train_config = TrainingConfig(
    batch_size = CONFIG.batch_size,
    learning_rate = CONFIG.learning_rate,
    warmup_steps = CONFIG.warmup_steps,
    total_steps = CONFIG.total_steps,
    eval_every = CONFIG.eval_every,
    log_every = CONFIG.log_every,
    mask_schedule = :cosine,
)

# Setup callbacks
callbacks = Dict{Symbol, Function}()

if CONFIG.save_checkpoints
    mkpath(CONFIG.checkpoint_dir)
    callbacks[:on_save] = function(ts)
        # Save checkpoint (basic - just print for now)
        println("  [Checkpoint] Step $(ts.step), Loss $(round(ts.best_loss, digits=4))")
    end
end

println("Training config:")
println("  Steps: $(train_config.total_steps)")
println("  Batch size: $(train_config.batch_size)")
println("  Learning rate: $(train_config.learning_rate)")
println("  Warmup steps: $(train_config.warmup_steps)")

# ============================================================================
# Train!
# ============================================================================

println("\n[4/4] Starting training...")
println("=" ^ 60)

train!(
    model,
    train_state,
    train_loader,
    train_config;
    val_data = val_loader,
    callbacks = callbacks,
    rng = rng,
)

println("\n" * "=" ^ 60)
println("Training complete!")
println("=" ^ 60)

# ============================================================================
# Generate Sample
# ============================================================================

println("\n[Bonus] Generating sample text...")

try
    generated_ids = generate(
        model,
        train_state.params,
        train_state.state,
        CONFIG.seq_length;
        num_steps = 10,
        batch_size = 1,
        rng = rng,
    )

    # Decode (subtract 1 for 0-indexed tokenizer)
    sample_ids = vec(generated_ids) .- 1
    sample_text = decode(tokenizer, sample_ids)
    println("\nGenerated text:")
    println("-" ^ 40)
    println(sample_text[1:min(500, length(sample_text))])
    println("-" ^ 40)
catch e
    println("Generation failed: $e")
end

println("\nDone!")
