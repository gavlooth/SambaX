#!/usr/bin/env julia
"""
Example training script for LLaDA text diffusion model.

Usage:
    julia --project=. examples/train_llada.jl [config_path]

If no config_path is provided, uses the small config for testing.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using Optimisers

# Load our module
include(joinpath(@__DIR__, "..", "src", "Ossamma.jl"))
using .Ossamma

# ============================================================================
# Synthetic Data Generator (for demonstration)
# ============================================================================

"""
Simple synthetic data: random token sequences.
In practice, replace with real tokenized text data.
"""
struct SyntheticDataLoader
    vocab_size::Int
    seq_length::Int
    batch_size::Int
    num_batches::Int
    rng::Random.AbstractRNG
end

function SyntheticDataLoader(;
    vocab_size::Int = 1000,
    seq_length::Int = 64,
    batch_size::Int = 8,
    num_batches::Int = 100,
    seed::Int = 42,
)
    return SyntheticDataLoader(vocab_size, seq_length, batch_size, num_batches, Random.MersenneTwister(seed))
end

Base.length(d::SyntheticDataLoader) = d.num_batches

function Base.iterate(d::SyntheticDataLoader, state=1)
    if state > d.num_batches
        return nothing
    end
    # Generate random token IDs (seq_length, batch_size)
    batch = rand(d.rng, 1:d.vocab_size, d.seq_length, d.batch_size)
    return (batch, state + 1)
end

# ============================================================================
# Main Training Script
# ============================================================================

function main(config_path::Union{String, Nothing} = nothing)
    println("=" ^ 60)
    println("LLaDA Text Diffusion Training")
    println("=" ^ 60)
    println()

    # Load configuration
    if config_path !== nothing && isfile(config_path)
        println("Loading config from: $config_path")
        config = load_config(config_path)
    else
        println("Using small config for testing...")
        config = small_config()
    end

    println("\nModel Configuration:")
    println("  vocab_size: $(config.vocab_size)")
    println("  max_sequence_length: $(config.max_sequence_length)")
    println("  embedding_dimension: $(config.embedding_dimension)")
    println("  number_of_heads: $(config.number_of_heads)")
    println("  number_of_layers: $(config.number_of_layers)")
    println()

    # Create model
    println("Building model...")
    model = LLaDAModel(config)
    println("  Done!")
    println()

    # Setup RNG
    rng = Random.default_rng()
    Random.seed!(rng, 42)

    # Initialize training state
    println("Initializing training state...")
    optimizer = Optimisers.Adam(1e-4f0)
    train_state = create_train_state(model, optimizer; rng=rng)
    println("  Done!")
    println()

    # Create data loaders
    println("Creating data loaders...")
    train_data = SyntheticDataLoader(
        vocab_size = config.vocab_size,
        seq_length = config.max_sequence_length,
        batch_size = 8,
        num_batches = 1000,  # Total batches for training
        seed = 42,
    )

    val_data = SyntheticDataLoader(
        vocab_size = config.vocab_size,
        seq_length = config.max_sequence_length,
        batch_size = 8,
        num_batches = 10,
        seed = 123,
    )
    println("  Train batches: $(length(train_data))")
    println("  Val batches: $(length(val_data))")
    println()

    # Training configuration
    train_config = TrainingConfig(
        batch_size = 8,
        learning_rate = 1e-4f0,
        min_learning_rate = 1e-6f0,
        warmup_steps = 100,
        total_steps = 500,  # Short for demo
        eval_every = 100,
        log_every = 50,
        save_every = 200,
        mask_schedule = config.mask_schedule,
    )

    # Callbacks
    callbacks = Dict{Symbol, Function}(
        :on_best => (state) -> println("  Saving best model checkpoint..."),
        :on_save => (state) -> println("  Saving checkpoint at step $(state.step)..."),
    )

    # Train!
    println("Starting training loop...")
    println("-" ^ 60)

    train!(
        model,
        train_state,
        train_data,
        train_config;
        val_data = val_data,
        callbacks = callbacks,
        rng = rng,
    )

    println()
    println("=" ^ 60)
    println("Training complete!")
    println("=" ^ 60)

    # Test generation
    println("\nTesting generation...")
    params, state = train_state.params, train_state.state

    generated = generate(
        model, params, state, 32;  # Generate 32 tokens
        num_steps = 10,
        batch_size = 1,
        rng = rng,
    )

    println("Generated tokens: ", generated[1:min(20, length(generated))], "...")
    println()

    return train_state
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    config_path = length(ARGS) > 0 ? ARGS[1] : nothing
    main(config_path)
end
