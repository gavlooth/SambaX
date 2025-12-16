#!/usr/bin/env julia
"""
Extended training script with checkpoint saving.
Trains for multiple epochs and saves checkpoints that can be committed.
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

println("Loading packages...")
using Random
using Lux
using Zygote
using Optimisers
using Statistics: mean
using Serialization

include(joinpath(dirname(@__DIR__), "src", "Ossamma.jl"))
using .Ossamma

include(joinpath(dirname(@__DIR__), "src", "DataLoader.jl"))
using .DataLoader

# ============================================================================
# Configuration
# ============================================================================
const CONFIG = (
    # Model
    embedding_dim = 256,
    num_heads = 4,
    num_layers = 6,
    seq_length = 128,

    # Training
    batch_size = 32,
    num_epochs = 10,
    learning_rate = 3e-4,
    warmup_steps = 200,

    # Checkpointing
    checkpoint_every_epoch = 1,  # Save every N epochs
    checkpoint_dir = "checkpoints",

    # Logging
    log_every = 50,
)

# ============================================================================
# Setup
# ============================================================================
println("=" ^ 70)
println("OSSAMMA Extended Training")
println("=" ^ 70)
println()
println("Config:")
println("  Epochs: $(CONFIG.num_epochs)")
println("  Batch size: $(CONFIG.batch_size)")
println("  Sequence length: $(CONFIG.seq_length)")
println()

# Create checkpoint directory
mkpath(CONFIG.checkpoint_dir)

# Load data
println("[1/3] Loading Gutenberg dataset (all books)...")
rng = Random.MersenneTwister(42)

train_loader, val_loader, tokenizer = prepare_gutenberg(;
    books = :all,
    seq_length = CONFIG.seq_length,
    batch_size = CONFIG.batch_size,
    max_vocab_size = 10000,
    rng = rng,
)

vocab_size = get_vocab_size(tokenizer)
mask_token_id = get_mask_token_id(tokenizer) + 1
num_batches = length(train_loader)
total_steps = num_batches * CONFIG.num_epochs

println("  Vocab size: $vocab_size")
println("  Batches per epoch: $num_batches")
println("  Total steps: $total_steps")
println()

# Create model
println("[2/3] Creating model...")
model_config = LLaDAConfig(
    vocab_size = vocab_size,
    max_sequence_length = CONFIG.seq_length,
    embedding_dimension = CONFIG.embedding_dim,
    number_of_heads = CONFIG.num_heads,
    number_of_layers = CONFIG.num_layers,
    mask_token_id = mask_token_id,
    time_dimension = 64,
    state_dimension = CONFIG.embedding_dim,
    window_size = 16,
    mask_schedule = :cosine,
)

model = LLaDAModel(model_config)
ps, st = Lux.setup(rng, model)

# Count parameters
function count_params(p)
    total = 0
    for (_, v) in pairs(p)
        if v isa AbstractArray
            total += length(v)
        elseif v isa NamedTuple
            total += count_params(v)
        end
    end
    return total
end

println("  Parameters: $(round(count_params(ps) / 1e6, digits=2))M")
println()

# ============================================================================
# Checkpoint Functions
# ============================================================================
function save_checkpoint(epoch, params, state, optimizer_state, loss, path)
    checkpoint = Dict(
        :epoch => epoch,
        :params => params,
        :state => state,
        :optimizer_state => optimizer_state,
        :loss => loss,
        :timestamp => now_string(),
    )
    serialize(path, checkpoint)
    println("  Checkpoint saved: $path")
end

function load_checkpoint(path)
    return deserialize(path)
end

function now_string()
    # Simple timestamp without Dates package
    return string(time())
end

# Save tokenizer vocab for later use
function save_tokenizer(tokenizer, path)
    vocab_data = Dict(
        :vocab => tokenizer.vocab,
        :inverse_vocab => tokenizer.inverse_vocab,
        :special_tokens => tokenizer.special_tokens,
        :vocab_size => tokenizer.vocab_size,
    )
    serialize(path, vocab_data)
end

save_tokenizer(tokenizer, joinpath(CONFIG.checkpoint_dir, "tokenizer.jls"))
println("Tokenizer saved to $(CONFIG.checkpoint_dir)/tokenizer.jls")

# ============================================================================
# Training Loop
# ============================================================================
println("\n[3/3] Starting training...")
println("=" ^ 70)

optimizer = Optimisers.Adam(Float32(CONFIG.learning_rate))
opt_state = Optimisers.setup(optimizer, ps)

global_step = 0
best_loss = Inf

for epoch in 1:CONFIG.num_epochs
    println("\n--- Epoch $epoch / $(CONFIG.num_epochs) ---")
    epoch_loss = 0.0
    num_steps = 0

    # Reset data loader
    reset!(train_loader)

    for (batch_idx, batch) in enumerate(train_loader)
        global_step += 1

        # Learning rate schedule
        lr = if global_step < CONFIG.warmup_steps
            Float32(CONFIG.learning_rate) * Float32(global_step) / Float32(CONFIG.warmup_steps)
        else
            Float32(CONFIG.learning_rate)
        end
        Optimisers.adjust!(opt_state, lr)

        # Sample mask ratio
        mask_ratio = sample_mask_ratio(:cosine; rng=rng)

        # Forward and backward pass
        (loss, st), grads = Zygote.withgradient(ps) do p
            inputs = (token_ids = batch, mask_ratio = mask_ratio)
            logits, new_st = model(inputs, p, st)

            # Compute loss on masked positions
            l = diffusion_loss(logits, batch, mask_ratio, mask_token_id)
            (l, new_st)
        end

        # Update parameters
        opt_state, ps = Optimisers.update(opt_state, ps, grads[1])

        epoch_loss += loss
        num_steps += 1

        # Logging
        if global_step % CONFIG.log_every == 0
            avg_loss = epoch_loss / num_steps
            println("  Step $global_step | Batch $batch_idx/$num_batches | Loss: $(round(loss, digits=4)) | Avg: $(round(avg_loss, digits=4)) | LR: $(round(lr, sigdigits=3))")
        end
    end

    # Epoch summary
    avg_epoch_loss = epoch_loss / num_steps
    println("\nEpoch $epoch complete | Avg Loss: $(round(avg_epoch_loss, digits=4))")

    # Validation
    println("Running validation...")
    reset!(val_loader)
    val_loss = 0.0
    val_steps = 0
    for batch in val_loader
        mask_ratio = 0.5f0  # Fixed for validation
        inputs = (token_ids = batch, mask_ratio = mask_ratio)
        logits, _ = model(inputs, ps, st)
        loss = diffusion_loss(logits, batch, mask_ratio, mask_token_id)
        val_loss += loss
        val_steps += 1
    end
    avg_val_loss = val_loss / val_steps
    println("  Validation Loss: $(round(avg_val_loss, digits=4))")

    # Save checkpoint
    if epoch % CONFIG.checkpoint_every_epoch == 0
        checkpoint_path = joinpath(CONFIG.checkpoint_dir, "checkpoint_epoch_$(epoch).jls")
        save_checkpoint(epoch, ps, st, opt_state, avg_val_loss, checkpoint_path)

        if avg_val_loss < best_loss
            best_loss = avg_val_loss
            best_path = joinpath(CONFIG.checkpoint_dir, "checkpoint_best.jls")
            save_checkpoint(epoch, ps, st, opt_state, avg_val_loss, best_path)
            println("  New best model!")
        end
    end

    # Generate sample
    println("\nGenerating sample text...")
    try
        generated_ids = generate(model, ps, st, 64; num_steps=15, batch_size=1, rng=rng)
        ids = vec(generated_ids) .- 1
        sample_text = decode(tokenizer, ids)
        println("  \"$(sample_text[1:min(100, length(sample_text))])...\"")
    catch e
        println("  Generation failed: $e")
    end
end

println("\n" * "=" ^ 70)
println("Training Complete!")
println("  Total steps: $global_step")
println("  Best validation loss: $(round(best_loss, digits=4))")
println("  Checkpoints saved to: $(CONFIG.checkpoint_dir)/")
println("=" ^ 70)

# Final generation samples
println("\n--- Final Generation Samples ---")
for i in 1:3
    println("\nSample $i:")
    generated_ids = generate(model, ps, st, 128; num_steps=25, batch_size=1, rng=Random.MersenneTwister(i))
    ids = vec(generated_ids) .- 1
    sample_text = decode(tokenizer, ids)
    println(sample_text)
end
