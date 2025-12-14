module Training

"""
Training utilities for LLaDA text diffusion model.

Includes:
- Loss functions (cross-entropy on masked positions)
- Training step
- Evaluation utilities
- Learning rate schedules
"""

using Lux
using Random
using NNlib
using Optimisers
using Zygote
using Statistics: mean
using TOML

using ..LLaDA: LLaDAModel, LLaDAConfig, apply_mask, sample_mask_ratio

# ============================================================================
# Loss Functions
# ============================================================================

"""
    masked_cross_entropy(logits, targets, mask)

Compute cross-entropy loss only on masked positions.

Arguments:
- logits: (vocab_size, seq_len, batch)
- targets: (seq_len, batch) - original token IDs
- mask: (seq_len, batch) - Bool, true where masked
"""
function masked_cross_entropy(logits, targets, mask)
    vocab_size = size(logits, 1)

    # Flatten for easier indexing
    # logits: (vocab, seq*batch), targets: (seq*batch,), mask: (seq*batch,)
    logits_flat = reshape(logits, vocab_size, :)
    targets_flat = vec(targets)
    mask_flat = vec(mask)

    # Log softmax for numerical stability
    log_probs = NNlib.logsoftmax(logits_flat, dims=1)

    # Gather log probs at target indices
    # For each position i, get log_probs[targets[i], i]
    n_positions = length(targets_flat)
    target_log_probs = [log_probs[targets_flat[i], i] for i in 1:n_positions]
    target_log_probs = reshape(target_log_probs, size(mask))

    # Only count masked positions
    n_masked = sum(mask_flat)
    if n_masked == 0
        return 0.0f0
    end

    # Negative log likelihood on masked positions
    loss = -sum(target_log_probs .* mask) / n_masked
    return loss
end

"""
    masked_cross_entropy_vectorized(logits, targets, mask)

Vectorized version of masked cross-entropy (more Zygote-friendly).
"""
function masked_cross_entropy_vectorized(logits, targets, mask)
    vocab_size, seq_len, batch_size = size(logits)

    # Log softmax
    log_probs = NNlib.logsoftmax(logits, dims=1)

    # One-hot encode targets: (vocab, seq, batch)
    targets_onehot = zeros(Float32, vocab_size, seq_len, batch_size)
    for b in 1:batch_size
        for s in 1:seq_len
            targets_onehot[targets[s, b], s, b] = 1.0f0
        end
    end

    # Element-wise multiply and sum over vocab dimension
    # Result: (seq, batch)
    target_log_probs = dropdims(sum(log_probs .* targets_onehot, dims=1), dims=1)

    # Mask and average
    n_masked = sum(mask)
    if n_masked == 0
        return 0.0f0
    end

    loss = -sum(target_log_probs .* mask) / n_masked
    return loss
end

"""
    diffusion_loss(model, params, state, token_ids, mask_token_id; rng, schedule)

Compute diffusion training loss:
1. Sample mask ratio t
2. Mask tokens according to t
3. Predict masked tokens
4. Compute cross-entropy on masked positions
"""
function diffusion_loss(
    model::LLaDAModel,
    params,
    state,
    token_ids::AbstractArray,
    mask_token_id::Int;
    rng::Random.AbstractRNG = Random.default_rng(),
    schedule::Symbol = :uniform,
)
    # Sample mask ratio
    t = sample_mask_ratio(rng; schedule=schedule)

    # Apply masking
    masked_ids, mask = apply_mask(token_ids, t, mask_token_id; rng=rng)

    # Forward pass
    inputs = (token_ids = masked_ids, mask_ratio = t)
    logits, new_state = model(inputs, params, state)

    # Compute loss on masked positions only
    loss = masked_cross_entropy_vectorized(logits, token_ids, mask)

    return loss, new_state
end

# ============================================================================
# Training Step
# ============================================================================

"""
    TrainState

Holds all state needed for training.
"""
mutable struct TrainState
    params::Any
    state::Any
    optimizer_state::Any
    step::Int
    epoch::Int
    best_loss::Float32
end

"""
    create_train_state(model, optimizer; rng)

Initialize training state.
"""
function create_train_state(
    model::LLaDAModel,
    optimizer;
    rng::Random.AbstractRNG = Random.default_rng(),
)
    params, state = Lux.setup(rng, model)
    optimizer_state = Optimisers.setup(optimizer, params)

    return TrainState(params, state, optimizer_state, 0, 0, Inf32)
end

"""
    train_step!(train_state, model, batch, mask_token_id; rng, schedule)

Perform a single training step. Updates train_state in place.
Returns the loss value.
"""
function train_step!(
    train_state::TrainState,
    model::LLaDAModel,
    batch::AbstractArray,
    mask_token_id::Int;
    rng::Random.AbstractRNG = Random.default_rng(),
    schedule::Symbol = :uniform,
)
    loss, grads = Zygote.withgradient(train_state.params) do params
        l, _ = diffusion_loss(model, params, train_state.state, batch, mask_token_id; rng=rng, schedule=schedule)
        l
    end

    # Update parameters
    train_state.optimizer_state, train_state.params = Optimisers.update(
        train_state.optimizer_state, train_state.params, grads[1]
    )

    train_state.step += 1

    return loss
end

# ============================================================================
# Learning Rate Schedules
# ============================================================================

"""
    warmup_cosine_schedule(step, warmup_steps, total_steps, base_lr, min_lr)

Cosine annealing with linear warmup.
"""
function warmup_cosine_schedule(
    step::Int,
    warmup_steps::Int,
    total_steps::Int,
    base_lr::Float32,
    min_lr::Float32 = 0.0f0,
)
    if step < warmup_steps
        # Linear warmup
        return base_lr * step / warmup_steps
    else
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5f0 * (base_lr - min_lr) * (1 + cos(Float32(π) * progress))
    end
end

"""
    create_scheduled_optimizer(base_optimizer, schedule_fn)

Wrap an optimizer with a learning rate schedule.
Returns a function that creates the optimizer for a given step.
"""
function create_scheduled_optimizer(base_lr::Float32, schedule_fn)
    return step -> Optimisers.Adam(schedule_fn(step))
end

# ============================================================================
# Evaluation
# ============================================================================

"""
    evaluate(model, params, state, dataloader, mask_token_id; num_batches)

Evaluate model on validation data.
"""
function evaluate(
    model::LLaDAModel,
    params,
    state,
    data_iterator,
    mask_token_id::Int;
    num_batches::Int = 10,
    rng::Random.AbstractRNG = Random.default_rng(),
)
    total_loss = 0.0f0
    count = 0

    for (i, batch) in enumerate(data_iterator)
        if i > num_batches
            break
        end

        loss, _ = diffusion_loss(model, params, state, batch, mask_token_id; rng=rng)
        total_loss += loss
        count += 1
    end

    return count > 0 ? total_loss / count : 0.0f0
end

"""
    compute_accuracy(logits, targets, mask)

Compute prediction accuracy on masked positions.
"""
function compute_accuracy(logits, targets, mask)
    predictions = dropdims(argmax(logits, dims=1), dims=1)

    correct = sum((predictions .== targets) .& mask)
    total = sum(mask)

    return total > 0 ? correct / total : 0.0f0
end

# ============================================================================
# Training Loop
# ============================================================================

"""
    TrainingConfig

Configuration for training loop.
"""
Base.@kwdef struct TrainingConfig
    batch_size::Int = 32
    learning_rate::Float32 = 1e-4f0
    min_learning_rate::Float32 = 1e-6f0
    warmup_steps::Int = 1000
    total_steps::Int = 100000
    eval_every::Int = 1000
    log_every::Int = 100
    save_every::Int = 5000
    mask_schedule::Symbol = :cosine
    gradient_clip::Float32 = 1.0f0
end

"""
    load_training_config(path::String) -> TrainingConfig

Load training configuration from a TOML file.
Looks for [training] section.
"""
function load_training_config(path::String)
    data = TOML.parsefile(path)
    return training_config_from_dict(data)
end

"""
    training_config_from_dict(data::Dict) -> TrainingConfig

Create TrainingConfig from parsed TOML dictionary.
"""
function training_config_from_dict(data::Dict)
    # Get training section
    train_data = get(data, "training", Dict())

    # Handle nested checkpoints section
    checkpoints = get(train_data, "checkpoints", Dict())

    # Convert mask_schedule string to Symbol
    schedule = get(train_data, "mask_schedule", "cosine")
    mask_schedule = schedule isa String ? Symbol(schedule) : schedule

    return TrainingConfig(
        batch_size = get(train_data, "batch_size", 32),
        learning_rate = Float32(get(train_data, "learning_rate", 1e-4)),
        min_learning_rate = Float32(get(train_data, "min_learning_rate", 1e-6)),
        warmup_steps = get(train_data, "warmup_steps", 1000),
        total_steps = get(train_data, "total_steps", 100000),
        eval_every = get(checkpoints, "eval_every", get(train_data, "eval_every", 1000)),
        log_every = get(checkpoints, "log_every", get(train_data, "log_every", 100)),
        save_every = get(checkpoints, "save_every", get(train_data, "save_every", 5000)),
        mask_schedule = mask_schedule,
        gradient_clip = Float32(get(train_data, "gradient_clip", 1.0)),
    )
end

"""
    train!(model, train_state, train_data, config; val_data, callbacks)

Main training loop.

Arguments:
- model: LLaDAModel
- train_state: TrainState from create_train_state
- train_data: Iterator yielding batches of token IDs
- config: TrainingConfig

Optional:
- val_data: Validation data iterator
- callbacks: Dict of callback functions
"""
function train!(
    model::LLaDAModel,
    train_state::TrainState,
    train_data,
    config::TrainingConfig;
    val_data = nothing,
    callbacks = Dict{Symbol, Function}(),
    rng::Random.AbstractRNG = Random.default_rng(),
)
    mask_token_id = model.mask_token_id

    println("Starting training...")
    println("  Total steps: $(config.total_steps)")
    println("  Batch size: $(config.batch_size)")
    println("  Learning rate: $(config.learning_rate)")
    println()

    running_loss = 0.0f0
    loss_count = 0

    for (step, batch) in enumerate(train_data)
        if train_state.step >= config.total_steps
            break
        end

        # Update learning rate
        lr = warmup_cosine_schedule(
            train_state.step,
            config.warmup_steps,
            config.total_steps,
            config.learning_rate,
            config.min_learning_rate,
        )
        Optimisers.adjust!(train_state.optimizer_state, lr)

        # Training step
        loss = train_step!(
            train_state, model, batch, mask_token_id;
            rng=rng, schedule=config.mask_schedule
        )

        running_loss += loss
        loss_count += 1

        # Logging
        if train_state.step % config.log_every == 0
            avg_loss = running_loss / loss_count
            println("Step $(train_state.step) | Loss: $(round(avg_loss, digits=4)) | LR: $(round(lr, sigdigits=3))")
            running_loss = 0.0f0
            loss_count = 0

            if haskey(callbacks, :on_log)
                callbacks[:on_log](train_state, avg_loss, lr)
            end
        end

        # Evaluation
        if val_data !== nothing && train_state.step % config.eval_every == 0
            val_loss = evaluate(model, train_state.params, train_state.state, val_data, mask_token_id; rng=rng)
            println("  Validation Loss: $(round(val_loss, digits=4))")

            if val_loss < train_state.best_loss
                train_state.best_loss = val_loss
                println("  New best!")

                if haskey(callbacks, :on_best)
                    callbacks[:on_best](train_state)
                end
            end

            if haskey(callbacks, :on_eval)
                callbacks[:on_eval](train_state, val_loss)
            end
        end

        # Saving
        if train_state.step % config.save_every == 0
            if haskey(callbacks, :on_save)
                callbacks[:on_save](train_state)
            end
        end
    end

    println("\nTraining complete!")
    println("Final step: $(train_state.step)")
    println("Best validation loss: $(train_state.best_loss)")

    return train_state
end

# ============================================================================
# Exports
# ============================================================================

export masked_cross_entropy, masked_cross_entropy_vectorized, diffusion_loss
export TrainState, create_train_state, train_step!
export warmup_cosine_schedule, create_scheduled_optimizer
export evaluate, compute_accuracy
export TrainingConfig, load_training_config, training_config_from_dict, train!

end # module
