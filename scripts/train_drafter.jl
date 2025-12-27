#!/usr/bin/env julia
# train_drafter.jl - Training script for OssammaDrafter
#
# This script trains the Ossamma drafter model for TiDAR-style generation.
# The drafter learns to predict masked tokens, optionally with distillation
# from a teacher AR model (e.g., Granite 4.0).
#
# Usage:
#   julia --project=. scripts/train_drafter.jl --config configs/drafter_granite.toml
#   julia --project=. scripts/train_drafter.jl --config configs/drafter_granite.toml --resume checkpoints/drafter/latest.jls

using ArgParse
using Dates
using Printf
using Random
using Statistics

using Ossamma
using Lux
using Optimisers
using Zygote

# =============================================================================
# Argument Parsing
# =============================================================================

function parse_commandline()
    s = ArgParseSettings(description="Train OssammaDrafter model")

    @add_arg_table! s begin
        "--config"
            help = "Path to drafter config TOML file"
            arg_type = String
            default = "configs/drafter_granite.toml"
        "--data"
            help = "Path to training data (JSONL with 'text' field)"
            arg_type = String
            default = ""
        "--val-data"
            help = "Path to validation data"
            arg_type = String
            default = ""
        "--batch-size"
            help = "Batch size"
            arg_type = Int
            default = 32
        "--max-steps"
            help = "Maximum training steps"
            arg_type = Int
            default = 100000
        "--lr"
            help = "Learning rate"
            arg_type = Float64
            default = 1e-4
        "--warmup-steps"
            help = "Warmup steps"
            arg_type = Int
            default = 1000
        "--save-every"
            help = "Save checkpoint every N steps"
            arg_type = Int
            default = 1000
        "--log-every"
            help = "Log every N steps"
            arg_type = Int
            default = 100
        "--checkpoint-dir"
            help = "Checkpoint directory"
            arg_type = String
            default = "checkpoints/drafter"
        "--resume"
            help = "Resume from checkpoint"
            arg_type = String
            default = ""
        "--seed"
            help = "Random seed"
            arg_type = Int
            default = 42
        "--mask-ratio"
            help = "Fraction of tokens to mask"
            arg_type = Float64
            default = 0.15
        "--mask-strategy"
            help = "Masking strategy: random | suffix | mixed"
            arg_type = String
            default = "mixed"
        "--draft-length"
            help = "Suffix length for TiDAR-style masking"
            arg_type = Int
            default = 8
        "--suffix-prob"
            help = "Probability of suffix masking when using mixed strategy"
            arg_type = Float64
            default = 0.5
        "--alpha"
            help = "MLM loss weight (1-alpha = distillation weight)"
            arg_type = Float64
            default = 1.0
    end

    return parse_args(s)
end

# =============================================================================
# Data Loading (Simple text-based)
# =============================================================================

struct TextDataset
    texts::Vector{String}
    max_length::Int
end

function load_text_data(path::String; max_length::Int = 512)
    texts = String[]

    if isfile(path)
        for line in eachline(path)
            line = strip(line)
            if !isempty(line)
                # Try to parse as JSON first
                try
                    import JSON3
                    obj = JSON3.read(line)
                    if haskey(obj, :text)
                        push!(texts, String(obj.text))
                    elseif haskey(obj, :content)
                        push!(texts, String(obj.content))
                    else
                        push!(texts, line)
                    end
                catch
                    push!(texts, line)
                end
            end
        end
    end

    return TextDataset(texts, max_length)
end

function sample_batch(
    dataset::TextDataset,
    batch_size::Int,
    vocab_size::Int,
    mask_token_id::Int;
    rng = Random.default_rng(),
    mask_ratio::Float64 = 0.15,
    mask_strategy::Symbol = :mixed,
    draft_length::Int = 8,
    suffix_prob::Float64 = 0.5
)
    n = length(dataset.texts)
    if n == 0
        error("Dataset is empty")
    end

    # Sample random texts
    indices = rand(rng, 1:n, batch_size)
    texts = [dataset.texts[i] for i in indices]

    # Simple character-level tokenization for demo
    # In production, use HFTokenizer
    max_len = dataset.max_length
    token_ids = zeros(Int, max_len, batch_size)
    mask_positions = falses(max_len, batch_size)

    for (b, text) in enumerate(texts)
        # Simple: use character codes as token IDs (clamped to vocab)
        chars = collect(text)
        seq_len = min(length(chars), max_len)

        for i in 1:seq_len
            # Map char to token ID (simple hash)
            token_ids[i, b] = (Int(chars[i]) % (vocab_size - 1)) + 1
        end

        # Apply masking strategy
        use_suffix = mask_strategy == :suffix ||
            (mask_strategy == :mixed && rand(rng) < suffix_prob)

        if use_suffix
            start_pos = max(seq_len - draft_length + 1, 1)
            for i in start_pos:seq_len
                mask_positions[i, b] = true
            end
        else
            for i in 1:seq_len
                if rand(rng) < mask_ratio
                    mask_positions[i, b] = true
                end
            end
        end
    end

    # Create masked version
    masked_ids = copy(token_ids)
    for i in eachindex(mask_positions)
        if mask_positions[i]
            masked_ids[i] = mask_token_id
        end
    end

    return (
        input_ids = masked_ids,
        target_ids = token_ids,
        mask_positions = mask_positions
    )
end

# =============================================================================
# Learning Rate Schedule
# =============================================================================

function warmup_cosine_schedule(step::Int, warmup_steps::Int, max_steps::Int, base_lr::Float64)
    if step < warmup_steps
        # Linear warmup
        return base_lr * (step / warmup_steps)
    else
        # Cosine decay
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return base_lr * 0.5 * (1.0 + cos(π * progress))
    end
end

# =============================================================================
# Training Step
# =============================================================================

function train_step!(model, params, state, opt_state, batch, config;
                     diffusion_t = 0.0f0)
    # Compute gradients
    (loss_val, (new_state, losses)), grads = Zygote.withgradient(params) do ps
        logits, st = model(batch.input_ids, diffusion_t, ps, state)

        # Compute MLM loss
        loss_tuple = combined_drafter_loss(
            logits,
            batch.target_ids,
            nothing,  # No teacher for now
            batch.mask_positions;
            alpha = Float32(config[:alpha])
        )

        return loss_tuple.total, (st, loss_tuple)
    end

    # Update parameters
    opt_state, params = Optimisers.update(opt_state, params, grads[1])

    return (
        params = params,
        state = new_state,
        opt_state = opt_state,
        loss = loss_val,
        losses = losses
    )
end

# =============================================================================
# Main Training Loop
# =============================================================================

function main()
    args = parse_commandline()

    # Set random seed
    Random.seed!(args["seed"])
    rng = Random.default_rng()

    println("=" ^ 60)
    println("OssammaDrafter Training")
    println("=" ^ 60)

    # Load model config
    println("\nLoading config from: $(args["config"])")
    model_config = load_drafter_config(args["config"])
    println("  AR model: $(model_config.ar_model)")
    println("  Vocab size: $(model_config.vocab_size)")
    println("  Embedding dim: $(model_config.embedding_dimension)")
    println("  Layers: $(model_config.number_of_layers)")

    # Create model
    println("\nCreating model...")
    model = OssammaDrafter(model_config)

    # Initialize or resume
    if !isempty(args["resume"]) && isfile(args["resume"])
        println("Resuming from: $(args["resume"])")
        checkpoint = load_drafter_checkpoint(args["resume"])
        params = checkpoint.params
        state = checkpoint.state
        opt_state = checkpoint.opt_state
        start_step = checkpoint.step + 1
        println("  Resumed at step $(start_step)")
    else
        println("Initializing fresh parameters...")
        params = Lux.initialparameters(rng, model)
        state = Lux.initialstates(rng, model)
        opt_state = nothing
        start_step = 1
    end

    # Create optimizer
    base_lr = args["lr"]
    optimizer = Optimisers.AdamW(base_lr, (0.9, 0.999), args["lr"] * 0.01)
    if opt_state === nothing
        opt_state = Optimisers.setup(optimizer, params)
    end

    # Load data
    data_path = args["data"]
    if isempty(data_path) || !isfile(data_path)
        println("\nWARNING: No training data provided. Using synthetic data for demo.")
        # Create synthetic dataset
        synthetic_texts = ["This is a test sentence number $i for training." for i in 1:1000]
        dataset = TextDataset(synthetic_texts, model_config.max_sequence_length)
    else
        println("\nLoading training data from: $data_path")
        dataset = load_text_data(data_path; max_length=model_config.max_sequence_length)
        println("  Loaded $(length(dataset.texts)) examples")
    end

    # Create checkpoint directory
    checkpoint_dir = args["checkpoint-dir"]
    mkpath(checkpoint_dir)

    # Training config
    training_config = Dict(
        :batch_size => args["batch-size"],
        :max_steps => args["max-steps"],
        :warmup_steps => args["warmup-steps"],
        :save_every => args["save-every"],
        :log_every => args["log-every"],
        :alpha => args["alpha"],
        :mask_ratio => args["mask-ratio"],
        :mask_strategy => args["mask-strategy"],
        :draft_length => args["draft-length"],
        :suffix_prob => args["suffix-prob"],
    )

    println("\nTraining config:")
    for (k, v) in training_config
        println("  $k: $v")
    end

    # Training loop
    println("\n" * "=" ^ 60)
    println("Starting training...")
    println("=" ^ 60)

    losses = Float32[]
    start_time = now()

    for step in start_step:args["max-steps"]
        # Update learning rate
        lr = warmup_cosine_schedule(step, args["warmup-steps"], args["max-steps"], base_lr)
        Optimisers.adjust!(opt_state, lr)

        # Sample batch
        batch = sample_batch(
            dataset,
            args["batch-size"],
            model_config.vocab_size,
            model_config.mask_token_id;
            rng = rng,
            mask_ratio = args["mask-ratio"],
            mask_strategy = Symbol(args["mask-strategy"]),
            draft_length = args["draft-length"],
            suffix_prob = args["suffix-prob"]
        )

        # Random diffusion time for this batch
        t = rand(rng, Float32)

        # Training step
        result = train_step!(model, params, state, opt_state, batch, training_config;
                            diffusion_t = t)

        params = result.params
        state = result.state
        opt_state = result.opt_state

        push!(losses, result.loss)

        # Logging
        if step % args["log-every"] == 0
            avg_loss = mean(losses[max(1, end-99):end])
            elapsed = Dates.value(now() - start_time) / 1000
            steps_per_sec = step / elapsed

            @printf("Step %6d | Loss: %.4f | Avg: %.4f | LR: %.2e | %.2f steps/s\n",
                    step, result.loss, avg_loss, lr, steps_per_sec)
        end

        # Checkpointing
        if step % args["save-every"] == 0
            checkpoint_path = joinpath(checkpoint_dir, "step_$(step).jls")
            save_drafter_checkpoint(checkpoint_path;
                model_config = model_config,
                params = params,
                state = state,
                opt_state = opt_state,
                step = step,
                loss = result.loss,
                metadata = Dict("args" => args)
            )

            # Also save as latest
            latest_path = joinpath(checkpoint_dir, "latest.jls")
            cp(checkpoint_path, latest_path, force=true)

            println("  Saved checkpoint: $checkpoint_path")
        end
    end

    # Final save
    final_path = joinpath(checkpoint_dir, "final.jls")
    save_drafter_checkpoint(final_path;
        model_config = model_config,
        params = params,
        state = state,
        opt_state = opt_state,
        step = args["max-steps"],
        loss = losses[end],
        metadata = Dict("args" => args)
    )

    println("\n" * "=" ^ 60)
    println("Training complete!")
    println("Final checkpoint: $final_path")
    println("=" ^ 60)
end

# Run
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
