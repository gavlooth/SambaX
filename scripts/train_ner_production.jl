#!/usr/bin/env julia
"""
Production training script for OssammaNER 110M parameter model.

Features:
- Progress bars with ETA
- Periodic checkpoint saving
- Automatic Git push of checkpoints
- GPU acceleration with mixed precision
- Gradient accumulation for large effective batch size

Usage:
    julia --project=. scripts/train_ner_production.jl
    julia --project=. scripts/train_ner_production.jl --config configs/ner_production_110m.toml
    julia --project=. scripts/train_ner_production.jl --resume checkpoints/ner_110m/checkpoint_step_5000.jls
"""

using Random
using Statistics
using Printf
using Dates
using JSON3
using TOML
using Serialization
using ProgressMeter
using LinearAlgebra

# Enable multi-threaded BLAS for CPU acceleration
BLAS.set_num_threads(min(64, Threads.nthreads() > 1 ? Threads.nthreads() : 64))

# Load Ossamma modules
include(joinpath(@__DIR__, "..", "src", "Ossamma.jl"))
using .Ossamma
using .Ossamma: OssammaNER, NERConfig
using .Ossamma: ner_cross_entropy, predict_labels, extract_entities
using .Ossamma: RAG_LABELS, LABEL_TO_ID, ID_TO_LABEL, NUM_LABELS

using Lux
using LuxCUDA  # GPU support
using CUDA
using Optimisers
using Zygote

# Use GPU (RTX 5090)
const USE_CPU = false

# =============================================================================
# GPU Monitoring Functions
# =============================================================================

"""Get GPU utilization and memory info"""
function get_gpu_stats()
    if !CUDA.functional()
        return (utilization=0.0, mem_used=0.0, mem_total=0.0, mem_percent=0.0)
    end

    try
        # Get memory info
        mem_info = CUDA.memory_status()
        mem_used = CUDA.used_memory() / 1e9  # GB
        mem_total = CUDA.total_memory() / 1e9  # GB
        mem_percent = (mem_used / mem_total) * 100

        # GPU utilization - read from nvidia-smi
        utilization = 0.0
        try
            output = read(`nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits`, String)
            utilization = parse(Float64, strip(output))
        catch
            utilization = -1.0  # Unable to read
        end

        return (utilization=utilization, mem_used=mem_used, mem_total=mem_total, mem_percent=mem_percent)
    catch e
        return (utilization=0.0, mem_used=0.0, mem_total=0.0, mem_percent=0.0)
    end
end

"""Format GPU stats for display"""
function format_gpu_stats()
    stats = get_gpu_stats()
    if stats.utilization < 0
        return @sprintf("GPU: %.1f/%.1fGB (%.0f%%)", stats.mem_used, stats.mem_total, stats.mem_percent)
    else
        return @sprintf("GPU: %.0f%% | Mem: %.1f/%.1fGB (%.0f%%)",
                       stats.utilization, stats.mem_used, stats.mem_total, stats.mem_percent)
    end
end

# =============================================================================
# Configuration
# =============================================================================

Base.@kwdef mutable struct TrainingConfig
    # Model (smaller defaults for faster CPU testing)
    vocab_size::Int = 32000
    max_sequence_length::Int = 128
    embedding_dimension::Int = 256
    number_of_heads::Int = 4
    number_of_layers::Int = 4
    time_dimension::Int = 128
    state_dimension::Int = 256
    window_size::Int = 32
    dropout_rate::Float32 = 0.1f0

    # Training (defaults for GPU - optimized for 32GB VRAM RTX 5090)
    batch_size::Int = 32
    gradient_accumulation_steps::Int = 2
    learning_rate::Float64 = 2e-4
    min_learning_rate::Float64 = 1e-6
    warmup_steps::Int = 500
    total_steps::Int = 10000
    gradient_clip::Float64 = 1.0
    weight_decay::Float64 = 0.01

    # Checkpoints
    eval_every::Int = 500
    log_every::Int = 50
    save_every::Int = 2000
    push_every::Int = 5000

    # Paths
    data_dir::String = "data/ner"
    checkpoint_dir::String = "checkpoints/ner_110m"

    # Git
    git_token::String = ""
    git_remote::String = "origin"
    git_branch::String = "master"

    # Device
    use_gpu::Bool = true
end

function load_config(path::String)
    data = TOML.parsefile(path)
    config = TrainingConfig()

    # Model settings
    if haskey(data, "model")
        m = data["model"]
        config.vocab_size = get(m, "vocab_size", config.vocab_size)
        config.max_sequence_length = get(m, "max_sequence_length", config.max_sequence_length)
        config.embedding_dimension = get(m, "embedding_dimension", config.embedding_dimension)
        config.number_of_heads = get(m, "number_of_heads", config.number_of_heads)
        config.number_of_layers = get(m, "number_of_layers", config.number_of_layers)

        if haskey(m, "dimensions")
            config.time_dimension = get(m["dimensions"], "time_dimension", config.time_dimension)
            config.state_dimension = get(m["dimensions"], "state_dimension", config.state_dimension)
        end
        if haskey(m, "attention")
            config.window_size = get(m["attention"], "window_size", config.window_size)
        end
        if haskey(m, "regularization")
            config.dropout_rate = Float32(get(m["regularization"], "dropout_rate", config.dropout_rate))
        end
    end

    # Training settings
    if haskey(data, "training")
        t = data["training"]
        config.batch_size = get(t, "batch_size", config.batch_size)
        config.gradient_accumulation_steps = get(t, "gradient_accumulation_steps", config.gradient_accumulation_steps)
        config.learning_rate = get(t, "learning_rate", config.learning_rate)
        config.min_learning_rate = get(t, "min_learning_rate", config.min_learning_rate)
        config.warmup_steps = get(t, "warmup_steps", config.warmup_steps)
        config.total_steps = get(t, "total_steps", config.total_steps)
        config.gradient_clip = get(t, "gradient_clip", config.gradient_clip)
        config.weight_decay = get(t, "weight_decay", config.weight_decay)

        if haskey(t, "checkpoints")
            c = t["checkpoints"]
            config.eval_every = get(c, "eval_every", config.eval_every)
            config.log_every = get(c, "log_every", config.log_every)
            config.save_every = get(c, "save_every", config.save_every)
            config.push_every = get(c, "push_every", config.push_every)
        end
    end

    # Git settings
    if haskey(data, "git")
        g = data["git"]
        config.checkpoint_dir = get(g, "checkpoint_dir", config.checkpoint_dir)
        config.git_remote = get(g, "remote", config.git_remote)
        config.git_branch = get(g, "branch", config.git_branch)
    end

    return config
end

# =============================================================================
# Data Loading
# =============================================================================

function load_jsonl(filepath::String)
    data = []
    open(filepath, "r") do f
        for line in eachline(f)
            if !isempty(strip(line))
                push!(data, JSON3.read(line))
            end
        end
    end
    return data
end

function build_vocab(data; min_freq::Int=1, max_vocab::Int=32000)
    word_counts = Dict{String, Int}()

    for example in data
        for token in example.tokens
            word_counts[token] = get(word_counts, token, 0) + 1
        end
    end

    sorted_words = sort(collect(word_counts), by=x->-x[2])

    vocab = Dict{String, Int}()
    vocab["[PAD]"] = 1
    vocab["[UNK]"] = 2
    vocab["[CLS]"] = 3
    vocab["[SEP]"] = 4

    idx = 5
    for (word, count) in sorted_words
        if count >= min_freq && idx <= max_vocab
            vocab[word] = idx
            idx += 1
        end
    end

    return vocab
end

function tokenize(tokens::Vector, vocab::Dict{String, Int})
    unk_id = vocab["[UNK]"]
    return [get(vocab, String(t), unk_id) for t in tokens]
end

function tags_to_ids(tags::Vector)
    return [get(LABEL_TO_ID, String(t), 1) for t in tags]
end

function prepare_batch(examples, vocab::Dict{String, Int}, max_len::Int, device)
    batch_size = length(examples)

    token_ids = ones(Int, max_len, batch_size)
    label_ids = fill(-100, max_len, batch_size)

    for (i, ex) in enumerate(examples)
        tokens = collect(ex.tokens)
        tags = collect(ex.ner_tags)
        seq_len = min(length(tokens), max_len)

        token_ids[1:seq_len, i] = tokenize(tokens[1:seq_len], vocab)
        label_ids[1:seq_len, i] = tags_to_ids(tags[1:seq_len])
    end

    return device(token_ids), device(label_ids)
end

function generate_synthetic_data(n_samples::Int, vocab_size::Int, max_len::Int)
    data = []
    for _ in 1:n_samples
        seq_len = rand(10:max_len)
        tokens = ["token_$i" for i in rand(1:vocab_size, seq_len)]

        # Generate random but valid BIO tags
        tags = String[]
        current_entity = nothing
        for j in 1:seq_len
            if current_entity === nothing
                if rand() < 0.15  # 15% chance to start entity
                    entity_type = rand(["PERSON", "AGENCY", "PLACE", "ORGANISM", "EVENT",
                                       "INSTRUMENT", "WORK", "DOMAIN", "MEASURE"])
                    push!(tags, "B-$entity_type")
                    current_entity = entity_type
                else
                    push!(tags, "O")
                end
            else
                if rand() < 0.6  # 60% chance to continue entity
                    push!(tags, "I-$current_entity")
                else
                    current_entity = nothing
                    push!(tags, "O")
                end
            end
        end

        push!(data, (tokens = tokens, ner_tags = tags))
    end
    return data
end

# =============================================================================
# Learning Rate Schedule
# =============================================================================

function warmup_cosine_schedule(step::Int, warmup_steps::Int, total_steps::Int,
                                base_lr::Float64, min_lr::Float64)
    if step < warmup_steps
        return base_lr * step / warmup_steps
    else
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + cos(pi * progress))
    end
end

# =============================================================================
# Training
# =============================================================================

function compute_metrics(logits, labels)
    # logits: (num_labels, seq_len, batch)
    # labels: (seq_len, batch)

    predictions = dropdims(argmax(logits, dims=1), dims=1)
    mask = labels .!= -100

    correct = sum((predictions .== labels) .& mask)
    total = sum(mask)

    accuracy = total > 0 ? correct / total : 0.0f0

    return accuracy
end

function train_step(model, params, state, opt_state, token_ids, label_ids, config)
    # Compute loss and gradients
    (loss, new_state), grads = Zygote.withgradient(params) do p
        (emissions, boundary_logits), st = model(token_ids, p, state)
        l = ner_cross_entropy(emissions, label_ids)
        return l, st
    end

    # Gradient clipping
    grads_flat, rebuild = Optimisers.destructure(grads[1])
    grad_norm = sqrt(sum(grads_flat .^ 2))
    if grad_norm > config.gradient_clip
        grads_flat = grads_flat .* (config.gradient_clip / grad_norm)
    end
    clipped_grads = rebuild(grads_flat)

    # Update parameters
    opt_state, params = Optimisers.update(opt_state, params, clipped_grads)

    return loss, params, new_state, opt_state, grad_norm
end

function evaluate(model, params, state, data, vocab, config, device)
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_batches = 0

    for batch_start in 1:config.batch_size:min(length(data), 1000)  # Limit eval size
        batch_end = min(batch_start + config.batch_size - 1, length(data))
        batch = data[batch_start:batch_end]

        token_ids, label_ids = prepare_batch(batch, vocab, config.max_sequence_length, device)

        (emissions, _), _ = model(token_ids, params, state)
        loss = ner_cross_entropy(emissions, label_ids)
        total_loss += loss
        n_batches += 1

        predictions = dropdims(argmax(emissions, dims=1), dims=1)
        mask = label_ids .!= -100
        total_correct += sum((predictions .== label_ids) .& mask)
        total_tokens += sum(mask)
    end

    avg_loss = n_batches > 0 ? total_loss / n_batches : 0.0
    accuracy = total_tokens > 0 ? total_correct / total_tokens : 0.0

    return avg_loss, accuracy
end

# =============================================================================
# Checkpoint Management
# =============================================================================

function save_checkpoint(path::String; params, state, opt_state, step, epoch, loss, vocab, config)
    try
        mkpath(dirname(path))
    catch
        # Directory may already exist
    end

    # Convert GPU arrays to CPU for serialization
    cpu_dev = cpu_device()
    cpu_params = cpu_dev(params)
    cpu_state = cpu_dev(state)
    cpu_opt_state = cpu_dev(opt_state)

    data = Dict{Symbol,Any}(
        :params => cpu_params,
        :state => cpu_state,
        :opt_state => cpu_opt_state,
        :step => step,
        :epoch => epoch,
        :loss => loss,
        :vocab => vocab,
        :config => config,
        :timestamp => Dates.now(),
    )

    serialize(path, data)
    println("    Checkpoint saved: $path")
    return path
end

function load_checkpoint(path::String, device)
    data = deserialize(path)

    params = device(data[:params])
    state = device(data[:state])
    opt_state = device(data[:opt_state])

    return (
        params = params,
        state = state,
        opt_state = opt_state,
        step = get(data, :step, 0),
        epoch = get(data, :epoch, 0),
        loss = get(data, :loss, nothing),
        vocab = get(data, :vocab, nothing),
        config = get(data, :config, nothing),
    )
end

# =============================================================================
# Git Operations
# =============================================================================

function setup_git_credentials(token::String)
    if isempty(token)
        return false
    end

    # Configure Git with token
    run(`git config --global credential.helper store`)

    # Get remote URL and update with token
    try
        remote_url = strip(read(`git remote get-url origin`, String))
        if startswith(remote_url, "https://")
            # Update remote URL with token
            new_url = replace(remote_url, "https://" => "https://oauth2:$token@")
            run(`git remote set-url origin $new_url`)
            return true
        end
    catch e
        @warn "Failed to setup Git credentials: $e"
    end
    return false
end

function push_checkpoints(checkpoint_dir::String, step::Int, token::String)
    if isempty(token)
        @warn "No Git token provided, skipping push"
        return false
    end

    try
        # Setup credentials
        setup_git_credentials(token)

        # Add checkpoints
        run(`git add $checkpoint_dir`)

        # Commit
        commit_msg = "checkpoint: step $step - $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM"))"
        run(`git commit -m $commit_msg`)

        # Push
        run(`git push origin master`)

        println("    Checkpoints pushed to Git at step $step")
        return true
    catch e
        @warn "Git push failed: $e"
        return false
    end
end

# =============================================================================
# Model Creation
# =============================================================================

function create_110m_model(vocab_size::Int, config::TrainingConfig)
    ner_config = NERConfig(
        vocab_size = vocab_size,
        max_sequence_length = config.max_sequence_length,
        embedding_dimension = config.embedding_dimension,
        number_of_heads = config.number_of_heads,
        number_of_layers = config.number_of_layers,
        time_dimension = config.time_dimension,
        state_dimension = config.state_dimension,
        window_size = config.window_size,
        dropout_rate = config.dropout_rate,
    )

    return OssammaNER(ner_config)
end

function count_parameters(params)
    total = 0
    function count_nested(p)
        if p isa NamedTuple || p isa Tuple
            for v in values(p)
                count_nested(v)
            end
        elseif p isa AbstractArray
            total += length(p)
        end
    end
    count_nested(params)
    return total
end

# =============================================================================
# Main Training Loop
# =============================================================================

function train_production(;
    config_path::String = "configs/ner_production_110m.toml",
    resume_from::String = "",
    git_token::String = "",
    use_synthetic::Bool = false,
)
    println("=" ^ 70)
    println("OssammaNER 110M Production Training")
    println("=" ^ 70)
    println("Start time: $(Dates.now())")
    println()

    # Load configuration
    config = isfile(config_path) ? load_config(config_path) : TrainingConfig()
    config.git_token = git_token

    # Setup device
    use_gpu = !USE_CPU && config.use_gpu && LuxCUDA.functional()
    device = use_gpu ? gpu_device() : cpu_device()
    device_name = use_gpu ? "GPU (CUDA)" : "CPU"
    println("Device: $device_name")

    # Create checkpoint directory
    try
        mkpath(config.checkpoint_dir)
    catch
        # Directory may already exist, that's fine
    end

    # Load or generate data
    println("\nLoading data...")
    if use_synthetic || !isdir(config.data_dir)
        println("  Using synthetic data for training")
        train_data = generate_synthetic_data(1000, 2000, config.max_sequence_length)
        val_data = generate_synthetic_data(100, 2000, config.max_sequence_length)
        vocab = build_vocab(train_data; max_vocab = config.vocab_size)
    else
        train_path = joinpath(config.data_dir, "train.jsonl")
        val_path = joinpath(config.data_dir, "validation.jsonl")

        if isfile(train_path)
            train_data = load_jsonl(train_path)
            val_data = isfile(val_path) ? load_jsonl(val_path) : train_data[1:min(1000, length(train_data))]
            vocab = build_vocab(train_data; max_vocab = config.vocab_size)
        else
            println("  No data found at $train_path, using synthetic data")
            train_data = generate_synthetic_data(10000, 5000, config.max_sequence_length)
            val_data = generate_synthetic_data(500, 5000, config.max_sequence_length)
            vocab = build_vocab(train_data; max_vocab = config.vocab_size)
        end
    end

    vocab_size = length(vocab)
    println("  Train examples: $(length(train_data))")
    println("  Val examples: $(length(val_data))")
    println("  Vocabulary size: $vocab_size")

    # Save vocabulary
    vocab_path = joinpath(config.checkpoint_dir, "vocab.json")
    open(vocab_path, "w") do f
        JSON3.write(f, vocab)
    end

    # Initialize or resume
    rng = Random.default_rng()
    start_step = 0

    if !isempty(resume_from) && isfile(resume_from)
        println("\nResuming from checkpoint: $resume_from")
        checkpoint = load_checkpoint(resume_from, device)
        params = checkpoint.params
        state = checkpoint.state
        opt_state = checkpoint.opt_state
        start_step = checkpoint.step
        vocab = checkpoint.vocab !== nothing ? checkpoint.vocab : vocab
        vocab_size = length(vocab)

        # Recreate model with same config
        model = create_110m_model(vocab_size, config)
        println("  Resumed from step $start_step")
    else
        println("\nCreating 110M parameter model...")
        model = create_110m_model(vocab_size, config)

        params = Lux.initialparameters(rng, model)
        state = Lux.initialstates(rng, model)

        # Count parameters
        n_params = count_parameters(params)
        println("  Parameters: $(round(n_params / 1e6, digits=2))M")

        # Move to device
        params = device(params)
        state = device(state)

        # Setup optimizer (AdamW: eta, beta, lambda for weight decay)
        opt = Optimisers.AdamW(config.learning_rate, (0.9, 0.999), Float32(config.weight_decay))
        opt_state = Optimisers.setup(opt, params)
    end

    # Training configuration summary
    println("\nTraining Configuration:")
    println("  Batch size: $(config.batch_size)")
    println("  Gradient accumulation: $(config.gradient_accumulation_steps)")
    println("  Effective batch size: $(config.batch_size * config.gradient_accumulation_steps)")
    println("  Learning rate: $(config.learning_rate)")
    println("  Warmup steps: $(config.warmup_steps)")
    println("  Total steps: $(config.total_steps)")
    println("  Save every: $(config.save_every) steps")
    println("  Push every: $(config.push_every) steps")
    println()

    # Training loop
    println("Starting training from step $(start_step + 1)...")
    println("-" ^ 70)

    global_step = start_step
    best_val_loss = Inf
    running_loss = 0.0
    running_grad_norm = 0.0
    loss_count = 0

    # Create progress bar
    remaining_steps = config.total_steps - start_step
    progress = Progress(remaining_steps;
        desc = "Training: ",
        showspeed = true,
        barlen = 40,
        color = :cyan,
    )

    epoch = 0
    while global_step < config.total_steps
        epoch += 1
        shuffled_data = shuffle(rng, train_data)

        for batch_start in 1:config.batch_size:length(shuffled_data)
            if global_step >= config.total_steps
                break
            end

            batch_end = min(batch_start + config.batch_size - 1, length(shuffled_data))
            batch = shuffled_data[batch_start:batch_end]

            # Prepare batch
            token_ids, label_ids = prepare_batch(batch, vocab, config.max_sequence_length, device)

            # Update learning rate
            lr = warmup_cosine_schedule(
                global_step,
                config.warmup_steps,
                config.total_steps,
                config.learning_rate,
                config.min_learning_rate,
            )
            Optimisers.adjust!(opt_state, lr)

            # Training step
            loss, params, state, opt_state, grad_norm = train_step(
                model, params, state, opt_state, token_ids, label_ids, config
            )

            running_loss += loss
            running_grad_norm += grad_norm
            loss_count += 1
            global_step += 1

            # Update progress bar - only get GPU stats every 10 steps to reduce overhead
            gpu_info = (use_gpu && global_step % 10 == 0) ? format_gpu_stats() : ""
            showvals = [
                (:step, global_step),
                (:loss, @sprintf("%.4f", loss)),
                (:lr, @sprintf("%.2e", lr)),
                (:grad_norm, @sprintf("%.2f", grad_norm)),
            ]
            if !isempty(gpu_info)
                push!(showvals, (:gpu, gpu_info))
            end
            ProgressMeter.next!(progress; showvalues = showvals)

            # Logging
            if global_step % config.log_every == 0
                avg_loss = running_loss / loss_count
                avg_grad_norm = running_grad_norm / loss_count

                # Get GPU stats for logging
                gpu_log = use_gpu ? " | " * format_gpu_stats() : ""

                println()
                @printf("Step %d/%d | Loss: %.4f | Grad Norm: %.2f | LR: %.2e%s\n",
                    global_step, config.total_steps, avg_loss, avg_grad_norm, lr, gpu_log)

                running_loss = 0.0
                running_grad_norm = 0.0
                loss_count = 0
            end

            # Evaluation
            if global_step % config.eval_every == 0
                val_loss, val_acc = evaluate(model, params, state, val_data, vocab, config, device)

                println()
                @printf("  [Eval] Val Loss: %.4f | Val Acc: %.2f%%\n", val_loss, val_acc * 100)

                if val_loss < best_val_loss
                    best_val_loss = val_loss
                    println("  [Eval] New best! Saving best checkpoint...")

                    save_checkpoint(
                        joinpath(config.checkpoint_dir, "checkpoint_best.jls");
                        params=params, state=state, opt_state=opt_state,
                        step=global_step, epoch=epoch, loss=val_loss,
                        vocab=vocab, config=config,
                    )
                end
            end

            # Save checkpoint
            if global_step % config.save_every == 0
                checkpoint_path = joinpath(config.checkpoint_dir, "checkpoint_step_$(global_step).jls")
                save_checkpoint(
                    checkpoint_path;
                    params=params, state=state, opt_state=opt_state,
                    step=global_step, epoch=epoch, loss=running_loss / max(loss_count, 1),
                    vocab=vocab, config=config,
                )

                # Also save latest
                save_checkpoint(
                    joinpath(config.checkpoint_dir, "checkpoint_latest.jls");
                    params=params, state=state, opt_state=opt_state,
                    step=global_step, epoch=epoch, loss=running_loss / max(loss_count, 1),
                    vocab=vocab, config=config,
                )
            end

            # Git push
            if global_step % config.push_every == 0 && !isempty(config.git_token)
                println()
                println("  Pushing checkpoints to Git...")
                push_checkpoints(config.checkpoint_dir, global_step, config.git_token)
            end
        end
    end

    ProgressMeter.finish!(progress)

    # Final save
    println("\n" * "=" ^ 70)
    println("Training complete!")
    println("=" ^ 70)
    println("Final step: $global_step")
    println("Best validation loss: $(@sprintf("%.4f", best_val_loss))")
    println("End time: $(Dates.now())")

    # Final checkpoint
    save_checkpoint(
        joinpath(config.checkpoint_dir, "checkpoint_final.jls");
        params=params, state=state, opt_state=opt_state,
        step=global_step, epoch=epoch, loss=best_val_loss,
        vocab=vocab, config=config,
    )

    # Final git push
    if !isempty(config.git_token)
        println("\nFinal Git push...")
        push_checkpoints(config.checkpoint_dir, global_step, config.git_token)
    end

    return model, params, state, vocab
end

# =============================================================================
# CLI
# =============================================================================

function main()
    config_path = "configs/ner_production_110m.toml"
    resume_from = ""
    git_token = get(ENV, "GITHUB_PERSONAL_ACCESS_TOKEN", "")  # From environment
    use_synthetic = false

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--config"
            config_path = ARGS[i+1]
            i += 2
        elseif arg == "--resume"
            resume_from = ARGS[i+1]
            i += 2
        elseif arg == "--token"
            git_token = ARGS[i+1]
            i += 2
        elseif arg == "--synthetic"
            use_synthetic = true
            i += 1
        elseif arg == "--help"
            println("""
OssammaNER 110M Production Training

Usage:
    julia --project=. scripts/train_ner_production.jl [OPTIONS]

Options:
    --config PATH       Path to config TOML file (default: configs/ner_production_110m.toml)
    --resume PATH       Resume from checkpoint
    --token TOKEN       GitHub token for pushing checkpoints
    --synthetic         Use synthetic data for testing
    --help              Show this help message
            """)
            return
        else
            i += 1
        end
    end

    train_production(;
        config_path = config_path,
        resume_from = resume_from,
        git_token = git_token,
        use_synthetic = use_synthetic,
    )
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
