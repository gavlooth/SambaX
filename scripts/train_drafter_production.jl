#!/usr/bin/env julia
"""
Production training script for OssammaDrafter with HF tokenizer and optional Granite 4 teacher.

Features:
- HuggingFace dataset download
- Granite 4 tokenizer auto-download
- Optional Granite 4 teacher logits for distillation
- Gradient accumulation + clipping
- Periodic eval + checkpointing

Usage:
  julia --project=. scripts/train_drafter_production.jl \
    --config configs/drafter_granite.toml \
    --dataset roneneldan/TinyStories \
    --num-train-rows 20000 --num-val-rows 1000

  julia --project=. scripts/train_drafter_production.jl \
    --config configs/drafter_granite.toml \
    --data data/train.jsonl --val-data data/val.jsonl \
    --use-teacher --teacher-model ibm-granite/granite-4.0-micro
"""

using ArgParse
using Dates
using JSON3
using LinearAlgebra
using Printf
using Random
using Statistics
using PyCall

include(joinpath(@__DIR__, "..", "src", "Ossamma.jl"))
include(joinpath(@__DIR__, "..", "src", "DataLoader.jl"))

using .Ossamma
using .Ossamma: OssammaDrafter, load_drafter_config
using .Ossamma: combined_drafter_loss, save_drafter_checkpoint, load_drafter_checkpoint
using .Ossamma.HFTokenizer: load_tokenizer, batch_encode, get_vocab_size, get_mask_token_id
using .DataLoader: download_hf_dataset, get_texts

using Lux
using Optimisers
using Zygote

# =============================================================================
# Argument Parsing
# =============================================================================

function parse_commandline()
    s = ArgParseSettings(description = "Production training for OssammaDrafter")

    @add_arg_table! s begin
        "--config"
            help = "Path to drafter config TOML file"
            arg_type = String
            default = "configs/drafter_granite.toml"
        "--data"
            help = "Path to training data JSONL with text/content field"
            arg_type = String
            default = ""
        "--val-data"
            help = "Path to validation data JSONL"
            arg_type = String
            default = ""
        "--dataset"
            help = "HuggingFace dataset name (e.g., roneneldan/TinyStories)"
            arg_type = String
            default = ""
        "--dataset-config"
            help = "Dataset config name (e.g., wikitext-2-raw-v1)"
            arg_type = String
            default = "default"
        "--text-column"
            help = "Text column name for HF dataset"
            arg_type = String
            default = "text"
        "--num-train-rows"
            help = "Max training rows to download"
            arg_type = Int
            default = 20000
        "--num-val-rows"
            help = "Max validation rows to download"
            arg_type = Int
            default = 1000
        "--batch-size"
            help = "Batch size"
            arg_type = Int
            default = 16
        "--gradient-accumulation-steps"
            help = "Gradient accumulation steps"
            arg_type = Int
            default = 2
        "--gradient-clip"
            help = "Max gradient norm"
            arg_type = Float64
            default = 1.0
        "--max-steps"
            help = "Maximum training steps"
            arg_type = Int
            default = 100000
        "--lr"
            help = "Learning rate"
            arg_type = Float64
            default = 1e-4
        "--weight-decay"
            help = "Weight decay"
            arg_type = Float64
            default = 0.01
        "--warmup-steps"
            help = "Warmup steps"
            arg_type = Int
            default = 1000
        "--save-every"
            help = "Save checkpoint every N steps"
            arg_type = Int
            default = 1000
        "--eval-every"
            help = "Run validation every N steps"
            arg_type = Int
            default = 500
        "--log-every"
            help = "Log every N steps"
            arg_type = Int
            default = 100
        "--checkpoint-dir"
            help = "Checkpoint directory"
            arg_type = String
            default = "checkpoints/drafter_production"
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
            help = "Probability of suffix masking (mixed)"
            arg_type = Float64
            default = 0.5
        "--alpha"
            help = "MLM loss weight (1-alpha = distillation weight)"
            arg_type = Float64
            default = 1.0
        "--temperature"
            help = "Distillation temperature"
            arg_type = Float64
            default = 1.0
        "--tokenizer-model"
            help = "HF tokenizer model name (Granite 4 default)"
            arg_type = String
            default = "ibm-granite/granite-4.0-micro"
        "--download-verifier"
            help = "Download Granite 4 verifier weights via huggingface_hub"
            action = :store_true
        "--verifier-model"
            help = "HF verifier model name (defaults to tokenizer model)"
            arg_type = String
            default = ""
        "--hf-cache-dir"
            help = "HuggingFace cache directory"
            arg_type = String
            default = ""
        "--use-teacher"
            help = "Enable teacher logits for distillation"
            action = :store_true
        "--teacher-model"
            help = "HF teacher model name (defaults to verifier model)"
            arg_type = String
            default = ""
        "--teacher-device"
            help = "Teacher device: cpu | cuda | cuda:0"
            arg_type = String
            default = "cpu"
        "--teacher-dtype"
            help = "Teacher dtype: float32 | float16 | bfloat16"
            arg_type = String
            default = "float16"
    end

    return parse_args(s)
end

# =============================================================================
# Utilities
# =============================================================================

function load_jsonl_texts(path::String)
    texts = String[]
    if isempty(path) || !isfile(path)
        return texts
    end
    for line in eachline(path)
        line = strip(line)
        if isempty(line)
            continue
        end
        try
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
    return texts
end

function ensure_hf_snapshot(model_name::String; cache_dir::String = "")
    huggingface_hub = pyimport("huggingface_hub")
    kwargs = Dict{Symbol,Any}(:repo_id => model_name)
    if !isempty(cache_dir)
        kwargs[:cache_dir] = cache_dir
    end
    path = huggingface_hub.snapshot_download(; kwargs...)
    return String(path)
end

mutable struct TeacherContext
    model
    torch
    device::String
end

function load_teacher(model_name::String; device::String = "cpu", dtype::String = "float16")
    transformers = pyimport("transformers")
    torch = pyimport("torch")

    dtype_map = Dict(
        "float32" => torch.float32,
        "float16" => torch.float16,
        "bfloat16" => torch.bfloat16,
    )
    torch_dtype = get(dtype_map, lowercase(dtype), torch.float16)

    model = transformers.AutoModelForCausalLM.from_pretrained(model_name; torch_dtype = torch_dtype)
    model = model.to(device)
    model.eval()

    return TeacherContext(model, torch, device)
end

function teacher_logits(ctx::TeacherContext, token_ids::AbstractMatrix, attention_mask::AbstractMatrix)
    torch = ctx.torch
    input_ids = permutedims(token_ids, (2, 1)) .- 1
    attn = permutedims(attention_mask, (2, 1))

    input_tensor = torch.tensor(input_ids; dtype = torch.long, device = ctx.device)
    attn_tensor = torch.tensor(attn; dtype = torch.long, device = ctx.device)

    outputs = ctx.model(input_ids = input_tensor, attention_mask = attn_tensor)
    logits = outputs[:logits]
    logits_np = logits.detach().cpu().numpy()
    logits_julia = Array{Float32}(logits_np)

    # Convert to (vocab, seq, batch)
    return permutedims(logits_julia, (3, 2, 1))
end

function warmup_cosine_schedule(step::Int, warmup_steps::Int, max_steps::Int, base_lr::Float64)
    if step < warmup_steps
        return base_lr * (step / warmup_steps)
    else
        progress = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
        return base_lr * 0.5 * (1.0 + cos(pi * progress))
    end
end

function add_grads(g1, g2)
    if g1 === nothing
        return g2
    elseif g2 === nothing
        return g1
    elseif g1 isa NamedTuple
        return NamedTuple{keys(g1)}(map(add_grads, values(g1), values(g2)))
    elseif g1 isa Tuple
        return map(add_grads, g1, g2)
    else
        return g1 .+ g2
    end
end

function apply_mask_strategy(
    token_ids::AbstractMatrix,
    attention_mask::AbstractMatrix,
    mask_token_id::Int;
    rng = Random.default_rng(),
    mask_ratio::Float64 = 0.15,
    mask_strategy::Symbol = :mixed,
    draft_length::Int = 8,
    suffix_prob::Float64 = 0.5
)
    seq_len, batch = size(token_ids)
    masked = copy(token_ids)
    mask_positions = falses(seq_len, batch)

    for b in 1:batch
        valid = findall(attention_mask[:, b])
        if isempty(valid)
            continue
        end

        use_suffix = mask_strategy == :suffix || (mask_strategy == :mixed && rand(rng) < suffix_prob)
        if use_suffix
            last_pos = last(valid)
            start_pos = max(last_pos - draft_length + 1, first(valid))
            for t in start_pos:last_pos
                masked[t, b] = mask_token_id
                mask_positions[t, b] = true
            end
        else
            for t in valid
                if rand(rng) < mask_ratio
                    masked[t, b] = mask_token_id
                    mask_positions[t, b] = true
                end
            end
        end
    end

    return masked, mask_positions
end

function sample_texts(texts::Vector{String}, batch_size::Int; rng = Random.default_rng())
    n = length(texts)
    idx = rand(rng, 1:n, batch_size)
    return [texts[i] for i in idx]
end

function make_batch(
    tokenizer,
    texts::Vector{String},
    seq_len::Int,
    mask_token_id::Int;
    rng = Random.default_rng(),
    mask_ratio::Float64,
    mask_strategy::Symbol,
    draft_length::Int,
    suffix_prob::Float64
)
    encoding = batch_encode(
        tokenizer,
        texts;
        add_special_tokens = true,
        max_length = seq_len,
        padding = true,
    )
    token_ids = encoding.input_ids
    attention_mask = encoding.attention_mask

    masked_ids, mask_positions = apply_mask_strategy(
        token_ids,
        attention_mask,
        mask_token_id;
        rng = rng,
        mask_ratio = mask_ratio,
        mask_strategy = mask_strategy,
        draft_length = draft_length,
        suffix_prob = suffix_prob
    )

    return (
        input_ids = masked_ids,
        target_ids = token_ids,
        mask_positions = mask_positions,
        attention_mask = attention_mask,
    )
end

function evaluate(
    model,
    params,
    state,
    tokenizer,
    texts::Vector{String},
    seq_len::Int,
    mask_token_id::Int,
    config,
    teacher_ctx::Union{TeacherContext, Nothing}
)
    if isempty(texts)
        return (total = 0.0f0, mlm = 0.0f0, distill = 0.0f0)
    end

    rng = Random.default_rng()
    batch_size = config[:batch_size]
    eval_batches = min(20, max(1, length(texts) ÷ batch_size))

    total = 0.0f0
    mlm = 0.0f0
    distill = 0.0f0

    for _ in 1:eval_batches
        batch_texts = sample_texts(texts, batch_size; rng = rng)
        batch = make_batch(
            tokenizer,
            batch_texts,
            seq_len,
            mask_token_id;
            rng = rng,
            mask_ratio = config[:mask_ratio],
            mask_strategy = config[:mask_strategy],
            draft_length = config[:draft_length],
            suffix_prob = config[:suffix_prob]
        )

        logits, _ = model(batch.input_ids, Float32(config[:mask_ratio]), params, state)

        teacher_logits_val = teacher_ctx === nothing ? nothing :
            teacher_logits(teacher_ctx, batch.target_ids, batch.attention_mask)

        losses = combined_drafter_loss(
            logits,
            batch.target_ids,
            teacher_logits_val,
            batch.mask_positions;
            alpha = Float32(config[:alpha]),
            temperature = Float32(config[:temperature])
        )

        total += losses.total
        mlm += losses.mlm
        distill += losses.distill
    end

    scale = 1.0f0 / eval_batches
    return (total = total * scale, mlm = mlm * scale, distill = distill * scale)
end

# =============================================================================
# Main
# =============================================================================

function main()
    args = parse_commandline()
    Random.seed!(args["seed"])
    rng = Random.default_rng()

    println("=" ^ 60)
    println("OssammaDrafter Production Training")
    println("=" ^ 60)

    println("\nLoading config: $(args["config"]) ")
    model_config = load_drafter_config(args["config"])

    tokenizer_model = args["tokenizer-model"]
    verifier_model = isempty(args["verifier-model"]) ? tokenizer_model : args["verifier-model"]
    teacher_model = isempty(args["teacher-model"]) ? verifier_model : args["teacher-model"]

    if args["download-verifier"]
        println("\nDownloading verifier model: $verifier_model")
        try
            path = ensure_hf_snapshot(verifier_model; cache_dir = args["hf-cache-dir"])
            println("  Verifier cached at: $path")
        catch e
            println("  Failed to download verifier: $e")
            println("  Install huggingface_hub in your Python environment.")
        end
    end

    println("\nLoading tokenizer: $tokenizer_model")
    tokenizer = load_tokenizer(tokenizer_model)

    if model_config.vocab_size != get_vocab_size(tokenizer)
        println("  WARNING: tokenizer vocab_size=$(get_vocab_size(tokenizer)) differs from model_config.vocab_size=$(model_config.vocab_size)")
    end

    mask_token_id = model_config.mask_token_id
    if mask_token_id == 0
        mask_token_id = get_mask_token_id(tokenizer)
    end

    println("\nCreating model...")
    model = OssammaDrafter(model_config)

    # Load data
    train_texts = load_jsonl_texts(args["data"])
    val_texts = load_jsonl_texts(args["val-data"])

    if isempty(train_texts) && !isempty(args["dataset"])
        println("\nDownloading dataset: $(args["dataset"]) ")
        train_dataset = download_hf_dataset(
            args["dataset"]; config = args["dataset-config"], split = "train",
            num_rows = args["num-train-rows"], text_column = args["text-column"]
        )
        train_texts = get_texts(train_dataset)

        val_dataset = try
            download_hf_dataset(
                args["dataset"]; config = args["dataset-config"], split = "validation",
                num_rows = args["num-val-rows"], text_column = args["text-column"]
            )
        catch
            nothing
        end
        if val_dataset !== nothing
            val_texts = get_texts(val_dataset)
        end
    end

    if isempty(train_texts)
        println("\nWARNING: No training data provided. Using synthetic prompts for smoke testing.")
        train_texts = ["This is a synthetic training sentence $(i)." for i in 1:2000]
        val_texts = ["This is a synthetic validation sentence $(i)." for i in 1:200]
    end

    println("\nTrain examples: $(length(train_texts))")
    println("Val examples: $(length(val_texts))")

    # Initialize or resume
    start_step = 1
    if !isempty(args["resume"]) && isfile(args["resume"])
        println("\nResuming from: $(args["resume"]) ")
        checkpoint = load_drafter_checkpoint(args["resume"])
        params = checkpoint.params
        state = checkpoint.state
        opt_state = checkpoint.opt_state
        start_step = checkpoint.step + 1
    else
        params = Lux.initialparameters(rng, model)
        state = Lux.initialstates(rng, model)
        opt_state = nothing
    end

    # Optimizer
    optimizer = Optimisers.AdamW(args["lr"], (0.9, 0.999), args["weight-decay"])
    if opt_state === nothing
        opt_state = Optimisers.setup(optimizer, params)
    end

    # Optional teacher
    teacher_ctx = nothing
    if args["use-teacher"]
        println("\nLoading teacher model: $teacher_model on $(args["teacher-device"]) ")
        teacher_ctx = load_teacher(teacher_model; device = args["teacher-device"], dtype = args["teacher-dtype"])
    end

    # Training config
    mask_strategy = Symbol(lowercase(args["mask-strategy"]))
    if !(mask_strategy in (:random, :suffix, :mixed))
        error("Invalid mask-strategy: $(args["mask-strategy"]). Use random, suffix, or mixed.")
    end

    train_cfg = Dict(
        :batch_size => args["batch-size"],
        :mask_ratio => args["mask-ratio"],
        :mask_strategy => mask_strategy,
        :draft_length => args["draft-length"],
        :suffix_prob => args["suffix-prob"],
        :alpha => args["alpha"],
        :temperature => args["temperature"],
    )

    mkpath(args["checkpoint-dir"])
    best_val = Inf

    println("\nStarting training...")
    losses = Float32[]
    accum_steps = max(1, args["gradient-accumulation-steps"])
    accum_counter = 0
    accum_grads = nothing

    for step in start_step:args["max-steps"]
        lr = warmup_cosine_schedule(step, args["warmup-steps"], args["max-steps"], args["lr"])
        Optimisers.adjust!(opt_state, lr)

        batch_texts = sample_texts(train_texts, args["batch-size"]; rng = rng)
        batch = make_batch(
            tokenizer,
            batch_texts,
            model_config.max_sequence_length,
            mask_token_id;
            rng = rng,
            mask_ratio = train_cfg[:mask_ratio],
            mask_strategy = train_cfg[:mask_strategy],
            draft_length = train_cfg[:draft_length],
            suffix_prob = train_cfg[:suffix_prob]
        )

        teacher_logits_val = teacher_ctx === nothing ? nothing :
            teacher_logits(teacher_ctx, batch.target_ids, batch.attention_mask)

        (loss_val, (new_state, loss_breakdown)), grads = Zygote.withgradient(params) do ps
            logits, st = model(batch.input_ids, Float32(train_cfg[:mask_ratio]), ps, state)
            loss_tuple = combined_drafter_loss(
                logits,
                batch.target_ids,
                teacher_logits_val,
                batch.mask_positions;
                alpha = Float32(train_cfg[:alpha]),
                temperature = Float32(train_cfg[:temperature])
            )
            return loss_tuple.total / accum_steps, (st, loss_tuple)
        end

        accum_grads = accum_grads === nothing ? grads[1] : add_grads(accum_grads, grads[1])
        accum_counter += 1

        if accum_counter >= accum_steps
            grads_flat, rebuild = Optimisers.destructure(accum_grads)
            grad_norm = sqrt(sum(grads_flat .^ 2))
            if grad_norm > args["gradient-clip"]
                grads_flat = grads_flat .* (args["gradient-clip"] / grad_norm)
            end
            clipped_grads = rebuild(grads_flat)

            opt_state, params = Optimisers.update(opt_state, params, clipped_grads)
            accum_grads = nothing
            accum_counter = 0
        end

        state = new_state
        push!(losses, Float32(loss_val * accum_steps))

        if step % args["log-every"] == 0
            @printf(
                "Step %d | loss=%.4f | mlm=%.4f | distill=%.4f | lr=%.2e\n",
                step,
                loss_val * accum_steps,
                loss_breakdown.mlm,
                loss_breakdown.distill,
                lr
            )
        end

        if step % args["eval-every"] == 0 && !isempty(val_texts)
            val_losses = evaluate(
                model,
                params,
                state,
                tokenizer,
                val_texts,
                model_config.max_sequence_length,
                mask_token_id,
                train_cfg,
                teacher_ctx
            )
            @printf(
                "Validation @ step %d | total=%.4f | mlm=%.4f | distill=%.4f\n",
                step,
                val_losses.total,
                val_losses.mlm,
                val_losses.distill
            )

            if val_losses.total < best_val
                best_val = val_losses.total
                save_drafter_checkpoint(
                    joinpath(args["checkpoint-dir"], "best.jls");
                    model_config = model_config,
                    params = params,
                    state = state,
                    opt_state = opt_state,
                    step = step,
                    loss = val_losses.total,
                    metadata = Dict("tokenizer" => tokenizer_model, "teacher" => teacher_model)
                )
            end
        end

        if step % args["save-every"] == 0
            save_drafter_checkpoint(
                joinpath(args["checkpoint-dir"], "step_$(step).jls");
                model_config = model_config,
                params = params,
                state = state,
                opt_state = opt_state,
                step = step,
                loss = loss_val * accum_steps,
                metadata = Dict("tokenizer" => tokenizer_model, "teacher" => teacher_model)
            )
        end
    end

    println("\nTraining complete.")
end

main()
