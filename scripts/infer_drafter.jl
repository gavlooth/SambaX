#!/usr/bin/env julia
# infer_drafter.jl - Inference script for TiDAR-style generation
#
# This script implements the TiDAR inference loop:
# 1. Drafter proposes K tokens in parallel (diffusion)
# 2. AR verifier validates via rejection sampling
# 3. Accept valid tokens, resample rejected ones
#
# Usage:
#   julia --project=. scripts/infer_drafter.jl --checkpoint checkpoints/drafter/latest.jls --prompt "Hello, world"
#
# Note: Full TiDAR requires an AR model (Granite 4.0). This script provides
# the drafter-only mode and a template for integration.

using ArgParse
using Printf
using Random
using Statistics

using Ossamma
using Lux
using NNlib

# =============================================================================
# Argument Parsing
# =============================================================================

function parse_commandline()
    s = ArgParseSettings(description="TiDAR-style inference with OssammaDrafter")

    @add_arg_table! s begin
        "--checkpoint"
            help = "Path to drafter checkpoint"
            arg_type = String
            required = true
        "--prompt"
            help = "Text prompt to continue"
            arg_type = String
            default = "The quick brown fox"
        "--max-new-tokens"
            help = "Maximum new tokens to generate"
            arg_type = Int
            default = 64
        "--draft-length"
            help = "Number of tokens to draft per step"
            arg_type = Int
            default = 8
        "--temperature"
            help = "Sampling temperature"
            arg_type = Float64
            default = 1.0
        "--top-p"
            help = "Top-p (nucleus) sampling threshold"
            arg_type = Float64
            default = 0.9
        "--seed"
            help = "Random seed"
            arg_type = Int
            default = 42
        "--mode"
            help = "Inference mode: 'draft-only' or 'tidar' (requires AR model)"
            arg_type = String
            default = "draft-only"
        "--ar-model"
            help = "AR model for TiDAR mode (HuggingFace model name)"
            arg_type = String
            default = "ibm-granite/granite-4.0-micro"
        "--add-bos-to-verifier"
            help = "Prepend BOS token to verifier input for alignment"
            arg_type = Bool
            default = true
        "--bos-token-id"
            help = "BOS token ID (1-based)"
            arg_type = Int
            default = 1
    end

    return parse_args(s)
end

# =============================================================================
# Sampling Utilities
# =============================================================================

"""
    sample_from_logits(logits; temperature=1.0, top_p=0.9, rng=Random.default_rng())

Sample token ID from logits using temperature and nucleus sampling.
"""
function sample_from_logits(
    logits::AbstractVector;
    temperature::Float64 = 1.0,
    top_p::Float64 = 0.9,
    rng = Random.default_rng()
)
    # Apply temperature
    scaled_logits = logits ./ Float32(temperature)

    # Convert to probabilities
    probs = NNlib.softmax(scaled_logits)

    # Top-p (nucleus) sampling
    sorted_indices = sortperm(probs, rev=true)
    cumsum_probs = cumsum(probs[sorted_indices])

    # Find cutoff
    cutoff_idx = findfirst(x -> x > top_p, cumsum_probs)
    if cutoff_idx === nothing
        cutoff_idx = length(probs)
    end

    # Zero out tokens beyond cutoff
    nucleus_mask = falses(length(probs))
    for i in 1:cutoff_idx
        nucleus_mask[sorted_indices[i]] = true
    end

    # Renormalize
    masked_probs = probs .* nucleus_mask
    masked_probs ./= sum(masked_probs)

    # Sample
    r = rand(rng)
    cumsum_masked = cumsum(masked_probs)
    token_id = findfirst(x -> x > r, cumsum_masked)

    return token_id
end

"""
    sample_batch_from_logits(logits; kwargs...)

Sample from batched logits (vocab_size, seq_len).
"""
function sample_batch_from_logits(
    logits::AbstractMatrix;
    temperature::Float64 = 1.0,
    top_p::Float64 = 0.9,
    rng = Random.default_rng()
)
    seq_len = size(logits, 2)
    tokens = Int[]

    for i in 1:seq_len
        token = sample_from_logits(logits[:, i]; temperature, top_p, rng)
        push!(tokens, token)
    end

    return tokens
end

# =============================================================================
# Draft-Only Generation
# =============================================================================

"""
    generate_draft_only(model, params, state, prompt_ids, config; kwargs...)

Generate tokens using drafter only (no AR verification).
This is useful for testing the drafter quality.
"""
function generate_draft_only(
    model,
    params,
    state,
    prompt_ids::Vector{Int},
    max_new_tokens::Int;
    draft_length::Int = 8,
    temperature::Float64 = 1.0,
    top_p::Float64 = 0.9,
    rng = Random.default_rng()
)
    generated = copy(prompt_ids)
    mask_token_id = model.mask_token_id

    tokens_generated = 0
    steps = 0

    while tokens_generated < max_new_tokens
        steps += 1

        # Create input with mask tokens for drafting
        current_len = length(generated)
        tokens_to_draft = min(draft_length, max_new_tokens - tokens_generated)

        # Append mask tokens
        input_ids = vcat(generated, fill(mask_token_id, tokens_to_draft))

        # Run drafter (at t=0 for fully denoised)
        logits, state = model(input_ids, 0.0f0, params, state)

        # Sample only the masked positions
        draft_start = current_len + 1
        draft_end = current_len + tokens_to_draft
        draft_logits = logits[:, draft_start:draft_end]

        drafted_tokens = sample_batch_from_logits(draft_logits;
            temperature, top_p, rng)

        # Accept all drafted tokens (no AR verification)
        append!(generated, drafted_tokens)
        tokens_generated += tokens_to_draft
    end

    return (
        tokens = generated,
        steps = steps,
        tokens_generated = tokens_generated
    )
end

# =============================================================================
# TiDAR Generation (with AR verification)
# =============================================================================

"""
    generate_tidar(drafter, drafter_params, drafter_state,
                   ar_model, ar_tokenizer,
                   prompt_ids, max_new_tokens; kwargs...)

Full TiDAR generation with AR verification.

Note: This is a template - full implementation requires integration with
an AR model via Python (e.g., using PyCall with transformers).
"""
function generate_tidar(
    drafter,
    drafter_params,
    drafter_state,
    prompt_ids::Vector{Int},
    max_new_tokens::Int;
    draft_length::Int = 8,
    temperature::Float64 = 1.0,
    top_p::Float64 = 0.9,
    add_bos_to_verifier::Bool = true,
    bos_token_id::Int = 1,
    rng = Random.default_rng(),
    ar_forward_fn = nothing  # Function: (token_ids[, state]) -> logits or (logits, state)
)
    if ar_forward_fn === nothing
        @warn "No AR model provided. Falling back to draft-only mode."
        return generate_draft_only(drafter, drafter_params, drafter_state,
            prompt_ids, max_new_tokens;
            draft_length, temperature, top_p, rng)
    end

    generated = copy(prompt_ids)
    mask_token_id = drafter.mask_token_id

    tokens_generated = 0
    total_drafted = 0
    total_accepted = 0
    steps = 0

    ar_state = nothing

    while tokens_generated < max_new_tokens
        steps += 1
        current_len = length(generated)
        tokens_to_draft = min(draft_length, max_new_tokens - tokens_generated)

        # === DRAFT PHASE ===
        # Drafter proposes tokens in parallel
        input_ids = vcat(generated, fill(mask_token_id, tokens_to_draft))
        draft_logits, drafter_state = drafter(input_ids, 0.0f0, drafter_params, drafter_state)

        # Sample from drafter
        draft_start = current_len + 1
        draft_end = current_len + tokens_to_draft
        draft_logits_slice = draft_logits[:, draft_start:draft_end]

        drafted_tokens = Int[]
        for i in 1:tokens_to_draft
            token = sample_from_logits(draft_logits[:, draft_start + i - 1];
                temperature, top_p, rng)
            push!(drafted_tokens, token)
        end

        total_drafted += tokens_to_draft

        # === VERIFY PHASE ===
        # AR model computes logits causally
        verify_input = add_bos_to_verifier ? vcat(bos_token_id, generated, drafted_tokens) : vcat(generated, drafted_tokens)
        ar_out = if applicable(ar_forward_fn, verify_input, ar_state)
            ar_forward_fn(verify_input, ar_state)
        else
            ar_forward_fn(verify_input)
        end
        ar_logits, ar_state = ar_out isa Tuple ? ar_out : (ar_out, ar_state)
        verifier_offset = add_bos_to_verifier ? 1 : 0

        accepted, _rejection_idx, replacement = verify_and_accept(
            vcat(generated, drafted_tokens), current_len, ar_logits;
            draft_logits = draft_logits_slice,
            temperature = Float32(temperature),
            mode = :rejection,
            top_p = Float32(top_p),
            verifier_offset = verifier_offset,
            rng = rng
        )

        total_accepted += accepted

        # Accept tokens
        if accepted == tokens_to_draft
            append!(generated, drafted_tokens)
            tokens_generated += tokens_to_draft
        else
            if accepted > 0
                append!(generated, drafted_tokens[1:accepted])
            end
            if replacement === nothing
                replacement = argmax(ar_logits[:, current_len + accepted + verifier_offset])
            end
            push!(generated, replacement)
            tokens_generated += accepted + 1
        end
    end

    acceptance_rate = total_accepted / max(total_drafted, 1)

    return (
        tokens = generated,
        steps = steps,
        tokens_generated = tokens_generated,
        total_drafted = total_drafted,
        acceptance_rate = acceptance_rate
    )
end

# =============================================================================
# Simple Tokenization (for demo)
# =============================================================================

function simple_encode(text::String, vocab_size::Int)
    # Simple character-level encoding
    return [(Int(c) % (vocab_size - 1)) + 1 for c in text]
end

function simple_decode(token_ids::Vector{Int})
    # Simple character-level decoding (approximate)
    return join([Char(clamp(id, 32, 126)) for id in token_ids])
end

# =============================================================================
# Main
# =============================================================================

function main()
    args = parse_commandline()

    Random.seed!(args["seed"])
    rng = Random.default_rng()

    println("=" ^ 60)
    println("TiDAR Inference with OssammaDrafter")
    println("=" ^ 60)

    # Load checkpoint
    println("\nLoading checkpoint: $(args["checkpoint"])")
    checkpoint = load_drafter_checkpoint(args["checkpoint"])
    model_config = checkpoint.model_config
    params = checkpoint.params
    state = checkpoint.state

    println("  Model: $(model_config.ar_model)")
    println("  Vocab size: $(model_config.vocab_size)")
    println("  Step: $(checkpoint.step)")

    # Create model
    model = OssammaDrafter(model_config)

    # Encode prompt
    prompt = args["prompt"]
    println("\nPrompt: \"$prompt\"")
    prompt_ids = simple_encode(prompt, model_config.vocab_size)
    println("  Encoded length: $(length(prompt_ids))")

    # Generate
    println("\nGenerating (mode: $(args["mode"]))...")
    println("  Draft length: $(args["draft-length"])")
    println("  Temperature: $(args["temperature"])")
    println("  Max new tokens: $(args["max-new-tokens"])")

    start_time = time()

    if args["mode"] == "draft-only"
        result = generate_draft_only(
            model, params, state,
            prompt_ids,
            args["max-new-tokens"];
            draft_length = args["draft-length"],
            temperature = args["temperature"],
            top_p = args["top-p"],
            rng = rng
        )
    else
        # TiDAR mode - would need AR model integration
        println("\nWARNING: Full TiDAR mode requires AR model integration.")
        println("         Falling back to draft-only mode.")
        result = generate_draft_only(
            model, params, state,
            prompt_ids,
            args["max-new-tokens"];
            draft_length = args["draft-length"],
            temperature = args["temperature"],
            top_p = args["top-p"],
            rng = rng
        )
    end

    elapsed = time() - start_time

    # Decode and display
    generated_text = simple_decode(result.tokens)

    println("\n" * "=" ^ 60)
    println("Generated text:")
    println("=" ^ 60)
    println(generated_text)

    println("\n" * "-" ^ 60)
    println("Statistics:")
    @printf("  Steps: %d\n", result.steps)
    @printf("  Tokens generated: %d\n", result.tokens_generated)
    @printf("  Time: %.2f s\n", elapsed)
    @printf("  Tokens/sec: %.2f\n", result.tokens_generated / elapsed)

    if haskey(result, :acceptance_rate)
        @printf("  Acceptance rate: %.2f%%\n", result.acceptance_rate * 100)
    end

    println("=" ^ 60)
end

# Run
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
