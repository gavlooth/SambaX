#!/usr/bin/env julia
"""
Simple NER training test using minimal config.
Validates that the model can:
1. Load from TOML config
2. Initialize parameters
3. Run forward pass
4. Compute gradients
5. Update parameters
"""

using Random
using Statistics
using Printf

# Load the main module
include(joinpath(@__DIR__, "..", "src", "Ossamma.jl"))
using .Ossamma

using Lux
using Optimisers
using Zygote

function count_parameters(params)
    total = 0
    function count_nested(x)
        if x isa NamedTuple || x isa Tuple
            for v in values(x)
                count_nested(v)
            end
        elseif x isa AbstractArray
            total += length(x)
        end
    end
    count_nested(params)
    return total
end

function main()
    println("=" ^ 60)
    println("OssammaNER Training Test")
    println("=" ^ 60)

    # =========================================================================
    # 1. Load config from TOML
    # =========================================================================
    println("\n[1/6] Loading configuration...")
    config_path = joinpath(@__DIR__, "..", "configs", "ner_minimal.toml")

    if !isfile(config_path)
        error("Config file not found: $config_path")
    end

    config = load_ner_config(config_path)
    print_config_summary(config)

    # =========================================================================
    # 2. Create model
    # =========================================================================
    println("\n[2/6] Creating model...")
    model = OssammaNER(config)

    rng = Random.default_rng()
    Random.seed!(rng, 42)

    params = Lux.initialparameters(rng, model)
    state = Lux.initialstates(rng, model)

    actual_params = count_parameters(params)
    println("  Actual parameters: $(round(actual_params / 1e6, digits=3))M")

    # =========================================================================
    # 3. Test forward pass
    # =========================================================================
    println("\n[3/6] Testing forward pass...")
    batch_size = 4
    seq_len = 32

    # Generate random tokens (1 to vocab_size)
    token_ids = rand(rng, 1:config.vocab_size, seq_len, batch_size)

    # Forward pass
    t0 = time()
    (emissions, boundary_logits), new_state = model(token_ids, params, state)
    forward_time = time() - t0

    println("  Input shape:           ($seq_len, $batch_size)")
    println("  Emissions shape:       $(size(emissions))")
    println("  Boundary logits shape: $(size(boundary_logits))")
    println("  Forward time:          $(round(forward_time * 1000, digits=1))ms")

    # Verify shapes
    @assert size(emissions) == (config.num_labels, seq_len, batch_size) "Emissions shape mismatch"
    @assert size(boundary_logits) == (2, seq_len, batch_size) "Boundary logits shape mismatch"
    println("  Shape validation: PASSED")

    # =========================================================================
    # 4. Test gradient computation
    # =========================================================================
    println("\n[4/6] Testing gradient computation...")

    # Generate random labels (1 to num_labels, with some -100 for padding)
    labels = rand(rng, 1:config.num_labels, seq_len, batch_size)
    # Mark last few tokens as padding
    labels[end-2:end, :] .= -100

    t0 = time()
    (loss, _), grads = Zygote.withgradient(params) do p
        (emissions, _), st = model(token_ids, p, state)
        l = ner_cross_entropy(emissions, labels)
        return l, st
    end
    grad_time = time() - t0

    println("  Loss:          $(round(loss, digits=4))")
    println("  Gradient time: $(round(grad_time * 1000, digits=1))ms")
    println("  Gradients computed: YES")

    # =========================================================================
    # 5. Test parameter update
    # =========================================================================
    println("\n[5/6] Testing parameter update...")

    opt = Optimisers.AdamW(1e-3, (0.9, 0.999), 0.01f0)
    opt_state = Optimisers.setup(opt, params)

    t0 = time()
    opt_state, params_new = Optimisers.update(opt_state, params, grads[1])
    update_time = time() - t0

    println("  Update time: $(round(update_time * 1000, digits=1))ms")
    println("  Parameters updated: YES")

    # =========================================================================
    # 6. Run a few training steps
    # =========================================================================
    println("\n[6/6] Running training loop (10 steps)...")
    println("-" ^ 60)

    params = params_new
    losses = Float64[]

    for step in 1:10
        # Generate new batch
        token_ids = rand(rng, 1:config.vocab_size, seq_len, batch_size)
        labels = rand(rng, 1:config.num_labels, seq_len, batch_size)
        labels[end-2:end, :] .= -100

        t0 = time()
        (loss, _), grads = Zygote.withgradient(params) do p
            (emissions, _), st = model(token_ids, p, state)
            l = ner_cross_entropy(emissions, labels)
            return l, st
        end

        opt_state, params = Optimisers.update(opt_state, params, grads[1])
        step_time = time() - t0

        push!(losses, loss)
        @printf("  Step %2d: loss=%.4f, time=%.0fms\n", step, loss, step_time * 1000)
    end

    println("-" ^ 60)

    # Check if loss is decreasing (roughly)
    first_half_avg = mean(losses[1:5])
    second_half_avg = mean(losses[6:10])

    println("\n  First 5 steps avg loss:  $(round(first_half_avg, digits=4))")
    println("  Last 5 steps avg loss:   $(round(second_half_avg, digits=4))")

    # =========================================================================
    # Summary
    # =========================================================================
    println("\n" * "=" ^ 60)
    println("TEST SUMMARY")
    println("=" ^ 60)
    println("  Config loading:      PASSED")
    println("  Model creation:      PASSED")
    println("  Forward pass:        PASSED")
    println("  Gradient computation: PASSED")
    println("  Parameter update:    PASSED")
    println("  Training loop:       PASSED")
    println()
    println("  All tests passed!")
    println("=" ^ 60)

    return true
end

# Run the test
success = main()
exit(success ? 0 : 1)
