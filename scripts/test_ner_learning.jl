#!/usr/bin/env julia
"""
Test if the NER model can learn patterns (not random data).

Creates synthetic NER data with learnable patterns:
- Tokens 1-200: Always label "O" (outside entity)
- Tokens 201-400: Always "B-PERSON"
- etc.

If the model learns, loss should drop well below random baseline (~2.94).
"""

using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "src", "Ossamma.jl"))
using .Ossamma

using Lux
using Optimisers
using Zygote

# Label mapping (5 classes for this test)
const LABEL_O = 1
const LABEL_B_PERSON = 2
const LABEL_I_PERSON = 3
const LABEL_B_PLACE = 4
const LABEL_I_PLACE = 5

"""
Generate synthetic data with learnable patterns.
Token ranges map deterministically to labels.
"""
function generate_patterned_batch(rng, vocab_size, seq_len, batch_size)
    token_ids = rand(rng, 1:vocab_size, seq_len, batch_size)
    labels = similar(token_ids)

    for b in 1:batch_size
        for t in 1:seq_len
            tok = token_ids[t, b]
            # Deterministic mapping: token range -> label
            if tok <= div(vocab_size, 5)
                labels[t, b] = LABEL_O
            elseif tok <= 2 * div(vocab_size, 5)
                labels[t, b] = LABEL_B_PERSON
            elseif tok <= 3 * div(vocab_size, 5)
                labels[t, b] = LABEL_I_PERSON
            elseif tok <= 4 * div(vocab_size, 5)
                labels[t, b] = LABEL_B_PLACE
            else
                labels[t, b] = LABEL_I_PLACE
            end
        end
    end

    return token_ids, labels
end

function main()
    println("=" ^ 60)
    println("OssammaNER Learning Test (Patterned Data)")
    println("=" ^ 60)
    println()
    println("This test uses PATTERNED data (not random) to verify")
    println("the model can learn actual token->label mappings.")
    println()
    println("Random baseline (19 classes): -log(1/19) = 2.944")
    println("Random baseline (5 classes):  -log(1/5)  = 1.609")
    println("Target: Loss should drop well below 1.0 if learning works")
    println()

    # Use minimal config for speed
    config_path = joinpath(@__DIR__, "..", "configs", "ner_minimal.toml")
    config = load_ner_config(config_path)

    println("Config: ner_minimal.toml")
    println("  vocab_size: $(config.vocab_size)")
    println("  embedding_dim: $(config.embedding_dimension)")
    println("  layers: $(config.number_of_layers)")

    # Create model
    println("\n[1/3] Creating model...")
    model = OssammaNER(config)

    rng = Random.default_rng()
    Random.seed!(rng, 42)

    params = Lux.initialparameters(rng, model)
    state = Lux.initialstates(rng, model)

    # Setup training
    batch_size = 16
    seq_len = 32
    n_steps = 200
    lr = 3e-3  # Higher LR for faster convergence

    opt = Optimisers.AdamW(Float32(lr), (0.9f0, 0.999f0), 0.01f0)
    opt_state = Optimisers.setup(opt, params)

    println("\n[2/3] Training on patterned data ($(n_steps) steps)...")
    println("-" ^ 60)

    losses = Float64[]

    for step in 1:n_steps
        # Generate patterned batch (learnable!)
        token_ids, labels = generate_patterned_batch(rng, config.vocab_size, seq_len, batch_size)

        # Training step
        (loss, _), grads = Zygote.withgradient(params) do p
            (emissions, _), st = model(token_ids, p, state)
            l = ner_cross_entropy(emissions, labels)
            return l, st
        end

        opt_state, params = Optimisers.update(opt_state, params, grads[1])
        push!(losses, loss)

        if step == 1 || step % 25 == 0 || step == n_steps
            @printf("  Step %3d: loss = %.4f\n", step, loss)
        end
    end

    println("-" ^ 60)

    # Analysis
    println("\n[3/3] Results")
    println("=" ^ 60)

    initial_loss = losses[1]
    final_loss = losses[end]
    min_loss = minimum(losses)
    random_baseline_19 = -log(1/19)
    random_baseline_5 = -log(1/5)

    println("  Initial loss:       $(round(initial_loss, digits=4))")
    println("  Final loss:         $(round(final_loss, digits=4))")
    println("  Minimum loss:       $(round(min_loss, digits=4))")
    println()
    println("  Random baseline (19): $(round(random_baseline_19, digits=4))")
    println("  Random baseline (5):  $(round(random_baseline_5, digits=4))")
    println()

    # Verdict
    if final_loss < 0.5
        println("  *** EXCELLENT: Loss < 0.5 - model learns patterns very well!")
        verdict = :excellent
    elseif final_loss < 1.0
        println("  ** GOOD: Loss < 1.0 - model is learning effectively")
        verdict = :good
    elseif final_loss < random_baseline_5
        println("  * OK: Below 5-class baseline - some learning")
        verdict = :ok
    elseif final_loss < random_baseline_19
        println("  ~ MARGINAL: Below 19-class baseline but weak")
        verdict = :marginal
    else
        println("  X POOR: At/above baseline - model not learning")
        verdict = :poor
    end

    println()
    reduction = (initial_loss - final_loss) / initial_loss * 100
    println("  Loss reduction: $(round(reduction, digits=1))%")

    println("=" ^ 60)

    return verdict in [:excellent, :good, :ok]
end

success = main()
exit(success ? 0 : 1)
