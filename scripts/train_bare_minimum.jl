#!/usr/bin/env julia
"""
Bare minimum training - just embedding + dense to verify pipeline works.
"""

using Random
using Statistics
using Printf
using Lux
using Optimisers
using Zygote
using NNlib

function main()
    println("=" ^ 60)
    println("Bare Minimum Training Test")
    println("=" ^ 60)

    # Simple model: Embedding -> Dense
    vocab_size = 5000
    embed_dim = 128
    num_labels = 19

    model = Lux.Chain(
        Lux.Embedding(vocab_size => embed_dim),
        Lux.Dense(embed_dim => num_labels)
    )

    rng = Random.default_rng()
    params = Lux.initialparameters(rng, model)
    state = Lux.initialstates(rng, model)

    println("Model created")

    # Training params
    batch_size = 16
    seq_len = 64
    n_steps = 50
    lr = 1e-3

    # Setup optimizer
    opt = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, params)

    # Loss function
    function compute_loss(logits, labels)
        # logits: (num_labels, seq, batch)
        # labels: (seq, batch)
        log_probs = NNlib.logsoftmax(logits; dims=1)
        n_labels = size(logits, 1)
        total_loss = 0.0f0
        count = 0
        for b in 1:size(labels, 2)
            for s in 1:size(labels, 1)
                label = labels[s, b]
                if label > 0 && label <= n_labels
                    total_loss -= log_probs[label, s, b]
                    count += 1
                end
            end
        end
        return count > 0 ? total_loss / count : 0.0f0
    end

    println("\nStarting training...")
    println("-" ^ 60)

    total_time = 0.0
    for step in 1:n_steps
        t0 = time()

        # Generate batch
        token_ids = rand(1:vocab_size, seq_len, batch_size)
        label_ids = rand(1:num_labels, seq_len, batch_size)

        # Training step
        (loss, _), grads = Zygote.withgradient(params) do p
            logits, st = model(token_ids, p, state)
            l = compute_loss(logits, label_ids)
            return l, st
        end

        # Update
        opt_state, params = Optimisers.update(opt_state, params, grads[1])

        step_time = time() - t0
        total_time += step_time

        if step == 1 || step % 10 == 0
            avg_time = total_time / step
            @printf("Step %3d: loss=%.4f, time=%.3fs, avg=%.3fs/step\n",
                    step, loss, step_time, avg_time)
        end
    end

    println("-" ^ 60)
    @printf("Done! Total: %.1fs, Avg: %.3fs/step\n", total_time, total_time / n_steps)
    println("=" ^ 60)
    println("SUCCESS!")
end

main()
