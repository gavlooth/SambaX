#!/usr/bin/env julia
"""
Fast training script using SimpleNER (no OSSM layer).
"""

using Random
using Statistics
using Printf
using Dates
using Serialization

include(joinpath(@__DIR__, "..", "src", "SimpleNER.jl"))
using .SimpleNER

using Lux
using Optimisers
using Zygote

function main()
    println("=" ^ 60)
    println("SimpleNER Training")
    println("=" ^ 60)

    # Configuration
    config = SimpleNERConfig(
        vocab_size = 5000,
        max_sequence_length = 128,
        embedding_dimension = 256,
        number_of_heads = 4,
        number_of_layers = 4,
        num_labels = 19,
        dropout_rate = 0.1f0,
    )

    println("\nCreating model...")
    model = SimpleNERModel(config)

    rng = Random.default_rng()
    params = Lux.initialparameters(rng, model)
    state = Lux.initialstates(rng, model)

    # Count parameters
    function count_params(p)
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
        count_nested(p)
        return total
    end

    n_params = count_params(params)
    println("  Parameters: $(round(n_params / 1e6, digits=3))M")

    # Training config
    batch_size = 8
    seq_len = 64
    n_steps = 100
    lr = 1e-3

    println("\nTraining config:")
    println("  Batch size: $batch_size")
    println("  Sequence length: $seq_len")
    println("  Steps: $n_steps")

    # Setup optimizer
    opt = Optimisers.AdamW(lr, (0.9, 0.999), 0.01f0)
    opt_state = Optimisers.setup(opt, params)

    println("\nStarting training...")
    println("-" ^ 60)

    total_time = 0.0
    for step in 1:n_steps
        t0 = time()

        # Generate batch
        token_ids = rand(1:config.vocab_size, seq_len, batch_size)
        label_ids = rand(1:19, seq_len, batch_size)

        # Training step
        (loss, _), grads = Zygote.withgradient(params) do p
            logits, st = model(token_ids, p, state)
            l = simple_ner_cross_entropy(logits, label_ids)
            return l, st
        end

        # Update
        opt_state, params = Optimisers.update(opt_state, params, grads[1])

        step_time = time() - t0
        total_time += step_time

        if step == 1 || step % 10 == 0
            avg_time = total_time / step
            @printf("Step %3d: loss=%.4f, time=%.2fs, avg=%.2fs/step\n",
                    step, loss, step_time, avg_time)
        end
    end

    println("-" ^ 60)
    @printf("Training complete! Total time: %.1fs, Avg: %.2fs/step\n",
            total_time, total_time / n_steps)

    # Save checkpoint
    checkpoint_dir = "checkpoints/simple_ner"
    isdir(checkpoint_dir) || mkpath(checkpoint_dir)
    checkpoint_path = joinpath(checkpoint_dir, "checkpoint_final.jls")
    serialize(checkpoint_path, Dict(
        :params => params,
        :state => state,
        :config => config,
        :step => n_steps,
    ))
    println("\nCheckpoint saved to: $checkpoint_path")

    println("\n" * "=" ^ 60)
    println("SUCCESS!")
    println("=" ^ 60)
end

main()
