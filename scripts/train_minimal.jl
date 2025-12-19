#!/usr/bin/env julia
"""
Minimal training script to verify NER model works before scaling up.
"""

using Random
using Statistics
using Printf
using Dates

# Load Ossamma modules
include(joinpath(@__DIR__, "..", "src", "Ossamma.jl"))
using .Ossamma
using .Ossamma: OssammaNER, NERConfig
using .Ossamma: ner_cross_entropy
using .Ossamma: LABEL_TO_ID

using Lux
using Optimisers
using Zygote

println("=" ^ 60)
println("Minimal NER Training Test")
println("=" ^ 60)

# Ultra-minimal config
config = NERConfig(
    vocab_size = 1000,
    max_sequence_length = 32,
    embedding_dimension = 64,
    number_of_heads = 2,
    number_of_layers = 2,
    time_dimension = 32,
    state_dimension = 64,
    window_size = 16,
    dropout_rate = 0.0f0,
)

println("\nCreating minimal model...")
model = OssammaNER(config)

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
println("  Parameters: $(round(n_params / 1e6, digits=3))M ($(n_params))")

# Tiny synthetic batch
batch_size = 2
seq_len = 32
token_ids = rand(1:config.vocab_size, seq_len, batch_size)
label_ids = rand(1:19, seq_len, batch_size)  # 19 NER labels

println("\nRunning forward pass...")
t0 = time()
logits, new_state = model(token_ids, params, state)
println("  Forward pass time: $(round(time() - t0, digits=2))s")
println("  Logits shape: $(size(logits))")

println("\nComputing loss...")
t0 = time()
loss = ner_cross_entropy(logits, label_ids)
println("  Loss: $loss")
println("  Loss computation time: $(round(time() - t0, digits=2))s")

println("\nComputing gradients...")
t0 = time()
(loss_val, _), grads = Zygote.withgradient(params) do p
    logits, st = model(token_ids, p, state)
    l = ner_cross_entropy(logits, label_ids)
    return l, st
end
println("  Gradient computation time: $(round(time() - t0, digits=2))s")
println("  Loss: $loss_val")

println("\nSetting up optimizer...")
opt = Optimisers.AdamW(1e-3, (0.9, 0.999), 0.01f0)
opt_state = Optimisers.setup(opt, params)

println("\nRunning 5 training steps...")
for step in 1:5
    t0 = time()

    # Generate new batch
    token_ids = rand(1:config.vocab_size, seq_len, batch_size)
    label_ids = rand(1:19, seq_len, batch_size)

    # Training step
    (loss, new_state), grads = Zygote.withgradient(params) do p
        logits, st = model(token_ids, p, state)
        l = ner_cross_entropy(logits, label_ids)
        return l, st
    end

    # Update
    opt_state, params = Optimisers.update(opt_state, params, grads[1])
    state = new_state

    step_time = time() - t0
    @printf("  Step %d: loss=%.4f, time=%.2fs\n", step, loss, step_time)
end

println("\n" * "=" ^ 60)
println("SUCCESS! Minimal model training works.")
println("=" ^ 60)
