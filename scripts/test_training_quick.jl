#!/usr/bin/env julia
"""
Quick test of the training script with a tiny model.
"""

using Random
using Statistics
using Printf
using Dates
using JSON3
using Serialization

# Load Ossamma modules
include(joinpath(@__DIR__, "..", "src", "Ossamma.jl"))
using .Ossamma
using .Ossamma: OssammaNER, NERConfig, tiny_ner
using .Ossamma: ner_cross_entropy
using .Ossamma: LABEL_TO_ID, NUM_LABELS

using Lux
using Optimisers
using Zygote

# Generate synthetic data
function generate_test_data(n_samples::Int)
    data = []
    for _ in 1:n_samples
        seq_len = rand(10:32)
        tokens = ["word_$i" for i in rand(1:100, seq_len)]
        tags = [rand() < 0.1 ? rand(["B-PERSON", "I-PERSON", "B-PLACE"]) : "O" for _ in 1:seq_len]
        push!(data, (tokens = tokens, ner_tags = tags))
    end
    return data
end

function build_vocab(data)
    vocab = Dict{String, Int}("[PAD]" => 1, "[UNK]" => 2)
    idx = 3
    for ex in data
        for t in ex.tokens
            if !haskey(vocab, t)
                vocab[t] = idx
                idx += 1
            end
        end
    end
    return vocab
end

function prepare_batch(examples, vocab, max_len)
    batch_size = length(examples)
    token_ids = ones(Int, max_len, batch_size)
    label_ids = fill(-100, max_len, batch_size)

    for (i, ex) in enumerate(examples)
        seq_len = min(length(ex.tokens), max_len)
        for j in 1:seq_len
            token_ids[j, i] = get(vocab, ex.tokens[j], 2)
            label_ids[j, i] = get(LABEL_TO_ID, String(ex.ner_tags[j]), 1)
        end
    end
    return token_ids, label_ids
end

println("=" ^ 60)
println("Quick Training Test")
println("=" ^ 60)

# Generate data
println("\nGenerating synthetic data...")
train_data = generate_test_data(100)
vocab = build_vocab(train_data)
println("  Samples: $(length(train_data))")
println("  Vocab: $(length(vocab))")

# Create tiny model
println("\nCreating tiny model...")
model = tiny_ner(vocab_size=length(vocab), max_sequence_length=32)

rng = Random.default_rng()
params = Lux.initialparameters(rng, model)
state = Lux.initialstates(rng, model)

# Count params
function count_params(p)
    total = 0
    if p isa NamedTuple || p isa Tuple
        for v in values(p)
            total += count_params(v)
        end
    elseif p isa AbstractArray
        total += length(p)
    end
    return total
end
println("  Parameters: $(count_params(params))")

# Setup optimizer
opt = Optimisers.Adam(1e-3)
opt_state = Optimisers.setup(opt, params)

# Training loop
println("\nTraining for 20 steps...")
batch_size = 8
max_len = 32

for step in 1:20
    # Sample batch
    batch_idx = rand(1:length(train_data), batch_size)
    batch = train_data[batch_idx]
    token_ids, label_ids = prepare_batch(batch, vocab, max_len)

    # Training step
    (loss, new_state), grads = Zygote.withgradient(params) do p
        logits, st = model(token_ids, p, state)
        l = ner_cross_entropy(logits, label_ids)
        return l, st
    end

    global opt_state, params = Optimisers.update(opt_state, params, grads[1])
    global state = new_state

    if step % 5 == 0
        @printf("Step %2d | Loss: %.4f\n", step, loss)
    end
end

# Test checkpoint save/load
println("\nTesting checkpoint save/load...")
checkpoint_path = "/tmp/test_checkpoint.jls"
serialize(checkpoint_path, Dict(:params => params, :state => state, :step => 20))
loaded = deserialize(checkpoint_path)
println("  Checkpoint saved and loaded successfully")
rm(checkpoint_path)

println("\n" * "=" ^ 60)
println("All tests passed!")
println("=" ^ 60)
