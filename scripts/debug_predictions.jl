#!/usr/bin/env julia
"""
Debug NER model predictions
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Ossamma
using Ossamma: OssammaNER, NERConfig
using Ossamma.NER: ID_TO_LABEL, LABEL_TO_ID
using Lux
using Serialization
using JSON3

# Define TrainingConfig for deserialization
Base.@kwdef mutable struct TrainingConfig
    vocab_size::Int = 32000
    max_sequence_length::Int = 128
    embedding_dimension::Int = 256
    number_of_heads::Int = 4
    number_of_layers::Int = 4
    time_dimension::Int = 128
    state_dimension::Int = 256
    window_size::Int = 32
    dropout_rate::Float32 = 0.1f0
    batch_size::Int = 32
    gradient_accumulation_steps::Int = 2
    learning_rate::Float64 = 2e-4
    min_learning_rate::Float64 = 1e-6
    warmup_steps::Int = 500
    total_steps::Int = 10000
    gradient_clip::Float64 = 1.0
    weight_decay::Float64 = 0.01
    eval_every::Int = 500
    log_every::Int = 50
    save_every::Int = 2000
    push_every::Int = 5000
    data_dir::String = "data/ner"
    checkpoint_dir::String = "checkpoints/ner_110m"
    git_token::String = ""
    git_remote::String = "origin"
    git_branch::String = "master"
    use_gpu::Bool = true
end

function main()
    # Load vocab
    vocab = JSON3.read(read("checkpoints/ner_110m/vocab.json", String), Dict{String,Int})
    println("Vocab size: $(length(vocab))")

    # Show sample vocab entries
    println("\nSample vocab entries:")
    for (i, (k, v)) in enumerate(vocab)
        i > 15 && break
        println("  \"$k\" => $v")
    end

    # Check for special tokens
    println("\n[UNK]: ", get(vocab, "[UNK]", "NOT FOUND"))
    println("<UNK>: ", get(vocab, "<UNK>", "NOT FOUND"))
    println("[PAD]: ", get(vocab, "[PAD]", "NOT FOUND"))
    println("<PAD>: ", get(vocab, "<PAD>", "NOT FOUND"))

    # Load checkpoint
    data = deserialize("checkpoints/ner_110m/checkpoint_best.jls")
    params = data[:params]
    state = Lux.testmode(data[:state])

    # Config
    cfg = NERConfig(
        vocab_size = length(vocab),
        max_sequence_length = 256,
        embedding_dimension = 640,
        number_of_heads = 10,
        number_of_layers = 10,
        time_dimension = 192,
        state_dimension = 640,
        window_size = 32,
        dropout_rate = 0.0f0,
        use_ffn = true,
        ffn_expansion = 1.334375f0,
    )

    model = OssammaNER(cfg)

    # Test tokenization and prediction
    text = "John Smith works at Google."
    unk_id = get(vocab, "[UNK]", get(vocab, "<UNK>", 2))

    tokens = split(text)
    token_ids = [get(vocab, lowercase(String(t)), get(vocab, String(t), unk_id)) for t in tokens]
    println("\nText: $text")
    println("Tokens: $tokens")
    println("Token IDs: $token_ids")
    println("UNK count: $(count(==(unk_id), token_ids))")

    # Pad and run model
    pad_id = 1
    seq_len = length(token_ids)
    padded = fill(pad_id, 256)
    padded[1:seq_len] = token_ids
    input_batch = reshape(padded, :, 1)

    (emissions, _), _ = model(input_batch, params, state)

    # Check predictions
    preds = vec(mapslices(argmax, Array(emissions[:, 1:seq_len, 1]), dims=1))
    labels = [get(ID_TO_LABEL, p, "O") for p in preds]

    println("\nPredictions (raw argmax indices): $preds")
    println("Labels: $labels")

    println("\nID_TO_LABEL mapping:")
    for (k, v) in sort(collect(ID_TO_LABEL), by=first)
        println("  $k => \"$v\"")
    end

    println("\nEmission values for first token (all 19 classes):")
    for (i, v) in enumerate(emissions[:, 1, 1])
        label = get(ID_TO_LABEL, i, "?")
        println("  $i ($label): $v")
    end
end

main()
