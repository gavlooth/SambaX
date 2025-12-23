#!/usr/bin/env julia
"""Test the trained OssammaNER model"""

using Random
using Serialization
using Printf
using JSON3

# Load Ossamma with full dependencies
include(joinpath(@__DIR__, "..", "src", "Ossamma.jl"))
using .Ossamma
using .Ossamma: OssammaNER, NERConfig, ID_TO_LABEL, LABEL_TO_ID

using Lux
using LuxCUDA
using CUDA

println("=" ^ 60)
println("OssammaNER Model Testing")
println("=" ^ 60)

# Load vocab separately
vocab_path = "checkpoints/ner_110m/vocab.json"
println("\nLoading vocabulary: $vocab_path")
vocab = JSON3.read(read(vocab_path, String), Dict{String,Int})
println("  Vocab size: $(length(vocab))")

# Load checkpoint with CUDA context
checkpoint_path = "checkpoints/ner_110m/checkpoint_final.jls"
println("\nLoading checkpoint: $checkpoint_path")

# Use GPU device for loading
device = gpu_device()
cpu_dev = cpu_device()

checkpoint = open(checkpoint_path, "r") do io
    deserialize(io)
end

println("  Loaded from step: $(checkpoint[:step])")
config = checkpoint[:config]

# Create model for inference
println("\nCreating model for inference...")
ner_config = NERConfig(
    vocab_size = length(vocab),
    max_sequence_length = config.max_sequence_length,
    embedding_dimension = config.embedding_dimension,
    number_of_heads = config.number_of_heads,
    number_of_layers = config.number_of_layers,
    time_dimension = config.time_dimension,
    state_dimension = config.state_dimension,
    window_size = config.window_size,
    dropout_rate = 0.0f0,  # No dropout for inference
)
model = OssammaNER(ner_config)

# Get parameters - they should already be on CPU from checkpoint save
params = checkpoint[:params]
state = checkpoint[:state]

# Set to test mode
rng = Random.default_rng()
state = Lux.testmode(state)

# Create reverse vocab for decoding
id_to_token = Dict(v => k for (k, v) in vocab)

# Test sentences
test_sentences = [
    "John Smith works at Google in New York City.",
    "The Amazon rainforest is located in South America.",
    "Dr. Emily Chen published a paper on quantum computing.",
    "Microsoft announced a new partnership with OpenAI.",
    "The Eiffel Tower in Paris attracts millions of visitors.",
]

println("\n" * "=" ^ 60)
println("Running Inference (CPU)")
println("=" ^ 60)

for (i, sentence) in enumerate(test_sentences)
    println("\n[$i] Input: \"$sentence\"")
    
    # Tokenize (simple whitespace + punctuation split, lowercase)
    raw_tokens = split(sentence, r"[\s]+", keepempty=false)
    tokens = String[]
    for t in raw_tokens
        # Remove trailing punctuation for lookup, keep original
        clean = replace(lowercase(t), r"[.,!?;:]$" => "")
        push!(tokens, clean)
    end
    
    # Convert to IDs
    unk_id = get(vocab, "<UNK>", get(vocab, "[UNK]", 1))
    pad_id = get(vocab, "<PAD>", get(vocab, "[PAD]", 0))
    token_ids = [get(vocab, t, unk_id) for t in tokens]
    
    # Pad to sequence length
    seq_len = min(length(token_ids), config.max_sequence_length)
    padded_ids = fill(pad_id, config.max_sequence_length)
    padded_ids[1:seq_len] = token_ids[1:seq_len]
    
    # Create batch (seq_len, batch_size)
    input_batch = reshape(padded_ids, :, 1)
    
    # Run inference
    try
        (emissions, boundary_logits), _ = model(input_batch, params, state)
        
        # Get predictions (argmax over label dimension)
        # emissions shape: (num_labels, seq_len, batch)
        preds = vec(mapslices(argmax, Array(emissions[:, 1:seq_len, 1]), dims=1))
        
        # Display results
        println("    Tokens: ", join(tokens[1:seq_len], " | "))
        labels = [get(ID_TO_LABEL, p, "O") for p in preds]
        println("    Labels: ", join(labels, " | "))
        
        # Show entities found
        entities = String[]
        current_entity = ""
        current_type = ""
        
        for (j, (token, label)) in enumerate(zip(tokens[1:seq_len], labels))
            if startswith(label, "B-")
                if !isempty(current_entity)
                    push!(entities, "$current_entity [$current_type]")
                end
                current_entity = token
                current_type = label[3:end]
            elseif startswith(label, "I-") && !isempty(current_entity)
                current_entity *= " " * token
            else
                if !isempty(current_entity)
                    push!(entities, "$current_entity [$current_type]")
                    current_entity = ""
                    current_type = ""
                end
            end
        end
        if !isempty(current_entity)
            push!(entities, "$current_entity [$current_type]")
        end
        
        if !isempty(entities)
            println("    Entities: ", join(entities, ", "))
        else
            println("    Entities: (none detected)")
        end
    catch e
        println("    Error: $e")
    end
end

println("\n" * "=" ^ 60)
println("Testing Complete!")
println("=" ^ 60)
