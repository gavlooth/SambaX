#!/usr/bin/env julia
"""
Checkpoint compatibility loader for different versions.
"""

using Dates
using Serialization

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Ossamma
using Ossamma: OssammaNER, NERConfig
using Ossamma.NER: ID_TO_LABEL

using Lux

# Define a compat config struct that matches what was serialized
mutable struct CompatConfig
    vocab_size::Int
    max_sequence_length::Int
    embedding_dimension::Int
    number_of_heads::Int
    number_of_layers::Int
    time_dimension::Int
    state_dimension::Int
    window_size::Int
    dropout_rate::Float32
    batch_size::Int
    gradient_accumulation_steps::Int
    learning_rate::Float64
    min_learning_rate::Float64
    warmup_steps::Int
    total_steps::Int
    gradient_clip::Float64
    weight_decay::Float64
    eval_every::Int
    log_every::Int
    save_every::Int
    push_every::Int
    data_dir::String
    checkpoint_dir::String
    git_token::String
    git_remote::String
    git_branch::String
    use_gpu::Bool
    use_ffn::Bool
    ffn_expansion::Float32
end

# Custom deserializer - handle type conversion
function my_deserialize(path::String)
    open(path, "r") do f
        s = Serializer(f)

        # First read the header/version
        header = read(f, 4)
        if header != [0x37, 0x4a, 0x4c, 0x1e]
            error("Invalid JLS header")
        end
        seekstart(f)

        # Fallback to standard deserialize and catch type errors
        deserialize(s)
    end
end

function load_checkpoint_compat(path::String)
    println("Attempting to load: $path")

    try
        data = deserialize(path)
        return data
    catch e
        println("Standard deserialize failed: $e")
        println("\nAttempting fallback...")
        return nothing
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    path = get(ARGS, 1, "checkpoints/ner_110m/checkpoint_final.jls")
    result = load_checkpoint_compat(path)
    if result !== nothing
        println("Success! Type: ", typeof(result))
        if isa(result, Dict)
            println("Keys: ", keys(result))
        end
    end
end
