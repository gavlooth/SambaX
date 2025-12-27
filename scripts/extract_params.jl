#!/usr/bin/env julia
"""
Extract model parameters from checkpoint for ONNX export.
This script extracts only the params field, avoiding deserialization of complex types.
"""

using Serialization
using NPZ
using Lux
using LuxCUDA
using CUDA
using JSON3

# Load project modules to get TrainingConfig and other types
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Ossamma
using Ossamma.Training: TrainingConfig

# Define a function to recursively extract arrays
function extract_arrays(data, prefix="")
    result = Dict{String, Array}()

    if data isa NamedTuple
        for k in keys(data)
            v = data[k]
            key = prefix == "" ? string(k) : prefix * "." * string(k)
            merge!(result, extract_arrays(v, key))
        end
    elseif data isa AbstractDict
        for (k, v) in data
            key = prefix == "" ? string(k) : prefix * "." * string(k)
            merge!(result, extract_arrays(v, key))
        end
    elseif data isa AbstractArray{<:Number}
        # Convert CUDA arrays to CPU
        arr = data isa CUDA.CuArray ? Array(data) : Array(data)
        result[prefix] = arr
    elseif data isa Number
        result[prefix] = [data]
    end

    return result
end

function main()
    if length(ARGS) < 2
        println("Usage: julia extract_params.jl <checkpoint.jls> <output.npz>")
        return
    end

    checkpoint_path = ARGS[1]
    output_path = ARGS[2]

    println("Loading checkpoint: $checkpoint_path")

    # Use try-catch to handle deserialization issues
    data = try
        deserialize(checkpoint_path)
    catch e
        println("Error during deserialization: $e")
        println("Trying partial extraction...")

        # Try loading file and manually extracting
        open(checkpoint_path, "r") do io
            s = Serializer(io)
            deserialize(s)
        end
    end

    println("Checkpoint type: $(typeof(data))")
    println("Keys: $(keys(data))")

    # Extract params
    params = if haskey(data, :params)
        data[:params]
    elseif data isa NamedTuple && hasproperty(data, :params)
        data.params
    else
        error("Cannot find params in checkpoint")
    end

    println("Extracting arrays from params...")
    arrays = extract_arrays(params)

    println("Found $(length(arrays)) parameter arrays:")
    total_params = 0
    for (k, v) in sort(collect(arrays), by=x->x[1])
        size_str = join(size(v), " × ")
        println("  $k: $size_str ($(length(v)) elements)")
        total_params += length(v)
    end
    println("\nTotal parameters: $(total_params) (~$(round(total_params / 1e6, digits=1))M)")

    println("\nSaving to: $output_path")
    NPZ.npzwrite(output_path, arrays)

    # Also extract config if available
    if haskey(data, :config)
        config = data[:config]
        println("Config found: $(typeof(config))")

        # Try to save config
        if config isa NamedTuple
            config_path = replace(output_path, ".npz" => ".config.json")
            println("Saving config to: $config_path")

            # Convert NamedTuple to Dict
            config_dict = Dict{Symbol, Any}()
            for k in keys(config)
                v = config[k]
                if v isa Number || v isa String || v isa Bool
                    config_dict[k] = v
                end
            end

            open(config_path, "w") do f
                JSON3.pretty(f, config_dict)
            end
        end
    end

    println("\nDone!")
end

main()
