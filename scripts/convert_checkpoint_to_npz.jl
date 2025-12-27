#!/usr/bin/env julia
"""
Convert Julia checkpoint to NPZ format for Python loading.
"""

using Serialization
using NPZ
using JSON3
using Optimisers
using Lux
using Random

# Ensure CUDA types can be deserialized
try
    using CUDA
catch
    # CUDA not available, that's fine
end

# Load Ossamma module (needed to deserialize TrainingConfig, etc.)
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Ossamma
using Ossamma.Training: TrainingConfig

function flatten_params(params, prefix="")
    result = Dict{String, Any}()

    if params isa NamedTuple
        for k in keys(params)
            v = getfield(params, k)
            key = prefix == "" ? string(k) : prefix * "." * string(k)
            merge!(result, flatten_params(v, key))
        end
    elseif params isa AbstractArray
        # Convert to CPU array if on GPU
        arr = Array(params)
        result[prefix] = arr
    elseif params isa Number
        result[prefix] = params
    else
        # Try to iterate as dict-like
        try
            for (k, v) in pairs(params)
                key = prefix == "" ? string(k) : prefix * "." * string(k)
                merge!(result, flatten_params(v, key))
            end
        catch
            @warn "Could not process: $prefix of type $(typeof(params))"
        end
    end

    return result
end

function main()
    if length(ARGS) < 2
        println("Usage: julia convert_checkpoint_to_npz.jl <checkpoint.jls> <output.npz>")
        return
    end

    checkpoint_path = ARGS[1]
    output_path = ARGS[2]

    println("Loading checkpoint: $checkpoint_path")
    checkpoint = deserialize(checkpoint_path)

    # Extract params
    params = if haskey(checkpoint, :params)
        checkpoint.params
    elseif haskey(checkpoint, :model_params)
        checkpoint.model_params
    else
        error("Unknown checkpoint format")
    end

    println("Flattening parameters...")
    flat_params = flatten_params(params)

    println("Found $(length(flat_params)) parameters")

    # Convert to NPZ-compatible format
    npz_params = Dict{String, Array}()
    for (k, v) in flat_params
        if v isa AbstractArray
            npz_params[k] = Array(v)
        elseif v isa Number
            npz_params[k] = [v]
        end
    end

    println("Saving to: $output_path")
    NPZ.npzwrite(output_path, npz_params)

    # Also save config if available
    if haskey(checkpoint, :config)
        config_path = replace(output_path, ".npz" => ".config.json")
        println("Saving config to: $config_path")
        open(config_path, "w") do f
            JSON3.pretty(f, checkpoint.config)
        end
    end

    println("Done!")
end

main()
