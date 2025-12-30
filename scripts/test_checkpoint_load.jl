#!/usr/bin/env julia
"""
Test checkpoint loading - Must be run from project root
"""

# Load the training script to define TrainingConfig in Main
include(joinpath(@__DIR__, "train_ner_production.jl"))

using Serialization

path = get(ARGS, 1, "checkpoints/ner_110m/checkpoint_best.jls")
println("Loading: $path")

try
    data = deserialize(path)
    println("Loaded successfully")
    println("Type: $(typeof(data))")
    if isa(data, Dict)
        println("Keys: $(keys(data))")
        if haskey(data, :step)
            println("Step: $(data[:step])")
        end
        if haskey(data, :config)
            cfg = data[:config]
            println("\nConfig type: $(typeof(cfg))")
            for fn in fieldnames(typeof(cfg))
                val = getfield(cfg, fn)
                println("  $fn: $val")
            end
        end
    else
        println("Value: $data")
    end
catch e
    println("Error: $e")
    println("\nStacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end
