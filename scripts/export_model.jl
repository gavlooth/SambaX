#!/usr/bin/env julia
"""
NER Model Export Script

Exports trained NER models for production deployment:
- Removes training-only components (dropout in training mode)
- Serializes model weights and config
- Generates checksums for integrity verification
- Creates metadata for deployment tracking

Usage:
    julia --project=. scripts/export_model.jl checkpoint.jls output_dir/

Options:
    --include-optimizer    Include optimizer state (for resuming training)
    --format jls|bson      Output format (default: jls)
    --validate             Run validation after export
"""

using Serialization
using SHA
using JSON3
using Dates
using Random

# Import project modules
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Ossamma
using Ossamma.NER: RAG_LABELS, ENTITY_TYPES, ID_TO_LABEL, LABEL_TO_ID, NUM_LABELS

# =============================================================================
# Configuration
# =============================================================================

const EXPORT_VERSION = "1.0.0"

Base.@kwdef struct ExportConfig
    checkpoint_path::String
    output_dir::String
    include_optimizer::Bool = false
    format::Symbol = :jls  # :jls or :bson
    validate::Bool = true
    model_name::String = "ossamma-ner"
end

# =============================================================================
# Checkpoint Loading
# =============================================================================

"""
Load a training checkpoint.
"""
function load_checkpoint(path::String)
    println("Loading checkpoint: $path")

    if !isfile(path)
        error("Checkpoint not found: $path")
    end

    data = deserialize(path)

    # Handle different checkpoint formats
    if haskey(data, :params) && haskey(data, :state)
        return data
    elseif haskey(data, :model_params)
        # Legacy format
        return (
            params = data.model_params,
            state = get(data, :model_state, (;)),
            config = get(data, :config, nothing),
            epoch = get(data, :epoch, 0),
            step = get(data, :step, 0),
            metrics = get(data, :metrics, nothing),
            optimizer_state = get(data, :optimizer_state, nothing),
        )
    else
        return data
    end
end

# =============================================================================
# Model Preparation
# =============================================================================

"""
Prepare model parameters for inference (remove training artifacts).
"""
function prepare_for_inference(params::NamedTuple)
    # For inference, we keep all parameters but the model should be
    # run in inference mode (dropout disabled via state)
    return params
end

"""
Prepare model state for inference (set inference mode).
"""
function prepare_state_for_inference(state::NamedTuple)
    # Recursively set training=false for dropout layers
    function set_inference_mode(s)
        if s isa NamedTuple
            new_fields = Dict{Symbol, Any}()
            for field in keys(s)
                val = getfield(s, field)
                if field == :training
                    new_fields[field] = Val(false)
                elseif field == :rng
                    # Keep RNG for any stochastic inference
                    new_fields[field] = val
                else
                    new_fields[field] = set_inference_mode(val)
                end
            end
            return NamedTuple{Tuple(keys(new_fields)...)}(values(new_fields))
        else
            return s
        end
    end

    return set_inference_mode(state)
end

# =============================================================================
# Export Functions
# =============================================================================

"""
Compute SHA256 checksum of a file.
"""
function compute_checksum(filepath::String)
    return bytes2hex(sha256(read(filepath)))
end

"""
Export model to production format.
"""
function export_model(config::ExportConfig)
    println("\n" * "=" ^ 60)
    println("Exporting NER Model for Production")
    println("=" ^ 60)

    # Load checkpoint
    checkpoint = load_checkpoint(config.checkpoint_path)

    # Create output directory
    if !isdir(config.output_dir)
        mkpath(config.output_dir)
        println("Created output directory: $(config.output_dir)")
    end

    # Prepare for inference
    println("\nPreparing model for inference...")
    inference_params = prepare_for_inference(checkpoint.params)
    inference_state = haskey(checkpoint, :state) ?
        prepare_state_for_inference(checkpoint.state) : (;)

    # Export model weights
    weights_filename = "model_weights.$(config.format)"
    weights_path = joinpath(config.output_dir, weights_filename)

    println("Saving model weights to: $weights_path")

    weights_data = (
        params = inference_params,
        state = inference_state,
    )

    if config.format == :jls
        serialize(weights_path, weights_data)
    elseif config.format == :bson
        # Would require BSON.jl
        error("BSON format not yet implemented. Use :jls format.")
    else
        error("Unknown format: $(config.format)")
    end

    weights_checksum = compute_checksum(weights_path)
    weights_size = filesize(weights_path)

    println("  Size: $(round(weights_size / 1024 / 1024, digits=2)) MB")
    println("  Checksum: $weights_checksum")

    # Export config
    model_config = haskey(checkpoint, :config) ? checkpoint.config : nothing
    if model_config !== nothing
        config_path = joinpath(config.output_dir, "config.json")
        println("\nSaving model config to: $config_path")

        # Convert config to dict for JSON serialization
        if model_config isa NamedTuple
            config_dict = Dict(pairs(model_config))
        else
            config_dict = Dict(
                k => getfield(model_config, k)
                for k in fieldnames(typeof(model_config))
            )
        end

        open(config_path, "w") do f
            JSON3.pretty(f, config_dict)
        end
    end

    # Export label schema
    labels_path = joinpath(config.output_dir, "labels.json")
    println("Saving label schema to: $labels_path")

    labels_data = Dict(
        "labels" => RAG_LABELS,
        "num_labels" => NUM_LABELS,
        "entity_types" => ENTITY_TYPES,
        "label_to_id" => LABEL_TO_ID,
    )

    open(labels_path, "w") do f
        JSON3.pretty(f, labels_data)
    end

    # Optionally export optimizer state
    if config.include_optimizer && haskey(checkpoint, :optimizer_state)
        opt_path = joinpath(config.output_dir, "optimizer_state.jls")
        println("Saving optimizer state to: $opt_path")
        serialize(opt_path, checkpoint.optimizer_state)
    end

    # Generate metadata
    metadata = Dict(
        "model_type" => "OssammaNER",
        "model_name" => config.model_name,
        "export_version" => EXPORT_VERSION,
        "num_labels" => NUM_LABELS,
        "label_schema" => "RAG-9",
        "weights_file" => weights_filename,
        "weights_checksum" => weights_checksum,
        "weights_size_bytes" => weights_size,
        "format" => string(config.format),
        "exported_at" => string(now()),
        "source_checkpoint" => basename(config.checkpoint_path),
    )

    # Add training info if available
    if haskey(checkpoint, :epoch)
        metadata["trained_epochs"] = checkpoint.epoch
    end
    if haskey(checkpoint, :step)
        metadata["trained_steps"] = checkpoint.step
    end
    if haskey(checkpoint, :metrics) && checkpoint.metrics !== nothing
        metadata["final_metrics"] = checkpoint.metrics
    end

    metadata_path = joinpath(config.output_dir, "metadata.json")
    println("Saving metadata to: $metadata_path")

    open(metadata_path, "w") do f
        JSON3.pretty(f, metadata)
    end

    # Validation
    if config.validate
        println("\nValidating exported model...")
        validate_export(config.output_dir, weights_checksum)
    end

    println("\n" * "=" ^ 60)
    println("Export Complete!")
    println("=" ^ 60)
    println("\nExported files:")
    for f in readdir(config.output_dir)
        size = filesize(joinpath(config.output_dir, f))
        println("  $f ($(round(size / 1024, digits=1)) KB)")
    end

    println("\nTo load the exported model:")
    println("  weights = deserialize(\"$(joinpath(config.output_dir, weights_filename))\")")
    println("  model = OssammaNER(config)")
    println("  # Use weights.params and weights.state for inference")

    return config.output_dir
end

# =============================================================================
# Validation
# =============================================================================

"""
Validate an exported model.
"""
function validate_export(output_dir::String, expected_checksum::String)
    println("  Checking file integrity...")

    # Load weights and verify checksum
    weights_path = joinpath(output_dir, "model_weights.jls")
    if !isfile(weights_path)
        error("Weights file not found: $weights_path")
    end

    actual_checksum = compute_checksum(weights_path)
    if actual_checksum != expected_checksum
        error("Checksum mismatch! Expected: $expected_checksum, Got: $actual_checksum")
    end
    println("  ✓ Checksum verified")

    # Try loading weights
    println("  Loading weights...")
    weights = deserialize(weights_path)

    if !haskey(weights, :params)
        error("Invalid weights format: missing 'params' field")
    end
    println("  ✓ Weights loadable")

    # Check labels
    labels_path = joinpath(output_dir, "labels.json")
    if isfile(labels_path)
        labels = JSON3.read(read(labels_path, String))
        if length(labels.labels) != NUM_LABELS
            @warn "Label count mismatch: expected $NUM_LABELS, got $(length(labels.labels))"
        end
        println("  ✓ Labels valid")
    end

    # Check metadata
    metadata_path = joinpath(output_dir, "metadata.json")
    if isfile(metadata_path)
        metadata = JSON3.read(read(metadata_path, String))
        println("  ✓ Metadata valid")
    end

    println("  ✓ All validation checks passed!")
    return true
end

# =============================================================================
# Batch Export
# =============================================================================

"""
Export multiple checkpoints (e.g., for model comparison).
"""
function batch_export(
    checkpoint_paths::Vector{String},
    output_base_dir::String;
    kwargs...
)
    for path in checkpoint_paths
        name = splitext(basename(path))[1]
        output_dir = joinpath(output_base_dir, name)

        config = ExportConfig(
            checkpoint_path = path,
            output_dir = output_dir;
            kwargs...
        )

        try
            export_model(config)
        catch e
            @warn "Failed to export $path: $e"
        end
    end
end

# =============================================================================
# Quick Export Function
# =============================================================================

"""
    quick_export(checkpoint_path, output_dir; kwargs...)

Quick export with sensible defaults.
"""
function quick_export(checkpoint_path::String, output_dir::String; kwargs...)
    config = ExportConfig(
        checkpoint_path = checkpoint_path,
        output_dir = output_dir;
        kwargs...
    )
    return export_model(config)
end

# =============================================================================
# Main
# =============================================================================

function print_usage()
    println("""
NER Model Export Script

Usage:
    julia --project=. scripts/export_model.jl <checkpoint.jls> <output_dir/>

Options:
    --include-optimizer    Include optimizer state for training resumption
    --format <jls|bson>    Output format (default: jls)
    --no-validate          Skip validation after export
    --name <model-name>    Model name for metadata

Examples:
    julia --project=. scripts/export_model.jl checkpoints/ner_best.jls models/ner_v1/
    julia --project=. scripts/export_model.jl ckpt.jls export/ --include-optimizer
""")
end

function main()
    if length(ARGS) < 2
        print_usage()
        return
    end

    checkpoint_path = ARGS[1]
    output_dir = ARGS[2]

    # Parse options
    include_optimizer = "--include-optimizer" in ARGS
    validate = !("--no-validate" in ARGS)

    format = :jls
    format_idx = findfirst(==("--format"), ARGS)
    if format_idx !== nothing && format_idx < length(ARGS)
        format = Symbol(ARGS[format_idx + 1])
    end

    model_name = "ossamma-ner"
    name_idx = findfirst(==("--name"), ARGS)
    if name_idx !== nothing && name_idx < length(ARGS)
        model_name = ARGS[name_idx + 1]
    end

    config = ExportConfig(
        checkpoint_path = checkpoint_path,
        output_dir = output_dir,
        include_optimizer = include_optimizer,
        format = format,
        validate = validate,
        model_name = model_name,
    )

    export_model(config)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

# Export functions for use as module
export ExportConfig, export_model, quick_export, batch_export
export load_checkpoint, validate_export
