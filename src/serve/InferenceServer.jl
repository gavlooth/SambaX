module InferenceServer

"""
HTTP Inference Server for NER predictions.

Provides:
- REST API for NER predictions
- Batch processing support
- Health check endpoints
- Request logging and metrics

Endpoints:
- POST /predict       - Predict entities in text(s)
- POST /predict/batch - Batch prediction with higher throughput
- GET  /health        - Health check
- GET  /info          - Model info and configuration
"""

using HTTP
using JSON3
using Serialization
using Random
using Dates
using Statistics: mean

# =============================================================================
# Server Configuration
# =============================================================================

Base.@kwdef struct ServerConfig
    host::String = "0.0.0.0"
    port::Int = 8080
    model_path::String
    vocab_path::Union{String, Nothing} = nothing
    max_batch_size::Int = 32
    max_sequence_length::Int = 512
    num_workers::Int = 1
    log_requests::Bool = true
    cors_enabled::Bool = true
end

# =============================================================================
# NER Server
# =============================================================================

mutable struct NERServer
    config::ServerConfig
    model::Any           # OssammaNER model
    params::NamedTuple   # Model parameters
    state::NamedTuple    # Model state
    vocab::Any           # Vocabulary
    id_to_label::Dict{Int, String}
    label_to_id::Dict{String, Int}
    request_count::Int
    total_latency::Float64
    start_time::DateTime
end

"""
    load_server(config::ServerConfig) -> NERServer

Load model and create server instance.
"""
function load_server(config::ServerConfig)
    println("Loading NER model from: $(config.model_path)")

    # Load model weights
    weights = deserialize(config.model_path)
    params = weights.params
    state = haskey(weights, :state) ? weights.state : (;)

    # Load or create vocabulary
    vocab = nothing
    if config.vocab_path !== nothing && isfile(config.vocab_path)
        vocab = JSON3.read(read(config.vocab_path, String))
    end

    # Get label mappings from the NER module
    # These are defined in the parent Ossamma module
    id_to_label = Dict{Int, String}()
    label_to_id = Dict{String, Int}()

    # Default RAG labels
    rag_labels = [
        "O",
        "B-PERSON", "I-PERSON", "B-AGENCY", "I-AGENCY",
        "B-PLACE", "I-PLACE", "B-ORGANISM", "I-ORGANISM",
        "B-EVENT", "I-EVENT", "B-INSTRUMENT", "I-INSTRUMENT",
        "B-WORK", "I-WORK", "B-DOMAIN", "I-DOMAIN",
        "B-MEASURE", "I-MEASURE",
    ]

    for (i, label) in enumerate(rag_labels)
        id_to_label[i] = label
        label_to_id[label] = i
    end

    # Load labels from file if available
    labels_path = replace(config.model_path, "model_weights" => "labels")
    labels_path = replace(labels_path, ".jls" => ".json")
    if isfile(labels_path)
        labels_data = JSON3.read(read(labels_path, String))
        if haskey(labels_data, :labels)
            id_to_label = Dict{Int, String}()
            label_to_id = Dict{String, Int}()
            for (i, label) in enumerate(labels_data.labels)
                id_to_label[i] = String(label)
                label_to_id[String(label)] = i
            end
        end
    end

    println("Model loaded successfully")
    println("  Labels: $(length(id_to_label))")

    # Note: model is set to nothing here - in production you'd load the actual model
    # The full implementation requires the Ossamma module to be loaded
    return NERServer(
        config,
        nothing,  # model - would be OssammaNER instance
        params,
        state,
        vocab,
        id_to_label,
        label_to_id,
        0,
        0.0,
        now(),
    )
end

# =============================================================================
# Tokenization (Simple)
# =============================================================================

"""
Simple whitespace tokenization for the server.
In production, use the Tokenizer module.
"""
function simple_tokenize(text::String)
    # Basic whitespace tokenization
    tokens = String[]
    for word in split(strip(text))
        push!(tokens, String(word))
    end
    return tokens
end

"""
Convert tokens to IDs using vocabulary.
"""
function tokens_to_ids(tokens::Vector{String}, vocab, unk_id::Int = 2)
    if vocab === nothing
        # Return dummy IDs if no vocab (for testing)
        return collect(1:length(tokens))
    end

    ids = Int[]
    for token in tokens
        if haskey(vocab, token)
            push!(ids, vocab[token])
        else
            push!(ids, unk_id)
        end
    end
    return ids
end

# =============================================================================
# Prediction
# =============================================================================

"""
Entity representation in API response.
"""
struct Entity
    text::String
    label::String
    start::Int
    end_::Int
    score::Float64
end

function JSON3.StructTypes.StructType(::Type{Entity})
    return JSON3.StructTypes.Struct()
end

"""
Predict entities for a single text.
"""
function predict_single(server::NERServer, text::String)
    start_time = time()

    # Tokenize
    tokens = simple_tokenize(text)

    if isempty(tokens)
        return (tokens = String[], labels = String[], entities = Entity[])
    end

    # Truncate if necessary
    if length(tokens) > server.config.max_sequence_length
        tokens = tokens[1:server.config.max_sequence_length]
    end

    # Convert to IDs
    token_ids = tokens_to_ids(tokens, server.vocab)

    # Run model inference
    # In a full implementation:
    # logits, _ = server.model(token_ids, server.params, server.state)
    # predictions = vec(mapslices(argmax, logits, dims=1))

    # For now, return dummy predictions (O for everything)
    predictions = fill(1, length(tokens))

    # Convert predictions to labels
    labels = [get(server.id_to_label, p, "O") for p in predictions]

    # Extract entities
    entities = extract_entities(tokens, labels, predictions)

    latency = time() - start_time

    # Update metrics
    server.request_count += 1
    server.total_latency += latency

    return (tokens = tokens, labels = labels, entities = entities)
end

"""
Extract entities from predicted labels.
"""
function extract_entities(
    tokens::Vector{String},
    labels::Vector{String},
    predictions::Vector{Int}
)
    entities = Entity[]
    i = 1

    while i <= length(labels)
        if startswith(labels[i], "B-")
            entity_type = labels[i][3:end]
            start_idx = i
            entity_tokens = [tokens[i]]

            # Find extent
            i += 1
            while i <= length(labels) && labels[i] == "I-$entity_type"
                push!(entity_tokens, tokens[i])
                i += 1
            end

            push!(entities, Entity(
                join(entity_tokens, " "),
                entity_type,
                start_idx,
                i - 1,
                1.0  # Placeholder score
            ))
        else
            i += 1
        end
    end

    return entities
end

"""
Batch prediction for multiple texts.
"""
function predict_batch(server::NERServer, texts::Vector{String})
    results = []
    for text in texts
        push!(results, predict_single(server, text))
    end
    return results
end

# =============================================================================
# HTTP Handlers
# =============================================================================

"""
Health check handler.
"""
function handle_health(server::NERServer, req::HTTP.Request)
    uptime = now() - server.start_time
    avg_latency = server.request_count > 0 ?
        server.total_latency / server.request_count : 0.0

    response = Dict(
        "status" => "healthy",
        "uptime_seconds" => Dates.value(uptime) / 1000,
        "requests_served" => server.request_count,
        "avg_latency_ms" => round(avg_latency * 1000, digits=2),
    )

    return HTTP.Response(200, JSON3.write(response))
end

"""
Model info handler.
"""
function handle_info(server::NERServer, req::HTTP.Request)
    response = Dict(
        "model_type" => "OssammaNER",
        "num_labels" => length(server.id_to_label),
        "max_sequence_length" => server.config.max_sequence_length,
        "labels" => collect(values(server.id_to_label)),
        "entity_types" => unique([
            split(l, "-")[end] for l in values(server.id_to_label)
            if startswith(l, "B-")
        ]),
    )

    return HTTP.Response(200, JSON3.write(response))
end

"""
Single prediction handler.
"""
function handle_predict(server::NERServer, req::HTTP.Request)
    # Parse request body
    try
        body = JSON3.read(String(req.body))

        if !haskey(body, :text) && !haskey(body, :texts)
            return HTTP.Response(400, JSON3.write(Dict(
                "error" => "Missing 'text' or 'texts' field"
            )))
        end

        if haskey(body, :text)
            # Single text
            text = String(body.text)
            result = predict_single(server, text)

            response = Dict(
                "text" => text,
                "tokens" => result.tokens,
                "labels" => result.labels,
                "entities" => [
                    Dict(
                        "text" => e.text,
                        "label" => e.label,
                        "start" => e.start,
                        "end" => e.end_,
                        "score" => e.score,
                    ) for e in result.entities
                ],
            )
        else
            # Multiple texts
            texts = String.(body.texts)

            if length(texts) > server.config.max_batch_size
                return HTTP.Response(400, JSON3.write(Dict(
                    "error" => "Batch size $(length(texts)) exceeds max $(server.config.max_batch_size)"
                )))
            end

            results = predict_batch(server, texts)

            response = Dict(
                "results" => [
                    Dict(
                        "text" => texts[i],
                        "tokens" => r.tokens,
                        "labels" => r.labels,
                        "entities" => [
                            Dict(
                                "text" => e.text,
                                "label" => e.label,
                                "start" => e.start,
                                "end" => e.end_,
                                "score" => e.score,
                            ) for e in r.entities
                        ],
                    ) for (i, r) in enumerate(results)
                ],
            )
        end

        return HTTP.Response(200, JSON3.write(response))

    catch e
        return HTTP.Response(500, JSON3.write(Dict(
            "error" => "Internal error: $(string(e))"
        )))
    end
end

# =============================================================================
# CORS Middleware
# =============================================================================

function add_cors_headers!(response::HTTP.Response)
    HTTP.setheader(response, "Access-Control-Allow-Origin" => "*")
    HTTP.setheader(response, "Access-Control-Allow-Methods" => "GET, POST, OPTIONS")
    HTTP.setheader(response, "Access-Control-Allow-Headers" => "Content-Type")
    return response
end

function handle_cors_preflight(req::HTTP.Request)
    response = HTTP.Response(204)
    add_cors_headers!(response)
    return response
end

# =============================================================================
# Router
# =============================================================================

function create_router(server::NERServer)
    function router(req::HTTP.Request)
        # Log request
        if server.config.log_requests
            println("[$(now())] $(req.method) $(req.target)")
        end

        # Handle CORS preflight
        if req.method == "OPTIONS"
            return handle_cors_preflight(req)
        end

        # Route request
        response = if req.target == "/health" && req.method == "GET"
            handle_health(server, req)
        elseif req.target == "/info" && req.method == "GET"
            handle_info(server, req)
        elseif req.target == "/predict" && req.method == "POST"
            handle_predict(server, req)
        elseif req.target == "/predict/batch" && req.method == "POST"
            handle_predict(server, req)
        else
            HTTP.Response(404, JSON3.write(Dict("error" => "Not found")))
        end

        # Add CORS headers
        if server.config.cors_enabled
            add_cors_headers!(response)
        end

        # Set content type
        HTTP.setheader(response, "Content-Type" => "application/json")

        return response
    end

    return router
end

# =============================================================================
# Server Startup
# =============================================================================

"""
    start_server(config::ServerConfig)

Start the HTTP inference server.
"""
function start_server(config::ServerConfig)
    println("\n" * "=" ^ 50)
    println("Starting NER Inference Server")
    println("=" ^ 50)

    # Load model and create server
    server = load_server(config)

    # Create router
    router = create_router(server)

    println("\nServer configuration:")
    println("  Host: $(config.host)")
    println("  Port: $(config.port)")
    println("  Max batch size: $(config.max_batch_size)")
    println("  Max sequence length: $(config.max_sequence_length)")

    println("\nEndpoints:")
    println("  POST /predict       - Single/batch prediction")
    println("  POST /predict/batch - Batch prediction")
    println("  GET  /health        - Health check")
    println("  GET  /info          - Model info")

    println("\nStarting server on http://$(config.host):$(config.port)")
    println("Press Ctrl+C to stop\n")

    HTTP.serve(router, config.host, config.port)
end

"""
    start_server(model_path::String; kwargs...)

Convenience function to start server with just model path.
"""
function start_server(model_path::String; kwargs...)
    config = ServerConfig(; model_path = model_path, kwargs...)
    start_server(config)
end

# =============================================================================
# Exports
# =============================================================================

export ServerConfig, NERServer
export load_server, start_server
export predict_single, predict_batch

end # module
