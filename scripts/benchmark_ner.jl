#!/usr/bin/env julia
"""
OssammaNER Benchmark Script

Tests inference speed and accuracy on sample data.
"""

using Random
using Serialization
using JSON3
using Statistics

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Ossamma
using Ossamma: OssammaNER, NERConfig
using Ossamma.NER: ID_TO_LABEL, LABEL_TO_ID

using Lux
using CUDA

# Define TrainingConfig here (must match the checkpoint's struct exactly)
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

const CHECKPOINT = "checkpoints/ner_110m/checkpoint_best.jls"
const VOCAB_PATH = "checkpoints/ner_110m/vocab.json"

# Test sentences with expected entities
const TEST_CASES = [
    (text="John Smith works at Google in New York.",
     expected=[("John Smith", "PER"), ("Google", "ORG"), ("New York", "LOC")]),
    (text="Apple Inc. announced new products in Cupertino, California.",
     expected=[("Apple Inc.", "ORG"), ("Cupertino", "LOC"), ("California", "LOC")]),
    (text="President Biden met with Chancellor Scholz in Berlin.",
     expected=[("Biden", "PER"), ("Scholz", "PER"), ("Berlin", "LOC")]),
    (text="Microsoft and Amazon are competing in cloud computing.",
     expected=[("Microsoft", "ORG"), ("Amazon", "ORG")]),
    (text="The Eiffel Tower in Paris attracts millions of tourists.",
     expected=[("Eiffel Tower", "LOC"), ("Paris", "LOC")]),
    (text="Dr. Sarah Johnson from Harvard University published a study.",
     expected=[("Sarah Johnson", "PER"), ("Harvard University", "ORG")]),
    (text="Tesla's CEO Elon Musk visited the factory in Shanghai.",
     expected=[("Tesla", "ORG"), ("Elon Musk", "PER"), ("Shanghai", "LOC")]),
    (text="The European Union headquarters is located in Brussels.",
     expected=[("European Union", "ORG"), ("Brussels", "LOC")]),
    (text="Leonardo DiCaprio starred in a film directed by Martin Scorsese.",
     expected=[("Leonardo DiCaprio", "PER"), ("Martin Scorsese", "PER")]),
    (text="The United Nations held a meeting in Geneva, Switzerland.",
     expected=[("United Nations", "ORG"), ("Geneva", "LOC"), ("Switzerland", "LOC")]),
]

# Known config for ner_110m model (from checkpoint_best.jls step 48500)
# FFN: Expand 854x640, Contract 640x427
# hidden = 854 = dim * expansion_factor → expansion_factor = 854/640 = 1.334375
const MODEL_CONFIG = (
    max_sequence_length = 256,
    embedding_dimension = 640,
    number_of_heads = 10,
    number_of_layers = 10,
    time_dimension = 192,
    state_dimension = 640,
    window_size = 32,
    use_ffn = true,
    ffn_expansion = 1.334375f0,
)

function load_model()
    println("Loading vocabulary: $VOCAB_PATH")
    vocab = JSON3.read(read(VOCAB_PATH, String), Dict{String,Int})
    id_to_token = Dict(v => k for (k, v) in vocab)
    println("  Vocab size: $(length(vocab))")

    println("Loading checkpoint: $CHECKPOINT")

    # Try to load checkpoint - may fail due to version mismatch
    local params, state, step
    try
        checkpoint = deserialize(CHECKPOINT)
        params = checkpoint[:params]
        state = Lux.testmode(checkpoint[:state])
        step = get(checkpoint, :step, 0)
        println("  Loaded from step: $step")
    catch e
        error("Failed to load checkpoint: $e\n\nThe checkpoint may have been created with a different Julia version.")
    end

    # Use known config values
    ner_config = NERConfig(
        vocab_size = length(vocab),
        max_sequence_length = MODEL_CONFIG.max_sequence_length,
        embedding_dimension = MODEL_CONFIG.embedding_dimension,
        number_of_heads = MODEL_CONFIG.number_of_heads,
        number_of_layers = MODEL_CONFIG.number_of_layers,
        time_dimension = MODEL_CONFIG.time_dimension,
        state_dimension = MODEL_CONFIG.state_dimension,
        window_size = MODEL_CONFIG.window_size,
        dropout_rate = 0.0f0,
        use_ffn = MODEL_CONFIG.use_ffn,
        ffn_expansion = MODEL_CONFIG.ffn_expansion,
    )

    model = OssammaNER(ner_config)

    return model, params, state, vocab, id_to_token, MODEL_CONFIG
end

function tokenize(text::String, vocab::Dict{String,Int})
    unk_id = get(vocab, "[UNK]", get(vocab, "<UNK>", 2))

    raw_tokens = String[]
    for word in split(text)
        m = match(r"^([.,!?;:\"'()\[\]{}]*)(.+?)([.,!?;:\"'()\[\]{}]*)$", word)
        if m !== nothing
            !isempty(m[1]) && push!(raw_tokens, m[1])
            push!(raw_tokens, m[2])
            !isempty(m[3]) && push!(raw_tokens, m[3])
        else
            push!(raw_tokens, String(word))
        end
    end

    token_ids = Int[]
    original_tokens = String[]
    for token in raw_tokens
        push!(original_tokens, token)
        token_lower = lowercase(token)
        push!(token_ids, get(vocab, token_lower, get(vocab, token, unk_id)))
    end

    return token_ids, original_tokens
end

function predict(model, params, state, token_ids::Vector{Int}, max_seq_len::Int; use_gpu=false)
    pad_id = 1
    seq_len = min(length(token_ids), max_seq_len)
    padded = fill(pad_id, max_seq_len)
    padded[1:seq_len] = token_ids[1:seq_len]

    input_batch = reshape(padded, :, 1)

    if use_gpu && CUDA.functional()
        input_batch = CuArray(input_batch)
    end

    (emissions, _), _ = model(input_batch, params, state)

    emissions_cpu = Array(emissions)
    preds = vec(mapslices(argmax, emissions_cpu[:, 1:seq_len, 1], dims=1))

    return preds
end

function format_entities(tokens::Vector{String}, labels::Vector{String})
    entities = Tuple{String, String}[]

    current_entity = String[]
    current_type = ""

    for (i, (token, label)) in enumerate(zip(tokens, labels))
        if startswith(label, "B-")
            if !isempty(current_entity)
                push!(entities, (join(current_entity, " "), current_type))
            end
            current_entity = [token]
            current_type = label[3:end]
        elseif startswith(label, "I-") && !isempty(current_entity)
            push!(current_entity, token)
        else
            if !isempty(current_entity)
                push!(entities, (join(current_entity, " "), current_type))
                current_entity = String[]
            end
        end
    end

    if !isempty(current_entity)
        push!(entities, (join(current_entity, " "), current_type))
    end

    return entities
end

function run_inference(text::String, model, params, state, vocab, max_seq_len; use_gpu=false)
    token_ids, tokens = tokenize(text, vocab)

    if isempty(token_ids)
        return String[], Tuple{String,String}[]
    end

    preds = predict(model, params, state, token_ids, max_seq_len; use_gpu=use_gpu)
    labels = [get(ID_TO_LABEL, p, "O") for p in preds]
    entities = format_entities(tokens, labels)

    return labels, entities
end

function benchmark_speed(model, params, state, vocab, max_seq_len; n_iterations=100, use_gpu=false)
    println("\n" * "="^60)
    println("SPEED BENCHMARK")
    println("="^60)

    test_texts = [tc.text for tc in TEST_CASES]

    # Warmup
    println("Warming up...")
    for text in test_texts[1:min(3, length(test_texts))]
        run_inference(text, model, params, state, vocab, max_seq_len; use_gpu=use_gpu)
    end

    # Benchmark
    println("Running $n_iterations iterations...")

    total_tokens = 0
    times = Float64[]

    for i in 1:n_iterations
        text = test_texts[(i-1) % length(test_texts) + 1]
        token_ids, _ = tokenize(text, vocab)

        t0 = time()
        run_inference(text, model, params, state, vocab, max_seq_len; use_gpu=use_gpu)
        t1 = time()

        push!(times, t1 - t0)
        total_tokens += length(token_ids)
    end

    total_time = sum(times)
    avg_time = mean(times)
    std_time = std(times)
    tokens_per_sec = total_tokens / total_time

    println("\nResults:")
    println("  Total iterations: $n_iterations")
    println("  Total tokens processed: $total_tokens")
    println("  Total time: $(round(total_time, digits=3))s")
    println("  Avg time per inference: $(round(avg_time * 1000, digits=2))ms ± $(round(std_time * 1000, digits=2))ms")
    println("  Throughput: $(round(tokens_per_sec, digits=1)) tokens/sec")
    println("  Throughput: $(round(n_iterations / total_time, digits=1)) sentences/sec")

    return tokens_per_sec
end

function evaluate_accuracy(model, params, state, vocab, max_seq_len; use_gpu=false)
    println("\n" * "="^60)
    println("ACCURACY EVALUATION")
    println("="^60)

    total_expected = 0
    total_found = 0
    correct_entities = 0
    correct_labels = 0

    for tc in TEST_CASES
        labels, entities = run_inference(tc.text, model, params, state, vocab, max_seq_len; use_gpu=use_gpu)

        println("\nText: $(tc.text)")
        println("Expected: $(tc.expected)")
        println("Found: $entities")

        # Count metrics
        total_expected += length(tc.expected)
        total_found += length(entities)

        # Check each expected entity
        for (exp_text, exp_label) in tc.expected
            for (found_text, found_label) in entities
                # Fuzzy match (entity text contained in found or vice versa)
                if occursin(lowercase(exp_text), lowercase(found_text)) ||
                   occursin(lowercase(found_text), lowercase(exp_text))
                    correct_entities += 1
                    if found_label == exp_label
                        correct_labels += 1
                    end
                    break
                end
            end
        end
    end

    println("\n" * "-"^40)
    println("Summary:")
    println("  Expected entities: $total_expected")
    println("  Found entities: $total_found")
    println("  Correct entities (fuzzy): $correct_entities")
    println("  Correct with label: $correct_labels")

    recall = total_expected > 0 ? correct_entities / total_expected : 0.0
    precision = total_found > 0 ? correct_entities / total_found : 0.0
    f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0

    println("\n  Recall: $(round(recall * 100, digits=1))%")
    println("  Precision: $(round(precision * 100, digits=1))%")
    println("  F1 Score: $(round(f1 * 100, digits=1))%")

    return f1
end

function main()
    println("="^60)
    println("OssammaNER Benchmark")
    println("="^60)

    # Check GPU
    use_gpu = CUDA.functional()
    println("\nGPU available: $use_gpu")
    if use_gpu
        println("GPU: $(CUDA.name(CUDA.device()))")
    end

    # Load model
    model, params, state, vocab, id_to_token, config = load_model()
    max_seq_len = config.max_sequence_length

    println("\nModel config:")
    println("  Embedding dim: $(config.embedding_dimension)")
    println("  Layers: $(config.number_of_layers)")
    println("  Heads: $(config.number_of_heads)")
    println("  Max seq len: $max_seq_len")

    # Move to GPU if available
    if use_gpu
        println("\nMoving model to GPU...")
        params = Lux.gpu(params)
        state = Lux.gpu(state)
    end

    # Run accuracy evaluation
    f1 = evaluate_accuracy(model, params, state, vocab, max_seq_len; use_gpu=use_gpu)

    # Run speed benchmark
    throughput = benchmark_speed(model, params, state, vocab, max_seq_len; n_iterations=100, use_gpu=use_gpu)

    println("\n" * "="^60)
    println("FINAL RESULTS")
    println("="^60)
    println("  F1 Score: $(round(f1 * 100, digits=1))%")
    println("  Throughput: $(round(throughput, digits=1)) tokens/sec")
    println("="^60)
end

main()
