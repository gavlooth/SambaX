module NERDataset

"""
Unified NER Dataset handling for training and evaluation.

Supports multiple input formats:
- CoNLL format (word TAB label per line, blank lines separate sentences)
- JSONL format ({"tokens": [...], "labels": [...]} per line)
- BIO and IOB2 tagging schemes

Provides:
- NERSample: Single annotated sequence
- NERDataLoader: Batched iteration with padding and masking
"""

using Random
using JSON3

# =============================================================================
# Label Mapping
# =============================================================================

# Default RAG-optimized 9-label schema
const DEFAULT_LABELS = [
    "O",
    "B-PERSON", "I-PERSON",
    "B-AGENCY", "I-AGENCY",
    "B-PLACE", "I-PLACE",
    "B-ORGANISM", "I-ORGANISM",
    "B-EVENT", "I-EVENT",
    "B-INSTRUMENT", "I-INSTRUMENT",
    "B-WORK", "I-WORK",
    "B-DOMAIN", "I-DOMAIN",
    "B-MEASURE", "I-MEASURE",
]

# Standard mappings from common NER datasets
const LABEL_MAPPING = Dict{String, Union{String, Nothing}}(
    # OntoNotes -> RAG Schema
    "PERSON" => "PERSON",
    "ORG" => "AGENCY",
    "GPE" => "PLACE",
    "LOC" => "PLACE",
    "FAC" => "PLACE",
    "EVENT" => "EVENT",
    "PRODUCT" => "INSTRUMENT",
    "WORK_OF_ART" => "WORK",
    "DATE" => "MEASURE",
    "TIME" => "MEASURE",
    "PERCENT" => "MEASURE",
    "MONEY" => "MEASURE",
    "QUANTITY" => "MEASURE",
    "CARDINAL" => "MEASURE",
    "ORDINAL" => "MEASURE",
    "NORP" => "DOMAIN",
    "LAW" => "WORK",
    "LANGUAGE" => "DOMAIN",

    # CoNLL -> RAG Schema
    "PER" => "PERSON",
    # "ORG" already mapped
    # "LOC" already mapped
    "MISC" => nothing,  # Context-dependent

    # Few-NERD fine-grained
    "person-actor" => "PERSON",
    "person-artist" => "PERSON",
    "person-scientist" => "PERSON",
    "person-athlete" => "PERSON",
    "person-politician" => "PERSON",
    "person-scholar" => "PERSON",
    "person-soldier" => "PERSON",
    "person-director" => "PERSON",
    "person-other" => "PERSON",

    "organization-company" => "AGENCY",
    "organization-government" => "AGENCY",
    "organization-political_party" => "AGENCY",
    "organization-education" => "AGENCY",
    "organization-religion" => "AGENCY",
    "organization-sports_team" => "AGENCY",
    "organization-sports_league" => "AGENCY",
    "organization-media" => "AGENCY",
    "organization-other" => "AGENCY",

    "location-city" => "PLACE",
    "location-country" => "PLACE",
    "location-GPE" => "PLACE",
    "location-road" => "PLACE",
    "location-bodiesofwater" => "PLACE",
    "location-park" => "PLACE",
    "location-mountain" => "PLACE",
    "location-island" => "PLACE",
    "location-other" => "PLACE",

    "building-airport" => "PLACE",
    "building-hospital" => "PLACE",
    "building-hotel" => "PLACE",
    "building-library" => "PLACE",
    "building-restaurant" => "PLACE",
    "building-sportsfacility" => "PLACE",
    "building-theater" => "PLACE",
    "building-other" => "PLACE",

    "event-war" => "EVENT",
    "event-disaster" => "EVENT",
    "event-election" => "EVENT",
    "event-protest" => "EVENT",
    "event-sportsevent" => "EVENT",
    "event-other" => "EVENT",

    "product-software" => "INSTRUMENT",
    "product-weapon" => "INSTRUMENT",
    "product-car" => "INSTRUMENT",
    "product-food" => "INSTRUMENT",
    "product-game" => "INSTRUMENT",
    "product-ship" => "INSTRUMENT",
    "product-airplane" => "INSTRUMENT",
    "product-other" => "INSTRUMENT",

    "art-film" => "WORK",
    "art-music" => "WORK",
    "art-writtenart" => "WORK",
    "art-painting" => "WORK",
    "art-other" => "WORK",

    "other-biologything" => "ORGANISM",
    "other-livingthing" => "ORGANISM",
    "other-chemicalthing" => "DOMAIN",
    "other-disease" => "DOMAIN",
    "other-scientificterm" => "DOMAIN",
    "other-astronomything" => "DOMAIN",
    "other-currency" => "MEASURE",
    "other-medical" => "INSTRUMENT",
    "other-award" => "WORK",
    "other-law" => "WORK",
    "other-god" => "PERSON",
    "other-language" => "DOMAIN",
    "other-educationaldegree" => "DOMAIN",

    # MultiNERD
    "PER" => "PERSON",
    "ANIM" => "ORGANISM",
    "BIO" => "ORGANISM",
    "CEL" => "PLACE",
    "DIS" => "DOMAIN",
    "EVE" => "EVENT",
    "FOOD" => "INSTRUMENT",
    "INST" => "INSTRUMENT",
    "MEDIA" => "WORK",
    "MYTH" => "PERSON",
    "PLANT" => "ORGANISM",
    "TIME" => "MEASURE",
    "VEHI" => "INSTRUMENT",
)

# =============================================================================
# NERSample
# =============================================================================

"""
    NERSample

A single NER-annotated sequence.

Fields:
- tokens: Vector of word strings
- labels: Vector of label IDs (integers)
- original_labels: Vector of original label strings (before mapping)
"""
struct NERSample
    tokens::Vector{String}
    labels::Vector{Int}
    original_labels::Vector{String}
end

function Base.length(sample::NERSample)
    return length(sample.tokens)
end

function Base.show(io::IO, sample::NERSample)
    print(io, "NERSample($(length(sample.tokens)) tokens)")
end

# =============================================================================
# Label Conversion
# =============================================================================

"""
    map_label(label::String, label_to_id::Dict; mapping=LABEL_MAPPING) -> Int

Map a label string to its ID, applying schema mapping if needed.
"""
function map_label(
    label::String,
    label_to_id::Dict{String, Int};
    mapping::Dict = LABEL_MAPPING
)
    # Handle O tag
    if label == "O"
        return label_to_id["O"]
    end

    # Parse BIO prefix
    if !startswith(label, "B-") && !startswith(label, "I-")
        # Try IOB1 format (no prefix means inside)
        if haskey(label_to_id, "B-$label")
            return label_to_id["B-$label"]
        end
        return label_to_id["O"]
    end

    prefix = label[1:2]
    entity_type = label[3:end]

    # Try direct match first
    mapped_label = "$prefix$entity_type"
    if haskey(label_to_id, mapped_label)
        return label_to_id[mapped_label]
    end

    # Try mapping
    mapped_type = get(mapping, entity_type, nothing)
    if mapped_type !== nothing
        mapped_label = "$prefix$mapped_type"
        if haskey(label_to_id, mapped_label)
            return label_to_id[mapped_label]
        end
    end

    # Unknown entity type -> O
    return label_to_id["O"]
end

"""
    create_label_dicts(labels::Vector{String}) -> (label_to_id, id_to_label)

Create label dictionaries from a list of labels.
"""
function create_label_dicts(labels::Vector{String} = DEFAULT_LABELS)
    label_to_id = Dict(label => i for (i, label) in enumerate(labels))
    id_to_label = Dict(i => label for (i, label) in enumerate(labels))
    return label_to_id, id_to_label
end

# =============================================================================
# Data Loading - CoNLL Format
# =============================================================================

"""
    load_conll(filepath::String; label_to_id=nothing) -> Vector{NERSample}

Load NER data from CoNLL format file.

Format: word TAB POS TAB chunk TAB label (or word TAB label)
Sentences separated by blank lines.
"""
function load_conll(
    filepath::String;
    label_to_id::Union{Dict{String, Int}, Nothing} = nothing,
    label_column::Int = -1  # -1 means last column
)
    if label_to_id === nothing
        label_to_id, _ = create_label_dicts()
    end

    samples = NERSample[]
    current_tokens = String[]
    current_labels = String[]

    for line in eachline(filepath)
        line = strip(line)

        # Skip comments
        if startswith(line, "#") || startswith(line, "-DOCSTART-")
            continue
        end

        if isempty(line)
            # End of sentence
            if !isempty(current_tokens)
                label_ids = [map_label(l, label_to_id) for l in current_labels]
                push!(samples, NERSample(current_tokens, label_ids, current_labels))
                current_tokens = String[]
                current_labels = String[]
            end
        else
            # Parse token line
            parts = split(line, r"\s+")
            if length(parts) >= 2
                token = String(parts[1])
                label = label_column == -1 ? String(parts[end]) : String(parts[label_column])

                push!(current_tokens, token)
                push!(current_labels, label)
            end
        end
    end

    # Handle last sentence if file doesn't end with blank line
    if !isempty(current_tokens)
        label_ids = [map_label(l, label_to_id) for l in current_labels]
        push!(samples, NERSample(current_tokens, label_ids, current_labels))
    end

    return samples
end

# =============================================================================
# Data Loading - JSONL Format
# =============================================================================

"""
    load_jsonl(filepath::String; label_to_id=nothing) -> Vector{NERSample}

Load NER data from JSONL format file.

Each line: {"tokens": ["word1", "word2", ...], "labels": ["O", "B-PER", ...]}
Alternative: {"tokens": [...], "ner_tags": [...]}
"""
function load_jsonl(
    filepath::String;
    label_to_id::Union{Dict{String, Int}, Nothing} = nothing
)
    if label_to_id === nothing
        label_to_id, _ = create_label_dicts()
    end

    samples = NERSample[]

    for line in eachline(filepath)
        line = strip(line)
        if isempty(line)
            continue
        end

        data = JSON3.read(line)

        tokens = String.(data.tokens)

        # Handle different label field names
        if haskey(data, :labels)
            original_labels = String.(data.labels)
        elseif haskey(data, :ner_tags)
            original_labels = String.(data.ner_tags)
        elseif haskey(data, :tags)
            original_labels = String.(data.tags)
        else
            error("No labels found in JSONL record")
        end

        label_ids = [map_label(l, label_to_id) for l in original_labels]
        push!(samples, NERSample(tokens, label_ids, original_labels))
    end

    return samples
end

"""
    load_dataset(filepath::String; format=:auto, kwargs...) -> Vector{NERSample}

Load NER dataset with automatic format detection.
"""
function load_dataset(
    filepath::String;
    format::Symbol = :auto,
    kwargs...
)
    if format == :auto
        if endswith(filepath, ".jsonl") || endswith(filepath, ".json")
            format = :jsonl
        else
            format = :conll
        end
    end

    if format == :jsonl
        return load_jsonl(filepath; kwargs...)
    else
        return load_conll(filepath; kwargs...)
    end
end

# =============================================================================
# NERDataLoader
# =============================================================================

"""
    NERDataLoader

Batched data loader for NER training.

Handles:
- Batching samples of similar length
- Padding sequences to batch max length
- Creating attention masks
- Shuffling between epochs
"""
mutable struct NERDataLoader
    samples::Vector{NERSample}
    batch_size::Int
    max_length::Int
    pad_token_id::Int
    ignore_label_id::Int
    shuffle::Bool
    indices::Vector{Int}
    current_idx::Int
end

function NERDataLoader(
    samples::Vector{NERSample};
    batch_size::Int = 16,
    max_length::Int = 512,
    pad_token_id::Int = 0,
    ignore_label_id::Int = -100,
    shuffle::Bool = true
)
    indices = collect(1:length(samples))
    if shuffle
        Random.shuffle!(indices)
    end

    return NERDataLoader(
        samples,
        batch_size,
        max_length,
        pad_token_id,
        ignore_label_id,
        shuffle,
        indices,
        1
    )
end

function Base.length(loader::NERDataLoader)
    return ceil(Int, length(loader.samples) / loader.batch_size)
end

"""
    reset!(loader::NERDataLoader)

Reset the data loader for a new epoch.
"""
function reset!(loader::NERDataLoader)
    loader.current_idx = 1
    if loader.shuffle
        Random.shuffle!(loader.indices)
    end
end

"""
    get_batch(loader::NERDataLoader) -> (tokens, labels, mask) or nothing

Get the next batch. Returns nothing when epoch is complete.

Returns:
- tokens: (seq_len, batch_size) - padded token strings
- labels: (seq_len, batch_size) - padded label IDs
- mask: (seq_len, batch_size) - attention mask (true = valid)
"""
function get_batch(loader::NERDataLoader)
    if loader.current_idx > length(loader.samples)
        return nothing
    end

    # Get batch indices
    start_idx = loader.current_idx
    end_idx = min(start_idx + loader.batch_size - 1, length(loader.samples))
    batch_indices = loader.indices[start_idx:end_idx]
    loader.current_idx = end_idx + 1

    batch_samples = [loader.samples[i] for i in batch_indices]
    actual_batch_size = length(batch_samples)

    # Find max sequence length in batch (capped by max_length)
    batch_max_len = min(
        maximum(length(s) for s in batch_samples),
        loader.max_length
    )

    # Create padded arrays
    tokens = fill("", batch_max_len, actual_batch_size)
    labels = fill(loader.ignore_label_id, batch_max_len, actual_batch_size)
    mask = fill(false, batch_max_len, actual_batch_size)

    for (b, sample) in enumerate(batch_samples)
        seq_len = min(length(sample), loader.max_length)

        for t in 1:seq_len
            tokens[t, b] = sample.tokens[t]
            labels[t, b] = sample.labels[t]
            mask[t, b] = true
        end
    end

    return (tokens = tokens, labels = labels, mask = mask)
end

# Iteration interface
function Base.iterate(loader::NERDataLoader)
    reset!(loader)
    batch = get_batch(loader)
    if batch === nothing
        return nothing
    end
    return (batch, loader)
end

function Base.iterate(loader::NERDataLoader, state)
    batch = get_batch(loader)
    if batch === nothing
        return nothing
    end
    return (batch, loader)
end

# =============================================================================
# Train/Val/Test Splits
# =============================================================================

"""
    train_val_test_split(samples; train_ratio=0.8, val_ratio=0.1, seed=42)

Split samples into train/val/test sets.
"""
function train_val_test_split(
    samples::Vector{NERSample};
    train_ratio::Float64 = 0.8,
    val_ratio::Float64 = 0.1,
    seed::Int = 42
)
    rng = Random.MersenneTwister(seed)
    indices = Random.shuffle(rng, collect(1:length(samples)))

    n = length(samples)
    train_end = round(Int, n * train_ratio)
    val_end = round(Int, n * (train_ratio + val_ratio))

    train_samples = [samples[i] for i in indices[1:train_end]]
    val_samples = [samples[i] for i in indices[train_end+1:val_end]]
    test_samples = [samples[i] for i in indices[val_end+1:end]]

    return train_samples, val_samples, test_samples
end

# =============================================================================
# Statistics
# =============================================================================

"""
    dataset_statistics(samples; id_to_label=nothing) -> Dict

Compute statistics about the dataset.
"""
function dataset_statistics(
    samples::Vector{NERSample};
    id_to_label::Union{Dict{Int, String}, Nothing} = nothing
)
    if id_to_label === nothing
        _, id_to_label = create_label_dicts()
    end

    num_samples = length(samples)
    total_tokens = sum(length(s) for s in samples)
    avg_length = total_tokens / num_samples

    # Label distribution
    label_counts = Dict{String, Int}()
    entity_counts = Dict{String, Int}()

    for sample in samples
        for (i, label_id) in enumerate(sample.labels)
            label = id_to_label[label_id]
            label_counts[label] = get(label_counts, label, 0) + 1

            # Count entity starts (B- tags)
            if startswith(label, "B-")
                entity_type = label[3:end]
                entity_counts[entity_type] = get(entity_counts, entity_type, 0) + 1
            end
        end
    end

    return Dict(
        "num_samples" => num_samples,
        "total_tokens" => total_tokens,
        "avg_length" => avg_length,
        "min_length" => minimum(length(s) for s in samples),
        "max_length" => maximum(length(s) for s in samples),
        "label_distribution" => label_counts,
        "entity_counts" => entity_counts,
    )
end

# =============================================================================
# Exports
# =============================================================================

export NERSample, NERDataLoader
export load_conll, load_jsonl, load_dataset
export create_label_dicts, map_label
export train_val_test_split, dataset_statistics
export reset!, get_batch
export LABEL_MAPPING, DEFAULT_LABELS

end # module
