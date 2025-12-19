module Augmentation

"""
Entity-aware data augmentation strategies for NER.

Provides augmentation techniques that preserve entity annotations:
1. Entity Replacement - swap entities of same type
2. Mention Dropout - randomly remove entity mentions
3. Context Shuffling - shuffle non-entity tokens
4. Synonym Replacement - replace non-entity words with synonyms
5. Token Dropout - randomly drop individual tokens
6. Case Augmentation - random case changes for non-entities
"""

using Random

# =============================================================================
# Entity Span Extraction
# =============================================================================

"""
    EntitySpan

Represents an entity mention with its position and type.
"""
struct EntitySpan
    start_idx::Int
    end_idx::Int
    entity_type::String
    tokens::Vector{String}
end

function Base.show(io::IO, span::EntitySpan)
    text = join(span.tokens, " ")
    print(io, "EntitySpan($(span.entity_type): \"$text\" @ $(span.start_idx):$(span.end_idx))")
end

"""
    extract_spans(tokens, labels, id_to_label) -> Vector{EntitySpan}

Extract entity spans from a labeled sequence.
"""
function extract_spans(
    tokens::Vector{String},
    labels::Vector{Int},
    id_to_label::Dict{Int, String}
)
    spans = EntitySpan[]
    i = 1

    while i <= length(labels)
        label = id_to_label[labels[i]]

        if startswith(label, "B-")
            entity_type = label[3:end]
            start_idx = i
            entity_tokens = [tokens[i]]

            # Find extent of entity
            i += 1
            while i <= length(labels)
                next_label = id_to_label[labels[i]]
                if next_label == "I-$entity_type"
                    push!(entity_tokens, tokens[i])
                    i += 1
                else
                    break
                end
            end

            push!(spans, EntitySpan(start_idx, i - 1, entity_type, entity_tokens))
        else
            i += 1
        end
    end

    return spans
end

"""
    get_non_entity_indices(labels, id_to_label) -> Vector{Int}

Get indices of non-entity (O) tokens.
"""
function get_non_entity_indices(labels::Vector{Int}, id_to_label::Dict{Int, String})
    return [i for i in 1:length(labels) if id_to_label[labels[i]] == "O"]
end

# =============================================================================
# Entity Bank
# =============================================================================

"""
    EntityBank

Collection of entity mentions organized by type for replacement augmentation.
"""
struct EntityBank
    entities::Dict{String, Vector{Vector{String}}}
end

function EntityBank()
    return EntityBank(Dict{String, Vector{Vector{String}}}())
end

"""
    add_entity!(bank, entity_type, tokens)

Add an entity to the bank.
"""
function add_entity!(bank::EntityBank, entity_type::String, tokens::Vector{String})
    if !haskey(bank.entities, entity_type)
        bank.entities[entity_type] = Vector{String}[]
    end
    push!(bank.entities[entity_type], tokens)
end

"""
    build_entity_bank(samples, id_to_label) -> EntityBank

Build an entity bank from a collection of NER samples.
"""
function build_entity_bank(
    samples::Vector,
    id_to_label::Dict{Int, String}
)
    bank = EntityBank()

    for sample in samples
        spans = extract_spans(sample.tokens, sample.labels, id_to_label)
        for span in spans
            add_entity!(bank, span.entity_type, span.tokens)
        end
    end

    return bank
end

"""
    get_random_entity(bank, entity_type; rng=Random.GLOBAL_RNG) -> Union{Vector{String}, Nothing}

Get a random entity of the given type.
"""
function get_random_entity(
    bank::EntityBank,
    entity_type::String;
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    if !haskey(bank.entities, entity_type) || isempty(bank.entities[entity_type])
        return nothing
    end
    return rand(rng, bank.entities[entity_type])
end

# =============================================================================
# Augmentation Functions
# =============================================================================

"""
    entity_replacement(tokens, labels, id_to_label, label_to_id, entity_bank; p=0.3, rng=GLOBAL_RNG)

Replace entities with other entities of the same type.

Arguments:
- tokens: Vector of word strings
- labels: Vector of label IDs
- id_to_label: ID to label string mapping
- label_to_id: Label string to ID mapping
- entity_bank: EntityBank with replacement entities
- p: Probability of replacing each entity
- rng: Random number generator

Returns: (new_tokens, new_labels)
"""
function entity_replacement(
    tokens::Vector{String},
    labels::Vector{Int},
    id_to_label::Dict{Int, String},
    label_to_id::Dict{String, Int},
    entity_bank::EntityBank;
    p::Float64 = 0.3,
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    spans = extract_spans(tokens, labels, id_to_label)

    if isempty(spans)
        return copy(tokens), copy(labels)
    end

    new_tokens = String[]
    new_labels = Int[]
    prev_end = 0

    for span in spans
        # Copy tokens before this entity
        for i in (prev_end + 1):(span.start_idx - 1)
            push!(new_tokens, tokens[i])
            push!(new_labels, labels[i])
        end

        # Decide whether to replace
        if rand(rng) < p
            replacement = get_random_entity(entity_bank, span.entity_type; rng=rng)

            if replacement !== nothing && replacement != span.tokens
                # Use replacement
                for (i, token) in enumerate(replacement)
                    push!(new_tokens, token)
                    if i == 1
                        push!(new_labels, label_to_id["B-$(span.entity_type)"])
                    else
                        push!(new_labels, label_to_id["I-$(span.entity_type)"])
                    end
                end
            else
                # Keep original
                for i in span.start_idx:span.end_idx
                    push!(new_tokens, tokens[i])
                    push!(new_labels, labels[i])
                end
            end
        else
            # Keep original
            for i in span.start_idx:span.end_idx
                push!(new_tokens, tokens[i])
                push!(new_labels, labels[i])
            end
        end

        prev_end = span.end_idx
    end

    # Copy remaining tokens
    for i in (prev_end + 1):length(tokens)
        push!(new_tokens, tokens[i])
        push!(new_labels, labels[i])
    end

    return new_tokens, new_labels
end

"""
    mention_dropout(tokens, labels, id_to_label; p=0.1, rng=GLOBAL_RNG)

Randomly drop entire entity mentions.

Arguments:
- tokens: Vector of word strings
- labels: Vector of label IDs
- id_to_label: ID to label string mapping
- p: Probability of dropping each entity
- rng: Random number generator

Returns: (new_tokens, new_labels)
"""
function mention_dropout(
    tokens::Vector{String},
    labels::Vector{Int},
    id_to_label::Dict{Int, String};
    p::Float64 = 0.1,
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    spans = extract_spans(tokens, labels, id_to_label)

    # Determine which spans to drop
    drop_indices = Set{Int}()
    for span in spans
        if rand(rng) < p
            for i in span.start_idx:span.end_idx
                push!(drop_indices, i)
            end
        end
    end

    # Filter out dropped tokens
    new_tokens = String[]
    new_labels = Int[]

    for i in 1:length(tokens)
        if !(i in drop_indices)
            push!(new_tokens, tokens[i])
            push!(new_labels, labels[i])
        end
    end

    return new_tokens, new_labels
end

"""
    context_shuffle(tokens, labels, id_to_label; window_size=5, rng=GLOBAL_RNG)

Shuffle non-entity tokens within local windows.

Arguments:
- tokens: Vector of word strings
- labels: Vector of label IDs
- id_to_label: ID to label string mapping
- window_size: Size of shuffle windows
- rng: Random number generator

Returns: (new_tokens, new_labels)
"""
function context_shuffle(
    tokens::Vector{String},
    labels::Vector{Int},
    id_to_label::Dict{Int, String};
    window_size::Int = 5,
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    new_tokens = copy(tokens)
    new_labels = copy(labels)

    # Get non-entity indices
    non_entity_idx = get_non_entity_indices(labels, id_to_label)

    if length(non_entity_idx) < 2
        return new_tokens, new_labels
    end

    # Group into windows and shuffle within each window
    for start in 1:window_size:length(non_entity_idx)
        window_end = min(start + window_size - 1, length(non_entity_idx))
        window_indices = non_entity_idx[start:window_end]

        if length(window_indices) > 1
            # Shuffle the tokens at these indices
            window_tokens = [tokens[i] for i in window_indices]
            shuffle!(rng, window_tokens)

            for (i, idx) in enumerate(window_indices)
                new_tokens[idx] = window_tokens[i]
            end
        end
    end

    return new_tokens, new_labels
end

"""
    synonym_replacement(tokens, labels, id_to_label, synonyms; p=0.1, rng=GLOBAL_RNG)

Replace non-entity words with synonyms.

Arguments:
- tokens: Vector of word strings
- labels: Vector of label IDs
- id_to_label: ID to label string mapping
- synonyms: Dict mapping words to lists of synonyms
- p: Probability of replacing each non-entity word
- rng: Random number generator

Returns: (new_tokens, new_labels)
"""
function synonym_replacement(
    tokens::Vector{String},
    labels::Vector{Int},
    id_to_label::Dict{Int, String},
    synonyms::Dict{String, Vector{String}};
    p::Float64 = 0.1,
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    new_tokens = copy(tokens)

    for i in 1:length(tokens)
        # Only replace non-entity tokens
        if id_to_label[labels[i]] == "O" && rand(rng) < p
            word_lower = lowercase(tokens[i])
            if haskey(synonyms, word_lower) && !isempty(synonyms[word_lower])
                # Preserve original case pattern
                replacement = rand(rng, synonyms[word_lower])
                if isuppercase(tokens[i][1])
                    replacement = uppercasefirst(replacement)
                end
                new_tokens[i] = replacement
            end
        end
    end

    return new_tokens, copy(labels)
end

"""
    token_dropout(tokens, labels, id_to_label; p=0.1, preserve_entities=true, rng=GLOBAL_RNG)

Randomly drop individual tokens.

Arguments:
- tokens: Vector of word strings
- labels: Vector of label IDs
- id_to_label: ID to label string mapping
- p: Probability of dropping each token
- preserve_entities: If true, never drop entity tokens
- rng: Random number generator

Returns: (new_tokens, new_labels)
"""
function token_dropout(
    tokens::Vector{String},
    labels::Vector{Int},
    id_to_label::Dict{Int, String};
    p::Float64 = 0.1,
    preserve_entities::Bool = true,
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    new_tokens = String[]
    new_labels = Int[]

    for i in 1:length(tokens)
        is_entity = id_to_label[labels[i]] != "O"

        if preserve_entities && is_entity
            # Always keep entity tokens
            push!(new_tokens, tokens[i])
            push!(new_labels, labels[i])
        elseif rand(rng) >= p
            # Keep non-entity tokens with probability 1-p
            push!(new_tokens, tokens[i])
            push!(new_labels, labels[i])
        end
    end

    return new_tokens, new_labels
end

"""
    case_augmentation(tokens, labels, id_to_label; p=0.1, preserve_entities=true, rng=GLOBAL_RNG)

Randomly change case of tokens.

Arguments:
- tokens: Vector of word strings
- labels: Vector of label IDs
- id_to_label: ID to label string mapping
- p: Probability of changing case for each token
- preserve_entities: If true, preserve case of entity tokens
- rng: Random number generator

Returns: (new_tokens, new_labels)
"""
function case_augmentation(
    tokens::Vector{String},
    labels::Vector{Int},
    id_to_label::Dict{Int, String};
    p::Float64 = 0.1,
    preserve_entities::Bool = true,
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    new_tokens = copy(tokens)

    for i in 1:length(tokens)
        is_entity = id_to_label[labels[i]] != "O"

        if (preserve_entities && is_entity) || rand(rng) >= p
            continue
        end

        # Random case transformation
        transform = rand(rng, 1:4)
        if transform == 1
            new_tokens[i] = lowercase(tokens[i])
        elseif transform == 2
            new_tokens[i] = uppercase(tokens[i])
        elseif transform == 3
            new_tokens[i] = uppercasefirst(lowercase(tokens[i]))
        end
        # transform == 4: keep original
    end

    return new_tokens, copy(labels)
end

# =============================================================================
# Augmentation Pipeline
# =============================================================================

"""
    AugmentationConfig

Configuration for the augmentation pipeline.
"""
Base.@kwdef struct AugmentationConfig
    entity_replacement_p::Float64 = 0.0
    mention_dropout_p::Float64 = 0.0
    context_shuffle_window::Int = 0  # 0 = disabled
    synonym_replacement_p::Float64 = 0.0
    token_dropout_p::Float64 = 0.0
    case_augmentation_p::Float64 = 0.0
    preserve_entities::Bool = true
end

"""
    default_augmentation_config()

Return sensible default augmentation settings.
"""
function default_augmentation_config()
    return AugmentationConfig(
        entity_replacement_p = 0.15,
        mention_dropout_p = 0.05,
        context_shuffle_window = 5,
        synonym_replacement_p = 0.1,
        token_dropout_p = 0.05,
        case_augmentation_p = 0.05,
        preserve_entities = true,
    )
end

"""
    augment(tokens, labels, id_to_label, label_to_id, config; entity_bank=nothing, synonyms=nothing, rng=GLOBAL_RNG)

Apply the full augmentation pipeline to a sample.
"""
function augment(
    tokens::Vector{String},
    labels::Vector{Int},
    id_to_label::Dict{Int, String},
    label_to_id::Dict{String, Int},
    config::AugmentationConfig;
    entity_bank::Union{EntityBank, Nothing} = nothing,
    synonyms::Union{Dict{String, Vector{String}}, Nothing} = nothing,
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    new_tokens = copy(tokens)
    new_labels = copy(labels)

    # Entity replacement
    if config.entity_replacement_p > 0 && entity_bank !== nothing
        new_tokens, new_labels = entity_replacement(
            new_tokens, new_labels, id_to_label, label_to_id, entity_bank;
            p = config.entity_replacement_p, rng = rng
        )
    end

    # Mention dropout
    if config.mention_dropout_p > 0
        new_tokens, new_labels = mention_dropout(
            new_tokens, new_labels, id_to_label;
            p = config.mention_dropout_p, rng = rng
        )
    end

    # Context shuffle
    if config.context_shuffle_window > 0
        new_tokens, new_labels = context_shuffle(
            new_tokens, new_labels, id_to_label;
            window_size = config.context_shuffle_window, rng = rng
        )
    end

    # Synonym replacement
    if config.synonym_replacement_p > 0 && synonyms !== nothing
        new_tokens, new_labels = synonym_replacement(
            new_tokens, new_labels, id_to_label, synonyms;
            p = config.synonym_replacement_p, rng = rng
        )
    end

    # Token dropout
    if config.token_dropout_p > 0
        new_tokens, new_labels = token_dropout(
            new_tokens, new_labels, id_to_label;
            p = config.token_dropout_p,
            preserve_entities = config.preserve_entities,
            rng = rng
        )
    end

    # Case augmentation
    if config.case_augmentation_p > 0
        new_tokens, new_labels = case_augmentation(
            new_tokens, new_labels, id_to_label;
            p = config.case_augmentation_p,
            preserve_entities = config.preserve_entities,
            rng = rng
        )
    end

    return new_tokens, new_labels
end

# =============================================================================
# Exports
# =============================================================================

export EntitySpan, EntityBank
export extract_spans, get_non_entity_indices
export add_entity!, build_entity_bank, get_random_entity
export entity_replacement, mention_dropout, context_shuffle
export synonym_replacement, token_dropout, case_augmentation
export AugmentationConfig, default_augmentation_config, augment

end # module
