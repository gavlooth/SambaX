module Tokenizer

"""
Tokenizer with NER label alignment for subword tokenization.

Handles the challenge of aligning word-level NER labels with subword tokens:
- First subword of each word gets the word's label
- Subsequent subwords get ignore_index (-100) for loss masking

Supports:
- Word-level tokenization (space-based)
- Character-level tokenization
- Subword tokenization with label alignment
- Vocabulary building from corpus
"""

using Random
using JSON3

# =============================================================================
# Vocabulary
# =============================================================================

# Special tokens
const PAD_TOKEN = "[PAD]"
const UNK_TOKEN = "[UNK]"
const CLS_TOKEN = "[CLS]"
const SEP_TOKEN = "[SEP]"
const MASK_TOKEN = "[MASK]"

const SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]

"""
    Vocabulary

Token-to-ID mapping with special tokens.
"""
struct Vocabulary
    token_to_id::Dict{String, Int}
    id_to_token::Dict{Int, String}
    vocab_size::Int
    pad_token_id::Int
    unk_token_id::Int
    cls_token_id::Int
    sep_token_id::Int
    mask_token_id::Int
end

function Vocabulary(tokens::Vector{String})
    # Add special tokens first
    all_tokens = copy(SPECIAL_TOKENS)

    # Add regular tokens (skip if already in special tokens)
    for token in tokens
        if !(token in all_tokens)
            push!(all_tokens, token)
        end
    end

    token_to_id = Dict(token => i for (i, token) in enumerate(all_tokens))
    id_to_token = Dict(i => token for (i, token) in enumerate(all_tokens))

    return Vocabulary(
        token_to_id,
        id_to_token,
        length(all_tokens),
        token_to_id[PAD_TOKEN],
        token_to_id[UNK_TOKEN],
        token_to_id[CLS_TOKEN],
        token_to_id[SEP_TOKEN],
        token_to_id[MASK_TOKEN],
    )
end

function Base.length(vocab::Vocabulary)
    return vocab.vocab_size
end

function Base.getindex(vocab::Vocabulary, token::String)
    return get(vocab.token_to_id, token, vocab.unk_token_id)
end

function Base.getindex(vocab::Vocabulary, id::Int)
    return get(vocab.id_to_token, id, UNK_TOKEN)
end

"""
    save_vocab(vocab::Vocabulary, filepath::String)

Save vocabulary to JSON file.
"""
function save_vocab(vocab::Vocabulary, filepath::String)
    data = Dict(
        "tokens" => [vocab.id_to_token[i] for i in 1:vocab.vocab_size],
        "special_tokens" => Dict(
            "pad" => vocab.pad_token_id,
            "unk" => vocab.unk_token_id,
            "cls" => vocab.cls_token_id,
            "sep" => vocab.sep_token_id,
            "mask" => vocab.mask_token_id,
        )
    )
    open(filepath, "w") do f
        JSON3.pretty(f, data)
    end
end

"""
    load_vocab(filepath::String) -> Vocabulary

Load vocabulary from JSON file.
"""
function load_vocab(filepath::String)
    data = JSON3.read(read(filepath, String))
    tokens = String.(data.tokens)

    # Remove special tokens from the list (they'll be added by constructor)
    regular_tokens = filter(t -> !(t in SPECIAL_TOKENS), tokens)

    return Vocabulary(regular_tokens)
end

# =============================================================================
# Vocabulary Building
# =============================================================================

"""
    build_vocab_from_tokens(token_lists; min_freq=1, max_vocab_size=nothing)

Build vocabulary from lists of tokens.
"""
function build_vocab_from_tokens(
    token_lists::Vector{Vector{String}};
    min_freq::Int = 1,
    max_vocab_size::Union{Int, Nothing} = nothing
)
    # Count token frequencies
    freq = Dict{String, Int}()
    for tokens in token_lists
        for token in tokens
            freq[token] = get(freq, token, 0) + 1
        end
    end

    # Filter by minimum frequency
    filtered = [(token, count) for (token, count) in freq if count >= min_freq]

    # Sort by frequency (descending)
    sort!(filtered, by = x -> -x[2])

    # Limit vocab size
    if max_vocab_size !== nothing
        max_regular = max_vocab_size - length(SPECIAL_TOKENS)
        filtered = filtered[1:min(length(filtered), max_regular)]
    end

    tokens = [t[1] for t in filtered]
    return Vocabulary(tokens)
end

"""
    build_vocab_from_corpus(filepath; kwargs...)

Build vocabulary from a text file (one sentence per line).
"""
function build_vocab_from_corpus(filepath::String; kwargs...)
    token_lists = Vector{String}[]
    for line in eachline(filepath)
        tokens = split(strip(line))
        if !isempty(tokens)
            push!(token_lists, String.(tokens))
        end
    end
    return build_vocab_from_tokens(token_lists; kwargs...)
end

# =============================================================================
# NERTokenizer
# =============================================================================

"""
    NERTokenizer

Tokenizer that handles NER label alignment with subword/word tokens.
"""
struct NERTokenizer
    vocab::Vocabulary
    max_length::Int
    lowercase::Bool
    add_special_tokens::Bool
end

function NERTokenizer(
    vocab::Vocabulary;
    max_length::Int = 512,
    lowercase::Bool = false,
    add_special_tokens::Bool = true
)
    return NERTokenizer(vocab, max_length, lowercase, add_special_tokens)
end

"""
    tokenize_word(tokenizer, word) -> Vector{String}

Tokenize a single word. For word-level tokenization, returns the word.
Override this for subword tokenization.
"""
function tokenize_word(tokenizer::NERTokenizer, word::String)
    if tokenizer.lowercase
        word = lowercase(word)
    end
    return [word]  # Word-level tokenization
end

"""
    tokenize_with_labels(tokenizer, tokens, labels; ignore_index=-100)

Tokenize a sequence with NER label alignment.

For subword tokenization:
- First subword gets the original label
- Subsequent subwords get ignore_index

Returns: (token_ids, aligned_labels, attention_mask)
"""
function tokenize_with_labels(
    tokenizer::NERTokenizer,
    tokens::Vector{String},
    labels::Vector{Int};
    ignore_index::Int = -100
)
    token_ids = Int[]
    aligned_labels = Int[]

    # Add [CLS] token
    if tokenizer.add_special_tokens
        push!(token_ids, tokenizer.vocab.cls_token_id)
        push!(aligned_labels, ignore_index)
    end

    # Process each word
    for (word, label) in zip(tokens, labels)
        subwords = tokenize_word(tokenizer, word)

        for (i, subword) in enumerate(subwords)
            # Check length limit (leave room for [SEP])
            if length(token_ids) >= tokenizer.max_length - 1
                break
            end

            token_id = tokenizer.vocab[subword]
            push!(token_ids, token_id)

            # First subword gets the label, rest get ignore_index
            if i == 1
                push!(aligned_labels, label)
            else
                push!(aligned_labels, ignore_index)
            end
        end

        if length(token_ids) >= tokenizer.max_length - 1
            break
        end
    end

    # Add [SEP] token
    if tokenizer.add_special_tokens
        push!(token_ids, tokenizer.vocab.sep_token_id)
        push!(aligned_labels, ignore_index)
    end

    # Create attention mask (all 1s for actual tokens)
    seq_len = length(token_ids)
    attention_mask = fill(true, seq_len)

    return token_ids, aligned_labels, attention_mask
end

"""
    tokenize(tokenizer, tokens) -> (token_ids, attention_mask)

Tokenize without labels (for inference).
"""
function tokenize(tokenizer::NERTokenizer, tokens::Vector{String})
    # Use dummy labels
    dummy_labels = fill(1, length(tokens))
    token_ids, _, attention_mask = tokenize_with_labels(tokenizer, tokens, dummy_labels)
    return token_ids, attention_mask
end

"""
    decode(tokenizer, token_ids) -> Vector{String}

Convert token IDs back to tokens.
"""
function decode(tokenizer::NERTokenizer, token_ids::Vector{Int})
    return [tokenizer.vocab[id] for id in token_ids]
end

# =============================================================================
# Batch Tokenization
# =============================================================================

"""
    TokenizedBatch

A batch of tokenized sequences with padding.
"""
struct TokenizedBatch
    token_ids::Matrix{Int}      # (seq_len, batch_size)
    labels::Matrix{Int}         # (seq_len, batch_size)
    attention_mask::Matrix{Bool} # (seq_len, batch_size)
    seq_lengths::Vector{Int}    # Original sequence lengths
end

"""
    tokenize_batch(tokenizer, batch_tokens, batch_labels; pad_to_max=false)

Tokenize a batch of sequences with padding.
"""
function tokenize_batch(
    tokenizer::NERTokenizer,
    batch_tokens::Vector{Vector{String}},
    batch_labels::Vector{Vector{Int}};
    pad_to_max::Bool = false,
    ignore_index::Int = -100
)
    batch_size = length(batch_tokens)

    # Tokenize all sequences
    tokenized = [
        tokenize_with_labels(tokenizer, tokens, labels; ignore_index=ignore_index)
        for (tokens, labels) in zip(batch_tokens, batch_labels)
    ]

    # Get sequence lengths
    seq_lengths = [length(t[1]) for t in tokenized]

    # Determine padding length
    if pad_to_max
        max_len = tokenizer.max_length
    else
        max_len = maximum(seq_lengths)
    end

    # Create padded arrays
    token_ids = fill(tokenizer.vocab.pad_token_id, max_len, batch_size)
    labels = fill(ignore_index, max_len, batch_size)
    attention_mask = fill(false, max_len, batch_size)

    for (i, (ids, lbls, mask)) in enumerate(tokenized)
        seq_len = length(ids)
        token_ids[1:seq_len, i] = ids
        labels[1:seq_len, i] = lbls
        attention_mask[1:seq_len, i] = mask
    end

    return TokenizedBatch(token_ids, labels, attention_mask, seq_lengths)
end

# =============================================================================
# Character-level Tokenizer
# =============================================================================

"""
    CharTokenizer <: NERTokenizer

Character-level tokenizer that splits words into individual characters.
Useful for morphologically rich languages or handling rare words.
"""
struct CharTokenizer
    vocab::Vocabulary
    max_length::Int
    lowercase::Bool
    add_special_tokens::Bool
    word_boundary::String  # Token to insert between words
end

function CharTokenizer(
    vocab::Vocabulary;
    max_length::Int = 512,
    lowercase::Bool = false,
    add_special_tokens::Bool = true,
    word_boundary::String = " "
)
    return CharTokenizer(vocab, max_length, lowercase, add_special_tokens, word_boundary)
end

function tokenize_word(tokenizer::CharTokenizer, word::String)
    if tokenizer.lowercase
        word = lowercase(word)
    end
    return [string(c) for c in word]
end

"""
    build_char_vocab(token_lists; include_word_boundary=true)

Build character-level vocabulary from token lists.
"""
function build_char_vocab(
    token_lists::Vector{Vector{String}};
    include_word_boundary::Bool = true
)
    chars = Set{String}()
    for tokens in token_lists
        for token in tokens
            for c in token
                push!(chars, string(c))
            end
        end
    end

    if include_word_boundary
        push!(chars, " ")
    end

    return Vocabulary(collect(chars))
end

# =============================================================================
# Subword Tokenizer (BPE-style)
# =============================================================================

"""
    SubwordTokenizer

Simple subword tokenizer using greedy longest-match.
For production, consider using BytePairEncoding.jl or similar.
"""
struct SubwordTokenizer
    vocab::Vocabulary
    max_length::Int
    lowercase::Bool
    add_special_tokens::Bool
    subword_prefix::String
end

function SubwordTokenizer(
    vocab::Vocabulary;
    max_length::Int = 512,
    lowercase::Bool = false,
    add_special_tokens::Bool = true,
    subword_prefix::String = "##"
)
    return SubwordTokenizer(vocab, max_length, lowercase, add_special_tokens, subword_prefix)
end

"""
    tokenize_word(tokenizer::SubwordTokenizer, word) -> Vector{String}

Greedy longest-match subword tokenization.
"""
function tokenize_word(tokenizer::SubwordTokenizer, word::String)
    if tokenizer.lowercase
        word = lowercase(word)
    end

    # Check if whole word is in vocab
    if haskey(tokenizer.vocab.token_to_id, word)
        return [word]
    end

    # Greedy longest match
    subwords = String[]
    start = 1
    is_first = true

    while start <= length(word)
        # Find longest matching subword
        end_pos = length(word)
        found = false

        while end_pos >= start
            subword = word[start:end_pos]

            # Add prefix for non-first subwords
            if !is_first
                subword = tokenizer.subword_prefix * subword
            end

            if haskey(tokenizer.vocab.token_to_id, subword)
                push!(subwords, subword)
                start = end_pos + 1
                is_first = false
                found = true
                break
            end

            end_pos -= 1
        end

        # If no match found, use single character
        if !found
            char = string(word[start])
            if !is_first
                char = tokenizer.subword_prefix * char
            end
            push!(subwords, char)
            start += 1
            is_first = false
        end
    end

    # If still empty, return [UNK]
    if isempty(subwords)
        return [UNK_TOKEN]
    end

    return subwords
end

# Make SubwordTokenizer work with tokenize_with_labels
function tokenize_with_labels(
    tokenizer::SubwordTokenizer,
    tokens::Vector{String},
    labels::Vector{Int};
    ignore_index::Int = -100
)
    token_ids = Int[]
    aligned_labels = Int[]

    # Add [CLS] token
    if tokenizer.add_special_tokens
        push!(token_ids, tokenizer.vocab.cls_token_id)
        push!(aligned_labels, ignore_index)
    end

    # Process each word
    for (word, label) in zip(tokens, labels)
        subwords = tokenize_word(tokenizer, word)

        for (i, subword) in enumerate(subwords)
            if length(token_ids) >= tokenizer.max_length - 1
                break
            end

            token_id = tokenizer.vocab[subword]
            push!(token_ids, token_id)

            if i == 1
                push!(aligned_labels, label)
            else
                push!(aligned_labels, ignore_index)
            end
        end

        if length(token_ids) >= tokenizer.max_length - 1
            break
        end
    end

    # Add [SEP] token
    if tokenizer.add_special_tokens
        push!(token_ids, tokenizer.vocab.sep_token_id)
        push!(aligned_labels, ignore_index)
    end

    seq_len = length(token_ids)
    attention_mask = fill(true, seq_len)

    return token_ids, aligned_labels, attention_mask
end

# =============================================================================
# Exports
# =============================================================================

export Vocabulary, NERTokenizer, CharTokenizer, SubwordTokenizer
export TokenizedBatch
export tokenize, tokenize_with_labels, tokenize_batch, decode
export tokenize_word, build_vocab_from_tokens, build_vocab_from_corpus
export build_char_vocab
export save_vocab, load_vocab
export PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN, SPECIAL_TOKENS

end # module
