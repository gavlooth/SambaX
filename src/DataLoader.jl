module DataLoader

"""
Data loading utilities for LLaDA training.

Supports:
- HuggingFace Datasets (automatic download)
- Local text files
- Character-level and BPE tokenization
"""

using Downloads
using JSON3
using Random

# ============================================================================
# Tokenizer
# ============================================================================

"""
    Tokenizer

Simple tokenizer supporting character-level or learned vocabulary.
"""
mutable struct Tokenizer
    vocab::Dict{String, Int}
    inverse_vocab::Dict{Int, String}
    special_tokens::Dict{Symbol, Int}
    vocab_size::Int

    function Tokenizer()
        # Initialize with special tokens
        vocab = Dict{String, Int}()
        special_tokens = Dict{Symbol, Int}(
            :pad => 0,
            :unk => 1,
            :bos => 2,
            :eos => 3,
            :mask => 4,
        )

        for (name, id) in special_tokens
            vocab[string(name)] = id
        end

        inverse_vocab = Dict(v => k for (k, v) in vocab)
        new(vocab, inverse_vocab, special_tokens, 5)
    end
end

"""
    build_vocab!(tokenizer, texts; max_vocab_size, min_freq)

Build vocabulary from texts. Uses character-level tokenization.
"""
function build_vocab!(
    tokenizer::Tokenizer,
    texts::Vector{String};
    max_vocab_size::Int = 32000,
    min_freq::Int = 1,
)
    # Count character frequencies
    char_counts = Dict{Char, Int}()
    for text in texts
        for char in text
            char_counts[char] = get(char_counts, char, 0) + 1
        end
    end

    # Filter by frequency and sort
    filtered = [(c, n) for (c, n) in char_counts if n >= min_freq]
    sort!(filtered, by = x -> -x[2])

    # Add to vocabulary (keep special tokens)
    next_id = tokenizer.vocab_size
    for (char, _) in filtered
        if next_id >= max_vocab_size
            break
        end
        char_str = string(char)
        if !haskey(tokenizer.vocab, char_str)
            tokenizer.vocab[char_str] = next_id
            tokenizer.inverse_vocab[next_id] = char_str
            next_id += 1
        end
    end

    tokenizer.vocab_size = next_id
    return tokenizer
end

"""
    encode(tokenizer, text) -> Vector{Int}

Encode text to token IDs.
"""
function encode(tokenizer::Tokenizer, text::String)
    tokens = Int[]
    push!(tokens, tokenizer.special_tokens[:bos])

    for char in text
        char_str = string(char)
        if haskey(tokenizer.vocab, char_str)
            push!(tokens, tokenizer.vocab[char_str])
        else
            push!(tokens, tokenizer.special_tokens[:unk])
        end
    end

    push!(tokens, tokenizer.special_tokens[:eos])
    return tokens
end

"""
    decode(tokenizer, token_ids) -> String

Decode token IDs back to text.
"""
function decode(tokenizer::Tokenizer, token_ids::Vector{Int})
    chars = String[]
    for id in token_ids
        if haskey(tokenizer.inverse_vocab, id)
            token = tokenizer.inverse_vocab[id]
            # Skip special tokens in output
            if !(token in ["pad", "unk", "bos", "eos", "mask"])
                push!(chars, token)
            end
        end
    end
    return join(chars)
end

"""
    get_mask_token_id(tokenizer) -> Int

Get the mask token ID for diffusion training.
"""
get_mask_token_id(tokenizer::Tokenizer) = tokenizer.special_tokens[:mask]

"""
    get_vocab_size(tokenizer) -> Int

Get vocabulary size.
"""
get_vocab_size(tokenizer::Tokenizer) = tokenizer.vocab_size

# ============================================================================
# HuggingFace Dataset Downloading
# ============================================================================

const HUGGINGFACE_DATASETS_URL = "https://datasets-server.huggingface.co"

"""
    HuggingFaceDataset

Represents a downloaded HuggingFace dataset.
"""
struct HuggingFaceDataset
    name::String
    config::String
    split::String
    data::Vector{Dict{String, Any}}
    text_column::String
end

"""
    list_hf_datasets()

Print some popular text datasets available on HuggingFace.
"""
function list_hf_datasets()
    datasets = [
        ("roneneldan/TinyStories", "default", "Small stories for LLM training (~2GB)"),
        ("wikitext", "wikitext-2-raw-v1", "Wikipedia text (~12MB)"),
        ("wikitext", "wikitext-103-raw-v1", "Wikipedia text (~500MB)"),
        ("openwebtext", "plain_text", "OpenAI's WebText (~40GB)"),
        ("bookcorpus", "plain_text", "Book corpus (~5GB)"),
        ("allenai/c4", "en", "Colossal Clean Crawled Corpus"),
        ("EleutherAI/pile", "default", "The Pile dataset (~800GB)"),
    ]

    println("Popular HuggingFace text datasets:")
    println("=" ^ 60)
    for (name, config, desc) in datasets
        println("  $name ($config)")
        println("    $desc")
        println()
    end
end

"""
    download_hf_dataset(dataset_name; config, split, num_rows, text_column, cache_dir)

Download a dataset from HuggingFace Hub.

Arguments:
- dataset_name: e.g., "roneneldan/TinyStories" or "wikitext"
- config: Dataset configuration (e.g., "wikitext-2-raw-v1")
- split: "train", "validation", or "test"
- num_rows: Maximum number of rows to download (default: 10000)
- text_column: Name of the text column (default: "text")
- cache_dir: Directory to cache downloaded data

Returns: HuggingFaceDataset
"""
function download_hf_dataset(
    dataset_name::String;
    config::String = "default",
    split::String = "train",
    num_rows::Int = 10000,
    text_column::String = "text",
    cache_dir::String = joinpath(homedir(), ".cache", "ossamma", "datasets"),
)
    # Create cache directory
    mkpath(cache_dir)

    # Check cache first
    safe_name = replace(dataset_name, "/" => "_")
    cache_file = joinpath(cache_dir, "$(safe_name)_$(config)_$(split)_$(num_rows).json")

    if isfile(cache_file)
        println("Loading from cache: $cache_file")
        data = JSON3.read(read(cache_file, String), Vector{Dict{String, Any}})
        return HuggingFaceDataset(dataset_name, config, split, data, text_column)
    end

    println("Downloading dataset: $dataset_name")
    println("  Config: $config")
    println("  Split: $split")
    println("  Rows: $num_rows")

    # Build API URL
    # Using the datasets-server API
    url = "$HUGGINGFACE_DATASETS_URL/rows?dataset=$dataset_name&config=$config&split=$split&offset=0&length=$num_rows"

    println("  URL: $url")

    try
        # Download
        response = Downloads.download(url, IOBuffer())
        json_str = String(take!(response))
        result = JSON3.read(json_str)

        # Extract rows
        if haskey(result, :rows)
            rows = result[:rows]
            data = [
                Dict{String, Any}(string(k) => v for (k, v) in pairs(row[:row]))
                for row in rows
            ]

            # Cache the result
            open(cache_file, "w") do f
                write(f, JSON3.write(data))
            end

            println("  Downloaded $(length(data)) rows")
            println("  Cached to: $cache_file")

            return HuggingFaceDataset(dataset_name, config, split, data, text_column)
        else
            error("Unexpected response format from HuggingFace API")
        end
    catch e
        println("Error downloading dataset: $e")
        println("\nTrying alternative method...")
        return download_hf_dataset_parquet(dataset_name; config, split, num_rows, text_column, cache_dir)
    end
end

"""
    download_hf_dataset_parquet(...)

Alternative download method using parquet files directly.
"""
function download_hf_dataset_parquet(
    dataset_name::String;
    config::String = "default",
    split::String = "train",
    num_rows::Int = 10000,
    text_column::String = "text",
    cache_dir::String = joinpath(homedir(), ".cache", "ossamma", "datasets"),
)
    # Try to get parquet URLs
    url = "$HUGGINGFACE_DATASETS_URL/parquet?dataset=$dataset_name&config=$config&split=$split"

    println("  Fetching parquet info: $url")

    try
        response = Downloads.download(url, IOBuffer())
        json_str = String(take!(response))
        result = JSON3.read(json_str)

        if haskey(result, :parquet_files)
            parquet_url = result[:parquet_files][1][:url]
            println("  Parquet URL: $parquet_url")
            println("\nNote: Direct parquet download requires additional setup.")
            println("Consider using the Python datasets library for large datasets.")
        end
    catch e
        println("Could not get parquet info: $e")
    end

    error("Failed to download dataset. Try using Python: `from datasets import load_dataset`")
end

"""
    get_texts(dataset::HuggingFaceDataset) -> Vector{String}

Extract text strings from dataset.
"""
function get_texts(dataset::HuggingFaceDataset)
    texts = String[]
    for row in dataset.data
        if haskey(row, dataset.text_column)
            text = row[dataset.text_column]
            if text isa String && !isempty(text)
                push!(texts, text)
            end
        end
    end
    return texts
end

# ============================================================================
# Data Loader
# ============================================================================

"""
    TextDataLoader

Iterable data loader for training.
"""
mutable struct TextDataLoader
    tokenizer::Tokenizer
    token_ids::Vector{Vector{Int}}
    seq_length::Int
    batch_size::Int
    current_idx::Int
    shuffle::Bool
    indices::Vector{Int}
    rng::Random.AbstractRNG
end

"""
    create_dataloader(tokenizer, texts; seq_length, batch_size, shuffle, rng)

Create a data loader from tokenized texts.
"""
function create_dataloader(
    tokenizer::Tokenizer,
    texts::Vector{String};
    seq_length::Int = 128,
    batch_size::Int = 32,
    shuffle::Bool = true,
    rng::Random.AbstractRNG = Random.default_rng(),
)
    # Tokenize all texts
    println("Tokenizing $(length(texts)) texts...")
    token_ids = [encode(tokenizer, text) for text in texts]

    # Filter to texts that are long enough
    token_ids = filter(t -> length(t) >= seq_length, token_ids)
    println("  $(length(token_ids)) texts long enough for seq_length=$seq_length")

    if isempty(token_ids)
        error("No texts are long enough. Try reducing seq_length or getting more data.")
    end

    indices = collect(1:length(token_ids))
    if shuffle
        Random.shuffle!(rng, indices)
    end

    return TextDataLoader(
        tokenizer,
        token_ids,
        seq_length,
        batch_size,
        1,
        shuffle,
        indices,
        rng,
    )
end

"""
    Base.iterate(loader::TextDataLoader)

Iterate over batches.
"""
function Base.iterate(loader::TextDataLoader, state=nothing)
    if loader.current_idx > length(loader.indices)
        # Reset for next epoch
        loader.current_idx = 1
        if loader.shuffle
            Random.shuffle!(loader.rng, loader.indices)
        end
        return nothing
    end

    # Build batch
    batch_indices = loader.indices[loader.current_idx:min(loader.current_idx + loader.batch_size - 1, length(loader.indices))]
    loader.current_idx += loader.batch_size

    # Create batch tensor (seq_length, batch_size)
    # Add 1 to all token IDs because Julia is 1-indexed and our vocab starts at 0
    batch = zeros(Int, loader.seq_length, length(batch_indices))
    for (i, idx) in enumerate(batch_indices)
        tokens = loader.token_ids[idx]
        # Random starting position for variety
        if length(tokens) > loader.seq_length
            start = rand(loader.rng, 1:length(tokens) - loader.seq_length + 1)
            batch[:, i] = tokens[start:start + loader.seq_length - 1] .+ 1
        else
            batch[1:length(tokens), i] = tokens .+ 1
        end
    end

    return batch, nothing
end

Base.length(loader::TextDataLoader) = ceil(Int, length(loader.indices) / loader.batch_size)

"""
    reset!(loader::TextDataLoader)

Reset loader to beginning.
"""
function reset!(loader::TextDataLoader)
    loader.current_idx = 1
    if loader.shuffle
        Random.shuffle!(loader.rng, loader.indices)
    end
end

# ============================================================================
# Convenience Functions
# ============================================================================

"""
    prepare_tinystories(; num_rows, seq_length, batch_size)

Download and prepare TinyStories dataset for training.
Returns (train_loader, val_loader, tokenizer).
"""
function prepare_tinystories(;
    num_train_rows::Int = 10000,
    num_val_rows::Int = 1000,
    seq_length::Int = 128,
    batch_size::Int = 32,
    max_vocab_size::Int = 10000,
    rng::Random.AbstractRNG = Random.default_rng(),
)
    println("=" ^ 60)
    println("Preparing TinyStories dataset")
    println("=" ^ 60)

    # Download training data
    train_dataset = download_hf_dataset(
        "roneneldan/TinyStories";
        split = "train",
        num_rows = num_train_rows,
    )

    # Download validation data
    val_dataset = download_hf_dataset(
        "roneneldan/TinyStories";
        split = "validation",
        num_rows = num_val_rows,
    )

    # Extract texts
    train_texts = get_texts(train_dataset)
    val_texts = get_texts(val_dataset)

    println("\nBuilding tokenizer...")
    tokenizer = Tokenizer()
    build_vocab!(tokenizer, train_texts; max_vocab_size = max_vocab_size)
    println("  Vocabulary size: $(get_vocab_size(tokenizer))")

    println("\nCreating data loaders...")
    train_loader = create_dataloader(tokenizer, train_texts; seq_length, batch_size, rng=rng)
    val_loader = create_dataloader(tokenizer, val_texts; seq_length, batch_size, shuffle=false, rng=rng)

    println("\nDataset ready!")
    println("  Train batches: $(length(train_loader))")
    println("  Val batches: $(length(val_loader))")
    println("  Vocab size: $(get_vocab_size(tokenizer))")
    println("  Mask token ID: $(get_mask_token_id(tokenizer) + 1)")  # +1 for Julia indexing

    return train_loader, val_loader, tokenizer
end

"""
    prepare_wikitext(; config, num_rows, seq_length, batch_size)

Download and prepare WikiText dataset for training.
Returns (train_loader, val_loader, tokenizer).
"""
function prepare_wikitext(;
    config::String = "wikitext-2-raw-v1",
    num_train_rows::Int = 5000,
    num_val_rows::Int = 500,
    seq_length::Int = 128,
    batch_size::Int = 32,
    max_vocab_size::Int = 10000,
    rng::Random.AbstractRNG = Random.default_rng(),
)
    println("=" ^ 60)
    println("Preparing WikiText dataset ($config)")
    println("=" ^ 60)

    # Download training data
    train_dataset = download_hf_dataset(
        "wikitext";
        config = config,
        split = "train",
        num_rows = num_train_rows,
    )

    # Download validation data
    val_dataset = download_hf_dataset(
        "wikitext";
        config = config,
        split = "validation",
        num_rows = num_val_rows,
    )

    # Extract texts
    train_texts = get_texts(train_dataset)
    val_texts = get_texts(val_dataset)

    # Filter empty texts
    train_texts = filter(t -> length(t) > 10, train_texts)
    val_texts = filter(t -> length(t) > 10, val_texts)

    println("\nBuilding tokenizer...")
    tokenizer = Tokenizer()
    build_vocab!(tokenizer, train_texts; max_vocab_size = max_vocab_size)
    println("  Vocabulary size: $(get_vocab_size(tokenizer))")

    println("\nCreating data loaders...")
    train_loader = create_dataloader(tokenizer, train_texts; seq_length, batch_size, rng=rng)
    val_loader = create_dataloader(tokenizer, val_texts; seq_length, batch_size, shuffle=false, rng=rng)

    println("\nDataset ready!")
    println("  Train batches: $(length(train_loader))")
    println("  Val batches: $(length(val_loader))")
    println("  Vocab size: $(get_vocab_size(tokenizer))")

    return train_loader, val_loader, tokenizer
end

"""
    prepare_custom_dataset(dataset_name; config, text_column, ...)

Download and prepare any HuggingFace text dataset.
Returns (train_loader, val_loader, tokenizer).
"""
function prepare_custom_dataset(
    dataset_name::String;
    config::String = "default",
    text_column::String = "text",
    num_train_rows::Int = 10000,
    num_val_rows::Int = 1000,
    seq_length::Int = 128,
    batch_size::Int = 32,
    max_vocab_size::Int = 10000,
    rng::Random.AbstractRNG = Random.default_rng(),
)
    println("=" ^ 60)
    println("Preparing dataset: $dataset_name")
    println("=" ^ 60)

    train_dataset = download_hf_dataset(
        dataset_name;
        config = config,
        split = "train",
        num_rows = num_train_rows,
        text_column = text_column,
    )

    val_dataset = try
        download_hf_dataset(
            dataset_name;
            config = config,
            split = "validation",
            num_rows = num_val_rows,
            text_column = text_column,
        )
    catch
        println("No validation split found, using subset of train")
        nothing
    end

    train_texts = get_texts(train_dataset)
    val_texts = val_dataset !== nothing ? get_texts(val_dataset) : train_texts[end-min(num_val_rows, length(train_texts)÷10)+1:end]

    if val_dataset === nothing
        train_texts = train_texts[1:end-length(val_texts)]
    end

    println("\nBuilding tokenizer...")
    tokenizer = Tokenizer()
    build_vocab!(tokenizer, train_texts; max_vocab_size = max_vocab_size)
    println("  Vocabulary size: $(get_vocab_size(tokenizer))")

    println("\nCreating data loaders...")
    train_loader = create_dataloader(tokenizer, train_texts; seq_length, batch_size, rng=rng)
    val_loader = create_dataloader(tokenizer, val_texts; seq_length, batch_size, shuffle=false, rng=rng)

    return train_loader, val_loader, tokenizer
end

# ============================================================================
# Direct Text Download (Reliable Fallback)
# ============================================================================

"""
    download_text_from_url(url; cache_dir) -> String

Download raw text directly from a URL. Useful as a fallback when HuggingFace APIs fail.
"""
function download_text_from_url(
    url::String;
    cache_dir::String = joinpath(homedir(), ".cache", "ossamma", "texts"),
)
    mkpath(cache_dir)

    # Create cache filename from URL
    safe_name = replace(replace(url, r"[^a-zA-Z0-9]" => "_"), r"_+" => "_")
    safe_name = safe_name[1:min(100, length(safe_name))]
    cache_file = joinpath(cache_dir, "$(safe_name).txt")

    if isfile(cache_file)
        println("Loading from cache: $cache_file")
        return read(cache_file, String)
    end

    println("Downloading: $url")
    try
        response = Downloads.download(url, IOBuffer())
        text = String(take!(response))

        # Cache the result
        open(cache_file, "w") do f
            write(f, text)
        end
        println("  Downloaded $(length(text)) characters")
        println("  Cached to: $cache_file")

        return text
    catch e
        error("Failed to download from $url: $e")
    end
end

# Project Gutenberg book URLs (public domain, reliable)
const GUTENBERG_BOOKS = Dict{Symbol, String}(
    :alice => "https://www.gutenberg.org/cache/epub/11/pg11.txt",           # Alice in Wonderland
    :pride => "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",       # Pride and Prejudice
    :sherlock => "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",    # Sherlock Holmes
    :frankenstein => "https://www.gutenberg.org/cache/epub/84/pg84.txt",    # Frankenstein
    :dracula => "https://www.gutenberg.org/cache/epub/345/pg345.txt",       # Dracula
    :moby_dick => "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",   # Moby Dick
    :war_peace => "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",   # War and Peace
    :tale_two_cities => "https://www.gutenberg.org/cache/epub/98/pg98.txt", # Tale of Two Cities
    :great_expectations => "https://www.gutenberg.org/cache/epub/1400/pg1400.txt", # Great Expectations
    :emma => "https://www.gutenberg.org/cache/epub/158/pg158.txt",          # Emma
)

"""
    prepare_gutenberg(; books, seq_length, batch_size, max_vocab_size, rng)

Download and prepare Project Gutenberg books for training.
This is a reliable fallback when HuggingFace APIs fail.

Arguments:
- books: List of book symbols from GUTENBERG_BOOKS, or :all for all books
- seq_length: Sequence length for training
- batch_size: Batch size
- max_vocab_size: Maximum vocabulary size
- rng: Random number generator

Returns (train_loader, val_loader, tokenizer).
"""
function prepare_gutenberg(;
    books::Union{Vector{Symbol}, Symbol} = [:alice, :pride, :sherlock],
    seq_length::Int = 128,
    batch_size::Int = 32,
    max_vocab_size::Int = 10000,
    val_split::Float64 = 0.1,
    rng::Random.AbstractRNG = Random.default_rng(),
)
    println("=" ^ 60)
    println("Preparing Project Gutenberg dataset")
    println("=" ^ 60)

    # Handle :all case
    book_list = if books == :all
        collect(keys(GUTENBERG_BOOKS))
    elseif books isa Symbol
        [books]
    else
        books
    end

    println("Books to download: $(join(book_list, ", "))")

    # Download all books
    all_texts = String[]
    for book in book_list
        if !haskey(GUTENBERG_BOOKS, book)
            println("Warning: Unknown book '$book', skipping")
            continue
        end

        url = GUTENBERG_BOOKS[book]
        println("\nDownloading: $book")
        try
            text = download_text_from_url(url)

            # Clean up Gutenberg header/footer
            text = clean_gutenberg_text(text)

            push!(all_texts, text)
            println("  Cleaned text: $(length(text)) characters")
        catch e
            println("  Failed to download $book: $e")
        end
    end

    if isempty(all_texts)
        error("No texts could be downloaded")
    end

    # Split texts into chunks for training
    println("\nProcessing texts...")
    chunk_size = seq_length * 4  # Reasonable chunk size
    chunks = String[]

    for text in all_texts
        # Split into paragraphs first
        paragraphs = split(text, r"\n\n+")
        current_chunk = ""

        for para in paragraphs
            para = strip(String(para))
            if isempty(para)
                continue
            end

            if length(current_chunk) + length(para) < chunk_size
                current_chunk *= " " * para
            else
                if length(current_chunk) > seq_length
                    push!(chunks, strip(current_chunk))
                end
                current_chunk = para
            end
        end

        if length(current_chunk) > seq_length
            push!(chunks, strip(current_chunk))
        end
    end

    println("  Created $(length(chunks)) text chunks")

    # Shuffle and split into train/val
    Random.shuffle!(rng, chunks)
    val_size = max(1, floor(Int, length(chunks) * val_split))
    train_texts = chunks[1:end-val_size]
    val_texts = chunks[end-val_size+1:end]

    println("  Train chunks: $(length(train_texts))")
    println("  Val chunks: $(length(val_texts))")

    # Build tokenizer
    println("\nBuilding tokenizer...")
    tokenizer = Tokenizer()
    build_vocab!(tokenizer, train_texts; max_vocab_size = max_vocab_size)
    println("  Vocabulary size: $(get_vocab_size(tokenizer))")

    # Create data loaders
    println("\nCreating data loaders...")
    train_loader = create_dataloader(tokenizer, train_texts; seq_length, batch_size, rng=rng)
    val_loader = create_dataloader(tokenizer, val_texts; seq_length, batch_size, shuffle=false, rng=rng)

    println("\nDataset ready!")
    println("  Train batches: $(length(train_loader))")
    println("  Val batches: $(length(val_loader))")
    println("  Vocab size: $(get_vocab_size(tokenizer))")
    println("  Mask token ID: $(get_mask_token_id(tokenizer) + 1)")

    return train_loader, val_loader, tokenizer
end

"""
    clean_gutenberg_text(text) -> String

Remove Project Gutenberg headers and footers from text.
"""
function clean_gutenberg_text(text::String)
    # Find start marker
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
    ]

    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]

    # Find content start
    content_start = 1
    for marker in start_markers
        idx = findfirst(marker, text)
        if idx !== nothing
            # Find the next newline after the marker
            newline_idx = findnext('\n', text, idx[end])
            if newline_idx !== nothing
                content_start = newline_idx + 1
            end
            break
        end
    end

    # Find content end
    content_end = length(text)
    for marker in end_markers
        idx = findfirst(marker, text)
        if idx !== nothing
            content_end = idx[1] - 1
            break
        end
    end

    cleaned = text[content_start:content_end]

    # Remove excessive whitespace
    cleaned = replace(cleaned, r"\r\n" => "\n")
    cleaned = replace(cleaned, r"\n{3,}" => "\n\n")
    cleaned = strip(cleaned)

    return cleaned
end

# ============================================================================
# Exports
# ============================================================================

export Tokenizer, build_vocab!, encode, decode, get_mask_token_id, get_vocab_size
export HuggingFaceDataset, list_hf_datasets, download_hf_dataset, get_texts
export TextDataLoader, create_dataloader, reset!
export prepare_tinystories, prepare_wikitext, prepare_custom_dataset
export prepare_gutenberg, download_text_from_url, GUTENBERG_BOOKS

end # module
