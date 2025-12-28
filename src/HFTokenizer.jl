# HFTokenizer.jl - HuggingFace Tokenizer wrapper for Julia
#
# Wraps HuggingFace tokenizers via PyCall for use with Granite/Qwen3/Llama.
# Provides a Julia-native interface for tokenization.

module HFTokenizer

using PyCall

export HuggingFaceTokenizer, load_tokenizer
export encode, decode, batch_encode, batch_decode
export get_vocab_size, get_mask_token_id, get_pad_token_id

# =============================================================================
# Tokenizer Wrapper
# =============================================================================

"""
    HuggingFaceTokenizer

Wrapper around HuggingFace AutoTokenizer.
"""
struct HuggingFaceTokenizer
    py_tokenizer::PyObject
    model_name::String
    vocab_size::Int
    pad_token_id::Int
    mask_token_id::Int
    eos_token_id::Int
    bos_token_id::Union{Int, Nothing}
end

"""
    load_tokenizer(model_name::String; trust_remote_code=true) -> HuggingFaceTokenizer

Load a HuggingFace tokenizer by model name.

# Examples
```julia
# Granite 4.0
tokenizer = load_tokenizer("ibm-granite/granite-4.0-micro")

# Qwen3
tokenizer = load_tokenizer("Qwen/Qwen3-4B")

# Llama 3
tokenizer = load_tokenizer("meta-llama/Llama-3.1-8B")
```
"""
function load_tokenizer(model_name::String; trust_remote_code::Bool = true)
    # Import transformers
    transformers = pyimport("transformers")

    # Load tokenizer
    py_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code = trust_remote_code
    )

    # Extract special token IDs
    vocab_size = py_tokenizer.vocab_size
    pad_token_id = something(py_tokenizer.pad_token_id, py_tokenizer.eos_token_id)
    eos_token_id = py_tokenizer.eos_token_id

    # Handle mask token (may not exist for all models)
    mask_token_id = try
        py_tokenizer.mask_token_id
    catch
        # If no mask token, use a placeholder
        vocab_size - 1
    end

    bos_token_id = try
        py_tokenizer.bos_token_id
    catch
        nothing
    end

    return HuggingFaceTokenizer(
        py_tokenizer,
        model_name,
        vocab_size,
        pad_token_id,
        mask_token_id !== nothing ? mask_token_id : vocab_size - 1,
        eos_token_id,
        bos_token_id
    )
end

# =============================================================================
# Tokenizer Interface
# =============================================================================

"""
    encode(tokenizer, text; add_special_tokens=true, max_length=nothing) -> Vector{Int}

Encode text to token IDs.
"""
function encode(
    tokenizer::HuggingFaceTokenizer,
    text::String;
    add_special_tokens::Bool = true,
    max_length::Union{Int, Nothing} = nothing
)
    kwargs = Dict{Symbol,Any}(
        :add_special_tokens => add_special_tokens,
        :return_tensors => nothing
    )

    if max_length !== nothing
        kwargs[:max_length] = max_length
        kwargs[:truncation] = true
    end

    encoding = tokenizer.py_tokenizer(text; kwargs...)
    input_ids = encoding["input_ids"]

    # Convert to Julia array (1-indexed)
    return [Int(id) + 1 for id in input_ids]
end

"""
    decode(tokenizer, token_ids; skip_special_tokens=true) -> String

Decode token IDs to text.
"""
function decode(
    tokenizer::HuggingFaceTokenizer,
    token_ids::AbstractVector{<:Integer};
    skip_special_tokens::Bool = true
)
    # Convert back to 0-indexed for Python
    py_ids = [id - 1 for id in token_ids]
    return tokenizer.py_tokenizer.decode(py_ids, skip_special_tokens=skip_special_tokens)
end

"""
    batch_encode(tokenizer, texts; add_special_tokens=true, max_length=nothing, padding=true)

Batch encode multiple texts.

Returns:
- `input_ids`: Matrix of token IDs (seq_len, batch)
- `attention_mask`: Matrix of attention masks (seq_len, batch)
"""
function batch_encode(
    tokenizer::HuggingFaceTokenizer,
    texts::Vector{String};
    add_special_tokens::Bool = true,
    max_length::Union{Int, Nothing} = nothing,
    padding::Bool = true
)
    kwargs = Dict{Symbol,Any}(
        :add_special_tokens => add_special_tokens,
        :padding => padding,
        :return_tensors => "np"  # NumPy for easy conversion
    )

    if max_length !== nothing
        kwargs[:max_length] = max_length
        kwargs[:truncation] = true
    end

    encoding = tokenizer.py_tokenizer(texts; kwargs...)

    # Convert to Julia arrays (1-indexed, transposed to seq_len × batch)
    np = pyimport("numpy")
    input_ids_np = encoding["input_ids"]
    attention_mask_np = encoding["attention_mask"]

    input_ids = Array{Int}(input_ids_np) .+ 1  # batch × seq_len
    attention_mask = Array{Bool}(attention_mask_np)  # batch × seq_len

    # Transpose to (seq_len, batch) convention
    return (
        input_ids = permutedims(input_ids, (2, 1)),
        attention_mask = permutedims(attention_mask, (2, 1))
    )
end

"""
    batch_decode(tokenizer, token_ids; skip_special_tokens=true) -> Vector{String}

Batch decode token IDs to texts.
"""
function batch_decode(
    tokenizer::HuggingFaceTokenizer,
    token_ids::AbstractMatrix{<:Integer};
    skip_special_tokens::Bool = true
)
    batch_size = size(token_ids, 2)
    results = String[]

    for b in 1:batch_size
        ids = token_ids[:, b]
        push!(results, decode(tokenizer, ids; skip_special_tokens))
    end

    return results
end

# =============================================================================
# Utility Functions
# =============================================================================

get_vocab_size(t::HuggingFaceTokenizer) = t.vocab_size
get_mask_token_id(t::HuggingFaceTokenizer) = t.mask_token_id + 1  # Julia 1-indexed
get_pad_token_id(t::HuggingFaceTokenizer) = t.pad_token_id + 1    # Julia 1-indexed

function Base.show(io::IO, t::HuggingFaceTokenizer)
    print(io, "HuggingFaceTokenizer(\"$(t.model_name)\", vocab_size=$(t.vocab_size))")
end

# =============================================================================
# Preset Tokenizers
# =============================================================================

"""
    load_granite_tokenizer(; model="ibm-granite/granite-4.0-micro")

Load the Granite 4.0 tokenizer.
"""
function load_granite_tokenizer(; model::String = "ibm-granite/granite-4.0-micro")
    return load_tokenizer(model)
end

"""
    load_qwen3_tokenizer(; model="Qwen/Qwen3-4B")

Load the Qwen3 tokenizer.
"""
function load_qwen3_tokenizer(; model::String = "Qwen/Qwen3-4B")
    return load_tokenizer(model)
end

export load_granite_tokenizer, load_qwen3_tokenizer

end # module HFTokenizer
