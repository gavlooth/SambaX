module NER

"""
Token-level NER using Ossamma architecture.

Unlike OssammaClassifier (sequence → single label), OssammaNER outputs
a label for each token position (sequence → sequence of labels).

RAG-optimized 9-label schema:
    PERSON, AGENCY, PLACE, ORGANISM, EVENT,
    INSTRUMENT, CREATIVE_WORK, DOMAIN, MEASURE
"""

using Lux
using Random
using NNlib
using Statistics: mean
using TOML
import CUDA
using Zygote: @ignore

# Import parent module components
import ..OssammaNERBlock
import ..TimeConditionedLayerNorm
import ..LinearChainCRF

const LuxLayer = Lux.AbstractLuxLayer

# =============================================================================
# NER Label Schema
# =============================================================================

# Few-NERD coarse labels (8 entity types + O = 17 labels)
const RAG_LABELS = [
    "O",           # Outside any entity
    "B-ART", "I-ART",
    "B-BUILDING", "I-BUILDING",
    "B-EVENT", "I-EVENT",
    "B-LOCATION", "I-LOCATION",
    "B-ORGANIZATION", "I-ORGANIZATION",
    "B-OTHER", "I-OTHER",
    "B-PERSON", "I-PERSON",
    "B-PRODUCT", "I-PRODUCT",
]

const NUM_LABELS = length(RAG_LABELS)  # 17 (O + 8 entity types × 2 for B/I)

const LABEL_TO_ID = Dict(label => i for (i, label) in enumerate(RAG_LABELS))
const ID_TO_LABEL = Dict(i => label for (i, label) in enumerate(RAG_LABELS))

# Entity types without B/I prefix
const ENTITY_TYPES = [
    "ART", "BUILDING", "EVENT", "LOCATION",
    "ORGANIZATION", "OTHER", "PERSON", "PRODUCT"
]

# =============================================================================
# Configuration
# =============================================================================

Base.@kwdef struct NERConfig
    # Architecture
    vocab_size::Int = 32000
    max_sequence_length::Int = 512
    embedding_dimension::Int = 256
    number_of_heads::Int = 4
    number_of_layers::Int = 4
    num_labels::Int = NUM_LABELS  # 19 for BIO tagging

    # Internal dimensions
    time_dimension::Int = 64
    state_dimension::Int = -1  # -1 means use embedding_dimension

    # Attention
    window_size::Int = 5

    # Oscillator SSM
    min_frequency::Float32 = 0.1f0
    max_frequency::Float32 = 10.0f0
    default_time_step::Float32 = 0.1f0

    # Training
    dropout_rate::Float32 = 0.1f0
    label_smoothing::Float32 = 0.0f0
    use_crf::Bool = true # New field: whether to use a CRF layer

    # FFN configuration (Option E)
    use_ffn::Bool = true           # Default: true (SwiGLU FFN after mixing)
    ffn_expansion::Float32 = 4f0 / 3f0  # FFN expansion factor (4/3 gives power-of-2 split)

    # GPU Parallelization (RTX 5090 optimization)
    use_parallel_scan::Bool = false    # Enable parallel associative scan (10-40× speedup)
    parallel_chunk_size::Int = 64      # Chunk size for parallel scan

    # Alpha mixing ablation options (all disabled by default)
    use_vector_gains::Bool = false      # Learnable per-dim gains s_g, s_l (+2d params/layer)
    use_per_head_alpha::Bool = false    # Per-head α instead of scalar (+d*h params/layer)
    use_branch_projections::Bool = false # Full d→d projections per branch (+2d² params/layer)
end

# =============================================================================
# TOML Configuration Loading
# =============================================================================

"""
    load_ner_config(path::String) -> NERConfig

Load NER configuration from a TOML file.

# Example
```julia
config = load_ner_config("configs/ner_production.toml")
model = OssammaNER(config)
```
"""
function load_ner_config(path::String)::NERConfig
    toml = TOML.parsefile(path)
    model = get(toml, "model", Dict())
    dims = get(model, "dimensions", Dict())
    attn = get(model, "attention", Dict())
    osc = get(model, "oscillator", Dict())
    reg = get(model, "regularization", Dict())
    ablation = get(model, "ablation", Dict())
    parallel = get(toml, "parallelization", Dict())

    return NERConfig(
        # Architecture
        vocab_size = get(model, "vocab_size", 32000),
        max_sequence_length = get(model, "max_sequence_length", 512),
        embedding_dimension = get(model, "embedding_dimension", 256),
        number_of_heads = get(model, "number_of_heads", 4),
        number_of_layers = get(model, "number_of_layers", 4),
        num_labels = get(model, "num_labels", NUM_LABELS),

        # Internal dimensions
        time_dimension = get(dims, "time_dimension", 64),
        state_dimension = get(dims, "state_dimension", -1),

        # Attention
        window_size = get(attn, "window_size", 5),

        # Oscillator SSM
        min_frequency = Float32(get(osc, "min_frequency", 0.1)),
        max_frequency = Float32(get(osc, "max_frequency", 10.0)),
        default_time_step = Float32(get(osc, "default_time_step", 0.1)),

        # Training/Regularization
        dropout_rate = Float32(get(reg, "dropout_rate", 0.1)),
        label_smoothing = Float32(get(reg, "label_smoothing", 0.0)),
        use_crf = get(reg, "use_crf", true),

        # FFN configuration
        use_ffn = get(ablation, "use_ffn", true),
        ffn_expansion = Float32(get(ablation, "ffn_expansion", 4.0 / 3.0)),

        # GPU Parallelization
        use_parallel_scan = get(parallel, "use_parallel_scan", false),
        parallel_chunk_size = get(parallel, "chunk_size", 64),

        # Alpha mixing ablation options (all disabled by default)
        use_vector_gains = get(ablation, "use_vector_gains", false),
        use_per_head_alpha = get(ablation, "use_per_head_alpha", false),
        use_branch_projections = get(ablation, "use_branch_projections", false),
    )
end

"""
    load_training_config(path::String) -> NamedTuple

Load training hyperparameters from a TOML file.
Returns a NamedTuple with training settings.
"""
function load_training_config(path::String)
    toml = TOML.parsefile(path)
    train = get(toml, "training", Dict())
    checkpoints = get(train, "checkpoints", Dict())
    data = get(toml, "data", Dict())
    hardware = get(toml, "hardware", Dict())

    return (
        # Training hyperparameters
        batch_size = get(train, "batch_size", 16),
        gradient_accumulation_steps = get(train, "gradient_accumulation_steps", 1),
        learning_rate = Float32(get(train, "learning_rate", 5e-4)),
        min_learning_rate = Float32(get(train, "min_learning_rate", 1e-6)),
        warmup_steps = get(train, "warmup_steps", 500),
        total_steps = get(train, "total_steps", 20000),
        gradient_clip = Float32(get(train, "gradient_clip", 1.0)),
        weight_decay = Float32(get(train, "weight_decay", 0.01)),

        # Checkpoints
        eval_every = get(checkpoints, "eval_every", 200),
        log_every = get(checkpoints, "log_every", 20),
        save_every = get(checkpoints, "save_every", 1000),
        push_every = get(checkpoints, "push_every", 0),

        # Data paths
        train_path = get(data, "train_path", ""),
        val_path = get(data, "val_path", ""),
        test_path = get(data, "test_path", ""),
        max_len = get(data, "max_len", 512),

        # Hardware
        device = Symbol(get(hardware, "device", "gpu")),
        mixed_precision = get(hardware, "mixed_precision", true),
        num_workers = get(hardware, "num_workers", 2),
    )
end

"""
    OssammaNER(config_path::String)

Create NER model from a TOML configuration file.

# Example
```julia
model = OssammaNER("configs/ner_production.toml")
```
"""
function OssammaNER(config_path::String)
    config = load_ner_config(config_path)
    return OssammaNER(config)
end

"""
    estimate_parameters(config::NERConfig) -> Int

Estimate the number of trainable parameters for a given configuration.
"""
function estimate_parameters(config::NERConfig)
    d = config.embedding_dimension
    h = config.number_of_heads
    L = config.number_of_layers
    V = config.vocab_size
    S = config.max_sequence_length
    d_t = config.time_dimension
    d_s = config.state_dimension == -1 ? d : config.state_dimension
    n_labels = config.num_labels

    # Embeddings
    token_emb = V * d
    pos_emb = S * d
    embeddings = token_emb + pos_emb

    # Per OssammaNERBlock
    time_cond_norm = d * 2 + d_t * d + d_t * d + d_t  # LayerNorm + Scale/Shift/AlphaBias proj
    glu_proj = d * 2d + 2d  # GLU projection
    glu_out_proj = d * d + d

    # LinearAttention
    lin_attn_qkvo = 4 * (d * d + d)
    lin_attn_features = 2 * (d ÷ h * 2 * d ÷ h)  # Approximate
    lin_attn_time = d_t * (d ÷ h) + d ÷ h
    lin_attn = lin_attn_qkvo + lin_attn_features + lin_attn_time

    # DLinOSS
    dlinoss = 3 * d_s + d_s * d + d * d_s  # log params + projections

    # Input gate (no bias)
    input_gate = d * d

    # SWAttention
    sw_attn = 4 * (d * d + d)

    # Alpha + output norm
    alpha_norm = d + 1 + d * 2

    per_block = time_cond_norm + glu_proj + glu_out_proj + lin_attn + dlinoss + input_gate + sw_attn + alpha_norm

    # Classification head
    class_head = d * 2 + d * n_labels + n_labels  # LayerNorm + Dense

    # Boundary head
    boundary_head = d * 2 + d * 2 + 2

    # CRF
    crf = n_labels * n_labels + 2 * n_labels

    total = embeddings + L * per_block + class_head + boundary_head + crf
    return total
end

"""
Print configuration summary with estimated parameters.
"""
function print_config_summary(config::NERConfig)
    params = estimate_parameters(config)
    params_m = params / 1_000_000

    println("=" ^ 60)
    println("OssammaNER Configuration Summary")
    println("=" ^ 60)
    println("Architecture:")
    println("  vocab_size:           $(config.vocab_size)")
    println("  max_sequence_length:  $(config.max_sequence_length)")
    println("  embedding_dimension:  $(config.embedding_dimension)")
    println("  number_of_heads:      $(config.number_of_heads)")
    println("  number_of_layers:     $(config.number_of_layers)")
    println("  num_labels:           $(config.num_labels)")
    println()
    println("Dimensions:")
    println("  time_dimension:       $(config.time_dimension)")
    println("  state_dimension:      $(config.state_dimension == -1 ? "$(config.embedding_dimension) (auto)" : config.state_dimension)")
    println("  window_size:          $(config.window_size)")
    println()
    println("Oscillator:")
    println("  min_frequency:        $(config.min_frequency)")
    println("  max_frequency:        $(config.max_frequency)")
    println("  default_time_step:    $(config.default_time_step)")
    println()
    println("Regularization:")
    println("  dropout_rate:         $(config.dropout_rate)")
    println("  label_smoothing:      $(config.label_smoothing)")
    println("  use_crf:              $(config.use_crf)")
    println()
    println("Estimated Parameters:   ~$(round(params_m, digits=1))M")
    println("=" ^ 60)
end

# =============================================================================
# Fixed Time Embedding (for NER, no diffusion)
# =============================================================================

struct FixedTimeEmbedding <: LuxLayer
    time_dimension::Int
    fixed_value::Float32
end

function FixedTimeEmbedding(time_dimension::Int; fixed_value::Float32 = 0.5f0)
    return FixedTimeEmbedding(time_dimension, fixed_value)
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::FixedTimeEmbedding)
    half_dim = layer.time_dimension ÷ 2
    freqs = exp.(-(log(10000.0f0)) .* collect(Float32, 0:half_dim-1) ./ half_dim)
    args = freqs .* layer.fixed_value
    init_embedding = vcat(sin.(args), cos.(args))
    return (embedding = init_embedding,)
end

Lux.initialstates(::Random.AbstractRNG, ::FixedTimeEmbedding) = (;)

function (layer::FixedTimeEmbedding)(batch_size::Int, params, state)
    embedding = repeat(reshape(params.embedding, :, 1), 1, batch_size)
    return embedding, state
end

# =============================================================================
# OssammaNER Model
# =============================================================================

struct OssammaNER{E, P, T, B, D, H, C, BH} <: LuxLayer
    vocab_size::Int
    max_sequence_length::Int
    embedding_dimension::Int
    number_of_heads::Int
    number_of_layers::Int
    num_labels::Int
    dropout_rate::Float32

    # Embeddings
    TokenEmbedding::E
    PositionEmbedding::P
    TimeEmbedding::T

    # Encoder blocks
    Blocks::B

    # Dropout before classification
    Dropout::D

    # Token classification head (per-token output)
    ClassificationHead::H

    # Conditional Random Field layer
    CRF::C

    # Boundary prediction head for auxiliary loss
    BoundaryHead::BH
end

"""
    OssammaNER(config::NERConfig)

Create NER model from configuration.
"""
function OssammaNER(config::NERConfig)
    state_dimension = config.state_dimension == -1 ? config.embedding_dimension : config.state_dimension

    return OssammaNER(;
        vocab_size = config.vocab_size,
        max_sequence_length = config.max_sequence_length,
        embedding_dimension = config.embedding_dimension,
        number_of_heads = config.number_of_heads,
        number_of_layers = config.number_of_layers,
        num_labels = config.num_labels,
        time_dimension = config.time_dimension,
        state_dimension = state_dimension,
        window_size = config.window_size, # Pass window_size from config
        min_frequency = config.min_frequency,
        max_frequency = config.max_frequency,
        default_time_step = config.default_time_step,
        dropout_rate = config.dropout_rate,
        use_ffn = config.use_ffn,                  # FFN configuration
        ffn_expansion = config.ffn_expansion,
        use_parallel_scan = config.use_parallel_scan,      # GPU parallelization
        parallel_chunk_size = config.parallel_chunk_size,
        # Alpha mixing ablation
        use_vector_gains = config.use_vector_gains,
        use_per_head_alpha = config.use_per_head_alpha,
        use_branch_projections = config.use_branch_projections,
    )
end

function OssammaNER(;
    vocab_size::Int,
    max_sequence_length::Int,
    embedding_dimension::Int,
    number_of_heads::Int,
    number_of_layers::Int,
    num_labels::Int = NUM_LABELS,
    time_dimension::Int = 64,
    state_dimension::Int = embedding_dimension,
    window_size::Int = 256, # Changed default window size to 256 as per docs/NER_TRAINING_PLAN.md
    min_frequency::Float32 = 0.1f0,
    max_frequency::Float32 = 10.0f0,
    default_time_step::Float32 = 0.1f0,
    dropout_rate::Float32 = 0.1f0,
    use_ffn::Bool = true,           # Default: true (SwiGLU FFN after mixing)
    ffn_expansion::Float32 = 4f0 / 3f0,  # FFN expansion factor
    use_parallel_scan::Bool = false,     # GPU parallelization (10-40× speedup)
    parallel_chunk_size::Int = 64,       # Chunk size for parallel scan
    # Alpha mixing ablation options (all disabled by default)
    use_vector_gains::Bool = false,      # Learnable per-dim gains s_g, s_l (+2d params/layer)
    use_per_head_alpha::Bool = false,    # Per-head α instead of scalar (+d*h params/layer)
    use_branch_projections::Bool = false, # Full d→d projections per branch (+2d² params/layer)
)
    # Build stack of OssammaNERBlocks (with dual gating)
    blocks = Tuple([
        OssammaNERBlock(
            embedding_dimension,
            max_sequence_length,
            number_of_heads,
            time_dimension;
            state_dimension = state_dimension,
            window_size = window_size,
            min_frequency = min_frequency,
            max_frequency = max_frequency,
            default_time_step = default_time_step,
            dropout_rate = dropout_rate,
            use_ffn = use_ffn,
            ffn_expansion = ffn_expansion,
            use_parallel_scan = use_parallel_scan,
            parallel_chunk_size = parallel_chunk_size,
            # Alpha mixing ablation
            use_vector_gains = use_vector_gains,
            use_per_head_alpha = use_per_head_alpha,
            use_branch_projections = use_branch_projections,
        )
        for _ in 1:number_of_layers
    ])

    return OssammaNER(
        vocab_size,
        max_sequence_length,
        embedding_dimension,
        number_of_heads,
        number_of_layers,
        num_labels,
        dropout_rate,
        # Embeddings
        Lux.Embedding(vocab_size => embedding_dimension),
        Lux.Embedding(max_sequence_length => embedding_dimension),
        FixedTimeEmbedding(time_dimension),
        # Encoder blocks
        blocks,
        # Dropout before classification
        Lux.Dropout(dropout_rate),
        # Per-token classification: LayerNorm → Dropout → Dense → Labels
        Lux.Chain(
            Lux.LayerNorm((embedding_dimension,)),
            Lux.Dropout(dropout_rate),
            Lux.Dense(embedding_dimension => num_labels),
        ),
        # CRF Layer
        LinearChainCRF(num_labels),
        # Boundary prediction head
        Lux.Chain(
            Lux.LayerNorm((embedding_dimension,)),
            Lux.Dense(embedding_dimension => 2), # 2 outputs: is_boundary or not
        )
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, model::OssammaNER)
    block_params = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(
        Tuple(Lux.initialparameters(rng, block) for block in model.Blocks)
    )

    return (
        TokenEmbedding = Lux.initialparameters(rng, model.TokenEmbedding),
        PositionEmbedding = Lux.initialparameters(rng, model.PositionEmbedding),
        TimeEmbedding = Lux.initialparameters(rng, model.TimeEmbedding),
        Blocks = block_params,
        Dropout = Lux.initialparameters(rng, model.Dropout),
        ClassificationHead = Lux.initialparameters(rng, model.ClassificationHead),
        CRF = Lux.initialparameters(rng, model.CRF),
        BoundaryHead = Lux.initialparameters(rng, model.BoundaryHead),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, model::OssammaNER)
    block_states = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(
        Tuple(Lux.initialstates(rng, block) for block in model.Blocks)
    )

    # Cache position indices to avoid allocation in forward pass
    position_indices = collect(1:model.max_sequence_length)

    return (
        TokenEmbedding = Lux.initialstates(rng, model.TokenEmbedding),
        PositionEmbedding = Lux.initialstates(rng, model.PositionEmbedding),
        TimeEmbedding = Lux.initialstates(rng, model.TimeEmbedding),
        Blocks = block_states,
        Dropout = Lux.initialstates(rng, model.Dropout),
        ClassificationHead = Lux.initialstates(rng, model.ClassificationHead),
        CRF = Lux.initialstates(rng, model.CRF),
        BoundaryHead = Lux.initialstates(rng, model.BoundaryHead),
        position_indices = position_indices,
    )
end

function (model::OssammaNER)(token_ids::AbstractArray, params, state)
    # token_ids: (seq_len,) or (seq_len, batch)
    # Output: ((emissions, boundary_logits), new_state)

    is_batched = ndims(token_ids) == 2
    seq_len = size(token_ids, 1)
    batch_size = is_batched ? size(token_ids, 2) : 1

    # Standardize to batched format
    token_ids_batched = is_batched ? token_ids : reshape(token_ids, :, 1)

    # =========================================================================
    # 1. Token Embedding
    # =========================================================================
    token_flat = vec(token_ids_batched)
    token_emb_flat, tok_state = model.TokenEmbedding(token_flat, params.TokenEmbedding, state.TokenEmbedding)
    # (embedding_dim, seq_len * batch)
    token_emb = reshape(token_emb_flat, model.embedding_dimension, seq_len, batch_size)

    # =========================================================================
    # 2. Position Embedding (using cached indices)
    # =========================================================================
    # Use copy instead of @view to avoid GPU scalar indexing issues
    position_indices = copy(state.position_indices[1:seq_len])
    pos_emb_raw, pos_state = model.PositionEmbedding(position_indices, params.PositionEmbedding, state.PositionEmbedding)
    pos_emb = reshape(pos_emb_raw, model.embedding_dimension, seq_len, 1)

    # =========================================================================
    # 3. Combine Embeddings
    # =========================================================================
    hidden = token_emb .+ pos_emb  # (embedding_dim, seq_len, batch)

    # =========================================================================
    # 4. Fixed Time Embedding
    # =========================================================================
    time_emb, time_state = model.TimeEmbedding(batch_size, params.TimeEmbedding, state.TimeEmbedding)

    # =========================================================================
    # 5. Process through OssammaBlocks
    # =========================================================================
    (hidden, block_states) = foldl(
        enumerate(model.Blocks);
        init = (hidden, ())
    ) do (h, states), (i, block)
        block_key = Symbol("Block_$i")
        block_params = params.Blocks[block_key]
        block_state = state.Blocks[block_key]

        new_h, new_block_state = block((h, time_emb), block_params, block_state)
        (new_h, (states..., new_block_state))
    end

    # =========================================================================
    # 6. Apply dropout before classification
    # =========================================================================
    # hidden: (embedding_dim, seq_len, batch)
    # Flatten for dropout
    hidden_flat = reshape(hidden, model.embedding_dimension, :)
    hidden_flat, dropout_state = model.Dropout(hidden_flat, params.Dropout, state.Dropout)

    # =========================================================================
    # 7. Per-token Classification
    # =========================================================================
    emissions_flat, head_state = model.ClassificationHead(hidden_flat, params.ClassificationHead, state.ClassificationHead)
    # emissions_flat: (num_labels, seq_len * batch)

    emissions = reshape(emissions_flat, model.num_labels, seq_len, batch_size)

    # =========================================================================
    # 8. Boundary Prediction Head (for auxiliary loss)
    # =========================================================================
    boundary_logits_flat, boundary_head_state = model.BoundaryHead(hidden_flat, params.BoundaryHead, state.BoundaryHead)
    # boundary_logits_flat: (2, seq_len * batch)

    boundary_logits = reshape(boundary_logits_flat, 2, seq_len, batch_size)

    # Remove batch dim if input wasn't batched
    final_emissions = is_batched ? emissions : dropdims(emissions, dims=3)
    final_boundary_logits = is_batched ? boundary_logits : dropdims(boundary_logits, dims=3)

    # =========================================================================
    # 9. Update State
    # =========================================================================
    new_block_states = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(
        block_states
    )

    new_state = (
        TokenEmbedding = tok_state,
        PositionEmbedding = pos_state,
        TimeEmbedding = time_state,
        Blocks = new_block_states,
        Dropout = dropout_state,
        ClassificationHead = head_state,
        CRF = state.CRF,
        BoundaryHead = boundary_head_state,
        position_indices = state.position_indices,
    )

    return (final_emissions, final_boundary_logits), new_state
end

# =============================================================================
# Convenience Constructors
# =============================================================================

"""
    tiny_ner(; vocab_size, kwargs...)

Tiny NER model for debugging.
"""
function tiny_ner(; vocab_size::Int = 1000, max_sequence_length::Int = 64, kwargs...)
    config = NERConfig(;
        vocab_size = vocab_size,
        max_sequence_length = max_sequence_length,
        embedding_dimension = 64,
        number_of_heads = 2,
        number_of_layers = 2,
        time_dimension = 32,
        kwargs...
    )
    return OssammaNER(config)
end

"""
    small_ner(; vocab_size, kwargs...)

Small NER model for experiments.
"""
function small_ner(; vocab_size::Int = 32000, max_sequence_length::Int = 256, kwargs...)
    config = NERConfig(;
        vocab_size = vocab_size,
        max_sequence_length = max_sequence_length,
        embedding_dimension = 256,
        number_of_heads = 4,
        number_of_layers = 4,
        time_dimension = 64,
        kwargs...
    )
    return OssammaNER(config)
end

"""
    base_ner(; vocab_size, kwargs...)

Base NER model for production.
"""
function base_ner(; vocab_size::Int = 32000, max_sequence_length::Int = 512, kwargs...)
    config = NERConfig(;
        vocab_size = vocab_size,
        max_sequence_length = max_sequence_length,
        embedding_dimension = 512,
        number_of_heads = 8,
        number_of_layers = 6,
        time_dimension = 128,
        kwargs...
    )
    return OssammaNER(config)
end

# =============================================================================
# Loss Functions
# =============================================================================

"""
    ner_cross_entropy(logits, labels; ignore_index=-100)

Cross-entropy loss for NER, ignoring padding tokens.
GPU-compatible using one-hot matrix multiplication.

- logits: (num_labels, seq_len, batch)
- labels: (seq_len, batch) integer labels
"""
function ner_cross_entropy(logits, labels; ignore_index::Int = -100)
    num_labels, seq_len, batch_size = size(logits)

    # Flatten
    logits_flat = reshape(logits, num_labels, :)  # (num_labels, n_positions)
    labels_flat = vec(labels)  # (n_positions,)
    n_positions = @ignore length(labels_flat)

    # Create mask for valid positions (use @ignore for non-differentiable comparison)
    valid_mask = @ignore labels_flat .!= ignore_index
    valid_count = @ignore Float32(sum(valid_mask))

    if @ignore valid_count < 1
        return sum(logits_flat) * 0.0f0  # Differentiable zero
    end

    # Create safe labels on same device (replace ignore_index with 1)
    safe_labels = @ignore ifelse.(valid_mask, labels_flat, 1)

    # Compute log softmax - this is the differentiable part
    log_probs = NNlib.logsoftmax(logits_flat, dims=1)  # (num_labels, n_positions)

    # Create one-hot matrix using broadcasting: (num_labels, n_positions)
    # one_hot[j, i] = 1.0 if safe_labels[i] == j, else 0.0
    # This must be done on the same device as logits
    label_indices = @ignore if logits_flat isa CUDA.CuArray
        CUDA.cu(collect(1:num_labels))
    else
        collect(1:num_labels)
    end

    # Broadcast: (num_labels, 1) == (1, n_positions) -> (num_labels, n_positions)
    one_hot_bool = @ignore reshape(label_indices, num_labels, 1) .== reshape(safe_labels, 1, n_positions)

    # Convert to Float32 on same device - this is a constant for backprop
    one_hot = @ignore if logits_flat isa CUDA.CuArray
        CUDA.cu(Float32.(Array(one_hot_bool)))
    else
        Float32.(one_hot_bool)
    end

    # Compute cross-entropy: -sum(one_hot .* log_probs, dims=1)
    # Gradients flow through log_probs, one_hot is treated as constant
    per_pos_ce = -sum(one_hot .* log_probs, dims=1)  # (1, n_positions)

    # Create mask for valid positions
    mask_float = @ignore if logits_flat isa CUDA.CuArray
        CUDA.cu(Float32.(valid_mask))
    else
        Float32.(valid_mask)
    end

    # Apply mask and compute mean
    masked_ce = vec(per_pos_ce) .* mask_float

    return sum(masked_ce) / valid_count
end

# =============================================================================
# Inference Utilities
# =============================================================================

"""
    predict_labels(model, params, state, token_ids) -> Vector{String}

Predict NER labels for a sequence.
"""
function predict_labels(model::OssammaNER, params, state, token_ids)
    logits, _ = model(token_ids, params, state)

    # Get argmax predictions
    predictions = vec(mapslices(argmax, logits, dims=1))

    # Convert to label strings
    return [ID_TO_LABEL[p] for p in predictions]
end

"""
    extract_entities(tokens, labels) -> Vector{NamedTuple}

Extract entities from tokens and BIO labels.
"""
function extract_entities(tokens::Vector{String}, labels::Vector{String})
    entities = NamedTuple{(:text, :label, :start, :end_), Tuple{String, String, Int, Int}}[]

    current_entity = nothing
    current_tokens = String[]
    current_start = 0

    for (i, (token, label)) in enumerate(zip(tokens, labels))
        if startswith(label, "B-")
            # Save previous entity if exists
            if current_entity !== nothing
                push!(entities, (
                    text = join(current_tokens, " "),
                    label = current_entity,
                    start = current_start,
                    end_ = i - 1,
                ))
            end

            # Start new entity
            current_entity = label[3:end]  # Remove "B-"
            current_tokens = [token]
            current_start = i

        elseif startswith(label, "I-") && current_entity == label[3:end]
            # Continue current entity
            push!(current_tokens, token)

        else
            # End current entity
            if current_entity !== nothing
                push!(entities, (
                    text = join(current_tokens, " "),
                    label = current_entity,
                    start = current_start,
                    end_ = i - 1,
                ))
                current_entity = nothing
                current_tokens = String[]
            end
        end
    end

    # Don't forget last entity
    if current_entity !== nothing
        push!(entities, (
            text = join(current_tokens, " "),
            label = current_entity,
            start = current_start,
            end_ = length(tokens),
        ))
    end

    return entities
end

# =============================================================================
# Exports
# =============================================================================

export OssammaNER, NERConfig
export tiny_ner, small_ner, base_ner
export ner_cross_entropy, predict_labels, extract_entities
export RAG_LABELS, ENTITY_TYPES, LABEL_TO_ID, ID_TO_LABEL, NUM_LABELS
export load_ner_config, load_training_config, estimate_parameters, print_config_summary

end # module
