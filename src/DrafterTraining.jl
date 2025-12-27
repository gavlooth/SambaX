# DrafterTraining.jl - Training utilities for OssammaDrafter
#
# Provides:
# - Distillation loss (KL divergence from AR model)
# - Masked language model loss
# - Checkpoint saving/loading specific to drafter
# - Training loop utilities

module DrafterTraining

using Lux
using LuxCore
using NNlib
using Random
using Statistics
using Serialization
using Dates

export drafter_mlm_loss, distillation_loss, combined_drafter_loss
export save_drafter_checkpoint, load_drafter_checkpoint
export DrafterTrainingConfig, create_drafter_training_state

# =============================================================================
# Loss Functions
# =============================================================================

"""
    drafter_mlm_loss(logits, targets, mask_positions; ignore_index=-100)

Compute masked language model loss for drafter training.

# Arguments
- `logits`: (vocab_size, seq_len, batch) model output
- `targets`: (seq_len, batch) ground truth token IDs
- `mask_positions`: (seq_len, batch) boolean mask (true = compute loss)
- `ignore_index`: Token ID to ignore in loss computation

# Returns
- Scalar loss value (mean cross-entropy over masked positions)
"""
function drafter_mlm_loss(
    logits::AbstractArray,
    targets::AbstractArray,
    mask_positions::AbstractArray;
    ignore_index::Int = -100
)
    vocab_size = size(logits, 1)

    # Flatten for easier computation
    logits_flat = reshape(logits, vocab_size, :)  # (V, seq*batch)
    targets_flat = vec(targets)                    # (seq*batch,)
    mask_flat = vec(mask_positions)                # (seq*batch,)

    # Compute log softmax
    log_probs = NNlib.logsoftmax(logits_flat, dims=1)  # (V, seq*batch)

    # Gather log probs at target positions
    # targets_flat contains token IDs, need to gather those positions
    total_loss = 0.0f0
    count = 0

    for i in eachindex(targets_flat)
        if mask_flat[i] && targets_flat[i] != ignore_index
            target_id = targets_flat[i]
            if 1 <= target_id <= vocab_size
                total_loss -= log_probs[target_id, i]
                count += 1
            end
        end
    end

    return count > 0 ? total_loss / count : 0.0f0
end

"""
    distillation_loss(student_logits, teacher_logits, mask_positions; temperature=1.0)

Compute KL divergence loss between student (drafter) and teacher (AR model).

# Arguments
- `student_logits`: (vocab_size, seq_len, batch) drafter output
- `teacher_logits`: (vocab_size, seq_len, batch) AR model output
- `mask_positions`: (seq_len, batch) boolean mask (true = compute loss)
- `temperature`: Softmax temperature for distillation (higher = softer)

# Returns
- Scalar KL divergence loss
"""
function distillation_loss(
    student_logits::AbstractArray,
    teacher_logits::AbstractArray,
    mask_positions::AbstractArray;
    temperature::Float32 = 1.0f0
)
    vocab_size = size(student_logits, 1)

    # Apply temperature scaling
    student_scaled = student_logits ./ temperature
    teacher_scaled = teacher_logits ./ temperature

    # Flatten
    student_flat = reshape(student_scaled, vocab_size, :)
    teacher_flat = reshape(teacher_scaled, vocab_size, :)
    mask_flat = vec(mask_positions)

    # Compute softmax distributions
    student_probs = NNlib.softmax(student_flat, dims=1)
    teacher_probs = NNlib.softmax(teacher_flat, dims=1)

    # KL divergence: sum_i p_teacher(i) * log(p_teacher(i) / p_student(i))
    # = sum_i p_teacher(i) * (log_p_teacher(i) - log_p_student(i))
    log_student = log.(student_probs .+ 1f-10)
    log_teacher = log.(teacher_probs .+ 1f-10)

    kl_per_position = sum(teacher_probs .* (log_teacher .- log_student), dims=1)
    kl_flat = vec(kl_per_position)

    # Average over masked positions
    total_kl = 0.0f0
    count = 0
    for i in eachindex(mask_flat)
        if mask_flat[i]
            total_kl += kl_flat[i]
            count += 1
        end
    end

    # Scale by temperature^2 (standard distillation practice)
    return count > 0 ? (total_kl / count) * temperature^2 : 0.0f0
end

"""
    combined_drafter_loss(student_logits, targets, teacher_logits, mask_positions;
                          alpha=0.5, temperature=1.0)

Combined loss: α * MLM_loss + (1-α) * distillation_loss

# Arguments
- `student_logits`: Drafter output logits
- `targets`: Ground truth token IDs
- `teacher_logits`: AR model output logits (or nothing for pure MLM)
- `mask_positions`: Boolean mask for loss computation
- `alpha`: Weight for MLM loss (1-alpha for distillation)
- `temperature`: Distillation temperature

# Returns
- (total_loss, mlm_loss, distill_loss) tuple
"""
function combined_drafter_loss(
    student_logits::AbstractArray,
    targets::AbstractArray,
    teacher_logits::Union{AbstractArray, Nothing},
    mask_positions::AbstractArray;
    alpha::Float32 = 0.5f0,
    temperature::Float32 = 1.0f0
)
    # MLM loss (always computed)
    mlm = drafter_mlm_loss(student_logits, targets, mask_positions)

    # Distillation loss (if teacher provided)
    distill = if teacher_logits !== nothing
        distillation_loss(student_logits, teacher_logits, mask_positions; temperature)
    else
        0.0f0
    end

    # Combined
    total = alpha * mlm + (1.0f0 - alpha) * distill

    return (total=total, mlm=mlm, distill=distill)
end

# =============================================================================
# Checkpoint Management
# =============================================================================

"""
    save_drafter_checkpoint(path; model_config, params, state, opt_state, step, loss, metadata)

Save a drafter checkpoint with full configuration.
"""
function save_drafter_checkpoint(
    path::String;
    model_config,
    params,
    state,
    opt_state = nothing,
    step::Int = 0,
    loss = nothing,
    metadata::Dict = Dict()
)
    # Ensure directory exists
    mkpath(dirname(path))

    data = Dict{Symbol,Any}(
        :model_config => model_config,
        :params => params,
        :state => state,
        :opt_state => opt_state,
        :step => step,
        :loss => loss,
        :timestamp => now(),
        :metadata => metadata,
    )

    serialize(path, data)
    return path
end

"""
    load_drafter_checkpoint(path) -> NamedTuple

Load a drafter checkpoint.

Returns a NamedTuple with fields:
- `model_config`: DrafterConfig
- `params`: Model parameters
- `state`: Model state
- `opt_state`: Optimizer state (may be nothing)
- `step`: Training step
- `loss`: Last loss value
- `timestamp`: When checkpoint was saved
- `metadata`: Additional metadata
"""
function load_drafter_checkpoint(path::String)
    data = deserialize(path)
    return (
        model_config = data[:model_config],
        params = data[:params],
        state = data[:state],
        opt_state = get(data, :opt_state, nothing),
        step = get(data, :step, 0),
        loss = get(data, :loss, nothing),
        timestamp = get(data, :timestamp, nothing),
        metadata = get(data, :metadata, Dict()),
    )
end

# =============================================================================
# Training Configuration
# =============================================================================

"""
    DrafterTrainingConfig

Configuration for drafter training.
"""
Base.@kwdef struct DrafterTrainingConfig
    # Data
    train_data_path::String = ""
    val_data_path::String = ""

    # Training
    batch_size::Int = 32
    max_steps::Int = 100000
    learning_rate::Float64 = 1e-4
    weight_decay::Float64 = 0.01
    warmup_steps::Int = 1000

    # Loss
    alpha::Float32 = 0.5f0           # MLM weight (1-alpha = distillation weight)
    temperature::Float32 = 1.0f0     # Distillation temperature
    mask_ratio::Float32 = 0.15f0     # Fraction of tokens to mask
    mask_strategy::Symbol = :mixed   # :random | :suffix | :mixed
    draft_length::Int = 8            # Suffix length for TiDAR-style masking
    suffix_prob::Float32 = 0.5f0     # Probability of suffix masking (mixed)

    # Checkpointing
    checkpoint_dir::String = "checkpoints/drafter"
    save_every::Int = 1000
    eval_every::Int = 500

    # Logging
    log_every::Int = 100
end

"""
    create_drafter_training_state(model, config; rng=Random.default_rng())

Initialize training state for drafter.
"""
function create_drafter_training_state(model, config::DrafterTrainingConfig; rng=Random.default_rng())
    params = Lux.initialparameters(rng, model)
    state = Lux.initialstates(rng, model)

    return (
        params = params,
        state = state,
        step = 0,
        best_loss = Inf,
    )
end

# =============================================================================
# Masking Utilities
# =============================================================================

"""
    apply_random_mask(token_ids, mask_token_id, mask_ratio; rng=Random.default_rng())

Apply random masking to token IDs for MLM training.

# Returns
- `masked_ids`: Token IDs with some replaced by mask_token_id
- `mask_positions`: Boolean array indicating masked positions
- `original_ids`: Original token IDs at masked positions
"""
function apply_random_mask(
    token_ids::AbstractArray,
    mask_token_id::Int,
    mask_ratio::Float32;
    rng = Random.default_rng()
)
    masked_ids = copy(token_ids)
    mask_positions = falses(size(token_ids))

    for i in eachindex(token_ids)
        if rand(rng) < mask_ratio
            mask_positions[i] = true
            masked_ids[i] = mask_token_id
        end
    end

    return (
        masked_ids = masked_ids,
        mask_positions = mask_positions,
        original_ids = token_ids,
    )
end

"""
    apply_block_mask(token_ids, mask_token_id, block_size, start_pos)

Apply block masking for TiDAR-style training.
Masks a contiguous block of tokens starting at start_pos.
"""
function apply_block_mask(
    token_ids::AbstractVector,
    mask_token_id::Int,
    block_size::Int,
    start_pos::Int
)
    seq_len = length(token_ids)
    end_pos = min(start_pos + block_size - 1, seq_len)

    masked_ids = copy(token_ids)
    mask_positions = falses(seq_len)

    for i in start_pos:end_pos
        masked_ids[i] = mask_token_id
        mask_positions[i] = true
    end

    return (
        masked_ids = masked_ids,
        mask_positions = mask_positions,
        original_ids = token_ids,
    )
end

"""
    apply_suffix_mask(token_ids, mask_token_id, block_size)

Apply suffix masking for TiDAR-style training.
Masks the last `block_size` tokens of the sequence.
"""
function apply_suffix_mask(
    token_ids::AbstractVector,
    mask_token_id::Int,
    block_size::Int
)
    seq_len = length(token_ids)
    if block_size <= 0
        return (
            masked_ids = copy(token_ids),
            mask_positions = falses(seq_len),
            original_ids = token_ids,
        )
    end

    start_pos = max(seq_len - block_size + 1, 1)
    return apply_block_mask(token_ids, mask_token_id, block_size, start_pos)
end

export apply_random_mask, apply_block_mask, apply_suffix_mask, DrafterTrainingConfig

end # module DrafterTraining
