module CRF

"""
Linear-chain Conditional Random Field (CRF) for BIO sequence labeling.

Enforces valid BIO transitions:
- O can go to O or B-*
- B-X can go to O, B-*, or I-X
- I-X can go to O, B-*, or I-X (same type only)

This prevents invalid sequences like O → I-PERSON or I-PERSON → I-AGENCY.
"""

using Lux
using Random
using NNlib
using Statistics: mean

# Import label mappings from parent
import ..NER: RAG_LABELS, LABEL_TO_ID, ID_TO_LABEL, NUM_LABELS, ENTITY_TYPES

const LuxLayer = Lux.AbstractLuxLayer

# =============================================================================
# Transition Validity
# =============================================================================

"""
    is_valid_transition(from_label::String, to_label::String) -> Bool

Check if transitioning from `from_label` to `to_label` is valid in BIO scheme.
"""
function is_valid_transition(from_label::String, to_label::String)
    # O can go to O or B-*
    if from_label == "O"
        return to_label == "O" || startswith(to_label, "B-")
    end

    # B-X can go to O, B-*, or I-X (same entity type)
    if startswith(from_label, "B-")
        entity_type = from_label[3:end]
        return to_label == "O" ||
               startswith(to_label, "B-") ||
               to_label == "I-$entity_type"
    end

    # I-X can go to O, B-*, or I-X (same entity type)
    if startswith(from_label, "I-")
        entity_type = from_label[3:end]
        return to_label == "O" ||
               startswith(to_label, "B-") ||
               to_label == "I-$entity_type"
    end

    return true
end

"""
    is_valid_transition(from_id::Int, to_id::Int) -> Bool

Check transition validity using label IDs.
"""
function is_valid_transition(from_id::Int, to_id::Int)
    from_label = ID_TO_LABEL[from_id]
    to_label = ID_TO_LABEL[to_id]
    return is_valid_transition(from_label, to_label)
end

"""
    build_transition_mask() -> Matrix{Float32}

Build a mask matrix where valid transitions are 0 and invalid are -Inf.
"""
function build_transition_mask()
    n = NUM_LABELS
    mask = zeros(Float32, n, n)

    for i in 1:n
        for j in 1:n
            if !is_valid_transition(i, j)
                mask[i, j] = -Inf32
            end
        end
    end

    return mask
end

# Pre-compute the transition mask
const TRANSITION_MASK = build_transition_mask()

# =============================================================================
# LinearChainCRF Layer
# =============================================================================

"""
    LinearChainCRF <: LuxLayer

Linear-chain CRF for sequence labeling with learned transition scores.

The CRF models:
    P(y|x) ∝ exp(∑ᵢ emission[yᵢ, i] + ∑ᵢ transition[yᵢ₋₁, yᵢ])

Training uses negative log-likelihood:
    loss = log Z(x) - score(x, y*)

where Z(x) is the partition function (sum over all possible label sequences).
"""
struct LinearChainCRF <: LuxLayer
    num_labels::Int

    function LinearChainCRF(num_labels::Int = NUM_LABELS)
        new(num_labels)
    end
end

function Lux.initialparameters(rng::Random.AbstractRNG, crf::LinearChainCRF)
    n = crf.num_labels

    # Initialize transitions with small random values
    # Valid transitions start near 0, invalid transitions are masked during forward
    transitions = randn(rng, Float32, n, n) * 0.1f0

    # Start transitions: score for starting with each label
    # O and B-* are valid starts, I-* are not
    start_transitions = randn(rng, Float32, n) * 0.1f0
    for i in 1:n
        label = ID_TO_LABEL[i]
        if startswith(label, "I-")
            start_transitions[i] = -10000.0f0
        end
    end

    # End transitions: score for ending with each label
    # All labels are valid ends
    end_transitions = randn(rng, Float32, n) * 0.1f0

    return (
        transitions = transitions,
        start_transitions = start_transitions,
        end_transitions = end_transitions,
    )
end

Lux.initialstates(::Random.AbstractRNG, ::LinearChainCRF) = (;)

# =============================================================================
# Forward Algorithm (Log Partition Function)
# =============================================================================

"""
    log_sum_exp(x; dims=1)

Numerically stable log-sum-exp.
"""
function log_sum_exp(x::AbstractArray; dims=1)
    max_x = maximum(x, dims=dims)
    return max_x .+ log.(sum(exp.(x .- max_x), dims=dims))
end

"""
    forward_algorithm(emissions, mask, params) -> log_partition

Compute log partition function Z(x) using forward algorithm.

Arguments:
- emissions: (num_labels, seq_len, batch) - emission scores from encoder
- mask: (seq_len, batch) - attention mask (true = valid token)
- params: CRF parameters

Returns:
- log_partition: (batch,) - log Z(x) for each sequence
"""
function forward_algorithm(
    emissions::AbstractArray{T, 3},
    mask::AbstractMatrix{Bool},
    params
) where T
    num_labels, seq_len, batch_size = size(emissions)

    # Get masked transitions (valid transitions + learned scores)
    trans = params.transitions .+ TRANSITION_MASK

    # Initialize with start transitions + first emissions
    # alpha[j, b] = log score of all paths ending in label j at position 1
    alpha = params.start_transitions .+ emissions[:, 1, :]  # (num_labels, batch)

    # Forward pass
    for t in 2:seq_len
        # Expand alpha for broadcasting: (num_labels, 1, batch)
        alpha_expanded = reshape(alpha, num_labels, 1, batch_size)

        # Transition scores: (num_labels, num_labels)
        # trans[i, j] = score of transitioning from label i to label j

        # For each target label j:
        # new_alpha[j] = logsumexp_i(alpha[i] + trans[i, j] + emission[j])

        # Compute all pairwise scores: (from_label, to_label, batch)
        # alpha[i, b] + trans[i, j] for all i, j
        scores = alpha_expanded .+ reshape(trans, num_labels, num_labels, 1)

        # Sum over source labels: (to_label, batch)
        new_alpha = dropdims(log_sum_exp(scores, dims=1), dims=1)

        # Add emission scores: (num_labels, batch)
        new_alpha = new_alpha .+ emissions[:, t, :]

        # Apply mask: keep old alpha for padded positions
        mask_t = reshape(mask[t, :], 1, batch_size)
        alpha = ifelse.(mask_t, new_alpha, alpha)
    end

    # Add end transitions and sum over labels
    alpha = alpha .+ params.end_transitions
    log_partition = dropdims(log_sum_exp(alpha, dims=1), dims=1)  # (batch,)

    return log_partition
end

# =============================================================================
# Gold Score
# =============================================================================

"""
    compute_gold_score(emissions, labels, mask, params) -> gold_score

Compute the score of the gold label sequence.

Arguments:
- emissions: (num_labels, seq_len, batch)
- labels: (seq_len, batch) - gold label IDs
- mask: (seq_len, batch) - attention mask
- params: CRF parameters

Returns:
- gold_score: (batch,) - score of gold sequence for each batch element
"""
function compute_gold_score(
    emissions::AbstractArray{T, 3},
    labels::AbstractMatrix{<:Integer},
    mask::AbstractMatrix{Bool},
    params
) where T
    num_labels, seq_len, batch_size = size(emissions)

    # Start with start transitions for first labels
    score = zeros(T, batch_size)

    for b in 1:batch_size
        first_label = labels[1, b]
        score[b] = params.start_transitions[first_label] + emissions[first_label, 1, b]
    end

    # Accumulate emission and transition scores
    for t in 2:seq_len
        for b in 1:batch_size
            if mask[t, b]
                prev_label = labels[t-1, b]
                curr_label = labels[t, b]

                # Transition score
                score[b] += params.transitions[prev_label, curr_label]

                # Emission score
                score[b] += emissions[curr_label, t, b]
            end
        end
    end

    # Add end transitions
    for b in 1:batch_size
        # Find last valid position
        last_pos = findlast(mask[:, b])
        if last_pos !== nothing
            last_label = labels[last_pos, b]
            score[b] += params.end_transitions[last_label]
        end
    end

    return score
end

# =============================================================================
# CRF Loss (Negative Log Likelihood)
# =============================================================================

"""
    crf_loss(crf, emissions, labels, mask, params, state) -> (loss, state)

Compute CRF negative log-likelihood loss.

Arguments:
- crf: LinearChainCRF layer
- emissions: (num_labels, seq_len, batch) - logits from encoder
- labels: (seq_len, batch) - gold label IDs (use ignore_index for padding)
- mask: (seq_len, batch) - attention mask (true = valid)
- params: CRF parameters
- state: CRF state (unused)

Returns:
- loss: scalar mean NLL
- state: unchanged state
"""
function crf_loss(
    crf::LinearChainCRF,
    emissions::AbstractArray{T, 3},
    labels::AbstractMatrix{<:Integer},
    mask::AbstractMatrix{Bool},
    params,
    state
) where T
    # Log partition function (sum over all possible sequences)
    log_partition = forward_algorithm(emissions, mask, params)

    # Gold sequence score
    gold_score = compute_gold_score(emissions, labels, mask, params)

    # NLL = log Z - gold_score
    nll = log_partition .- gold_score

    return mean(nll), state
end

# =============================================================================
# Viterbi Decoding
# =============================================================================

"""
    viterbi_decode(crf, emissions, mask, params, state) -> (predictions, state)

Find the most likely label sequence using Viterbi algorithm.

Arguments:
- crf: LinearChainCRF layer
- emissions: (num_labels, seq_len, batch)
- mask: (seq_len, batch) - attention mask
- params: CRF parameters
- state: CRF state

Returns:
- predictions: (seq_len, batch) - best label sequence
- state: unchanged state
"""
function viterbi_decode(
    crf::LinearChainCRF,
    emissions::AbstractArray{T, 3},
    mask::AbstractMatrix{Bool},
    params,
    state
) where T
    num_labels, seq_len, batch_size = size(emissions)

    # Get masked transitions
    trans = params.transitions .+ TRANSITION_MASK

    # Viterbi scores and backpointers
    viterbi = zeros(T, num_labels, seq_len, batch_size)
    backpointers = zeros(Int, num_labels, seq_len, batch_size)

    # Initialize with start transitions + first emissions
    viterbi[:, 1, :] = params.start_transitions .+ emissions[:, 1, :]

    # Forward pass (find best paths)
    for t in 2:seq_len
        for b in 1:batch_size
            if !mask[t, b]
                # Copy previous scores for padded positions
                viterbi[:, t, b] = viterbi[:, t-1, b]
                continue
            end

            for j in 1:num_labels
                # Score of reaching label j at position t from each previous label
                scores = viterbi[:, t-1, b] .+ trans[:, j]

                # Best previous label
                best_i = argmax(scores)
                best_score = scores[best_i]

                viterbi[j, t, b] = best_score + emissions[j, t, b]
                backpointers[j, t, b] = best_i
            end
        end
    end

    # Add end transitions and find best final label
    predictions = zeros(Int, seq_len, batch_size)

    for b in 1:batch_size
        # Find last valid position
        last_pos = findlast(mask[:, b])
        if last_pos === nothing
            last_pos = 1
        end

        # Best final label
        final_scores = viterbi[:, last_pos, b] .+ params.end_transitions
        best_last = argmax(final_scores)
        predictions[last_pos, b] = best_last

        # Backtrack
        for t in (last_pos-1):-1:1
            predictions[t, b] = backpointers[predictions[t+1, b], t+1, b]
        end

        # Fill padding with O (label 1)
        for t in (last_pos+1):seq_len
            predictions[t, b] = 1
        end
    end

    return predictions, state
end

# =============================================================================
# Combined CRF + Encoder Model
# =============================================================================

"""
    CRFTagger <: LuxLayer

Combines an encoder (producing emissions) with a CRF layer for sequence labeling.
"""
struct CRFTagger{E} <: LuxLayer
    encoder::E
    crf::LinearChainCRF
end

function CRFTagger(encoder; num_labels::Int = NUM_LABELS)
    return CRFTagger(encoder, LinearChainCRF(num_labels))
end

function Lux.initialparameters(rng::Random.AbstractRNG, model::CRFTagger)
    return (
        encoder = Lux.initialparameters(rng, model.encoder),
        crf = Lux.initialparameters(rng, model.crf),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, model::CRFTagger)
    return (
        encoder = Lux.initialstates(rng, model.encoder),
        crf = Lux.initialstates(rng, model.crf),
    )
end

"""
Forward pass returning emissions (for training with crf_loss).
"""
function (model::CRFTagger)(token_ids, params, state)
    emissions, encoder_state = model.encoder(token_ids, params.encoder, state.encoder)
    new_state = (encoder = encoder_state, crf = state.crf)
    return emissions, new_state
end

"""
    predict(model::CRFTagger, token_ids, mask, params, state)

Get predictions using Viterbi decoding.
"""
function predict(model::CRFTagger, token_ids, mask, params, state)
    emissions, new_state = model(token_ids, params, state)
    predictions, _ = viterbi_decode(model.crf, emissions, mask, params.crf, state.crf)
    return predictions, new_state
end

# =============================================================================
# Exports
# =============================================================================

export LinearChainCRF, CRFTagger
export crf_loss, viterbi_decode, predict
export is_valid_transition, build_transition_mask
export log_sum_exp, forward_algorithm, compute_gold_score

end # module
