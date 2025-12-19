module NERMetrics

"""
Comprehensive NER evaluation metrics.

Provides:
- Token-level accuracy
- Span-level precision, recall, F1 (strict matching)
- Per-entity type metrics
- Confusion matrix
- Partial matching metrics (for analysis)

Span-level evaluation follows CoNLL-2003 standard:
- An entity is correct only if both boundary AND type match exactly
"""

using Statistics: mean
using Printf

# =============================================================================
# Entity Span for Evaluation
# =============================================================================

"""
    EvalSpan

Represents an entity span for evaluation purposes.
"""
struct EvalSpan
    start_idx::Int
    end_idx::Int
    entity_type::String
end

function Base.:(==)(a::EvalSpan, b::EvalSpan)
    return a.start_idx == b.start_idx &&
           a.end_idx == b.end_idx &&
           a.entity_type == b.entity_type
end

function Base.hash(span::EvalSpan, h::UInt)
    return hash(span.start_idx, hash(span.end_idx, hash(span.entity_type, h)))
end

"""
    extract_eval_spans(labels, id_to_label) -> Set{EvalSpan}

Extract entity spans from a label sequence for evaluation.
"""
function extract_eval_spans(
    labels::Vector{Int},
    id_to_label::Dict{Int, String}
)
    spans = Set{EvalSpan}()
    i = 1

    while i <= length(labels)
        label = id_to_label[labels[i]]

        if startswith(label, "B-")
            entity_type = label[3:end]
            start_idx = i
            i += 1

            # Find extent
            while i <= length(labels)
                next_label = id_to_label[labels[i]]
                if next_label == "I-$entity_type"
                    i += 1
                else
                    break
                end
            end

            push!(spans, EvalSpan(start_idx, i - 1, entity_type))
        else
            i += 1
        end
    end

    return spans
end

# =============================================================================
# Metrics Computation
# =============================================================================

"""
    NERResults

Container for NER evaluation results.
"""
struct NERResults
    # Overall metrics (micro-averaged)
    f1_micro::Float64
    precision_micro::Float64
    recall_micro::Float64

    # Macro-averaged metrics
    f1_macro::Float64
    precision_macro::Float64
    recall_macro::Float64

    # Per-entity metrics
    per_entity_f1::Dict{String, Float64}
    per_entity_precision::Dict{String, Float64}
    per_entity_recall::Dict{String, Float64}
    per_entity_support::Dict{String, Int}

    # Token-level accuracy
    token_accuracy::Float64

    # Confusion matrix (optional)
    confusion_matrix::Union{Matrix{Int}, Nothing}

    # Counts for debugging
    total_predictions::Int
    total_gold::Int
    total_correct::Int
end

function Base.show(io::IO, results::NERResults)
    println(io, "NER Evaluation Results:")
    println(io, "=" ^ 50)
    @printf(io, "  Micro F1:     %.4f\n", results.f1_micro)
    @printf(io, "  Precision:    %.4f\n", results.precision_micro)
    @printf(io, "  Recall:       %.4f\n", results.recall_micro)
    println(io, "-" ^ 50)
    @printf(io, "  Macro F1:     %.4f\n", results.f1_macro)
    @printf(io, "  Token Acc:    %.4f\n", results.token_accuracy)
    println(io, "-" ^ 50)
    println(io, "  Per-entity F1:")
    for (entity, f1) in sort(collect(results.per_entity_f1))
        support = get(results.per_entity_support, entity, 0)
        @printf(io, "    %-12s: %.4f (n=%d)\n", entity, f1, support)
    end
end

"""
    compute_f1(precision, recall) -> Float64

Compute F1 score from precision and recall.
"""
function compute_f1(precision::Float64, recall::Float64)
    if precision + recall == 0
        return 0.0
    end
    return 2 * precision * recall / (precision + recall)
end

"""
    evaluate_ner(predictions, gold_labels, id_to_label; entity_types=nothing) -> NERResults

Compute comprehensive NER evaluation metrics.

Arguments:
- predictions: Vector of prediction sequences (Vector{Vector{Int}})
- gold_labels: Vector of gold label sequences (Vector{Vector{Int}})
- id_to_label: Dict mapping label IDs to strings
- entity_types: Optional list of entity types to evaluate (default: all found)

Returns: NERResults struct with all metrics
"""
function evaluate_ner(
    predictions::Vector{Vector{Int}},
    gold_labels::Vector{Vector{Int}},
    id_to_label::Dict{Int, String};
    entity_types::Union{Vector{String}, Nothing} = nothing,
    compute_confusion::Bool = false
)
    @assert length(predictions) == length(gold_labels) "Predictions and gold must have same length"

    # Extract entity types from id_to_label if not provided
    if entity_types === nothing
        entity_types = String[]
        for label in values(id_to_label)
            if startswith(label, "B-")
                push!(entity_types, label[3:end])
            end
        end
        entity_types = unique(entity_types)
    end

    # Initialize counters
    true_positives = Dict{String, Int}()
    false_positives = Dict{String, Int}()
    false_negatives = Dict{String, Int}()

    for entity in entity_types
        true_positives[entity] = 0
        false_positives[entity] = 0
        false_negatives[entity] = 0
    end

    # Token-level counters
    total_tokens = 0
    correct_tokens = 0

    # Process each sequence
    for (pred_seq, gold_seq) in zip(predictions, gold_labels)
        # Token-level accuracy
        min_len = min(length(pred_seq), length(gold_seq))
        for i in 1:min_len
            total_tokens += 1
            if pred_seq[i] == gold_seq[i]
                correct_tokens += 1
            end
        end

        # Span-level evaluation
        pred_spans = extract_eval_spans(pred_seq, id_to_label)
        gold_spans = extract_eval_spans(gold_seq, id_to_label)

        # Count true positives (correct predictions)
        for span in pred_spans
            if span in gold_spans
                true_positives[span.entity_type] = get(true_positives, span.entity_type, 0) + 1
            else
                false_positives[span.entity_type] = get(false_positives, span.entity_type, 0) + 1
            end
        end

        # Count false negatives (missed gold entities)
        for span in gold_spans
            if !(span in pred_spans)
                false_negatives[span.entity_type] = get(false_negatives, span.entity_type, 0) + 1
            end
        end
    end

    # Compute per-entity metrics
    per_entity_f1 = Dict{String, Float64}()
    per_entity_precision = Dict{String, Float64}()
    per_entity_recall = Dict{String, Float64}()
    per_entity_support = Dict{String, Int}()

    for entity in entity_types
        tp = get(true_positives, entity, 0)
        fp = get(false_positives, entity, 0)
        fn = get(false_negatives, entity, 0)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = compute_f1(precision, recall)

        per_entity_precision[entity] = precision
        per_entity_recall[entity] = recall
        per_entity_f1[entity] = f1
        per_entity_support[entity] = tp + fn  # Gold count
    end

    # Compute micro-averaged metrics
    total_tp = sum(values(true_positives))
    total_fp = sum(values(false_positives))
    total_fn = sum(values(false_negatives))

    precision_micro = total_tp / max(total_tp + total_fp, 1)
    recall_micro = total_tp / max(total_tp + total_fn, 1)
    f1_micro = compute_f1(precision_micro, recall_micro)

    # Compute macro-averaged metrics
    valid_entities = [e for e in entity_types if per_entity_support[e] > 0]
    if isempty(valid_entities)
        f1_macro = 0.0
        precision_macro = 0.0
        recall_macro = 0.0
    else
        f1_macro = mean([per_entity_f1[e] for e in valid_entities])
        precision_macro = mean([per_entity_precision[e] for e in valid_entities])
        recall_macro = mean([per_entity_recall[e] for e in valid_entities])
    end

    # Token accuracy
    token_accuracy = total_tokens > 0 ? correct_tokens / total_tokens : 0.0

    # Confusion matrix (optional)
    confusion = nothing
    if compute_confusion
        num_labels = length(id_to_label)
        confusion = zeros(Int, num_labels, num_labels)
        for (pred_seq, gold_seq) in zip(predictions, gold_labels)
            for (p, g) in zip(pred_seq, gold_seq)
                confusion[g, p] += 1
            end
        end
    end

    return NERResults(
        f1_micro, precision_micro, recall_micro,
        f1_macro, precision_macro, recall_macro,
        per_entity_f1, per_entity_precision, per_entity_recall, per_entity_support,
        token_accuracy,
        confusion,
        total_tp + total_fp,
        total_tp + total_fn,
        total_tp
    )
end

# =============================================================================
# Partial Matching Metrics (for Analysis)
# =============================================================================

"""
    PartialMatchResults

Results for partial matching evaluation (useful for error analysis).
"""
struct PartialMatchResults
    exact_match_f1::Float64      # Both boundary and type correct
    type_match_f1::Float64       # Type correct, boundary overlap
    boundary_match_f1::Float64   # Boundary correct, any type
end

"""
    spans_overlap(a::EvalSpan, b::EvalSpan) -> Bool

Check if two spans have any overlap.
"""
function spans_overlap(a::EvalSpan, b::EvalSpan)
    return !(a.end_idx < b.start_idx || b.end_idx < a.start_idx)
end

"""
    evaluate_partial(predictions, gold_labels, id_to_label) -> PartialMatchResults

Evaluate with partial matching for error analysis.
"""
function evaluate_partial(
    predictions::Vector{Vector{Int}},
    gold_labels::Vector{Vector{Int}},
    id_to_label::Dict{Int, String}
)
    exact_tp, exact_fp, exact_fn = 0, 0, 0
    type_tp, type_fp, type_fn = 0, 0, 0
    boundary_tp, boundary_fp, boundary_fn = 0, 0, 0

    for (pred_seq, gold_seq) in zip(predictions, gold_labels)
        pred_spans = collect(extract_eval_spans(pred_seq, id_to_label))
        gold_spans = collect(extract_eval_spans(gold_seq, id_to_label))

        pred_matched = falses(length(pred_spans))
        gold_matched = falses(length(gold_spans))

        # Find matches
        for (pi, pred) in enumerate(pred_spans)
            for (gi, gold) in enumerate(gold_spans)
                if pred == gold
                    # Exact match
                    exact_tp += 1
                    type_tp += 1
                    boundary_tp += 1
                    pred_matched[pi] = true
                    gold_matched[gi] = true
                    break
                elseif spans_overlap(pred, gold)
                    if pred.entity_type == gold.entity_type
                        # Type match with boundary error
                        type_tp += 1
                    end
                    # Boundary overlap
                    boundary_tp += 1
                    pred_matched[pi] = true
                    gold_matched[gi] = true
                    break
                end
            end
        end

        # Count unmatched
        exact_fp += count(.!pred_matched)
        exact_fn += count(.!gold_matched)
    end

    # Compute F1 scores
    exact_prec = exact_tp / max(exact_tp + exact_fp, 1)
    exact_rec = exact_tp / max(exact_tp + exact_fn, 1)
    exact_f1 = compute_f1(exact_prec, exact_rec)

    # Note: type and boundary F1 are approximations for analysis
    type_f1 = type_tp / max(exact_tp + exact_fn, 1)  # Simplified
    boundary_f1 = boundary_tp / max(exact_tp + exact_fn, 1)

    return PartialMatchResults(exact_f1, type_f1, boundary_f1)
end

# =============================================================================
# Error Analysis
# =============================================================================

"""
    ErrorAnalysis

Detailed error breakdown for analysis.
"""
struct ErrorAnalysis
    boundary_errors::Vector{Tuple{String, Int, Int, Int, Int}}  # (type, pred_start, pred_end, gold_start, gold_end)
    type_errors::Vector{Tuple{String, String, Int, Int}}        # (pred_type, gold_type, start, end)
    false_positives::Vector{Tuple{String, Int, Int}}            # (type, start, end)
    false_negatives::Vector{Tuple{String, Int, Int}}            # (type, start, end)
end

"""
    analyze_errors(predictions, gold_labels, id_to_label) -> ErrorAnalysis

Perform detailed error analysis.
"""
function analyze_errors(
    predictions::Vector{Vector{Int}},
    gold_labels::Vector{Vector{Int}},
    id_to_label::Dict{Int, String}
)
    boundary_errors = Tuple{String, Int, Int, Int, Int}[]
    type_errors = Tuple{String, String, Int, Int}[]
    false_positives = Tuple{String, Int, Int}[]
    false_negatives = Tuple{String, Int, Int}[]

    for (seq_idx, (pred_seq, gold_seq)) in enumerate(zip(predictions, gold_labels))
        pred_spans = collect(extract_eval_spans(pred_seq, id_to_label))
        gold_spans = collect(extract_eval_spans(gold_seq, id_to_label))

        pred_matched = falses(length(pred_spans))
        gold_matched = falses(length(gold_spans))

        for (pi, pred) in enumerate(pred_spans)
            for (gi, gold) in enumerate(gold_spans)
                if spans_overlap(pred, gold)
                    if pred == gold
                        # Correct
                        pred_matched[pi] = true
                        gold_matched[gi] = true
                    elseif pred.entity_type == gold.entity_type
                        # Boundary error
                        push!(boundary_errors, (pred.entity_type,
                            pred.start_idx, pred.end_idx,
                            gold.start_idx, gold.end_idx))
                        pred_matched[pi] = true
                        gold_matched[gi] = true
                    else
                        # Type error
                        push!(type_errors, (pred.entity_type, gold.entity_type,
                            pred.start_idx, pred.end_idx))
                        pred_matched[pi] = true
                        gold_matched[gi] = true
                    end
                    break
                end
            end
        end

        # False positives
        for (pi, pred) in enumerate(pred_spans)
            if !pred_matched[pi]
                push!(false_positives, (pred.entity_type, pred.start_idx, pred.end_idx))
            end
        end

        # False negatives
        for (gi, gold) in enumerate(gold_spans)
            if !gold_matched[gi]
                push!(false_negatives, (gold.entity_type, gold.start_idx, gold.end_idx))
            end
        end
    end

    return ErrorAnalysis(boundary_errors, type_errors, false_positives, false_negatives)
end

"""
    print_error_summary(analysis::ErrorAnalysis)

Print a summary of errors.
"""
function print_error_summary(analysis::ErrorAnalysis)
    println("Error Analysis Summary:")
    println("=" ^ 50)
    println("Boundary errors: $(length(analysis.boundary_errors))")
    println("Type errors:     $(length(analysis.type_errors))")
    println("False positives: $(length(analysis.false_positives))")
    println("False negatives: $(length(analysis.false_negatives))")

    if !isempty(analysis.type_errors)
        println("\nType confusion:")
        type_confusion = Dict{Tuple{String, String}, Int}()
        for (pred, gold, _, _) in analysis.type_errors
            key = (pred, gold)
            type_confusion[key] = get(type_confusion, key, 0) + 1
        end
        for ((pred, gold), count) in sort(collect(type_confusion), by=x->-x[2])
            println("  $pred → $gold: $count")
        end
    end

    if !isempty(analysis.false_positives)
        println("\nFalse positive types:")
        fp_counts = Dict{String, Int}()
        for (t, _, _) in analysis.false_positives
            fp_counts[t] = get(fp_counts, t, 0) + 1
        end
        for (t, c) in sort(collect(fp_counts), by=x->-x[2])
            println("  $t: $c")
        end
    end

    if !isempty(analysis.false_negatives)
        println("\nFalse negative types:")
        fn_counts = Dict{String, Int}()
        for (t, _, _) in analysis.false_negatives
            fn_counts[t] = get(fn_counts, t, 0) + 1
        end
        for (t, c) in sort(collect(fn_counts), by=x->-x[2])
            println("  $t: $c")
        end
    end
end

# =============================================================================
# Seqeval-compatible Output
# =============================================================================

"""
    classification_report(results::NERResults) -> String

Generate a sklearn-style classification report string.
"""
function classification_report(results::NERResults)
    io = IOBuffer()

    println(io, @sprintf("%-15s %10s %10s %10s %10s", "", "precision", "recall", "f1-score", "support"))
    println(io, "")

    # Per-entity rows
    for entity in sort(collect(keys(results.per_entity_f1)))
        p = results.per_entity_precision[entity]
        r = results.per_entity_recall[entity]
        f = results.per_entity_f1[entity]
        s = results.per_entity_support[entity]
        println(io, @sprintf("%-15s %10.4f %10.4f %10.4f %10d", entity, p, r, f, s))
    end

    println(io, "")
    println(io, @sprintf("%-15s %10.4f %10.4f %10.4f %10d", "micro avg",
        results.precision_micro, results.recall_micro, results.f1_micro,
        results.total_gold))
    println(io, @sprintf("%-15s %10.4f %10.4f %10.4f %10d", "macro avg",
        results.precision_macro, results.recall_macro, results.f1_macro,
        results.total_gold))

    return String(take!(io))
end

# =============================================================================
# Exports
# =============================================================================

export NERResults, EvalSpan
export evaluate_ner, extract_eval_spans
export compute_f1, classification_report
export PartialMatchResults, evaluate_partial
export ErrorAnalysis, analyze_errors, print_error_summary

end # module
