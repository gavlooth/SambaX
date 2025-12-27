module LogicGated

using Lux
using LuxCore
using NNlib
using Random
using Zygote: @ignore
using Statistics: mean

export TokenRouter, GatedExperts, top1_expert, build_spans
export heuristic_labels, heuristic_labels_batch
export router_supervision_loss, router_accuracy
export anneal_temperature, topk_gates, ste_gates, apply_logic_floor
export gate_entropy, expert_utilization, load_balance_loss, expert_dropout
export disagreement_score, conflict_mask
export fuse_experts_gated_sum, gather_spans, scatter_spans
export ExpertCache, init_cache, update_cache, apply_cache
export active_experts_schedule, apply_expert_mask
export expert_confusion_matrix, collapse_alert, router_metrics, misroute_rate
export router_loss, logic_mask_from_tokens, logic_mask_from_labels
export force_experts, reroute_on_failure
export EXPERT_LOGIC, EXPERT_LANGUAGE, EXPERT_MATH, EXPERT_MEMORY, EXPERT_NAMES

const EXPERT_LOGIC = 1
const EXPERT_LANGUAGE = 2
const EXPERT_MATH = 3
const EXPERT_MEMORY = 4
const EXPERT_NAMES = (:logic, :language, :math, :memory)
const DEFAULT_LOGIC_FLOOR = 0.05f0

"""
    TokenRouter(d, num_experts; hidden_dim=128, temperature=1.0)

Lightweight per-token router. Produces gate probabilities for each token.
"""
struct TokenRouter <: LuxCore.AbstractLuxLayer
    embedding_dimension::Int
    num_experts::Int
    hidden_dim::Int
    temperature::Float32
    RouterMLP::Lux.Chain
end

function TokenRouter(
    embedding_dimension::Int,
    num_experts::Int;
    hidden_dim::Int = 128,
    temperature::Float32 = 1.0f0
)
    mlp = Lux.Chain(
        Lux.Dense(embedding_dimension => hidden_dim, NNlib.gelu),
        Lux.Dense(hidden_dim => num_experts)
    )
    return TokenRouter(embedding_dimension, num_experts, hidden_dim, temperature, mlp)
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::TokenRouter)
    return (RouterMLP = Lux.initialparameters(rng, layer.RouterMLP),)
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::TokenRouter)
    return (RouterMLP = Lux.initialstates(rng, layer.RouterMLP),)
end

"""
    (router::TokenRouter)(x, params, state; temperature=router.temperature)

Inputs:
- `x`: (d, seq, batch) or (d, seq)

Returns:
- `gates`: (num_experts, seq, batch)
- `new_state`
"""
function (router::TokenRouter)(x, params, state; temperature::Float32 = router.temperature)
    if ndims(x) == 2
        x = reshape(x, size(x, 1), size(x, 2), 1)
        was_unbatched = true
    else
        was_unbatched = false
    end

    d, seq_len, batch = size(x)
    x_flat = reshape(x, d, :)
    logits_flat, mlp_state = router.RouterMLP(x_flat, params.RouterMLP, state.RouterMLP)
    logits = reshape(logits_flat, router.num_experts, seq_len, batch)
    gates = NNlib.softmax(logits ./ temperature, dims = 1)

    new_state = (RouterMLP = mlp_state,)
    if was_unbatched
        gates = dropdims(gates, dims = 3)
    end
    return gates, new_state
end

"""
    GatedExperts(router, experts; top_k=2, use_ste=false, logic_floor=DEFAULT_LOGIC_FLOOR)

Minimal MoE wrapper that runs all experts and fuses outputs with gates.
"""
struct GatedExperts{R,E} <: LuxCore.AbstractLuxLayer
    Router::R
    Experts::Vector{E}
    num_experts::Int
    top_k::Int
    use_ste::Bool
    logic_floor::Float32
end

function GatedExperts(
    router::TokenRouter,
    experts::Vector;
    top_k::Int = 2,
    use_ste::Bool = false,
    logic_floor::Float32 = DEFAULT_LOGIC_FLOOR
)
    return GatedExperts(router, experts, length(experts), top_k, use_ste, logic_floor)
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::GatedExperts)
    return (
        Router = Lux.initialparameters(rng, layer.Router),
        Experts = [Lux.initialparameters(rng, ex) for ex in layer.Experts],
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::GatedExperts)
    return (
        Router = Lux.initialstates(rng, layer.Router),
        Experts = [Lux.initialstates(rng, ex) for ex in layer.Experts],
    )
end

"""
    (layer::GatedExperts)(inputs, params, state; logic_mask=nothing, temperature=1.0, return_gates=false)

Routes inputs through all experts and fuses outputs with gates.
"""
function (layer::GatedExperts)(
    inputs,
    params,
    state;
    logic_mask = nothing,
    temperature::Float32 = 1.0f0,
    return_gates::Bool = false
)
    x = inputs isa Tuple ? inputs[1] : inputs

    gates, router_state = layer.Router(x, params.Router, state.Router; temperature = temperature)
    if logic_mask !== nothing
        gates = apply_logic_floor(gates, logic_mask; floor = layer.logic_floor)
    end
    if layer.top_k > 0
        gates, _ = topk_gates(gates; k = layer.top_k, temperature = 1.0f0, ste = layer.use_ste)
    end

    # Run all experts (dense, safe baseline)
    outputs = Vector{Any}(undef, layer.num_experts)
    new_states = Vector{Any}(undef, layer.num_experts)
    for i in 1:layer.num_experts
        out, st = layer.Experts[i](inputs, params.Experts[i], state.Experts[i])
        outputs[i] = out
        new_states[i] = st
    end

    # Stack expert outputs for fusion
    first_out = outputs[1]
    expert_outputs = if ndims(first_out) == 2
        E = layer.num_experts
        d, seq = size(first_out)
        buf = zeros(Float32, E, d, seq)
        for e in 1:E
            buf[e, :, :] .= outputs[e]
        end
        buf
    else
        E = layer.num_experts
        d, seq, batch = size(first_out)
        buf = zeros(Float32, E, d, seq, batch)
        for e in 1:E
            buf[e, :, :, :] .= outputs[e]
        end
        buf
    end

    fused = fuse_experts_gated_sum(expert_outputs, gates)
    new_state = (Router = router_state, Experts = new_states)
    return return_gates ? (fused, gates, new_state) : (fused, new_state)
end

"""
    anneal_temperature(step; start=2.0, stop=0.5, total_steps=10000, schedule=:cosine)

Compute a temperature value for soft-to-hard routing.
"""
function anneal_temperature(
    step::Int;
    start::Float32 = 2.0f0,
    stop::Float32 = 0.5f0,
    total_steps::Int = 10000,
    schedule::Symbol = :cosine
)
    if total_steps <= 0
        return stop
    end
    t = clamp(step / total_steps, 0.0f0, 1.0f0)
    if schedule == :linear
        return start + (stop - start) * t
    elseif schedule == :cosine
        return stop + (start - stop) * 0.5f0 * (1.0f0 + cos(Float32(pi) * t))
    else
        return stop
    end
end

"""
    topk_gates(logits; k=2, temperature=1.0, ste=false)

Convert logits to (hard or soft) top-k gates.
Returns `(gates, top_idx)`.
"""
function topk_gates(
    logits::AbstractArray;
    k::Int = 2,
    temperature::Float32 = 1.0f0,
    ste::Bool = false
)
    gates = NNlib.softmax(logits ./ temperature, dims = 1)
    if k <= 0
        return gates, nothing
    end

    hard = zeros(Float32, size(gates))
    top_idx = topk_indices(gates, k)
    if ndims(gates) == 2
        for t in 1:size(gates, 2)
            for j in 1:k
                hard[top_idx[j, t], t] = 1.0f0
            end
        end
    else
        for b in 1:size(gates, 3)
            for t in 1:size(gates, 2)
                for j in 1:k
                    hard[top_idx[j, t, b], t, b] = 1.0f0
                end
            end
        end
    end

    if ste
        hard = ste_gates(hard, gates)
    end

    return hard, top_idx
end

"""
    ste_gates(hard_gates, soft_gates)

Straight-through estimator: forward uses hard gates, backward uses soft gates.
"""
function ste_gates(hard_gates::AbstractArray, soft_gates::AbstractArray)
    soft_detached = @ignore soft_gates
    return hard_gates .+ (soft_gates .- soft_detached)
end

"""
    apply_logic_floor(gates, logic_mask; floor=DEFAULT_LOGIC_FLOOR)

Ensure the logic expert has at least `floor` mass for tokens in `logic_mask`.
"""
function apply_logic_floor(
    gates::AbstractArray,
    logic_mask::AbstractArray;
    floor::Float32 = DEFAULT_LOGIC_FLOOR
)
    g = copy(gates)
    if ndims(g) == 2
        for t in 1:size(g, 2)
            if logic_mask[t]
                g[EXPERT_LOGIC, t] = max(g[EXPERT_LOGIC, t], floor)
            end
        end
    else
        for b in 1:size(g, 3), t in 1:size(g, 2)
            if logic_mask[t, b]
                g[EXPERT_LOGIC, t, b] = max(g[EXPERT_LOGIC, t, b], floor)
            end
        end
    end
    # Renormalize
    norm = sum(g, dims = 1)
    return g ./ norm
end

"""
    gate_entropy(gates)

Compute mean entropy across tokens.
"""
function gate_entropy(gates::AbstractArray)
    eps = 1f-10
    log_g = log.(gates .+ eps)
    ent = -sum(gates .* log_g, dims = 1)
    return mean(ent)
end

"""
    expert_utilization(gates; hard=false)

Return mean utilization per expert.
"""
function expert_utilization(gates::AbstractArray; hard::Bool = false)
    if hard
        top_idx = top1_expert(gates)
        counts = zeros(Float32, size(gates, 1))
        for idx in top_idx
            counts[idx] += 1
        end
        return counts ./ length(top_idx)
    else
        return vec(mean(gates, dims = (2, 3)))
    end
end

"""
    load_balance_loss(gates; hard=false)

Switch-style load balancing loss.
"""
function load_balance_loss(gates::AbstractArray; hard::Bool = false)
    num_experts = size(gates, 1)
    p = expert_utilization(gates; hard = false)
    f = expert_utilization(gates; hard = hard)
    return num_experts * sum(p .* f)
end

"""
    expert_dropout(gates; drop_prob=0.1, rng=Random.default_rng())

Randomly drop one expert per batch with probability `drop_prob`.
"""
function expert_dropout(
    gates::AbstractArray;
    drop_prob::Float32 = 0.1f0,
    rng = Random.default_rng()
)
    g = copy(gates)
    num_experts = size(g, 1)
    if ndims(g) == 2
        if rand(rng) < drop_prob
            drop_idx = rand(rng, 1:num_experts)
            g[drop_idx, :] .= 0
        end
    else
        for b in 1:size(g, 3)
            if rand(rng) < drop_prob
                drop_idx = rand(rng, 1:num_experts)
                g[drop_idx, :, b] .= 0
            end
        end
    end
    # Renormalize
    norm = sum(g, dims = 1)
    return g ./ (norm .+ 1f-10)
end

"""
    active_experts_schedule(step; logic_start=10000, math_start=20000, memory_start=30000)

Warmup schedule for expert activation. Returns a Bool vector (num_experts).
"""
function active_experts_schedule(
    step::Int;
    logic_start::Int = 10000,
    math_start::Int = 20000,
    memory_start::Int = 30000
)
    active = trues(4)
    # Always keep Language (index 2) active
    active .= false
    active[EXPERT_LANGUAGE] = true
    if step >= logic_start
        active[EXPERT_LOGIC] = true
    end
    if step >= math_start
        active[EXPERT_MATH] = true
    end
    if step >= memory_start
        active[EXPERT_MEMORY] = true
    end
    return active
end

"""
    apply_expert_mask(gates, active)

Zero out inactive experts and renormalize.
"""
function apply_expert_mask(gates::AbstractArray, active::AbstractVector{Bool})
    g = copy(gates)
    for e in 1:length(active)
        if !active[e]
            g[e, :, :] .= 0
        end
    end
    norm = sum(g, dims = 1)
    return g ./ (norm .+ 1f-10)
end

"""
    force_experts(gates; experts, mask=nothing, floor=0.1)

Force specified experts to have at least `floor` mass on masked tokens.
"""
function force_experts(
    gates::AbstractArray;
    experts::AbstractVector{<:Integer},
    mask::Union{Nothing, AbstractArray} = nothing,
    floor::Float32 = 0.1f0
)
    g = copy(gates)
    if ndims(g) == 2
        for t in 1:size(g, 2)
            if mask === nothing || mask[t]
                for e in experts
                    g[e, t] = max(g[e, t], floor)
                end
            end
        end
    else
        for b in 1:size(g, 3), t in 1:size(g, 2)
            if mask === nothing || mask[t, b]
                for e in experts
                    g[e, t, b] = max(g[e, t, b], floor)
                end
            end
        end
    end
    norm = sum(g, dims = 1)
    return g ./ (norm .+ 1f-10)
end

"""
    reroute_on_failure(gates, failure_mask; forced=(EXPERT_LOGIC, EXPERT_LANGUAGE))

Simple reroute fallback: boost logic + language for tokens flagged by verifier.
"""
function reroute_on_failure(
    gates::AbstractArray,
    failure_mask::AbstractArray;
    forced::Tuple{Int,Int} = (EXPERT_LOGIC, EXPERT_LANGUAGE),
    floor::Float32 = 0.2f0
)
    return force_experts(gates; experts = collect(forced), mask = failure_mask, floor = floor)
end

"""
    disagreement_score(expert_outputs)

Compute per-token disagreement based on variance across experts.
"""
function disagreement_score(expert_outputs::AbstractArray)
    # expert_outputs: (E, d, seq, batch) or (E, d, seq)
    if ndims(expert_outputs) == 3
        E, d, seq = size(expert_outputs)
        mean_e = mean(expert_outputs, dims = 1)
        var_e = mean((expert_outputs .- mean_e).^2, dims = 1)
        score = mean(var_e, dims = 1)
        return dropdims(score, dims = (1, 2))
    else
        E, d, seq, batch = size(expert_outputs)
        mean_e = mean(expert_outputs, dims = 1)
        var_e = mean((expert_outputs .- mean_e).^2, dims = 1)
        score = mean(var_e, dims = 1)
        return dropdims(score, dims = (1, 2))
    end
end

"""
    conflict_mask(score; threshold=0.1)

Return a boolean mask for tokens above a disagreement threshold.
"""
conflict_mask(score::AbstractArray; threshold::Float32 = 0.1f0) = score .> threshold

"""
    topk_indices(gates, k)

Return top-k indices per token.
"""
function topk_indices(gates::AbstractArray, k::Int)
    if ndims(gates) == 2
        _, seq_len = size(gates)
        idx = Array{Int}(undef, k, seq_len)
        for t in 1:seq_len
            idx[:, t] = partialsortperm(view(gates, :, t), 1:k; rev = true)
        end
        return idx
    else
        _, seq_len, batch = size(gates)
        idx = Array{Int}(undef, k, seq_len, batch)
        for b in 1:batch
            for t in 1:seq_len
                idx[:, t, b] = partialsortperm(view(gates, :, t, b), 1:k; rev = true)
            end
        end
        return idx
    end
end

"""
    expert_confusion_matrix(preds, labels; num_experts=4, ignore_index=0)

Compute confusion matrix of predicted vs. target expert.
"""
function expert_confusion_matrix(
    preds::AbstractArray,
    labels::AbstractArray;
    num_experts::Int = 4,
    ignore_index::Int = 0
)
    cm = zeros(Int, num_experts, num_experts)
    for i in eachindex(labels)
        lbl = labels[i]
        if lbl != ignore_index
            cm[lbl, preds[i]] += 1
        end
    end
    return cm
end

"""
    collapse_alert(utilization; threshold=0.6)

Return true if any expert exceeds usage threshold.
"""
collapse_alert(utilization::AbstractVector; threshold::Float32 = 0.6f0) =
    any(utilization .>= threshold)

"""
    router_metrics(gates, labels; hard=false, ignore_index=0)

Convenience bundle of routing metrics.
"""
function router_metrics(
    gates::AbstractArray,
    labels::AbstractArray;
    hard::Bool = false,
    ignore_index::Int = 0
)
    acc = router_accuracy(gates, labels; ignore_index = ignore_index)
    ent = gate_entropy(gates)
    util = expert_utilization(gates; hard = hard)
    bal = load_balance_loss(gates; hard = hard)
    return (
        accuracy = acc,
        misroute = 1.0f0 - acc,
        entropy = ent,
        utilization = util,
        balance_loss = bal,
        collapse = collapse_alert(util)
    )
end

"""
    router_loss(gates, labels; λ_balance=0.01, λ_entropy=0.01, hard=false)

Combined router loss: supervision + load balance + entropy regularization.
"""
function router_loss(
    gates::AbstractArray,
    labels::AbstractArray;
    λ_balance::Float32 = 0.01f0,
    λ_entropy::Float32 = 0.01f0,
    hard::Bool = false
)
    sup = router_supervision_loss(gates, labels)
    bal = load_balance_loss(gates; hard = hard)
    ent = gate_entropy(gates)
    return sup + λ_balance * bal - λ_entropy * ent
end

"""
    misroute_rate(gates, labels; ignore_index=0)

Return 1 - routing accuracy.
"""
function misroute_rate(
    gates::AbstractArray,
    labels::AbstractArray;
    ignore_index::Int = 0
)
    return 1.0f0 - router_accuracy(gates, labels; ignore_index = ignore_index)
end

# -----------------------------------------------------------------------------
# Fusion and Span Utilities
# -----------------------------------------------------------------------------

"""
    fuse_experts_gated_sum(expert_outputs, gates)

Weighted sum across experts. `expert_outputs` is (E, d, seq, batch),
`gates` is (E, seq, batch).
"""
function fuse_experts_gated_sum(
    expert_outputs::AbstractArray,
    gates::AbstractArray
)
    if ndims(expert_outputs) == 3
        # (E, d, seq)
        E, d, seq = size(expert_outputs)
        fused = zeros(Float32, d, seq)
        for e in 1:E
            fused .+= expert_outputs[e, :, :] .* reshape(gates[e, :], 1, seq)
        end
        return fused
    else
        E, d, seq, batch = size(expert_outputs)
        fused = zeros(Float32, d, seq, batch)
        for e in 1:E
            fused .+= expert_outputs[e, :, :, :] .* reshape(gates[e, :, :], 1, seq, batch)
        end
        return fused
    end
end

"""
    gather_spans(x, spans)

Collect per-expert span slices from `x` (d, seq, batch).
Returns `Vector{Vector{AbstractArray}}` where each inner vector holds slices.
"""
function gather_spans(
    x::AbstractArray,
    spans::Vector{Vector{Tuple{Int,Int,Int}}}
)
    gathered = [Vector{AbstractArray}() for _ in spans]
    for (expert, list) in enumerate(spans)
        for (b, s, e) in list
            push!(gathered[expert], view(x, :, s:e, b))
        end
    end
    return gathered
end

"""
    scatter_spans!(y, spans, expert_outputs)

Scatter per-expert outputs back into `y` (d, seq, batch).
"""
function scatter_spans!(
    y::AbstractArray,
    spans::Vector{Vector{Tuple{Int,Int,Int}}},
    expert_outputs::Vector{Vector{AbstractArray}}
)
    for (expert, list) in enumerate(spans)
        for (i, (b, s, e)) in enumerate(list)
            y[:, s:e, b] .= expert_outputs[expert][i]
        end
    end
    return y
end

# -----------------------------------------------------------------------------
# Expert Caching (prefix reuse)
# -----------------------------------------------------------------------------

struct ExpertCache{T}
    prefix_len::Int
    outputs::Vector{T}  # per-expert cached outputs for prefix
end

function init_cache(num_experts::Int)
    return ExpertCache(0, [nothing for _ in 1:num_experts])
end

"""
    update_cache(cache, expert_outputs, prefix_len)

Cache expert outputs for the prefix.
"""
function update_cache(cache::ExpertCache, expert_outputs::AbstractArray, prefix_len::Int)
    num_experts = size(expert_outputs, 1)
    outputs = Vector{Any}(undef, num_experts)
    for e in 1:num_experts
        outputs[e] = copy(view(expert_outputs, e, :, 1:prefix_len, :))
    end
    return ExpertCache(prefix_len, outputs)
end

"""
    apply_cache(cache, expert_outputs)

Overwrite prefix region with cached outputs.
"""
function apply_cache(cache::ExpertCache, expert_outputs::AbstractArray)
    if cache.prefix_len == 0
        return expert_outputs
    end
    num_experts = size(expert_outputs, 1)
    for e in 1:num_experts
        if cache.outputs[e] !== nothing
            expert_outputs[e, :, 1:cache.prefix_len, :] .= cache.outputs[e]
        end
    end
    return expert_outputs
end

"""
    top1_expert(gates)

Return top-1 expert indices for each token.
"""
function top1_expert(gates::AbstractArray)
    if ndims(gates) == 2
        # (E, seq)
        _, seq_len = size(gates)
        top_idx = similar(gates, Int, seq_len)
        for t in 1:seq_len
            top_idx[t] = argmax(view(gates, :, t))
        end
        return reshape(top_idx, :, 1)
    end

    _, seq_len, batch = size(gates)
    top_idx = Matrix{Int}(undef, seq_len, batch)
    for b in 1:batch
        for t in 1:seq_len
            top_idx[t, b] = argmax(view(gates, :, t, b))
        end
    end
    return top_idx
end

"""
    build_spans(top_idx, num_experts; mask=nothing)

Build contiguous spans per expert from token-level top-1 routing.

Returns `Vector{Vector{Tuple{Int,Int,Int}}}` where each tuple is
`(batch_idx, start, end)` (1-based, inclusive).
"""
function build_spans(
    top_idx::AbstractMatrix{<:Integer},
    num_experts::Int;
    mask::Union{Nothing, AbstractMatrix{Bool}} = nothing
)
    seq_len, batch = size(top_idx)
    spans = [Vector{Tuple{Int,Int,Int}}() for _ in 1:num_experts]

    for b in 1:batch
        t = 1
        while t <= seq_len
            if mask !== nothing && !mask[t, b]
                t += 1
                continue
            end
            expert = top_idx[t, b]
            start = t
            t += 1
            while t <= seq_len &&
                  (mask === nothing || mask[t, b]) &&
                  top_idx[t, b] == expert
                t += 1
            end
            push!(spans[expert], (b, start, t - 1))
        end
    end

    return spans
end

# -----------------------------------------------------------------------------
# Heuristic Router Labels
# -----------------------------------------------------------------------------

const LOGIC_REGEX = r"(∀|∃|→|->|=>|⊢|⊨|¬|∧|∨|\biff\b|\bimplies\b|\btherefore\b|\bproof\b)"
const MATH_REGEX = r"([0-9]|[+\-*/^=]|\bsolve\b|\bcalculate\b|\bequation\b)"
const MEMORY_REGEX = r"(\bwho\b|\bwhat\b|\bwhen\b|\bwhere\b|\bwhich\b|\bdefine\b|\bcapital\b|\bdate\b)"

"""
    heuristic_labels(tokens)

Assign heuristic expert labels to tokens.
"""
function heuristic_labels(tokens::AbstractVector{<:AbstractString})
    labels = fill(EXPERT_LANGUAGE, length(tokens))
    for i in eachindex(tokens)
        tok = lowercase(tokens[i])
        if occursin(LOGIC_REGEX, tok)
            labels[i] = EXPERT_LOGIC
        elseif occursin(MATH_REGEX, tok)
            labels[i] = EXPERT_MATH
        elseif occursin(MEMORY_REGEX, tok)
            labels[i] = EXPERT_MEMORY
        end
    end
    return labels
end

logic_mask_from_labels(labels::AbstractArray) = labels .== EXPERT_LOGIC

function logic_mask_from_tokens(tokens::AbstractVector{<:AbstractString})
    labels = heuristic_labels(tokens)
    return logic_mask_from_labels(labels)
end

"""
    heuristic_labels_batch(batch_tokens; pad_label=0)

Return (seq_len, batch) label matrix for a batch of token lists.
"""
function heuristic_labels_batch(
    batch_tokens::Vector{<:AbstractVector{<:AbstractString}};
    pad_label::Int = 0
)
    batch = length(batch_tokens)
    max_len = maximum(length.(batch_tokens))
    labels = fill(pad_label, max_len, batch)
    for b in 1:batch
        tok = batch_tokens[b]
        lbl = heuristic_labels(tok)
        labels[1:length(lbl), b] .= lbl
    end
    return labels
end

"""
    router_supervision_loss(gates, labels; ignore_index=0)

Cross-entropy loss between router gates and token labels.
"""
function router_supervision_loss(
    gates::AbstractArray,
    labels::AbstractArray;
    ignore_index::Int = 0
)
    log_gates = log.(gates .+ 1f-10)
    total = 0.0f0
    count = 0

    if ndims(gates) == 2
        # (E, seq), labels: (seq,) or (seq,1)
        seq_len = size(gates, 2)
        for t in 1:seq_len
            label = labels[t]
            if label != ignore_index
                total -= log_gates[label, t]
                count += 1
            end
        end
    else
        _, seq_len, batch = size(gates)
        for b in 1:batch
            for t in 1:seq_len
                label = labels[t, b]
                if label != ignore_index
                    total -= log_gates[label, t, b]
                    count += 1
                end
            end
        end
    end

    return count > 0 ? total / count : 0.0f0
end

"""
    router_accuracy(gates, labels; ignore_index=0)

Compute top-1 routing accuracy.
"""
function router_accuracy(
    gates::AbstractArray,
    labels::AbstractArray;
    ignore_index::Int = 0
)
    preds = top1_expert(gates)
    correct = 0
    total = 0

    if ndims(labels) == 1
        for t in eachindex(labels)
            label = labels[t]
            if label != ignore_index
                total += 1
                correct += (preds[t] == label)
            end
        end
    else
        for b in 1:size(labels, 2)
            for t in 1:size(labels, 1)
                label = labels[t, b]
                if label != ignore_index
                    total += 1
                    correct += (preds[t, b] == label)
                end
            end
        end
    end

    return total > 0 ? correct / total : 0.0f0
end

end # module LogicGated
