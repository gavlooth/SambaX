using Lux
using Random
using Test

include("../src/LogicGated.jl")
using .LogicGated

@testset "TokenRouter Shapes and Spans" begin
    d = 16
    seq_len = 8
    batch = 2
    num_experts = 4

    router = TokenRouter(d, num_experts; hidden_dim = 8, temperature = 1.0f0)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, router)

    x = randn(Float32, d, seq_len, batch)
    gates, _ = router(x, ps, st)

    @test size(gates) == (num_experts, seq_len, batch)

    # Gates should sum to ~1 along expert dimension
    for b in 1:batch, t in 1:seq_len
        s = sum(gates[:, t, b])
        @test isapprox(s, 1.0f0; atol = 1e-4)
    end

    top_idx = top1_expert(gates)
    @test size(top_idx) == (seq_len, batch)

    spans = build_spans(top_idx, num_experts)
    @test length(spans) == num_experts
end

@testset "Routing Utilities" begin
    logits = randn(Float32, 4, 6)
    hard, _ = topk_gates(logits; k = 2, temperature = 1.0f0, ste = false)
    @test size(hard) == size(logits)

    soft = NNlib.softmax(logits, dims = 1)
    logic_mask = trues(6)
    floored = apply_logic_floor(soft, logic_mask; floor = 0.1f0)
    @test all(floored[EXPERT_LOGIC, :] .>= 0.1f0)

    ent = gate_entropy(soft)
    @test ent > 0

    util = expert_utilization(soft)
    @test length(util) == 4

    lb = load_balance_loss(soft)
    @test lb > 0
end

@testset "Fusion and Cache Utilities" begin
    E, d, seq, batch = 4, 8, 5, 2
    expert_outputs = randn(Float32, E, d, seq, batch)
    gates = NNlib.softmax(randn(Float32, E, seq, batch), dims = 1)
    fused = fuse_experts_gated_sum(expert_outputs, gates)
    @test size(fused) == (d, seq, batch)

    cache = init_cache(E)
    cache = update_cache(cache, expert_outputs, 3)
    updated = apply_cache(cache, copy(expert_outputs))
    @test size(updated) == size(expert_outputs)
end

@testset "Metrics and Schedules" begin
    gates = NNlib.softmax(randn(Float32, 4, 6), dims = 1)
    labels = rand(1:4, 6)
    metrics = router_metrics(gates, labels)
    @test haskey(metrics, :accuracy)
    @test haskey(metrics, :collapse)

    active = active_experts_schedule(15000)
    @test active[EXPERT_LOGIC]
    @test active[EXPERT_LANGUAGE]

    masked = apply_expert_mask(gates, active)
    @test size(masked) == size(gates)

    preds = top1_expert(gates)
    cm = expert_confusion_matrix(preds, labels)
    @test size(cm) == (4, 4)

    loss = router_loss(gates, labels)
    @test loss > 0

    tokens = ["forall", "x", "->", "y"]
    logic_mask = logic_mask_from_tokens(tokens)
    @test any(logic_mask)

    forced = force_experts(gates; experts = [EXPERT_LOGIC], mask = logic_mask, floor = 0.2f0)
    @test all(forced[EXPERT_LOGIC, logic_mask] .>= 0.2f0)

    rerouted = reroute_on_failure(gates, logic_mask)
    @test size(rerouted) == size(gates)
end

@testset "GatedExperts Wrapper" begin
    d = 8
    seq = 6
    batch = 2
    router = TokenRouter(d, 2; hidden_dim = 4)
    experts = [Lux.Dense(d => d), Lux.Dense(d => d)]
    layer = GatedExperts(router, experts; top_k = 1, use_ste = false)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, layer)

    x = randn(Float32, d, seq, batch)
    y, _ = layer(x, ps, st)
    @test size(y) == size(x)

    y2, gates, _ = layer(x, ps, st; return_gates = true)
    @test size(y2) == size(x)
    @test size(gates) == (2, seq, batch)
end
