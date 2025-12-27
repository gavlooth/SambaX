using Lux
using Random
using Test

include("../src/Ossamma.jl")
using .Ossamma

@testset "MoETModel Shapes" begin
    config = MoETConfig(
        vocab_size = 64,
        max_sequence_length = 16,
        embedding_dimension = 32,
        number_of_heads = 4,
        number_of_experts = 4,
        layers_per_expert = 2,
        time_dimension = 16,
        router_hidden_dim = 32,
        top_k = 0,
        dropout_rate = 0.0f0,
    )

    model = MoETModel(config)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    seq_len = 12
    batch = 3
    token_ids = rand(1:config.vocab_size, seq_len, batch)

    logits, gates, new_state = model((token_ids = token_ids,), ps, st; return_gates = true)

    @test size(logits) == (config.vocab_size, seq_len, batch)
    @test size(gates) == (config.number_of_experts, seq_len, batch)
    @test !isnothing(new_state)
end
