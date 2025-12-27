module MoET

using Lux
using LuxCore
using Random

import ..Ossamma: OssammaBlock
import ..LogicGated: TokenRouter, GatedExperts

# Helper to detect and transfer to GPU arrays without requiring CUDA at compile time
function to_device_like(target, x::AbstractArray)
    target_type = string(typeof(target))
    if occursin("CuArray", target_type)
        cuda_mod = parentmodule(typeof(target))
        while cuda_mod !== Main && !isdefined(cuda_mod, :CuArray)
            cuda_mod = parentmodule(cuda_mod)
        end
        if isdefined(cuda_mod, :CuArray)
            return cuda_mod.CuArray(x)
        end
    end
    return x
end

# =============================================================================
# Configuration
# =============================================================================

Base.@kwdef struct MoETConfig
    # Vocabulary and shapes
    vocab_size::Int = 32000
    max_sequence_length::Int = 512
    embedding_dimension::Int = 256
    number_of_heads::Int = 4
    number_of_experts::Int = 4
    layers_per_expert::Int = 2

    # Internal dimensions
    time_dimension::Int = 64
    state_dimension::Int = -1  # -1 means use embedding_dimension

    # Router
    router_hidden_dim::Int = 128
    top_k::Int = 2
    use_ste::Bool = false
    logic_floor::Float32 = 0.05f0

    # Attention
    window_size::Int = 256

    # Oscillator SSM
    min_frequency::Float32 = 0.1f0
    max_frequency::Float32 = 10.0f0
    default_time_step::Float32 = 0.1f0

    # Training
    dropout_rate::Float32 = 0.1f0

    # FFN configuration
    use_ffn::Bool = true
    ffn_expansion::Float32 = 3f0 / 2f0

    # GPU Parallelization
    use_parallel_scan::Bool = false
    parallel_chunk_size::Int = 64
end

# =============================================================================
# Expert Tower
# =============================================================================

struct ExpertTower{B,N} <: LuxCore.AbstractLuxLayer
    embedding_dimension::Int
    number_of_layers::Int
    Blocks::B
    FinalNorm::N
end

function ExpertTower(
    embedding_dimension::Int,
    max_sequence_length::Int,
    number_of_heads::Int,
    time_dimension::Int;
    number_of_layers::Int = 2,
    state_dimension::Int = embedding_dimension,
    window_size::Int = 256,
    min_frequency::Float32 = 0.1f0,
    max_frequency::Float32 = 10.0f0,
    default_time_step::Float32 = 0.1f0,
    dropout_rate::Float32 = 0.1f0,
    use_ffn::Bool = true,
    ffn_expansion::Float32 = 3f0 / 2f0,
    use_parallel_scan::Bool = false,
    parallel_chunk_size::Int = 64,
)
    blocks = Tuple([
        OssammaBlock(
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
        )
        for _ in 1:number_of_layers
    ])

    return ExpertTower(
        embedding_dimension,
        number_of_layers,
        blocks,
        Lux.LayerNorm((embedding_dimension,)),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, tower::ExpertTower)
    block_params = NamedTuple{ntuple(i -> Symbol("Block_$i"), tower.number_of_layers)}(
        Tuple(Lux.initialparameters(rng, block) for block in tower.Blocks)
    )

    return (
        Blocks = block_params,
        FinalNorm = Lux.initialparameters(rng, tower.FinalNorm),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, tower::ExpertTower)
    block_states = NamedTuple{ntuple(i -> Symbol("Block_$i"), tower.number_of_layers)}(
        Tuple(Lux.initialstates(rng, block) for block in tower.Blocks)
    )

    return (
        Blocks = block_states,
        FinalNorm = Lux.initialstates(rng, tower.FinalNorm),
    )
end

function (tower::ExpertTower)(inputs::Tuple, params, state)
    hidden, time_input = inputs

    new_states = Vector{Any}(undef, tower.number_of_layers)
    for i in 1:tower.number_of_layers
        block = tower.Blocks[i]
        block_key = Symbol("Block_$i")
        block_params = getproperty(params.Blocks, block_key)
        block_state = getproperty(state.Blocks, block_key)
        hidden, new_block_state = block((hidden, time_input), block_params, block_state)
        new_states[i] = new_block_state
    end

    hidden, norm_state = tower.FinalNorm(hidden, params.FinalNorm, state.FinalNorm)

    new_state = (
        Blocks = NamedTuple{ntuple(i -> Symbol("Block_$i"), tower.number_of_layers)}(Tuple(new_states)),
        FinalNorm = norm_state,
    )

    return hidden, new_state
end

# =============================================================================
# MoET Model
# =============================================================================

struct MoETModel{E,P,M,N,O} <: LuxCore.AbstractLuxLayer
    vocab_size::Int
    max_sequence_length::Int
    embedding_dimension::Int
    number_of_heads::Int
    time_dimension::Int
    number_of_experts::Int

    TokenEmbedding::E
    PositionEmbedding::P
    MoE::M
    FinalNorm::N
    OutputHead::O
end

function MoETModel(config::MoETConfig)
    state_dimension = config.state_dimension == -1 ? config.embedding_dimension : config.state_dimension
    return MoETModel(
        vocab_size = config.vocab_size,
        max_sequence_length = config.max_sequence_length,
        embedding_dimension = config.embedding_dimension,
        number_of_heads = config.number_of_heads,
        number_of_experts = config.number_of_experts,
        layers_per_expert = config.layers_per_expert,
        time_dimension = config.time_dimension,
        state_dimension = state_dimension,
        router_hidden_dim = config.router_hidden_dim,
        top_k = config.top_k,
        use_ste = config.use_ste,
        logic_floor = config.logic_floor,
        window_size = config.window_size,
        min_frequency = config.min_frequency,
        max_frequency = config.max_frequency,
        default_time_step = config.default_time_step,
        dropout_rate = config.dropout_rate,
        use_ffn = config.use_ffn,
        ffn_expansion = config.ffn_expansion,
        use_parallel_scan = config.use_parallel_scan,
        parallel_chunk_size = config.parallel_chunk_size,
    )
end

function MoETModel(;
    vocab_size::Int,
    max_sequence_length::Int,
    embedding_dimension::Int,
    number_of_heads::Int,
    number_of_experts::Int,
    layers_per_expert::Int,
    time_dimension::Int = 64,
    state_dimension::Int = embedding_dimension,
    router_hidden_dim::Int = 128,
    top_k::Int = 2,
    use_ste::Bool = false,
    logic_floor::Float32 = 0.05f0,
    window_size::Int = 256,
    min_frequency::Float32 = 0.1f0,
    max_frequency::Float32 = 10.0f0,
    default_time_step::Float32 = 0.1f0,
    dropout_rate::Float32 = 0.1f0,
    use_ffn::Bool = true,
    ffn_expansion::Float32 = 3f0 / 2f0,
    use_parallel_scan::Bool = false,
    parallel_chunk_size::Int = 64,
)
    experts = [
        ExpertTower(
            embedding_dimension,
            max_sequence_length,
            number_of_heads,
            time_dimension;
            number_of_layers = layers_per_expert,
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
        )
        for _ in 1:number_of_experts
    ]

    router = TokenRouter(embedding_dimension, number_of_experts; hidden_dim = router_hidden_dim)
    moe = GatedExperts(router, experts; top_k = top_k, use_ste = use_ste, logic_floor = logic_floor)

    return MoETModel(
        vocab_size,
        max_sequence_length,
        embedding_dimension,
        number_of_heads,
        time_dimension,
        number_of_experts,
        Lux.Embedding(vocab_size => embedding_dimension),
        Lux.Embedding(max_sequence_length => embedding_dimension),
        moe,
        Lux.LayerNorm((embedding_dimension,)),
        Lux.Dense(embedding_dimension => vocab_size; use_bias = false),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, model::MoETModel)
    return (
        TokenEmbedding = Lux.initialparameters(rng, model.TokenEmbedding),
        PositionEmbedding = Lux.initialparameters(rng, model.PositionEmbedding),
        MoE = Lux.initialparameters(rng, model.MoE),
        FinalNorm = Lux.initialparameters(rng, model.FinalNorm),
        OutputHead = Lux.initialparameters(rng, model.OutputHead),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, model::MoETModel)
    return (
        TokenEmbedding = Lux.initialstates(rng, model.TokenEmbedding),
        PositionEmbedding = Lux.initialstates(rng, model.PositionEmbedding),
        MoE = Lux.initialstates(rng, model.MoE),
        FinalNorm = Lux.initialstates(rng, model.FinalNorm),
        OutputHead = Lux.initialstates(rng, model.OutputHead),
    )
end

function (model::MoETModel)(
    inputs,
    params,
    state;
    logic_mask = nothing,
    temperature::Float32 = 1.0f0,
    return_gates::Bool = false,
)
    if inputs isa NamedTuple
        token_ids = inputs.token_ids
        time_input = haskey(inputs, :time_input) ? inputs.time_input : nothing
    else
        token_ids = inputs
        time_input = nothing
    end

    seq_len = size(token_ids, 1)
    is_batched = ndims(token_ids) == 2
    batch_size = is_batched ? size(token_ids, 2) : 1

    token_ids_batched = is_batched ? token_ids : reshape(token_ids, :, 1)
    token_flat = vec(token_ids_batched)
    token_emb_flat, tok_state = model.TokenEmbedding(token_flat, params.TokenEmbedding, state.TokenEmbedding)
    token_emb = reshape(token_emb_flat, model.embedding_dimension, seq_len, batch_size)

    position_indices_cpu = collect(1:seq_len)
    position_indices = to_device_like(token_ids, position_indices_cpu)
    pos_emb_raw, pos_state = model.PositionEmbedding(position_indices, params.PositionEmbedding, state.PositionEmbedding)
    pos_emb = reshape(pos_emb_raw, model.embedding_dimension, seq_len, 1)

    hidden = token_emb .+ pos_emb

    if time_input === nothing
        t_input_cpu = is_batched ? zeros(Float32, model.time_dimension, batch_size) : zeros(Float32, model.time_dimension)
        time_input = to_device_like(token_ids, t_input_cpu)
    end

    if return_gates
        hidden, gates, moe_state = model.MoE(
            (hidden, time_input),
            params.MoE,
            state.MoE;
            logic_mask = logic_mask,
            temperature = temperature,
            return_gates = true,
        )
    else
        hidden, moe_state = model.MoE(
            (hidden, time_input),
            params.MoE,
            state.MoE;
            logic_mask = logic_mask,
            temperature = temperature,
            return_gates = false,
        )
    end

    hidden, norm_state = model.FinalNorm(hidden, params.FinalNorm, state.FinalNorm)
    logits, head_state = model.OutputHead(hidden, params.OutputHead, state.OutputHead)

    new_state = (
        TokenEmbedding = tok_state,
        PositionEmbedding = pos_state,
        MoE = moe_state,
        FinalNorm = norm_state,
        OutputHead = head_state,
    )

    return return_gates ? (logits, gates, new_state) : (logits, new_state)
end

export MoETConfig, ExpertTower, MoETModel

end
