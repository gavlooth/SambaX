using Test

include("../src/TiDAR.jl")
using .TiDAR

@testset "TiDAR Verifier Alignment" begin
    vocab = 8
    prefix_ids = [2, 3]
    drafted_ids = [2, 3, 4, 5]  # prefix + 2 drafts
    prefix_len = length(prefix_ids)

    # Build verifier logits with BOS offset = 1
    # We want verifier argmax at positions:
    # pos=3 -> col=3, pos=4 -> col=4
    seq_len_with_bos = length(drafted_ids) + 1
    verifier_logits = fill(-10.0f0, vocab, seq_len_with_bos)
    verifier_logits[4, 3] = 10.0f0
    verifier_logits[5, 4] = 10.0f0

    accepted, rejection_idx, replacement = verify_and_accept(
        drafted_ids, prefix_len, verifier_logits;
        mode = :argmax,
        verifier_offset = 1
    )

    @test accepted == 2
    @test rejection_idx === nothing
    @test replacement === nothing
end
