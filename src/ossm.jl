module ossm

import LuxCore, Lux, Random, NNlib

struct ossm <: Lux.AbstractLuxLayer
    dim_in::Int
    dim_out::Int
    oscillators_count::Int
end

@inline function ossm_dim(block::ossm)
    # state_dim = 2 * oscillators_count (two coordinates per oscillator)
    2 * block.oscillators_count
end

function apply_oscillation(block, x, ρ, θ)
    H = block.oscillators_count
    @assert size(x) == (2H, 1)             # Convention A: state is a (2H,1) column
    @assert length(ρ) == H && length(θ) == H

    # View state as H oscillator columns, each a 2-vector
    x_view = reshape(x, 2, H)              # (2, H)
    slices = eachslice(x_view; dims = 2)   # iterate columns => one oscillator state xi ∈ ℝ^2

    # Apply per-oscillator damped rotation: xi ↦ ρi * R(θi) * xi
    cols = [
        ρi * [cos(θi) -sin(θi); sin(θi) cos(θi)] * xi for (ρi, θi, xi) in zip(ρ, θ, slices)
    ]                                      # length-H collection of 2-vectors

    X_next = reduce(hcat, cols)            # (2, H)
    return reshape(X_next, 2H, 1)          # back to (2H, 1)
end

function Lux.initialparameters(rng, block::ossm)
    state_dim = ossm_dim(block)
    H = block.oscillators_count

    return (
        # Continuous-time-ish params (learned)
        ω = randn(rng, H),                 # frequencies per oscillator (unconstrained)
        α = randn(rng, H),                 # raw damping params (we'll softplus in forward to force α>0)

        # Input/output mixing
        B = randn(rng, state_dim, block.dim_in),
        C = randn(rng, block.dim_out, state_dim),
        D = zeros(Float32, block.dim_out, block.dim_in),

        # Selective step: Δt = softplus(WΔ * u_t + bΔ)
        WΔ = randn(rng, H, block.dim_in),  # (H, dim_in)
        bΔ = randn(rng, H),                # (H,)
    )
end

function Lux.initialstates(rng, block::ossm)
    # Convention A: keep state as a (2H,1) column
    (; oscillation_state = zeros(Float32, ossm_dim(block), 1))
end

function oscillator_step(block, params, xt, ut)
    # xt :: (2H,1), ut :: (dim_in,1)
    (; ω, α, B, C, D, WΔ, bΔ) = params

    @assert size(xt) == (ossm_dim(block), 1)
    @assert size(ut) == (block.dim_in, 1)

    # --- Selective step size Δt ---
    # WΔ * ut => (H,1); bΔ is length-H, so reshape to (H,1) before adding.
    Δt = NNlib.softplus.(WΔ * ut .+ reshape(bΔ, :, 1))  # (H,1)
    Δt = vec(Δt)                                        # (H,) so it zips nicely later

    # --- Stable damping and phase advance ---
    # Force α positive so ρ = exp(-α Δt) stays in (0,1]
    αpos = NNlib.softplus.(α)            # (H,)
    ρ = exp.(-(αpos .* Δt))              # (H,)
    θ = ω .* Δt                          # (H,)

    # State update: x_{t+1} = A(ρ,θ) x_t + B u_t
    x_next = apply_oscillation(block, xt, ρ, θ) + B * ut   # (2H,1)

    # Output: y_t = C x_t + D u_t  (you can switch to x_next if you prefer)
    y = C * xt + D * ut                                     # (dim_out,1)

    return (y, (; oscillation_state = x_next))
end

function (block::ossm)(u, params, state)
    # u :: (dim_in, T), columns are tokens
    xt0 = state.oscillation_state
    @assert size(xt0) == (ossm_dim(block), 1)

    T = size(u, 2)
    Y = similar(u, block.dim_out, T)  # preallocate (dim_out, T)

    it = enumerate(eachslice(u; dims = 2))  # (t, slice) where slice is a length-dim_in vector
    xt_final = foldl(it; init = xt0) do xt, (t, slice)
        ut = reshape(slice, block.dim_in, 1)          # Convention A: (dim_in, 1)
        y, st = oscillator_step(block, params, xt, ut)

        @views Y[:, t] .= vec(y)                      # store y (dim_out,1) into column t
        st.oscillation_state                           # return next state as fold accumulator
    end

    return (Y, (; oscillation_state = xt_final))
end
end
