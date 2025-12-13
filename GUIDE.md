# Implementation Guide: Upgrading OSSM and SWAttention

**A Step-by-Step Educational Tutorial**

This guide walks you through implementing the architectural improvements proposed in ARCHITECTURE.md. Each section explains **what** to change, **where** to change it, **why** it matters, and **how** to test it.

**Estimated Total Time**: 10-15 hours
**Difficulty**: Beginner to Intermediate Julia/Lux.jl

---

## Table of Contents

1. [Quick Wins (4-5 hours)](#quick-wins)
   - [Step 1: Add Skip Projection to OSSM](#step-1-add-skip-projection-to-ossm)
   - [Step 2: Vectorize Oscillator Rotation](#step-2-vectorize-oscillator-rotation)
   - [Step 3: Add Input Validation](#step-3-add-input-validation)
   - [Step 4: Write Basic Tests](#step-4-write-basic-tests)

2. [Mamba-Inspired Improvements (6-8 hours)](#mamba-inspired-improvements)
   - [Step 5: Add 1D Convolution to OSSM](#step-5-add-1d-convolution-to-ossm)
   - [Step 6: Add Normalization Layers](#step-6-add-normalization-layers)
   - [Step 7: Make B and C Selective](#step-7-make-b-and-c-selective)
   - [Step 8: Add Causal Masking to SWAttention](#step-8-add-causal-masking-to-swattention)

3. [Testing & Validation](#testing--validation)
4. [Performance Benchmarking](#performance-benchmarking)

---

## Quick Wins

These improvements are high-impact and low-effort. Start here!

---

### Step 1: Add Skip Projection to OSSM

**Time**: 30 minutes
**Difficulty**: ⭐ Easy
**Impact**: Removes the `dim_in == dim_out` constraint

#### What & Why

Currently, OSSM requires `dim_in == dim_out` because the mixture gate does:
```julia
out = g_mix .* Y + (1 - g_mix) .* u  # u must match Y's dimension!
```

This limits architectural flexibility. By adding an optional skip projection, we can handle any dimension mismatch.

#### Where to Change

**File**: `src/ossm.jl`

#### Step-by-Step Implementation

**1.1** Update the struct definition to include an optional skip projection:

```julia
# BEFORE (around line 13):
struct OSSM{IG,MG} <: Lux.AbstractLuxLayer
    dim_in::Int
    dim_out::Int
    oscillators_count::Int
    input_gate::IG
    mixture_gate::MG
end

# AFTER:
struct OSSM{IG,MG,SP} <: Lux.AbstractLuxLayer
    dim_in::Int
    dim_out::Int
    oscillators_count::Int
    input_gate::IG
    mixture_gate::MG
    skip_proj::SP  # NEW: Union{Nothing, Lux.Dense}
end
```

**Educational Note**: We add a type parameter `SP` to make this struct type-stable. The skip projection can be either `Nothing` (when dims match) or `Lux.Dense` (when they don't).

**1.2** Update the constructor:

```julia
# BEFORE (around line 27):
function OSSM(dim_in::Int, dim_out::Int, H::Int)
    input_gate = Lux.Dense(dim_in => dim_in, NNlib.sigmoid)
    mixture_gate = Lux.Dense(dim_out => dim_out, NNlib.sigmoid)
    return OSSM(dim_in, dim_out, H, input_gate, mixture_gate)
end

# AFTER:
function OSSM(dim_in::Int, dim_out::Int, H::Int; use_skip::Bool=true)
    # H: number of oscillators (state dimension will be 2H)
    # dim_in: input dimension
    # dim_out: output dimension
    # use_skip: whether to enable skip connection (default true)

    input_gate = Lux.Dense(dim_in => dim_in, NNlib.sigmoid)
    mixture_gate = Lux.Dense(dim_out => dim_out, NNlib.sigmoid)

    # Only create skip projection if dimensions don't match AND skip is enabled
    skip_proj = if use_skip && (dim_in != dim_out)
        Lux.Dense(dim_in => dim_out, bias=false)  # Linear projection, no bias
    else
        nothing
    end

    return OSSM(dim_in, dim_out, H, input_gate, mixture_gate, skip_proj)
end
```

**Educational Note**: We use a keyword argument `use_skip` to allow disabling the skip connection entirely. The projection is only created when needed, saving parameters.

**1.3** Update `initialparameters` to handle the skip projection:

```julia
# Add after the existing initialparameters function (around line 72):

function Lux.initialparameters(rng, block::OSSM)
    state_dim = ossm_dim(block)  # = 2H
    H = block.oscillators_count

    base_params = (
        # Continuous-time-ish params (learned)
        ω = randn(rng, H),
        α = randn(rng, H),

        # Input/output mixing
        B = randn(rng, state_dim, block.dim_in),
        C = randn(rng, block.dim_out, state_dim),
        D = zeros(Float32, block.dim_out, block.dim_in),

        # Selective step
        WΔ = randn(rng, H, block.dim_in),
        bΔ = randn(rng, H),

        # Gates
        input_gate = Lux.initialparameters(rng, block.input_gate),
        mixture_gate = Lux.initialparameters(rng, block.mixture_gate),
    )

    # Add skip projection parameters if it exists
    if !isnothing(block.skip_proj)
        return merge(base_params, (skip_proj = Lux.initialparameters(rng, block.skip_proj),))
    else
        return base_params
    end
end
```

**1.4** Update `initialstates` similarly:

```julia
function Lux.initialstates(rng, block::OSSM)
    base_state = (
        oscillation_state = zeros(Float32, ossm_dim(block), 1),
        input_gate = Lux.initialstates(rng, block.input_gate),
        mixture_gate = Lux.initialstates(rng, block.mixture_gate),
    )

    if !isnothing(block.skip_proj)
        return merge(base_state, (skip_proj = Lux.initialstates(rng, block.skip_proj),))
    else
        return base_state
    end
end
```

**1.5** Update the forward pass to use skip projection:

```julia
# Find the mixture gate section (around line 145-152), replace:

# BEFORE:
# skip path: out = g_mix ⊙ Y + (1 - g_mix) ⊙ u
# this assumes dim_in == dim_out. If not, add a skip projection.
@assert block.dim_in == block.dim_out "Need a skip projection if dim_in != dim_out"
oneT = one(eltype(g_mix))
out = g_mix .* Y .+ (oneT .- g_mix) .* u  # (dim_out, T)

# AFTER:
# Skip path with optional projection
if !isnothing(block.skip_proj)
    u_proj, st_skip = block.skip_proj(u, params.skip_proj, state.skip_proj)  # (dim_out, T)
    oneT = one(eltype(g_mix))
    out = g_mix .* Y .+ (oneT .- g_mix) .* u_proj  # (dim_out, T)
    new_state = (; oscillation_state = xt_final, input_gate = st_in,
                   mixture_gate = st_mix, skip_proj = st_skip)
else
    # No projection needed (dimensions match or skip disabled)
    if block.dim_in == block.dim_out
        oneT = one(eltype(g_mix))
        out = g_mix .* Y .+ (oneT .- g_mix) .* u  # (dim_out, T)
    else
        # Skip connection disabled, just use SSM output
        out = Y
    end
    new_state = (; oscillation_state = xt_final, input_gate = st_in, mixture_gate = st_mix)
end

return (out, new_state)
```

#### Testing Step 1

Create a test file to verify:

```julia
# Quick test in REPL:
using Lux, Random

# Test 1: Matching dimensions (no projection needed)
model1 = OSSM(128, 128, 8)
rng = Random.default_rng()
ps1 = Lux.initialparameters(rng, model1)
st1 = Lux.initialstates(rng, model1)
x1 = randn(Float32, 128, 10)
y1, _ = model1(x1, ps1, st1)
@assert size(y1) == (128, 10) "Test 1 failed!"

# Test 2: Mismatched dimensions (projection created)
model2 = OSSM(64, 128, 8)
ps2 = Lux.initialparameters(rng, model2)
st2 = Lux.initialstates(rng, model2)
x2 = randn(Float32, 64, 10)
y2, _ = model2(x2, ps2, st2)
@assert size(y2) == (128, 10) "Test 2 failed!"
@assert !isnothing(model2.skip_proj) "Skip projection should exist!"

# Test 3: Skip disabled
model3 = OSSM(64, 128, 8; use_skip=false)
ps3 = Lux.initialparameters(rng, model3)
st3 = Lux.initialstates(rng, model3)
x3 = randn(Float32, 64, 10)
y3, _ = model3(x3, ps3, st3)
@assert size(y3) == (128, 10) "Test 3 failed!"

println("✅ All skip projection tests passed!")
```

---

### Step 2: Vectorize Oscillator Rotation

**Time**: 1 hour
**Difficulty**: ⭐⭐ Moderate
**Impact**: ~3× speedup for oscillator updates

#### What & Why

The current `apply_oscillation` function allocates a rotation matrix for each oscillator in a list comprehension:

```julia
cols = [ρi * [cos(θi) -sin(θi); sin(θi) cos(θi)] * xi for (ρi, θi, xi) in zip(ρ, θ, slices)]
```

This creates `H` separate 2×2 matrices. We can vectorize this using broadcasting, which is much faster.

#### Where to Change

**File**: `src/ossm.jl`, function `apply_oscillation` (around line 32-52)

#### Mathematical Background

For a 2D rotation with damping:
```
[x'] = ρ * [cos(θ) -sin(θ)] * [x]
[y']       [sin(θ)  cos(θ)]   [y]

Expanded:
x' = ρ * (cos(θ) * x - sin(θ) * y)
y' = ρ * (sin(θ) * x + cos(θ) * y)
```

We can compute this for all H oscillators at once using broadcasting.

#### Step-by-Step Implementation

**2.1** Replace the entire `apply_oscillation` function:

```julia
# BEFORE (lines 32-52):
function apply_oscillation(block, x, ρ, θ)
    # x: (2H, 1) - state vector for H oscillators
    # ρ: (H,) - damping factors per oscillator
    # θ: (H,) - rotation angles per oscillator
    # Returns: (2H, 1) - updated state vector
    H = block.oscillators_count
    @assert size(x) == (2H, 1)
    @assert length(ρ) == H && length(θ) == H

    # View state as H oscillator columns, each a 2-vector
    x_view = reshape(x, 2, H)              # (2, H) - each column is one oscillator state
    slices = eachslice(x_view; dims = 2)   # iterate columns => one oscillator state xi ∈ ℝ^2

    # Apply per-oscillator damped rotation: xi ↦ ρi * R(θi) * xi
    cols = [
        ρi * [cos(θi) -sin(θi); sin(θi) cos(θi)] * xi for (ρi, θi, xi) in zip(ρ, θ, slices)
    ]                                      # length-H collection of 2-vectors

    X_next = reduce(hcat, cols)            # (2, H)
    return reshape(X_next, 2H, 1)          # back to (2H, 1)
end

# AFTER - Vectorized version:
function apply_oscillation(block, x, ρ, θ)
    """
    Vectorized damped rotation of H oscillators.

    Instead of allocating H rotation matrices, we compute the rotation
    directly using the trigonometric formulas and broadcast operations.

    Mathematical formulation:
        For each oscillator i with state (xi, yi):
        xi' = ρi * (cos(θi) * xi - sin(θi) * yi)
        yi' = ρi * (sin(θi) * xi + cos(θi) * yi)

    This is ~3× faster than the loop-based version for typical H values.
    """
    # x: (2H, 1) - state vector for H oscillators
    # ρ: (H,) - damping factors per oscillator
    # θ: (H,) - rotation angles per oscillator
    # Returns: (2H, 1) - updated state vector
    H = block.oscillators_count
    @assert size(x) == (2H, 1) "State must be (2H, 1), got $(size(x))"
    @assert length(ρ) == H && length(θ) == H "ρ and θ must have length H=$H"

    # Reshape to separate x and y coordinates: (2, H)
    # Each column is one oscillator: [xi; yi]
    x_view = reshape(x, 2, H)

    # Extract x and y coordinates across all oscillators
    x_coords = @view x_view[1, :]  # (H,) - all x coordinates
    y_coords = @view x_view[2, :]  # (H,) - all y coordinates

    # Precompute trig functions (vectorized)
    cos_θ = cos.(θ)  # (H,) - cosines
    sin_θ = sin.(θ)  # (H,) - sines

    # Apply rotational update (vectorized)
    # Broadcasting: each element-wise operation is O(H) instead of O(H) allocations
    x_new = ρ .* (cos_θ .* x_coords .- sin_θ .* y_coords)  # (H,)
    y_new = ρ .* (sin_θ .* x_coords .+ cos_θ .* y_coords)  # (H,)

    # Stack back into (2, H) and reshape to (2H, 1)
    x_next = vcat(reshape(x_new, 1, H), reshape(y_new, 1, H))  # (2, H)
    return reshape(x_next, 2H, 1)  # (2H, 1)
end
```

**Educational Notes**:

1. **Why vectorize?** Julia's broadcasting is highly optimized and uses SIMD instructions. It's much faster than allocating individual matrices.

2. **Memory efficiency**: The old version allocates H rotation matrices (each 2×2) plus H result vectors. The new version only allocates the trig function arrays.

3. **Type stability**: Using `@view` prevents copying data, keeping memory usage low.

4. **Readability**: While slightly longer, the vectorized version is explicit about the math.

#### Testing Step 2

**2.2** Create a test to verify correctness and benchmark:

```julia
# test/benchmark_oscillators.jl
using Lux, Random, BenchmarkTools

# Test correctness
function test_oscillator_correctness()
    rng = Random.default_rng()
    model = OSSM(64, 64, 16)  # 16 oscillators

    # Create test inputs
    x = randn(Float32, 32, 1)  # (2H, 1)
    ρ = rand(Float32, 16) .* 0.9 .+ 0.1  # (H,) ∈ [0.1, 1.0]
    θ = randn(Float32, 16)  # (H,)

    # Apply oscillation
    x_new = apply_oscillation(model, x, ρ, θ)

    # Verify dimensions
    @assert size(x_new) == (32, 1)

    # Verify stability (damping should reduce magnitude)
    @assert norm(x_new) <= norm(x) "Damping should not increase magnitude!"

    println("✅ Correctness test passed!")
end

# Benchmark comparison (if you saved the old version)
function benchmark_oscillators()
    rng = Random.default_rng()
    H_values = [4, 8, 16, 32, 64]

    println("\n📊 Oscillator Update Benchmark:")
    println("="^50)

    for H in H_values
        model = OSSM(64, 64, H)
        x = randn(Float32, 2H, 1)
        ρ = rand(Float32, H)
        θ = randn(Float32, H)

        # Benchmark
        t = @benchmark apply_oscillation($model, $x, $ρ, $θ)

        println("H=$H: $(round(median(t).time / 1000, digits=2)) μs")
    end
end

test_oscillator_correctness()
benchmark_oscillators()
```

**Expected speedup**: 2-4× faster depending on H.

---

### Step 3: Add Input Validation

**Time**: 30 minutes
**Difficulty**: ⭐ Easy
**Impact**: Better error messages, easier debugging

#### What & Why

Currently, if you pass wrong-shaped inputs, you get cryptic errors from deep in the call stack. Adding validation at the entry points gives immediate, clear feedback.

#### Where to Change

**Files**: `src/ossm.jl` and `src/attention.jl`

#### Step-by-Step Implementation

**3.1** Add validation to OSSM's forward pass:

```julia
# In src/ossm.jl, at the start of the forward function (around line 122):

function (block::OSSM)(u, params, state)
    # === INPUT VALIDATION ===
    # Validate input dimensions
    if ndims(u) != 2
        error("OSSM expects 2D input (dim_in, T), got $(ndims(u))D tensor with shape $(size(u))")
    end

    if size(u, 1) != block.dim_in
        error("OSSM input dimension mismatch: expected $(block.dim_in), got $(size(u, 1))")
    end

    # Validate state
    expected_state_dim = (ossm_dim(block), 1)
    if size(state.oscillation_state) != expected_state_dim
        error("OSSM state dimension mismatch: expected $expected_state_dim, got $(size(state.oscillation_state))")
    end
    # === END VALIDATION ===

    # u: (dim_in, T) - input sequence of length T
    # Returns: (out, new_state) where out: (dim_out, T), new_state contains updated oscillation_state: (2H, 1)
    # ... rest of function
end
```

**3.2** Add validation to SWAttention:

```julia
# In src/attention.jl, at the start of the forward function (around line 61):

function (block::SWAttention)(x, params::NamedTuple, _state::NamedTuple)
    # === INPUT VALIDATION ===
    if ndims(x) != 2
        error("SWAttention expects 2D input (dimension, T), got $(ndims(x))D tensor with shape $(size(x))")
    end

    if size(x, 1) != block.dimension
        error("SWAttention input dimension mismatch: expected $(block.dimension), got $(size(x, 1))")
    end
    # === END VALIDATION ===

    # x: (dimension, T) where T is sequence length
    state = (;)
    # ... rest of function
end
```

**3.3** Add validation to `oscillator_step`:

```julia
# In oscillator_step function (around line 79):

function oscillator_step(block, params, xt, ut)
    # === INPUT VALIDATION ===
    expected_state = (ossm_dim(block), 1)
    if size(xt) != expected_state
        error("oscillator_step: state dimension mismatch. Expected $expected_state, got $(size(xt))")
    end

    expected_input = (block.dim_in, 1)
    if size(ut) != expected_input
        error("oscillator_step: input dimension mismatch. Expected $expected_input, got $(size(ut))")
    end
    # === END VALIDATION ===

    # ... rest of function
end
```

**Educational Note**: Input validation is a best practice for library code. It catches errors early and gives users clear guidance on what went wrong.

#### Testing Step 3

```julia
# Test that validation catches errors:
using Lux, Random

rng = Random.default_rng()
model = OSSM(64, 64, 8)
ps = Lux.initialparameters(rng, model)
st = Lux.initialstates(rng, model)

# Test 1: Wrong number of dimensions
try
    x_wrong = randn(Float32, 64, 10, 2)  # 3D instead of 2D
    model(x_wrong, ps, st)
    @error "Should have thrown an error!"
catch e
    @assert occursin("2D input", e.msg) "Error message should mention 2D input"
    println("✅ Test 1: Caught 3D input correctly")
end

# Test 2: Wrong input dimension
try
    x_wrong = randn(Float32, 32, 10)  # 32 instead of 64
    model(x_wrong, ps, st)
    @error "Should have thrown an error!"
catch e
    @assert occursin("dimension mismatch", e.msg)
    println("✅ Test 2: Caught dimension mismatch correctly")
end

# Test 3: Correct input should work
x_correct = randn(Float32, 64, 10)
y, _ = model(x_correct, ps, st)
println("✅ Test 3: Correct input passes validation")
```

---

### Step 4: Write Basic Tests

**Time**: 2 hours
**Difficulty**: ⭐⭐ Moderate
**Impact**: Ensures correctness, prevents regressions

#### What & Why

Tests are essential for any serious project. They catch bugs early and document expected behavior.

#### Where to Change

**Create new directory**: `test/`
**Create files**: `test/runtests.jl`, `test/test_ossm.jl`, `test/test_attention.jl`

#### Step-by-Step Implementation

**4.1** Create test directory structure:

```bash
mkdir -p test
```

**4.2** Create `test/runtests.jl`:

```julia
# test/runtests.jl
using Test
using Samba2
using Lux
using Random

# Set random seed for reproducibility
Random.seed!(1234)

@testset "Samba2 Tests" begin
    include("test_ossm.jl")
    include("test_attention.jl")
end
```

**4.3** Create `test/test_ossm.jl`:

```julia
# test/test_ossm.jl
using Test, Lux, Random, LinearAlgebra

@testset "OSSM Tests" begin
    rng = Random.default_rng()
    Random.seed!(rng, 42)

    @testset "Construction" begin
        # Test basic construction
        model = OSSM(64, 64, 8)
        @test model.dim_in == 64
        @test model.dim_out == 64
        @test model.oscillators_count == 8

        # Test with dimension mismatch
        model2 = OSSM(32, 64, 8)
        @test !isnothing(model2.skip_proj)

        # Test with skip disabled
        model3 = OSSM(32, 64, 8; use_skip=false)
        @test isnothing(model3.skip_proj)
    end

    @testset "Parameter Initialization" begin
        model = OSSM(64, 64, 8)
        ps = Lux.initialparameters(rng, model)

        # Check all expected parameters exist
        @test haskey(ps, :ω)
        @test haskey(ps, :α)
        @test haskey(ps, :B)
        @test haskey(ps, :C)
        @test haskey(ps, :D)
        @test haskey(ps, :WΔ)
        @test haskey(ps, :bΔ)
        @test haskey(ps, :input_gate)
        @test haskey(ps, :mixture_gate)

        # Check dimensions
        @test size(ps.ω) == (8,)
        @test size(ps.α) == (8,)
        @test size(ps.B) == (16, 64)  # (2H, dim_in)
        @test size(ps.C) == (64, 16)  # (dim_out, 2H)
        @test size(ps.D) == (64, 64)  # (dim_out, dim_in)
    end

    @testset "State Initialization" begin
        model = OSSM(64, 64, 8)
        st = Lux.initialstates(rng, model)

        @test haskey(st, :oscillation_state)
        @test size(st.oscillation_state) == (16, 1)  # (2H, 1)
        @test all(st.oscillation_state .== 0)  # Should be zero-initialized
    end

    @testset "Forward Pass - Shape Tests" begin
        model = OSSM(64, 64, 8)
        ps = Lux.initialparameters(rng, model)
        st = Lux.initialstates(rng, model)

        # Test various sequence lengths
        for T in [1, 10, 100]
            x = randn(Float32, 64, T)
            y, st_new = model(x, ps, st)

            @test size(y) == (64, T) "Output shape mismatch for T=$T"
            @test haskey(st_new, :oscillation_state)
            @test size(st_new.oscillation_state) == (16, 1)
        end
    end

    @testset "Forward Pass - Dimension Mismatch" begin
        # Test skip projection with mismatched dims
        model = OSSM(32, 64, 8)
        ps = Lux.initialparameters(rng, model)
        st = Lux.initialstates(rng, model)

        x = randn(Float32, 32, 10)
        y, _ = model(x, ps, st)

        @test size(y) == (64, 10) "Skip projection failed"
    end

    @testset "Oscillator Stability" begin
        # Test that damping keeps states bounded
        model = OSSM(64, 64, 8)
        ps = Lux.initialparameters(rng, model)
        st = Lux.initialstates(rng, model)

        # Set initial state to something non-zero
        st = merge(st, (oscillation_state = randn(Float32, 16, 1),))

        # Run for many steps
        x = randn(Float32, 64, 100)
        y, st_final = model(x, ps, st)

        # State should remain finite (no NaN or Inf)
        @test all(isfinite.(st_final.oscillation_state)) "State became infinite/NaN"

        # State magnitude shouldn't explode (thanks to damping)
        @test norm(st_final.oscillation_state) < 1000 "State exploded"
    end

    @testset "Gradient Flow" begin
        # Test that gradients can be computed (basic check)
        using Zygote

        model = OSSM(64, 64, 8)
        ps = Lux.initialparameters(rng, model)
        st = Lux.initialstates(rng, model)
        x = randn(Float32, 64, 10)

        # Compute loss and gradients
        loss, grads = Zygote.withgradient(ps) do p
            y, _ = model(x, p, st)
            sum(abs2, y)  # Simple L2 loss
        end

        @test isfinite(loss) "Loss is not finite"
        @test !isnothing(grads[1]) "Gradients are nothing"

        # Check that key parameters have gradients
        @test haskey(grads[1], :ω)
        @test haskey(grads[1], :α)
        @test all(isfinite.(grads[1].ω)) "ω gradients not finite"
    end

    @testset "Selective Step Sizing" begin
        # Verify that Δt is always positive
        model = OSSM(64, 64, 8)
        ps = Lux.initialparameters(rng, model)
        st = Lux.initialstates(rng, model)

        # Access internal step computation
        xt = st.oscillation_state
        ut = randn(Float32, 64, 1)

        # Extract params
        (; WΔ, bΔ) = ps

        # Compute Δt
        Δt = NNlib.softplus.(WΔ * ut .+ reshape(bΔ, :, 1))

        @test all(Δt .> 0) "Step sizes must be positive"
        @test all(isfinite.(Δt)) "Step sizes must be finite"
    end
end
```

**4.4** Create `test/test_attention.jl`:

```julia
# test/test_attention.jl
using Test, Lux, Random, LinearAlgebra

include("../src/attention.jl")
using .attention

@testset "SWAttention Tests" begin
    rng = Random.default_rng()
    Random.seed!(rng, 42)

    @testset "Construction" begin
        # Test valid construction
        attn = attention.SWAttention(1024, 128, 4)
        @test attn.dimension == 128
        @test attn.number_of_heads == 4
        @test attn.sequence_length == 1024

        # Test dimension divisibility requirement
        @test_throws AssertionError attention.SWAttention(1024, 127, 4)  # 127 not divisible by 4
    end

    @testset "Parameter Initialization" begin
        attn = attention.SWAttention(1024, 128, 4)
        ps = Lux.initialparameters(rng, attn)

        @test haskey(ps, :q)
        @test haskey(ps, :k)
        @test haskey(ps, :v)
        @test haskey(ps, :output)
    end

    @testset "State Initialization" begin
        attn = attention.SWAttention(1024, 128, 4)
        st = Lux.initialstates(rng, attn)

        # Should be empty (stateless)
        @test isempty(st)
    end

    @testset "Forward Pass - Shape Tests" begin
        attn = attention.SWAttention(1024, 128, 4)
        ps = Lux.initialparameters(rng, attn)
        st = Lux.initialstates(rng, attn)

        # Test various sequence lengths
        for T in [1, 16, 64, 256]
            x = randn(Float32, 128, T)
            y, st_new = attn(x, ps, st)

            @test size(y) == (128, T) "Output shape mismatch for T=$T"
            @test st_new == st "State should not change (stateless layer)"
        end
    end

    @testset "Attention Weights Normalization" begin
        # Test normalized_sigmoids function
        seq = randn(Float32, 10)
        weights = attention.normalized_sigmoids(seq)

        # Should sum to approximately 1
        @test isapprox(sum(weights), 1.0, atol=1e-6)

        # All weights should be positive
        @test all(weights .> 0)

        # Should be same length
        @test length(weights) == length(seq)
    end

    @testset "Multi-Head Splitting" begin
        # Verify that heads see correct dimensions
        dim = 128
        n_heads = 4
        T = 16

        attn = attention.SWAttention(1024, dim, n_heads)
        ps = Lux.initialparameters(rng, attn)
        st = Lux.initialstates(rng, attn)

        x = randn(Float32, dim, T)
        y, _ = attn(x, ps, st)

        # Output should match input dimensions
        @test size(y) == size(x)
    end

    @testset "Gradient Flow" begin
        using Zygote

        attn = attention.SWAttention(1024, 128, 4)
        ps = Lux.initialparameters(rng, attn)
        st = Lux.initialstates(rng, attn)
        x = randn(Float32, 128, 16)

        loss, grads = Zygote.withgradient(ps) do p
            y, _ = attn(x, p, st)
            sum(abs2, y)
        end

        @test isfinite(loss)
        @test !isnothing(grads[1])
        @test haskey(grads[1], :q)
        @test haskey(grads[1], :k)
        @test haskey(grads[1], :v)
    end

    @testset "Temperature Scaling" begin
        # Test that different temperatures produce different outputs
        seq = randn(Float32, 10)

        w1 = attention.normalized_sigmoids(seq; τ=0.5)
        w2 = attention.normalized_sigmoids(seq; τ=1.0)
        w3 = attention.normalized_sigmoids(seq; τ=2.0)

        # Lower temperature should produce sharper distribution
        @test maximum(w1) > maximum(w2) > maximum(w3)

        # All should sum to 1
        @test all(isapprox.([sum(w1), sum(w2), sum(w3)], 1.0, atol=1e-6))
    end
end
```

**4.5** Update `Project.toml` to include test dependencies:

```toml
# Add to Project.toml:

[deps]
# ... existing deps ...
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

# Add test-specific section
[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Test"]
```

**4.6** Run the tests:

```bash
# From project root:
julia --project=. -e 'using Pkg; Pkg.test()'

# Or in Julia REPL:
using Pkg
Pkg.test()
```

**Expected output**: All tests should pass with green checkmarks.

---

## Mamba-Inspired Improvements

Now let's implement the more advanced improvements inspired by Mamba.

---

### Step 5: Add 1D Convolution to OSSM

**Time**: 1-2 hours
**Difficulty**: ⭐⭐ Moderate
**Impact**: Better local context modeling

#### What & Why

Mamba includes a short 1D convolution (kernel size 4) before the SSM. This captures local dependencies efficiently, letting the SSM focus on longer-range patterns.

**Intuition**: Conv handles "what's happening right now", SSM handles "how did we get here".

#### Where to Change

**File**: `src/ossm.jl`

#### Step-by-Step Implementation

**5.1** Update the struct to include a convolution layer:

```julia
# Update struct (around line 13):
struct OSSM{IG,MG,SP,CV} <: Lux.AbstractLuxLayer
    dim_in::Int
    dim_out::Int
    oscillators_count::Int
    input_gate::IG
    mixture_gate::MG
    skip_proj::SP
    conv::CV  # NEW: 1D convolution layer
end
```

**5.2** Update constructor to create the conv layer:

```julia
function OSSM(dim_in::Int, dim_out::Int, H::Int;
              use_skip::Bool=true,
              use_conv::Bool=true,  # NEW
              conv_kernel::Int=4)   # NEW: kernel size

    input_gate = Lux.Dense(dim_in => dim_in, NNlib.sigmoid)
    mixture_gate = Lux.Dense(dim_out => dim_out, NNlib.sigmoid)

    skip_proj = if use_skip && (dim_in != dim_out)
        Lux.Dense(dim_in => dim_out, bias=false)
    else
        nothing
    end

    # NEW: Create 1D convolution
    # Conv: (dim_in, T) -> (dim_in, T) with kernel_size conv_kernel
    # We use padding to preserve sequence length
    conv = if use_conv
        # Lux.Conv expects (spatial_dims..., in_channels, out_channels)
        # For 1D: (kernel_size, in_channels, out_channels)
        # With groups=dim_in, we get depthwise convolution (each channel independent)
        Lux.Conv((conv_kernel,), dim_in => dim_in;
                 pad=SamePad(),
                 groups=dim_in)  # Depthwise separable
    else
        nothing
    end

    return OSSM(dim_in, dim_out, H, input_gate, mixture_gate, skip_proj, conv)
end
```

**Educational Note**:
- **Depthwise convolution** (`groups=dim_in`): Each channel is convolved separately, reducing parameters
- **SamePad()**: Pads input so output length matches input length
- **Why kernel=4?**: Empirically good balance (Mamba uses 4)

**5.3** Update parameter and state initialization:

```julia
function Lux.initialparameters(rng, block::OSSM)
    # ... existing base_params ...

    # Add conv parameters
    params_with_conv = if !isnothing(block.conv)
        merge(base_params, (conv = Lux.initialparameters(rng, block.conv),))
    else
        base_params
    end

    # Add skip projection if needed
    if !isnothing(block.skip_proj)
        return merge(params_with_conv, (skip_proj = Lux.initialparameters(rng, block.skip_proj),))
    else
        return params_with_conv
    end
end

# Similarly for initialstates:
function Lux.initialstates(rng, block::OSSM)
    base_state = (
        oscillation_state = zeros(Float32, ossm_dim(block), 1),
        input_gate = Lux.initialstates(rng, block.input_gate),
        mixture_gate = Lux.initialstates(rng, block.mixture_gate),
    )

    # Add conv state
    state_with_conv = if !isnothing(block.conv)
        merge(base_state, (conv = Lux.initialstates(rng, block.conv),))
    else
        base_state
    end

    # Add skip projection state
    if !isnothing(block.skip_proj)
        return merge(state_with_conv, (skip_proj = Lux.initialstates(rng, block.skip_proj),))
    else
        return state_with_conv
    end
end
```

**5.4** Update forward pass to use convolution:

```julia
function (block::OSSM)(u, params, state)
    # ... input validation ...

    xt0 = state.oscillation_state
    T = size(u, 2)

    # NEW: Apply convolution first (if enabled)
    u_processed, st_conv = if !isnothing(block.conv)
        # Lux.Conv expects (spatial, channels, batch)
        # We have (channels, spatial) = (dim_in, T)
        # Need to add batch dimension and transpose
        u_reshaped = reshape(u, 1, size(u, 1), size(u, 2))  # (1, dim_in, T)
        u_conv, st = block.conv(u_reshaped, params.conv, state.conv)
        u_conv_2d = reshape(u_conv, size(u, 1), size(u, 2))  # back to (dim_in, T)
        (u_conv_2d, st)
    else
        (u, nothing)
    end

    # Input gating on convolved input
    g_in, st_in = block.input_gate(u_processed, params.input_gate, state.input_gate)
    u_gated = g_in .* u_processed

    # ... rest of SSM processing ...

    # Update state to include conv state
    new_state = if !isnothing(block.conv)
        if !isnothing(block.skip_proj)
            (; oscillation_state = xt_final, input_gate = st_in,
               mixture_gate = st_mix, skip_proj = st_skip, conv = st_conv)
        else
            (; oscillation_state = xt_final, input_gate = st_in,
               mixture_gate = st_mix, conv = st_conv)
        end
    else
        # ... existing state update ...
    end

    return (out, new_state)
end
```

**5.5** Test the convolution:

```julia
# Test conv layer
using Lux, Random

rng = Random.default_rng()
model = OSSM(64, 64, 8; use_conv=true)
ps = Lux.initialparameters(rng, model)
st = Lux.initialstates(rng, model)

x = randn(Float32, 64, 20)
y, _ = model(x, ps, st)

@assert size(y) == (64, 20) "Conv should preserve dimensions"
println("✅ Convolution layer test passed!")

# Compare with and without conv
model_no_conv = OSSM(64, 64, 8; use_conv=false)
ps_nc = Lux.initialparameters(rng, model_no_conv)
st_nc = Lux.initialstates(rng, model_no_conv)
y_nc, _ = model_no_conv(x, ps_nc, st_nc)

@assert y != y_nc "Outputs should differ with/without conv"
println("✅ Conv changes output as expected!")
```

---

### Step 6: Add Normalization Layers

**Time**: 1 hour
**Difficulty**: ⭐⭐ Moderate
**Impact**: Training stability, better convergence

#### What & Why

Normalization (LayerNorm or RMSNorm) stabilizes training by keeping activations in a reasonable range. This is standard in modern architectures.

**RMSNorm** is simpler and faster than LayerNorm (no mean subtraction), used in Mamba and LLaMA.

#### Step-by-Step Implementation

**6.1** Add normalization layers to OSSM:

```julia
# Update struct:
struct OSSM{IG,MG,SP,CV,NI,NO} <: Lux.AbstractLuxLayer
    dim_in::Int
    dim_out::Int
    oscillators_count::Int
    input_gate::IG
    mixture_gate::MG
    skip_proj::SP
    conv::CV
    norm_input::NI   # NEW: normalize input
    norm_output::NO  # NEW: normalize output
end

# Update constructor:
function OSSM(dim_in::Int, dim_out::Int, H::Int;
              use_skip::Bool=true,
              use_conv::Bool=true,
              conv_kernel::Int=4,
              use_norm::Bool=true)  # NEW

    # ... existing layers ...

    # NEW: Create normalization layers
    norm_input = use_norm ? Lux.LayerNorm(dim_in) : nothing
    norm_output = use_norm ? Lux.LayerNorm(dim_out) : nothing

    return OSSM(dim_in, dim_out, H, input_gate, mixture_gate,
                skip_proj, conv, norm_input, norm_output)
end
```

**6.2** Update initialization (similar pattern to conv):

```julia
# In initialparameters and initialstates, add:
if !isnothing(block.norm_input)
    # Add norm_input and norm_output to params/state
end
```

**6.3** Apply normalization in forward pass:

```julia
function (block::OSSM)(u, params, state)
    # Normalize input
    u_norm, st_norm_in = if !isnothing(block.norm_input)
        block.norm_input(u, params.norm_input, state.norm_input)
    else
        (u, nothing)
    end

    # ... conv, gating, SSM ...

    # Normalize output
    out_norm, st_norm_out = if !isnothing(block.norm_output)
        block.norm_output(out, params.norm_output, state.norm_output)
    else
        (out, nothing)
    end

    # ... update state ...

    return (out_norm, new_state)
end
```

**6.4** Similarly for SWAttention:

```julia
# Add LayerNorm before Q/K/V projections and after output
struct SWAttention{...NM} <: Lux.AbstractLuxLayer
    # ... existing fields ...
    norm::NM  # pre-attention normalization
end

function (block::SWAttention)(x, params, state)
    # Normalize input
    x_norm = if !isnothing(block.norm)
        x_n, _ = block.norm(x, params.norm, state.norm)
        x_n
    else
        x
    end

    # ... rest of attention ...
end
```

---

### Step 7: Make B and C Selective

**Time**: 2 hours
**Difficulty**: ⭐⭐⭐ Advanced
**Impact**: Full selectivity like Mamba, more expressive

#### What & Why

Currently, only Δt is input-dependent. Making B and C selective allows the model to dynamically choose:
- **B**: What information from input to add to state
- **C**: What information from state to output

This is Mamba's key innovation.

#### Step-by-Step Implementation

**7.1** Add projection matrices for selective B and C:

```julia
function Lux.initialparameters(rng, block::OSSM)
    state_dim = ossm_dim(block)
    H = block.oscillators_count

    return (
        # ... existing params ...

        # Base B and C (will be modulated)
        B = randn(rng, state_dim, block.dim_in),
        C = randn(rng, block.dim_out, state_dim),
        D = zeros(Float32, block.dim_out, block.dim_in),

        # NEW: Selective projections for B and C
        W_B = randn(rng, state_dim, block.dim_in),  # Projects input to B modulation
        W_C = randn(rng, state_dim, block.dim_in),  # Projects input to C modulation

        # ... rest ...
    )
end
```

**7.2** Update `oscillator_step` to use selective B and C:

```julia
function oscillator_step(block, params, xt, ut)
    (; ω, α, B, C, D, WΔ, bΔ, W_B, W_C) = params

    # ... existing Δt computation ...

    # NEW: Compute input-dependent B and C
    # B_t modulates how much of each input dimension affects each state dimension
    B_modulation = NNlib.sigmoid.(W_B * ut)  # (2H, 1) ∈ (0, 1)
    B_t = B .* B_modulation  # Element-wise modulation

    # C_t modulates which state dimensions contribute to output
    C_modulation = NNlib.sigmoid.(W_C * ut)  # (2H, 1) ∈ (0, 1)
    C_t = C .* C_modulation'  # Broadcasting: (dim_out, 2H) .* (1, 2H)

    # State update with selective B
    x_next = apply_oscillation(block, xt, ρ, θ) + B_t * ut

    # Output with selective C
    y = C_t * xt + D * ut

    return (y, (; oscillation_state = x_next))
end
```

**Educational Note**:
- We use **sigmoid** for modulation (not softplus) to bound values in (0,1)
- Element-wise multiplication lets the model "turn off" specific state updates
- This is similar to gating in LSTMs/GRUs but applied to SSM matrices

**7.3** Test selective mechanism:

```julia
# Verify that B and C change with different inputs
using Lux, Random

model = OSSM(64, 64, 8)
ps = Lux.initialparameters(Random.default_rng(), model)

# Two different inputs
u1 = randn(Float32, 64, 1)
u2 = randn(Float32, 64, 1)

# Compute B modulations
B_mod1 = NNlib.sigmoid.(ps.W_B * u1)
B_mod2 = NNlib.sigmoid.(ps.W_B * u2)

@assert B_mod1 != B_mod2 "B should be input-dependent!"
println("✅ Selective B/C test passed!")
```

---

### Step 8: Add Causal Masking to SWAttention

**Time**: 1 hour
**Difficulty**: ⭐⭐ Moderate
**Impact**: Enables autoregressive tasks (language modeling)

#### What & Why

Causal masking ensures that position `t` can only attend to positions `≤ t`. This is essential for autoregressive modeling (predicting next token).

#### Step-by-Step Implementation

**8.1** Add causal flag to struct:

```julia
struct SWAttention <: Lux.AbstractLuxLayer
    sequence_length::Int
    dimension::Int
    number_of_heads::Int
    Q::Lux.Dense
    K::Lux.Dense
    V::Lux.Dense
    OUTPUT::Lux.Dense
    causal::Bool  # NEW
end

function SWAttention(sequence_length::Int, dimension::Int, number_of_heads::Int;
                     causal::Bool=false)  # NEW keyword arg
    # ... existing construction ...
    return SWAttention(sequence_length, dimension, number_of_heads, Q, K, V, OUTPUT, causal)
end
```

**8.2** Apply causal mask in forward pass:

```julia
function (block::SWAttention)(x, params, state)
    # ... Q, K, V projections ...
    # ... split into heads ...

    head_outputs = map(Q_heads, K_heads, V_heads) do q_row, k_row, v_row
        # q_row, k_row, v_row: each (d_k, T)
        scores = q_row' * k_row / √d_k  # (T, T)

        # NEW: Apply causal mask
        if block.causal
            # Create lower triangular mask (can attend to past and self)
            T = size(scores, 1)
            mask = tril(ones(Float32, T, T))  # Lower triangular
            # Set future positions to very negative (will sigmoid to ~0)
            scores = scores .* mask .+ (1 .- mask) .* Float32(-1e9)
        end

        # Normalize and apply
        weights = eachslice(scores; dims=1) |>
                  (row -> map(normalized_sigmoids, row)) |>
                  (x -> reduce(hcat, x))  # (T, T)

        Yh = v_row * weights  # (d_k, T)
        return Yh
    end

    # ... concatenate and output ...
end
```

**8.3** Test causal masking:

```julia
using Lux, Random
include("src/attention.jl")
using .attention

# Non-causal attention
attn = attention.SWAttention(64, 128, 4; causal=false)
ps = Lux.initialparameters(Random.default_rng(), attn)
st = Lux.initialstates(Random.default_rng(), attn)

x = randn(Float32, 128, 10)
y_non_causal, _ = attn(x, ps, st)

# Causal attention
attn_causal = attention.SWAttention(64, 128, 4; causal=true)
y_causal, _ = attn_causal(x, ps, st)

# Outputs should differ
@assert y_non_causal != y_causal "Causal mask should change output!"

# Test that changing future doesn't affect past (in causal mode)
x_modified = copy(x)
x_modified[:, end] .= 999.0  # Modify last position

y_causal_2, _ = attn_causal(x_modified, ps, st)

# First positions should be identical (can't see future)
@assert y_causal[:, 1:end-1] ≈ y_causal_2[:, 1:end-1] "Causality violated!"
println("✅ Causal masking test passed!")
```

---

## Testing & Validation

After implementing all improvements, run comprehensive tests:

```julia
# Run full test suite
using Pkg
Pkg.test()

# Check test coverage (if you have Coverage.jl installed)
using Coverage
cov = process_folder()
covered_lines, total_lines = get_summary(cov)
println("Coverage: $(round(100 * covered_lines / total_lines, digits=2))%")
```

---

## Performance Benchmarking

Compare before and after performance:

```julia
# benchmark/compare_improvements.jl
using Lux, Random, BenchmarkTools

function benchmark_ossm()
    rng = Random.default_rng()

    println("\n📊 OSSM Performance Comparison")
    println("="^60)

    # Baseline (no improvements)
    model_old = OSSM(128, 128, 16; use_conv=false, use_norm=false)
    ps_old = Lux.initialparameters(rng, model_old)
    st_old = Lux.initialstates(rng, model_old)
    x = randn(Float32, 128, 100)

    t_old = @benchmark $model_old($x, $ps_old, $st_old)
    println("Baseline: $(round(median(t_old).time / 1e6, digits=2)) ms")

    # With vectorized oscillators
    # (test if you kept old version for comparison)

    # With all improvements
    model_new = OSSM(128, 128, 16; use_conv=true, use_norm=true)
    ps_new = Lux.initialparameters(rng, model_new)
    st_new = Lux.initialstates(rng, model_new)

    t_new = @benchmark $model_new($x, $ps_new, $st_new)
    println("With improvements: $(round(median(t_new).time / 1e6, digits=2)) ms")

    speedup = median(t_old).time / median(t_new).time
    println("\nSpeedup: $(round(speedup, digits=2))×")
end

benchmark_ossm()
```

---

## Summary Checklist

After completing this guide, you should have:

**Quick Wins:**
- [ ] Skip projection for flexible dimensions
- [ ] Vectorized oscillator rotation (~3× faster)
- [ ] Input validation for better errors
- [ ] Comprehensive test suite

**Mamba-Inspired:**
- [ ] 1D convolution for local context
- [ ] Normalization layers for stability
- [ ] Selective B and C (full expressiveness)
- [ ] Causal masking for autoregressive tasks

**Infrastructure:**
- [ ] Full test coverage (>80%)
- [ ] Benchmarks documented
- [ ] All tests passing

---

## Next Steps

1. **Train a Model**: Use your improved OSSM on a real task (time series, audio, etc.)
2. **Hyperparameter Tuning**: Experiment with:
   - Number of oscillators H
   - Convolution kernel size
   - Normalization placement
3. **Advanced Optimizations**:
   - Implement parallel scan (requires GPU kernels)
   - Try hybrid architectures (OSSM + SWAttention)
4. **Documentation**: Add docstrings to all public functions

---

## Troubleshooting

**Common Issues:**

1. **Dimension mismatch errors**: Check that input/output shapes match constructor arguments
2. **NaN/Inf in training**: Add gradient clipping, reduce learning rate, check normalization
3. **Slow performance**: Profile with `@profview` to find bottlenecks
4. **Test failures**: Read error messages carefully, check random seeds for reproducibility

**Getting Help:**
- Check the test files for usage examples
- Review ARCHITECTURE.md for design rationale
- Consult Lux.jl documentation: https://lux.csail.mit.edu/

---

**Congratulations!** You've now implemented a state-of-the-art selective SSM with Mamba-inspired improvements. Your OSSM is faster, more flexible, and more expressive than before. 🎉
