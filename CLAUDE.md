# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## DIRECTIVE: Autonomous GPU Training Mode

**PRIMARY OBJECTIVE**: Train OssammaNER model autonomously on GPU (RTX 5090 32GB).

### Training Commands
```bash
# Start training (background)
nohup julia --project=. scripts/train_ner_production.jl --synthetic > training.log 2>&1 &

# Monitor progress
tail -f training.log | grep -E "(step:|loss:|grad_norm:|Step|ETA)"

# GPU status
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
```

### Autonomous Behavior
1. **DO NOT INTERRUPT** training for questions - training takes priority
2. Monitor progress every 60s: step, loss, grad_norm, GPU%, memory, ETA
3. Checkpoints save automatically every 1000 steps to `checkpoints/ner_110m/`
4. If training crashes: diagnose error, fix, and restart immediately
5. Report progress periodically without stopping training

### Current Training Target
- **Model**: OssammaNER (~15M parameters)
- **Task**: Named Entity Recognition (19 labels)
- **Steps**: 50,000 total
- **Expected Loss**: Start ~2.94, converge to <1.0
- **GPU Utilization**: 30-35% (limited by sequential oscillator architecture)
- **Step Time**: ~7-10 seconds
- **ETA**: ~4-6 days

### Progress Indicators
| Metric | Good | Warning | Bad |
|--------|------|---------|-----|
| Loss | <2.5 | 2.5-2.9 | >2.9 (not learning) |
| Grad Norm | 1-100 | 100-500 | >1000 (unstable) |
| GPU Util | >30% | 20-30% | <20% (bottleneck) |

---

## Project Overview

Samba2 is a Julia package implementing custom neural network layers using the Lux.jl framework. The project focuses on two novel architectures:

1. **SWAttention** (Sliding Window Attention): A multi-head attention mechanism that uses normalized sigmoids instead of softmax for attention weights
2. **OSSM** (Oscillatory State Space Model): A state space model based on learnable damped oscillators with selective step sizing

## Development Commands

### Julia REPL
Start Julia REPL in the project directory:
```bash
julia --project=.
```

### Package Development
Inside the Julia REPL:
```julia
using Revise                    # Auto-reload code changes
using Samba2
using Lux, Random

# Example: Create and test SWAttention
include("src/attention.jl")
using .attention
attn = attention.SWAttention(1024, 128, 4)
rng = Random.default_rng()
ps = Lux.initialparameters(rng, attn)
st = Lux.initialstates(rng, attn)

# Example: Create and test OSSM
include("src/ossm.jl")
using .ossm
model = ossm.OSSM(128, 128, 8)  # dim_in, dim_out, num_oscillators
ps = Lux.initialparameters(rng, model)
st = Lux.initialstates(rng, model)
```

## Architecture

### Module Structure
Both components are implemented as standalone modules that can be included individually:
- `src/attention.jl` - Sliding window attention module
- `src/ossm.jl` - Oscillatory state space model module

### Lux Integration
All layers follow Lux.jl conventions:
- Extend `Lux.AbstractLuxLayer`
- Implement `Lux.initialparameters(rng, block)` for parameter initialization
- Implement `Lux.initialstates(rng, block)` for state initialization
- Implement call operator `(block::Layer)(x, params, state)` returning `(output, new_state)`

### SWAttention Design
- Uses normalized sigmoids with temperature scaling instead of softmax
- Multi-head attention with dimension split across heads
- Requires `dimension % number_of_heads == 0`
- Input/output dimensions are always equal to `dimension`
- State convention: Returns empty NamedTuple `(;)` as this layer is stateless

### OSSM Design
- State space model with `H` independent damped oscillators
- State dimension is `2H` (two coordinates per oscillator)
- **Selective step sizing**: Learns per-oscillator step size `Δt = softplus(WΔ * u + bΔ)`
- **Damped rotation**: Each oscillator applies `ρ * R(θ)` where:
  - `ρ = exp(-α * Δt)` with `α > 0` (forced via softplus)
  - `θ = ω * Δt` for learned frequency `ω`
- **Gating mechanisms**:
  - Input gate: `u_gated = sigmoid(input_gate(u)) .* u`
  - Mixture gate: `out = g_mix .* Y + (1 - g_mix) .* u` (requires `dim_in == dim_out`)
- State convention: Maintains `oscillation_state` as `(2H, 1)` column vector
- Processes sequences via `foldl` scan over time dimension

### Key Implementation Details

**Attention normalized_sigmoids**:
- Temperature parameter `τ = 1.0` controls sharpness
- Epsilon `1e-12` prevents division by zero
- In-place normalization via `map!`

**OSSM oscillator_step**:
- Input: `xt :: (2H, 1)`, `ut :: (dim_in, 1)`
- Reshapes state to `(2, H)` for per-oscillator operations
- Applies damped rotation per oscillator then flattens back
- Output computed from current state: `y = C * xt + D * ut`

**State management**:
- SWAttention is stateless (empty state)
- OSSM is stateful (carries oscillation_state and gate states)
- Always return `(output, new_state)` tuple from forward pass

## Dependencies
- **Lux.jl** (v1.26.0): Main neural network framework
- **LuxCore.jl** (v1.4.2): Core abstractions
- **NNlib.jl**: Neural network primitives (sigmoid, softplus)
- **MLStyle.jl** (v0.4.17): Pattern matching utilities
- **Revise.jl** (v3.12.1): Auto-reload code during development
