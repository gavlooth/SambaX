# Depth Scaling Strategy: Exploiting O(T) Complexity

## The Core Insight

For a fixed compute budget `C`, compare layer allocation:

```
Transformer:  C = L_t × T² × d    →  L_t = C / (T² × d)
Ossamma:      C = L_o × T × d²    →  L_o = C / (T × d²)

Ratio:  L_o / L_t = T / d
```

**For T=2048, d=512:**  Ossamma can afford **4× more layers** at the same compute cost.

**For T=4096, d=512:**  Ossamma can afford **8× more layers**.

This is the key advantage to exploit.

---

## Strategy 1: Deep & Narrow Architecture

### Principle

Trade width (embedding dimension) for depth (layers). Ossamma's linear complexity makes this viable.

### Comparison: Equivalent Compute Models

| Model | Layers | Dim | Heads | Params | FLOPs/token |
|-------|--------|-----|-------|--------|-------------|
| **Transformer-Base** | 12 | 768 | 12 | 110M | O(T² × 768) |
| **Ossamma-Deep** | 48 | 384 | 6 | ~110M | O(T × 384²) |
| **Ossamma-VeryDeep** | 96 | 256 | 4 | ~100M | O(T × 256²) |

### Why Depth > Width for Ossamma

1. **O(T) makes depth cheap**: Each layer costs O(T×d²), not O(T²×d)
2. **Oscillator memory compounds**: DLinOSS state carries through layers - more layers = richer state evolution
3. **Gradient flow with residuals**: Post-norm + residual scaling handles deep networks
4. **Hierarchical abstraction**: More layers = more levels of representation

### Proposed Configuration: `ossamma_deep_48L`

```toml
[model]
name = "ossamma_deep_48L"
vocab_size = 32000
max_sequence_length = 4096
embedding_dimension = 384      # Narrower than typical
number_of_heads = 6
number_of_layers = 48          # 4x deeper than typical 12L
time_dimension = 64

[oscillator]
state_dimension = 384
use_parallel_scan = true       # CRITICAL: O(log T) depth
min_frequency = 0.01
max_frequency = 100.0          # Wide frequency range for 48 layers

[training]
dropout_rate = 0.1
layer_scale_init = 0.1         # For deep network stability
use_stochastic_depth = true    # Drop layers randomly during training
stochastic_depth_rate = 0.1
```

---

## Strategy 2: Parallel Scan is Mandatory

### Current Bottleneck

Without parallel scan, OSSM is O(T) sequential steps - **kills GPU utilization**.

```
Sequential OSSM:  GPU util = 30-35% (from CLAUDE.md)
Parallel Scan:    GPU util = 80-90% (theoretical)
```

### Implementation Requirement

```julia
# In DlinossParallel.jl - already implemented!
# The key is the associative combination:
#   (A, b) ⊕ (A', b') = (A'·A, A'·b + b')

# Enable via:
oscillator_layer = DLinOSSParallel(
    dim, state_dim, dim,
    min_freq, max_freq, default_dt;
    chunk_size = 64  # Tune for GPU
)
```

### Parallel Scan Speedup

| Sequence Length | Sequential Steps | Parallel Depth | Speedup |
|-----------------|------------------|----------------|---------|
| 512 | 512 | 9 | 57× |
| 2048 | 2048 | 11 | 186× |
| 8192 | 8192 | 13 | 630× |

**This is where O(T) complexity becomes real.**

---

## Strategy 3: Hierarchical Oscillator Frequencies

### Problem

With 48 layers, a single frequency band is wasteful. Different layers should capture different timescales.

### Solution: Layer-Dependent Frequency Ranges

```julia
function layer_frequency_range(layer_idx::Int, total_layers::Int)
    # Early layers: high frequency (local patterns, syntax)
    # Middle layers: mid frequency (phrases, clauses)
    # Late layers: low frequency (document-level, semantics)

    progress = layer_idx / total_layers  # 0.0 → 1.0

    # Exponential frequency decay
    max_freq = 100.0 * exp(-3.0 * progress)  # 100 → 5
    min_freq = 0.1 * exp(-2.0 * progress)    # 0.1 → 0.01

    return (min_freq, max_freq)
end

# Example for 48 layers:
# Layer 1:  freq ∈ [0.10, 100.0]  - high frequency, local
# Layer 12: freq ∈ [0.05, 47.0]   - mid-high
# Layer 24: freq ∈ [0.02, 22.0]   - mid
# Layer 36: freq ∈ [0.01, 10.0]   - mid-low
# Layer 48: freq ∈ [0.01, 5.0]    - low frequency, global
```

### Intuition

```
Early layers (high freq):
  "The [adjective] [noun] [verb]..."
   ↑ Fast oscillations track local syntax

Late layers (low freq):
  "[Topic established] ... [reference back]"
   ↑ Slow oscillations maintain document state
```

---

## Strategy 4: Deep Residual Scaling

### Problem

48+ layers need careful initialization to avoid gradient vanishing/exploding.

### Solution: Layer Scale (from CaiT, DeepViT)

```julia
struct OssammaBlockDeep <: LuxCore.AbstractLuxLayer
    # ... existing fields ...
    layer_scale_init::Float32  # e.g., 0.1 or 1e-4
end

function (block::OssammaBlockDeep)(inputs, params, state)
    # ... compute output ...

    # Scale residual contribution
    # For layer i: scale = layer_scale_init * (i / L)^0.5
    scaled_output = output .* params.layer_scale

    return residual .+ scaled_output, new_state
end

function Lux.initialparameters(rng, block::OssammaBlockDeep)
    # Initialize layer_scale small for deep networks
    layer_scale = fill(block.layer_scale_init, block.embedding_dimension)
    # ... rest of params ...
end
```

### Stochastic Depth (optional but recommended)

```julia
function (block::OssammaBlockDeep)(inputs, params, state; training::Bool=true)
    if training && rand() < block.drop_rate
        # Skip this layer entirely during training
        return inputs, state
    end

    # Normal forward pass...
end
```

---

## Strategy 5: Diffusion for Parallel Generation

### Why Diffusion + Deep Ossamma

1. **No autoregressive bottleneck**: Generate all T tokens in parallel
2. **Iterative refinement**: K steps (4-10) vs T steps (512-4096)
3. **Deep model amortization**: Each refinement step uses full 48L model

### Generation Comparison

| Method | Steps | Tokens/Step | Total Compute |
|--------|-------|-------------|---------------|
| AR (GPT-style) | T | 1 | T × L × O(T²×d) |
| Diffusion + Transformer | K | T | K × L × O(T²×d) |
| **Diffusion + Ossamma** | K | T | **K × L × O(T×d²)** |

For K=8, T=2048, L=48:
- AR Transformer: 2048 × 12 × O(T²) = very slow
- Diffusion + Ossamma-48L: 8 × 48 × O(T) = **much faster**

### OssammaDrafter Configuration

```toml
[model]
type = "OssammaDrafter"
vocab_size = 100352          # Granite 4.0
max_sequence_length = 4096
embedding_dimension = 384
number_of_heads = 6
number_of_layers = 48        # Deep!
time_dimension = 64

[diffusion]
num_refinement_steps = 8
confidence_threshold = 0.85
temperature = 0.9

[oscillator]
use_parallel_scan = true     # MANDATORY for speed
```

---

## Strategy 6: Efficient Attention in Deep Networks

### Problem

Even with O(T×d²), 48 layers of full attention is expensive.

### Solution: Attention Budget Allocation

```
Layer Type Distribution (48 layers):

Layers 1-12:   SWAttention only (window=64)   - local syntax
Layers 13-24:  SWAttention (window=128)       - clause-level
Layers 25-36:  LinearAttention + DLinOSS      - global semantics
Layers 37-48:  Full OssammaBlock              - final integration
```

### Implementation

```julia
function create_deep_ossamma(;
    num_layers::Int = 48,
    embedding_dim::Int = 384,
    # ...
)
    blocks = []

    for i in 1:num_layers
        if i <= num_layers ÷ 4
            # Early: Local only
            push!(blocks, SWAttentionBlock(embedding_dim, window=64))
        elseif i <= num_layers ÷ 2
            # Early-mid: Local with larger window
            push!(blocks, SWAttentionBlock(embedding_dim, window=128))
        elseif i <= 3 * num_layers ÷ 4
            # Late-mid: Global linear only (OssammaDrafterBlock style)
            push!(blocks, OssammaDrafterBlock(embedding_dim, ...))
        else
            # Late: Full OssammaBlock
            push!(blocks, OssammaBlock(embedding_dim, ...))
        end
    end

    return blocks
end
```

---

## Strategy 7: Memory-Efficient Deep Training

### Gradient Checkpointing

For 48 layers, activation memory is huge. Checkpoint every N layers:

```julia
function forward_with_checkpointing(blocks, x, params, states; checkpoint_every=4)
    for (i, block) in enumerate(blocks)
        if i % checkpoint_every == 0
            # Checkpoint: don't store activations, recompute in backward
            x, states[i] = Zygote.checkpointed(block, x, params[i], states[i])
        else
            x, states[i] = block(x, params[i], states[i])
        end
    end
    return x, states
end
```

### Memory Budget

| Layers | Without Checkpoint | With Checkpoint (every 4) |
|--------|-------------------|---------------------------|
| 12 | 12 × activations | 12 × activations |
| 48 | 48 × activations | 12 × activations + recompute |
| 96 | OOM | 24 × activations + recompute |

---

## Concrete Model Configurations

### Config 1: `ossamma_deep_48L` (Recommended Start)

```toml
# ~120M parameters, comparable to BERT-base
[model]
embedding_dimension = 384
number_of_layers = 48
number_of_heads = 6
max_sequence_length = 4096

[oscillator]
state_dimension = 384
use_parallel_scan = true
frequency_scaling = "hierarchical"  # Layer-dependent

[training]
batch_size = 16
gradient_accumulation = 4
checkpoint_every = 4
layer_scale_init = 0.1
```

### Config 2: `ossamma_ultra_96L` (Research)

```toml
# ~100M parameters, extremely deep
[model]
embedding_dimension = 256
number_of_layers = 96
number_of_heads = 4
max_sequence_length = 8192

[oscillator]
state_dimension = 256
use_parallel_scan = true
frequency_scaling = "hierarchical"

[attention]
# Mixed attention types by layer
early_layers_attention = "sliding_window_64"
mid_layers_attention = "linear_only"
late_layers_attention = "full_ossamma"

[training]
batch_size = 8
gradient_accumulation = 8
checkpoint_every = 6
use_stochastic_depth = true
stochastic_depth_rate = 0.2
```

### Config 3: `ossamma_long_context` (Long Sequences)

```toml
# Optimized for very long sequences (16K+)
[model]
embedding_dimension = 512
number_of_layers = 32
number_of_heads = 8
max_sequence_length = 16384

[oscillator]
state_dimension = 512
use_parallel_scan = true  # CRITICAL for 16K
frequency_scaling = "logarithmic"
min_frequency = 0.001  # Very low for long-range
max_frequency = 50.0

[training]
batch_size = 4
gradient_accumulation = 16
checkpoint_every = 4
```

---

## Implementation Roadmap

### Phase 1: Enable Parallel Scan (Week 1)
- [ ] Verify DLinOSSParallel works correctly
- [ ] Benchmark parallel vs sequential scan
- [ ] Integrate into OssammaBlock with `use_parallel_scan` flag
- [ ] Target: 10-40× speedup on GPU

### Phase 2: Deep Architecture Support (Week 2)
- [ ] Add layer scale initialization
- [ ] Add stochastic depth
- [ ] Implement gradient checkpointing
- [ ] Test 24L → 48L → 96L scaling

### Phase 3: Hierarchical Frequencies (Week 2-3)
- [ ] Implement `layer_frequency_range()`
- [ ] Per-layer oscillator configuration
- [ ] Ablation: uniform vs hierarchical frequencies

### Phase 4: Mixed Attention Layers (Week 3)
- [ ] Create layer type selector
- [ ] Benchmark mixed vs uniform architectures
- [ ] Find optimal layer type distribution

### Phase 5: Full Training Run (Week 4+)
- [ ] Train `ossamma_deep_48L` on target dataset
- [ ] Compare to 12L baseline (same params, different depth/width)
- [ ] Evaluate on downstream tasks

---

## Expected Outcomes

### Speed Improvement

With parallel scan + diffusion:
```
Current (sequential OSSM, AR generation):
  7-10 sec/step, 30-35% GPU util

Target (parallel scan + diffusion):
  0.5-1 sec/step, 80-90% GPU util

Speedup: 7-20×
```

### Quality Improvement (Hypothesis)

Deeper networks should show:
1. **Better long-range coherence** (oscillator state compounds over layers)
2. **Richer representations** (more abstraction levels)
3. **Better sample efficiency** (depth > width for language)

### Compute Efficiency

For same FLOP budget:
```
Transformer-12L-768d:  L=12,  d=768,  T²=T² bottleneck
Ossamma-48L-384d:      L=48,  d=384,  T bottleneck

At T=4096:
  Transformer: ~67B FLOPs/sequence
  Ossamma:     ~8B FLOPs/sequence (8× cheaper)

Can afford 48 layers for price of 6 transformer layers.
```

---

## Summary

| Strategy | Key Insight | Implementation |
|----------|-------------|----------------|
| **Deep & Narrow** | O(T) allows 4-8× more layers | 48L/384d instead of 12L/768d |
| **Parallel Scan** | O(log T) depth, not O(T) | Use DLinOSSParallel |
| **Hierarchical Freq** | Different layers, different timescales | `layer_frequency_range()` |
| **Deep Residuals** | Stabilize 48+ layers | Layer scale + stochastic depth |
| **Diffusion** | Parallel generation | OssammaDrafter with K steps |
| **Mixed Attention** | Allocate attention budget wisely | Early=local, late=global |
| **Checkpointing** | Fit 48-96L in memory | Checkpoint every 4-6 layers |

**The fundamental insight**: Ossamma's O(T) complexity is only useful if you **spend the saved compute on depth**. Going deeper with parallel scan + diffusion is the winning strategy.
