# Parallelization Strategy for Ossamma

This document outlines strategies for parallelizing training and inference in the Ossamma architecture.

## Current Bottlenecks

### 1. DLinOSS Sequential Scan
The oscillatory state space model processes timesteps sequentially:
```julia
state = foldl(oscillator_step, timesteps, init=initial_state)
```
This is O(T) and cannot utilize GPU parallelism effectively.

### 2. Diffusion Inference Steps
LLaDA inference requires K denoising steps (typically K=50-1000), each running the full model sequentially.

### 3. Single GPU Bound
Training currently runs on a single GPU with batch parallelism only.

---

## Strategy 1: Parallel Associative Scan for DLinOSS

### Concept
The linear state update `x_{t+1} = A·x_t + B·u_t` can be reformulated as an associative operation, enabling parallel prefix-sum computation.

### Mathematical Formulation
Define tuples `(A_t, b_t)` where the combination operator is:
```
(A, b) ⊕ (A', b') = (A'·A, A'·b + b')
```

This is associative: `((A,b) ⊕ (A',b')) ⊕ (A'',b'') = (A,b) ⊕ ((A',b') ⊕ (A'',b''))`

### Parallel Scan Tree
```
Level 0:  (A₁,b₁)   (A₂,b₂)   (A₃,b₃)   (A₄,b₄)   (A₅,b₅)   (A₆,b₆)   (A₇,b₇)   (A₈,b₈)
              \       /           \       /           \       /           \       /
Level 1:      (A₁:₂,b₁:₂)       (A₃:₄,b₃:₄)       (A₅:₆,b₅:₆)       (A₇:₈,b₇:₈)
                    \               /                     \               /
Level 2:            (A₁:₄,b₁:₄)                         (A₅:₈,b₅:₈)
                            \                               /
Level 3:                            (A₁:₈,b₁:₈)

Complexity: O(log T) parallel steps instead of O(T) sequential
```

### Advantages
- **Speedup**: 10-40× for long sequences (T > 256)
- **GPU utilization**: Fully utilizes parallel cores
- **Scalable**: Benefits increase with sequence length

### Pitfalls
- **Memory overhead**: Must store intermediate results at each level (2× memory)
- **Numerical stability**: Repeated matrix multiplications can accumulate errors
- **Implementation complexity**: Requires custom CUDA kernel or careful use of `CUDA.jl`
- **Small sequence overhead**: For T < 64, sequential may be faster due to kernel launch overhead
- **Non-linear extensions**: If state update becomes non-linear (e.g., gating), associativity breaks

### Implementation Notes
```julia
# Pseudo-code for parallel scan
function parallel_ssm_scan(A_seq, b_seq)
    # A_seq: (2H, 2H, T) - per-timestep transition matrices
    # b_seq: (2H, T) - per-timestep inputs

    # Up-sweep (reduce)
    for level in 1:log2(T)
        parallel_for i in 1:T÷(2^level)
            left = 2^level * i - 2^(level-1)
            right = 2^level * i
            A[right] = A[right] * A[left]
            b[right] = A[right] * b[left] + b[right]
        end
    end

    # Down-sweep (distribute)
    # ... (Blelloch scan pattern)
end
```

---

## Strategy 2: Convolution Approximation

### Concept
For linear time-invariant (LTI) systems, the SSM response equals convolution with an impulse response kernel.

### Mathematical Formulation
```
y_t = Σ_{k=0}^{t} C·A^k·B·u_{t-k} = (kernel * input)_t

where kernel_k = C·A^k·B  (precomputed)
```

### Advantages
- **Simplicity**: Use existing optimized `conv1d` operations
- **Speed**: cuDNN convolutions are highly optimized
- **No custom kernels**: Works with standard deep learning frameworks

### Pitfalls
- **Fixed sequence length**: Kernel size must match or exceed sequence length
- **Memory**: Full kernel storage is O(T × state_dim²)
- **Time-varying systems**: Doesn't work if A, B change per-timestep (selective scan)
- **Truncation error**: Must truncate kernel at some length, losing long-range info
- **Not applicable to DLinOSS**: Our A matrix is input-dependent (selective Δt), breaking LTI assumption

### When to Use
- Short sequences (T < 512)
- Non-selective SSM variants
- Quick prototyping before implementing parallel scan

---

## Strategy 3: Chunked Parallel Processing

### Concept
Divide sequence into chunks, process chunks in parallel, propagate states between chunks.

### Architecture
```
Sequence:  [----Chunk 1----][----Chunk 2----][----Chunk 3----][----Chunk 4----]
                  ↓                 ↓                 ↓                 ↓
Process:      parallel          parallel          parallel          parallel
                  ↓                 ↓                 ↓                 ↓
States:        s₁ ──────────────► s₂ ──────────────► s₃ ──────────────► s₄
                    (sequential state propagation)
```

### Two-Pass Algorithm
```
Pass 1 (parallel): Process each chunk assuming zero initial state
Pass 2 (sequential): Propagate final states and correct chunk outputs
```

### Advantages
- **Moderate speedup**: num_chunks × faster (typically 4-8×)
- **Memory efficient**: Only process chunk_size tokens at once
- **Simple implementation**: No custom CUDA kernels needed
- **Works with selective scan**: Unlike convolution

### Pitfalls
- **State propagation overhead**: Sequential pass between chunks limits speedup
- **Chunk boundary artifacts**: Information flow interrupted at boundaries
- **Suboptimal for short sequences**: Overhead > benefit when T < 128
- **Hyperparameter tuning**: Optimal chunk_size depends on hardware

### Implementation
```julia
function chunked_ssm(model, x, ps, st; chunk_size=64)
    T = size(x, 2)
    n_chunks = ceil(Int, T / chunk_size)

    # Pass 1: Process chunks in parallel (zero initial state)
    chunk_outputs = Vector{Any}(undef, n_chunks)
    chunk_final_states = Vector{Any}(undef, n_chunks)

    @sync for i in 1:n_chunks
        @async begin
            chunk = get_chunk(x, i, chunk_size)
            out, new_st = model(chunk, ps, st)
            chunk_outputs[i] = out
            chunk_final_states[i] = new_st
        end
    end

    # Pass 2: Correct for state propagation
    corrected = [chunk_outputs[1]]
    running_state = chunk_final_states[1]

    for i in 2:n_chunks
        correction = compute_state_correction(running_state, chunk_outputs[i])
        push!(corrected, chunk_outputs[i] + correction)
        running_state = propagate_state(running_state, chunk_final_states[i])
    end

    return cat(corrected..., dims=2)
end
```

---

## Strategy 4: Multi-GPU Data Parallelism

### Concept
Distribute batches across multiple GPUs, synchronize gradients.

### Architecture
```
              ┌─────────────┐
              │   Batch     │
              │  (B samples)│
              └──────┬──────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
    ┌───────┐    ┌───────┐    ┌───────┐
    │ GPU 0 │    │ GPU 1 │    │ GPU 2 │
    │ B/3   │    │ B/3   │    │ B/3   │
    └───┬───┘    └───┬───┘    └───┬───┘
        │            │            │
        └────────────┼────────────┘
                     ▼
              ┌─────────────┐
              │  AllReduce  │
              │  Gradients  │
              └─────────────┘
```

### Advantages
- **Linear scaling**: N GPUs ≈ N× throughput (with good interconnect)
- **Larger effective batch**: Better gradient estimates
- **Standard technique**: Well-supported in frameworks

### Pitfalls
- **Communication overhead**: Gradient sync can bottleneck (use NCCL)
- **Memory duplication**: Each GPU holds full model copy
- **Diminishing returns**: Communication costs dominate at high GPU counts
- **Batch size constraints**: May need to adjust learning rate for larger batches
- **Hardware requirement**: Needs multiple GPUs with fast interconnect

### Implementation with Lux
```julia
using Lux.DistributedUtils

backend = NCCLBackend()
ps_distributed = distribute(ps, backend)

for batch in dataloader
    local_batch = scatter(batch, backend)
    loss, grads = compute_gradients(model, local_batch, ps_distributed)

    # AllReduce gradients
    grads_synced = allreduce(grads, backend, +) / world_size()

    ps_distributed = update(optimizer, ps_distributed, grads_synced)
end
```

---

## Strategy 5: Diffusion Inference Parallelism

### Current Inference Flow
```
x_T (fully masked) → denoise → x_{T-1} → denoise → ... → x_0 (output)
                     step 1              step 2           step K

K steps executed sequentially
```

### 5a. Batch Parallel Inference

**Concept**: Process multiple sequences through all K steps together.

```
Batch of N sequences:
[seq_1, seq_2, ..., seq_N] → K steps → [out_1, out_2, ..., out_N]
                            (batched)
```

**Advantages**:
- Simple, no algorithm changes
- Good GPU utilization

**Pitfalls**:
- Doesn't reduce latency for single sequence
- Memory scales with batch size

### 5b. Fewer Denoising Steps

**Concept**: Reduce K through better noise schedules or training.

| Method | Steps | Quality | Training Cost |
|--------|-------|---------|---------------|
| Standard DDPM | 1000 | Best | Baseline |
| DDIM | 50-100 | Good | None (inference change) |
| Distillation | 4-8 | Good | High (student training) |
| Consistency | 1-2 | Moderate | High (new objective) |

**Advantages**:
- Directly reduces inference time
- DDIM requires no retraining

**Pitfalls**:
- Quality degradation with fewer steps
- Distillation requires training separate model
- Consistency models need architecture changes

### 5c. Speculative Parallel Decoding

**Concept**: Run multiple denoising trajectories in parallel, select best.

```
         ┌─► trajectory A ─► score_A
x_T ─────┼─► trajectory B ─► score_B  ─► select best
         └─► trajectory C ─► score_C
```

**Advantages**:
- Can improve quality with same compute
- Embarrassingly parallel

**Pitfalls**:
- Doesn't reduce compute, just latency
- Need good scoring function
- Memory multiplied by number of trajectories

### 5d. Parallel Token Unmasking

**Concept**: In mask-predict models, unmask multiple tokens per step based on confidence.

```
Step t:   "The [M] [M] on the [M]"
          confidence: [0.9, 0.3, 0.8]

Step t+1: "The cat [M] on the mat"  (unmasked 2 tokens)
```

**Advantages**:
- Already implemented in LLaDA (top-k unmasking)
- Adaptive compute per sequence

**Pitfalls**:
- Can unmask wrong tokens early (error propagation)
- Confidence calibration matters
- Diminishing returns with aggressive unmasking

---

## Comparison Summary

| Strategy | Speedup | Complexity | Memory | Applicability |
|----------|---------|------------|--------|---------------|
| Parallel Scan | 10-40× | High | 2× | DLinOSS |
| Convolution | 5-10× | Low | High | Non-selective SSM only |
| Chunked | 2-8× | Medium | 1× | All SSM variants |
| Multi-GPU | N× | Medium | N× | Training |
| Batch Inference | N× | Low | N× | Inference throughput |
| Fewer Steps | K/k× | Varies | 1× | Diffusion inference |

## Recommended Priority

1. **Chunked Processing** - Quick win, moderate speedup, low risk
2. **Multi-GPU** - If hardware available, standard technique
3. **DDIM Sampling** - Free speedup for inference, no retraining
4. **Parallel Scan** - Best long-term, but significant engineering effort

## Architecture Considerations

### Why Two Arrows from GLU to Local Branch?

The OssammaNERBlock uses **dual gating**:

```
              GLU Output
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
   Input Gate          Output Gate
        │                   │
        ▼                   ▼
   x * σ(W₁·glu)    local + σ(W₂·glu)·glu
        │                   │
        ▼                   │
   SWAttention              │
        │                   │
        └───────────────────┘
                │
                ▼
          Final Output
```

**Purpose**:
1. **Input Gate**: GLU context decides what Local attention should focus on
2. **Output Gate**: Injects global information back into the local output

This creates an information flow where global understanding (from GLU/LinearAttention) guides local precision (from SWAttention).
