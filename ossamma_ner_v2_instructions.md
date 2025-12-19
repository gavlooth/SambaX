# OSSAMMA-NER v2 Implementation Instructions

## Overview

You are implementing a modified OSSAMMA (Oscillatory State Space Attention Masked Mixer) architecture for Named Entity Recognition. The key modification is **dual gating** where the GLU-Global branch guides the Local-Sharp branch through two gates.

---

## Architecture Summary

```
Input → Embeddings → [OSSAMMA Block × N] → NER Head → Entity Labels
```

Each OSSAMMA Block contains:
1. Time-Conditioned LayerNorm
2. GLU-Global Branch (processes first)
3. Local-Sharp Branch (receives guidance from GLU)
4. Adaptive Mixing
5. Residual + LayerNorm

---

## Core Modification: Dual Gating

The Local branch receives two gating signals from GLU-Global:

### Input Gate (before attention)
- **Purpose**: GLU controls what features Local should attend to
- **Formula**: `gated_input = x ⊙ sigmoid(W_input_gate @ GLU_out)`
- **Effect**: Suppresses irrelevant features before Local processes them

### Output Gate (after attention)
- **Purpose**: GLU injects global context where Local needs help
- **Formula**: `Local_final = local_out + sigmoid(W_output_gate @ GLU_out) ⊙ GLU_out`
- **Effect**: Adds global information at positions where it's needed

---

## Detailed Component Specifications

### 1. Embeddings

```
Token Embedding: nn.Embedding(vocab_size, d_model)
Position Embedding: Rotary Position Embedding (RoPE)
Segment Embedding: nn.Embedding(max_segments, d_model)  # optional, for multi-doc RAG

Output: x = token_emb + segment_emb  # RoPE applied later in attention
Shape: (batch, seq_len, d_model)
```

### 2. Time-Conditioned LayerNorm

```
Input: x, time_emb
Process:
    scale = 1 + linear_scale(time_emb)  # (batch, 1, d_model)
    shift = linear_shift(time_emb)       # (batch, 1, d_model)
    x_norm = LayerNorm(x)
    output = x_norm * scale + shift
```

### 3. GLU-Global Branch

```
Input: x (after LayerNorm)

Step 1 - Expand:
    expanded = Dense(d_model → 2*d_model)(x)
    path_a, path_b = split(expanded, dim=-1)

Step 2 - Process paths:
    attn_out = LinearAttention(path_a)
    osc_out = DLinOSS(path_b)  # Oscillator/state space

Step 3 - Gate and combine:
    gated = attn_out * sigmoid(osc_out)

Step 4 - Project:
    GLU_out = Dense(d_model → d_model)(gated)

Output shape: (batch, seq_len, d_model)
```

### 4. Local-Sharp Branch (WITH DUAL GATING)

```
Input: x (after LayerNorm), GLU_out (from Global branch)

Step 1 - Input Gate:
    input_gate = sigmoid(W_input_gate @ GLU_out)  # W_input_gate: (d_model, d_model)
    gated_x = x * input_gate

Step 2 - Sliding Window Attention:
    Q = W_q(gated_x)
    K = W_k(gated_x)
    V = W_v(gated_x)
    
    # Apply RoPE to Q and K
    Q, K = apply_rotary_emb(Q, K, positions)
    
    # Sliding window attention (window_size = 256)
    local_out = sliding_window_attention(Q, K, V, window_size=256)

Step 3 - Output Gate:
    output_gate = sigmoid(W_output_gate @ GLU_out)  # W_output_gate: (d_model, d_model)
    gated_global = output_gate * GLU_out

Step 4 - Combine:
    Local_final = local_out + gated_global

Output shape: (batch, seq_len, d_model)
```

### 5. Adaptive Mixing

```
Input: GLU_out, Local_final, x (original), time_emb

Compute alpha:
    content_signal = W_alpha @ x          # (batch, seq_len, 1)
    time_bias = linear_alpha(time_emb)    # (batch, 1, 1)
    alpha = sigmoid(content_signal + time_bias)

Mix:
    mixed = alpha * GLU_out + (1 - alpha) * Local_final

Output shape: (batch, seq_len, d_model)
```

### 6. Block Output

```
Input: mixed, x_residual (input to block)

output = LayerNorm(mixed + x_residual)
```

---

## NER Head Specification

```
Input: final hidden states (batch, seq_len, d_model)

Step 1 - Span Boundary Pooling:
    # For each position, concatenate with neighboring representations
    boundary_features = concat([
        hidden,
        hidden_shifted_left,
        hidden_shifted_right,
        hidden * hidden_shifted_left,   # interaction features
    ], dim=-1)
    pooled = Dense(4*d_model → d_model)(boundary_features)

Step 2 - Classification:
    logits = Dense(d_model → num_labels)(pooled)
    # num_labels = 19 (9 entity types × 2 for B/I + 1 for O)

Step 3 - CRF Layer:
    # Apply CRF for valid BIO sequence constraints
    # Transition matrix: (num_labels, num_labels)
    # Invalid transitions (e.g., O → I-PERSON) get -inf score
    predictions = crf.decode(logits)
    loss = -crf.log_likelihood(logits, targets)

Auxiliary Task - Boundary Detection:
    boundary_logits = Dense(d_model → 2)(pooled)  # binary: is_boundary or not
    boundary_loss = cross_entropy(boundary_logits, boundary_targets)

Total Loss:
    loss = crf_loss + 0.2 * boundary_loss
```

---

## Entity Labels

```python
labels = [
    "O",           # Outside
    "B-PERSON",    # Begin Person
    "I-PERSON",    # Inside Person
    "B-AGENCY",    # Begin Agency (companies, governments, institutions)
    "I-AGENCY",
    "B-PLACE",     # Begin Place (geographic, addresses)
    "I-PLACE",
    "B-ORGANISM",  # Begin Organism (animals, plants, microbes)
    "I-ORGANISM",
    "B-EVENT",     # Begin Event (wars, incidents, eras)
    "I-EVENT",
    "B-INSTRUMENT",# Begin Instrument (tools, products, devices)
    "I-INSTRUMENT",
    "B-WORK",      # Begin Work (books, papers, films, datasets)
    "I-WORK",
    "B-DOMAIN",    # Begin Domain (sciences, methods, fields)
    "I-DOMAIN",
    "B-MEASURE",   # Begin Measure (numbers, dates, money)
    "I-MEASURE",
]
```

---

## Hyperparameters

### Model Sizes

| Size | Layers | d_model | Heads | Head Dim | FFN Mult | Params |
|------|--------|---------|-------|----------|----------|--------|
| Small | 6 | 384 | 6 | 64 | 2 | ~30M |
| Base | 12 | 768 | 12 | 64 | 2 | ~110M |
| Large | 24 | 1024 | 16 | 64 | 2 | ~350M |

### Attention

```
window_size: 256
num_heads: d_model // 64
head_dim: 64
dropout: 0.1
attention_dropout: 0.1
```

### Oscillator (DLinOSS)

```
oscillator_dim: 64
num_oscillators: 8
damping: 0.1
```

### Training

```
# Pretraining (Masked Diffusion)
mask_ratio: 0.15
num_timesteps: 1000
noise_schedule: "cosine"
learning_rate: 1e-4
warmup_steps: 2000
batch_size: 32
gradient_accumulation: 4

# Fine-tuning (NER)
learning_rate: 2e-5
epochs: 20
label_smoothing: 0.1
crf_learning_rate: 1e-3  # CRF often needs higher LR
```

---

## Implementation Checklist

### Core Components
- [ ] Rotary Position Embedding (RoPE)
- [ ] Time-Conditioned LayerNorm
- [ ] Linear Attention (for GLU branch)
- [ ] DLinOSS Oscillator
- [ ] Sliding Window Attention
- [ ] **Input Gate: `sigmoid(W₁ @ GLU_out)`**
- [ ] **Output Gate: `sigmoid(W₂ @ GLU_out) * GLU_out`**
- [ ] Adaptive alpha mixing
- [ ] CRF layer for BIO decoding

### Dual Gating (THE KEY MODIFICATION)
- [ ] `W_input_gate`: Linear(d_model, d_model) - no bias recommended
- [ ] `W_output_gate`: Linear(d_model, d_model) - no bias recommended
- [ ] Input gate applied BEFORE sliding window attention
- [ ] Output gate applied AFTER sliding window attention
- [ ] Output gate multiplies GLU_out, then ADDS to local_out

### Forward Pass Order
1. Time-Conditioned LayerNorm on input
2. GLU-Global branch processes → produces GLU_out
3. Compute input_gate from GLU_out
4. Apply input_gate to x
5. Local branch attention on gated input
6. Compute output_gate from GLU_out
7. Add gated GLU_out to local output
8. Adaptive mixing of GLU_out and Local_final
9. Add residual, final LayerNorm

---

## Critical Implementation Notes

### Dual Gating Flow
```python
# CORRECT order - GLU must complete before Local starts
GLU_out = glu_global_branch(x_norm)

# Input gate
input_gate = torch.sigmoid(self.W_input_gate(GLU_out))
gated_x = x_norm * input_gate

# Local attention on gated input
local_out = self.sliding_window_attention(gated_x)

# Output gate
output_gate = torch.sigmoid(self.W_output_gate(GLU_out))
Local_final = local_out + output_gate * GLU_out
```

### Why This Design
- **Input gate**: GLU knows global context, tells Local "ignore these features, they're not relevant here"
- **Output gate**: GLU says "Local might be uncertain here, let me inject my knowledge"
- **Both are position-wise**: Different positions get different gating
- **GLU_out used twice**: As the signal to compute gates AND as the value added via output gate

### Initialization
- Initialize gate projection weights with small values (std=0.02)
- This starts gates near 0.5 (sigmoid(0) = 0.5), allowing gradual learning
- Alternatively, initialize bias to small negative value so gates start more "closed"

---

## Testing Your Implementation

### Sanity Checks
1. Without gates (set to all 1s), should behave like original OSSAMMA
2. Input gate all 0s → Local sees nothing → output should rely on output gate
3. Output gate all 0s → No GLU injection → pure local output
4. Gradients should flow through both gate paths

### Shape Checks
```python
# All should be (batch, seq_len, d_model)
assert GLU_out.shape == x.shape
assert input_gate.shape == x.shape
assert gated_x.shape == x.shape
assert local_out.shape == x.shape
assert output_gate.shape == x.shape
assert Local_final.shape == x.shape
```
