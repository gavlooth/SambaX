# Ossamma

**Oscillatory State Space Attention Masked Mixer Architecture** - A Julia-based neural network framework implementing novel state space models and attention mechanisms for Named Entity Recognition.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OssammaNER Architecture                             │
└─────────────────────────────────────────────────────────────────────────────┘

  Input: "Barack Obama visited Paris"
                    │
                    ▼
    ┌───────────────────────────────┐
    │     Token Embeddings          │  vocab_size → embedding_dim
    │     + Position Embeddings     │  seq_len → embedding_dim
    │     + Time Embedding          │  (fixed t=0.5 for NER)
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                  OssammaNERBlock (×N layers)                  │
    │  ┌─────────────────────────────────────────────────────────┐  │
    │  │            Time-Conditioned LayerNorm                   │  │
    │  │         (scale/shift modulated by timestep)             │  │
    │  └────────────────────┬────────────────────────────────────┘  │
    │                       │                                       │
    │          ┌────────────┴────────────┐                          │
    │          │                         │                          │
    │          ▼                         ▼                          │
    │  ┌───────────────┐        ┌────────────────┐                  │
    │  │  GLU Branch   │        │  Local Branch  │                  │
    │  │   (Global)    │        │   (Precise)    │                  │
    │  │               │        │                │                  │
    │  │ LinearAttn ⊙  │───────►│  Input Gate    │ σ(W·glu) gates   │
    │  │   DLinOSS     │        │       ↓        │ input features   │
    │  │       ↓       │        │  SWAttention   │                  │
    │  │   glu_out     │        │                │                  │
    │  └───────┬───────┘        └───────┬────────┘                  │
    │          │                        │                           │
    │          └──────────┬─────────────┘                           │
    │                     ▼                                         │
    │           ┌─────────────────┐                                 │
    │           │  Adaptive Mix   │  α·GLU + (1-α)·Local            │
    │           │  (α-mixing)     │  where α = σ(learned + t_bias)  │
    │           └────────┬────────┘                                 │
    │                    ▼                                          │
    │           ┌─────────────────┐                                 │
    │           │   SwiGLU FFN    │  d → 3d/2 → split → swish⊙ → d  │
    │           │   + Residual    │  (transform nonlinearity)       │
    │           └─────────────────┘                                 │
    └───────────────────────────────────────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │    Classification Head        │  LayerNorm → Dense(emb → 19)
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │    CRF Layer (optional)       │  Viterbi decoding for valid
    │                               │  BIO sequences
    └───────────────┬───────────────┘
                    │
                    ▼
  Output: [B-PERSON, I-PERSON, O, B-PLACE]
```

## Core Components

### DLinOSS (Damped Linear Oscillatory State Space Model)

Physics-inspired recurrent layer using coupled damped harmonic oscillators:

```
State Evolution:  x_{t+1} = ρ · R(θ) · x_t + B · u_t
                  where ρ = 1/(1 + Δt·damping)  (implicit damping)
                        θ = ω · Δt              (rotation angle)
```

Each oscillator maintains 2D state (position, velocity) with learnable:
- **Frequency (ω)**: Controls oscillation speed
- **Damping (α)**: Controls decay rate
- **Step size (Δt)**: Selective time discretization

### SWAttention (Sliding Window Attention)

Local attention restricted to a window around each position:

```
Attention(Q, K, V) = sigsoftmax(QK^T / √d · mask) · V

where mask[i,j] = -∞ if |i-j| > window_size
```

Uses `sigsoftmax` (sigmoid-enhanced softmax) for sharper attention patterns.

### LinearAttention

O(n) global attention using the kernel trick:

```
Instead of:  softmax(QK^T)V     → O(n²)
Use:         φ(Q)(φ(K)^T V)    → O(n)
```

Provides global context efficiently, complementing local SWAttention.

### SwiGLU FFN

Swish-Gated Linear Unit feed-forward network from "GLU Variants Improve Transformer" (Shazeer, 2020):

```
FFN(x) = Dense(Swish(a) ⊙ b) where [a, b] = split(Dense(x))

Expansion: d → 3d/2 → split → swish(half) ⊙ other → d
```

Provides transform-type nonlinearity after the α-mixing step. The 3/2 expansion factor (e.g., 384 → 576 → 288 split → 384).

## NER Label Schema

OssammaNER uses a RAG-optimized 9-entity-type schema with BIO tagging (**19 labels** total).

### Entity Types

| Type | Description | Examples |
|------|-------------|----------|
| **PERSON** | Individual people | "Barack Obama", "Marie Curie" |
| **AGENCY** | Organizations, companies, institutions | "Google", "United Nations", "FDA" |
| **PLACE** | Locations, geographic entities | "Paris", "Mount Everest", "Europe" |
| **ORGANISM** | Living things: animals, plants, species | "dolphin", "oak tree", "E. coli" |
| **EVENT** | Occurrences, happenings | "World War II", "Olympics", "IPO" |
| **INSTRUMENT** | Tools, devices, equipment | "microscope", "Python", "MRI scanner" |
| **WORK** | Creative outputs, publications | "Hamlet", "Nature journal", "GPT-4" |
| **DOMAIN** | Fields, categories, topics | "astrology", "media", "quantum physics" |
| **MEASURE** | Quantities, dates, money, time | "500kg", "2024", "$1M", "3 hours" |

### Semantic Coverage

```
Who?        → PERSON, AGENCY
What?       → ORGANISM, INSTRUMENT, WORK
Where?      → PLACE
When/How?   → MEASURE
What happened? → EVENT
What field?    → DOMAIN
```

### BIO Tagging Example

```
"Barack Obama visited Paris"
 B-PERSON I-PERSON O      B-PLACE
```

### Design Rationale

| This Schema | Standard NER Equivalent |
|-------------|-------------------------|
| AGENCY | ORG (same semantics) |
| MEASURE | DATE + TIME + QUANTITY + MONEY (consolidated) |
| DOMAIN | No direct equivalent (catch-all for categorical concepts) |

### Possible Improvements

1. **Split MEASURE** - Separate DATE/TIME from QUANTITY/MONEY for finer temporal queries
2. **Add NORP** - Nationalities, religious, political groups ("Republicans", "French citizens")
3. **PRODUCT vs INSTRUMENT** - Commercial products may warrant separate handling
4. **LANGUAGE type** - For multilingual RAG ("English", "Mandarin")
5. **Clarify DOMAIN boundaries** - Is "AI" a DOMAIN or INSTRUMENT?
6. **Nested entity support** - "New York Times" is both AGENCY and WORK

## Project Structure

```
Ossamma/
├── src/
│   ├── Ossamma.jl           # Main module, OssammaBlock, OssammaNERBlock
│   ├── ossm.jl              # Basic OSSM layer
│   ├── Dlinoss.jl           # DLinOSS (Damped Linear Oscillatory SSM)
│   ├── Attention.jl         # SWAttention (Sliding Window)
│   ├── linearAttention.jl   # O(n) Linear Attention
│   ├── NER.jl               # OssammaNER model
│   ├── CRF.jl               # Conditional Random Field
│   ├── Training.jl          # Loss functions, training utilities
│   ├── data/
│   │   ├── NERDataset.jl    # Data loading and batching
│   │   ├── Tokenizer.jl     # BPE tokenization
│   │   └── Augmentation.jl  # Data augmentation
│   ├── evaluation/
│   │   └── NERMetrics.jl    # F1, precision, recall
│   └── serve/
│       ├── InferenceServer.jl  # HTTP API
│       └── Monitoring.jl       # GPU monitoring
├── scripts/
│   ├── train_ner_production.jl  # Production training
│   ├── export_model.jl          # Model serialization
│   └── download_ner_data.jl     # Data utilities
├── configs/
│   ├── ner_production_110m.toml # Production config
│   └── ner_dev.toml             # Development config
├── checkpoints/                  # Saved model weights
└── docs/
    ├── OSSAMMA_NER_ARCHITECTURE.md
    └── NER_TRAINING_PLAN.md
```

## Quick Start

### Installation

```bash
cd Ossamma
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Training

```bash
# Start training on GPU
julia --project=. scripts/train_ner_production.jl --synthetic

# Or with config file
julia --project=. scripts/train_ner_production.jl --config configs/ner_production_110m.toml
```

### Inference

```julia
using Ossamma

# Load model
config = load_ner_config("configs/ner_production_110m.toml")
model = OssammaNER(config)
ps, st = load_checkpoint("checkpoints/ner_110m/latest.jls")

# Predict
text = "Barack Obama visited Paris"
tokens, labels, entities = predict(model, ps, st, text)
# entities: [(text="Barack Obama", label="PERSON"), (text="Paris", label="PLACE")]
```

## Model Configurations

| Config | Embedding | Layers | Heads | Params | Use Case |
|--------|-----------|--------|-------|--------|----------|
| `tiny` | 64 | 2 | 2 | ~500K | Debugging |
| `small` | 256 | 4 | 4 | ~5M | Experiments |
| `base` | 384 | 6 | 6 | ~15M | Production |
| `large` | 512 | 12 | 8 | ~50M | High accuracy |

## Training Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Raw Text    │────►│  Tokenizer   │────►│  NERDataset  │
│  + Labels    │     │  (BPE 32k)   │     │  (batching)  │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
┌──────────────┐     ┌──────────────┐     ┌──────▼───────┐
│  Checkpoint  │◄────│   Optimizer  │◄────│   Model      │
│  (every 1k)  │     │ (AdamW+cos)  │     │  Forward     │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                     ┌──────────────┐     ┌──────▼───────┐
                     │   Metrics    │◄────│  NER Loss    │
                     │  (F1, etc)   │     │  + CRF Loss  │
                     └──────────────┘     └──────────────┘
```

## Dependencies

- **Lux.jl** - Neural network framework
- **NNlib.jl** - Neural network primitives
- **Zygote.jl** - Automatic differentiation
- **CUDA.jl** - GPU support
- **Optimisers.jl** - Adam, learning rate schedules

## License

MIT

## Citation

```bibtex
@software{ossamma2024,
  title={Ossamma: Oscillatory State Space Attention for NER},
  year={2024},
  url={https://github.com/your-repo/ossamma}
}
```
