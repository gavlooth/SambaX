# Mixture of Experts for Thought Generation (MoET)

## Overview

A novel **Mixture-of-Experts architecture for reasoning**, not just token routing. Instead of hoping the model learns logic from text patterns, we train dedicated **Expert Towers** for different modes of thought and gate their activation based on reasoning requirements.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MoET: MIXTURE OF EXPERT THOUGHTS                         │
│                                                                             │
│   "Different thoughts require different circuits"                           │
│                                                                             │
│   Unlike standard MoE:                                                       │
│     Traditional: Route TOKENS to expert FFNs                                │
│     MoET: Route REASONING MODES to expert towers                            │
│                                                                             │
│   Total: 168 layers (but only ~40 active per forward pass)                  │
│   ├── Logic Expert: 48 layers (formal reasoning)                           │
│   ├── Language Expert: 48 layers (semantics/context)                       │
│   ├── Math Expert: 48 layers (arithmetic/algebra)                          │
│   ├── Memory Expert: 16 layers (retrieval/recall)                          │
│   └── Fusion: 8 layers (combines activated experts)                        │
│                                                                             │
│   Effective depth: 40-60 layers (sparse routing)                            │
│   Compute: O(T × d²) per expert = extremely efficient                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Insight: MoE for Thoughts, Not Tokens

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TRADITIONAL MoE (Switch Transformer)                     │
│                                                                             │
│   Token → Router → Expert FFN 1                                             │
│                  → Expert FFN 2   → Combine → Output                        │
│                  → Expert FFN 3                                             │
│                                                                             │
│   Problem: All experts are the same architecture (FFN)                      │
│            Just different weights                                           │
│            No specialization for different TYPES of reasoning               │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                     MoET (Our Approach)                                      │
│                                                                             │
│   Input → Router → Logic Tower (48L, trained on proofs)                     │
│                  → Language Tower (48L, trained on text)                    │
│                  → Math Tower (48L, trained on equations)     → Fusion      │
│                  → Memory Tower (16L, trained on retrieval)                 │
│                                                                             │
│   Key differences:                                                          │
│   1. Experts are FULL TOWERS, not single FFN layers                        │
│   2. Experts trained on DIFFERENT DATA for different skills                 │
│   3. Routing is by REASONING MODE, not load balancing                       │
│   4. O(T) complexity allows DEEP experts cheaply                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Full Expert Tower Architecture

### Why We Can Go Deep

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     COMPUTE BUDGET ANALYSIS                                  │
│                                                                             │
│   Transformer MoE (DeepSeek-style):                                         │
│     Per token: O(T² × d) attention + O(d²) expert FFN                       │
│     Bottleneck: Attention is O(T²)                                          │
│     Result: Can't go very deep without massive compute                      │
│                                                                             │
│   Ossamma MoET:                                                              │
│     Per token: O(T × d²) per expert tower                                   │
│     No O(T²) bottleneck!                                                    │
│     Result: Can afford 48-96 layers per expert                              │
│                                                                             │
│   With sparse routing (1-2 experts active):                                 │
│     Effective compute = 1.5 × O(T × d²) × L_active                         │
│                                                                             │
│   For T=4096, d=384, L=48:                                                  │
│     Transformer: 4096² × 384 × 48 = 300B FLOPs                             │
│     Ossamma: 4096 × 384² × 48 × 1.5 = 43B FLOPs                            │
│                                                                             │
│     7× cheaper → can afford 7× more layers!                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Four Expert Towers

```
                              Input Tokens
                                   │
                                   ▼
                      ┌────────────────────────┐
                      │   Shared Embedding     │
                      │   + Position + Time    │
                      └───────────┬────────────┘
                                  │
                      ┌───────────┴───────────┐
                      │     Expert Router     │
                      │   (learned gating)    │
                      └───────────┬───────────┘
                                  │
          ┌───────────┬───────────┼───────────┬───────────┐
          │           │           │           │           │
          ▼           ▼           ▼           ▼           │
   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
   │   LOGIC     │ │  LANGUAGE   │ │    MATH     │ │   MEMORY    │
   │   EXPERT    │ │   EXPERT    │ │   EXPERT    │ │   EXPERT    │
   │             │ │             │ │             │ │             │
   │  48 layers  │ │  48 layers  │ │  48 layers  │ │  16 layers  │
   │             │ │             │ │             │ │             │
   │ Trained on: │ │ Trained on: │ │ Trained on: │ │ Trained on: │
   │ - Prop/FOL  │ │ - Text      │ │ - Arithmetic│ │ - QA pairs  │
   │ - SAT/SMT   │ │ - NLI       │ │ - Algebra   │ │ - Facts     │
   │ - Proofs    │ │ - Semantics │ │ - Equations │ │ - Retrieval │
   │ - Inference │ │ - Context   │ │ - Calc      │ │ - Lookup    │
   │             │ │             │ │             │ │             │
   │ Oscillators:│ │ Oscillators:│ │ Oscillators:│ │ Oscillators:│
   │ Fast→Slow   │ │ Fast→Slow   │ │ Fast→Slow   │ │ Fast only   │
   │ (planning)  │ │ (discourse) │ │ (carry/bor) │ │ (no state)  │
   └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
          │               │               │               │
          └───────────────┴───────┬───────┴───────────────┘
                                  │
                        Weighted combination
                        g_l×L + g_a×A + g_m×M + g_r×R
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │    Fusion Layers (8)   │
                      │  Cross-expert attention │
                      │  + Conflict resolution  │
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │        LM Head         │
                      └────────────────────────┘
```

### Expert Specifications

#### Logic Expert (48 Layers)

```
Purpose: Formal logical reasoning and proof generation

Training Data:
  - Propositional logic (10M examples)
  - First-order logic (5M examples)
  - SAT/SMT problems (5M examples)
  - Proof traces from theorem provers (2M examples)
  - Logical fallacy detection (1M examples)

Oscillator Configuration:
  Layers 1-16:  freq ∈ [10, 100]   "Symbol manipulation"
  Layers 17-32: freq ∈ [1, 30]    "Rule application"
  Layers 33-48: freq ∈ [0.1, 10]  "Proof planning"

Activation Pattern:
  High on: "if...then", "therefore", "∀", "∃", "implies", "valid"
  Low on: Names, descriptions, narratives
```

#### Language Expert (48 Layers)

```
Purpose: Semantic understanding and natural language processing

Training Data:
  - Wikipedia, books, web text (100M examples)
  - Natural language inference (10M examples)
  - Reading comprehension (5M examples)
  - Semantic similarity (5M examples)
  - Discourse coherence (5M examples)

Oscillator Configuration:
  Layers 1-16:  freq ∈ [5, 50]    "Syntax, local coherence"
  Layers 17-32: freq ∈ [0.5, 15]  "Phrases, clauses"
  Layers 33-48: freq ∈ [0.05, 5]  "Document context, coreference"

Activation Pattern:
  High on: Narratives, descriptions, dialogue
  Low on: Formulas, code, structured data
```

#### Math Expert (48 Layers)

```
Purpose: Arithmetic, algebra, and numerical reasoning

Training Data:
  - Arithmetic problems (20M examples)
  - Algebraic equations (10M examples)
  - Word problems → equations (10M examples)
  - Step-by-step solutions (5M examples)
  - Numerical estimation (2M examples)

Oscillator Configuration:
  Layers 1-16:  freq ∈ [20, 200]  "Digit manipulation, carry/borrow"
  Layers 17-32: freq ∈ [2, 40]   "Operation sequencing"
  Layers 33-48: freq ∈ [0.2, 10] "Problem decomposition"

Activation Pattern:
  High on: Numbers, operators, "calculate", "solve", "="
  Low on: Pure text, logical connectives without numbers
```

#### Memory Expert (16 Layers)

```
Purpose: Fact retrieval and context lookup

Training Data:
  - Question-answer pairs (50M examples)
  - Fact verification (10M examples)
  - Entity attributes (10M examples)
  - Temporal facts (5M examples)

Oscillator Configuration:
  All layers: freq ∈ [1, 50]  "Fast retrieval, no long-term state"

Activation Pattern:
  High on: "Who is", "What is", "When did", factual questions
  Low on: Reasoning, computation, generation

Note: Smaller because retrieval is pattern matching, not deep reasoning.
      Could be replaced with actual retrieval system in production.
```

### Layer Count Justification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     WHY 48 LAYERS PER EXPERT?                                │
│                                                                             │
│   Research on reasoning depth:                                              │
│   - Simple inference (MP): ~4-8 layers sufficient                          │
│   - Chain of 3 rules: ~12-16 layers                                        │
│   - Chain of 5 rules: ~20-24 layers                                        │
│   - Chain of 10 rules: ~40-48 layers                                       │
│   - Open-ended proof search: 48+ layers                                    │
│                                                                             │
│   With 48 layers, can handle:                                               │
│   - 10-step logical proofs                                                  │
│   - 5-level nested quantifiers                                              │
│   - Complex algebraic simplification                                        │
│   - Multi-paragraph discourse                                               │
│                                                                             │
│   Total model: 48×3 + 16 + 8 = 168 layers                                  │
│   Active per forward: ~48-56 layers (1-2 experts + fusion)                 │
│   Parameters: ~400M (comparable to BERT-large but MUCH deeper)              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Expert Routing

```
Router Network:

  Input: x (d, seq, batch)

  Compute expert scores:
    h = LayerNorm(mean(x, dim=seq))           # (d, batch)
    scores = Dense(d → 4)(h)                   # (4, batch) for 4 experts
    gates = softmax(scores / temperature)      # (4, batch)

  Top-k routing (k=2 default):
    active_experts = top_k(gates, k=2)
    gate_weights = normalize(gates[active_experts])

  Sparse computation:
    output = Σ gate_weights[i] × Expert_i(x)  for i in active_experts

  Load balancing loss:
    L_balance = CV(expert_usage)²             # Coefficient of variation

  Auxiliary losses:
    L_router = L_balance + λ_entropy × H(gates)  # Encourage exploration
```

### Routing Examples

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Input: "If all mammals are warm-blooded and whales are mammals,            │
│         what can we conclude about whales?"                                │
│                                                                             │
│ Router scores: Logic=0.7, Language=0.2, Math=0.05, Memory=0.05             │
│ Active: Logic (0.78) + Language (0.22)                                      │
│                                                                             │
│ Logic Expert: "∀x: Mammal(x)→WarmBlooded(x), Mammal(whale) ⊢ WarmBlooded"  │
│ Language Expert: Contextualizes "whales", "warm-blooded"                   │
│ Fusion: "Whales are warm-blooded."                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ Input: "Calculate 23 × 47 step by step"                                    │
│                                                                             │
│ Router scores: Logic=0.1, Language=0.1, Math=0.75, Memory=0.05             │
│ Active: Math (0.88) + Language (0.12)                                      │
│                                                                             │
│ Math Expert: "23×47 = 23×40 + 23×7 = 920 + 161 = 1081"                    │
│ Language Expert: Formats as readable steps                                 │
│ Fusion: "Step 1: 23×40=920. Step 2: 23×7=161. Step 3: 920+161=1081."     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Input: "Who was the first person to walk on the moon?"                     │
│                                                                             │
│ Router scores: Logic=0.05, Language=0.15, Math=0.0, Memory=0.8             │
│ Active: Memory (0.84) + Language (0.16)                                    │
│                                                                             │
│ Memory Expert: Retrieves "Neil Armstrong, Apollo 11, 1969"                 │
│ Language Expert: Forms natural response                                    │
│ Fusion: "Neil Armstrong was the first person to walk on the moon in 1969."│
├─────────────────────────────────────────────────────────────────────────────┤
│ Input: "Explain why the proof of Fermat's Last Theorem was significant"    │
│                                                                             │
│ Router scores: Logic=0.3, Language=0.3, Math=0.3, Memory=0.1               │
│ Active: All four (balanced)                                                │
│                                                                             │
│ This requires: Logic (proof structure), Language (explanation),            │
│                Math (theorem content), Memory (historical facts)           │
│ Fusion: Combines all perspectives                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Problem Statement

### Current Approaches (Others)

```
Training: "If it rains, the ground is wet. It rained. Therefore the ground is wet."
Hope: Model figures out modus ponens from examples
Reality: Pattern matching, not actual logical inference
         Breaks on novel combinations of premises
         Inconsistent on slightly rephrased problems
```

### Our Solution

```
Logic Core (20L): Trained on propositional/FOL → actually does inference
Language Shell (20L): Maps natural language ↔ logical form
Gating: Routes through logic tower when reasoning needed

Result: Guaranteed logical consistency on structured problems
        Compositional generalization to novel premise combinations
        Verifiable reasoning on pure logic subset
```

---

## Architecture

### High-Level Structure

```
                            Input Tokens
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Shared Embedding     │
                    │   (vocab_size → d)     │
                    └───────────┬────────────┘
                                │
                    ┌───────────┴───────────┐
                    │     Gate Network      │
                    │   g = σ(MLP(x))       │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
        ┌───────────────────┐   ┌───────────────────┐
        │   LOGIC TOWER     │   │  LANGUAGE TOWER   │
        │   (20 layers)     │   │  (20 layers)      │
        │                   │   │                   │
        │   Trained on:     │   │   Trained on:     │
        │   - Prop logic    │   │   - Natural text  │
        │   - FOL           │   │   - Semantics     │
        │   - SAT problems  │   │   - Context       │
        │   - Proof traces  │   │   - Reasoning in  │
        │                   │   │     natural lang  │
        └─────────┬─────────┘   └─────────┬─────────┘
                  │                       │
                  └───────────┬───────────┘
                              │
                    g × Logic + (1-g) × Language
                              │
                              ▼
                    ┌────────────────────────┐
                    │   Fusion Layers (4L)   │
                    │   Combines both views  │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │      LM Head           │
                    │   (d → vocab_size)     │
                    └───────────┬────────────┘
                                │
                                ▼
                           Output Logits
```

### Component Details

#### 1. Shared Embedding Layer

```
Input: token_ids (seq_len, batch)
Output: embeddings (d, seq_len, batch)

- Uses Granite vocabulary (49,155 tokens)
- Shared between both towers
- Includes position embeddings
- Includes time embeddings (for diffusion)
```

#### 2. Gate Network

```
Input: x (d, seq_len, batch)
Output: g (1, seq_len, batch) or (d, seq_len, batch)

Architecture:
  x → LayerNorm → Dense(d → d//4) → GELU → Dense(d//4 → 1) → Sigmoid

Interpretation:
  g ≈ 1.0: "This token needs logical reasoning"
  g ≈ 0.0: "This token is natural language / pattern matching"
  g ≈ 0.5: "Uncertain, use both towers"

Training signal:
  - Logic problems → push g toward 1
  - Pure text → push g toward 0
  - Sparsity regularization → push g away from 0.5
```

#### 3. Logic Tower (20 Layers)

```
Architecture: 20 × OssammaDrafterBlockDeep

Each block:
  Input ──► TimeConditionedLayerNorm ──► GLU Projection (d → 2d)
                                              │
                              ┌───────────────┴───────────────┐
                              │                               │
                              ▼                               ▼
                      LinearAttention              DLinOSS (Oscillators)
                              │                               │
                              └───────────┬───────────────────┘
                                          │
                                   ⊙ (GLU gating)
                                          │
                                   SwiGLU FFN
                                          │
                                   + Residual
                                          │
                                      LayerNorm
                                          │
                                       Output

Oscillator Configuration (Logic Tower):
  - Layers 1-7:   freq ∈ [10, 100]   "Fast logic" (symbol manipulation)
  - Layers 8-14:  freq ∈ [1, 30]    "Medium logic" (rule application)
  - Layers 15-20: freq ∈ [0.1, 10]  "Slow logic" (proof planning)

Training Data:
  - Propositional logic (AND, OR, NOT, IMPLIES, IFF)
  - First-order logic (∀, ∃, predicates, functions)
  - SAT/SMT problems
  - Proof traces with step-by-step reasoning
```

#### 4. Language Tower (20 Layers)

```
Architecture: 20 × OssammaDrafterBlockDeep (same structure as Logic Tower)

Oscillator Configuration (Language Tower):
  - Layers 1-7:   freq ∈ [5, 50]    "Fast language" (syntax, local coherence)
  - Layers 8-14:  freq ∈ [0.5, 15]  "Medium language" (phrases, clauses)
  - Layers 15-20: freq ∈ [0.05, 5]  "Slow language" (document context)

Training Data:
  - Standard text corpora
  - Natural language inference
  - Reading comprehension
  - Semantic similarity
```

#### 5. Fusion Layers (4 Layers)

```
Architecture: 4 × OssammaDrafterBlockDeep

Purpose:
  - Combine logical and linguistic representations
  - Resolve conflicts between towers
  - Produce coherent output

Input: g × LogicOut + (1-g) × LanguageOut
Output: Final hidden states for LM head
```

---

## Sparse Routing (Efficiency)

### Motivation

Running both towers for every token is wasteful. Most tokens are clearly either:
- Logic-heavy (formulas, operators, quantifiers)
- Language-heavy (articles, prepositions, common phrases)

### Sparse Routing Algorithm

```
Input: x (d, seq_len, batch), threshold τ = 0.7

1. Compute gate values: g = GateNetwork(x)

2. Classify tokens:
   logic_mask = g > τ           # Clearly needs logic
   language_mask = g < (1 - τ)  # Clearly natural language
   both_mask = τ ≤ g ≤ (1-τ)    # Uncertain, run both

3. Route and compute:
   output[logic_mask] = LogicTower(x[logic_mask])
   output[language_mask] = LanguageTower(x[language_mask])
   output[both_mask] = g[both_mask] × LogicTower(x[both_mask]) +
                       (1-g[both_mask]) × LanguageTower(x[both_mask])

4. Apply fusion layers to all tokens

Effective computation:
  - Pure logic tokens: 20L logic + 4L fusion = 24 layers
  - Pure language tokens: 20L language + 4L fusion = 24 layers
  - Mixed tokens: 40L + 4L fusion = 44 layers

Expected distribution (on reasoning tasks):
  - ~40% pure logic
  - ~40% pure language
  - ~20% mixed

Average effective depth: 0.4×24 + 0.4×24 + 0.2×44 = 28 layers
```

### Batched Sparse Routing (GPU-Efficient)

```
For GPU efficiency, route by groups rather than individual tokens:

1. Compute mean gate per sequence: g_seq = mean(g, dim=seq)
2. Classify sequences (not tokens):
   - g_seq > 0.6 → Logic-heavy sequence → Logic tower only
   - g_seq < 0.4 → Language-heavy sequence → Language tower only
   - Otherwise → Run both towers

This allows batched matrix operations without scatter/gather overhead.
```

---

## Training Pipeline

### Stage 1: Logic Tower Pretraining (50K steps)

```
Data: Pure formal logic
  - 40% Propositional logic
  - 30% First-order logic
  - 20% SAT problems
  - 10% Proof traces

Frozen: Language tower, Gate network
Training: Logic tower only

Loss: MLM loss on logic expressions
      + Validity classification loss
      + Proof step prediction loss

Objective: Logic tower learns to do actual logical inference,
           not pattern matching on surface forms.
```

### Stage 2: Language Tower Pretraining (50K steps)

```
Data: Natural language text
  - 50% General text (Wikipedia, books)
  - 30% Natural language inference
  - 20% Reading comprehension

Frozen: Logic tower, Gate network
Training: Language tower only

Loss: Standard MLM loss

Objective: Language tower learns semantics and context,
           complementary to logic tower.
```

### Stage 3: Joint Finetuning with Gating (100K steps)

```
Data: Mixed logic + language
  - 30% Pure logic problems
  - 30% Pure natural language
  - 40% Mixed (logic in natural language, word problems)

Training: All parameters (both towers + gate + fusion)

Loss: L_total = L_task + λ_sparse × L_sparsity + λ_balance × L_balance

Where:
  L_task = MLM loss + (optional) distillation from Granite

  L_sparsity = -mean(g × log(g) + (1-g) × log(1-g))
               Pushes g toward 0 or 1 (sparse routing)

  L_balance = |mean(g) - 0.5|
              Prevents mode collapse (all logic or all language)

Hyperparameters:
  λ_sparse = 0.01
  λ_balance = 0.01
```

### Stage 4: TiDAR Integration (Optional, 50K steps)

```
Data: Reasoning tasks with thinking traces

Training: Full model with Granite distillation

Loss: L_total = α × L_mlm + (1-α) × L_distill

Objective: Drafter learns to propose thinking tokens
           that Granite will accept.
```

---

## Logic Training Data Generation

### Propositional Logic Dataset

```
Inference Rules to Learn:

1. Modus Ponens:       P → Q, P ⊢ Q
2. Modus Tollens:      P → Q, ¬Q ⊢ ¬P
3. Hypothetical Syll:  P → Q, Q → R ⊢ P → R
4. Disjunctive Syll:   P ∨ Q, ¬P ⊢ Q
5. Conjunction:        P, Q ⊢ P ∧ Q
6. Simplification:     P ∧ Q ⊢ P
7. Addition:           P ⊢ P ∨ Q
8. Resolution:         P ∨ Q, ¬P ∨ R ⊢ Q ∨ R
9. Contradiction:      P, ¬P ⊢ ⊥
10. Double Negation:   ¬¬P ⊢ P

Fallacies to Reject:

1. Affirming Consequent: P → Q, Q ⊬ P
2. Denying Antecedent:   P → Q, ¬P ⊬ ¬Q
3. Illicit Major/Minor:  Syllogism errors
4. Undistributed Middle: Quantifier errors

Example Format:

  Input:  "P → Q. Q → R. P. Therefore ?"
  Output: "R. [Proof: P→Q, P ⊢ Q (MP). Q→R, Q ⊢ R (MP).]"

  Input:  "P → Q. Q. Therefore ?"
  Output: "INVALID. [Affirming the consequent: Q does not prove P]"
```

### First-Order Logic Dataset

```
Rules to Learn:

1. Universal Instantiation:   ∀x: P(x) ⊢ P(a)
2. Universal Generalization:  P(a) for arbitrary a ⊢ ∀x: P(x)
3. Existential Instantiation: ∃x: P(x) ⊢ P(c) for fresh c
4. Existential Generalization: P(a) ⊢ ∃x: P(x)
5. Unification:               Match terms to apply rules

Example Format:

  Input:  "∀x: Human(x) → Mortal(x). Human(Socrates). Therefore ?"
  Output: "Mortal(Socrates). [UI on ∀x with x=Socrates, then MP.]"

  Input:  "∃x: Prime(x) ∧ Even(x). Therefore ?"
  Output: "Prime(c) ∧ Even(c) for some c. [EI introduces fresh constant.]"
```

### SAT Dataset

```
Format: CNF clauses + satisfiability + solution/proof

Example (SAT):
  Input:  "[[1, 2], [-1, 3], [-2, -3]]"
  Output: "SAT. Assignment: [true, true, true]"

Example (UNSAT):
  Input:  "[[1], [-1]]"
  Output: "UNSAT. [Resolution: [1] + [-1] = [], empty clause.]"

Generation Strategy:
  - Phase transition at ~4.26 clauses per variable
  - Mix easy SAT (underconstrained), hard SAT (phase transition),
    easy UNSAT (obvious contradiction), hard UNSAT (requires resolution)
```

---

## Integration with TiDAR

### Unified Generation Pipeline

```
Input: User question (natural language)

Phase 1: Parse and Route
  ┌─────────────────────────────────────────────────────────────┐
  │ "What is the conclusion if all humans are mortal and        │
  │  Socrates is human?"                                        │
  │                                                             │
  │ Gate network detects: "all...are" → quantifier pattern      │
  │                       "if...and" → logical connective       │
  │                                                             │
  │ Route: 70% Logic, 30% Language                              │
  └─────────────────────────────────────────────────────────────┘

Phase 2: Generate Thinking Tokens (Drafter)
  ┌─────────────────────────────────────────────────────────────┐
  │ <think>                                                     │
  │ [MASK] [MASK] [MASK] ... [MASK]  (K=8 tokens)              │
  │                                                             │
  │ Drafter proposes (via Logic Tower):                         │
  │ "Let P(x) = Human(x), Q(x) = Mortal(x). Given ∀x: P(x)→Q(x)│
  │  and P(Socrates). By UI and MP, Q(Socrates)."              │
  │ </think>                                                    │
  └─────────────────────────────────────────────────────────────┘

Phase 3: Verify with Granite
  ┌─────────────────────────────────────────────────────────────┐
  │ Granite verifies each thinking token:                       │
  │                                                             │
  │ "Let P(x)..." ✓ accept                                      │
  │ "Given ∀x..."  ✓ accept                                     │
  │ "By UI..."     ✓ accept (Granite knows this is valid)       │
  │ "Q(Socrates)"  ✓ accept                                     │
  │                                                             │
  │ Acceptance rate: 100% (clean logical reasoning)             │
  └─────────────────────────────────────────────────────────────┘

Phase 4: Generate Answer
  ┌─────────────────────────────────────────────────────────────┐
  │ "Socrates is mortal."                                       │
  │                                                             │
  │ (Language Tower handles natural phrasing)                   │
  └─────────────────────────────────────────────────────────────┘
```

### Why This Beats Others

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Scenario: Novel logical structure the model hasn't seen before              │
│                                                                             │
│ Problem: "If (A → B) and (B → C) and (C → D) and A, what follows?"         │
│          (4-step hypothetical syllogism chain)                              │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ GPT-4/Claude (pattern matching):                                            │
│   Might get it right, might not                                            │
│   Depends on whether similar chains appeared in training                    │
│   No guarantee of correctness                                               │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ Logic-Gated Ossamma:                                                        │
│   Logic Tower trained on: HS rule (P→Q, Q→R ⊢ P→R)                        │
│   Applies HS three times: A→B, B→C ⊢ A→C                                   │
│                           A→C, C→D ⊢ A→D                                   │
│   Then MP: A→D, A ⊢ D                                                       │
│   GUARANTEED correct by construction                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## MCP Tool Layer (General Tool Interface)

Add an MCP-based tool layer so all external tools (retrieval, calculators, proof checkers, web, DBs, etc.) share a single protocol. This makes tool usage uniform and observable, while keeping the core towers unchanged.

```
User Input
   │
   ▼
MoET Router ──► Tool Decision Head (needs_tool?, tool_name, args)
   │                         │
   │                         ▼
   │                    MCP Client
   │                         │
   │               call_tool / read_resource
   │                         │
   ▼                         ▼
Expert Towers           <tool_result>
   │                         │
   └───────────────┬─────────┘
                   ▼
                Fusion
                   ▼
                Answer
```

### MCP Integration Rules
- Tool registry: MCP client lists tools at startup; cache schemas and argument shapes.
- Tool selection: route to a small tool head (binary + tool ID + args). If `needs_tool=false`, skip MCP.
- Result injection: append tool output to the input stream as a structured block (e.g., `<tool_result name="X">...</tool_result>`), then continue generation.
- Tool-aware gating: add a compact tool-context embedding (tool name + result summary) into the router input so gates shift after tool results (e.g., bias toward Logic after a proof checker).
- Safety: enforce allowlists and argument validation before calling MCP tools; add a fallback to re-answer without tools if tool calls fail.

### Training Signals (Minimal)
- Supervise tool usage: add labels for when a tool is required and which tool should be called.
- Argument correctness: train a small loss on structured tool arguments.
- Post-tool correctness: use standard task loss after tool results are injected.

---

## Competitive Advantages

### 1. Guaranteed Logical Consistency

```
Claim: For pure propositional/FOL inputs, output is provably valid.

Proof sketch:
  - Logic Tower trained to convergence on complete logic dataset
  - All valid inference rules covered
  - All common fallacies trained as negative examples
  - If input is well-formed logic, output follows valid rules

Verification: Can check Logic Tower output against SAT solver
              100% agreement = verified reasoning
```

### 2. Compositional Generalization

```
Training: Individual rules (MP, MT, HS, etc.)
Testing: Novel combinations of 5-10 rules

Others: Fail on unseen combinations (memorization)
Ours: Succeed because rules compose (actual inference)

Benchmark: CLUTRR, LogiQA, bAbI reasoning tasks
Expected: Significant improvement on novel structures
```

### 3. Efficient Routing

```
Token classification:
  - Logic operators (∧, ∨, →, ¬, ∀, ∃): g ≈ 1.0
  - Natural language (the, is, because): g ≈ 0.0
  - Mixed (therefore, implies, hence): g ≈ 0.5

Average computation:
  - Pure logic problem: 24 layers (fast)
  - Pure language: 24 layers (fast)
  - Mixed problem: ~28 layers (still efficient)

vs. running full 44 layers always
```

### 4. Interpretable Reasoning

```
Debugging capability:

  Input: "All cats are mammals. Felix is a cat. Felix is cold-blooded."

  Gate values: "All" → 0.9 (logic), "cats" → 0.3 (mixed),
               "mammals" → 0.4 (mixed), "Felix" → 0.2 (language),
               "cold-blooded" → 0.3 (mixed)

  Logic Tower output: "∀x: Cat(x) → Mammal(x), Cat(Felix) ⊢ Mammal(Felix)"
  Language Tower: Confused by "cold-blooded" (contradicts mammal)

  Diagnosis: Logic Tower correctly infers Felix is mammal.
             Language Tower detects semantic contradiction.
             Fusion should flag inconsistency.
```

### 5. Verifiable Subset

```
For inputs that are pure propositional logic:

1. Parse input to AST
2. Run Logic Tower
3. Parse output to AST
4. Verify with SAT solver / proof checker

If verification passes: Output is PROVABLY correct
No other LLM offers formal verification of reasoning.

Use case: High-stakes reasoning (legal, medical, financial)
          Where correctness is more important than fluency
```

---

## Model Configurations

### Config 1: MoET-168L (Full, Recommended)

```toml
[model]
name = "moet_168L"
vocab_size = 49155  # Granite
embedding_dimension = 384
number_of_heads = 6
total_layers = 168  # But only ~56 active per forward
active_experts = 2  # Top-k routing

[experts]
num_experts = 4

[experts.logic]
num_layers = 48
oscillator_freq_ranges = [
    [10.0, 100.0],  # Layers 1-16: symbol manipulation
    [1.0, 30.0],    # Layers 17-32: rule application
    [0.1, 10.0],    # Layers 33-48: proof planning
]
state_dimension = 384
use_parallel_scan = true
dropout = 0.1

[experts.language]
num_layers = 48
oscillator_freq_ranges = [
    [5.0, 50.0],    # Layers 1-16: syntax
    [0.5, 15.0],    # Layers 17-32: clauses
    [0.05, 5.0],    # Layers 33-48: discourse
]
state_dimension = 384
use_parallel_scan = true
dropout = 0.1

[experts.math]
num_layers = 48
oscillator_freq_ranges = [
    [20.0, 200.0],  # Layers 1-16: digit ops
    [2.0, 40.0],    # Layers 17-32: operation seq
    [0.2, 10.0],    # Layers 33-48: decomposition
]
state_dimension = 384
use_parallel_scan = true
dropout = 0.1

[experts.memory]
num_layers = 16
oscillator_freq_ranges = [[1.0, 50.0]]  # Fast retrieval
state_dimension = 256
use_parallel_scan = true
dropout = 0.05

[fusion]
num_layers = 8
oscillator_freq_ranges = [[0.1, 20.0]]
cross_expert_attention = true

[router]
hidden_dim = 96
temperature = 1.0
top_k = 2
load_balance_weight = 0.01

[training]
# Stage 1: Expert pretraining (parallel)
logic_pretrain_steps = 100000
language_pretrain_steps = 100000
math_pretrain_steps = 100000
memory_pretrain_steps = 50000

# Stage 2: Joint finetuning
joint_finetune_steps = 200000

# Stage 3: TiDAR integration
tidar_finetune_steps = 100000

[training.optimizer]
lr = 1e-4
weight_decay = 0.01
warmup_steps = 5000

# Estimated parameters
# Logic: 48L × ~2M/layer = ~96M
# Language: 48L × ~2M/layer = ~96M
# Math: 48L × ~2M/layer = ~96M
# Memory: 16L × ~1.5M/layer = ~24M
# Fusion: 8L × ~2M/layer = ~16M
# Embeddings: ~19M
# Total: ~350M parameters
# Active: ~120M per forward (1-2 experts + fusion)
```

### Config 2: MoET-104L (Efficient)

```toml
[model]
name = "moet_104L"
vocab_size = 49155
embedding_dimension = 384
total_layers = 104
active_experts = 2

[experts.logic]
num_layers = 32

[experts.language]
num_layers = 32

[experts.math]
num_layers = 32

[experts.memory]
num_layers = 0  # Disabled, use external retrieval

[fusion]
num_layers = 8

# Parameters: ~250M total, ~90M active
# Good for: Single-GPU training, faster iteration
```

### Config 3: MoET-232L (Research/Maximum)

```toml
[model]
name = "moet_232L"
vocab_size = 49155
embedding_dimension = 512
number_of_heads = 8
total_layers = 232
active_experts = 2

[experts.logic]
num_layers = 64

[experts.language]
num_layers = 64

[experts.math]
num_layers = 64

[experts.memory]
num_layers = 24

[fusion]
num_layers = 16

# Parameters: ~700M total, ~200M active
# For: Multi-GPU training, maximum reasoning depth
# Can handle: 15+ step proofs, complex multi-part problems
```

### Config 4: MoET-Lite-56L (Deployment)

```toml
[model]
name = "moet_lite_56L"
vocab_size = 49155
embedding_dimension = 256
total_layers = 56
active_experts = 1  # Single expert for speed

[experts.logic]
num_layers = 24

[experts.language]
num_layers = 24

[experts.math]
num_layers = 0  # Merged into logic

[experts.memory]
num_layers = 0  # Use external retrieval

[fusion]
num_layers = 8

# Parameters: ~80M total, ~40M active
# For: Edge deployment, real-time inference
# Trade-off: Less specialized, but much faster
```

### Parameter Comparison

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Config          │ Total Layers │ Active │ Total Params │ Active Params │     │
├──────────────────────────────────────────────────────────────────────────────┤
│ MoET-168L       │ 168          │ ~56    │ ~350M        │ ~120M         │ ★   │
│ MoET-104L       │ 104          │ ~40    │ ~250M        │ ~90M          │     │
│ MoET-232L       │ 232          │ ~80    │ ~700M        │ ~200M         │     │
│ MoET-Lite-56L   │ 56           │ ~32    │ ~80M         │ ~40M          │     │
├──────────────────────────────────────────────────────────────────────────────┤
│ Comparison:                                                                  │
│ DeepSeek-R1     │ ~60          │ ~60    │ 671B         │ ~20B active   │     │
│ GPT-4 (est.)    │ ~120         │ ~120   │ ~1.7T        │ ~200B active  │     │
│ Llama-3-8B      │ 32           │ 32     │ 8B           │ 8B            │     │
│                                                                              │
│ MoET advantage: 168 layers with only 120M active params!                    │
│                 O(T) complexity makes this possible.                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Evaluation Plan

### Logic Benchmarks

| Benchmark | Task | Expected Improvement |
|-----------|------|---------------------|
| LogiQA | Logical reasoning MCQ | +15-20% over baseline |
| ReClor | Reading comprehension + logic | +10-15% |
| FOLIO | First-order logic NLI | +20-30% |
| ProofWriter | Proof generation | +25-35% |
| bAbI (tasks 15-20) | Basic reasoning | Near 100% |

### Generalization Tests

| Test | Description | Expected |
|------|-------------|----------|
| Rule composition | 5-10 rule chains unseen in training | 90%+ accuracy |
| Variable scaling | 20+ variables (trained on 10) | Minimal degradation |
| Negation depth | Triple/quadruple negation | Correct handling |
| Quantifier mixing | ∀∃∀ patterns | Correct scoping |

### Efficiency Metrics

| Metric | Target |
|--------|--------|
| Average effective layers | ≤28 (of 44) |
| Logic-only token % | 30-50% on reasoning tasks |
| Routing overhead | <5% of total compute |
| Sparse routing accuracy | >90% (vs. running both) |

---

## Implementation Roadmap

### Phase 1: Logic Data Generation (Week 1-2)
- [ ] Implement propositional logic generator
- [ ] Implement FOL generator
- [ ] Implement SAT generator
- [ ] Generate 1M+ logic examples
- [ ] Validate with external solvers

### Phase 2: Architecture Implementation (Week 2-3)
- [ ] Implement LogicGatedBlock
- [ ] Implement sparse routing
- [ ] Implement gate network
- [ ] Unit tests for each component

### Phase 3: Logic Tower Training (Week 3-4)
- [ ] Train logic tower on pure logic data
- [ ] Evaluate on held-out logic problems
- [ ] Verify with SAT solver

### Phase 4: Language Tower Training (Week 4-5)
- [ ] Train language tower on text data
- [ ] Evaluate on language benchmarks

### Phase 5: Joint Finetuning (Week 5-6)
- [ ] Joint training with gating
- [ ] Tune sparsity/balance hyperparameters
- [ ] Evaluate on mixed benchmarks

### Phase 6: TiDAR Integration (Week 6-7)
- [ ] Integrate with Granite verifier
- [ ] Evaluate on reasoning tasks with thinking
- [ ] Benchmark speed vs. accuracy

### Phase 7: Ablations and Paper (Week 7-8)
- [ ] Ablate: gating vs. no gating
- [ ] Ablate: sparse vs. dense routing
- [ ] Ablate: logic pretraining vs. joint from scratch
- [ ] Write up results

---

## TODO: Concrete Fixes for Gating Architecture (Expanded)

### TODO: Global–Local Gated Block Improvements (Requested)
- [ ] Add explicit residual form: `y = x + α ⊙ g + (1-α) ⊙ l` (do not mix raw outputs without the skip).
- [ ] Use token-wise mixing (preferred): `α_t = σ(Wα · h_t + bα)` where `h_t` is token hidden state at position `t`.
- [ ] Define `h_t` clearly and consistently (default): `h_t = RMSNorm(x_t)` (token embedding at position t, normalized).
  - [ ] Alternative experiment: `h_t = RMSNorm(g_t)` (gate driven by global context).
  - [ ] Alternative experiment: `h_t = concat(RMSNorm(x_t), RMSNorm(g_t))` (richer gate).
- [ ] Add branch output projections (separate Dense layers): `g = Wo_g(GlobalAttn(...))`, `l = Wo_l(LocalAttn(...))`.
- [ ] Normalize branches before mixing: `ĝ = Norm(g)`, `l̂ = Norm(l)` (RMSNorm or LayerNorm).
- [ ] Prevent α collapse (α → 0 or 1):
  - [ ] Initialize α near 0.5 (bias init).
  - [ ] Add small entropy regularization on α.
  - [ ] Use a mild depth bias schedule (`t_bias`) so early layers don’t collapse.
- [ ] Stabilize gate signal magnitude: `α_t = σ((Wα · h_t)/√d + bα)` (scale logits).
- [ ] Reduce global-branch dominance through gate:
  - [ ] Stop-gradient through gate input for early training (stopgrad(g) into gate only).
  - [ ] Use a smaller learning rate on gate parameters.
  - [ ] Gate on normalized features only (`RMSNorm(g_t)`).
- [ ] Choose scalar vs vector α explicitly:
  - [ ] Start with scalar α per token (stable).
  - [ ] Only consider vector α per token if needed, and only with strong normalization.
- [ ] Ensure local branch capacity is sufficient:
  - [ ] Verify window size / heads aren’t too small (avoid global dominance).
  - [ ] Optionally increase window or heads in deeper layers.
- [ ] Add diagnostics to verify both branches are used:
  - [ ] Log α distribution (mean, std, histogram).
  - [ ] Log % tokens with α > 0.8 / < 0.2.
  - [ ] Log branch norms and gradient norms.

### Critique / Gaps To Resolve
- [ ] Granularity mismatch: current router is sequence-level, but inputs are often mixed at token level; fix by moving to token/span gating and keep a small global gate only for load balancing.
- [ ] Objective mismatch: balance + entropy does not encode "correct expert"; add supervised routing targets and an explicit misroute penalty so gates learn semantics, not just uniformity.
- [ ] Correctness claims depend on routing: without a verifier-driven fallback or logic-always path, a single misroute breaks the "guaranteed" story; add hard failover logic.
- [ ] Fusion compute conflict: cross-expert attention reintroduces O(T^2); define an O(T * d^2) fusion path and document the exact compute.
- [ ] Missing contracts: tool use, verifier alignment, and routing expectations are not formally specified; add explicit contracts and tests.

### P0: Router Granularity + Supervision
- [ ] Token/Span Gate Head: add a lightweight classifier (embedding + 1-2 shallow blocks + linear head) to emit per-token expert logits; pool contiguous same-label tokens into spans; route per token/span; keep a global gate for load balancing only.
- [ ] Router Targets: build heuristic labels (regex for ∀, ∃, ->, digits, math operators, "therefore", "because", named entities); optionally add a tiny propositional/FOL parser for stronger labels; train router with CE to labels.
- [ ] Consistency Loss: penalize disagreement between router labels and expert activations (e.g., if math tokens do not activate Math expert); add an auxiliary "router-expert agreement" metric.
- [ ] Routing Contracts: codify which token patterns map to which expert; include examples and edge cases; freeze into unit tests with expected gate ranges.

### P0: Robust Routing + Safe Defaults
- [ ] Soft-to-Hard Gating: train with soft gates and anneal temperature; at inference use top-k with a floor on Logic for formal inputs and Language as default for mixed inputs.
- [ ] Logic-Always Fallback: if input contains logical operators or parsable FOL/prop structure, force Logic expert into top-k regardless of router score.
- [ ] Verification Failover: if verifier (SAT/proof checker/Granite) rejects output, re-run with forced Logic+Language and a higher Logic weight; log fallback rate.

### P0: Fusion Without O(T^2)
- [ ] Linear Fusion: replace cross-expert attention with per-token gated sums + linear (kernelized) attention across experts; only allow quadratic attention on a small subset of disagreement tokens (top-k by expert variance).
- [ ] Disagreement Detector: compute per-token variance across expert outputs; if variance > threshold, mark token for "conflict resolution" path; keep this path bounded.
- [ ] Compute Spec: document exact complexity and memory, including gating, expert calls, and fusion; add a table comparing dense vs. sparse.

### P1: Training Curriculum + Expert Health
- [ ] Expert Warmup: train Language alone for N steps; add Logic next; then Math; then Memory; finally enable full router; prevents early collapse.
- [ ] Expert Dropout: randomly disable one expert per batch; measure recovery in other experts; log redundancy score.
- [ ] STE / Gumbel Option: try STE or Gumbel-softmax instead of temperature annealing; choose based on routing stability and utilization.
- [ ] Load-Balancing Loss: implement Switch-style `L_balance = num_experts * sum(f_i * p_i)` and track coefficient of variation to avoid expert starvation.

### P1: Systems + Efficiency
- [ ] Span Batching Strategy: decide how to batch heterogeneous spans: (a) gather by expert type, (b) merge adjacent spans, (c) pad to fixed span size + mask; benchmark memory vs. throughput.
- [ ] Inference Caching: cache expert outputs for unchanged prefix; recompute only new tokens or a sliding window; include cache invalidation rules.

### P2: Observability + Metrics
- [ ] Gate Debugger: per-layer gate heatmaps, per-token expert attribution, misroute detector vs. heuristic labels; integrate with tensorboard/wandb.
- [ ] Specialization Metrics: gradient norms per expert, cosine similarity of expert outputs, confusion matrix of "correct expert vs. selected expert", collapse alerts (>60% usage).
- [ ] Acceptance/Accuracy KPIs: report routing accuracy, verifier fallback rate, expert utilization entropy, and task-level accuracy by reasoning type.

---

## References

### Logic and Reasoning
- Propositional Logic: Classical inference rules
- First-Order Logic: Quantifiers and predicates
- SAT Solvers: DPLL, CDCL algorithms
- Proof Assistants: Coq, Lean, Isabelle

### Neural Reasoning
- Neural Theorem Provers (NTPs)
- Differentiable Reasoning (∂ILP)
- Neuro-Symbolic Integration

### Mixture of Experts
- Switch Transformer (sparse routing)
- Expert Choice routing
- Soft MoE

### Relevant Benchmarks
- LogiQA, ReClor, FOLIO
- ProofWriter, EntailmentBank
- bAbI reasoning tasks
- CLUTRR (compositional reasoning)

---

## Summary: Why MoET is Different

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MoET: MIXTURE OF EXPERT THOUGHTS                         │
│                                                                             │
│   "Not just different weights - different ways of thinking"                 │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   KEY INNOVATIONS:                                                          │
│                                                                             │
│   1. Expert Towers, Not Expert FFNs                                         │
│      Traditional MoE: Same architecture, different weights                  │
│      MoET: Full 48-layer towers trained on different data                  │
│                                                                             │
│   2. Routing by Reasoning Mode, Not Load Balancing                          │
│      Traditional MoE: Balance tokens across experts for efficiency          │
│      MoET: Route by semantic need (logic vs math vs language)              │
│                                                                             │
│   3. O(T) Enables Extreme Depth                                             │
│      Traditional: O(T²) limits depth to ~60-120 layers                     │
│      MoET: O(T) allows 168+ total layers (48 per expert)                   │
│                                                                             │
│   4. Verifiable Logic Subset                                                │
│      Traditional: Hope model learned logic from text                        │
│      MoET: Logic expert trained on formal logic, verifiable via SAT        │
│                                                                             │
│   5. Specialized Oscillator Frequencies                                     │
│      Logic: Fast→Slow (symbol→planning)                                    │
│      Math: Very Fast→Medium (digits→decomposition)                         │
│      Language: Medium→Slow (syntax→discourse)                              │
│      Memory: Fast only (retrieval, no state)                               │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   COMPETITIVE POSITIONING:                                                  │
│                                                                             │
│   vs DeepSeek-R1:                                                           │
│     - Same MoE concept but for THOUGHT MODES not tokens                    │
│     - 350M params vs 671B params                                            │
│     - Verifiable logic vs pattern matching                                  │
│                                                                             │
│   vs OpenAI o1/o3:                                                          │
│     - Open architecture (vs black box)                                      │
│     - Parallel thinking via TiDAR (vs sequential AR)                       │
│     - Inspectable expert routing (vs opaque reasoning)                     │
│                                                                             │
│   vs Traditional Transformers:                                              │
│     - 168 layers possible vs ~60 max                                       │
│     - Specialized experts vs monolithic model                              │
│     - O(T) vs O(T²) complexity                                              │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   THE PITCH:                                                                │
│                                                                             │
│   "MoET: The first model that thinks in different modes.                    │
│                                                                             │
│    - Logic problems activate the Logic Expert (trained on proofs)          │
│    - Math problems activate the Math Expert (trained on equations)          │
│    - Language tasks activate the Language Expert (trained on text)          │
│    - Facts activate the Memory Expert (trained on QA)                       │
│                                                                             │
│    168 layers of depth. 350M parameters. 120M active per forward.           │
│    Verifiable reasoning on the logic subset.                                │
│    7× more efficient than comparable transformers.                          │
│                                                                             │
│    Different thoughts require different circuits.                           │
│    MoET has them."                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Next Steps

1. **Implement LogicDataGenerator.jl** - Generate training data for Logic Expert
2. **Implement MathDataGenerator.jl** - Generate training data for Math Expert
3. **Implement MoETBlock.jl** - The multi-expert gated block
4. **Implement ExpertRouter.jl** - Top-k routing with load balancing
5. **Train experts in parallel** - Each expert can train independently
6. **Joint finetuning** - Train router and fusion layers
7. **Integrate with TiDAR** - Use MoET as the drafter, Granite as verifier
8. **Benchmark on reasoning tasks** - LogiQA, FOLIO, GSM8K, MATH
