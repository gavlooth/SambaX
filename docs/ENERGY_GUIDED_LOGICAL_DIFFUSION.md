# Energy-Guided Diffusion for Logical Thought Generation

> Research notes on embedding "Laws of Thought" into the OssammaDrafter diffusion process.

## Table of Contents

1. [Motivation](#motivation)
2. [Core Idea](#core-idea-energy-guided-diffusion)
3. [The Efficiency Problem](#the-efficiency-problem)
4. [Efficient Implementation Strategies](#efficient-implementation-strategies)
5. [Approach 1: Entailment Energy](#approach-1-entailment-energy)
6. [Approach 2: Contradiction Detector](#approach-2-contradiction-detector)
7. [Approach 3: Symbolic Logic Embedding](#approach-3-symbolic-logic-embedding)
8. [Approach 4: Reasoning Graph Energy](#approach-4-reasoning-graph-energy)
9. [Approach 5: Learned Logic Prior](#approach-5-learned-logic-prior)
10. [Hybrid Neuro-Symbolic Approach](#hybrid-approach-neuro-symbolic-energy)
11. [Implementation Roadmap](#implementation-roadmap)
12. [References](#references)

---

## Motivation

The OssammaDrafter generates "thought tokens" via diffusion - these are internal reasoning steps before producing the final output. We want these thoughts to follow logical structure, not just be coherent text, but **valid reasoning chains**.

### Why Logic Matters for Thought Tokens

Unlike regular text generation where fluency is primary, thought tokens serve as intermediate reasoning. Bad reasoning leads to bad conclusions, even if each sentence is grammatically correct.

**Example of fluent but illogical thought chain:**
```
Thought 1: "The function returns an integer"
Thought 2: "Integers cannot be null in this language"
Thought 3: "We should add a null check for the return value"  <- CONTRADICTION!
```

Each sentence is well-formed, but Thought 3 contradicts the logical chain.

### Classical Laws of Thought

**Aristotelian Logic (foundational)**:
1. **Law of Identity**: A thing is itself (A = A)
   - In reasoning: A concept must maintain consistent meaning throughout an argument
   - Violation example: Using "bank" to mean financial institution, then river bank

2. **Law of Non-Contradiction**: Nothing can be both A and not-A simultaneously
   - not(A AND not-A)
   - Violation example: "X is null" and "X.value is 5" in same context

3. **Law of Excluded Middle**: Everything is either A or not-A
   - A OR not-A
   - In reasoning: Conclusions must be definite (though uncertainty is valid)

### Reasoning Rules We Want to Enforce

**Deductive Rules**:
| Rule | Form | Example |
|------|------|---------|
| Modus Ponens | P, P→Q ⊢ Q | "It's raining" + "If rain then wet" → "Ground is wet" |
| Modus Tollens | ¬Q, P→Q ⊢ ¬P | "Ground not wet" + "If rain then wet" → "Not raining" |
| Hypothetical Syllogism | P→Q, Q→R ⊢ P→R | "A→B" + "B→C" → "A→C" (transitivity) |
| Disjunctive Syllogism | P∨Q, ¬P ⊢ Q | "X or Y" + "not X" → "Y" |

**Common Fallacies to Penalize**:
| Fallacy | Form | Why It's Wrong |
|---------|------|----------------|
| Affirming Consequent | Q, P→Q ⊢ P | "Wet ground" doesn't prove rain (could be sprinkler) |
| Denying Antecedent | ¬P, P→Q ⊢ ¬Q | "No rain" doesn't mean "not wet" |
| Non Sequitur | P ⊢ Q (unrelated) | Conclusion doesn't follow from premises |
| Circular Reasoning | P because P | Using conclusion as premise |

---

## Core Idea: Energy-Guided Diffusion

### Standard Diffusion Recap

In discrete diffusion (like LLaDA/MDLM that OssammaDrafter uses):

```
Forward process:  x_0 → x_1 → ... → x_T  (gradually add noise/masks)
Reverse process:  x_T → x_{T-1} → ... → x_0  (denoise to recover)

At each step t:
  logits = model(x_t, t)
  x_{t-1} = sample(logits)
```

The model learns p(x_{t-1} | x_t) - what the less-noisy version should be.

### Adding Energy Guidance

Energy-Based Models (EBMs) define probability via energy:
```
p(x) ∝ exp(-E(x))
```
- Low energy → High probability (desirable states)
- High energy → Low probability (undesirable states)

**Guided diffusion** modifies sampling:
```
logits_guided = logits - β * ∇_x E(x)

Where:
  logits       = model's predicted log-probabilities
  E(x)         = energy function (HIGH for illogical, LOW for logical)
  β            = guidance strength (hyperparameter)
  ∇_x E(x)     = gradient of energy w.r.t. token logits
```

For discrete tokens (no true gradient), we approximate:
```
For each candidate token t:
  E(t) = energy if we choose token t
  logits_guided[t] = logits[t] - β * E(t)
```

### Guidance Strength Schedule

β should vary with diffusion timestep:
```julia
function guidance_schedule(t, T; strategy=:linear)
    # t = current timestep, T = total timesteps
    # t=T is pure noise, t=0 is final output

    progress = 1 - t/T  # 0 at start, 1 at end

    if strategy == :linear
        return β_max * progress
    elseif strategy == :cosine
        return β_max * (1 - cos(π * progress)) / 2
    elseif strategy == :step
        return t < T/2 ? 0.0 : β_max  # Only guide in second half
    end
end
```

**Intuition**: Early steps are exploratory (low guidance), later steps refine (high guidance).

---

## The Efficiency Problem

### Naive Approach is Too Slow

The obvious implementation runs external models at every diffusion step:

```julia
# NAIVE: O(T * inference_cost) per generation
for t in T:-1:1
    logits = drafter(x_t, t)

    # EXPENSIVE: Full NLI inference for energy
    for token in vocabulary
        candidate = insert_token(x_t, token)
        energy[token] = nli_model(context, candidate)  # ~50ms each!
    end

    logits_guided = logits - β * energy
    x_{t-1} = sample(logits_guided)
end
```

**Cost analysis**:
- Diffusion steps T ≈ 50-100
- Vocabulary size V ≈ 32,000
- NLI inference ≈ 10-50ms per call
- Total: T × V × 50ms = **hours per generation** (completely impractical)

Even with top-k filtering (k=50):
- Total: T × k × 50ms = 50 × 50 × 50ms = **125 seconds** (still too slow)

### We Need Better Solutions

The key insight: **Move computation from inference to training**.

---

## Efficient Implementation Strategies

### Strategy 1: Train-Time Energy (Recommended)

**Core Idea**: Use energy as auxiliary loss during training, not at inference.

```julia
# TRAINING: Learn to generate logical thoughts
function train_step(drafter, x_clean, context)
    # Standard diffusion loss
    t = rand(1:T)
    x_noisy = add_noise(x_clean, t)
    logits = drafter(x_noisy, t)
    diffusion_loss = cross_entropy(logits, x_clean)

    # Energy-based auxiliary loss
    generated = sample(logits)
    energy = compute_logic_energy(generated, context)
    energy_loss = energy  # Minimize energy

    # Combined
    total_loss = diffusion_loss + λ * energy_loss
    return total_loss
end

# INFERENCE: No energy computation needed!
function generate(drafter, context)
    x = initialize_with_masks()
    for t in T:-1:1
        logits = drafter(x, t)  # Model already learned logic
        x = sample(logits)
    end
    return x
end
```

**Pros**:
- Zero inference overhead
- Model internalizes logical constraints
- Works with any energy function during training

**Cons**:
- Requires retraining
- Energy signal may be sparse/weak

### Strategy 2: Energy Head (Built-in Fast Approximation)

**Core Idea**: Add a small head to the drafter that predicts energy.

```julia
struct DrafterWithEnergyHead
    backbone::OssammaDrafter      # Main model
    energy_head::Chain            # Small MLP: hidden_dim → 1
end

function forward(model::DrafterWithEnergyHead, x, t)
    hidden, logits = model.backbone(x, t)  # Get hidden states too

    # Energy head predicts logic score
    energy = model.energy_head(mean(hidden, dims=2))  # Pool over sequence

    return logits, energy
end

# Training: Distill external NLI into energy head
function train_energy_head(model, x, context, nli_model)
    _, energy_pred = forward(model, x, t)

    # Target: what would NLI say?
    with_no_grad() do
        energy_true = nli_model(context, x)  # External model
    end

    loss = mse(energy_pred, energy_true)
    return loss
end

# Inference: Use cheap internal energy head
function generate_guided(model, context)
    x = initialize_with_masks()
    for t in T:-1:1
        logits, energy = forward(model, x, t)
        logits_guided = logits .- β * energy
        x = sample(logits_guided)
    end
    return x
end
```

**Architecture for Energy Head**:
```
hidden_states: (dim, seq_len, batch)
      ↓
Mean Pool over seq_len: (dim, batch)
      ↓
Dense(dim → dim/4) + ReLU
      ↓
Dense(dim/4 → 1)
      ↓
energy: (1, batch)
```

**Cost**: ~0.1% overhead (tiny MLP on existing hidden states)

### Strategy 3: Checkpoint-Based Guidance

**Core Idea**: Only compute energy at key checkpoints, not every step.

```julia
function generate_checkpoint_guided(drafter, context, nli_model)
    x = initialize_with_masks()

    # Define checkpoints (e.g., 4 checks for T=100)
    checkpoints = [100, 75, 50, 25, 10]  # Timesteps to check

    for t in T:-1:1
        logits = drafter(x, t)

        if t in checkpoints
            # Only compute expensive energy at checkpoints
            energy = compute_full_energy(x, context, nli_model)
            logits = logits .- β * energy

            # Increase guidance at later checkpoints
            β *= 1.5
        end

        x = sample(logits)
    end
    return x
end
```

**Cost reduction**: 100 steps → 5 energy computations (20x speedup)

### Strategy 4: End-Only Rejection Sampling

**Core Idea**: Generate full thought, check logic once, accept/reject.

```julia
function generate_with_rejection(drafter, context, logic_checker; max_attempts=5)
    for attempt in 1:max_attempts
        # Generate complete thought (no guidance)
        thought = generate(drafter, context)

        # Check logic once at the end
        is_logical = logic_checker(thought, context)

        if is_logical
            return thought, attempt
        end

        # Log rejection for analysis
        log_rejection(thought, attempt)
    end

    # Fallback: return best attempt or error
    return best_attempt, max_attempts
end
```

**Pros**:
- Simple to implement
- Full-power logic checking
- Clear success/failure signal

**Cons**:
- May waste computation on rejected generations
- Doesn't guide the generation process

### Strategy 5: Distilled Tiny Logic Model

**Core Idea**: Train a small, fast model to mimic large NLI.

```julia
# Teacher: Large NLI model (350M params, 50ms inference)
teacher = load_model("deberta-v3-large-mnli")

# Student: Tiny logic scorer (10M params, 1ms inference)
student = Chain(
    Embedding(vocab_size, 128),
    TransformerBlock(128, 4, 256) × 2,
    MeanPool(),
    Dense(128, 3)  # entail, neutral, contradict
)

# Distillation training
function distill_step(student, teacher, premise, hypothesis)
    # Teacher predictions (soft targets)
    teacher_logits = teacher(premise, hypothesis)
    teacher_probs = softmax(teacher_logits / temperature)

    # Student predictions
    student_logits = student(concat(premise, hypothesis))
    student_probs = softmax(student_logits / temperature)

    # KL divergence loss
    loss = kl_divergence(student_probs, teacher_probs)
    return loss
end
```

**Speed comparison**:
| Model | Params | Inference | Accuracy |
|-------|--------|-----------|----------|
| DeBERTa-v3-large | 350M | 50ms | 92% |
| Distilled-small | 10M | 1ms | 85% |
| Distilled-tiny | 3M | 0.3ms | 78% |

### Strategy 6: Cached Energy Patterns

**Core Idea**: Cache energy for common reasoning patterns.

```julia
struct EnergyCache
    pattern_cache::Dict{PatternHash, Float32}
    max_size::Int
    hit_count::Int
    miss_count::Int
end

function cached_energy(cache::EnergyCache, thought, context, nli_model)
    # Extract pattern (abstract away specific entities)
    pattern = extract_pattern(thought, context)
    hash = hash_pattern(pattern)

    if haskey(cache.pattern_cache, hash)
        cache.hit_count += 1
        return cache.pattern_cache[hash]
    end

    # Cache miss - compute and store
    cache.miss_count += 1
    energy = nli_model(context, thought)

    if length(cache.pattern_cache) < cache.max_size
        cache.pattern_cache[hash] = energy
    end

    return energy
end

# Pattern extraction example:
# "The variable X is null" → "The variable [VAR] is null"
# "Function foo returns int" → "Function [FUNC] returns [TYPE]"
```

### Recommended Combination

**Best practice: Strategy 1 + 2 together**

```julia
struct LogicalDrafter
    drafter::OssammaDrafter
    energy_head::Chain
end

# Phase 1: Train drafter with energy auxiliary loss
# Phase 2: Distill external NLI into energy head
# Phase 3: Inference uses internal energy head only

function train_logical_drafter(model, data, nli_model)
    for (x, context) in data
        # Diffusion loss
        t = rand(1:T)
        logits = model.drafter(add_noise(x, t), t)
        L_diffusion = diffusion_loss(logits, x)

        # Energy loss (external NLI during training)
        energy_true = nli_model(context, x)
        L_energy = energy_true  # Minimize

        # Energy head distillation loss
        _, energy_pred = forward(model, x, t)
        L_distill = mse(energy_pred, energy_true)

        L_total = L_diffusion + λ₁ * L_energy + λ₂ * L_distill
        backward(L_total)
    end
end

function generate(model::LogicalDrafter, context)
    x = initialize_with_masks()
    for t in T:-1:1
        logits, energy = forward(model, x, t)
        β = guidance_schedule(t)
        logits_guided = logits .- β .* energy
        x = sample(logits_guided)
    end
    return x
end
```

**Inference cost**: Same as base drafter + negligible energy head overhead.

---

## Approach 1: Entailment Energy

### Concept

Each reasoning step must logically **follow from** previous steps. Natural Language Inference (NLI) measures this relationship.

NLI classifies (premise, hypothesis) pairs into:
- **Entailment**: Hypothesis follows from premise
- **Neutral**: No clear logical relationship
- **Contradiction**: Hypothesis conflicts with premise

### Formal Definition

For a thought chain [t₁, t₂, ..., tₙ], the entailment energy of new thought tₙ₊₁ is:

```
E_entail(tₙ₊₁) = -log P(entails | context, tₙ₊₁)

Where context = concatenation of [t₁, t₂, ..., tₙ]
```

**Interpretation**:
- P(entails) ≈ 1.0 → E ≈ 0 (good, logically follows)
- P(entails) ≈ 0.1 → E ≈ 2.3 (bad, doesn't follow)
- P(entails) ≈ 0.01 → E ≈ 4.6 (very bad)

### Detailed Examples

**Example 1: Valid mathematical reasoning**
```
Context (premises):
  "Let n be a positive integer greater than 1"
  "n has no divisors other than 1 and itself"

Candidate thoughts:
  A: "n is a prime number"          → ENTAILS (E ≈ 0.1)
  B: "n is divisible by 3"          → CONTRADICTS (E ≈ 5.0)
  C: "n might be even"              → NEUTRAL (E ≈ 1.5)
```

**Example 2: Code reasoning**
```
Context:
  "The function get_user() returns Optional<User>"
  "We call user = get_user()"

Candidate thoughts:
  A: "user might be None"           → ENTAILS (E ≈ 0.2)
  B: "user.name is always valid"    → CONTRADICTS (E ≈ 4.0)
  C: "We should check if user exists" → ENTAILS (E ≈ 0.3)
```

### Implementation Details

```julia
using Transformers  # HuggingFace wrapper

struct EntailmentEnergy
    model::NLIModel
    tokenizer::Tokenizer
    max_length::Int

    # Caching for efficiency
    cache::LRUCache{String, Vector{Float32}}
end

function EntailmentEnergy(model_name="microsoft/deberta-v3-base-mnli")
    model = load_model(model_name)
    tokenizer = load_tokenizer(model_name)
    return EntailmentEnergy(model, tokenizer, 512, LRUCache(10000))
end

function compute_energy(ee::EntailmentEnergy, premise::String, hypothesis::String)
    # Check cache first
    cache_key = hash(premise * "|||" * hypothesis)
    if haskey(ee.cache, cache_key)
        return ee.cache[cache_key]
    end

    # Tokenize
    inputs = ee.tokenizer(
        premise, hypothesis,
        max_length=ee.max_length,
        truncation=true,
        padding=true
    )

    # Forward pass
    logits = ee.model(inputs)  # Shape: (3,) for entail/neutral/contradict
    probs = softmax(logits)

    # Energy = -log P(entails)
    energy = -log(probs[1] + 1e-10)  # Add epsilon for stability

    # Cache result
    ee.cache[cache_key] = energy

    return energy
end

function compute_chain_energy(ee::EntailmentEnergy, thought_chain::Vector{String})
    if length(thought_chain) < 2
        return 0.0
    end

    total_energy = 0.0

    # Check each thought against accumulated context
    context = thought_chain[1]
    for i in 2:length(thought_chain)
        new_thought = thought_chain[i]
        energy = compute_energy(ee, context, new_thought)
        total_energy += energy

        # Accumulate context
        context = context * " " * new_thought
    end

    return total_energy
end
```

### Batch Processing for Efficiency

```julia
function compute_energy_batch(ee::EntailmentEnergy,
                               premises::Vector{String},
                               hypotheses::Vector{String})
    # Batch tokenization
    inputs = ee.tokenizer(
        premises, hypotheses,
        max_length=ee.max_length,
        truncation=true,
        padding=true,
        return_tensors="pt"
    )

    # Batch forward pass
    logits = ee.model(inputs)  # Shape: (batch, 3)
    probs = softmax(logits, dims=2)

    # Energies for whole batch
    energies = -log.(probs[:, 1] .+ 1e-10)

    return energies
end
```

### Limitations and Mitigations

| Limitation | Mitigation |
|------------|------------|
| Sentence-level only | Chunk long contexts, use hierarchical attention |
| May miss implicit logic | Combine with symbolic approach |
| Sensitive to phrasing | Data augmentation during training |
| Computational cost | Use distillation (Strategy 5) |

---

## Approach 2: Contradiction Detector

### Concept

Explicitly detect and heavily penalize any contradictions between statements in the reasoning chain. Unlike entailment (which checks "does B follow from A?"), contradiction detection checks "do A and B conflict?"

### Why Separate from Entailment?

While NLI includes contradiction, having a dedicated detector allows:
- **Higher precision**: Fine-tuned specifically for contradictions
- **All-pairs checking**: Check every pair of statements, not just sequential
- **Stronger penalty**: Contradictions are worse than non-entailment

### Formal Definition

For a thought chain [t₁, t₂, ..., tₙ], contradiction energy is:

```
E_contra(chain) = Σᵢ Σⱼ₍ⱼ<ᵢ₎ w(i,j) * P(contradicts | tᵢ, tⱼ)

Where w(i,j) = weighting function (e.g., 1/|i-j| for recency bias)
```

### Detailed Examples

**Example 1: Direct contradiction**
```
t₁: "The API uses REST architecture"
t₂: "Requests are made via GraphQL queries"

Analysis:
  - REST and GraphQL are mutually exclusive paradigms
  - P(contradicts | t₁, t₂) ≈ 0.95
  - Energy contribution: high
```

**Example 2: Implicit contradiction**
```
t₁: "The list is guaranteed to be non-empty"
t₂: "We return early if the list has no elements"

Analysis:
  - If list is guaranteed non-empty, the early return never triggers
  - This is a logical inconsistency (dead code)
  - P(contradicts | t₁, t₂) ≈ 0.7
```

**Example 3: Temporal contradiction**
```
t₁: "The user has not logged in yet"
t₂: "We retrieve the user's saved preferences"

Analysis:
  - Can't retrieve preferences without login (typically)
  - P(contradicts | t₁, t₂) ≈ 0.6
```

### Implementation Details

```julia
struct ContradictionDetector
    model::NLIModel
    tokenizer::Tokenizer
    memory::Vector{String}      # Previous thoughts
    memory_window::Int          # How many to keep (for efficiency)
    contradiction_threshold::Float32  # Below this, don't penalize
end

function add_thought!(cd::ContradictionDetector, thought::String)
    push!(cd.memory, thought)

    # Sliding window for efficiency
    if length(cd.memory) > cd.memory_window
        popfirst!(cd.memory)
    end
end

function compute_energy(cd::ContradictionDetector, new_thought::String)
    if isempty(cd.memory)
        return 0.0
    end

    total_energy = 0.0

    for (i, old_thought) in enumerate(cd.memory)
        # Compute contradiction probability
        inputs = cd.tokenizer(old_thought, new_thought)
        logits = cd.model(inputs)
        probs = softmax(logits)

        p_contra = probs[3]  # Index 3 is typically contradiction

        if p_contra > cd.contradiction_threshold
            # Recency weighting: recent contradictions matter more
            recency_weight = 1.0 / (length(cd.memory) - i + 1)
            total_energy += recency_weight * p_contra * 10.0  # Heavy penalty
        end
    end

    return total_energy
end

function reset!(cd::ContradictionDetector)
    empty!(cd.memory)
end
```

### Bidirectional Checking

Contradiction is symmetric: if A contradicts B, then B contradicts A. But NLI models can be asymmetric in practice. Solution:

```julia
function compute_symmetric_contradiction(model, thought_a, thought_b)
    # Forward direction
    p_forward = get_contradiction_prob(model, thought_a, thought_b)

    # Backward direction
    p_backward = get_contradiction_prob(model, thought_b, thought_a)

    # Take maximum (if either direction detects contradiction)
    return max(p_forward, p_backward)
end
```

### Efficient All-Pairs Checking

Naive O(n²) checking is expensive. Optimizations:

```julia
function efficient_contradiction_check(detector, thoughts::Vector{String})
    n = length(thoughts)

    # Optimization 1: Batch all pairs
    pairs = [(thoughts[i], thoughts[j]) for i in 1:n for j in 1:i-1]

    if length(pairs) > 100
        # Optimization 2: Embedding-based pre-filtering
        embeddings = encode_batch(detector.encoder, thoughts)

        # Only check pairs with low cosine similarity (potential contradictions)
        suspicious_pairs = filter(pairs) do (a, b)
            i, j = indexof(a, thoughts), indexof(b, thoughts)
            cosine_sim(embeddings[i], embeddings[j]) < 0.3
        end

        pairs = suspicious_pairs
    end

    # Check filtered pairs
    energies = compute_batch_contradiction(detector, pairs)

    return sum(energies)
end
```

---

## Approach 3: Symbolic Logic Embedding

### Concept

Parse natural language into formal logic (First-Order Logic or Propositional Logic), then use SAT/SMT solvers to check satisfiability. This provides **hard guarantees** - if the solver says UNSAT, there's definitely a contradiction.

### Why Symbolic?

| Neural (NLI) | Symbolic (SAT/SMT) |
|--------------|---------------------|
| Fuzzy probabilities | Hard true/false |
| May miss edge cases | Complete within logic |
| Works on any text | Requires successful parsing |
| Fast once trained | Can be slow on complex formulas |

### The Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Natural    │────▶│   Semantic   │────▶│     FOL      │────▶│   SAT/SMT    │
│   Language   │     │    Parser    │     │   Formulas   │     │    Solver    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                      │
                     ┌────────────────────────────────────────────────┘
                     ▼
              ┌──────────────┐
              │ SAT: E = 0   │
              │ UNSAT: E = ∞ │
              └──────────────┘
```

### Detailed Parsing Examples

**Example 1: Quantified statements**
```
Natural: "All prime numbers greater than 2 are odd"
FOL:     ∀x. (Prime(x) ∧ x > 2) → Odd(x)

Natural: "There exists a prime number that is even"
FOL:     ∃x. Prime(x) ∧ Even(x)

Natural: "2 is prime and even"
FOL:     Prime(2) ∧ Even(2)

Combined check: SAT (the statements are consistent - 2 is the only even prime)
```

**Example 2: Conditional reasoning**
```
Natural: "If the input is null, we throw an exception"
FOL:     ∀x. Input(x) ∧ Null(x) → ThrowException(x)

Natural: "The function returns normally"
FOL:     ∃x. Input(x) ∧ ReturnsNormally(x)

Natural: "The input is null"
FOL:     ∃x. Input(x) ∧ Null(x)

Combined check:
  From 1 and 3: ThrowException triggered
  From 2: ReturnsNormally
  If ThrowException → ¬ReturnsNormally, then UNSAT (contradiction!)
```

**Example 3: Set relationships**
```
Natural: "The variable is either an integer or a string"
FOL:     Integer(x) ∨ String(x)

Natural: "If it's an integer, it must be positive"
FOL:     Integer(x) → Positive(x)

Natural: "The variable is not positive"
FOL:     ¬Positive(x)

Derivation:
  From 2 and 3: ¬Integer(x)  (by Modus Tollens)
  From 1 and ¬Integer(x): String(x)  (by Disjunctive Syllogism)

Result: SAT, and we can derive String(x)
```

### Implementation with Z3

```julia
using Z3  # Julia bindings for Z3 SMT solver

struct SymbolicLogicEnergy
    ctx::Z3.Context
    parser::SemanticParser  # NL → FOL
    formula_cache::Dict{String, Z3.Expr}
end

function parse_to_fol(sle::SymbolicLogicEnergy, text::String)
    # Use semantic parser (could be neural, rule-based, or LLM)
    fol_string = sle.parser(text)

    # Parse FOL string to Z3 expression
    expr = Z3.parse_smt2_string(sle.ctx, fol_string)

    return expr
end

function check_consistency(sle::SymbolicLogicEnergy, thoughts::Vector{String})
    solver = Z3.Solver(sle.ctx)

    for thought in thoughts
        try
            formula = parse_to_fol(sle, thought)
            Z3.add(solver, formula)
        catch e
            # Parsing failed - skip this thought for symbolic check
            @warn "Failed to parse: $thought"
            continue
        end
    end

    result = Z3.check(solver)

    if result == Z3.unsat
        # Contradiction found!
        # Optionally get unsat core (minimal contradicting subset)
        core = Z3.unsat_core(solver)
        return Inf, core
    else
        return 0.0, nothing
    end
end

function compute_energy(sle::SymbolicLogicEnergy,
                        thought_chain::Vector{String},
                        new_thought::String)
    all_thoughts = vcat(thought_chain, [new_thought])
    energy, unsat_core = check_consistency(sle, all_thoughts)

    if isinf(energy) && unsat_core !== nothing
        # Log which thoughts caused contradiction
        @warn "Contradiction found involving: $unsat_core"
    end

    return energy
end
```

### Differentiable Relaxation

SAT is discrete (true/false). For gradient-based training, we need a soft version.

**Fuzzy Logic Approach**:
```julia
# T-norms for conjunction (AND)
function fuzzy_and(a, b; method=:product)
    if method == :product
        return a * b
    elseif method == :lukasiewicz
        return max(0, a + b - 1)
    elseif method == :godel
        return min(a, b)
    end
end

# T-conorms for disjunction (OR)
function fuzzy_or(a, b; method=:product)
    if method == :product
        return a + b - a * b
    elseif method == :lukasiewicz
        return min(1, a + b)
    elseif method == :godel
        return max(a, b)
    end
end

# Negation
fuzzy_not(a) = 1 - a

# Implication: a → b ≡ ¬a ∨ b
fuzzy_implies(a, b) = fuzzy_or(fuzzy_not(a), b)

function evaluate_fuzzy_formula(formula, variable_values::Dict)
    # Recursively evaluate formula with fuzzy semantics
    if formula isa Variable
        return variable_values[formula.name]
    elseif formula isa And
        return fuzzy_and(
            evaluate_fuzzy_formula(formula.left, variable_values),
            evaluate_fuzzy_formula(formula.right, variable_values)
        )
    elseif formula isa Or
        return fuzzy_or(
            evaluate_fuzzy_formula(formula.left, variable_values),
            evaluate_fuzzy_formula(formula.right, variable_values)
        )
    elseif formula isa Not
        return fuzzy_not(evaluate_fuzzy_formula(formula.arg, variable_values))
    elseif formula isa Implies
        return fuzzy_implies(
            evaluate_fuzzy_formula(formula.antecedent, variable_values),
            evaluate_fuzzy_formula(formula.consequent, variable_values)
        )
    end
end

function soft_sat_energy(formulas::Vector, variable_values::Dict)
    # Energy = how far are we from satisfying all formulas?
    total_violation = 0.0

    for formula in formulas
        truth_value = evaluate_fuzzy_formula(formula, variable_values)
        violation = max(0, 1 - truth_value)  # 0 if true, positive if not
        total_violation += violation
    end

    return total_violation
end
```

### Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| NL→FOL parsing is hard | Use LLMs for parsing, fallback to neural if fails |
| Not all NL maps to FOL | Hybrid: symbolic for structured, neural for rest |
| Solver can be slow | Cache results, use incremental solving |
| Brittleness | Soft fallback to neural when parsing fails |

---

## Approach 4: Reasoning Graph Energy

### Concept

Build a graph where nodes are propositions and edges are logical relationships. Measure consistency by analyzing graph structure.

### Graph Components

**Nodes**: Extracted propositions from each thought
```
Thought: "The user is logged in and has admin privileges"
Nodes:
  - P1: "The user is logged in"
  - P2: "The user has admin privileges"
```

**Edges**: Logical relationships between propositions
```
@enum RelationType begin
    IMPLIES       # P1 → P2 (if P1 then P2)
    CONTRADICTS   # P1 ⊕ P2 (cannot both be true)
    SUPPORTS      # P1 provides evidence for P2 (weaker than implies)
    EQUIVALENT    # P1 ↔ P2 (same meaning)
    INDEPENDENT   # No logical relationship
end
```

### Detailed Graph Example

```
Reasoning trace:
  T1: "X is a positive integer"
  T2: "X is less than 10"
  T3: "X is prime"
  T4: "X is odd"
  T5: "X equals 6"  ← PROBLEMATIC

Graph:
              ┌─────────────────┐
              │ X is positive   │
              │    integer      │
              └────────┬────────┘
                       │ (context)
                       ▼
              ┌─────────────────┐
              │   X < 10        │
              └────────┬────────┘
                       │ (context)
                       ▼
              ┌─────────────────┐
    ┌─────────│   X is prime    │─────────┐
    │         └─────────────────┘         │
    │ IMPLIES                             │ IMPLIES
    │ (primes > 2                         │ (primes have
    │  are odd)                           │  no divisors)
    ▼                                     │
┌─────────────────┐                       │
│   X is odd      │                       │
└────────┬────────┘                       │
         │                                │
         │ CONTRADICTS                    │
         ▼                                │
┌─────────────────┐     CONTRADICTS       │
│   X = 6         │◀──────────────────────┘
└─────────────────┘
    (6 is even,
     6 is composite)

Energy calculation:
  - CONTRADICTS edge between "X is odd" and "X = 6": +10
  - CONTRADICTS edge between "X is prime" and "X = 6": +10
  - Total energy: 20 (high, reject this thought)
```

### Implementation Details

```julia
struct Proposition
    text::String
    embedding::Vector{Float32}  # For similarity matching
    source_thought::Int         # Which thought this came from
end

struct LogicalEdge
    source::Int
    target::Int
    relation::RelationType
    confidence::Float32
end

mutable struct ReasoningGraph
    nodes::Vector{Proposition}
    edges::Vector{LogicalEdge}

    # For efficient lookup
    node_index::Dict{String, Int}
    adjacency::Dict{Int, Vector{Int}}

    # Relation classifier (neural)
    relation_model::RelationClassifier

    # Proposition extractor (neural)
    prop_extractor::PropositionExtractor
end

function add_thought!(graph::ReasoningGraph, thought::String, thought_idx::Int)
    # Extract propositions from thought
    propositions = graph.prop_extractor(thought)

    new_node_indices = Int[]

    for prop_text in propositions
        # Create node
        embedding = encode(graph.relation_model.encoder, prop_text)
        prop = Proposition(prop_text, embedding, thought_idx)

        push!(graph.nodes, prop)
        node_idx = length(graph.nodes)
        push!(new_node_indices, node_idx)
        graph.node_index[prop_text] = node_idx
        graph.adjacency[node_idx] = Int[]
    end

    # Find relationships with existing nodes
    for new_idx in new_node_indices
        new_prop = graph.nodes[new_idx]

        for old_idx in 1:(new_idx - 1)
            old_prop = graph.nodes[old_idx]

            # Skip if same thought
            if old_prop.source_thought == thought_idx
                continue
            end

            # Classify relationship
            relation, confidence = classify_relation(
                graph.relation_model,
                old_prop.text,
                new_prop.text
            )

            if relation != INDEPENDENT && confidence > 0.5
                edge = LogicalEdge(old_idx, new_idx, relation, confidence)
                push!(graph.edges, edge)
                push!(graph.adjacency[old_idx], new_idx)
                push!(graph.adjacency[new_idx], old_idx)
            end
        end
    end

    return new_node_indices
end

function compute_energy(graph::ReasoningGraph)
    energy = 0.0

    # 1. Direct contradictions (highest penalty)
    for edge in graph.edges
        if edge.relation == CONTRADICTS
            energy += 10.0 * edge.confidence
        end
    end

    # 2. Transitive inconsistencies
    # Find: A IMPLIES B, B IMPLIES C, but A CONTRADICTS C
    energy += find_transitive_inconsistencies(graph) * 5.0

    # 3. Circular implications (potential issue)
    # A IMPLIES B, B IMPLIES C, C IMPLIES A (may indicate confused reasoning)
    cycles = find_implication_cycles(graph)
    energy += length(cycles) * 2.0

    # 4. Unsupported claims
    # Propositions with no incoming IMPLIES or SUPPORTS edges
    for (idx, node) in enumerate(graph.nodes)
        incoming = filter(e -> e.target == idx &&
                              e.relation in [IMPLIES, SUPPORTS], graph.edges)
        if isempty(incoming) && node.source_thought > 1
            # First thought can be premise, later thoughts need support
            energy += 1.0
        end
    end

    return energy
end

function find_transitive_inconsistencies(graph::ReasoningGraph)
    count = 0

    # For each node, find what it implies (transitively)
    for start_idx in 1:length(graph.nodes)
        implied = transitive_closure(graph, start_idx, IMPLIES)

        # Check if any implied node contradicts start
        for implied_idx in implied
            for edge in graph.edges
                if edge.source == start_idx &&
                   edge.target == implied_idx &&
                   edge.relation == CONTRADICTS
                    count += 1
                end
            end
        end
    end

    return count
end

function transitive_closure(graph::ReasoningGraph, start::Int, relation::RelationType)
    visited = Set{Int}()
    queue = [start]

    while !isempty(queue)
        current = popfirst!(queue)
        if current in visited
            continue
        end
        push!(visited, current)

        for edge in graph.edges
            if edge.source == current && edge.relation == relation
                push!(queue, edge.target)
            end
        end
    end

    delete!(visited, start)  # Don't include start node
    return visited
end
```

### Relation Classifier

```julia
struct RelationClassifier
    encoder::TransformerEncoder
    classifier::Chain
end

function classify_relation(rc::RelationClassifier, text_a::String, text_b::String)
    # Encode both texts
    emb_a = rc.encoder(text_a)
    emb_b = rc.encoder(text_b)

    # Create relation features
    features = vcat(
        emb_a,
        emb_b,
        emb_a .* emb_b,           # Element-wise product
        abs.(emb_a .- emb_b)      # Absolute difference
    )

    # Classify
    logits = rc.classifier(features)
    probs = softmax(logits)

    relation = RelationType(argmax(probs))
    confidence = maximum(probs)

    return relation, confidence
end
```

### Incremental Updates

For efficiency during generation:

```julia
function incremental_energy_update(graph::ReasoningGraph, new_thought::String, thought_idx::Int)
    # Only compute energy contribution of new nodes
    old_energy = graph.cached_energy

    new_indices = add_thought!(graph, new_thought, thought_idx)

    delta_energy = 0.0

    # Only check edges involving new nodes
    for edge in graph.edges
        if edge.source in new_indices || edge.target in new_indices
            if edge.relation == CONTRADICTS
                delta_energy += 10.0 * edge.confidence
            end
        end
    end

    # Check new transitive paths
    for new_idx in new_indices
        delta_energy += check_new_transitives(graph, new_idx) * 5.0
    end

    graph.cached_energy = old_energy + delta_energy
    return delta_energy
end
```

---

## Approach 5: Learned Logic Prior

### Concept

Train a model on valid reasoning traces to learn what "good reasoning" looks like. This is data-driven: instead of hand-coding logic rules, learn them from examples.

### Training Data Sources

| Source | Format | Volume | Quality |
|--------|--------|--------|---------|
| Mathematical proofs | Lean/Coq/Isabelle exports | 100K+ | Very high |
| Logic textbooks | Problem + solution | 10K | High |
| Chain-of-thought (filtered) | Prompt + reasoning + answer (correct only) | 1M+ | Medium |
| Debate corpora | Argument trees with validity labels | 50K | Medium |
| Natural Logic datasets | NLI with reasoning steps | 100K | High |
| Synthetic | Generated from templates | Unlimited | Varies |

### Positive Examples (Valid Reasoning)

**Example 1: Modus Ponens**
```json
{
  "premises": [
    "If a number is divisible by 4, it is divisible by 2",
    "16 is divisible by 4"
  ],
  "reasoning": [
    "We know 16 is divisible by 4",
    "The rule states: divisible by 4 implies divisible by 2",
    "Applying modus ponens: since 16 is divisible by 4, 16 must be divisible by 2"
  ],
  "conclusion": "16 is divisible by 2",
  "valid": true,
  "rule": "modus_ponens"
}
```

**Example 2: Proof by Contradiction**
```json
{
  "premises": [
    "Assume sqrt(2) is rational",
    "Then sqrt(2) = p/q for some integers p, q with no common factors"
  ],
  "reasoning": [
    "Squaring both sides: 2 = p²/q²",
    "Therefore p² = 2q²",
    "This means p² is even, so p must be even",
    "Let p = 2k for some integer k",
    "Then 4k² = 2q², so 2k² = q²",
    "This means q² is even, so q must be even",
    "But if both p and q are even, they have common factor 2",
    "This contradicts our assumption that p/q has no common factors"
  ],
  "conclusion": "sqrt(2) is irrational",
  "valid": true,
  "rule": "proof_by_contradiction"
}
```

### Negative Examples (Invalid Reasoning)

**Example 1: Affirming the Consequent (Fallacy)**
```json
{
  "premises": [
    "If it rains, the ground gets wet",
    "The ground is wet"
  ],
  "reasoning": [
    "We observe the ground is wet",
    "We know rain causes wet ground",
    "Therefore it must have rained"
  ],
  "conclusion": "It rained",
  "valid": false,
  "fallacy": "affirming_consequent",
  "explanation": "The ground could be wet for other reasons (sprinkler, spill)"
}
```

**Example 2: Non Sequitur**
```json
{
  "premises": [
    "The function is slow",
    "The function uses a database"
  ],
  "reasoning": [
    "The function is slow",
    "We should use a database",
    "Therefore we should add more RAM"
  ],
  "conclusion": "Add more RAM",
  "valid": false,
  "fallacy": "non_sequitur",
  "explanation": "The conclusion doesn't follow from the premises"
}
```

### Generating Negative Examples

```julia
function generate_negatives(valid_trace::ReasoningTrace)
    negatives = ReasoningTrace[]

    # Strategy 1: Shuffle steps (breaks logical flow)
    shuffled = deepcopy(valid_trace)
    shuffle!(shuffled.reasoning)
    shuffled.valid = false
    shuffled.fallacy = "incoherent_order"
    push!(negatives, shuffled)

    # Strategy 2: Remove crucial step
    for i in 1:length(valid_trace.reasoning)
        incomplete = deepcopy(valid_trace)
        deleteat!(incomplete.reasoning, i)
        incomplete.valid = false
        incomplete.fallacy = "missing_step"
        push!(negatives, incomplete)
    end

    # Strategy 3: Inject contradiction
    contradicted = deepcopy(valid_trace)
    random_step = rand(1:length(contradicted.reasoning))
    contradicted.reasoning[random_step] = negate(contradicted.reasoning[random_step])
    contradicted.valid = false
    contradicted.fallacy = "contradiction"
    push!(negatives, contradicted)

    # Strategy 4: Apply known fallacy
    for fallacy in [:affirming_consequent, :denying_antecedent, :circular]
        fallacious = apply_fallacy(valid_trace, fallacy)
        if fallacious !== nothing
            push!(negatives, fallacious)
        end
    end

    # Strategy 5: Wrong conclusion
    wrong_conclusion = deepcopy(valid_trace)
    wrong_conclusion.conclusion = generate_wrong_conclusion(valid_trace)
    wrong_conclusion.valid = false
    wrong_conclusion.fallacy = "wrong_conclusion"
    push!(negatives, wrong_conclusion)

    return negatives
end

function negate(statement::String)
    # Simple negation heuristics
    if startswith(statement, "not ") || startswith(statement, "Not ")
        return statement[5:end]
    elseif contains(statement, " is ")
        return replace(statement, " is " => " is not ")
    elseif contains(statement, " are ")
        return replace(statement, " are " => " are not ")
    else
        return "It is not the case that " * lowercase(statement)
    end
end
```

### Model Architecture

```julia
struct LogicPrior
    # Encoder for reasoning traces
    encoder::TransformerEncoder

    # Validity prediction head
    validity_head::Chain

    # Auxiliary heads (multi-task learning)
    fallacy_classifier::Chain      # What type of fallacy, if any
    step_importance::Chain         # Which steps are crucial

    # Configuration
    max_seq_length::Int
    hidden_dim::Int
end

function LogicPrior(;
    vocab_size=32000,
    hidden_dim=768,
    num_layers=6,
    num_heads=12,
    max_seq_length=1024
)
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_length=max_seq_length
    )

    validity_head = Chain(
        Dense(hidden_dim, hidden_dim ÷ 2, gelu),
        Dropout(0.1),
        Dense(hidden_dim ÷ 2, 1)
    )

    fallacy_classifier = Chain(
        Dense(hidden_dim, hidden_dim ÷ 2, gelu),
        Dropout(0.1),
        Dense(hidden_dim ÷ 2, NUM_FALLACY_TYPES)
    )

    step_importance = Chain(
        Dense(hidden_dim, hidden_dim ÷ 2, gelu),
        Dense(hidden_dim ÷ 2, 1)
    )

    return LogicPrior(encoder, validity_head, fallacy_classifier, step_importance,
                      max_seq_length, hidden_dim)
end

function forward(lp::LogicPrior, reasoning_trace::String)
    # Tokenize and encode
    hidden_states = lp.encoder(reasoning_trace)  # (hidden_dim, seq_len)

    # Pool for sequence-level prediction
    pooled = mean(hidden_states, dims=2)[:, 1]  # (hidden_dim,)

    # Validity prediction
    validity_logit = lp.validity_head(pooled)[1]
    validity_prob = sigmoid(validity_logit)

    # Fallacy classification (if invalid)
    fallacy_logits = lp.fallacy_classifier(pooled)

    # Step importance (per-token)
    step_scores = dropdims(lp.step_importance(hidden_states), dims=1)

    return (
        validity = validity_prob,
        fallacy_logits = fallacy_logits,
        step_importance = step_scores
    )
end

function compute_energy(lp::LogicPrior, reasoning_trace::String)
    result = forward(lp, reasoning_trace)

    # Energy = -log(validity probability)
    energy = -log(result.validity + 1e-10)

    return energy
end
```

### Training Objective

```julia
function train_step(lp::LogicPrior, batch)
    total_loss = 0.0

    for (trace, label) in batch
        result = forward(lp, trace)

        # Primary: Binary cross-entropy for validity
        validity_loss = binary_cross_entropy(result.validity, label.valid)

        # Auxiliary 1: Fallacy classification (only for invalid traces)
        fallacy_loss = 0.0
        if !label.valid && label.fallacy !== nothing
            fallacy_loss = cross_entropy(result.fallacy_logits, label.fallacy_idx)
        end

        # Auxiliary 2: Step importance (if annotations available)
        step_loss = 0.0
        if label.crucial_steps !== nothing
            step_loss = binary_cross_entropy(
                sigmoid.(result.step_importance),
                label.crucial_steps
            )
        end

        # Combined loss
        loss = validity_loss + 0.3 * fallacy_loss + 0.2 * step_loss
        total_loss += loss
    end

    return total_loss / length(batch)
end
```

### Contrastive Training

```julia
function contrastive_loss(lp::LogicPrior, valid_trace, invalid_trace; margin=1.0)
    e_valid = compute_energy(lp, valid_trace)
    e_invalid = compute_energy(lp, invalid_trace)

    # We want: e_valid < e_invalid by at least margin
    # Loss = max(0, margin - (e_invalid - e_valid))
    loss = max(0, margin - (e_invalid - e_valid))

    return loss
end

function train_contrastive(lp::LogicPrior, valid_traces)
    total_loss = 0.0

    for valid_trace in valid_traces
        # Generate negative examples
        negatives = generate_negatives(valid_trace)

        for negative in negatives
            loss = contrastive_loss(lp, valid_trace.text, negative.text)
            total_loss += loss
        end
    end

    return total_loss
end
```

---

## Hybrid Approach: Neuro-Symbolic Energy

### Combining Multiple Energies

Different approaches have complementary strengths:

| Approach | Strength | Weakness |
|----------|----------|----------|
| Entailment | Semantic understanding | May miss formal logic |
| Contradiction | Catches conflicts | Pairwise only |
| Symbolic | Hard guarantees | Parsing can fail |
| Graph | Multi-hop reasoning | Needs good relation classifier |
| Learned | Flexible, data-driven | Needs training data |

### Weighted Combination

```julia
struct HybridLogicalEnergy
    # Component energies
    entailment::EntailmentEnergy
    contradiction::ContradictionDetector
    symbolic::SymbolicLogicEnergy
    graph::ReasoningGraph
    learned::LogicPrior

    # Learnable weights
    weights::Vector{Float32}  # 5 weights, one per component

    # Fallback strategy
    fallback_order::Vector{Symbol}  # Which to use if others fail
end

function compute_energy(hle::HybridLogicalEnergy, thought_chain::Vector{String})
    energies = Float32[]
    mask = Bool[]  # Which components succeeded

    # 1. Entailment energy
    try
        e_ent = compute_chain_energy(hle.entailment, thought_chain)
        push!(energies, e_ent)
        push!(mask, true)
    catch
        push!(energies, 0.0)
        push!(mask, false)
    end

    # 2. Contradiction energy
    try
        e_contra = 0.0
        reset!(hle.contradiction)
        for thought in thought_chain
            e_contra += compute_energy(hle.contradiction, thought)
            add_thought!(hle.contradiction, thought)
        end
        push!(energies, e_contra)
        push!(mask, true)
    catch
        push!(energies, 0.0)
        push!(mask, false)
    end

    # 3. Symbolic energy
    try
        e_sym, _ = check_consistency(hle.symbolic, thought_chain)
        push!(energies, min(e_sym, 100.0))  # Cap infinite
        push!(mask, true)
    catch
        push!(energies, 0.0)
        push!(mask, false)
    end

    # 4. Graph energy
    try
        for (i, thought) in enumerate(thought_chain)
            add_thought!(hle.graph, thought, i)
        end
        e_graph = compute_energy(hle.graph)
        push!(energies, e_graph)
        push!(mask, true)
    catch
        push!(energies, 0.0)
        push!(mask, false)
    end

    # 5. Learned prior energy
    try
        trace_text = join(thought_chain, " [SEP] ")
        e_learned = compute_energy(hle.learned, trace_text)
        push!(energies, e_learned)
        push!(mask, true)
    catch
        push!(energies, 0.0)
        push!(mask, false)
    end

    # Weighted combination (only successful components)
    weights_masked = hle.weights .* mask
    weights_normalized = weights_masked ./ (sum(weights_masked) + 1e-10)

    total_energy = dot(energies, weights_normalized)

    return total_energy, (energies=energies, mask=mask)
end
```

### Learning the Weights

```julia
function learn_weights(hle::HybridLogicalEnergy, training_data)
    # training_data: [(thought_chain, is_valid), ...]

    optimizer = Adam(0.01)

    for epoch in 1:100
        total_loss = 0.0

        for (chain, is_valid) in training_data
            energy, _ = compute_energy(hle, chain)

            # Valid chains should have low energy
            # Invalid chains should have high energy
            target = is_valid ? 0.0 : 5.0
            loss = (energy - target)^2

            # Gradient w.r.t. weights
            grads = gradient(hle.weights) do w
                energies = [...]  # Recompute with current weights
                e = dot(energies, w)
                (e - target)^2
            end

            optimizer(hle.weights, grads)
            total_loss += loss
        end

        # Ensure weights stay positive
        hle.weights .= max.(hle.weights, 0.01)

        println("Epoch $epoch, Loss: $(total_loss / length(training_data))")
    end
end
```

### Adaptive Selection

```julia
function adaptive_energy(hle::HybridLogicalEnergy, thought_chain::Vector{String})
    # Different strategies based on content

    # Detect if thoughts contain formal/mathematical content
    is_formal = any(contains(t, r"∀|∃|→|∧|∨|¬|≡|⊢") for t in thought_chain) ||
                any(contains(t, r"\b(theorem|lemma|proof|QED)\b"i) for t in thought_chain)

    # Detect if thoughts are code-related
    is_code = any(contains(t, r"function|class|if|for|while|return") for t in thought_chain)

    if is_formal
        # Prioritize symbolic for formal reasoning
        weights = Float32[0.1, 0.1, 0.5, 0.2, 0.1]
    elseif is_code
        # Prioritize graph for code reasoning (tracks variable states)
        weights = Float32[0.2, 0.3, 0.1, 0.3, 0.1]
    else
        # Balanced for natural language reasoning
        weights = Float32[0.25, 0.25, 0.1, 0.2, 0.2]
    end

    # Temporarily override weights
    original_weights = copy(hle.weights)
    hle.weights .= weights

    energy, details = compute_energy(hle, thought_chain)

    # Restore weights
    hle.weights .= original_weights

    return energy, details
end
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

**Goal**: Get basic energy-guided diffusion working

1. **Add energy head to OssammaDrafter**
   ```julia
   # Modify src/Drafter.jl
   struct OssammaDrafter
       ...
       energy_head::Chain  # NEW
   end
   ```

2. **Implement simple entailment energy**
   - Use pre-trained NLI via PyCall/HuggingFace
   - Test on simple reasoning traces

3. **Training loop modification**
   ```julia
   loss = diffusion_loss + λ * energy_loss
   ```

### Phase 2: Efficiency (Week 3-4)

**Goal**: Make it fast enough for practical use

1. **Distill NLI into energy head**
   - Generate (input, energy) pairs using external NLI
   - Train energy head to predict these

2. **Implement checkpoint-based guidance**
   - Only compute full energy at key timesteps

3. **Add caching**
   - Cache embeddings and energy scores

### Phase 3: Sophistication (Week 5-6)

**Goal**: Add stronger logical guarantees

1. **Implement contradiction detector**
   - All-pairs checking with efficient pre-filtering

2. **Add reasoning graph**
   - Proposition extraction
   - Relation classification
   - Graph consistency checking

3. **Train logic prior**
   - Collect/generate reasoning traces
   - Train validity classifier

### Phase 4: Integration (Week 7-8)

**Goal**: Full hybrid system

1. **Combine all energies**
   - Implement weighted combination
   - Learn optimal weights

2. **Evaluation**
   - Measure logical consistency of generated thoughts
   - Compare rejection rates with AR verifier
   - Human evaluation of reasoning quality

3. **Optimization**
   - Profile and optimize bottlenecks
   - Consider ONNX/TensorRT for energy models

### Milestones

| Milestone | Criteria | Target |
|-----------|----------|--------|
| M1: Basic | Energy head training works | Week 2 |
| M2: Fast | <10% overhead vs base drafter | Week 4 |
| M3: Logical | 50% reduction in contradictions | Week 6 |
| M4: Production | Full system, documented | Week 8 |

---

## Evaluation Metrics

### Automatic Metrics

1. **Contradiction Rate**: % of generated chains with detected contradictions
2. **Entailment Score**: Average P(entails) across chain steps
3. **Rejection Rate**: % rejected by AR verifier (should decrease with good energy)
4. **SAT Rate**: % of chains satisfiable by symbolic solver

### Human Evaluation

1. **Logical Coherence**: 1-5 scale, does the reasoning flow logically?
2. **Conclusion Validity**: Does conclusion follow from premises?
3. **Step Necessity**: Are all steps necessary? Any missing?

---

## References

### Energy-Based Models
- LeCun et al., "A Tutorial on Energy-Based Learning" (2006)
- Du & Mordatch, "Implicit Generation and Generalization with Energy-Based Models" (2019)

### Guided Diffusion
- Ho & Salimans, "Classifier-Free Diffusion Guidance" (2022)
- Dhariwal & Nichol, "Diffusion Models Beat GANs" (2021)

### Logical Reasoning in NLP
- Clark et al., "Transformers as Soft Reasoners over Language" (2020)
- Tafjord et al., "ProofWriter" (2021)
- Creswell et al., "Selection-Inference" (2022)

### Neuro-Symbolic
- Garcez et al., "Neural-Symbolic Computing" (2019)
- Badreddine et al., "Logic Tensor Networks" (2022)
- Wang et al., "SATNet" (2019)

### Chain-of-Thought
- Wei et al., "Chain-of-Thought Prompting" (2022)
- Wang et al., "Self-Consistency Improves CoT" (2023)
- Kojima et al., "Zero-shot CoT" (2022)

### NLI and Entailment
- Bowman et al., "SNLI" (2015)
- Williams et al., "MultiNLI" (2018)
- He et al., "DeBERTa" (2021)

---

## Status

- [ ] Add energy head to OssammaDrafter
- [ ] Implement entailment energy with NLI model
- [ ] Implement contradiction detector
- [ ] Training with energy auxiliary loss
- [ ] Distill NLI into energy head
- [ ] Implement reasoning graph
- [ ] Collect/generate reasoning training data
- [ ] Train logic prior model
- [ ] Implement hybrid energy combination
- [ ] Evaluation on reasoning benchmarks
- [ ] Integration with TiDAR verification loop
