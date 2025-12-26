# Energy-Guided Diffusion for Logical Thought Generation

> Research notes on embedding "Laws of Thought" into the OssammaDrafter diffusion process.

## Motivation

The drafter generates "thought tokens" via diffusion. We want these thoughts to follow logical structure - not just be coherent text, but valid reasoning chains.

**Classical Laws of Thought** (Aristotle/Boole):
1. **Identity**: A = A
2. **Non-Contradiction**: not(A and not-A)
3. **Excluded Middle**: A or not-A

**Reasoning Rules**:
- **Modus Ponens**: P, P->Q therefore Q
- **Transitivity**: A->B, B->C therefore A->C
- **Consistency**: No contradictions across steps

## Core Idea: Energy-Guided Diffusion

Standard diffusion samples from learned distribution. Energy-guided diffusion biases sampling towards low-energy (logically valid) states:

```
logits_guided = logits - beta * gradient(E)
```

Where E(x) is an energy function that is HIGH for illogical thoughts and LOW for logical ones.

---

## Approach 1: Entailment Energy

**Concept**: Each thought step must logically follow from what came before. Use Natural Language Inference (NLI) to measure this.

### Example
```
Premise: "All mammals are warm-blooded. Whales are mammals."
Hypothesis (new thought): "Whales are warm-blooded."
-> NLI: ENTAILS (low energy)

Hypothesis (bad thought): "Whales are cold-blooded."
-> NLI: CONTRADICTS (high energy)
```

### Energy Function
```
E(thought_t) = -log P(entails | thoughts_1..t-1, thought_t)
```

### Implementation
```julia
# At each denoising step
for t in T:-1:1
    # Standard denoising
    logits = drafter(x_t, t)

    # Compute entailment score for each candidate token
    candidates = top_k(logits, k=50)
    for token in candidates
        hypothetical_thought = current_thought + token
        entailment_score = nli_model(context, hypothetical_thought)
        energy[token] = -log(entailment_score[:entails])
    end

    # Guide towards entailing tokens
    logits_guided = logits - beta * energy
    x_next = sample(logits_guided)
end
```

### Pros/Cons
- **Pros**: Captures semantic entailment, pre-trained NLI models available (DeBERTa-v3-large-mnli)
- **Cons**: NLI is sentence-level, may struggle with complex multi-step reasoning; computational cost per token

---

## Approach 2: Contradiction Detector

**Concept**: Explicitly detect and penalize contradictions between any pair of statements in the reasoning chain.

### Example
```
Thought 1: "The package uses Python 3.10"
Thought 2: "We need to upgrade from Python 2.7"
Thought 3: "The Python 3.10 syntax is required"  <- consistent
Thought 3': "This only works on Python 2"        <- contradicts Thought 1
```

### Energy Function
```
E(chain) = sum_i sum_j(j<i) P(contradicts | thought_i, thought_j)
```

### Implementation
```julia
struct ContradictionEnergy
    nli_model      # Fine-tuned for contradiction detection
    memory::Vector # Previous thoughts
end

function compute_energy(ce::ContradictionEnergy, new_thought)
    total_energy = 0.0
    for old_thought in ce.memory
        # Check if new thought contradicts any previous thought
        probs = ce.nli_model(old_thought, new_thought)
        total_energy += probs[:contradiction]
    end
    return total_energy
end
```

### Windowed Version (for efficiency)
```
E(thought_t) = sum_j(j=t-W to t-1) P(contradicts | thought_t, thought_j)
```
Only check last W thoughts (working memory window).

---

## Approach 3: Symbolic Logic Embedding

**Concept**: Parse natural language into First-Order Logic (FOL), then use SAT/SMT solvers to check satisfiability.

### Example
```
Natural Language -> FOL:
"All birds can fly"      -> forall x. Bird(x) -> CanFly(x)
"Penguins are birds"     -> forall x. Penguin(x) -> Bird(x)
"Penguins cannot fly"    -> forall x. Penguin(x) -> not CanFly(x)

SAT Check: UNSATISFIABLE (contradiction!)
Energy = infinity (or large penalty)
```

### Pipeline
```
Thought     ->  Semantic    ->  FOL         ->  SAT/SMT
Tokens          Parser          Formulas        Solver
                                                  |
                 Energy = {0 if SAT, inf if UNSAT}
```

### Soft Relaxation (differentiable)
```julia
# Instead of hard SAT, use fuzzy logic
function soft_sat_energy(formulas)
    # Convert to differentiable constraints
    # Use t-norms for conjunction: a AND b ~ a * b
    # Use t-conorms for disjunction: a OR b ~ a + b - a*b

    violations = 0.0
    for formula in formulas
        truth_value = evaluate_fuzzy(formula)
        violations += max(0, 1 - truth_value)  # Penalize if not fully true
    end
    return violations
end
```

### Tools
- Semantic parsing: AMR parsers, semantic role labeling
- SAT solvers: Z3, MiniSat
- Differentiable: SATNet, Logic Tensor Networks

---

## Approach 4: Reasoning Graph Energy

**Concept**: Build a graph of propositions and their logical relationships, then measure graph consistency.

### Graph Structure
```
         +---------------------+
         | "X is a prime > 2"  |
         +---------+-----------+
                   | IMPLIES
                   v
         +---------------------+
         | "X is odd"          |<--- SUPPORTS ---+
         +---------+-----------+                 |
                   | SUPPORTS            +-------+-------+
                   v                     | "X = 7"       |
         +---------------------+         +---------------+
         | "X is not even"     |
         +---------------------+
                   |
                   | CONTRADICTS (high energy!)
                   v
         +---------------------+
         | "X is divisible by 2"  <- Bad thought, penalized
         +---------------------+
```

### Implementation
```julia
struct ReasoningGraph
    nodes::Vector{Proposition}
    edges::Vector{LogicalRelation}  # (src, dst, type)
end

@enum RelationType IMPLIES CONTRADICTS SUPPORTS INDEPENDENT

function add_thought!(graph::ReasoningGraph, thought::String)
    # Extract propositions
    props = extract_propositions(thought)

    for prop in props
        push!(graph.nodes, prop)

        # Find relations to existing nodes
        for existing in graph.nodes[1:end-1]
            relation = classify_relation(existing, prop)
            push!(graph.edges, (existing, prop, relation))
        end
    end
end
```

### Energy Computation
```julia
function graph_energy(graph::ReasoningGraph)
    energy = 0.0

    # Direct contradictions
    for (src, dst, rel) in graph.edges
        if rel == CONTRADICTS
            energy += 10.0  # Heavy penalty
        end
    end

    # Transitive inconsistencies
    # If A->B and B->C but A contradicts C
    for path in find_implication_paths(graph)
        if contradicts(path.start, path.end)
            energy += 5.0
        end
    end

    # Missing support (floating claims)
    for node in graph.nodes
        if !has_support(graph, node) && !is_premise(node)
            energy += 1.0  # Mild penalty for unsupported claims
        end
    end

    return energy
end
```

---

## Approach 5: Learned Logic Prior

**Concept**: Train a model on valid reasoning traces to learn what "good reasoning" looks like.

### Training Data Sources
- Mathematical proofs (Lean, Coq exports)
- Logic textbook examples
- Chain-of-thought traces (filtered for correctness)
- Debate/argument corpora with validity labels

### Examples
```
Valid reasoning trace (Modus Ponens):
  P1: "If it rains, the ground is wet"
  P2: "It is raining"
  C:  "Therefore, the ground is wet"

Invalid reasoning trace (Affirming the Consequent - fallacy!):
  P1: "If it rains, the ground is wet"
  P2: "The ground is wet"
  C:  "Therefore, it is raining"
```

### Model Architecture
```julia
struct LogicPrior
    encoder::Transformer      # Encode reasoning context
    validity_head::Dense      # Predict: valid reasoning? (0-1)
end

function compute_energy(lp::LogicPrior, thought_chain)
    encoding = lp.encoder(thought_chain)
    validity = sigmoid(lp.validity_head(encoding))
    return -log(validity)  # Low validity -> high energy
end
```

### Training Objective
```
L = -E[log P(valid | valid_trace)] - E[log(1 - P(valid | invalid_trace))]
```

### Data Augmentation (generate invalid examples)
```julia
function generate_negative(valid_trace)
    # Strategies:
    # 1. Shuffle reasoning steps
    shuffled = shuffle(valid_trace.steps)

    # 2. Inject contradictions
    contradicted = inject_contradiction(valid_trace)

    # 3. Apply logical fallacies
    fallacious = apply_fallacy(valid_trace, :affirming_consequent)

    # 4. Remove crucial steps
    incomplete = remove_random_step(valid_trace)

    return [shuffled, contradicted, fallacious, incomplete]
end
```

---

## Hybrid Approach: Neuro-Symbolic Energy

Combine neural flexibility with symbolic guarantees:

```
+------------------------------------------------------------------+
|                    Thought Diffusion                              |
|                                                                   |
|  x_T --> Denoise --> x_{T-1} --> ... --> x_1 --> x_0             |
|              |                                                    |
|              v                                                    |
|     +-------------------------------------------+                 |
|     |         Energy Computation                |                 |
|     |                                           |                 |
|     |  E_total = a1 * E_entailment              |                 |
|     |          + a2 * E_contradiction           |                 |
|     |          + a3 * E_symbolic                |                 |
|     |          + a4 * E_graph                   |                 |
|     |          + a5 * E_learned                 |                 |
|     +-------------------------------------------+                 |
|              |                                                    |
|              v                                                    |
|     logits_guided = logits - beta * grad(E_total)                 |
|                                                                   |
+------------------------------------------------------------------+
```

### Implementation Sketch
```julia
struct LogicalDiffusionGuide
    entailment_model::NLIModel
    contradiction_detector::NLIModel
    semantic_parser::AMRParser
    sat_solver::Z3Context
    reasoning_graph::ReasoningGraph
    logic_prior::LogicPrior

    # Weights
    weights::NamedTuple{(:ent, :contra, :sym, :graph, :learned)}
end

function guide_diffusion_step(guide::LogicalDiffusionGuide,
                               logits, context, t)
    # Compute all energy components
    E_ent = entailment_energy(guide.entailment_model, context)
    E_con = contradiction_energy(guide.contradiction_detector, context)
    E_sym = symbolic_energy(guide.semantic_parser, guide.sat_solver, context)
    E_gra = graph_energy(guide.reasoning_graph)
    E_lea = learned_energy(guide.logic_prior, context)

    # Weighted combination
    E_total = guide.weights.ent * E_ent +
              guide.weights.contra * E_con +
              guide.weights.sym * E_sym +
              guide.weights.graph * E_gra +
              guide.weights.learned * E_lea

    # Guidance strength increases as t -> 0 (more confident near end)
    beta = guidance_schedule(t)

    return logits .- beta .* E_total
end
```

---

## Comparison

| Approach | Complexity | Effectiveness | Differentiable | Notes |
|----------|------------|---------------|----------------|-------|
| Entailment | Low | Medium | Yes | Quick start, use HuggingFace NLI |
| Contradiction | Low | Medium | Yes | Pairs well with entailment |
| Symbolic SAT | High | High (hard logic) | Partial | Guarantees but brittle parsing |
| Reasoning Graph | Medium | Medium-High | Yes | Good for multi-hop |
| Learned Prior | Medium | High | Yes | Needs good training data |

---

## Recommended Implementation Path

### Phase 1: Quick Win
1. **Entailment + Contradiction** using pre-trained NLI (DeBERTa-mnli)
2. Simple integration into diffusion loop
3. Measure: rejection rate, logical consistency scores

### Phase 2: Learned Reasoning
1. Collect/generate reasoning traces (math proofs, CoT data)
2. Train **Logic Prior** model
3. Fine-tune on domain-specific reasoning

### Phase 3: Symbolic Guarantees
1. Add **semantic parser** for key logical constructs
2. Integrate **Z3** for hard constraint checking
3. Use for high-stakes reasoning (proofs, legal, medical)

---

## References

- Classifier-Free Guidance (Ho & Salimans, 2022)
- NeuroLogic Decoding (Lu et al., 2021)
- Logic Tensor Networks (Badreddine et al., 2022)
- SATNet: Bridging deep learning and logical reasoning (Wang et al., 2019)
- Chain-of-Thought Prompting (Wei et al., 2022)
- Self-Consistency (Wang et al., 2023)

---

## Status

- [ ] Implement entailment energy with NLI model
- [ ] Implement contradiction detector
- [ ] Integrate into OssammaDrafter diffusion loop
- [ ] Collect reasoning trace training data
- [ ] Train logic prior model
- [ ] Evaluate logical consistency metrics
