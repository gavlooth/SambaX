# OSSAMMA-NER Production Training Plan

## Overview

This plan covers end-to-end training for a production-quality Named Entity Recognition system using the OSSAMMA architecture with dual-gating. The target is the RAG-optimized 9-label schema.

---

## Label Schema (19 classes)

| Entity Type | Description | Examples |
|-------------|-------------|----------|
| PERSON | Individual humans | "Albert Einstein", "Dr. Smith" |
| AGENCY | Organizations, companies, governments | "Google", "FDA", "United Nations" |
| PLACE | Geographic locations, addresses | "New York", "Mount Everest", "123 Main St" |
| ORGANISM | Animals, plants, microbes | "E. coli", "golden retriever", "oak tree" |
| EVENT | Wars, incidents, eras, occurrences | "World War II", "COVID-19 pandemic" |
| INSTRUMENT | Tools, products, devices | "iPhone", "CRISPR", "electron microscope" |
| WORK | Books, papers, films, datasets | "Nature", "ImageNet", "The Godfather" |
| DOMAIN | Sciences, methods, fields | "machine learning", "quantum physics" |
| MEASURE | Numbers, dates, money, quantities | "$500", "January 2024", "5.2 kg" |

BIO encoding: `O, B-PERSON, I-PERSON, B-AGENCY, I-AGENCY, ...` (19 total)

---

## Phase 1: Data Acquisition

### 1.1 Primary Datasets

#### A. OntoNotes 5.0 (Foundation)
```bash
# Requires LDC license - https://catalog.ldc.upenn.edu/LDC2013T19
# 1.7M tokens, 18 entity types
# Map to our schema:
#   PERSON -> PERSON
#   ORG -> AGENCY
#   GPE, LOC, FAC -> PLACE
#   EVENT -> EVENT
#   PRODUCT -> INSTRUMENT
#   WORK_OF_ART -> WORK
#   DATE, TIME, PERCENT, MONEY, QUANTITY, CARDINAL, ORDINAL -> MEASURE
#   NORP, LAW, LANGUAGE -> DOMAIN (contextual)
```

#### B. CoNLL-2003 (English)
```bash
# Classic NER benchmark - https://www.clips.uantwerpen.be/conll2003/ner/
# ~300K tokens
# wget https://data.deepai.org/conll2003.zip
# Map: PER->PERSON, ORG->AGENCY, LOC->PLACE, MISC->contextual
```

#### C. Few-NERD (Fine-grained)
```bash
# 188K sentences, 66 fine-grained types
# https://github.com/thunlp/Few-NERD
git clone https://github.com/thunlp/Few-NERD.git
# Provides hierarchical types that map well to our schema
```

#### D. WNUT-17 (Emerging Entities)
```bash
# Social media, novel entities - good for robustness
# https://github.com/leondz/emerging_entities_17
```

#### E. SciERC (Scientific)
```bash
# Scientific papers - excellent for DOMAIN, WORK, INSTRUMENT
# https://nlp.cs.washington.edu/sciIE/
```

#### F. BioNLP / BC5CDR (Biomedical)
```bash
# Critical for ORGANISM entities
# https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/
```

### 1.2 Data Download Script

```julia
# scripts/download_ner_data.jl
using Downloads
using JSON3
using ProgressMeter

const DATA_DIR = "data/ner"

function download_conll2003()
    url = "https://data.deepai.org/conll2003.zip"
    dest = joinpath(DATA_DIR, "conll2003.zip")
    mkpath(DATA_DIR)

    println("Downloading CoNLL-2003...")
    Downloads.download(url, dest)

    # Unzip
    run(`unzip -o $dest -d $(joinpath(DATA_DIR, "conll2003"))`)
end

function download_fewnerd()
    println("Downloading Few-NERD...")
    run(`git clone https://github.com/thunlp/Few-NERD.git $(joinpath(DATA_DIR, "fewnerd"))`)
end

function download_wnut17()
    println("Downloading WNUT-17...")
    urls = [
        "https://raw.githubusercontent.com/leondz/emerging_entities_17/master/wnut17train.conll",
        "https://raw.githubusercontent.com/leondz/emerging_entities_17/master/wnut17dev.conll",
        "https://raw.githubusercontent.com/leondz/emerging_entities_17/master/wnut17test.conll"
    ]
    dest_dir = joinpath(DATA_DIR, "wnut17")
    mkpath(dest_dir)

    for url in urls
        filename = basename(url)
        Downloads.download(url, joinpath(dest_dir, filename))
    end
end

function main()
    download_conll2003()
    download_fewnerd()
    download_wnut17()
    println("Download complete!")
end

main()
```

### 1.3 Label Mapping Configuration

```julia
# config/label_mapping.jl
const LABEL_MAPPING = Dict(
    # OntoNotes -> Our Schema
    "PERSON" => "PERSON",
    "ORG" => "AGENCY",
    "GPE" => "PLACE",
    "LOC" => "PLACE",
    "FAC" => "PLACE",
    "EVENT" => "EVENT",
    "PRODUCT" => "INSTRUMENT",
    "WORK_OF_ART" => "WORK",
    "DATE" => "MEASURE",
    "TIME" => "MEASURE",
    "PERCENT" => "MEASURE",
    "MONEY" => "MEASURE",
    "QUANTITY" => "MEASURE",
    "CARDINAL" => "MEASURE",
    "ORDINAL" => "MEASURE",
    "NORP" => "DOMAIN",      # Nationalities, religious, political groups
    "LAW" => "WORK",         # Laws, treaties
    "LANGUAGE" => "DOMAIN",

    # CoNLL -> Our Schema
    "PER" => "PERSON",
    "LOC" => "PLACE",
    "MISC" => nothing,       # Requires contextual mapping

    # Few-NERD fine-grained (examples)
    "person-actor" => "PERSON",
    "person-artist" => "PERSON",
    "person-scientist" => "PERSON",
    "organization-company" => "AGENCY",
    "organization-government" => "AGENCY",
    "location-city" => "PLACE",
    "location-country" => "PLACE",
    "event-war" => "EVENT",
    "event-disaster" => "EVENT",
    "product-software" => "INSTRUMENT",
    "product-weapon" => "INSTRUMENT",
    "art-film" => "WORK",
    "art-music" => "WORK",
    "other-biologything" => "ORGANISM",
    "other-disease" => "DOMAIN",
    "other-scientificterm" => "DOMAIN",
)
```

---

## Phase 2: Data Preprocessing

### 2.1 Unified Data Format

```julia
# src/data/NERDataset.jl
module NERDataset

using JSON3
using Random

struct NERSample
    tokens::Vector{String}
    labels::Vector{Int}      # Label IDs (1-19)
    token_ids::Vector{Int}   # Tokenizer output
    attention_mask::Vector{Bool}
end

struct NERDataLoader
    samples::Vector{NERSample}
    batch_size::Int
    shuffle::Bool
    current_idx::Int
end

function load_conll_format(filepath::String, label_map::Dict)
    samples = NERSample[]
    current_tokens = String[]
    current_labels = String[]

    for line in eachline(filepath)
        line = strip(line)
        if isempty(line)
            if !isempty(current_tokens)
                push!(samples, create_sample(current_tokens, current_labels, label_map))
                current_tokens = String[]
                current_labels = String[]
            end
        else
            parts = split(line)
            if length(parts) >= 2
                push!(current_tokens, parts[1])
                push!(current_labels, parts[end])
            end
        end
    end

    return samples
end

function create_sample(tokens, labels, label_map)
    mapped_labels = map_labels(labels, label_map)
    # Tokenization happens later with actual tokenizer
    return NERSample(tokens, mapped_labels, Int[], Bool[])
end

function map_labels(labels::Vector{String}, label_map::Dict)
    result = Int[]
    for label in labels
        if label == "O"
            push!(result, 1)  # O is always 1
        elseif startswith(label, "B-") || startswith(label, "I-")
            prefix = label[1:2]
            entity_type = label[3:end]
            mapped_type = get(label_map, entity_type, nothing)
            if mapped_type !== nothing
                new_label = prefix * mapped_type
                push!(result, LABEL_TO_ID[new_label])
            else
                push!(result, 1)  # Unknown -> O
            end
        else
            push!(result, 1)
        end
    end
    return result
end

end # module
```

### 2.2 Tokenization Strategy

```julia
# src/data/Tokenizer.jl
module Tokenizer

using BytePairEncoding
using JSON3

struct NERTokenizer
    bpe::BPE
    vocab::Dict{String, Int}
    vocab_size::Int
    pad_token_id::Int
    unk_token_id::Int
    cls_token_id::Int
    sep_token_id::Int
end

function load_tokenizer(vocab_path::String, merges_path::String)
    vocab = JSON3.read(read(vocab_path, String), Dict{String, Int})
    bpe = BPE(merges_path)

    return NERTokenizer(
        bpe,
        vocab,
        length(vocab),
        vocab["[PAD]"],
        vocab["[UNK]"],
        vocab["[CLS]"],
        vocab["[SEP]"]
    )
end

"""
Tokenize with label alignment for NER.
Handles subword tokenization by assigning first subword the label,
subsequent subwords get -100 (ignore in loss).
"""
function tokenize_with_labels(
    tokenizer::NERTokenizer,
    tokens::Vector{String},
    labels::Vector{Int};
    max_length::Int = 512
)
    token_ids = Int[tokenizer.cls_token_id]
    aligned_labels = Int[-100]  # CLS gets ignored

    for (token, label) in zip(tokens, labels)
        subwords = tokenize(tokenizer.bpe, token)
        subword_ids = [get(tokenizer.vocab, sw, tokenizer.unk_token_id) for sw in subwords]

        if length(token_ids) + length(subword_ids) >= max_length - 1
            break
        end

        append!(token_ids, subword_ids)
        push!(aligned_labels, label)  # First subword gets the label
        append!(aligned_labels, fill(-100, length(subword_ids) - 1))  # Rest ignored
    end

    push!(token_ids, tokenizer.sep_token_id)
    push!(aligned_labels, -100)

    # Pad
    pad_length = max_length - length(token_ids)
    append!(token_ids, fill(tokenizer.pad_token_id, pad_length))
    append!(aligned_labels, fill(-100, pad_length))

    attention_mask = [id != tokenizer.pad_token_id for id in token_ids]

    return token_ids, aligned_labels, attention_mask
end

end # module
```

### 2.3 Data Augmentation

```julia
# src/data/Augmentation.jl
module Augmentation

using Random

"""
Entity-aware augmentation strategies for NER.
"""

# 1. Entity Replacement - swap entities of same type
function replace_entities(tokens, labels, entity_bank::Dict{String, Vector{Vector{String}}})
    new_tokens = copy(tokens)
    i = 1
    while i <= length(labels)
        if startswith(ID_TO_LABEL[labels[i]], "B-")
            entity_type = ID_TO_LABEL[labels[i]][3:end]

            # Find entity span
            j = i + 1
            while j <= length(labels) && ID_TO_LABEL[labels[j]] == "I-$entity_type"
                j += 1
            end

            # Replace with random entity of same type (30% chance)
            if rand() < 0.3 && haskey(entity_bank, entity_type)
                replacement = rand(entity_bank[entity_type])
                new_tokens[i:j-1] = replacement
                # Adjust labels if length differs
            end
            i = j
        else
            i += 1
        end
    end
    return new_tokens
end

# 2. Mention Dropout - randomly drop entity mentions
function mention_dropout(tokens, labels; p::Float32 = 0.1f0)
    mask = ones(Bool, length(tokens))
    i = 1
    while i <= length(labels)
        if startswith(ID_TO_LABEL[labels[i]], "B-")
            j = i + 1
            while j <= length(labels) && startswith(ID_TO_LABEL[labels[j]], "I-")
                j += 1
            end
            if rand() < p
                mask[i:j-1] .= false
            end
            i = j
        else
            i += 1
        end
    end
    return tokens[mask], labels[mask]
end

# 3. Context Shuffling - shuffle non-entity tokens
function shuffle_context(tokens, labels; window::Int = 3)
    # Implementation preserves entity spans but shuffles surrounding context
    # ...
end

# 4. Synonym Replacement for non-entities
function synonym_replacement(tokens, labels, synonyms::Dict; p::Float32 = 0.1f0)
    new_tokens = copy(tokens)
    for i in eachindex(tokens)
        if labels[i] == 1 && rand() < p  # Only O tokens
            token_lower = lowercase(tokens[i])
            if haskey(synonyms, token_lower)
                new_tokens[i] = rand(synonyms[token_lower])
            end
        end
    end
    return new_tokens
end

end # module
```

---

## Phase 3: Pretraining (Masked Diffusion)

### 3.1 Pretraining Configuration

```julia
# config/pretrain_config.jl
const PRETRAIN_CONFIG = (
    # Model
    model_size = :base,  # :small (30M), :base (110M), :large (350M)
    vocab_size = 32000,
    max_sequence_length = 512,

    # Architecture (Base)
    embedding_dimension = 768,
    number_of_heads = 12,
    number_of_layers = 12,
    time_dimension = 256,

    # Diffusion
    num_timesteps = 1000,
    noise_schedule = :cosine,
    mask_ratio_min = 0.0,
    mask_ratio_max = 1.0,

    # Training
    batch_size = 32,
    gradient_accumulation_steps = 4,  # Effective batch = 128
    learning_rate = 1e-4,
    weight_decay = 0.01,
    warmup_steps = 10000,
    max_steps = 500000,

    # Hardware
    precision = :bf16,
    num_gpus = 4,

    # Checkpointing
    checkpoint_every = 5000,
    eval_every = 1000,
)
```

### 3.2 Pretraining Data

For diffusion pretraining, use large unlabeled corpora:

```julia
# scripts/prepare_pretrain_data.jl

# 1. Wikipedia (English) - ~16GB
# Download from https://dumps.wikimedia.org/

# 2. BookCorpus - ~5GB
# Available through HuggingFace datasets

# 3. OpenWebText - ~38GB
# https://skylion007.github.io/OpenWebTextCorpus/

# 4. Domain-specific corpora for better NER:
#    - PubMed abstracts (biomedical)
#    - arXiv papers (scientific)
#    - News articles (events, people, places)

const PRETRAIN_CORPORA = [
    ("wikipedia", "data/pretrain/wikipedia/"),
    ("bookcorpus", "data/pretrain/bookcorpus/"),
    ("openwebtext", "data/pretrain/openwebtext/"),
    ("pubmed", "data/pretrain/pubmed/"),
    ("arxiv", "data/pretrain/arxiv/"),
]
```

### 3.3 Pretraining Script

```julia
# scripts/pretrain.jl
using Ossamma
using Lux
using Optimisers
using Zygote
using CUDA
using ProgressMeter
using TensorBoardLogger

function pretrain(config)
    # Initialize model
    model = LLaDAModel(LLaDAConfig(
        vocab_size = config.vocab_size,
        max_sequence_length = config.max_sequence_length,
        embedding_dimension = config.embedding_dimension,
        number_of_heads = config.number_of_heads,
        number_of_layers = config.number_of_layers,
        time_dimension = config.time_dimension,
    ))

    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    ps = ps |> gpu
    st = st |> gpu

    # Optimizer with warmup
    opt = Optimisers.AdamW(config.learning_rate, (0.9, 0.999), config.weight_decay)
    opt_state = Optimisers.setup(opt, ps)

    # Learning rate schedule
    lr_schedule = warmup_cosine_schedule(
        config.learning_rate,
        config.warmup_steps,
        config.max_steps
    )

    # Data loader
    dataloader = create_pretrain_dataloader(config)

    # Logging
    logger = TBLogger("logs/pretrain")

    # Training loop
    step = 0
    @showprogress for epoch in 1:1000
        for batch in dataloader
            step += 1
            if step > config.max_steps
                break
            end

            # Update learning rate
            current_lr = lr_schedule(step)
            Optimisers.adjust!(opt_state, current_lr)

            # Forward + backward
            batch = batch |> gpu
            loss, grads = Zygote.withgradient(ps) do p
                diffusion_loss(model, batch, p, st)
            end

            # Gradient accumulation
            if step % config.gradient_accumulation_steps == 0
                opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
            end

            # Logging
            if step % 100 == 0
                log_value(logger, "train/loss", loss, step=step)
                log_value(logger, "train/lr", current_lr, step=step)
            end

            # Checkpointing
            if step % config.checkpoint_every == 0
                save_checkpoint("checkpoints/pretrain_step_$step.jls", ps, st, opt_state, step)
            end
        end
    end

    return ps, st
end
```

---

## Phase 4: NER Fine-tuning

### 4.1 Fine-tuning Configuration

```julia
# config/finetune_config.jl
const FINETUNE_CONFIG = (
    # Model (load from pretrained)
    pretrained_checkpoint = "checkpoints/pretrain_best.jls",

    # NER specific
    num_labels = 19,
    use_crf = true,
    boundary_loss_weight = 0.2,

    # Training
    batch_size = 16,
    gradient_accumulation_steps = 2,
    learning_rate = 2e-5,
    crf_learning_rate = 1e-3,  # CRF needs higher LR
    epochs = 20,
    warmup_ratio = 0.1,

    # Regularization
    dropout = 0.1,
    label_smoothing = 0.1,
    max_grad_norm = 1.0,

    # Early stopping
    patience = 5,
    min_delta = 0.001,

    # Evaluation
    eval_every_epoch = true,
    eval_metric = :f1_micro,
)
```

### 4.2 CRF Layer Implementation

```julia
# src/CRF.jl
module CRF

using Lux
using NNlib
using Random

"""
Linear-chain CRF for BIO sequence labeling.
Enforces valid transitions (e.g., O cannot follow I-PERSON directly).
"""
struct LinearChainCRF <: Lux.AbstractLuxLayer
    num_labels::Int

    function LinearChainCRF(num_labels::Int)
        new(num_labels)
    end
end

function Lux.initialparameters(rng::Random.AbstractRNG, crf::LinearChainCRF)
    n = crf.num_labels
    # Transition matrix: transitions[i,j] = score of transitioning from label i to label j
    transitions = randn(rng, Float32, n, n) * 0.1f0

    # Initialize invalid transitions to large negative value
    # Invalid: O -> I-*, I-X -> I-Y where X != Y
    for i in 1:n
        for j in 1:n
            if !is_valid_transition(i, j)
                transitions[i, j] = -10000.0f0
            end
        end
    end

    return (transitions = transitions,)
end

function is_valid_transition(from_label::Int, to_label::Int)
    from_str = ID_TO_LABEL[from_label]
    to_str = ID_TO_LABEL[to_label]

    # O can go to O or B-*
    if from_str == "O"
        return to_str == "O" || startswith(to_str, "B-")
    end

    # B-X can go to O, B-*, or I-X
    if startswith(from_str, "B-")
        entity_type = from_str[3:end]
        return to_str == "O" || startswith(to_str, "B-") || to_str == "I-$entity_type"
    end

    # I-X can go to O, B-*, or I-X
    if startswith(from_str, "I-")
        entity_type = from_str[3:end]
        return to_str == "O" || startswith(to_str, "B-") || to_str == "I-$entity_type"
    end

    return true
end

Lux.initialstates(::Random.AbstractRNG, ::LinearChainCRF) = (;)

"""
Compute log-likelihood of label sequence given emissions.
"""
function log_likelihood(crf::LinearChainCRF, emissions, labels, mask, params, state)
    # emissions: (num_labels, seq_len, batch)
    # labels: (seq_len, batch)
    # mask: (seq_len, batch)

    batch_size = size(emissions, 3)
    seq_len = size(emissions, 2)

    # Score of gold sequence
    gold_score = compute_gold_score(emissions, labels, mask, params.transitions)

    # Partition function (sum over all possible sequences)
    log_Z = compute_log_partition(emissions, mask, params.transitions)

    # Negative log-likelihood
    nll = log_Z .- gold_score

    return mean(nll), state
end

"""
Viterbi decoding to find best label sequence.
"""
function decode(crf::LinearChainCRF, emissions, mask, params, state)
    # emissions: (num_labels, seq_len, batch)
    batch_size = size(emissions, 3)
    seq_len = size(emissions, 2)
    num_labels = size(emissions, 1)

    best_paths = zeros(Int, seq_len, batch_size)

    for b in 1:batch_size
        # Viterbi forward pass
        viterbi = zeros(Float32, num_labels, seq_len)
        backpointers = zeros(Int, num_labels, seq_len)

        viterbi[:, 1] = emissions[:, 1, b]

        for t in 2:seq_len
            if !mask[t, b]
                break
            end
            for j in 1:num_labels
                scores = viterbi[:, t-1] .+ params.transitions[:, j] .+ emissions[j, t, b]
                best_prev = argmax(scores)
                viterbi[j, t] = scores[best_prev]
                backpointers[j, t] = best_prev
            end
        end

        # Backtrack
        seq_end = sum(mask[:, b])
        best_last = argmax(viterbi[:, seq_end])
        best_paths[seq_end, b] = best_last

        for t in (seq_end-1):-1:1
            best_paths[t, b] = backpointers[best_paths[t+1, b], t+1]
        end
    end

    return best_paths, state
end

export LinearChainCRF, log_likelihood, decode

end # module
```

### 4.3 Fine-tuning Script

```julia
# scripts/finetune_ner.jl
using Ossamma
using Ossamma.NER
using Ossamma.CRF
using Lux
using Optimisers
using Zygote
using CUDA
using Statistics
using ProgressMeter

function finetune_ner(config)
    # Load pretrained model
    println("Loading pretrained model...")
    pretrained = load_checkpoint(config.pretrained_checkpoint)

    # Create NER model with OssammaNERBlock
    model = create_ner_model(config, pretrained)

    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    # Initialize encoder from pretrained
    ps = transfer_encoder_weights(ps, pretrained.params)
    ps = ps |> gpu
    st = st |> gpu

    # Separate optimizer for CRF (higher LR)
    encoder_opt = Optimisers.AdamW(config.learning_rate)
    crf_opt = Optimisers.AdamW(config.crf_learning_rate)

    opt_state = (
        encoder = Optimisers.setup(encoder_opt, ps.encoder),
        crf = Optimisers.setup(crf_opt, ps.crf),
        head = Optimisers.setup(encoder_opt, ps.head),
    )

    # Data
    train_loader, val_loader, test_loader = create_ner_dataloaders(config)

    # Training loop
    best_f1 = 0.0
    patience_counter = 0

    for epoch in 1:config.epochs
        println("\n=== Epoch $epoch ===")

        # Training
        train_loss = train_epoch!(model, train_loader, ps, st, opt_state, config)
        println("Train loss: $(round(train_loss, digits=4))")

        # Validation
        val_metrics = evaluate_ner(model, val_loader, ps, st)
        println("Val F1: $(round(val_metrics.f1_micro, digits=4))")
        println("Val Precision: $(round(val_metrics.precision, digits=4))")
        println("Val Recall: $(round(val_metrics.recall, digits=4))")

        # Per-entity metrics
        println("\nPer-entity F1:")
        for (entity, f1) in val_metrics.per_entity_f1
            println("  $entity: $(round(f1, digits=4))")
        end

        # Early stopping
        if val_metrics.f1_micro > best_f1 + config.min_delta
            best_f1 = val_metrics.f1_micro
            patience_counter = 0
            save_checkpoint("checkpoints/ner_best.jls", ps, st, epoch, val_metrics)
            println("New best model saved!")
        else
            patience_counter += 1
            if patience_counter >= config.patience
                println("Early stopping triggered")
                break
            end
        end
    end

    # Final test evaluation
    println("\n=== Test Evaluation ===")
    ps, st, _, _ = load_checkpoint("checkpoints/ner_best.jls")
    test_metrics = evaluate_ner(model, test_loader, ps, st)

    println("Test F1: $(round(test_metrics.f1_micro, digits=4))")
    println("Test Precision: $(round(test_metrics.precision, digits=4))")
    println("Test Recall: $(round(test_metrics.recall, digits=4))")

    return ps, st, test_metrics
end

function train_epoch!(model, dataloader, ps, st, opt_state, config)
    total_loss = 0.0
    num_batches = 0

    for (batch_idx, batch) in enumerate(dataloader)
        batch = batch |> gpu

        # Forward + backward
        loss, grads = Zygote.withgradient(ps) do p
            compute_ner_loss(model, batch, p, st, config)
        end

        # Gradient clipping
        grads = clip_gradients(grads, config.max_grad_norm)

        # Update
        opt_state.encoder, ps_encoder = Optimisers.update(opt_state.encoder, ps.encoder, grads[1].encoder)
        opt_state.crf, ps_crf = Optimisers.update(opt_state.crf, ps.crf, grads[1].crf)
        opt_state.head, ps_head = Optimisers.update(opt_state.head, ps.head, grads[1].head)

        ps = (encoder = ps_encoder, crf = ps_crf, head = ps_head)

        total_loss += loss
        num_batches += 1
    end

    return total_loss / num_batches
end

function compute_ner_loss(model, batch, ps, st, config)
    token_ids, labels, attention_mask = batch

    # Forward pass
    emissions, st = model.encoder(token_ids, ps.encoder, st.encoder)
    logits, st = model.head(emissions, ps.head, st.head)

    # CRF loss
    crf_loss, _ = CRF.log_likelihood(model.crf, logits, labels, attention_mask, ps.crf, st.crf)

    # Boundary detection auxiliary loss
    boundary_labels = compute_boundary_labels(labels)
    boundary_logits, _ = model.boundary_head(emissions, ps.boundary_head, st.boundary_head)
    boundary_loss = NNlib.logitcrossentropy(boundary_logits, boundary_labels)

    # Combined loss
    total_loss = crf_loss + config.boundary_loss_weight * boundary_loss

    return total_loss
end
```

---

## Phase 5: Evaluation Metrics

### 5.1 NER Evaluation Module

```julia
# src/evaluation/NERMetrics.jl
module NERMetrics

using Statistics

struct NERResults
    f1_micro::Float64
    f1_macro::Float64
    precision::Float64
    recall::Float64
    per_entity_f1::Dict{String, Float64}
    per_entity_precision::Dict{String, Float64}
    per_entity_recall::Dict{String, Float64}
    confusion_matrix::Matrix{Int}
end

"""
Compute span-level F1 (strict matching).
An entity is correct only if both boundary AND type match exactly.
"""
function compute_span_f1(predictions::Vector{Vector{Int}},
                         gold::Vector{Vector{Int}},
                         label_map::Dict{Int, String})

    true_positives = Dict{String, Int}()
    false_positives = Dict{String, Int}()
    false_negatives = Dict{String, Int}()

    for entity_type in ENTITY_TYPES
        true_positives[entity_type] = 0
        false_positives[entity_type] = 0
        false_negatives[entity_type] = 0
    end

    for (pred_seq, gold_seq) in zip(predictions, gold)
        pred_entities = extract_entities(pred_seq, label_map)
        gold_entities = extract_entities(gold_seq, label_map)

        # Match entities
        for gold_ent in gold_entities
            if gold_ent in pred_entities
                true_positives[gold_ent.type] += 1
            else
                false_negatives[gold_ent.type] += 1
            end
        end

        for pred_ent in pred_entities
            if pred_ent ∉ gold_entities
                false_positives[pred_ent.type] += 1
            end
        end
    end

    # Compute metrics
    per_entity_f1 = Dict{String, Float64}()
    per_entity_precision = Dict{String, Float64}()
    per_entity_recall = Dict{String, Float64}()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for entity_type in ENTITY_TYPES
        tp = true_positives[entity_type]
        fp = false_positives[entity_type]
        fn = false_negatives[entity_type]

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)

        per_entity_precision[entity_type] = precision
        per_entity_recall[entity_type] = recall
        per_entity_f1[entity_type] = f1
    end

    # Micro F1
    micro_precision = total_tp / max(total_tp + total_fp, 1)
    micro_recall = total_tp / max(total_tp + total_fn, 1)
    micro_f1 = 2 * micro_precision * micro_recall / max(micro_precision + micro_recall, 1e-10)

    # Macro F1
    macro_f1 = mean(values(per_entity_f1))

    return NERResults(
        micro_f1,
        macro_f1,
        micro_precision,
        micro_recall,
        per_entity_f1,
        per_entity_precision,
        per_entity_recall,
        zeros(Int, 19, 19)  # Confusion matrix computed separately
    )
end

function extract_entities(label_ids::Vector{Int}, label_map::Dict{Int, String})
    entities = Set{NamedTuple{(:type, :start, :end_), Tuple{String, Int, Int}}}()

    i = 1
    while i <= length(label_ids)
        label = label_map[label_ids[i]]
        if startswith(label, "B-")
            entity_type = label[3:end]
            start_idx = i
            i += 1

            # Find entity end
            while i <= length(label_ids) && label_map[label_ids[i]] == "I-$entity_type"
                i += 1
            end

            push!(entities, (type = entity_type, start = start_idx, end_ = i - 1))
        else
            i += 1
        end
    end

    return entities
end

export NERResults, compute_span_f1

end # module
```

### 5.2 Target Metrics for Production

| Metric | Target | Notes |
|--------|--------|-------|
| Overall F1 (micro) | > 0.90 | Primary metric |
| Overall F1 (macro) | > 0.85 | Ensures all entity types work |
| PERSON F1 | > 0.95 | Most common, must be excellent |
| AGENCY F1 | > 0.90 | Second most common |
| PLACE F1 | > 0.92 | Geographic entities |
| MEASURE F1 | > 0.88 | Numbers, dates |
| ORGANISM F1 | > 0.85 | Domain-specific |
| EVENT F1 | > 0.82 | Challenging |
| WORK F1 | > 0.80 | Titles are hard |
| DOMAIN F1 | > 0.78 | Abstract concepts |
| INSTRUMENT F1 | > 0.80 | Products, tools |
| Inference latency | < 50ms | For 512 tokens |
| Throughput | > 100 seq/s | Batch processing |

---

## Phase 6: Production Deployment

### 6.1 Model Export

```julia
# scripts/export_model.jl
using Ossamma
using Serialization
using SHA

function export_for_production(checkpoint_path::String, output_dir::String)
    # Load best checkpoint
    ps, st, epoch, metrics = load_checkpoint(checkpoint_path)

    # Remove training-only components
    ps_inference = remove_dropout_params(ps)
    st_inference = set_inference_mode(st)

    # Save model weights
    weights_path = joinpath(output_dir, "model_weights.jls")
    serialize(weights_path, (params = ps_inference, state = st_inference))

    # Save config
    config_path = joinpath(output_dir, "config.json")
    save_config(config_path)

    # Compute checksum
    checksum = bytes2hex(sha256(read(weights_path)))

    # Save metadata
    metadata = Dict(
        "model_type" => "OssammaNER",
        "version" => "2.0.0",
        "num_labels" => 19,
        "label_schema" => RAG_LABELS,
        "checkpoint_epoch" => epoch,
        "val_f1" => metrics.f1_micro,
        "checksum" => checksum,
        "exported_at" => string(now()),
    )

    open(joinpath(output_dir, "metadata.json"), "w") do f
        JSON3.pretty(f, metadata)
    end

    println("Model exported to $output_dir")
    println("Checksum: $checksum")
end
```

### 6.2 Inference Server

```julia
# src/serve/InferenceServer.jl
module InferenceServer

using HTTP
using JSON3
using Ossamma
using Ossamma.NER
using CUDA

struct NERServer
    model::OssammaNER
    params::NamedTuple
    state::NamedTuple
    tokenizer::NERTokenizer
    batch_size::Int
    max_length::Int
end

function load_server(model_dir::String; device = :gpu)
    # Load model
    weights = deserialize(joinpath(model_dir, "model_weights.jls"))
    config = load_config(joinpath(model_dir, "config.json"))

    model = OssammaNER(config)
    ps = weights.params
    st = weights.state

    if device == :gpu
        ps = ps |> gpu
        st = st |> gpu
    end

    tokenizer = load_tokenizer(joinpath(model_dir, "tokenizer"))

    return NERServer(model, ps, st, tokenizer, 32, 512)
end

function predict(server::NERServer, texts::Vector{String})
    # Tokenize
    batch = tokenize_batch(server.tokenizer, texts, server.max_length)
    batch = batch |> gpu

    # Inference
    CUDA.@sync begin
        logits, _ = server.model(batch.token_ids, server.params, server.state)
        predictions, _ = CRF.decode(server.model.crf, logits, batch.attention_mask,
                                     server.params.crf, server.state.crf)
    end

    # Decode labels
    results = []
    for (i, text) in enumerate(texts)
        tokens = server.tokenizer.decode(batch.token_ids[:, i])
        labels = [ID_TO_LABEL[p] for p in predictions[:, i]]
        entities = extract_entities(tokens, labels)
        push!(results, Dict(
            "text" => text,
            "entities" => entities,
            "tokens" => tokens,
            "labels" => labels
        ))
    end

    return results
end

function start_server(server::NERServer; port::Int = 8080)
    router = HTTP.Router()

    HTTP.register!(router, "POST", "/predict") do req
        body = JSON3.read(String(req.body))
        texts = body["texts"]

        results = predict(server, texts)

        return HTTP.Response(200, JSON3.write(results))
    end

    HTTP.register!(router, "GET", "/health") do req
        return HTTP.Response(200, JSON3.write(Dict("status" => "healthy")))
    end

    println("Starting NER server on port $port")
    HTTP.serve(router, "0.0.0.0", port)
end

end # module
```

### 6.3 Monitoring & Logging

```julia
# src/serve/Monitoring.jl
module Monitoring

using Prometheus
using Statistics

# Metrics
const REQUEST_COUNTER = Counter("ner_requests_total", "Total NER requests")
const REQUEST_LATENCY = Histogram("ner_request_latency_seconds", "Request latency",
                                   buckets = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0])
const ENTITY_COUNTER = CounterVec("ner_entities_total", "Entities extracted",
                                   labels = ["entity_type"])
const BATCH_SIZE = Histogram("ner_batch_size", "Batch sizes",
                              buckets = [1, 2, 4, 8, 16, 32])

function record_request(latency::Float64, entities::Vector, batch_size::Int)
    inc(REQUEST_COUNTER)
    observe(REQUEST_LATENCY, latency)
    observe(BATCH_SIZE, batch_size)

    for entity in entities
        inc(ENTITY_COUNTER, labels = Dict("entity_type" => entity.type))
    end
end

# Alert thresholds
const LATENCY_THRESHOLD_P99 = 0.1  # 100ms
const ERROR_RATE_THRESHOLD = 0.01  # 1%

function check_alerts(metrics)
    alerts = []

    if metrics.latency_p99 > LATENCY_THRESHOLD_P99
        push!(alerts, "High latency: P99 = $(metrics.latency_p99)s")
    end

    if metrics.error_rate > ERROR_RATE_THRESHOLD
        push!(alerts, "High error rate: $(metrics.error_rate * 100)%")
    end

    return alerts
end

end # module
```

---

## Phase 7: Training Schedule

### Week 1-2: Data Preparation
- [ ] Download all datasets
- [ ] Implement label mapping
- [ ] Create unified data format
- [ ] Implement tokenization with label alignment
- [ ] Data augmentation pipeline
- [ ] Train/val/test splits

### Week 3-4: Pretraining
- [ ] Set up distributed training infrastructure
- [ ] Run masked diffusion pretraining (~500K steps)
- [ ] Monitor loss curves, adjust hyperparameters
- [ ] Select best checkpoint

### Week 5-6: Fine-tuning
- [ ] Implement CRF layer
- [ ] Fine-tune on combined NER datasets
- [ ] Hyperparameter search (LR, dropout, label smoothing)
- [ ] Ablation studies on dual gating

### Week 7: Evaluation & Analysis
- [ ] Comprehensive evaluation on all test sets
- [ ] Error analysis by entity type
- [ ] Failure case analysis
- [ ] Performance profiling

### Week 8: Production Deployment
- [ ] Model export and optimization
- [ ] Inference server setup
- [ ] Load testing
- [ ] Monitoring setup
- [ ] Documentation

---

## Appendix: Hyperparameter Search Space

```julia
const HP_SEARCH_SPACE = (
    learning_rate = [1e-5, 2e-5, 3e-5, 5e-5],
    crf_learning_rate = [5e-4, 1e-3, 2e-3],
    dropout = [0.0, 0.1, 0.2],
    label_smoothing = [0.0, 0.05, 0.1],
    warmup_ratio = [0.05, 0.1, 0.15],
    batch_size = [8, 16, 32],
    window_size = [128, 256, 512],  # For sliding window attention
)
```

---

## Appendix: Expected Results

Based on similar architectures and the dual-gating modification:

| Dataset | Expected F1 | SOTA F1 |
|---------|-------------|---------|
| CoNLL-2003 | 93.5 | 94.6 |
| OntoNotes 5.0 | 90.0 | 92.1 |
| Few-NERD | 70.0 | 72.3 |
| WNUT-17 | 52.0 | 55.2 |

The dual-gating mechanism should particularly help with:
- Long-range entity dependencies (through GLU global context)
- Entity boundary detection (through input gating)
- Rare entity types (through output gate injection)
