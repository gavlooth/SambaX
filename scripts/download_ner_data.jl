#!/usr/bin/env julia
"""
NER Dataset Download Script

Downloads and prepares standard NER datasets for training:
1. CoNLL-2003 (English) - Classic NER benchmark
2. WNUT-17 - Emerging entities from social media
3. Few-NERD - Fine-grained NER (requires git)
4. MultiNERD - Multilingual NER (via HuggingFace)

Usage:
    julia --project=. scripts/download_ner_data.jl [--all|--conll|--wnut|--fewnerd|--multinerd]
"""

using Downloads
using ProgressMeter
using JSON3

const DATA_DIR = joinpath(@__DIR__, "..", "data", "ner")
const CACHE_DIR = joinpath(DATA_DIR, ".cache")

# =============================================================================
# Utility Functions
# =============================================================================

function ensure_dir(path::String)
    if !isdir(path)
        mkpath(path)
        println("Created directory: $path")
    end
end

function download_with_progress(url::String, dest::String; description::String = "Downloading")
    if isfile(dest)
        println("  Already exists: $dest")
        return dest
    end

    println("  $description: $url")

    try
        Downloads.download(url, dest)
        println("  Saved to: $dest")
    catch e
        println("  Error downloading: $e")
        return nothing
    end

    return dest
end

# =============================================================================
# CoNLL-2003
# =============================================================================

"""
Download CoNLL-2003 English NER dataset.
Note: Original requires LDC license. Using publicly available mirror.
"""
function download_conll2003()
    println("\n" * "=" ^ 50)
    println("Downloading CoNLL-2003...")
    println("=" ^ 50)

    dest_dir = joinpath(DATA_DIR, "conll2003")
    ensure_dir(dest_dir)

    # Public mirrors (may have licensing restrictions for commercial use)
    urls = Dict(
        "train" => "https://raw.githubusercontent.com/davidsbatista/NER-datasets/master/CONLL2003/train.txt",
        "valid" => "https://raw.githubusercontent.com/davidsbatista/NER-datasets/master/CONLL2003/valid.txt",
        "test" => "https://raw.githubusercontent.com/davidsbatista/NER-datasets/master/CONLL2003/test.txt",
    )

    for (split, url) in urls
        dest = joinpath(dest_dir, "$split.txt")
        download_with_progress(url, dest; description = "CoNLL-2003 $split")
    end

    # Create info file
    info = Dict(
        "name" => "CoNLL-2003",
        "description" => "Classic NER benchmark with PER, ORG, LOC, MISC",
        "format" => "conll",
        "labels" => ["PER", "ORG", "LOC", "MISC"],
        "source" => "https://www.clips.uantwerpen.be/conll2003/ner/",
        "license" => "Research use only - requires original LDC license for commercial use",
    )

    open(joinpath(dest_dir, "info.json"), "w") do f
        JSON3.pretty(f, info)
    end

    println("CoNLL-2003 download complete!")
    return dest_dir
end

# =============================================================================
# WNUT-17
# =============================================================================

"""
Download WNUT-17 (Workshop on Noisy User-generated Text) dataset.
Contains emerging entities from social media.
"""
function download_wnut17()
    println("\n" * "=" ^ 50)
    println("Downloading WNUT-17...")
    println("=" ^ 50)

    dest_dir = joinpath(DATA_DIR, "wnut17")
    ensure_dir(dest_dir)

    base_url = "https://raw.githubusercontent.com/leondz/emerging_entities_17/master"

    files = Dict(
        "train" => "wnut17train.conll",
        "dev" => "wnut17dev.conll",
        "test" => "wnut17test.conll",
    )

    for (split, filename) in files
        url = "$base_url/$filename"
        dest = joinpath(dest_dir, "$split.conll")
        download_with_progress(url, dest; description = "WNUT-17 $split")
    end

    # Create info file
    info = Dict(
        "name" => "WNUT-17",
        "description" => "Emerging and rare entities from social media",
        "format" => "conll",
        "labels" => ["person", "location", "corporation", "group", "creative-work", "product"],
        "source" => "https://noisy-text.github.io/2017/emerging-rare-entities.html",
        "license" => "CC BY 4.0",
    )

    open(joinpath(dest_dir, "info.json"), "w") do f
        JSON3.pretty(f, info)
    end

    println("WNUT-17 download complete!")
    return dest_dir
end

# =============================================================================
# Few-NERD
# =============================================================================

"""
Download Few-NERD dataset via git clone.
Fine-grained NER with 66 entity types.
"""
function download_fewnerd()
    println("\n" * "=" ^ 50)
    println("Downloading Few-NERD...")
    println("=" ^ 50)

    dest_dir = joinpath(DATA_DIR, "fewnerd")

    if isdir(dest_dir) && isfile(joinpath(dest_dir, "data", "supervised", "train.txt"))
        println("  Few-NERD already downloaded")
        return dest_dir
    end

    # Clone the repository
    println("  Cloning Few-NERD repository...")
    try
        if isdir(dest_dir)
            rm(dest_dir, recursive=true)
        end
        run(`git clone --depth 1 https://github.com/thunlp/Few-NERD.git $dest_dir`)
    catch e
        println("  Error cloning Few-NERD: $e")
        println("  Make sure git is installed and accessible")
        return nothing
    end

    # Create info file
    info = Dict(
        "name" => "Few-NERD",
        "description" => "Fine-grained NER with 66 types in 8 coarse categories",
        "format" => "conll",
        "data_path" => "data/supervised/",
        "labels" => "66 fine-grained types (see taxonomy)",
        "source" => "https://github.com/thunlp/Few-NERD",
        "license" => "MIT",
    )

    open(joinpath(dest_dir, "info.json"), "w") do f
        JSON3.pretty(f, info)
    end

    println("Few-NERD download complete!")
    return dest_dir
end

# =============================================================================
# MultiNERD (via HuggingFace)
# =============================================================================

"""
Download MultiNERD dataset from HuggingFace.
Requires Python with datasets library.
"""
function download_multinerd()
    println("\n" * "=" ^ 50)
    println("Downloading MultiNERD...")
    println("=" ^ 50)

    dest_dir = joinpath(DATA_DIR, "multinerd")
    ensure_dir(dest_dir)

    # Check if already downloaded
    if isfile(joinpath(dest_dir, "train.jsonl"))
        println("  MultiNERD already downloaded")
        return dest_dir
    end

    # Create Python download script
    python_script = """
import json
from datasets import load_dataset

print("Loading MultiNERD from HuggingFace...")
dataset = load_dataset("Babelscape/multinerd", "en")

label_names = dataset["train"].features["ner_tags"].feature.names

for split in ["train", "validation", "test"]:
    print(f"Processing {split}...")
    with open(f"$dest_dir/{split}.jsonl", "w") as f:
        for example in dataset[split]:
            tokens = example["tokens"]
            labels = [label_names[t] for t in example["ner_tags"]]
            json.dump({"tokens": tokens, "labels": labels}, f)
            f.write("\\n")

# Save label info
with open("$dest_dir/labels.json", "w") as f:
    json.dump({"labels": label_names}, f, indent=2)

print("MultiNERD download complete!")
"""

    script_path = joinpath(CACHE_DIR, "download_multinerd.py")
    ensure_dir(CACHE_DIR)
    write(script_path, python_script)

    println("  Running Python download script...")
    println("  (Requires: pip install datasets)")

    try
        run(`python3 $script_path`)
    catch e
        println("  Error running Python script: $e")
        println("  Make sure Python 3 and 'datasets' package are installed:")
        println("    pip install datasets")
        return nothing
    end

    # Create info file
    info = Dict(
        "name" => "MultiNERD",
        "description" => "Multilingual NER with 15 entity types",
        "format" => "jsonl",
        "labels" => [
            "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
            "B-ANIM", "I-ANIM", "B-BIO", "I-BIO", "B-CEL", "I-CEL",
            "B-DIS", "I-DIS", "B-EVE", "I-EVE", "B-FOOD", "I-FOOD",
            "B-INST", "I-INST", "B-MEDIA", "I-MEDIA", "B-MYTH", "I-MYTH",
            "B-PLANT", "I-PLANT", "B-TIME", "I-TIME", "B-VEHI", "I-VEHI"
        ],
        "source" => "https://huggingface.co/datasets/Babelscape/multinerd",
        "license" => "CC BY-SA 4.0",
    )

    open(joinpath(dest_dir, "info.json"), "w") do f
        JSON3.pretty(f, info)
    end

    return dest_dir
end

# =============================================================================
# OntoNotes 5.0 Instructions
# =============================================================================

"""
Print instructions for obtaining OntoNotes 5.0 (requires LDC license).
"""
function print_ontonotes_instructions()
    println("\n" * "=" ^ 50)
    println("OntoNotes 5.0 Instructions")
    println("=" ^ 50)
    println("""
OntoNotes 5.0 requires a license from the Linguistic Data Consortium (LDC).

To obtain OntoNotes 5.0:
1. Visit: https://catalog.ldc.upenn.edu/LDC2013T19
2. Request access (free for LDC members, fee for non-members)
3. Download and extract to: $(joinpath(DATA_DIR, "ontonotes"))

After downloading, run the preprocessing script:
    julia --project=. scripts/preprocess_ontonotes.jl

OntoNotes provides:
- 1.7M tokens with 18 entity types
- High-quality annotations
- Multiple genres (news, broadcast, web, etc.)
""")
end

# =============================================================================
# Dataset Statistics
# =============================================================================

"""
Print statistics about downloaded datasets.
"""
function print_dataset_stats()
    println("\n" * "=" ^ 50)
    println("Downloaded Dataset Statistics")
    println("=" ^ 50)

    datasets = ["conll2003", "wnut17", "fewnerd", "multinerd"]

    for name in datasets
        dir = joinpath(DATA_DIR, name)
        if isdir(dir)
            # Count files
            files = readdir(dir)
            data_files = filter(f -> endswith(f, ".txt") || endswith(f, ".conll") || endswith(f, ".jsonl"), files)

            # Load info if available
            info_path = joinpath(dir, "info.json")
            if isfile(info_path)
                info = JSON3.read(read(info_path, String))
                println("\n$name:")
                println("  Format: $(get(info, :format, "unknown"))")
                println("  Files: $(length(data_files))")
                println("  License: $(get(info, :license, "unknown"))")
            else
                println("\n$name: $(length(data_files)) files")
            end
        else
            println("\n$name: Not downloaded")
        end
    end
end

# =============================================================================
# Main
# =============================================================================

function main()
    println("NER Dataset Download Script")
    println("Data directory: $DATA_DIR")

    ensure_dir(DATA_DIR)
    ensure_dir(CACHE_DIR)

    # Parse arguments
    args = ARGS
    if isempty(args)
        args = ["--all"]
    end

    if "--all" in args || "--conll" in args
        download_conll2003()
    end

    if "--all" in args || "--wnut" in args
        download_wnut17()
    end

    if "--all" in args || "--fewnerd" in args
        download_fewnerd()
    end

    if "--all" in args || "--multinerd" in args
        download_multinerd()
    end

    if "--ontonotes" in args
        print_ontonotes_instructions()
    end

    print_dataset_stats()

    println("\n" * "=" ^ 50)
    println("Download complete!")
    println("=" ^ 50)
    println("\nTo use these datasets in training:")
    println("  using Ossamma.NERDataset")
    println("  samples = load_dataset(\"$(joinpath(DATA_DIR, "conll2003", "train.txt"))\")")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
