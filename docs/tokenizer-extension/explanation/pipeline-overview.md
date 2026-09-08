---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "How the four tokenizer_extension steps exchange artifacts and hand a resized checkpoint to continued pretraining."
topics: ["Tokenizer Extension", "Architecture"]
tags: ["Explanation", "Tokenizer Extension"]
content:
  type: "Explanation"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Pipeline Overview

The `tokenizer_extension` category is a typed-artifact pipeline.
Each step declares in its `step.toml` the artifact type it consumes and the artifact type it produces, and the output directory of one step is the input path of the next.

## Steps and Artifacts

```{mermaid}
flowchart LR
    A[checkpoint_hf: base] --> B[extend]
    B --> C[tokenizer]
    C --> D[init_embeddings]
    A --> D
    D --> E[checkpoint_hf: resized]
    E --> F[pretrain/megatron_bridge]
    C --> G[evaluate]
    G --> H[eval_results: fertility]
    E --> I[eval_init]
    I --> J[eval_results: BPB]
```

| Step | Consumes | Produces | Compute |
|------|----------|----------|---------|
| `extend` | `checkpoint_hf` (only the tokenizer is read) and a text corpus | `tokenizer`: `output_dir/<method>/` with `summary.json` | CPU |
| `init_embeddings` | `tokenizer` and the base `checkpoint_hf` | `checkpoint_hf`: `output_dir/` with weights and tokenizer | GPU |
| `evaluate` | `tokenizer` | `eval_results`: fertility JSON | CPU |
| `eval_init` | one or more resized `checkpoint_hf` | `eval_results`: loss, perplexity, and BPB JSON | GPU |

The `extend` step trains one byte-pair encoding (BPE) on the corpus and splices the resulting tokens into the base tokenizer using the *arm* selected by `method`.
An arm is one of the three construction methods, `add`, `replace`, or `expand`, and a job builds exactly one arm.
See {doc}`extension-methods`.

The `init_embeddings` step attaches the extended tokenizer to the base model, resizes the input embedding and language-model head, initializes the new rows with the engine selected by `method`, and saves the result as a Hugging Face checkpoint.
See {doc}`embedding-initialization`.

The two evaluation steps are independent of each other.
`evaluate` scores a tokenizer alone; `eval_init` scores a model and therefore needs a GPU.
See {doc}`evaluation-metrics`.

## Hand-Off to Continued Pretraining

The `init_embeddings` output directory is the `hf_model_path` value for the `pretrain/megatron_bridge` step.
Post-CPT model evaluation and downstream benchmarks are not part of this category; use the `eval` step catalog described in {doc}`../../model-eval/index`.

## Design Choices

- Each step is self-contained. `step.py` is the executor and its supporting modules live beside it. The only shared modules are `languages.py` and `script_ranges.py` at the category root.
- The execution backend (Slurm, Lepton, or DGX Cloud) is chosen at run time by the environment profile that you pass with `--run` or `--batch`, not by the step. See {doc}`../how-to/run-on-a-cluster`.
- `language` is the single key that selects the corpus normalizer, the prune script for Replace, the auxiliary encoder, and the fastText vectors. Every one of those remains individually overridable, but an explicit override takes precedence over the profile. See {doc}`../how-to/add-a-language`.
- Missing optional dependencies are errors, not fallbacks. When `indic-nlp-library` or `fasttext-wheel` is required and absent, the step raises with the install command rather than training or initializing with a different procedure.

## Where to Read More

| Topic | Repository document |
|-------|---------------------|
| Category overview and quickstart | [`src/nemotron/steps/tokenizer_extension/guide.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/tokenizer_extension/guide.md) |
| Measured results and decision rules | [`src/nemotron/steps/tokenizer_extension/guidebook/README.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/tokenizer_extension/guidebook/README.md) |
| Datasets and hyperparameters behind the guidebook | [`src/nemotron/steps/tokenizer_extension/guidebook/REPRODUCIBILITY.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/tokenizer_extension/guidebook/REPRODUCIBILITY.md) |
