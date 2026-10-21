---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Remove training documents that near-duplicate a held-out split, then cut nested token-budget tiers from the decontaminated corpus."
topics: ["Curation", "How-To", "Decontamination", "Subset"]
tags: ["How-To", "Curation", "Decontamination", "Subset"]
content:
  type: "How-To"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Decontaminate and Subset a Corpus

Use this guide after filtering, when the corpus must be checked against a held-out evaluation split and cut into fixed-size tiers for a training comparison.
Run the two steps in this order: a tier cut before decontamination carries exactly the documents decontamination was asked to remove, and no later report says so.
The flow orders the steps and wires the paths automatically; this guide covers running them by hand.

## Prerequisites

- A filtered JSONL corpus with a stable `id` field so the report can identify matches from both splits.
- The held-out split in JSONL, with the same `id_field` and `text_field`.
- For the MinHash/LSH similarity pass, one GPU and `uv sync --extra curate-gpu`. Without a GPU, set `skip_similarity: true` and run the exact identity pass only.
- For token-counted tiers, network access to the tokenizer on the Hugging Face Hub, or a warm `token_cache`.

## Decontaminate

### Choose the Scope

The step detects whole-document near-duplicates.
It does not detect a short benchmark question embedded inside a long training document; that is substring contamination and needs a containment-oriented method.

| Setting | Meaning | Default |
| --- | --- | --- |
| `threshold` | Minimum exact Jaccard similarity that removes a candidate during the similarity pass | `0.8` |
| `shingle_kind` | Shingle unit for the exact verification; must match the MinHash generator | `char` |
| `normalization` | Text normalization applied before shingling and recorded in the report | NFC, case folding, whitespace collapse, punctuation stripping |
| `minhash.char_ngrams` | Shingle width for MinHash and for the exact verification | `24` |
| `skip_similarity` | Run only the exact source-identity pass | `false` |
| `shared_id_space` | Treat an equal `id` in both splits as the same document | `true` |

Set `shared_id_space: false` only when the two splits were built separately and both number from zero, so that an equal `id` is a collision rather than a match.
The step uses two independent removal rules:

- **Source-identity match**: The step removes the training document regardless of `threshold`.
- **Similarity match**: The step removes the training document only when its exact Jaccard similarity is at least `threshold`.

### Run the Step

```bash
uv run --no-sync nemotron steps run curate/decontamination -c default \
  train_glob="${PWD}/output/vi/filtered_jsonl/**/*.jsonl" \
  holdout_glob="${PWD}/data/holdout/**/*.jsonl" \
  output_dir="${PWD}/output/vi/decontaminated" \
  id_field=id threshold=0.8
```

On a CPU-only host add `skip_similarity=true`.
The report then states that near-duplicate overlap was not measured, rather than reporting none.

### Read the Report

`<output_dir>/decontamination_report.json` records the normalization, MinHash parameters, removal counts, limited evidence lists, and similarity pass status.
The aggregate counts include every detected match and removal.
The detailed evidence lists have these limits:

- Up to 50 shared source-identity groups.
- Up to 200 removed pairs with the highest similarity.
- Up to 50 candidate pairs that the step could not verify.

The decontaminated corpus is `<output_dir>/train_decontaminated.jsonl`.
The holdout is never modified.

## Subset

### Choose the Budgets

Every tier is produced from one plan in one run, so a smaller tier is contained in a larger one.
That property is what makes a filtering ablation interpretable: two policies compared at the same token budget differ only in the policy.

Nesting is guaranteed; filling the budget is not.
Tokens delivered are at most the budget, and the difference is reported per tier as `token_shortfall` and per stratum as `per_stratum_deviation`.
The shortfall is never made up by drawing from another stratum.

| Setting | Meaning | Default |
| --- | --- | --- |
| `token_budgets` | One tier per budget, sorted ascending | `[100000000, 500000000, 2000000000]` |
| `tokenizer.name` / `tokenizer.revision` | Tokenizer used to count tokens; the revision is required because counts from two revisions are not comparable | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` at a pinned commit |
| `tokenizer: null` | Count whitespace words instead; every artifact records `unit: words` | Not set |
| `source_field` | Stratification key; without it every shard path is its own source | `source` |
| `length_bands` | Upper edges of the length bands, in the budget unit | `[128, 512, 2048, 8192]` |
| `quality_score_field` | Optional `__<signal>` column whose deciles join the stratum key | `null` |
| `token_cache` | Counts cached by tokenizer, revision, `id`, and content digest | `./cache/subset/token_counts.json` |
| `seed` | Seed for the stable per-document ordering | `0` |

If `quality_score_field` is set and the column is absent from the data, the run fails rather than stratifying on less.
The only producer of `__<signal>` columns is `curate/nemo_curator` under `mode: annotate` or `mode: both` with a policy.

### Run the Step

Point `input_glob` at the decontaminated corpus:

```bash
uv run --no-sync nemotron steps run curate/subset -c default \
  input_glob="${PWD}/output/vi/decontaminated/train_decontaminated.jsonl" \
  output_dir="${PWD}/output/vi/subset" \
  id_field=id source_field=source \
  token_cache="${PWD}/cache/vi/token_counts.json"
```

### Read the Outputs

| Path | Content |
| --- | --- |
| `<output_dir>/plan.json` | The single plan every tier was cut from: strata, ordering seed, and per-tier prefix lengths |
| `<output_dir>/subset_report.json` | Per-tier token totals, `token_shortfall`, and `per_stratum_deviation` |
| `<output_dir>/budget_<budget>_<unit>/subset.jsonl` | One tier per budget, for example `budget_100000000_tokens/subset.jsonl` |

## Next Steps

- Field reference: {doc}`../reference/decontamination` and {doc}`../reference/subset`.
- Why the ordering matters: {doc}`../explanation/pipeline-artifacts`.
