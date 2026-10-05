---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Reference for the curate/decontamination step: parameters, artifacts, strategies, and errors."
topics: ["Curation", "Reference", "Decontamination"]
tags: ["Reference", "Curation", "Steps"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Data Scientist"]
---

# curate/decontamination

The `curate/decontamination` step removes training documents that near-duplicate a held-out split.
It combines an exact source-identity pass with MinHash/LSH candidate generation verified by exact Jaccard similarity.
Only the training split shrinks; the holdout is never modified.

The step detects whole-document near-duplicates only.
A benchmark question embedded inside a long document is substring contamination and is out of scope.

## Syntax

```bash
uv run --no-sync nemotron steps run curate/decontamination \
    [-c <config-name-or-path>] \
    [-d] \
    [<dotlist-overrides>...]
```

Refer to the [Nemotron Steps CLI Reference](../../train-models/reference/cli-reference.md) for the shared flag set.

The similarity pass calls GPU-backed NeMo Curator MinHash and LSH stages and requires the `curate-gpu` extra and a GPU.
With `skip_similarity: true`, the exact source-identity pass runs on the CPU-only `curate` extra.

## Configuration Files

The step ships two configuration files under `src/nemotron/steps/curate/nemo_curator/decontamination/config/`.

| File | Purpose |
| --- | --- |
| `default.yaml` | Full run with the similarity pass enabled, character shingles, Jaccard threshold 0.8, and NFC, case-fold, whitespace, and punctuation normalization. |
| `tiny.yaml` | Identity-only run (`skip_similarity: true`) over packaged train and holdout fixtures. The paths are container paths; override them for local runs. |

## Inputs and Outputs

| Direction | Artifact type | Content |
| --- | --- | --- |
| Consumes | `filtered_jsonl` | The training split to decontaminate and the held-out split to protect. Both carry a stable document identifier. |
| Produces | `decontaminated_jsonl` | `train_decontaminated.jsonl`: the training split with overlapping documents removed. |
| Produces | `decontamination_report` | `decontamination_report.json`: removal counts, normalization and shingling settings, and limited evidence lists. |

The aggregate counts include every detected match and removal.
The detailed evidence lists have these limits:

- Up to 50 shared source-identity groups.
- Up to 200 removed pairs with the highest similarity.
- Up to 50 candidate pairs that the step could not verify.

## Parameters

```{option} train_glob

The training split.
This is the only split that shrinks.
```

```{option} holdout_glob

The held-out split to protect.
Read only; the run fails if anything attempts to modify it.
```

```{option} output_dir

Directory receiving `train_decontaminated.jsonl` and `decontamination_report.json`.
```

```{option} id_field

Required, and unique within each split.
```

```{option} text_field

Record field holding the text to compare.

Default: `text`.
```

```{option} threshold

The minimum exact Jaccard similarity that removes a MinHash/LSH candidate.
This threshold does not apply to source-identity matches.
The step removes source-identity matches regardless of text similarity.

Default: `0.8`.
```

```{option} shingle_kind

`char` or `word`.
Must match the candidate generator; NeMo Curator shingles on characters.

Default: `char`.
```

```{option} normalization

Transformations applied before shingling: `nfc`, `casefold`, `collapse_whitespace`, and `strip_punctuation`.
Recorded in the report.
Punctuation stripping is category-based, so combining marks such as Devanagari vowel signs are preserved.

Default: all four enabled.
```

```{option} minhash

Passed to NeMo Curator's `FuzzyDeduplicationWorkflow`: `seed`, `char_ngrams`, `num_bands`, and `minhashes_per_band`.
`char_ngrams` is also the width used by the exact verification.

Default: `{seed: 42, char_ngrams: 24, num_bands: 20, minhashes_per_band: 13}`.
```

```{option} prefer_id_field

Group on `id_field` before URL aliases rather than after.
Set this when the URL column is coarse, such as a `source_url` that holds a site homepage; otherwise a single holdout page can mark every training document from that site as overlapping.

Default: `false`.
```

```{option} shared_id_space

Whether the two splits are parts of one corpus, so that an equal identifier means the same document.
Set `false` only when the splits are separately built corpora that both number from zero.

Default: `true`.
```

```{option} work_dir

Scratch directory for the union corpus and NeMo Curator's deduplication cache.

Default: `./cache/decontamination`.
```

```{option} skip_similarity

Run only the exact source-identity pass.
The report then states that near-duplicate overlap was not measured, rather than reporting zero overlap.

Default: `false`.
```

## Strategies

| When | Then |
| --- | --- |
| You want an inexpensive check before committing a GPU | Set `skip_similarity: true`. The source-identity pass catches same-page leaks regardless of how much the text was rewritten. |
| You need to know what the candidate generator missed | Call `runtime/decon.candidate_recall` on a sample small enough to compare every pair. LSH recall depends on the band structure and the corpus, so a figure quoted from elsewhere does not transfer. |
| A benchmark question appears verbatim inside long training documents | This step does not detect it. Substring contamination needs a containment-oriented design. |
| You are tuning the threshold | Read `unverifiable_pairs` and `removed_pairs` in the report first; similarity is reported per pair. |

## Common Errors

```{option} holdout_modified

Something attempted to remove documents from the held-out split.
This is never permitted.
```

```{option} id_field_missing_or_duplicated

Every record in both splits needs a unique identifier.
```

```{option} unverifiable_pairs

A warning, not a failure.
Candidate pairs whose text was too short to shingle, or whose documents could not be found, are reported and not removed.
```

```{option} shingling_mismatch

`shingle_kind` and `minhash.char_ngrams` must describe the same shingling the candidates were generated with.
```

```{option} no_gpu_available

The similarity pass requires the `curate-gpu` extra (or the `curate-gpu` isolated runtime) and a GPU.
Set `skip_similarity: true` to run the identity pass on the CPU-only `curate` extra.
```

## Examples

Identity-only pass on the CPU:

```bash
uv run --no-sync nemotron steps run curate/decontamination -c default \
    train_glob='./output/vi/filtered_jsonl/**/*.jsonl' \
    holdout_glob='./data/holdout/vi/*.jsonl' \
    output_dir=./output/vi/decontaminated \
    id_field=id \
    skip_similarity=true
```

## Related Pages

- [`src/nemotron/steps/curate/nemo_curator/decontamination/README.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/curate/nemo_curator/decontamination/README.md)
- {doc}`../how-to/decontaminate-and-subset`
