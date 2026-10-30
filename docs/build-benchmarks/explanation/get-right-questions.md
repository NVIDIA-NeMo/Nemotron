<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Getting the Right Questions From the Source Benchmark

Choose the best rows from the Hugging Face source benchmark, such as Massive Multitask Language Understanding (MMLU), to use as few-shot examples for each target, from broad choices down to very specific tags.

You control which source questions appear as few-shots by configuring coarse filters (`split`, `subset`, `hf_dataset`, `source_subjects`) and fine filters (`target_source_mapping`, and optional `tags` backed by `metadata_file`).
The intent is to show the model exemplars that match the subject you are generating for.

In `nemotron steps run byob/mcq`, each key in `target_source_mapping` must match a folder of `.txt` files or a `*.parquet` file under `input_dir`, not an abstract label on its own.

## The Funnel: Coarse to Fine-Grained Control

Most question-answer benchmark datasets expose many subjects, such as astronomy, econometrics, elementary mathematics, nutrition, and so on.
Coarse filters shrink that universe before you attach sources to each local target.

You can narrow step by step until each target draws from the pool you intend.

```{image} ../_images/funnel.png
:class: .sd-p-1
:width: 640px
:alt: Funnel from coarse to fine filtering: split, subset, source_subjects, then target_source_mapping with optional subjects and tags per target.
```

```{rubric} Coarse (top of funnel)
```

- `hf_dataset`: which Hugging Face benchmark to load, for example `cais/mmlu`.
- `split`: which split to read first, such as `test` or `train`, applied when the loader pulls the benchmark.
- `subset`: which variant of that benchmark, such as `all` for MMLU or `kn` for MMLU Indic, also applied at load time.
- `source_subjects`: which benchmark taxonomy labels may appear in the pool.
  If you leave this list empty, the pipeline expands it to every subject available for the chosen `hf_dataset`, `subset`, and `split`.
  Mappings only sample from subjects that remain in this pool.

```{rubric} Finer (middle of funnel)
```

- `target_source_mapping`: for each target key that exists under `input_dir`, you list which source subjects to sample and optionally assign weights.
  This ties each corpus folder or Parquet file to the part of the benchmark taxonomy you want the model to imitate.

```{rubric} Finest (bottom of funnel)
```

- `tags`: optional.
  When `metadata_file` lists question identifiers with tags, each target can filter few-shots further by tag strings such as `reasoning` or `unambiguous`, or by comma-joined combinations such as `reasoning,unambiguous`, optionally with weights.
  Tags are the tightest lever on which rows become few-shots.

## How It Fits Together

1. Load: the pipeline loads the benchmark for `hf_dataset`, `split`, and `subset`, then keeps only the subjects in `source_subjects`, or all subjects when that list is empty after expansion.
1. Map: for each target, `target_source_mapping` names source subjects and optional tag sets to sample from.
   Source subjects and tags each support explicit weights; otherwise sampling is uniform over the listed entries.
1. Sample: while building the seed dataset, the prepare step samples subject-tag pairs according to those weights and draws few-shot questions from the filtered source tables.

The coarse settings define the global pool.
`target_source_mapping`, and tags when enabled, narrow which slice of that pool each target should use when it pairs exemplars with your domain chunks.

## Configuration at a Glance

| Control | Where | Effect |
| ------- | ----- | ------ |
| `hf_dataset` | Top-level config | Which Hugging Face dataset supplies few-shot rows. |
| `split` | Top-level config | Dataset split, such as `test`, taken from the source benchmark. |
| `subset` | Top-level config | Dataset subset, such as `all` or `kn`, applied before subject filtering. |
| `source_subjects` | Top-level config | Restricts which benchmark subjects remain in the pool; an empty list expands to all subjects for the chosen dataset, split, and subset. |
| `target_source_mapping` | Per target directory or Parquet | For each target under `input_dir`, which source subjects and optional tags to use, with optional weights. |

## Related documentation

- For full option details and validation rules, see {doc}`../reference/generate-config`.
- For allowed Hugging Face benchmarks and subsets, see {doc}`../reference/benchmarks`.
- For how the prepare step uses this mapping to build the seed dataset, see {doc}`data-preparation`.
