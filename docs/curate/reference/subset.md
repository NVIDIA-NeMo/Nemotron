---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Reference for the curate/subset step: parameters, artifacts, strategies, and errors."
topics: ["Curation", "Reference", "Subsets"]
tags: ["Reference", "Curation", "Steps"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Data Scientist"]
---

# curate/subset

The `curate/subset` step draws stratified subsets of a corpus at several fixed token budgets in one run.
Every smaller tier is contained in every larger one, so two corpora subset to the same budgets differ in content rather than in size.
Strata are formed from the source, a length band, and optionally the deciles of a quality-score column.

## Syntax

```bash
uv run --no-sync nemotron steps run curate/subset \
    [-c <config-name-or-path>] \
    [-d] \
    [<dotlist-overrides>...]
```

Refer to the [Nemotron Steps CLI Reference](../../train-models/reference/cli-reference.md) for the shared flag set.

## Configuration Files

The step ships two configuration files under `src/nemotron/steps/curate/nemo_curator/subset/config/`.

| File | Purpose |
| --- | --- |
| `default.yaml` | Three budgets of 100 million, 500 million, and 2 billion tokens, counted with a pinned revision of the `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` tokenizer and cached at `./cache/subset/token_counts.json`. |
| `tiny.yaml` | Three small budgets over the packaged 120-document fixture, counted in whitespace words (`tokenizer: null`), so no tokenizer is downloaded. The input path is a container path; override it for local runs. |

## Inputs and Outputs

| Direction | Artifact type | Content |
| --- | --- | --- |
| Consumes | `filtered_jsonl` | A curated corpus carrying its own document identifier, optionally with `__<signal>` score columns from `curate/nemo_curator`. When decontamination runs, this must be its `decontaminated_jsonl` output. |
| Produces | `filtered_jsonl` | One corpus per budget at `budget_<N>_<unit>/subset.jsonl`. Records pass through unchanged. |
| Produces | `subset_plan` | `plan.json`: every tier's per-stratum quota, the stratum membership, the seed, and the tokenizer. Written before any corpus is materialized. |
| Produces | `subset_report` | `subset_report.json`: per tier, `achieved_tokens`, `token_shortfall`, `per_stratum_deviation`, `documents_refilled`, and `strata_exhausted`, plus the tokenizer name and revision. |

## Parameters

```{option} input_glob

JSONL file path or glob to subset.
```

```{option} output_dir

Directory receiving one subdirectory per tier, `plan.json`, and `subset_report.json`.
```

```{option} id_field

Required.
The corpus's own document identifier.
Non-unique or empty values are an error.
```

```{option} text_field

Record field holding the text to count.

Default: `text`.
```

```{option} source_field

Field naming the corpus a record came from; the first component of the stratum key.
Without it, each shard path becomes its own source.

Default: `null`.
```

```{option} token_budgets

One tier per positive integer budget.
All tiers are planned together in a single run; tiers produced by separate runs cannot be shown to nest.
```

```{option} tokenizer

Mapping with `name` and `revision`, forwarded to NeMo Curator's `TokenCountFilter`.
`revision` is required.
Set the block to `null` to count whitespace words; artifacts then record `unit: words`.
```

```{option} token_cache

Path for token counts cached by tokenizer, revision, identifier, and content SHA-256.
A cache from another tokenizer, revision, or schema is ignored.

Default: `null`.
```

```{option} quality_score_field

Optional score column whose deciles join the stratum key.
When set but absent from the data, the run fails rather than stratifying on less than was requested.

Default: `null`.
```

```{option} length_bands

Strictly increasing positive integer upper edges of the length bands, in the budget's unit.

Default: `[128, 512, 2048, 8192]`.
```

```{option} seed

Seed for the per-stratum ordering.
The ordering is a hash of the document identifier, never Python's process-salted `hash()`.

Default: `0`.
```

## Strategies

| When | Then |
| --- | --- |
| You want to compare two filtering policies | Run each policy through `curate/nemo_curator`, then subset both to the same `token_budgets`. |
| You need a quick check that the step runs | Use `config/tiny.yaml`. |
| You have score columns and want them to shape the subset | Set `quality_score_field` to a column produced by `curate/nemo_curator` with `mode: annotate` or `mode: both`. |
| You need to add a larger tier later | Re-run with the original budgets plus the new one, the same seed, and the same tokenizer revision. Running the new budget alone may produce a corpus that does not contain the tiers already used. |
| You want to inspect the composition before anything is written | Read `plan.json`; it is written before the first tier is materialized. |

## Common Errors

```{option} id_field_not_unique

The error reports the count and three examples.
Deduplicate on the corpus's own identifier, or add one upstream with `curate/ingest`.
```

```{option} id_field_missing

Set `id_field` to the corpus's own document identifier.
An identifier assigned after filtering cannot be traced back to the input.
```

```{option} quality_score_field_absent

The column is named in the configuration but missing from the records.
Produce it with `curate/nemo_curator` in `mode: annotate` or `mode: both`, or unset `quality_score_field`.
```

```{option} quality_score_non_finite

At least one configured quality score is NaN or infinite.
Repair or remove those scores before stratifying.
```

```{option} token_count_non_positive

At least one usable document has a zero or negative token count.
Check the tokenizer, the cache, and the input text.
```

```{option} invalid_budget_or_length_band

`token_budgets` must contain positive integers and `length_bands` must contain strictly increasing positive integers.
Values are refused rather than coerced.
```

```{option} tier_write_mismatch

A planned document identifier could not be written during materialization.
The run removes every tier and the success report rather than publishing an artifact that disagrees with `plan.json`.
Ensure that the input does not change during the run.
```

```{option} tokenizer_revision_missing

Set `tokenizer.revision`, or set `tokenizer` to `null` to count whitespace words.
```

```{option} starved_strata

A warning, not a failure.
The budget divided across the strata gives some of them less than their shortest document.
Raise the budget, widen `length_bands`, or unset `quality_score_field`.
```

```{option} token_shortfall

Expected, not a fault.
Nesting and filling the budget exactly are not simultaneously satisfiable, so a tier may achieve fewer tokens than its budget.
Shortfall is reported per stratum and never made up from another stratum.
```

```{option} tiers_do_not_nest

A defect in the step, not in the configuration.
The run fails rather than writing subsets that do not nest.
Report it with `plan.json` attached.
```

## Examples

Cut three nested tiers from a decontaminated corpus, stratified on a policy score:

```bash
uv run --no-sync nemotron steps run curate/subset -c default \
    input_glob=./output/vi/decontaminated/train_decontaminated.jsonl \
    output_dir=./output/vi/subset \
    id_field=id \
    source_field=source \
    quality_score_field=__script_ratio
```

## Related Pages

- [`src/nemotron/steps/curate/nemo_curator/subset/README.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/curate/nemo_curator/subset/README.md)
- {doc}`../how-to/decontaminate-and-subset`
