---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Reference for the curate/audit step: parameters, modes, artifacts, strategies, and errors."
topics: ["Curation", "Reference", "Audit"]
tags: ["Reference", "Curation", "Steps"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Data Scientist"]
---

# curate/audit

The `curate/audit` step produces integrity evidence for a curated corpus: per-shard readability, row counts, and a content digest, compared against the manifest declared by the producing step.
Completeness is claimed only relative to that manifest.
The step detects loss; attributing loss to a cause requires the ledger that the producing step emitted.

## Syntax

```bash
uv run --no-sync nemotron steps run curate/audit \
    [-c <config-name-or-path>] \
    [-d] \
    [<dotlist-overrides>...]
```

Refer to the [Nemotron Steps CLI Reference](../../train-models/reference/cli-reference.md) for the shared flag set.

## Configuration Files

The step ships two configuration files under `src/nemotron/steps/curate/nemo_curator/audit/config/`.

| File | Purpose |
| --- | --- |
| `default.yaml` | Audits `./output/filtered_jsonl/**/*.jsonl` in `integrity` mode with no manifest or ledger. Row counts are informational until `declared_manifest` is set. |
| `tiny.yaml` | Audits the packaged fixture in `digest` mode against its checked-in `run_manifest.json`. The paths are container paths; override them for local runs. |

## Inputs and Outputs

| Direction | Artifact type | Required | Content |
| --- | --- | --- | --- |
| Consumes | `filtered_jsonl` | Yes | The corpus to audit. `decontaminated_jsonl` is accepted. |
| Consumes | `curation_manifest` | No | `run_manifest.json` from the producing step. Without it, completeness is not claimed. |
| Consumes | `curation_ledger` | No | `curation_ledger.json` from the producing step. Its presence turns detection into attribution. |
| Produces | `curation_report` | | `audit_report.json`: per-shard readability and digests, row counts, manifest comparison, and an optional reference delta. |

## Parameters

```{option} target_glob

JSONL files to audit.
```

```{option} output_dir

Directory for `audit_report.json`.
```

```{option} ledger_glob

Ledgers written by the producing steps.
Without them the audit detects loss but cannot attribute it.
With them, the report states how many records each gate removed and how many left for an unrecorded reason.

Default: `null`.
```

```{option} declared_manifest

Manifest emitted by the producing step.
Required for any completeness claim; without it, row counts are informational.

Default: `null`.
```

```{option} reference_glob

Upstream corpus from which the target was derived.
`null` disables the row delta.
Required for `containment` mode.

Default: `null`.
```

```{option} mode

One of `integrity`, `digest`, `containment`, or `all`.

Default: `integrity`.
```

```{option} comparison_fields

Fields compared in `containment` mode.
An empty list fails in that mode, because the pipeline adds language and domain columns, and an implicit "all common fields" comparison would report the pipeline's own annotations as differences.

Default: `[]`.
```

```{option} source_field

Field naming each record's source corpus, for per-source counts.
Taken from the declared manifest when one is supplied.

Default: `null`.
```

```{option} digest_root

Directory that shard paths in the digest are made relative to, so the digest survives moving the corpus.

Default: `null`.
```

## Strategies

| When | Then |
| --- | --- |
| `curate/nemo_curator` produced less output than expected | Set `declared_manifest` to the manifest it emitted; the per-source delta shows where records went. |
| You must prove that a re-delivered corpus is unchanged | Set `mode=digest` and compare the digest against the previous release. The digest is independent of enumeration order and sensitive to shard names, so the report can identify which shard moved. |
| You must prove that a subset was drawn from a specific corpus | Set `mode=containment`, `reference_glob` to the parent corpus, and `comparison_fields` to `[id]` when the corpus carries a stable identifier. |

## Common Errors

```{option} attribution_disagreement

The run manifest and the curation ledger attribute the same removals to different gates.
Re-run the producing step; if the disagreement persists, treat the breakdown as unavailable rather than correcting it by hand.
```

```{option} ledger_unreadable

A file matched by `ledger_glob` is not a readable ledger.
The audit reports it rather than skipping it.
```

```{option} unaccounted_gain

The output holds more records than the input minus the declared removals.
Filtering cannot add rows, so this indicates a retried or duplicated write or an output directory that was not empty before the run.
```

```{option} unexplained_loss

Records left the pipeline for a reason no stage recorded.
Check the producing stage for a caught exception that did not record a terminal state.
```

```{option} ledger_imbalanced

A stage reported success while `n_input` did not equal `n_success + n_filtered + n_failed + n_quarantined`.
Fix the stage before trusting the corpus.
```

```{option} lost_units

One or more shards or sources could not be processed.
The record count in the message is a floor, because a shard too damaged to open reports zero rows.
Re-run the failed units before using the corpus.
```

```{option} unreadable_shard

A shard holds a record that does not parse.
The finding names the file and the byte offset.
Re-run the producing stage for that partition.
An empty but well-formed shard is reported separately as the `zero_row_shard` observation.
```

```{option} manifest_incomplete

The manifest has no `completed_at`; the producing run did not finish.
Re-run before shipping.
```

```{option} manifest_mismatch

Row counts disagree with the producer's declared manifest.
Check the producer's logs for swallowed worker exceptions.
```

```{option} containment_unverifiable

Records could not be keyed, usually because `comparison_fields` names a field the corpus does not carry.
```

```{option} unreadable_reference_shard

A shard in `reference_glob` is damaged, which makes the target appear to have gained rows.
Repair the reference before reading any comparison against it.
```

```{option} duplicate_ids

The identifier repeats while the producer's manifest declares `duplicate_ids: reject`.
Either the identifier does not identify a document, or the manifest should declare `multiset`.
```

```{option} containment_violation

Rows in the target are absent from the reference corpus.
Check that `reference_glob` names the corpus the target was derived from and that `comparison_fields` identifies records rather than fields the pipeline rewrites.
```

## Examples

Audit a filtered corpus against its manifest and ledger:

```bash
uv run --no-sync nemotron steps run curate/audit -c default \
    target_glob='./output/vi/filtered_jsonl/**/*.jsonl' \
    declared_manifest=./output/vi/filtered_jsonl/run_manifest.json \
    ledger_glob=./output/vi/filtered_jsonl/curation_ledger.json \
    output_dir=./output/vi/audit \
    mode=all \
    comparison_fields='[id]' \
    reference_glob='./output/vi/ingested/*.jsonl'
```

## Related Pages

- [`src/nemotron/steps/curate/nemo_curator/audit/README.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/curate/nemo_curator/audit/README.md)
- {doc}`../how-to/audit-a-curated-corpus`
- {doc}`../explanation/pipeline-artifacts`
