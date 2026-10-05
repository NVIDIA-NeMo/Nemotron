---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Reference for the curate flow driver: command, configuration keys, derived paths, preflight checks, and the flow report."
topics: ["Curation", "Reference", "Flow"]
tags: ["Reference", "Curation", "Configuration"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Data Scientist"]
---

# Curate Flow Configuration

The *curate flow* runs the six curation steps from one configuration file, derives every cross-step path so that a producer and its consumer cannot disagree, and refuses a misconfigured run before any step does work.
The flow is a script, not a registered step: it takes a configuration path and accepts no `key=value` overrides.

The driver is `src/nemotron/steps/curate/nemo_curator/scripts/run_flow.py`.

## Syntax

```bash
uv run --extra curate --extra xenna \
  python -m nemotron.steps.curate.nemo_curator.scripts.run_flow \
  --config <path/to/flow.yaml> [--plan]
```

| Flag | Meaning |
| --- | --- |
| `--config` | Path to the flow configuration file. |
| `--plan` | Write `flow_plan.json` under `output_root`, print what would run, and exit without running any step. |

Both extras are required.
NeMo Curator executes the filter step on Ray, and Ray resolves the worker interpreter independently of the launching one; without `xenna` on the worker side, the run fails inside a worker with `ModuleNotFoundError: No module named 'cosmos_xenna'` after the plan has already been printed.

## Configuration Files

Two worked-example configurations ship under `src/nemotron/steps/curate/nemo_curator/config/`.

| File | Purpose |
| --- | --- |
| `vi_c4_measure.yaml` | First run: ingest, profile, filter with language and length gates only, and audit. `approve` is `null`. |
| `vi_c4_apply.yaml` | Second run: identical above `steps:`, with `steps.profile` disabled, `steps.subset` enabled, and the `approve` block filled in. |

Copy both files and edit `corpus.input`, `output_root`, and `corpus.langpack_dir`.
Those three values must be identical between the two runs.

## Top-Level Keys

```{option} corpus

Describes the corpus that every step reads.
Refer to [Corpus Block](#corpus-block).
```

```{option} output_root

Directory under which every step writes.
Refer to [Derived Paths](#derived-paths).

Default: `./output/curate`.
```

```{option} steps

One block per step, keyed `ingest`, `profile`, `filter`, `audit`, `decontamination`, and `subset`.
Each block carries `enabled` plus any parameter of the underlying step; a key written in a step block overrides the value the flow derived for it.
The `filter` block configures the registered `curate/nemo_curator` step.
```

```{option} approve

Promotes a candidate policy into an approved policy before the steps run.
Refer to [Approve Block](#approve-block).

Default: `null`.
```

## Corpus Block

```{option} corpus.input

Required.
Glob, directory, or list of raw corpus files.
```

```{option} corpus.text_field

Column holding the text in the raw corpus.

Default: `text`.
```

```{option} corpus.id_field

Document identifier field.
When `steps.ingest` is enabled, every downstream step reads the canonical `id` field that ingestion writes.
```

```{option} corpus.id_field_in_source

Set `true` when the raw corpus already carries `corpus.id_field`; ingestion then keeps it rather than minting one.
```

```{option} corpus.id_fields

Columns from which ingestion mints a content-derived identifier.
Defaults to `url` (when it is a metadata field) and the text field.
```

```{option} corpus.id_prefix

Prefix for minted identifiers.

Default: `""`.
```

```{option} corpus.source_field

Field naming the source of each record.
When ingestion is enabled, downstream steps read the canonical `source` field.
```

```{option} corpus.source_field_in_source

Column in the raw corpus to map onto `source`.
```

```{option} corpus.source_value

A constant source label when the raw corpus has no source column.
```

```{option} corpus.metadata_fields

Columns carried through filtering.
Ingestion keeps every listed field other than `id` and `source`, which it produces itself.
```

```{option} corpus.language

BCP-47 tag that selects the language pack for profiling.
Required when `steps.profile` is enabled; there is no default language.
This is separate from `steps.filter.language_codes`, which decides which languages are kept.
```

```{option} corpus.langpack_dir

Root directory of the language packs.
Required when `steps.profile` is enabled; the flow never selects a pack root implicitly.
```

## Approve Block

```{option} approve.from

Path of the candidate policy to promote.

Default: `<output_root>/profile/candidate_policies.yaml`.
```

```{option} approve.thresholds

Required.
List of `{signal, min}`, `{signal, max}`, or `{signal, min, max}` entries.
Every signal must have been profiled, and each bound must match the signal's registered direction.
```

```{option} approve.approver

Person recording the approval.
Required by the approved-policy contract.
```

```{option} approve.evidence

What was examined and why these values were chosen.
Required by the approved-policy contract.
```

```{option} approve.method

`manual` or `ablation`.
```

```{option} approve.date

Optional provenance.
```

```{option} approve.verify_corpus

Compare the fingerprint recorded in the candidate policy against the corpus that the filter step will read.
A mismatch is refused.
When ingestion is enabled, verification is deferred until ingestion has written the prepared corpus.

Default: `true`.
```

The promoted policy is written to `<output_root>/policy/approved_policy.yaml` and set as `steps.filter.heuristic_filters.approved_policy`.
Refer to {doc}`policy-file` for the resulting document.

## Derived Paths

The flow computes every artifact path from `output_root`.

| Artifact | Path |
| --- | --- |
| `prepared` | `<output_root>/ingested/` |
| `candidates` | `<output_root>/profile/candidate_policies.yaml` |
| `profile_report` | `<output_root>/profile/profile_report.json` |
| `approved_policy` | `<output_root>/policy/approved_policy.yaml` |
| `corpus` | `<output_root>/filtered_jsonl/` |
| `manifest` | `<output_root>/filtered_jsonl/run_manifest.json` |
| `ledger` | `<output_root>/filtered_jsonl/curation_ledger.json` |
| `decontaminated` | `<output_root>/decontaminated/train_decontaminated.jsonl` |

Per-step derivations:

- `profile` and `filter` read the ingested JSONL when `steps.ingest` is enabled, otherwise `corpus.input`.
- `filter` receives `emit_manifest` and `emit_ledger` set to the manifest and ledger paths.
- `audit` receives `declared_manifest`, `ledger_glob`, `reference_glob`, and `digest_root` from the same table.
- `decontamination` reads `steps.decontamination.holdout` as its `holdout_glob`; the training split is the filtered corpus.
- `subset` reads the decontaminated file when `steps.decontamination` is enabled, otherwise the filtered corpus.

## Preflight Checks

The flow refuses to run when any of the following hold:

- `corpus.input` is missing.
- `approve` is set while `steps.profile` is enabled. Profiling and approval are separate runs.
- `approve.from` names a file that does not exist.
- `approve.thresholds` is empty, names a signal the profile did not measure, or uses a bound the signal does not accept.
- `approve.verify_corpus` is `true` and the corpus fingerprint does not match the candidate policy.
- `steps.filter.heuristic_filters.approved_policy` is set to a path other than the one the flow promotes.
- `steps.profile` is enabled without `corpus.language` or without an explicit `langpack_dir`.
- `steps.decontamination` is enabled without `holdout`.
- `steps.subset.quality_score_field` names a `__<signal>` column, but no policy is configured, the signal is not among `approve.thresholds`, or `steps.filter.mode` is `filter`.
- `<output_root>/filtered_jsonl/` already holds corpus shards from an earlier run. The filter step appends rather than replaces; delete the directory or choose another `output_root`.
- A disabled step's artifact is required by an enabled step and is absent under `output_root`.
- `steps.ingest` is enabled and `corpus.source_field` is set without `corpus.source_field_in_source` or `corpus.source_value`, so ingestion would have nothing to write into that column.

The flow warns, but runs, when:

- `approve` is set while `steps.filter` is disabled, so the policy is promoted but nothing applies it.
- `steps.audit` runs without the manifest or ledger of a disabled producer; completeness is then not claimed and removals are not attributed.
- `steps.subset` runs with `steps.decontamination` disabled while a decontaminated corpus from an earlier run exists; the tiers are cut from the corpus before decontamination.
- `steps.decontamination` runs the GPU similarity pass rather than `skip_similarity: true`.

## Flow Report

After a run, the flow writes `<output_root>/flow_report.json`:

| Field | Meaning |
| --- | --- |
| `schema_version` | Report schema version. |
| `step_id` | `curate/flow`. |
| `started_at`, `completed_at` | UTC timestamps. |
| `output_root` | The resolved output root. |
| `artifacts` | The derived-path table. |
| `steps` | Per-step results in execution order. |
| `warnings` | Preflight and promotion warnings. |
| `audit_passed` | The audit's verdict, or `null` when the audit did not run. |
| `status` | `ok` or `failed`. |
| `policy_applied` | Whether the filter applied policy thresholds. |
| `policy_promoted` | Whether this run carried an `approve` block. |
| `policy_status` | `approved`, `override_unvalidated`, `unapproved`, or `unresolved`, read from the filter's manifest. |

`--plan` writes `flow_plan.json` with the same derived configurations, so the preview cannot describe a run other than the one that executes.

## Related Pages

- {doc}`../how-to/run-the-measure-apply-flow`
- {doc}`../explanation/two-run-curation`
- {doc}`../explanation/pipeline-artifacts`
- [`src/nemotron/steps/curate/README.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/curate/README.md)
