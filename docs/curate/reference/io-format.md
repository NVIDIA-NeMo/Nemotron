---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Input and output format for the curation steps and the artifacts they exchange."
topics: ["Curation", "Reference", "IO"]
tags: ["Reference", "JSONL", "Curation"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Data Scientist"]
---

# Curation Input and Output Format

## Input

`curate/ingest` reads Parquet or JSONL.
Every other curation step reads JSON Lines: each line is a JSON object, and the configured `text_field` must exist on each record.

```json
{"id": "doc-001", "source": "c4_vi", "text": "The text to curate."}
```

By default, `text_field` is `text`.
`curate/subset` and `curate/decontamination` additionally require a unique document identifier in `id_field`.
A corpus without one should pass through `curate/ingest`, which mints a content-derived identifier.

## Output of curate/nemo_curator

The step writes JSONL shards under `output_dir` using NeMo Curator `JsonlWriter`.
The output contains `text_field` and the fields listed in `metadata_fields`; `JsonlReader` treats fields as a projection, so any field not listed is dropped at the first stage.

Typical output with no gates and no metadata fields:

```json
{"text": "The text to curate."}
```

When language filtering is enabled, the pipeline adds a language score field used by the filter.
When domain classification is enabled, classifier output fields depend on the installed NeMo Curator classifier implementation.
When an approved policy is applied with `mode: annotate` or `mode: both`, each record carries one `__<signal>` column per policy signal.

## Artifacts Exchanged Between Steps

| Artifact type | File | Producer | Consumers |
| --- | --- | --- | --- |
| `prepared_jsonl` | `part_<n>.jsonl`, `ingest_report.json` | `curate/ingest` | `curate/profile`, `curate/nemo_curator` |
| `profile_report` | `profile_report.json`, `profile_summary.md`, `sample_manifest.json` | `curate/profile` | A reviewer |
| `filter_policy` | `candidate_policies.yaml`, `approved_policy.yaml` | `curate/profile`, then a reviewer | `curate/nemo_curator` |
| `filtered_jsonl` | JSONL shards | `curate/nemo_curator` | `curate/audit`, `curate/decontamination`, `curate/subset`, downstream steps |
| `curation_manifest` | `run_manifest.json` | `curate/nemo_curator` | `curate/audit` |
| `curation_ledger` | `curation_ledger.json` | `curate/nemo_curator` | `curate/audit` |
| `curation_report` | `audit_report.json` | `curate/audit` | A reviewer |
| `decontaminated_jsonl` | `train_decontaminated.jsonl`, `decontamination_report.json` | `curate/decontamination` | `curate/subset`, downstream steps |
| `subset_plan`, `subset_report` | `plan.json`, `subset_report.json` | `curate/subset` | A reviewer |

Refer to {doc}`../explanation/pipeline-artifacts` for how these artifacts relate and {doc}`policy-file` for the policy schema.

## Downstream Use

Use the output of `curate/nemo_curator`, `curate/decontamination`, or a `curate/subset` tier as `filtered_jsonl`.
Common downstream paths are:

- Use `translate/nemo_curator` for corpus translation.
- Use `data_prep/pretrain_prep` for pretraining data preparation.
- Use `data_prep/sft_packing` when the curated records are already in the required supervised fine-tuning (SFT) format.

If a downstream step needs fields beyond `text`, verify that the curation reader/writer path preserves those fields before scaling the run.
