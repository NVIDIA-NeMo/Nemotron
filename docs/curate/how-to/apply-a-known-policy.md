---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Filter a corpus with curate/nemo_curator and an existing approved policy, without the flow driver."
topics: ["Curation", "How-To", "Filter Policy"]
tags: ["How-To", "Curation", "Policy"]
content:
  type: "How-To"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Apply a Known Policy

Use this guide when an approved policy already exists for the corpus and language pack, and you want to run the filter step alone.
To choose thresholds for a new corpus, use {doc}`run-the-measure-apply-flow` instead.

## Prerequisites

- An `approved_policy.yaml` with `approved: true` and a complete `approval` block. Refer to {doc}`../reference/policy-file`.
- The language pack the policy was measured against, at a path you can name in `langpack_dir`.
- A corpus in JSONL. Parquet must pass through `curate/ingest` first.

## Confirm the Policy Matches the Corpus

The policy records the fingerprint of the corpus it was measured on and the content hash of the language pack.
The step verifies both before applying a threshold and stops on a mismatch (`approval_corpus_mismatch`, `policy_langpack_mismatch`).
If the corpus has changed since the policy was approved, re-profile rather than override.

## Write the Configuration

Create a configuration that names the policy and the pack root:

```yaml
input_glob: ./output/vi/ingested/*.jsonl
output_dir: ./output/vi/filtered_jsonl
text_field: text
id_field: id
source_field: source
metadata_fields: [id, source, url]
mode: both
heuristic_filters:
  approved_policy: ./output/vi/policy/approved_policy.yaml
  langpack_dir: ./langpacks
language_codes: []
domains: []
quality_filters: {}
dataset: null
emit_manifest: ./output/vi/filtered_jsonl/run_manifest.json
emit_ledger: ./output/vi/filtered_jsonl/curation_ledger.json
```

Choose `mode` by what downstream steps need:

| Need | `mode` |
| --- | --- |
| The smallest output, with no score columns | `filter` |
| Every row retained, scores recorded for later re-thresholding | `annotate` |
| Rejected rows dropped, scores retained for `curate/subset` stratification | `both` |

`mode` governs only the policy's signals.
If `language_codes`, `quality_filters`, or `domains` are also set, those gates drop rows under every mode.

Set `emit_manifest` and `emit_ledger` so that `curate/audit` can later prove completeness and attribute removals.

## Run the Step

```bash
uv run --no-sync nemotron steps run curate/nemo_curator -c ./my_policy_run.yaml
```

Or apply the policy on top of an existing configuration with dotlist overrides:

```bash
uv run --no-sync nemotron steps run curate/nemo_curator -c tiny \
  input_glob="${PWD}/output/vi/ingested/*.jsonl" \
  output_dir="${PWD}/output/vi/filtered_jsonl" \
  mode=both \
  heuristic_filters.approved_policy="${PWD}/output/vi/policy/approved_policy.yaml" \
  heuristic_filters.langpack_dir="${PWD}/langpacks"
```

## Verify the Result

Open `run_manifest.json`:

| Field | Expected value |
| --- | --- |
| `producer.completed_at` | Present. Its absence means the run stopped before its write barrier. |
| `policy.status` | `approved` |
| `policy.thresholds_applied` | The number of thresholds in the policy file |
| `policy.approval_declared` | The `approval.method` recorded in the policy file |

Then run {doc}`audit-a-curated-corpus` against the output.

## Applying an Unapproved Policy

For an experiment where review has not happened, `heuristic_filters.allow_unvalidated_policy: true` applies a candidate or incomplete policy.
The run logs a warning naming the policy, the manifest records `policy_status: override_unvalidated`, and every later artifact carries that status.
The override is recorded, not hidden; it is not a substitute for approval.
