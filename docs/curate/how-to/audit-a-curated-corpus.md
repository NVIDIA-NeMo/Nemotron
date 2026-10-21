---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Verify a curated corpus with curate/audit: readability, completeness against the producer's manifest, and gate attribution."
topics: ["Curation", "How-To", "Audit"]
tags: ["How-To", "Curation", "Audit"]
content:
  type: "How-To"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Audit a Curated Corpus

Use this guide to verify a corpus that `curate/nemo_curator` wrote, whether it came from the flow or from a stand-alone run.
The audit reads files on disk; it starts no Ray cluster and needs no GPU.

## Prerequisites

- The corpus shards, usually `<output_dir>/**/*.jsonl`.
- For a completeness claim, the `run_manifest.json` the producing step wrote (`emit_manifest`).
- For gate attribution, the `curation_ledger.json` the producing step wrote (`emit_ledger`).
- For containment, the corpus the target was derived from.

Without a manifest, the audit reports counts and readability only.
A corpus can parse cleanly and still be missing rows, and a filter is expected to remove rows, so a bare count cannot distinguish loss from filtering.

## Choose a Mode

The audit modes are not cumulative.
In particular, `containment` does not include a digest.

| `mode` | Checks | Requires |
| --- | --- | --- |
| `integrity` | Every shard is readable; row counts; manifest reconciliation when a manifest is supplied | Nothing beyond the shards |
| `digest` | `integrity` plus a content digest that is independent of file enumeration order | Nothing beyond the shards |
| `containment` | `integrity` plus containment, without a digest | `reference_glob` and `comparison_fields` |
| `all` | `integrity`, digest, and containment checks | `reference_glob` and `comparison_fields` |

Set `comparison_fields` explicitly for containment.
The pipeline adds language and domain columns, so an implicit "all common fields" comparison would report differences that are the pipeline working as intended.
Use `[id]` when the corpus carries a stable identifier.

## Write the Configuration

```yaml
target_glob: ./output/vi/filtered_jsonl/**/*.jsonl
output_dir: ./output/vi/audit
declared_manifest: ./output/vi/filtered_jsonl/run_manifest.json
ledger_glob: ./output/vi/filtered_jsonl/curation_ledger.json
reference_glob: ./output/vi/ingested/**/*.jsonl
mode: all
comparison_fields: [id]
source_field: null
digest_root: ./output/vi
```

`source_field` is taken from the declared manifest when one is supplied.
`digest_root` makes shard paths in the digest relative, so moving the corpus does not change the digest.

## Run the Audit

```bash
uv run --no-sync nemotron steps run curate/audit -c ./my_audit.yaml
```

Or with dotlist overrides on the shipped starter configuration:

```bash
uv run --no-sync nemotron steps run curate/audit -c default \
  target_glob="${PWD}/output/vi/filtered_jsonl/**/*.jsonl" \
  output_dir="${PWD}/output/vi/audit" \
  declared_manifest="${PWD}/output/vi/filtered_jsonl/run_manifest.json" \
  ledger_glob="${PWD}/output/vi/filtered_jsonl/curation_ledger.json" \
  mode=digest
```

The command uses these exit codes:

- `0`: The audit completes without findings.
- `1`: The audit completes with findings and writes `audit_report.json`.
- `2`: The audit rejects the configuration.

Exit code `1` reports a finding, not a crash.

## Read the Report

Open `<output_dir>/audit_report.json`.

| Section | What it states |
| --- | --- |
| Readability | Per-shard readability and row counts |
| Completeness | File and row count comparison, plus a `manifest_incomplete` finding when the manifest has no `completed_at` |
| Attribution | `filtered_by_reason`, the per-gate removal counts from the ledger, and `unexplained`, the records that left the pipeline for a reason no stage recorded |
| Digest | The content digest, when `mode` is `digest` or `all` |
| Containment | Target records absent from the reference, when `mode` is `containment` or `all` |

A manifest without `completed_at` still participates in the file and row count comparison.
The `manifest_incomplete` finding prevents the audit from passing, even when those counts match.

Every record figure in a loss report is a floor: a shard truncated by a terminated job reports zero rows, so the audit counts units rather than trusting record counts alone.
Any non-zero `unexplained` figure is a finding.
One cause is the same gate declared twice, for example length gated in both `quality_filters` and the policy, which makes the producer discard its per-gate breakdown.
Refer to {doc}`../explanation/pipeline-artifacts` for how the manifest, ledger, and audit report relate.

## Next Steps

- Field reference: {doc}`../reference/audit`.
- Failure identifiers: {doc}`../reference/troubleshooting`.
