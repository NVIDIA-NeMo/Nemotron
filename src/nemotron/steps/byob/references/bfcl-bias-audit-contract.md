# BFCL bias-audit contract v1.0

The bias audit is a deterministic, read-only remeasurement of a frozen BFCL
release. It never regenerates, filters, balances, or edits benchmark rows.
`bias_audit_report.json` is the machine contract;
`bias_audit_report.md` is a deterministic rendering of the same report.

## Inputs and binding

Required:

- `run_manifest.json`, including every B1–B16 and dotted B3 applicability entry.
- `benchmark.parquet`, or a publication export whose tree hash is committed by
  the run manifest.

Provide these whenever the corresponding bias is applicable:

- `benchmark_raw.parquet`.
- The stage-cache directory containing `task_instances.parquet` and
  `rendered_conversations.parquet`.
- The exact Oracle-pack manifest for template, held-out, and tool evidence.
- Evaluation `contamination_report.json` files for B9.
- Reviewed B10 distractor evidence and B13 truth-creep evidence.
- B12 portability evidence and any approved exceptions.

Every input hash is recorded in the report. Manifest-declared publication, raw,
and expanded hashes are rechecked before use; all consumed inputs are hashed
again before return. Missing applicable evidence produces a failed metric.
Malformed, drifted, or incorrectly bound evidence aborts the audit.

## Metric contract

Each record has one `primary_metric`, zero or more
`supporting_diagnostics`, source evidence, applicability, evidence completeness,
and approved exceptions.

| ID | Primary metric | Gate |
|---|---|---|
| B1 | `category_balance_score` | ≥0.70 plus category floors/cap |
| B2 | `difficulty_mix_max_abs` | ≤0.05 |
| B3 | `edge_policy_coverage` | =1.0 plus non-single policy floor |
| B4 | `tool_usage_balance_score` | ≥0.60, no non-exempt orphan |
| B5 | `tool_call_count_mix_max_abs` | ≤0.05 plus multi-call survivor |
| B6 | `id_concentration_max` | ≤0.25 plus three IDs per collection |
| B7 | `held_out_leak_count` | =0 across expanded/raw/published |
| B8 | `paraphrase_leak_escape_count` | =0 across expanded/raw/published |
| B9 | `contamination_violations` | =0 with release-bound eval evidence |
| B10 | `distractor_gold_agreement` | ≥0.90 on seeded sample |
| B11 | `post_dedup_edge_coverage` | =1.0 |
| B12 | `agnostic_portability_pass` | true |
| B13 | `truth_creep_incidents` | =0 |
| B14 | `negative_or_irrelevant_rate` | ≥0.10 plus failure-family floor |
| B15 | `intent_balance_score` | ≥0.65, no orphan, configured per-category target +0.05 |
| B16 | `turn_mix_abs_error` | ≤0.05 |

For B5, an applicable release must pin all three `tool_call_count_mix`
buckets in `run_manifest.bias_targets`. The auditor does not inject the generic
design example after generation, because Stage 11 can only be audited against a
target it actually consumed. A missing target therefore yields `value: null`,
incomplete evidence, and a failed B5 verdict.

For B15, an applicable release must likewise pin `max_intent_share` in
`run_manifest.bias_targets`. The observed maximum within each category may
exceed that target by at most five percentage points. The auditor does not
invent the previous `0.50` design default when the manifest omits it; legacy
releases need an approved exception or a regenerated manifest from a new run.

An N/A record has a stable non-empty reason, `value: null`, and does not count
as an applicable pass. An approved exception records `affected_metric`,
`rationale`, `owner`, and ISO `approval_date`; it changes only the report status,
not the measured verdict.

## Reviewed B10 evidence

Schema `1.0`, kind `distractor_gold_agreement`, and
`run_manifest_hash` bind the review. `reviewers` contains at least two distinct
identities. `seed` selects exactly `min(30, eligible rows)` by the contract's
SHA-256 ordering. Each sampled row carries one annotation per distinct reviewer:

```json
{
  "task_id": "task-id",
  "annotations": [
    {"reviewer": "reviewer-a", "selected_tools": ["tool_a"]},
    {"reviewer": "reviewer-b", "selected_tools": ["tool_a"]}
  ]
}
```

Disagreement additionally requires `adjudicated_tools` and a non-empty
`adjudication_rationale`. A strict annotation majority remains authoritative;
adjudication resolves ties and documents all disagreements.

## Reviewed B13 evidence

Schema `1.0`, kind `judge_truth_creep`, and the manifest hash bind the review.
`prompt_hashes` contains every reviewed prompt content hash and must include the
judge prompt committed by generation. `seed` and `sample_size` select the exact
rationale sample. Incidents use one of:

- `tool_correctness_label`
- `gold_rewrite`
- `truth_based_keep_drop`

Each incident records a declared reviewer and non-empty detail. When no surface
judge was enabled, B13 deterministically reports zero incidents without external
evidence.

## CLI

```bash
python -m nemotron.steps.byob.scripts.audit_bfcl_bias \
  --run-manifest /release/run_manifest.json \
  --raw /release/benchmark_raw.parquet \
  --expanded /release/stage_cache \
  --pack-manifest /pack/manifest.yaml \
  --contamination-report /eval/contamination_report.json \
  --distractor-evidence /review/b10.json \
  --judge-evidence /review/b13.json \
  --portability-evidence /review/b12.json \
  --output-dir /audit
```

Paths adjacent to the manifest are discovered when present. Exit status is `0`
for passed or fully excepted reports, `1` for completed audits with unexcepted
metric failures, and `2` for invalid, tampered, or unusable evidence.

## Restoring a publication-trimmed rendered cache

When a release retained `benchmark_raw.parquet` but omitted
`stage_cache/rendered_conversations.parquet`, restore the stage projection
without rerunning generation:

```bash
python -m nemotron.steps.byob.scripts.restore_bfcl_rendered_conversations \
  --run-manifest /release/run_manifest.json \
  --raw-benchmark /release/benchmark/benchmark_raw.parquet \
  --output /release/stage_cache/rendered_conversations.parquet
```

The command verifies the raw-table hash and schema, reverses the lossless Stage
12 message projection, writes with the canonical stage schema, and publishes
the result only when its bytes reproduce the `rendered_conversations` hash
already committed by the manifest. It never calls a model or modifies the
frozen benchmark tables.
