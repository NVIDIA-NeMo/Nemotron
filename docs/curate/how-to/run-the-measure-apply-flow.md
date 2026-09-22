---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Run the curate flow twice: measure a corpus, review the candidate thresholds, then approve and apply them."
topics: ["Curation", "How-To", "Flow"]
tags: ["How-To", "Curation", "Policy"]
content:
  type: "How-To"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Run the Measure and Apply Flow

Use this guide to curate a corpus whose filter thresholds have not been chosen.
The flow driver runs the six curation steps from one configuration file; you run it once to measure and once to apply.
For the reasoning behind the two runs, refer to {doc}`../explanation/two-run-curation`.

## Prerequisites

- `uv sync --extra curate` completed in the repository clone.
- A raw corpus in Parquet or JSONL.
- The BCP-47 tag of the corpus language and a reviewed language pack for it. The shipped `vi` pack under `src/nemotron/steps/curate/nemo_curator/data/langpacks/` is example data.
- Optionally, the FastText `lid.176.bin` model, exported as `FASTTEXT_LANGID_MODEL`, for the language-composition table and the language gate.
- For decontamination with the similarity pass, one GPU and `--extra curate-gpu`.

## Copy the Configuration Files

Copy both worked-example files and edit the paths in each:

```bash
cp src/nemotron/steps/curate/nemo_curator/config/vi_c4_measure.yaml ./my_corpus_measure.yaml
cp src/nemotron/steps/curate/nemo_curator/config/vi_c4_apply.yaml ./my_corpus_apply.yaml
```

Set `corpus.input`, `output_root`, and `corpus.langpack_dir` in both, and keep those three values identical between them.
The approved policy carries a fingerprint of the corpus it was measured on, and the second run refuses an approval whose corpus differs from the one present.

Set `corpus.language` to the corpus language tag and `steps.filter.language_codes` to the languages to keep.
The two keys are separate decisions: the first selects the pack used for measurement, the second decides which documents survive the language gate.

## Print the Plan

```bash
uv run --extra curate --extra xenna \
  python -m nemotron.steps.curate.nemo_curator.scripts.run_flow \
  --config ./my_corpus_measure.yaml --plan
```

`--plan` resolves the per-step configurations, runs every preflight check, writes `<output_root>/flow_plan.json`, and exits without running a step.
Read the plan and correct any refused configuration before continuing.

## Run the Measurement

```bash
uv run --extra curate --extra xenna \
  python -m nemotron.steps.curate.nemo_curator.scripts.run_flow \
  --config ./my_corpus_measure.yaml
```

In the measurement configuration, `steps.profile.enabled` is `true` and `approve` is `null`.
The run ingests the corpus, profiles the unfiltered documents, applies only the language and length gates you configured, and audits the result.
No policy is applied because none has been approved.

## Read the Summary

Open `<output_root>/profile/profile_summary.md`.

1. Read the **Policy simulation** section first. The per-signal tables show each gate applied alone, and gates overlap, so the sum of per-gate removals exceeds the union.
2. Read the **Approve block** section. It lists each candidate threshold with its retention on the same line so the two cannot be mispaired.
3. Read the histograms for the signals you intend to gate. A signal with an empty region between two populations tolerates any threshold inside that region; a signal with a smooth decay has no natural cut point, and any threshold is a budget decision.

Read thresholds from the summary rather than from `candidate_policies.yaml`.
In the candidate file, a band's `threshold_low` and `threshold_high` are two alternative single bounds, not a range, and choosing the wrong end is accepted without a warning.

The approve block lists one-sided signals only.
`word_count`, `mean_word_length`, and `token_count` gate from both sides; write those by hand with both `min` and `max`, or leave length gating to `steps.filter.quality_filters`.
Do not gate length in both places: the ledger then cannot reconcile the per-gate counts and records the removals as unattributed.

## Write the Approval

Open `./my_corpus_apply.yaml` and confirm that everything above `steps:` matches the measurement file exactly.
Then:

1. Set `steps.profile.enabled: false`. Re-profiling during the application run costs a second full pass and would replace the reviewed candidate measurements.
2. Paste the chosen thresholds into `approve.thresholds`. Each signal takes `min`, `max`, or both as the registry fixes; a threshold written in the wrong direction is refused.
3. Fill in `approve.approver`, `approve.date`, `approve.method` (`manual` or `ablation`), and `approve.evidence`. The evidence text is the only record of why these values were chosen.
4. Optionally enable `steps.decontamination` with a `holdout`, and `steps.subset` with `token_budgets`. If you set `steps.subset.quality_score_field`, it must name a `__<signal>` column from `approve.thresholds`, and `steps.filter.mode` must be `annotate` or `both`.

Every value in the approve block should be a swept grid point from the summary.
The flow warns when a chosen threshold was not measured.

## Remove Stale Output

The filter step appends to `filtered_jsonl/` rather than replacing it, and the flow refuses to run into a directory that already holds corpus shards.
Remove the outputs the second run rewrites; keep `ingested/` and `profile/`, which it does not.

```bash
rm -rf ./output/my_corpus/filtered_jsonl ./output/my_corpus/audit ./output/my_corpus/subset
```

## Run the Application

```bash
uv run --extra curate --extra xenna \
  python -m nemotron.steps.curate.nemo_curator.scripts.run_flow \
  --config ./my_corpus_apply.yaml
```

At preflight the flow verifies the corpus fingerprint, writes `<output_root>/policy/approved_policy.yaml`, and wires it into the filter step.
It then filters, audits, and, if enabled, decontaminates and subsets.

## Inspect the Result

Open `<output_root>/flow_report.json` and check:

| Field | Expected value |
| --- | --- |
| `status` | `ok` |
| `policy_status` | `approved` |
| `policy_applied` | `true` |
| `artifacts.approved_policy` | The path of `approved_policy.yaml` |
| `audit_passed` | `true` |
| `warnings` | Empty, or warnings you have read and accepted |

`policy_applied` reports whether the filter applied policy thresholds.
`artifacts.approved_policy` identifies the policy file that the flow produced.

`<output_root>/filtered_jsonl/run_manifest.json` records the policy digest and approver, and `<output_root>/audit/audit_report.json` records completeness and gate attribution.
Refer to {doc}`../explanation/pipeline-artifacts` for what each file proves.

## Next Steps

- Verify an existing corpus without re-running the flow: {doc}`audit-a-curated-corpus`.
- Score the approved policy against labelled documents: {doc}`evaluate-a-policy`.
- Field reference: {doc}`../reference/flow-config`.
