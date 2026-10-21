---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "How the artifacts of the six curation steps relate: manifests, ledgers, reports, policies, and the corpus they describe."
topics: ["Curation", "Explanation", "Artifacts"]
tags: ["Explanation", "Curation", "Provenance"]
content:
  type: "Explanation"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Curation Pipeline Artifacts

Each curation step writes a corpus artifact, a report about that artifact, or both.
The reports exist so that a later step, or a reviewer, can verify a claim about the corpus without re-reading it.
This page explains what each artifact asserts and which other artifacts it depends on.

## The Artifact Graph

```{mermaid}
flowchart TD
    RAW[Raw Parquet or JSONL] --> ING[curate/ingest]
    ING --> PREP[ingested/ shards + ingest_report.json]
    PREP --> PROF[curate/profile]
    PROF --> CAND[candidate_policies.yaml]
    PROF --> PREPORT[profile_report.json + profile_summary.md]
    CAND -->|review and approve| APPR[approved_policy.yaml]
    PREP --> FILT[curate/nemo_curator]
    APPR --> FILT
    FILT --> CORPUS[filtered_jsonl/ shards]
    FILT --> MAN[run_manifest.json]
    FILT --> LED[curation_ledger.json]
    CORPUS --> AUD[curate/audit]
    MAN --> AUD
    LED --> AUD
    AUD --> AREPORT[audit_report.json]
    CORPUS --> DECON[curate/decontamination]
    HOLD[Holdout JSONL] --> DECON
    DECON --> DCORPUS[train_decontaminated.jsonl + decontamination_report.json]
    DCORPUS --> SUB[curate/subset]
    SUB --> TIERS[tier directories + plan.json + subset_report.json]
```

In a flow run, `flow_plan.json` records the derived paths before any step runs, and `flow_report.json` records what happened.

## The Corpus and Its Identity

`curate/ingest` establishes document identity.
When a raw corpus carries no identifier, ingestion mints one from the text content, so the same document yields the same identifier on every run and duplicates are detected rather than duplicated.
This identifier, carried in `id_field`, is the only one that survives resharding; the positional identifiers NeMo Curator can add do not.
Later artifacts that name documents, such as the decontamination report and the subset tiers, refer to them by it.

A corpus is also identified as a whole.
The profile records a `corpus.fingerprint` digest of its input, the manifest records per-shard digests of its output, and an approved policy carries the fingerprint of the corpus it was measured on.
These digests are how the pipeline detects that thresholds are being applied to data they were not calibrated for.

## The Manifest and the Ledger

The `curate/nemo_curator` step can write two accounting artifacts, and each answers a different question.

The *run manifest* (`run_manifest.json`) records what the run read and wrote: input and output counts, per-source counts when `source_field` is set, per-shard row counts and digests, the policy applied with its digest and approval status, and a `producer` block.
The producer block carries `completed_at` only if the run reached its write barrier.
A manifest without `completed_at` is the signal that a run stopped before finishing, and an auditor treats that as an incomplete corpus rather than a small one.

The *curation ledger* (`curation_ledger.json`) records why records left: how many entered, how many were written, and which gate rejected the rest.
Without it, an audit can detect that records are missing but cannot distinguish a record removed on purpose from one lost to a swallowed exception.
Because NeMo Curator does not report per-stage removal counts, removals it performs are attributed to `unattributed`, and the ledger states so in its own notes rather than naming a gate it could not observe.

Neither artifact is written unless `emit_manifest` and `emit_ledger` are set; the flow sets both.

## What the Audit Proves

`curate/audit` reads the corpus on disk and compares it with the manifest and the ledger.
Its `audit_report.json` distinguishes three strengths of claim:

- With only the corpus, the audit proves readability and computes digests; it cannot say whether anything is missing.
- With the manifest, it can prove completeness: the shards present match what the producer declared, and the run completed.
- With the ledger as well, it can attribute the difference between input and output to specific gates, and flag any loss that no gate explains (`unexplained_loss`).

An audit that passes without a ledger has not proven that no records were lost silently; it has proven only that the shards match the manifest.
The report says which inputs it had.

## Reports Are Not Corpus Artifacts

`profile_report.json`, `audit_report.json`, `decontamination_report.json`, and `subset_report.json` describe a corpus; they are not consumed as data by later steps.
The profile report carries a `producer` block with timestamps, a configuration hash, and the tool revision so it cannot be mistaken for a report about a different run, and a `profile_digest` computed over everything except that block, so re-profiling the same corpus under the same configuration does not make an approved policy look stale.

## Policies as Artifacts

`candidate_policies.yaml` is an output of profiling and an input to human review.
`approved_policy.yaml` is an output of that review and an input to filtering.
Both carry the corpus fingerprint, the language-pack content hash, and the profile digest, and the filter step verifies each before applying a threshold.
The manifest and the flow report then record the policy digest and the resulting `policy_status`, so the corpus, the decision, and the evidence remain linked.
Refer to {doc}`two-run-curation` for why the two files are distinct and {doc}`../reference/policy-file` for their schema.

## Downstream Order

Decontamination runs before subsetting so that tier budgets are counted over documents that will be trained on.
If `steps.decontamination` is disabled, the subset reads the filtered corpus directly, and the flow warns when a stale `train_decontaminated.jsonl` exists beside it.
Subset tiers nest: every document in a smaller tier is present in every larger one, so a scaling experiment compares data quantity and nothing else.

## Related Pages

- {doc}`../reference/io-format`
- {doc}`../reference/audit`
- {doc}`../reference/flow-config`
