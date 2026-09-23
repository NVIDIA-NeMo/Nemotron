---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Why Nemotron curation measures a corpus before it applies thresholds, and how candidate, approved, and overridden policies differ."
topics: ["Curation", "Explanation", "Filter Policy"]
tags: ["Explanation", "Curation", "Policy"]
content:
  type: "Explanation"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Two-Run Curation: Measure, Approve, Apply

Nemotron curation separates deciding what a threshold removes from deciding whether removing it is right.
The first is a measurement the software can make; the second is a judgement a person must record.
The workflow therefore takes two runs over the same corpus with a review between them.

## The Problem With Applying Thresholds First

A heuristic filter such as a word-count bound or a non-alphanumeric ratio was calibrated on some corpus, usually English web text.
Applied to a different language or a different source, the same threshold can retain most of the corpus or almost none of it, and a single small output count does not reveal which gate was responsible.
The `empty_or_tiny_output` failure that {doc}`../reference/troubleshooting` documents is the visible symptom of this: the software can detect a surprising loss but cannot diagnose it after the fact.

Measuring first replaces that guess with a retention curve per signal: for each candidate threshold, how much of the corpus it keeps, on both a per-source (`macro`) and a per-document (`micro`) weighting.
The `curate/profile` step produces these curves without dropping a single record.

## What Measurement Does Not Establish

A retention curve is descriptive.
It answers *how much does this threshold remove*; it does not answer *is what it removes bad*.
A corpus can carry a small but valuable tail or a large body of boilerplate, and a distribution cannot distinguish them.

For that reason the profile output uses deliberate vocabulary.
A *candidate policy* is a set of thresholds derived from the measured distributions and the analysis constraints you supplied in `band_search` (`min_keep_rate`, `max_keep_rate`).
It is written to `candidate_policies.yaml` with `approved: false`, and the `curate/nemo_curator` step refuses to apply it in that state.

## Approval Is a Recorded Act

An *approved policy* is a candidate policy that a person has promoted by adding an `approval` block and setting `approved: true`.
The approval block records who approved, when, by what method (`manual` or `ablation`), and what evidence was examined.
The step validates every one of these fields before it applies a threshold, and the run manifest records the approver and the policy digest, so a curated corpus can always be traced to the decision that shaped it.

Two identities travel with the policy so that approval cannot be transferred silently:

- `corpus.fingerprint` is a digest of the input the profile measured. The flow's `approve` block verifies it against the corpus present at approval time (`verify_corpus: true` by default), and the filter step verifies it again before applying thresholds. Thresholds calibrated on one corpus are not applied to another without an explicit decision.
- `langpack.content_hash` is a digest of the language pack whose word lists and character set the signals used. If the run loads a pack with a different hash, the step stops: a stopword ratio measured against one word list is a different quantity against another.

Refer to {doc}`../reference/policy-file` for the schema.

## The Override Is Not Approval

`heuristic_filters.allow_unvalidated_policy: true` applies a policy that does not meet the approval contract.
It exists for experiments where the review has not happened and the operator accepts the consequence.
It is not a shortcut to the same result: the run logs a warning naming the policy on every execution, the manifest records `policy_status: override_unvalidated`, and `curate/audit` and the flow report carry that status forward.
A corpus produced under the override is distinguishable from an approved one in every artifact that describes it.

## Why the Flow Enforces Two Runs

The policy is promoted at preflight, before any step runs.
A configuration that sets `approve` before a candidate file exists is therefore refused (`approve_before_profile`), and when `steps.profile` is also enabled in that configuration the refusal explains that profiling in the same run cannot help: the candidates the run is about to measure are not present to approve from.
When a candidate file already exists, leaving `steps.profile` enabled during the application run is discouraged for a different reason: re-profiling replaces the candidate measurements a person reviewed with measurements nobody reviewed (`profile_enabled_during_approval`).
The two-run structure is therefore a property of the software, not only a recommendation:

1. A *measurement run* with `steps.profile.enabled: true` and no `approve` block. It ingests, profiles, applies any language and length gates written by hand, and audits. Its policy outputs are `profile_summary.md` and `candidate_policies.yaml`.
2. A person reads the summary, chooses thresholds, and writes the `approve` block into the configuration.
3. An *application run* with `steps.profile.enabled: false` and the `approve` block present. It promotes the candidate to `approved_policy.yaml`, filters, audits, decontaminates, and subsets.

{doc}`../how-to/run-the-measure-apply-flow` walks through both runs.

## When One Run Is Enough

The two-run model applies when thresholds have to be chosen.
If an approved policy already exists for this corpus and pack, or if only the language, word-count, and domain gates are needed, `curate/nemo_curator` can be run once with `heuristic_filters.approved_policy` pointing at the policy.
{doc}`../how-to/apply-a-known-policy` covers that path.

## Related Pages

- {doc}`signals-and-language-packs`
- {doc}`pipeline-artifacts`
- {doc}`../reference/profile`
- {doc}`../reference/flow-config`
