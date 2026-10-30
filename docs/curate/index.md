---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Curate multilingual text corpora with the six Nemotron curate steps: ingest, profile, filter, audit, decontaminate, and subset."
topics: ["Curation", "NeMo Curator", "JSONL"]
tags: ["Curation", "Documentation"]
content:
  type: "Explanation"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

(curate-index)=
# About Data Curation

The `curate` category turns a raw text corpus into a filtered, verified, decontaminated training set with a recorded rationale for every threshold that was applied.
It consists of six registered steps built on NeMo Curator and a flow driver that runs them from one configuration.

| Step | What it does | Drops rows |
| --- | --- | --- |
| `curate/ingest` | Normalizes Parquet or JSONL, mints content-derived document identifiers, and maps source columns | No |
| `curate/profile` | Measures quality-signal distributions on the unfiltered corpus and writes candidate thresholds | No |
| `curate/nemo_curator` | Applies language, word-count, and domain gates and an approved filter policy | Yes |
| `curate/audit` | Verifies the filtered corpus against the manifest and ledger the filter step wrote | No |
| `curate/decontamination` | Removes training documents that overlap a held-out evaluation set | Yes |
| `curate/subset` | Cuts nested token-budget tiers for scaling experiments | No |

The steps run individually with `nemotron steps run`, or together through the flow driver.

## Two Runs, Not One

Choosing a threshold requires knowing what it removes.
The flow therefore runs twice over the same corpus:

1. A *measurement run* ingests and profiles the corpus and writes `candidate_policies.yaml` with `approved: false`. Nothing is filtered.
2. A person reads `profile_summary.md`, chooses thresholds, and records the approval in the configuration.
3. An *application run* promotes the candidate to an approved policy, filters, audits, decontaminates, and subsets.

The filter step refuses a policy that has not been approved.
{doc}`explanation/two-run-curation` explains why the two runs cannot be collapsed into one.

```{mermaid}
flowchart LR
    RAW[Raw Parquet or JSONL] --> ING[ingest]
    ING --> PROF[profile]
    PROF --> REV{{Review and approve}}
    REV --> FILT[filter]
    FILT --> AUD[audit]
    FILT --> DEC[decontamination]
    DEC --> SUB[subset]
    SUB --> OUT[Tiered training corpus]
```

## Choosing an Entry Point

| Situation | Start with |
| --- | --- |
| A new corpus or a new language, and thresholds have not been chosen | {doc}`how-to/run-the-measure-apply-flow` |
| An approved policy already exists for this corpus and language pack | {doc}`how-to/apply-a-known-policy` |
| Only language, word-count, or domain gates are needed, without a policy | {doc}`how-to/enable-filters` |
| A filtered corpus exists and its integrity has to be verified | {doc}`how-to/audit-a-curated-corpus` |
| A filtered corpus has to be protected against evaluation leakage or cut into tiers | {doc}`how-to/decontaminate-and-subset` |
| An approved policy has to be scored against labelled documents | {doc}`how-to/evaluate-a-policy` |
| A first local run to verify the installation | {doc}`getting-started` |

```{note}
The category does not crawl web pages, extract Common Crawl WARC files, or run corpus-wide deduplication.
Use a dedicated NeMo Curator recipe for those jobs before ingestion.
```

## Documentation Series

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Tutorial
:link: getting-started
:link-type: doc
Install the Nemotron CLI, run a local tiny JSONL verification, and inspect output shards.
+++
{bdg-secondary}`hands-on`
:::

:::{grid-item-card} {octicon}`light-bulb;1.5em;sd-mr-1` Explanation
:link: explanation/index
:link-type: doc
Why curation measures before it filters, how signals depend on language packs, and what each artifact proves.
+++
{bdg-secondary}`concepts`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` How-To Guides
:link: how-to/index
:link-type: doc
Run the measure and apply flow, apply a known policy, audit, decontaminate, subset, and evaluate.
+++
{bdg-secondary}`task-based`
:::

:::{grid-item-card} {octicon}`list-unordered;1.5em;sd-mr-1` Reference
:link: reference/index
:link-type: doc
Per-step parameters and errors, flow configuration, policy schema, CLI syntax, and troubleshooting.
+++
{bdg-secondary}`lookup`
:::

::::

## All Documentation

````{tab-set}

```{tab-item} Tutorial

| Guide | What you do |
| --- | --- |
| {doc}`getting-started` | Run `curate/nemo_curator` on the packaged tiny JSONL fixture |

```

```{tab-item} Explanation

| Page | Question it answers |
| --- | --- |
| {doc}`explanation/two-run-curation` | Why is measurement separated from application, and what does approval record? |
| {doc}`explanation/signals-and-language-packs` | Which signals transfer between languages, and what does a language pack supply? |
| {doc}`explanation/pipeline-artifacts` | What do the manifest, ledger, reports, and policies each prove? |

```

```{tab-item} How-To Guides

| Guide | Focus |
| --- | --- |
| {doc}`how-to/run-the-measure-apply-flow` | Measurement run, review, and application run with the flow driver |
| {doc}`how-to/apply-a-known-policy` | Single-step filtering with an existing approved policy |
| {doc}`how-to/audit-a-curated-corpus` | Integrity, completeness, and attribution checks |
| {doc}`how-to/decontaminate-and-subset` | Holdout overlap removal and nested token-budget tiers |
| {doc}`how-to/evaluate-a-policy` | Precision and recall of an approved policy on labelled documents |
| {doc}`how-to/enable-filters` | Language, word-count, and domain gates |
| {doc}`how-to/run-local-jsonl` | Local JSONL reader/writer path |
| {doc}`how-to/use-huggingface-snapshot` | `dataset` block and Hugging Face snapshot download |

```

```{tab-item} Reference

| Page | Content |
| --- | --- |
| {doc}`reference/cli-curate` | Command syntax for every step and the flow driver |
| {doc}`reference/ingest` | `curate/ingest` parameters, outputs, and errors |
| {doc}`reference/profile` | `curate/profile` parameters, signals, outputs, and errors |
| {doc}`reference/curate-config` | `curate/nemo_curator` parameters and gate semantics |
| {doc}`reference/audit` | `curate/audit` parameters, modes, and errors |
| {doc}`reference/decontamination` | `curate/decontamination` parameters and errors |
| {doc}`reference/subset` | `curate/subset` parameters and errors |
| {doc}`reference/flow-config` | Flow configuration file, derived paths, and preflight checks |
| {doc}`reference/policy-file` | Candidate and approved policy schema |
| {doc}`reference/io-format` | Input records and the artifacts steps exchange |
| {doc}`reference/troubleshooting` | Error identifiers grouped by cause |

```

````

## What You Need

- A corpus in Parquet or JSONL with one text field, usually named `text`.
- The BCP-47 tag of the corpus language and a reviewed language pack for it. Nemotron ships example packs for `en`, `vi`, and `hi` and selects none of them implicitly.
- Optional model assets when gates are enabled, such as a FastText language identification model for `language_codes`.
- A held-out evaluation set when decontamination is enabled, and one GPU for its similarity pass.
- A writable output root.
