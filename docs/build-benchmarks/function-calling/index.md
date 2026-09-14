<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(build-function-calling-benchmarks-index)=
# About Building Function-Calling Benchmarks

<!-- Explanation and navigation hub for the bring your own benchmark (BYOB) BFCL series. -->

This section describes how to build a custom function-calling benchmark as Apache Parquet files with the `nemotron steps run byob/bfcl` command, and how to evaluate a candidate model against the result.

Unlike the multiple-choice flow, generation here does not ask a model to invent content.
You supply an **Oracle Pack**: a tool catalog, conversation templates, an executable backend or an HTTP endpoint, fixtures, and assertions.
The pipeline renders conversations from your templates, derives the expected tool calls, then replays every task against the real backend and checks the assertions.
A task that the oracle cannot reproduce twice does not reach the benchmark.

That difference is the point of the design. The pack, not a model, is the source of truth about what a correct tool call looks like, so the benchmark can state why each expected answer is correct.

:::{tip}
New to this flow? Follow {doc}`getting-started` once with the bundled tiny pack, then use the grids below to jump to a task guide, a concept, or a field reference.
:::

## When to Use

The `nemotron steps run byob/bfcl` command enables the following outcomes.

- A function-calling benchmark over **your own** tools, in your own domain, where each expected call is justified by an executable oracle rather than by a model's opinion.
- A repeatable Parquet artifact plus a `run_manifest.json` that pins the exact pack bytes the benchmark came from, so a published benchmark can always be traced to its source.
- Evaluation of one or more candidate models against that benchmark, either by comparing against recorded gold calls or by replaying the candidate's calls against a live oracle and running the assertions.
- Optional translation of a published benchmark into another language, as a separate run.

## Pipeline Summary

At a high level, the step performs the following work.

1. **Prepare**: normalize the configuration, load and fingerprint the pack, then validate it. Validation awards a certification tier, and a publication-eligible run requires the Gold tier.
2. **Generate**: expand templates into task instances, plan conversations, render turns, derive expected traces, validate against the tool schemas, and replay each task against the oracle. Optional stages add surface-quality checks and deduplication or balancing before publication.
3. **Translate**, optional: localize a published benchmark and write a new `benchmark.parquet`.
4. **Evaluate**, a separate run: score candidate models against a published benchmark and write a report.

See {doc}`explanation/pipeline-overview` for the stage-by-stage account.

## Choose A Way To Get A Pack

The pack is the hard part. If you have a tool interface, records, and behavior but have
not chosen a route yet, start at {doc}`how-to/start-from-domain-data`. The three flows
below converge on the same generation pipeline and the same Gold gate, which means the
trust story does not depend on how the pack was written.

| Flow | You start from | Guide |
| --- | --- | --- |
| Choose a route | Domain assets, no pack yet | {doc}`how-to/start-from-domain-data` |
| Manual | Your own knowledge of the domain and its tools | {doc}`how-to/author-a-pack` |
| Assisted, conventional source | A Python package or a reviewed HTTP service | {doc}`how-to/assisted-authoring` |
| Assisted, MCP source | A running MCP server | {doc}`how-to/mcp-server` |

In the assisted flows a model may propose pack semantics, but it can never award a certification tier, approve its own output, or bypass executable replay. {doc}`explanation/authoring-flows` explains where the human decisions sit and why they are separate.

## Documentation Series

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Tutorial
:link: getting-started
:link-type: doc
Install the `byob` extra, run the bundled tiny pack end to end, and inspect the benchmark and manifest it writes.
+++
{bdg-secondary}`hands-on`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` How-To Guides
:link: how-to/index
:link-type: doc
Author a pack by hand or with model assistance, onboard an MCP server, publish at scale, and evaluate candidates.
+++
{bdg-secondary}`task-based`
:::

:::{grid-item-card} {octicon}`light-bulb;1.5em;sd-mr-1` Concepts
:link: explanation/index
:link-type: doc
What an Oracle Pack is, how the stages fit together, where the authorization boundaries sit, and how scoring works.
+++
{bdg-secondary}`concept-focused`
:::

:::{grid-item-card} {octicon}`list-unordered;1.5em;sd-mr-1` Reference
:link: reference/index
:link-type: doc
Oracle Pack fields, backend and template contracts, run configuration, output
artifacts, and the symptom-to-fix index.
+++
{bdg-secondary}`specification`
:::

::::

## All Documentation

````{tab-set}

```{tab-item} Tutorial

| Guide | What you will do |
| --- | --- |
| {doc}`./getting-started` | Run `nemotron steps run byob/bfcl` with `tiny.yaml` and inspect the outputs |

```

```{tab-item} How-To Guides

| Guide | What you will do |
| --- | --- |
| {doc}`how-to/start-from-domain-data` | Choose manual or model-assisted authoring and turn reviewed domain assets into an executable pack |
| {doc}`how-to/author-a-pack` | Scaffold, fill in, and validate an Oracle Pack of your own |
| {doc}`how-to/assisted-authoring` | Draft a pack from a Python package or HTTP service with model assistance |
| {doc}`how-to/mcp-server` | Onboard a running MCP server as the oracle |
| {doc}`how-to/publish-a-release` | Choose a publication budget and produce a released benchmark |
| {doc}`how-to/translate` | Localize a published benchmark without changing oracle truth |
| {doc}`how-to/run-evaluation` | Score candidate models and read the evaluation report |

```

```{tab-item} Concepts

| Guide | What you will learn |
| --- | --- |
| {doc}`explanation/pipeline-overview` | What each stage consumes, transforms, writes, and requires from the operator |
| {doc}`explanation/pipeline-worked-example` | How one missing-slot task changes from an abstract template into a published row |
| {doc}`explanation/oracle-pack` | Pack layout, certification tiers, and the Gold gate |
| {doc}`explanation/authoring-flows` | The three authoring flows and the two authorization boundaries |
| {doc}`explanation/evaluation` | Trace and executable scoring, and the gates that run before inference |

```

```{tab-item} Reference

| Guide | What you will find |
| --- | --- |
| {doc}`reference/oracle-pack-inputs` | Required pack files, manifest fields, fill order, and cross-file tool lineage |
| {doc}`reference/domain-brief` | Human-owned assisted-authoring context, content, and safety rules |
| {doc}`reference/probe-plan` | Intake probe cases, A2 coverage, timeout cleanup, and `check_probe_plan` |
| {doc}`reference/manifest` | Manifest paths, languages, shared text, confirmation vocabulary, and examples |
| {doc}`reference/tools-and-fixtures` | Public tool schemas, deterministic records, and fixture-slot contracts |
| {doc}`reference/python-backend` | The four required backend callables and executable-oracle invariants |
| {doc}`reference/task-templates` | Core template fields, slots, milestones, and turn-policy examples |
| {doc}`reference/assertions` | Assertion signatures, exports, capabilities, outcomes, and failures |
| {doc}`reference/validation-cases` | Probe fields, result classes, coverage, chaining, and failures |
| {doc}`reference/endpoint-config` | Endpoint identity, credentials, TLS, attestation, and HTTP routes |
| {doc}`reference/held-out-policy` | Fixture and template reservations and leak enforcement |
| {doc}`reference/generate-config` | Generation YAML fields, block by block |
| {doc}`reference/eval-config` | Evaluation YAML fields and the three envelopes |
| {doc}`reference/output-files` | Every path written under `output_dir` / `expt_name` |
| {doc}`reference/troubleshooting` | Symptom-to-fix index drawn from the error taxonomy |

```

````

## What You Need

- A Nemotron clone with dependencies installed, including the `byob` extra from `uv sync --extra byob`.
- An Oracle Pack. To learn the flow first, use the bundled `src/nemotron/steps/byob/data/tiny_oracle_pack`, which exists to exercise the plumbing quickly.
- For evaluation, a candidate model endpoint and its credentials. Generation itself calls no model unless you explicitly enable a model-authored surface role.
- For the assisted authoring flows, a model endpoint for drafting and the corresponding feature flag, as described in {doc}`how-to/assisted-authoring`.

## Quick Start

1. Follow {doc}`getting-started` if you have not run the step yet.
2. Use {doc}`how-to/start-from-domain-data` to choose manual or model-assisted
   authoring for your own tool interface, state, and business behavior.
3. Open {doc}`reference/oracle-pack-inputs` for the required files and its standardized
   reference link for each artifact you are filling.
4. Read {doc}`explanation/oracle-pack` for the trust model and Gold tier, then follow
   the selected authoring guide.
5. Open {doc}`reference/generate-config` or {doc}`reference/eval-config` for run-level
   YAML fields.

## Limitations and Considerations

- **Executable replay costs time.** Every candidate task is reset and replayed against the oracle, twice, and its assertions are evaluated. A publication-scale run is bounded by your backend's speed, not by model latency.
- **Pack admission and publication are separate gates.** Generation refuses a pack
  that is not Gold-eligible. A Gold pack may still run under
  `lineage.policy: smoke_no_publication`; that run writes benchmark artifacts for
  plumbing checks but records that its lineage is not eligible for release. See
  {doc}`explanation/oracle-pack` and {doc}`how-to/publish-a-release`.
- **Pack code executes.** The pipeline imports and runs your backend and assertions. It does so in a separate process with a sanitized environment and enforced timeouts, and Gold requires that isolation, but the pack is still code you are choosing to trust.
- **Model roles are opt-in and pinned.** Enabling a model-authored surface role requires a pinned, unambiguous model identity, because a benchmark whose wording came from an unrecorded model cannot be reproduced.
- **The MCP transport is experimental.** Only Mode A is implemented, and it is disabled unless you opt in. See {doc}`how-to/mcp-server`.
