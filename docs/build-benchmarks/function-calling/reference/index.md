<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Reference

Field-level and artifact-level detail for `nemotron steps run byob/bfcl` with the `bfcl` family.

```{toctree}
:maxdepth: 1
:hidden:

Oracle Pack Inputs <oracle-pack-inputs>
Domain Brief <domain-brief>
Data Designer Provider Configuration <data-designer-provider>
Probe Plan <probe-plan>
Manifest <manifest>
Tool Catalog And Fixtures <tools-and-fixtures>
Python Backend <python-backend>
Task Templates <task-templates>
Assertions <assertions>
Validation Cases <validation-cases>
Endpoint Configuration <endpoint-config>
Held-Out Policy <held-out-policy>
Generation Config <generate-config>
Evaluation Config <eval-config>
Output Files <output-files>
Troubleshooting <troubleshooting>
```

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Oracle Pack inputs
:link: oracle-pack-inputs
:link-type: doc
Required files, manifest fields, fill order, and how tool names connect the pack.
+++
{bdg-secondary}`pack files`
:::

:::{grid-item-card} {octicon}`note;1.5em;sd-mr-1` Domain brief
:link: domain-brief
:link-type: doc
Required assisted-authoring context, content boundaries, safety rules, and a complete example.
+++
{bdg-secondary}`authoring input`
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Probe plan
:link: probe-plan
:link-type: doc
Intake cases, A2 coverage including timeout cleanup, `check_probe_plan`, and common refusals.
+++
{bdg-secondary}`authoring input`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Manifest
:link: manifest
:link-type: doc
Identity, paths, languages, shared text, confirmation vocabulary, and field examples.
+++
{bdg-secondary}`manifest.yaml`
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Tool catalog and fixtures
:link: tools-and-fixtures
:link-type: doc
Public function schemas, deterministic reset records, minimal examples, and validation.
+++
{bdg-secondary}`JSON`
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Python backend
:link: python-backend
:link-type: doc
The four required callables, deterministic reset, structured errors, and confirmation behavior.
+++
{bdg-secondary}`backend.py`
:::

:::{grid-item-card} {octicon}`list-unordered;1.5em;sd-mr-1` Task templates
:link: task-templates
:link-type: doc
Core fields, slot binding, milestones, turn policies, and complete conversation shapes.
+++
{bdg-secondary}`task_templates.yaml`
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Assertions
:link: assertions
:link-type: doc
Signatures, exports, capabilities, outcome semantics, examples, and failure reasons.
+++
{bdg-secondary}`assertions.py`
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Validation cases
:link: validation-cases
:link-type: doc
Case fields, result classes, coverage, chaining, examples, and executable checks.
+++
{bdg-secondary}`validation_cases.yaml`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Endpoint configuration
:link: endpoint-config
:link-type: doc
Pinned identity, credentials, TLS, attestation, HTTP routes, and validation failures.
+++
{bdg-secondary}`endpoint_config.yaml`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Held-out policy
:link: held-out-policy
:link-type: doc
Reserved fixture and template ids, backend-state policy, validation, and leak checks.
+++
{bdg-secondary}`held_out.yaml`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Generation config
:link: generate-config
:link-type: doc
Every accepted key in a generation YAML, block by block, with types and defaults.
+++
{bdg-secondary}`YAML`
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Evaluation config
:link: eval-config
:link-type: doc
Scoring contract, modes, candidate declarations, and the difference between the three bundled envelopes.
+++
{bdg-secondary}`YAML`
:::

:::{grid-item-card} {octicon}`file-directory;1.5em;sd-mr-1` Output files
:link: output-files
:link-type: doc
Every path written under `output_dir` / `expt_name`, and which artifacts are content-addressed.
+++
{bdg-secondary}`artifacts`
:::

:::{grid-item-card} {octicon}`bug;1.5em;sd-mr-1` Troubleshooting
:link: troubleshooting
:link-type: doc
Symptom-to-fix index derived from the step's declared error taxonomy.
+++
{bdg-secondary}`errors`
:::

::::

## Bundled Configurations

The step ships runnable configurations under `src/nemotron/steps/byob/bfcl/config/`. Copy the one closest to your intent rather than starting from an empty file.

Every number in the example files is a worked example for the pack that file points at, not a framework default. Copying another pack's task counts and diversity limits is the most common way to make the balancing stage infeasible.

| File | Purpose |
| --- | --- |
| `tiny.yaml` | Plumbing smoke run against the bundled tiny pack. Not publication-eligible. |
| `default.yaml` | Annotated template. Its pack path is a placeholder, so it cannot publish an example domain by omission. |
| `smoke.example.yaml` | Domain-sized smoke run. Copy it and repoint it at your own pack. |
| `publication.example.yaml` | Publication-scale, template-only Gold profile with a worked budget and balancing targets. |
| `publication.paraphrase.example.yaml` | The same executable cases with an opt-in model-authored surface role. |
| `eval.default.yaml` | Annotated evaluation template to resolve into your own config. |
| `eval.cli.yaml` | Direct evaluation envelope. |
| `eval.launcher.yaml` | Launcher evaluation envelope. |
| `translate.yaml` | Localization of an already published benchmark. |

## Authoring Input Examples

Those configurations drive generation, which starts only once a reviewed pack exists. The inputs the authoring flows take *before* generation ship as worked examples under `src/nemotron/steps/byob/references/`, taken from a published release.

| File | Used by | Purpose |
| --- | --- | --- |
| `bfcl-domain-brief.example.txt` | `--brief` | The reviewed statement of what a source is for, sanitized and bound into the evidence. See {doc}`domain-brief`. |
| `bfcl-domain-brief.skeleton.txt` | copy, complete, then pass to `--brief` | The same thing with the domain taken out: an instruction header to delete, then seven bracketed blocks to replace, each naming what it feeds downstream. Intake rejects any remaining `BFCL-SKELETON` block, so do not pass this file itself. Start with {doc}`domain-brief` rather than editing the banking example. |
| `bfcl-probe-plan.example.json` | `--probe-plan` | A complete plan for certification tier A2: a success per published tool, a state-changing case per mutating tool, structured errors naming their codes, and the required timeout case. Its `fixtures` block is abridged to the records its own cases reach. See {doc}`probe-plan`. |
| `bfcl-authoring-policy.example.yaml` | policy | Organizational defaults a guided session should not ask for twice. |
| `bfcl-endpoint-config.example.yaml` | `endpoint_config.yaml` | A complete endpoint-backed pack declaration, with credentials referenced by environment-variable name only. |

## Command-line Conventions

These pages use `nemotron steps run byob/bfcl -c <CONFIG> stage=<STAGE>`. The same code path
is reachable directly as `python -m nemotron.steps.byob.scripts.run --config <CONFIG>
--stage <STAGE>`, which is convenient when the `nemotron` console script is not on `PATH`.
Every shipped configuration declares `family: bfcl`, so the direct form needs no `--family`;
a configuration that omits the key falls back to the MCQ family, so keep it declared in
configs you write yourself.

The helper commands under `nemotron.steps.byob.scripts` — the pack validator, the bias auditor, the release archiver, the authoring and MCP release commands — share one exit contract, so a wrapper can branch on the status alone:

| Status | Meaning | Output |
| --- | --- | --- |
| `0` | The command ran and the answer was yes: the pack is Gold-eligible, the audit passed, the artifact was written. | The result document on stdout. |
| `1` | The command could not reach an answer. A path was missing, a file would not parse, an invariant was violated. | A JSON failure envelope on stderr with `status`, `error_type`, and `reason`. |
| `2` | The command ran and the answer was no. The pack is not Gold-eligible, the audit found an unexcepted failure, the review packet is blocked. | The full verdict document on stdout, so you can see which check said no. |

The distinction between `1` and `2` is what makes these commands safe to automate: retry on `1`, because a crash may be transient; never retry on `2`, because the verdict will not change until a human changes the inputs.

The evaluator is the exception. `nemotron steps run byob/bfcl` with `stage=eval` publishes a wider taxonomy — `2` through `7` — because an operator needs to know whether to edit a config, fix a candidate endpoint, or investigate a contamination finding. See {doc}`../how-to/run-evaluation`.

## Normative Contracts

These pages describe the operator-facing surface. The normative contracts live in the source tree, beside the code that enforces them:

| Contract | File |
| --- | --- |
| Oracle Pack layout, tiers, and Gold rules | `src/nemotron/steps/byob/references/bfcl-oracle-pack.md` |
| Evaluation scoring | `src/nemotron/steps/byob/references/bfcl-eval-scoring-contract.md` |
| Bias audit dimensions | `src/nemotron/steps/byob/references/bfcl-bias-audit-contract.md` |
| MCP oracle profile | `src/nemotron/steps/byob/references/bfcl-mcp-oracle-contract.md` |
| MCP trust boundaries | `src/nemotron/steps/byob/references/bfcl-mcp-threat-model.md` |
| Supported, experimental, and refused capabilities | `src/nemotron/steps/byob/references/bfcl-authoring-support-matrix.md` |
| Supported, experimental, and refused MCP transports | `src/nemotron/steps/byob/references/bfcl-mcp-support-matrix.md` |
| Shared source-intake spine and evidence envelope | `src/nemotron/steps/byob/references/bfcl-transport-neutral-intake.md` |
| Review, approval, and freeze record shapes | `src/nemotron/steps/byob/references/bfcl-authoring-release-v2.md` |
| Adapter enablement policy and `BFCL_ENABLE_*` variables | `src/nemotron/steps/byob/references/bfcl-authoring-rollout.md` |
| End-to-end manual pack lifecycle, including endpoint pins | `src/nemotron/steps/byob/references/bfcl-manual-oracle-pack-flow.md` |
| Source package layouts and the dependency-lock format | `src/nemotron/steps/byob/references/bfcl-conventional-source-packages.md` |
| Certification tiers and the stable refusal-code registry | `src/nemotron/steps/byob/references/bfcl-source-adapter-certification-profiles.md` |
| Credential-reference lifecycle and authorization digests | `src/nemotron/steps/byob/references/bfcl-authoring-credentials.md` |
| Authoring event-log payload allowlist | `src/nemotron/steps/byob/references/bfcl-authoring-events.md` |
| Release revocation registry | `src/nemotron/steps/byob/references/bfcl-authoring-revocation.md` |
| Cache retention and the `purge-cache` audit record | `src/nemotron/steps/byob/references/bfcl-authoring-cache-retention.md` |
| Why MCP reaches the pipeline through a gateway | `src/nemotron/steps/byob/references/bfcl-mcp-architecture-decision.md` |

The step's own declared inputs, outputs, and error taxonomy are in `src/nemotron/steps/byob/bfcl/step.toml`.
