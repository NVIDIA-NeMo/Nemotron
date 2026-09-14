<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# The Oracle Pack

An oracle pack is the unit of truth for a function-calling benchmark.
It holds the tool schemas a model may call, an executable oracle that actually performs those calls, the templates that describe the conversations, and the assertions that decide whether a replay really succeeded.
All domain content lives in the pack; the generation runtime under `runtime/benchmark_families/bfcl` stays domain-agnostic and contains no branch for any particular domain.

That split is what makes the pipeline reusable. Onboarding a new domain means authoring a pack, not modifying the pipeline, and it also means the pipeline can make claims about a benchmark — that its gold calls execute, that its assertions pass, that its rows are reproducible — without knowing anything about the subject matter.

## Canonical File Layout

A pack is a directory under an `oracle_runtime.allowed_roots` entry. File locations come from `manifest.yaml.paths`, from `oracle_pack.*` overrides in the run configuration, or from these defaults:

| File | Required | Purpose |
| --- | --- | --- |
| `manifest.yaml` | Yes | Identifies the pack (`pack_id`, `version`) and declares languages, path overrides, primary keys, absent ids, prompts, assistant turn text, and confirmation vocabulary. |
| `tools.json` | Yes | The model-facing function schemas, plus the pack-local `x-mutates` and `x-requires-confirmation` annotations that the pipeline reads but never exposes to a model. |
| `task_templates.yaml` | Yes | Intents, slot sources, conversation policies, milestones, and the success assertions each template claims. |
| `validation_cases.yaml` | Yes | Positive and negative probes proving backend and schema alignment, determinism, error shapes, and confirmation behavior. |
| `assertions.py` | Yes | Deterministic checks over the final oracle state and the executed trace. |
| `backend.py` | One oracle required | A local executable oracle exposing `list_tools`, `reset`, `call_tool`, and `get_state`. |
| `endpoint_config.yaml` | One oracle required | A pinned HTTPS BFCL Oracle HTTP v1 service. |
| `fixtures.json` | Optional | Deterministic records and the initial oracle state passed to `reset`. |
| `held_out.yaml` | Optional | Fixture primary ids and template ids reserved out of ordinary generation. |

Exactly one oracle is allowed: `backend.py` or `endpoint_config.yaml`, never both.
Two oracles would make it impossible to say which one certified a row, so the ambiguity is refused rather than resolved by a precedence rule.
An endpoint pack pins the remote oracle's id, version, and content digest, and that identity must still match during validation, replay, and publication; only HTTPS is accepted and credentials are referenced by environment-variable name rather than stored in the pack.

Two files carry more weight than their size suggests.
`validation_cases.yaml` is what turns "the backend seems to work" into observed evidence, because every tool needs at least one success probe and one negative probe before the pack can be certified.
`assertions.py` is what turns "the trace ran" into "the trace was right": a template with no success assertion has no statement of what success means, so replay could only confirm that its calls executed.

## Certification Tiers And The Gold Gate

`stage=prepare` normalizes the pack and writes `oracle_validation_report.json` containing a tier, the gold-eligibility verdict, the pack fingerprint, per-check failures, and pack statistics.
The checks cover template tool references, slot sources, backend and schema alignment, assertion importability, the declared validation probes, confirmation policy, and a representative generation contract that expands, renders, replays, and asserts the first deterministic instance of every template.

The tier is derived from those individual checks rather than read from a summary flag:

| Tier | Meaning |
| --- | --- |
| `gold` | Every check passed and the pack has an oracle, templates, and assertions. Gold-eligible. |
| `silver` | The pack has templates and tools but at least one check did not pass. Not gold-eligible. |
| `prototype` | The pack does not yet reach silver. Not gold-eligible. |

`stage=generate` refuses a pack that is not gold-eligible.
A check whose preconditions failed is recorded as `skipped`, never as a pass, so an unrun check keeps a pack below gold instead of letting it inherit one.
The report on disk is a human-readable artifact, not a signed attestation, so generation never trusts one written by an earlier run: it reuses a verdict only when the same process produced it for the same pack and configuration fingerprints.

Because the gold gate is what a released benchmark's credibility rests on, it is worth saying what it is *not*: it certifies that the pack generates and that its own claims hold under execution. It says nothing about whether the domain modeling is a good benchmark of anything.
That judgment stays with the reviewer; {doc}`../how-to/author-a-pack` describes the manual review boundary.

:::{important}
Gold eligibility requires `oracle_runtime.worker: process`.
A run may configure `worker: thread` as a debugging aid, but such a run can never reach gold.
:::

## Pack Code Runs In A Separate Process

Pack code is executed through a process worker, never inside the process that scores a candidate.
That boundary exists for three separate reasons, and none of them is redundant.
A separate process is the only place a hanging tool can be stopped on a hard deadline.
It is also what sanitizes the environment, so a backend cannot read the caller's environment or wall-clock time and must instead take the frozen clock, seed, timeout, and task id the pipeline hands it.

Errors follow the same logic. A tool returns a failure as data — a structured `{"error": {"code": ...}}` envelope — rather than raising, because a domain rejection is a legitimate outcome the benchmark wants to score, and an exception would be indistinguishable from infrastructure breaking.

## The Fingerprint Pins A Benchmark To Its Source

Generation records a pack fingerprint covering every file in the pack tree, along with a per-file hash map, and verifies it before validation, after validation, and again before final output.

The whole tree counts, including files that look inert. A helper module the backend imports changes what the oracle does, and there is no read sandbox that would make a Markdown file provably unreadable to a backend that can open its own directory.
The consequence is worth planning for: publishing a benchmark freezes the pack directory.
Keep operational notes outside the pack, and publish a new release rather than editing a frozen pack.

The aggregate fingerprint proves only that something moved; the per-file map is what lets a drift report name the file and say whether any declared oracle input was involved.
Its versioned serialization length-prefixes each logical path and hashes a fixed-width
SHA-256 digest for each file, so delimiters inside file bytes cannot create the same
aggregate as a different file layout.
Executable Python bytecode (`__pycache__`, `.pyc`, and `.pyo`) is rejected rather than
ignored: unchecked-hash bytecode can override changed source and therefore cannot sit
outside the reviewed fingerprint.
Anything a pack imports from outside its own tree is invisible to the fingerprint, which is why a pack should keep its dependencies inside itself wherever it can.

## The Bundled Packs

Two packs ship under `src/nemotron/steps/byob/data/` and serve different purposes.

`tiny_oracle_pack` is the smallest end-to-end example. It covers single-turn, confirmation, parallel call-group, and irrelevant shapes with no model calls, which makes it the right pack for checking that plumbing, isolation, and output paths work before a real pack exists.

`banking_vn_oracle_pack` is the domain-scale reference. It is worth reading before authoring your own, because it declares a template for every conversation shape the pipeline supports — `single_turn`, `missing_slot`, `confirmation`, `correction`, `multi_tool`, `dependent_call`, `negative_path`, `clarify_only`, and `irrelevant` — and no template narrows `tools_present`, so every row must select its calls out of the full tool catalog.
Read it as a worked example of the pack contract rather than as a default: its inventory, scale, and mix are properties of that pack, not of the framework.

## Related Information

- {doc}`../reference/oracle-pack-inputs` for the developer-facing file inventory,
  manifest fields, and fill order.
- {doc}`../reference/manifest` and {doc}`../reference/tools-and-fixtures` for the
  foundational declarations and examples.
- {doc}`../reference/python-backend`, {doc}`../reference/task-templates`,
  {doc}`../reference/assertions`, and {doc}`../reference/validation-cases` for the
  executable and conversation contracts.
- {doc}`../reference/endpoint-config` and {doc}`../reference/held-out-policy` for
  remote-oracle and reservation fields.
- `src/nemotron/steps/byob/references/bfcl-oracle-pack.md` for the complete normative pack contract, including slot sources, turn policies, and every validation rule.
- {doc}`../how-to/author-a-pack` for the hands-on authoring sequence.
- {doc}`pipeline-overview` for how the pipeline consumes a validated pack.
- {doc}`../getting-started` for a first run against a bundled pack.
