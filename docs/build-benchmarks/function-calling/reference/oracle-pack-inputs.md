<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Oracle Pack Inputs

This page is the inventory and cross-file map for the inputs generation reads. Use the
linked artifact references for field definitions, examples, validation behavior, and
common failures.

The complete normative contract is
`src/nemotron/steps/byob/references/bfcl-oracle-pack.md`. When an operator guide and
that contract disagree, the normative contract is authoritative.

## Keep Three Input Layers Separate

1. **Authoring inputs**, such as a {doc}`domain-brief`, {doc}`probe-plan`, or reviewed
   supplement, help produce a pack. Generation does not read them.
2. **Oracle Pack inputs** define executable domain truth. Stage 1 loads and fingerprints
   them.
3. **Run configuration** selects stages, pack paths, budgets, optional model roles, and
   publication policy. See {doc}`generate-config`.

`manifest.yaml` and `task_templates.yaml` are pack declarations, not run
configuration. Editing any file inside a published pack changes its fingerprint.

## Canonical Pack Layout

```text
oracle-pack/
├── manifest.yaml             required
├── tools.json                required
├── backend.py                choose this ...
├── endpoint_config.yaml      ... or this, never both
├── fixtures.json             optional at load
├── task_templates.yaml       required
├── assertions.py             required
├── validation_cases.yaml     required
└── held_out.yaml             optional
```

`fixtures.json` is required in practice when templates bind fixture slots or a local
backend resets from fixture state. `held_out.yaml` is referenced by top-level
`manifest.held_out`.

The loader rejects explicit dual-oracle declarations and a canonical directory that
contains both `backend.py` and `endpoint_config.yaml`.

## Artifact References

Each reference follows the same sequence: creation path, contract, minimal example,
validation, common failures, and complete examples.

- {doc}`manifest` — identity, paths, languages, shared text, and confirmation.
- {doc}`tools-and-fixtures` — public function schemas and deterministic records.
- {doc}`python-backend` — the four local-oracle callables.
- {doc}`task-templates` — slots, milestones, policies, and conversation shapes.
- {doc}`assertions` — deterministic replay predicates and capabilities.
- {doc}`validation-cases` — direct success, rejection, and confirmation probes.
- {doc}`endpoint-config` — identity pins, credentials, TLS, and conformance.
- {doc}`held-out-policy` — fixture and template reservations.

## Recommended Fill Order

Each artifact constrains the next:

1. Declare the public interface in `tools.json`.
2. Implement that interface in `backend.py`, or pin a reviewed endpoint.
3. Add deterministic reset state and slot inventory in `fixtures.json`.
4. Connect identity, paths, languages, keys, and shared text in `manifest.yaml`.
5. Author conversation shapes in `task_templates.yaml`.
6. Implement the assertions those templates name.
7. Probe every tool's success and negative behavior in `validation_cases.yaml`.
8. Add `held_out.yaml` after deciding which existing ids or templates are reserved.

Run validation throughout authoring; do not wait until every file appears complete.

## Create A Complete Starter

```bash
python -m nemotron.steps.byob.scripts.scaffold_oracle_pack \
  --domain my_domain \
  --target /srv/bfcl/packs/my_domain \
  --transport python \
  --language en \
  --version 0.1.0
```

Use `--transport endpoint` to emit `endpoint_config.yaml`. Add
`--include-held-out` to emit a held-out example. The target must not already exist.

The starter is runnable plumbing, not domain truth. Replace its `get_record` names,
fixture values, behavior, tasks, assertions, and cases with independently reviewed
domain content.

## Create And Validate Each Artifact

| File | Supported creation path | Earliest useful check |
| --- | --- | --- |
| `manifest.yaml` | Pack scaffold, then manual review against sibling artifacts. | Whole-pack load and validation; no standalone manifest CLI. |
| `tools.json` | Pack scaffold or a human-reviewed existing catalog. | Whole-pack catalog/backend alignment is authoritative. |
| `backend.py` | Pack scaffold, then reviewed domain implementation. | Whole-pack isolated import, reset, and executable probes. |
| `fixtures.json` | Pack scaffold, then reviewed deterministic records. | Whole-pack reset and slot validation. |
| `task_templates.yaml` | Pack scaffold, then reviewed conversation plans. | Representative whole-pack generation. |
| `assertions.py` | Pack scaffold, then manual executable predicates. | `assertions_importable` and replay. |
| `validation_cases.yaml` | Pack scaffold, then reviewed success and rejection cases. | Whole-pack execution; there is no standalone case validator. |
| `endpoint_config.yaml` | Endpoint pack scaffold, then reviewed identity and conformance pins. | Whole-pack endpoint loading and conformance; there is no standalone config validator. |
| `held_out.yaml` | `scaffold_oracle_pack --include-held-out`, then reviewed reservations. | Whole-pack load and generation leak checks. |

Where no standalone validator exists, the file is still checked in context. A template
tool name is meaningful only relative to `tools.json`, and an assertion name only
relative to `assertions.py`.

## Validate The Complete Pack

```bash
python -m nemotron.steps.byob.scripts.validate_oracle_pack \
  --config /srv/bfcl/packs/my_domain/validate.yaml \
  --output-dir /tmp/bfcl-my-domain-validation
```

The equivalent pipeline entry is:

```bash
nemotron steps run byob/bfcl \
  -c /srv/bfcl/packs/my_domain/validate.yaml \
  stage=prepare \
  family=bfcl
```

Exit `0` means Gold-eligible. Exit `2` means validation completed with a non-Gold
verdict. Exit `1` means loading or execution prevented a verdict.

Whole-pack validation checks schema alignment, slot sources, assertion import and
execution, direct cases, confirmation, mutation declaration, deterministic reset,
timeouts, isolation, held-out bindings, endpoint conformance when applicable, and
representative replay.

## Cross-File Tool Lineage

One public name joins declarations with different responsibilities:

```text
tools.json
  declares get_record and its argument schema
        ↓
backend.py or endpoint metadata
  exposes and executes get_record
        ↓
task_templates.yaml
  makes the tool available and expects calls
        ↓
validation_cases.yaml
  probes known success and negative outcomes
        ↓
assertions.py
  decides whether replayed task behavior succeeded
```

Only the public name must match. Private backend helpers are implementation details.

## Where To Go Next

- Follow the artifact references above while filling each file.
- Use {doc}`../how-to/author-a-pack` for the manual scaffold-to-smoke sequence.
- Use {doc}`../how-to/start-from-domain-data` to choose an authoring route.
- Use {doc}`troubleshooting` to map a refusal to the responsible artifact.
