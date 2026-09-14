<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Manifest Fields

`manifest.yaml` identifies an Oracle Pack and connects its files, languages, fixture
keys, shared text, confirmation vocabulary, and optional held-out policy. It is a pack
declaration, not a generation configuration.

The runtime requires `pack_id` and `version` to load a manifest. Other fields become
required only when the pack uses the behavior they configure. Prefer stable string
values for identity even though the loader normalizes scalar identity values.

## Create A Manifest

Create a complete runnable pack, including a starter manifest, with:

```bash
python -m nemotron.steps.byob.scripts.scaffold_oracle_pack \
  --domain my_domain \
  --target /srv/bfcl/packs/my_domain \
  --transport python \
  --language en \
  --version 0.1.0
```

Use `--transport endpoint` for an HTTP oracle and `--include-held-out` when the pack
needs a held-out policy. Review the manifest against the executable oracle and pack
artifacts before validation.

## Field Contract

| Field | Required when | Meaning |
| --- | --- | --- |
| `pack_id` | Always | Stable pack identifier carried into fingerprints and row provenance. |
| `version` | Always | Pack revision carried into provenance. Change it when publishing changed pack bytes. |
| `paths` | Optional | Pack-relative filename overrides. Canonical filenames are used when entries are omitted. |
| `description` | Optional | Human-readable pack metadata; it does not configure runtime behavior. |
| `languages` | Optional | Preferred language codes available in user and assistant surfaces. |
| `default_language` | Optional | Preferred default rendered language. |
| `clock` | Optional | Intended frozen time as unvalidated metadata. Execution uses `oracle_runtime.clock` from run configuration. |
| `primary_keys.<collection>` | When fixture key inference is ambiguous | Field that uniquely identifies a row in one fixture collection. |
| `absent_ids.<collection>` | When a template uses an `absent:` source | Author-declared identifiers intended not to occur in that collection. |
| `assistant_turn_templates.<type>.<lang>` | For text milestones not overridden by a task | Shared `ask_for_slot`, `ask_confirm`, `decline`, and `final_answer` text. |
| `system_prompt` | Optional | Inline model-facing system prompt; a non-empty value takes precedence over a path. |
| `system_prompt_path` | Optional | Pack-relative system-prompt file used when no non-empty inline prompt is declared. |
| `confirmation` | Optional | Names the confirmation parameter, result status field, and pending status. |
| `surface_guards.tool_names_exempt` | Only for tool names that are ordinary domain language | Allows those names in user-facing text without a leakage finding. |
| `held_out` | When reservations apply | Pack-relative path to `held_out.yaml`; it is not part of `paths`. |

`paths` accepts these keys:

| Key | Canonical target |
| --- | --- |
| `tools` | `tools.json` |
| `backend` | `backend.py` |
| `endpoint` | `endpoint_config.yaml` |
| `fixtures` | `fixtures.json` |
| `templates` | `task_templates.yaml` |
| `assertions` | `assertions.py` |
| `validation_cases` | `validation_cases.yaml` |

Declare exactly one of `paths.backend` and `paths.endpoint`. If neither path is
explicit, the loader detects the canonical file and still refuses both-or-neither
layouts.

When no run language is selected, rendering tries `default_language`, then the first
manifest language that has a surface, then the sole available surface language.
Multiple unresolved surface languages require an explicit run selection.

`confirmation` has three optional string fields:

```yaml
confirmation:
  parameter: confirm
  status_field: status
  pending_status: awaiting_confirmation
```

These are the defaults. A confirmation-gated tool schema, backend result, validation
cases, and task templates must use the same vocabulary.

## Minimal Example

```yaml
pack_id: my_domain
version: "0.1.0"
languages: [en]
clock: "2026-01-01T00:00:00Z"
paths:
  tools: tools.json
  backend: backend.py
  fixtures: fixtures.json
  templates: task_templates.yaml
  assertions: assertions.py
  validation_cases: validation_cases.yaml
primary_keys:
  records: record_id
absent_ids:
  records: [REC-ABSENT-1]
assistant_turn_templates:
  ask_for_slot:
    en: "Please provide the {slot_name}."
  ask_confirm:
    en: "Please confirm before I continue."
  decline:
    en: "The available tools cannot complete that request."
  final_answer:
    en: "Done."
```

The example assumes matching sibling files. It is not a complete pack by itself.

## Validate A Manifest

There is no standalone manifest validator because most claims refer to sibling files.
Run whole-pack preparation:

```bash
python -m nemotron.steps.byob.scripts.validate_oracle_pack \
  --config /srv/bfcl/packs/my_domain/validate.yaml \
  --output-dir /tmp/bfcl-my-domain-validation
```

Manifest parsing and path resolution happen before the validation report can be
completed. Exit `1` means the validator could not load or execute the contract; exit
`2` means validation completed but the pack was not Gold-eligible; exit `0` means
Gold-eligible.

## Common Failures

- **Missing sibling file:** correct the relevant `paths` value or restore the file.
- **Both backend and endpoint declared:** select one oracle transport.
- **Invalid confirmation mapping:** use only `parameter`, `status_field`, and
  `pending_status`, each with a non-empty string.
- **Missing rendered language:** add the selected language to every required user and
  assistant text block, or select one in run configuration.
- **Unexpected time behavior:** `manifest.clock` is metadata and is not compared with
  execution. Set `oracle_runtime.clock` and make backend code use `ctx.clock`.
- **Declared absent id exists in fixtures:** `absent_ids` is an author assertion rather
  than a loader proof. Review it against fixture primary keys before publication.
- **Missing held-out file:** point top-level `held_out` at an existing policy.

## Complete Example

See `src/nemotron/steps/byob/data/tiny_oracle_pack/manifest.yaml`. The Banking VN pack
contains additional domain-local metadata; keys used only by one pack are not
automatically framework fields.

## Related Information

- {doc}`oracle-pack-inputs` for the complete file map and fill order.
- {doc}`tools-and-fixtures` for the catalog and reset data referenced here.
- {doc}`endpoint-config` for endpoint identity and security fields.
- {doc}`held-out-policy` for the file referenced by `held_out`.
