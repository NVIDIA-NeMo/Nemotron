<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Tool Catalog And Fixtures

`tools.json` is the public function interface shown to a candidate model.
`fixtures.json` supplies deterministic reset records and values that task slots can
bind. The catalog states what may be called; fixtures do not define what a call means.
Behavior belongs in `backend.py` or the pinned endpoint.

## Create The Files

`scaffold_oracle_pack` writes a matching catalog and fixture collection:

```bash
python -m nemotron.steps.byob.scripts.scaffold_oracle_pack \
  --domain my_domain \
  --target /srv/bfcl/packs/my_domain \
  --transport python \
  --language en \
  --version 0.1.0
```

Treat `tools.json` as a human-reviewed public contract. Keep fixture records and
backend behavior aligned with the reviewed function names and schemas.

The optional `--draft-with-model` lane may propose fixture values before source
certification. It does not rewrite `tools.json`. Prefer domain-owned records and an
independently implemented source: model-proposed distributions can introduce
benchmark-construction bias even when the output is deterministic.

## `tools.json` Field Contract

The root is a JSON array with unique public function names.

| Field | Required | Meaning |
| --- | --- | --- |
| `type` | Optional | Defaults to `function`; no other value is accepted. |
| `function` | Yes | Function declaration object. |
| `function.name` | Yes | Non-empty public name used by templates, cases, and the oracle. |
| `function.description` | Optional | Model-facing description. It must be a string. |
| `function.strict` | Optional | Boolean strict-mode declaration. |
| `function.parameters` | Required for Gold | JSON Schema object for call arguments. Its type defaults to `object`. |
| `x-mutates` | Optional | Pack-only boolean declaring observable state mutation. |
| `x-requires-confirmation` | Optional | Pack-only boolean declaring confirmation-gated behavior. |

The two `x-*` fields are removed from the catalog shown to the candidate. A tool marked
`x-requires-confirmation` must include the manifest's confirmation parameter and must
not mutate state when that argument is false.

The validator supports a bounded JSON Schema subset and rejects unsupported or
ambiguous constructs such as `anyOf`, `oneOf`, and `not`. Keep schemas explicit:
declare property types, required fields, bounds or enums where meaningful, and
`additionalProperties: false` when undeclared arguments should be refused.

Tool error codes are domain-defined strings, not a framework enum.

## Minimal Tool Example

```json
[
  {
    "type": "function",
    "function": {
      "name": "get_record",
      "description": "Return one record by its stable identifier.",
      "parameters": {
        "type": "object",
        "properties": {
          "record_id": {"type": "string"}
        },
        "required": ["record_id"],
        "additionalProperties": false
      }
    }
  }
]
```

The same public name must appear in `backend.py` or endpoint metadata and wherever the
tool is referenced in templates and validation cases.

## `fixtures.json` Contract

The root is a JSON object. A conventional fixture-backed pack maps collection names to
arrays of deterministic row objects:

```json
{
  "records": [
    {
      "record_id": "REC-001",
      "name": "Example record",
      "status": "active"
    }
  ]
}
```

A source such as `fixture:records.record_id` selects `record_id` from this collection.
When key inference is ambiguous, declare `primary_keys.records: record_id` in
`manifest.yaml`.

`fixtures.json` is load-optional. It is required in practice when templates use
fixture-backed slots or a local backend resets from fixture state. Use stable synthetic
or approved records, remove secrets and personal data, and ensure every episode can
reset from a defensive copy.

Author-declared absent ids belong under `manifest.absent_ids`; review them against
fixture primary keys because the loader does not prove absence. Existing ids reserved
from normal generation belong in `held_out.yaml`. They are different contracts.

## Validate The Files

For a conventional local source, run the static pre-check:

```bash
python -m nemotron.steps.byob.scripts.check_source_package \
  --source /srv/sources/my-domain
```

It checks catalog/backend name alignment, fixture root shape, and unresolved
`BFCL-TODO` markers. It does not prove behavior or Gold eligibility.

Run whole-pack validation after catalog, fixture, backend, or template changes:

```bash
python -m nemotron.steps.byob.scripts.validate_oracle_pack \
  --config /srv/bfcl/packs/my_domain/validate.yaml \
  --output-dir /tmp/bfcl-my-domain-validation
```

Gold validation checks schema support, tool implementation alignment, success and
negative case coverage, fixture slot sources, primary keys, mutation declarations,
confirmation behavior, deterministic reset, and representative replay.

## Common Failures

- **`schema_without_implementation`:** a catalog name is absent from the oracle.
- **`implementation_without_schema`:** the oracle exposes an undeclared public name.
- **Unsupported schema keyword:** simplify the parameter schema to the supported
  deterministic subset.
- **`unknown_collection`:** a template names a collection absent from fixtures.
- **`fixture_row_missing_primary_key` or ambiguous key:** add consistent row keys and
  `manifest.primary_keys`.
- **`filter_matches_zero`:** correct the fixture values or template filter.
- **`undeclared_mutation`:** set `x-mutates: true` when successful calls change state.
- **Incomplete validation coverage:** add at least one success and one negative case for
  each public tool.

## Complete Examples

- `src/nemotron/steps/byob/data/tiny_oracle_pack/tools.json`
- `src/nemotron/steps/byob/data/tiny_oracle_pack/fixtures.json`

The names and library records in that pack are illustrative, not defaults.

## Related Information

- {doc}`manifest` for paths, primary keys, absent ids, and confirmation vocabulary.
- {doc}`python-backend` for executable tool behavior and reset.
- {doc}`task-templates` for fixture slot sources and tool references.
- {doc}`validation-cases` for direct behavior probes.
- {doc}`held-out-policy` for reservations.
