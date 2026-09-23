<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Validation Case Fields

`validation_cases.yaml` is a list of direct oracle probes. Cases establish observed
success, rejection, confirmation, mutation, and determinism behavior before benchmark
tasks are generated.

Validation cases test the oracle itself. They are not model conversations and do not
replace task assertions.

## Create Validation Cases

The manual pack scaffold writes one successful case plus not-found and invalid-argument
negative cases for its starter tool:

```bash
uv run python -m nemotron.steps.byob.scripts.scaffold_oracle_pack \
  --domain my_domain \
  --target /srv/bfcl/packs/my_domain \
  --transport python \
  --language en \
  --version 0.1.0
```

Add enough reviewed cases to cover every public tool. In assisted authoring, model
drafts are proposals with a planning schema, not final pack YAML. A human supplies the
final pack-format cases through the reviewed supplement; assembly checks their tool
references before whole-pack validation executes them.

## Case Field Contract

| Field | Required | Default | Meaning |
| --- | --- | --- | --- |
| `id` | Yes | None | Unique stable string within the file. |
| `tool` | Yes | None | Public name from `tools.json`. |
| `arguments` | Optional | `{}` | Call arguments; declare `{}` explicitly for readability on a no-argument tool. Negative probes may deliberately violate the public schema. |
| `reset_before` | Optional | `true` | Reset the episode before this case. |
| `expect` | Recommended | `{}` | Expected result classification, error, state behavior, or result fields. |
| `coverage` | Optional | None | `negative` marks a success-shaped business outcome as negative coverage. |
| `notes` | Optional | None | Reviewer context; ignored by runtime. |

`expect` supports:

| Field | Value | Meaning |
| --- | --- | --- |
| `result_class` | `success`, `structured_error`, or `awaiting_confirmation` | Expected runtime classification. |
| `error_code` | Domain-defined string or `null` | Expected `result.error.code`. |
| `state_unchanged` | Boolean | Whether observable state must equal the pre-call snapshot. |
| Any other key | Any JSON value | Compared with the same top-level field in the tool result. |

Error-code values are defined by the domain. `not_found` and `invalid_argument` are
examples, not a closed framework vocabulary.

## Result Classification

The validator classifies a returned object in this order:

1. A dictionary under `error` produces `structured_error`; `error.code` identifies the
   rejection.
2. A result whose configured status field equals the configured pending status
   produces `awaiting_confirmation`.
3. Every other object produces `success`.

The status field and pending value come from `manifest.confirmation`; their defaults
are `status` and `awaiting_confirmation`.

Every public tool needs at least one success case and one negative case for Gold.
Structured errors and awaiting-confirmation results count as negative. Use
`coverage: negative` only when a valid business rejection intentionally has a
success-shaped result.

## Minimal Example

```yaml
- id: success_get_record
  tool: get_record
  arguments:
    record_id: REC-001
  expect:
    result_class: success
    error_code: null
  reset_before: true

- id: missing_get_record
  tool: get_record
  arguments:
    record_id: REC-ABSENT-1
  expect:
    result_class: structured_error
    error_code: not_found
    state_unchanged: true
  reset_before: true
```

The example assumes `REC-001` exists, `REC-ABSENT-1` does not, and the backend returns
the shown domain error code.

## Chained Cases

Set `reset_before: false` only when a case intentionally continues the previous
episode, for example to inspect a confirmed mutation after a pending call. The first
case cannot set it to false.

```yaml
- id: checkout_pending
  tool: checkout_record
  arguments: {record_id: REC-001, confirm: false}
  expect:
    result_class: awaiting_confirmation
    state_unchanged: true
  reset_before: true

- id: checkout_confirmed
  tool: checkout_record
  arguments: {record_id: REC-001, confirm: true}
  expect:
    result_class: success
  reset_before: false
```

Use domain-appropriate tool names and outcomes; this shape only demonstrates episode
chaining.

## Validate Cases

There is no standalone case validator. Run:

```bash
uv run python -m nemotron.steps.byob.scripts.validate_oracle_pack \
  --config /srv/bfcl/packs/my_domain/validate.yaml \
  --output-dir /tmp/bfcl-my-domain-validation
```

The `declared_validation_cases` check resets, snapshots state, calls each tool,
classifies the result, compares expectations, and tracks per-tool coverage. Extra Gold
checks reuse these observations for mutation declarations, determinism, structured
error shape, and confirmation behavior.

A success-classified case must satisfy the public parameter schema. A schema-invalid
argument may be used only as a negative probe whose observed outcome is negative.

## Common Failures

- **`validation_cases_empty`:** add reviewed probes.
- **`unknown_tool`:** correct the name or catalog.
- **`reset_before_false_without_predecessor`:** start a chain with a reset.
- **`non_object_result`:** every tool call must return a JSON object.
- **`result_class_mismatch`:** align the oracle or reviewed expectation.
- **`error_code_mismatch`:** use the backend's reviewed domain code.
- **`result_field_mismatch`:** correct an extra expected result field.
- **`state_changed`:** a case marked `state_unchanged` observed a mutation.
- **`successful_validation_case_schema_mismatch`:** the success arguments violate
  `tools.json`.
- **`incomplete_validation_coverage`:** add missing success or negative coverage.

## Complete Examples

- `src/nemotron/steps/byob/data/tiny_oracle_pack/validation_cases.yaml`
- `src/nemotron/steps/byob/data/banking_vn_oracle_pack/validation_cases.yaml`

The second is a larger localized example, not a default domain.

## Related Information

- {doc}`tools-and-fixtures` for tool schemas and reset records.
- {doc}`python-backend` for result and state behavior.
- {doc}`assertions` for task-level replay predicates.
- {doc}`manifest` for confirmation vocabulary.
