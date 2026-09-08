<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Probe Plan

A probe plan is the human-reviewed list of calls intake may execute against a
conventional source. It is passed to assisted authoring as `--probe-plan`. The pipeline
measures those calls and derives certification tier A0, A1, or A2 from the
observations. A Gold freeze requires A2.

The plan is not oracle truth and is not a pack file. Generation never reads it.
`tools.json` names the public interface; the plan names which situations intake is
allowed to observe.

## Create A Probe Plan

Copy the A2-shaped banking example, then replace its tools, fixture ids, and cases:

```bash
cp src/nemotron/steps/byob/references/bfcl-probe-plan.example.json \
  /srv/sources/library-probe-plan.json
```

Do not pass the example unchanged to a library or warehouse source. An optional
model-drafted plan is produced by `draft_probe_plan`; review that draft the same way
you would review a handwritten plan. See {doc}`../how-to/assisted-authoring`.

A `local_python` source may omit `fixtures` when the reviewed `fixtures.json` already
supplies reset state. A session-based HTTP or MCP plan must carry `fixtures`, because
those sessions are handed their world when they open.

## Required Coverage

Write cases that together cover:

1. **Every published tool**, at least one `expectation: success` case.
2. **Mutations**, a success case with `expected_state_change: true` for each tool
   marked `x-mutates`.
3. **Confirmation**, a success path for each tool marked `x-requires-confirmation`.
4. **Structured errors**, at least one `expectation: structured_error` case with
   `expected_error_code` when the source has error codes. Omitting error cases does
   not by itself block A2; it leaves error-shape evidence unobserved.
5. **Timeout cleanup**, at most one `expectation: timeout` case. Without it,
   `check_probe_plan` reports `no_timeout_case` with `impact: blocks_a2`, and intake
   cannot certify above A1.

A timeout case must name a call that actually cannot finish inside the probe deadline.
Inventing a timeout against a tool that returns promptly fails the probe rather than
proving cleanup. Timeout cases must not set `expected_state_change` or
`expected_error_code`.

Success cases require `expected_state_change` and must not set an error code.
Structured-error cases require `expected_error_code` and cannot claim a mutation.

## Example

This compact library fragment matches the `tiny_oracle_pack` catalog used in
{doc}`../how-to/start-from-domain-data`. It is not a complete A2 plan: the bundled
tiny backend has no honest hanging call, so a Gold freeze still needs a timeout case
against a tool that actually blocks, as the assisted-authoring walkthrough adds with
`rebuild_catalog_index`.

```json
{
  "schema_version": "bfcl-local-probe-plan-v1",
  "clock": "2026-03-02T09:00:00+07:00",
  "seed": 7,
  "fixtures": null,
  "cases": [
    {
      "case_id": "a_status_available",
      "tool": "get_book_status",
      "arguments": {"book_id": "BK-100"},
      "expectation": "success",
      "expected_state_change": false
    },
    {
      "case_id": "b_checkout_committed",
      "tool": "checkout_book",
      "arguments": {"book_id": "BK-100", "patron_id": "P-1", "confirm": true},
      "expectation": "success",
      "expected_state_change": true
    },
    {
      "case_id": "c_unknown_book",
      "tool": "get_book_status",
      "arguments": {"book_id": "BK-ABSENT-1"},
      "expectation": "structured_error",
      "expected_error_code": "not_found"
    }
  ]
}
```

Clock values must be ISO-8601 with an explicit timezone. The banking example uses
`2026-03-02T09:00:00+07:00`; that is the same instant as `2026-03-02T02:00:00Z`.

## Validate

Check the plan against a local Python source without executing probes:

```bash
python -m nemotron.steps.byob.scripts.check_probe_plan \
  --source /srv/sources/library \
  --probe-plan /srv/sources/library-probe-plan.json
```

| Exit code | Meaning |
| --- | --- |
| `0` | The plan is well-formed and does not report an A2 blocker. |
| `2` | The plan is usable for intake but cannot attain A2, usually `no_timeout_case`. |
| `1` | The plan or source could not be checked. Standard error carries a JSON envelope with `refusal_code` and `reason`. |

Intake remains authoritative: only it executes the probes and observes reset,
isolation, confirmation, timeout cleanup, and result behavior.

## Common Failures

| Symptom | What it means |
| --- | --- |
| `no_timeout_case` | Add one `expectation: timeout` case, or accept A1 and do not freeze as Gold. |
| `mutation_declaration_mismatch` | A success case claims `expected_state_change: true` on a read-only tool. Align the case with `x-mutates`. |
| Unknown tool or unsafe `case_id` | Case ids and tool names must match `^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$` and tools already in the catalog. |
| Timeout case with an outcome field | Remove `expected_state_change` and `expected_error_code` from the timeout case. |
| More than one timeout case | Keep a single timeout probe. |
| Clock without a timezone | Use an ISO-8601 value with an explicit offset or `Z`. |

## Related Information

- {doc}`../how-to/start-from-domain-data` for preparing the catalog, source, brief,
  and plan before intake.
- {doc}`../how-to/assisted-authoring` for `draft_probe_plan` and the guided `author`
  command.
- {doc}`domain-brief` for the drafting context that accompanies the plan.
- {doc}`oracle-pack-inputs` for the distinction between authoring inputs and Oracle
  Pack files.
- `src/nemotron/steps/byob/references/bfcl-probe-plan.example.json` for a complete
  A2-shaped banking example.
