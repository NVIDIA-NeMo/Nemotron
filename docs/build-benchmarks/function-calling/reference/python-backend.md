<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Python Backend Contract

A local Oracle Pack uses `backend.py` as the executable source of domain truth. This
page describes the four functions the pipeline calls and the invariants validation
enforces. The normative source of truth is
`src/nemotron/steps/byob/references/bfcl-oracle-pack.md`.

Use an HTTPS `endpoint_config.yaml` instead when the oracle already runs as a service.
A pack declares exactly one of the two transports.

## Required Interface

```python
def list_tools() -> list[str]: ...
def reset(*, ctx, fixtures=None) -> None: ...
def call_tool(name: str, arguments: dict, *, ctx) -> dict: ...
def get_state() -> dict: ...
```

The pipeline imports this module in a separate process worker. Gold validation requires
process isolation so a timed-out call can be terminated and the environment can be
sanitized.

Library names in the short snippets come from the bundled English reference pack.
They illustrate dispatch and state handling only; the four-function interface is
domain- and language-independent.

## Create A Backend Skeleton

Prefer an existing domain-owned implementation, or a backend written from independently
reviewed specifications and records. That keeps benchmark truth anchored in the domain
rather than in an authoring model's learned conventions.

For a manual Oracle Pack, scaffold the complete pack rather than creating
`backend.py` in isolation:

```bash
python -m nemotron.steps.byob.scripts.scaffold_oracle_pack \
  --domain my_domain \
  --target /srv/bfcl/packs/my_domain \
  --transport python \
  --language en \
  --version 0.1.0
```

The generated backend implements all four functions with one runnable `get_record`
tool. Replace the domain names and behavior while preserving the interface.

For model-assisted authoring, first place a reviewed `tools.json` inside a new source
directory, then create a source-package skeleton:

```bash
python -m nemotron.steps.byob.scripts.scaffold_source_package \
  --tools /srv/sources/my-domain/tools.json \
  --output /srv/sources/my-domain \
  --collection records \
  --dependency-lock
```

This command writes `backend.py`, `fixtures.json`, and optionally
`dependency-lock.json`; it does not copy `tools.json`. The mechanical backend carries
`BFCL-TODO` at behavior the catalog cannot decide. Intake refuses the source until
those markers are resolved and reviewed.

An opt-in `--draft-with-model` mode can propose fixture data and behavior in a bounded
declarative schema. The pipeline compiles that declaration into Python; it does not
accept arbitrary Python written by the model. The mode requires a human-authored domain
brief and pinned model identity arguments, never overwrites an existing backend or
fixtures file, and still requires static checks and executable certification probes.
Its command shape is:

```text
python -m nemotron.steps.byob.scripts.scaffold_source_package \
  --tools /srv/sources/my-domain/tools.json \
  --output /srv/sources/my-domain \
  --draft-with-model \
  --domain-brief /srv/sources/my-domain-brief.txt \
  --model-alias <ROUTE_ALIAS> \
  --model-provider <PROVIDER> \
  --model <MODEL_NAME> \
  --model-canonical-id <IMMUTABLE_MODEL_ID> \
  --dependency-lock
```

The model chooses only values representable by the fixed declarative vocabulary.
Review the compiled source, remove no marker without implementing its decision, then
run the same static and executable checks below. See
{doc}`data-designer-provider` for provider setup and
{doc}`../how-to/assisted-authoring` for certification.

:::{caution}
Model-assisted backend drafting is optional and is not the preferred source of oracle
semantics. Use it only when no suitable independently implemented source exists, and
review every proposed behavior against specifications, domain records, and tests that
were not produced by the same model.

An authoring model can import its own priors into fixture values, wording, error shapes,
and state transitions. That can skew task coverage and may favor conventions familiar
to the authoring model or its model family. Static checks, executable replay, and Gold
validation prove determinism and cross-file consistency; they do not prove domain
fidelity, representativeness, or absence of benchmark-construction bias.
:::

## Validate A Backend

Before source intake, run the static source-package check:

```bash
python -m nemotron.steps.byob.scripts.check_source_package \
  --source /srv/sources/my-domain
```

It catches missing required functions, unresolved `BFCL-TODO` markers, catalog names
the backend does not expose, and invalid fixture shape. It cannot prove behavior.

For a complete Oracle Pack, run executable Gold validation:

```bash
python -m nemotron.steps.byob.scripts.validate_oracle_pack \
  --config /srv/bfcl/packs/my_domain/validate.yaml \
  --output-dir /tmp/bfcl-my-domain-validation
```

This imports the backend in a process worker, compares `list_tools()` with
`tools.json`, executes validation cases, checks reset isolation and confirmation
behavior, builds representative template instances, and replays them twice. Exit code
`0` means Gold-eligible, `2` means a completed non-Gold verdict, and `1` means the
validator could not reach a verdict.

## `list_tools()`

Returns the public function names the backend implements:

```python
def list_tools() -> list[str]:
    return ["get_book_status", "checkout_book"]
```

The set must match the names in `tools.json`. The catalog owns descriptions and
parameter schemas; `list_tools()` is only the executable inventory. Returning a name
not present in the catalog, or omitting one the catalog declares, fails backend/schema
alignment.

Private Python helpers do not need to match public names. A backend may dispatch
`"get_book_status"` to `_get_book_status()`.

## `reset(*, ctx, fixtures=None)`

Creates a fresh deterministic episode:

```python
def reset(*, ctx, fixtures=None) -> None:
    global _SNAPSHOT, _STATE
    if fixtures is not None:
        _SNAPSHOT = copy.deepcopy(fixtures)
    if _SNAPSHOT is None:
        raise RuntimeError("reset requires fixtures on the first call")
    _STATE = copy.deepcopy(_SNAPSHOT)
```

The pipeline owns reset and calls it before replay. A backend should:

- initialize its fixture snapshot when fixtures are supplied;
- restore a deep copy for each episode;
- reset counters or other deterministic local state;
- use `ctx.clock` and `ctx.seed` rather than wall-clock time or ambient randomness;
- never read secrets or configuration from the process environment.

An infrastructure failure such as missing initial fixtures may raise. Expected domain
rejections belong in `call_tool()` results instead.

## `call_tool(name, arguments, *, ctx)`

Dispatches one public tool and returns a JSON-serializable object:

```python
def call_tool(name: str, arguments: dict, *, ctx) -> dict:
    if name == "get_book_status":
        return _get_book_status(arguments)
    if name == "checkout_book":
        return _checkout_book(arguments)
    return {"error": {"code": "invalid_argument", "field": "name"}}
```

The backend must validate the argument values it depends on even though the generation
pipeline separately validates calls against `tools.json`. A validation case may
deliberately send a wrong type to prove the structured error behavior.

Return expected business failures as data:

```json
{
  "error": {
    "code": "not_found",
    "entity": "books",
    "id": "BK-9",
    "field": "book_id",
    "message": "book BK-9 not found"
  }
}
```

Only `error.code` is required. `entity`, `id`, `field`, and `message` are optional
reviewer detail. Do not raise for an expected rejection: an exception is an
infrastructure failure and cannot be scored as domain behavior.

For a tool marked `x-requires-confirmation`:

- the tool schema exposes the pack's confirmation parameter;
- a call with that parameter false returns the configured pending status;
- state remains unchanged until a confirming call;
- a confirming call performs the mutation exactly once.

## `get_state()`

Returns the complete observable state assertions use after replay:

```python
def get_state() -> dict:
    return copy.deepcopy(_STATE)
```

Return a defensive copy. If a caller receives the live state, it could mutate the
oracle without a tool call and make replay evidence meaningless.

Read-only packs still implement `get_state()`. It may return unchanged fixture state or
another deterministic JSON object.

## `RunContext`

The keyword-only `ctx` argument carries run-owned values:

| Field | Use |
| --- | --- |
| `clock` | Frozen domain time. |
| `seed` | Deterministic randomness source. |
| `timeout_s` | Current call deadline. |
| `task_id` | Identity of the task being replayed. |
| `turn_index` | Assistant/tool turn being executed. |

The pipeline, not the backend, owns these values. Reading wall-clock time, process
environment, or unseeded randomness makes two replays diverge and prevents Gold.

## What Validation Checks

`stage=prepare` verifies:

- all four public functions exist with callable-compatible signatures;
- `list_tools()` and `tools.json` agree;
- returned tool values and `get_state()` are JSON-serializable;
- reset produces isolated, deterministic episodes;
- tool behavior matches declared validation cases;
- mutations match `x-mutates`;
- unconfirmed calls do not mutate when confirmation is required;
- timeouts can be enforced by the process worker.

Later, `executable_replay` resets and runs each generated task twice, compares results,
state, and assertion outcomes, then refuses any divergence.

## Common Failures

- **Missing required callable:** preserve all four public functions even for read-only
  domains.
- **Backend/schema mismatch:** make `list_tools()` match `tools.json` exactly.
- **Non-object or non-serializable result:** return a JSON object from every tool call.
- **Expected rejection raised as an exception:** return
  `{"error": {"code": "..."}}`; reserve exceptions for infrastructure failures.
- **Reset divergence:** deep-copy fixtures and use `ctx.clock` and `ctx.seed`.
- **Confirmation mutation:** leave state unchanged until the configured confirmation
  argument is true.
- **Timeout or environment finding:** remove ambient I/O and secrets; keep calls
  bounded by the supplied context.
- **Unresolved `BFCL-TODO`:** implement and independently review every generated
  behavior before intake.

## Complete Example

Read
`src/nemotron/steps/byob/data/tiny_oracle_pack/backend.py` for a complete English
backend with:

- read-only `get_book_status`;
- confirmation-protected `checkout_book`;
- structured invalid-argument and not-found errors;
- deterministic loan identifiers;
- defensive reset and state copies.

To generate a smaller starter:

```bash
python -m nemotron.steps.byob.scripts.scaffold_oracle_pack \
  --domain my_domain \
  --target /srv/bfcl/packs/my_domain \
  --transport python \
  --language en \
  --version 0.1.0
```

Replace the starter's `get_record` domain while preserving the four-function
interface.

## Related Information

- {doc}`oracle-pack-inputs` for the complete pack file map and cross-file lineage.
- {doc}`manifest` for paths and confirmation vocabulary.
- {doc}`tools-and-fixtures` for the public schema and reset records.
- {doc}`task-templates` for how a conversation names backend tools.
- {doc}`assertions` for predicates that inspect final state and trace.
- {doc}`validation-cases` for direct executable probes.
- {doc}`../how-to/author-a-pack` for validation and smoke-run commands.
- {doc}`troubleshooting` for backend/schema, confirmation, and replay failures.
