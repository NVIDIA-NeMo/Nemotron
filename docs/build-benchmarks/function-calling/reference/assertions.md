<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Assertion Contract

`assertions.py` contains deterministic predicates that decide whether an executed task
succeeded. Assertions evaluate replay evidence; they do not generate assistant text or
repair a failed trajectory.

Every Gold template names at least one assertion in `success_assertions`.

## Create Assertions

The manual pack scaffold writes working result, state, and no-call assertions:

```bash
python -m nemotron.steps.byob.scripts.scaffold_oracle_pack \
  --domain my_domain \
  --target /srv/bfcl/packs/my_domain \
  --transport python \
  --language en \
  --version 0.1.0
```

Extend the generated module with domain predicates derived from independently reviewed
requirements.

In assisted authoring, a model may propose bounded declarative assertion
specifications. The compiler accepts only supported trace predicates such as
tool-called, tool-not-called, and tool-called-after; it does not accept arbitrary
model-written Python. Assertions requiring custom state or result semantics remain
human-authored.

The specification's `assertion_id` and exported Python callable are related but not
identical: the compiler prefixes the callable with `assert_`. For example,
`tool_called_lookup` becomes `assert_tool_called_lookup`. A reviewed supplement's
`success_assertions` must name the compiled callable shown in `assertions.py`; using
only the specification id produces `supplement_assertion_unknown`.

Review that the draft produced at least one direct tool-called assertion for every
published tool a task template may exercise independently. Order assertions are useful
for genuine dependencies, but they do not replace direct capability coverage and must
never compare one tool with itself.

## Function Contract

Use this explicit keyword-only signature:

```python
def assert_something(*, state, trace, task, ctx) -> None:
    ...
```

The worker enforces that the callable can receive `state`, `trace`, `task`, and `ctx`
as keywords; positional-or-keyword parameters with those names and `**kwargs` are also
call-compatible. The explicit signature above is the normative authoring form because
it catches accidental parameters during review.

| Argument | Content |
| --- | --- |
| `state` | Complete observable oracle state after replay. |
| `trace` | Ordered `{"tool", "arguments", "result"}` entries for executed calls. |
| `task` | Bound task instance, including slots and template metadata. |
| `ctx` | Run-owned frozen clock, seed, timeout, task id, and turn information. |

An assertion has three valid outcomes:

- return `None` to pass;
- raise `AssertionError` to fail;
- return `{"status": "not_applicable", "detail": "..."}` when the declared predicate
  cannot apply.

`not_applicable` is not a pass. Any other return value or exception is an
infrastructure error.

## Export Contract

Prefer an explicit mapping:

```python
ASSERTIONS = {
    "assert_record_reported": assert_record_reported,
}
```

If `ASSERTIONS` is absent, module-level callable names beginning with `assert_` are
discovered. When the mapping exists, only callable entries in that mapping are
exported.

`ASSERTION_CAPABILITIES` is optional but recommended:

```python
ASSERTION_CAPABILITIES = {
    "assert_record_reported": {
        "trace": True,
        "executable": True,
        "category": "result",
    }
}
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `trace` | Boolean | `false` | Predicate can be represented in trace-oriented scoring. |
| `executable` | Boolean | `true` | Predicate may run against executable replay. |
| `category` | Enum | `unclassified` | `state`, `path`, `result`, `final_answer`, or `unclassified`. |

The mapping must be a literal assignment. Unknown fields, unknown assertion names, or
computed capability mappings fail validation.

## Minimal Examples

Result assertion:

```python
def assert_record_reported(*, state, trace, task, ctx) -> None:
    del state, ctx
    expected_id = task["slots"]["record_id"]
    assert trace, "expected one tool call"
    result = trace[-1]["result"]
    assert result.get("record_id") == expected_id
```

No-call assertion:

```python
def assert_no_tool_called(*, state, trace, task, ctx) -> None:
    del state, task, ctx
    assert trace == [], "expected no tool call"
```

Export both and declare capabilities that match what each predicate actually reads.

## Validate Assertions

There is no standalone assertion validator. Run:

```bash
python -m nemotron.steps.byob.scripts.validate_oracle_pack \
  --config /srv/bfcl/packs/my_domain/validate.yaml \
  --output-dir /tmp/bfcl-my-domain-validation
```

The `assertions_importable` check imports the module in an isolated worker, verifies
signatures and capabilities, and resolves every template reference. Representative
generation then executes each assertion after deterministic replay. Generated tasks
are replayed twice to detect divergent results, state, or assertion outcomes.

## Common Failures

- **`assertions_import_failed`:** fix syntax, import closure, or module initialization.
- **`template_without_success_assertion`:** name at least one meaningful predicate.
- **`missing_assertion`:** export the name referenced by the template.
- **`invalid_signature`:** make the callable accept all four required names as keyword
  arguments; prefer the explicit keyword-only form above.
- **`invalid_assertion_capability`:** use a literal mapping with supported fields and
  categories.
- **`assertion_not_executable_compatible`:** an executable task references an
  assertion declared with `executable: false`.
- **`assertion_failed`:** inspect the replay trace and final state; do not weaken the
  assertion merely to admit a row.

## Model-Assisted Bias Boundary

Model-proposed assertions can overfit the author's planned trajectories or omit domain
invariants. Treat them as coverage suggestions, not independent evidence of success.
Review predicates against requirements and tests that were not produced by the same
model. Compilation and Gold validation prove supported, deterministic execution; they
do not prove that an assertion captures every important domain condition.

## Complete Examples

- `src/nemotron/steps/byob/data/tiny_oracle_pack/assertions.py`
- `src/nemotron/steps/byob/scripts/scaffold_oracle_pack.py`

## Related Information

- {doc}`task-templates` for `success_assertions`.
- {doc}`python-backend` for the state and trace an assertion observes.
- {doc}`validation-cases` for direct oracle probes, which serve a different purpose.
- {doc}`troubleshooting` for replay and assertion failures.
