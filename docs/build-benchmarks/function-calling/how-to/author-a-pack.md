<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Hand-Author an Oracle Pack

Use this guide to write an Oracle Pack yourself, validate it, and smoke-run it before you spend a publication budget on it. You supply the tools, the executable oracle, the fixtures, the conversation templates, and the assertions; no model participates in authoring.

## Before You Start

- Install the BYOB dependencies with `uv sync --extra byob`.
- Decide whether the pack will use a local Python backend or an HTTPS service; Step 1
  scaffolds the selected transport.
- Read {doc}`../explanation/oracle-pack` for what each file means. The normative contract, including every validation rule, lives at `src/nemotron/steps/byob/references/bfcl-oracle-pack.md`.

## Step 1: Choose The Oracle Transport And Scaffold

A pack declares exactly one executable oracle. Choose the transport before scaffolding,
because the command writes a transport-specific file and manifest path.

| Transport | File | Use it when |
| --- | --- | --- |
| `python` | `backend.py` | Domain logic can run as deterministic Python. |
| `endpoint` | `endpoint_config.yaml` | An existing HTTPS service implements BFCL Oracle HTTP v1. |

`scaffold_oracle_pack.py` writes a complete starter whose domain is one `get_record`
tool. The command below selects local Python; replace `python` with `endpoint` when
that is the reviewed transport choice.

```bash
python -m nemotron.steps.byob.scripts.scaffold_oracle_pack \
  --domain warehouse_assets \
  --target /srv/bfcl/packs/warehouse_assets \
  --transport python \
  --language en \
  --version 0.1.0
```

`--domain` is normalized into the pack identifier, so it must reduce to a string matching `[a-z][a-z0-9_]*`. `--target` must not exist: the scaffold writes atomically and never overwrites existing work. Add `--include-held-out` to also emit a `held_out.yaml` example that reserves one fixture row. The command prints the created directory, and beside the pack files it writes a `README.md` and a `validate.yaml` you can use immediately.

:::{warning}
Declaring both transports is refused. An endpoint pack stores only credential
references, never secret values. Replace the scaffold's identity placeholders with
reviewed values from `GET /v1/metadata`, and pin the conformance digest from
`GET /v1/conformance`; a Gold endpoint pack requires both identity and attestation.
See {doc}`../reference/endpoint-config`.
:::

## Step 2: Fill In Each Pack File

Work through the files in this order, because each one constrains the next.
{doc}`../reference/oracle-pack-inputs` is the inventory and links to one standardized
reference for every artifact. In particular, use {doc}`../reference/manifest`,
{doc}`../reference/tools-and-fixtures`, {doc}`../reference/python-backend`,
{doc}`../reference/task-templates`, {doc}`../reference/assertions`, and
{doc}`../reference/validation-cases` while filling the local-Python layout. Endpoint
and reservation fields are in {doc}`../reference/endpoint-config` and
{doc}`../reference/held-out-policy`.

| File | What you write |
| --- | --- |
| `tools.json` | The model-facing function schemas, plus the pack-local `x-mutates` and `x-requires-confirmation` annotations. These annotations stay out of the schema the candidate sees. |
| `backend.py` or `endpoint_config.yaml` | The executable oracle. Structured business rejections must use the `{"error": {"code": ...}}` shape, or validation cannot tell a rejection from a failure. |
| `fixtures.json` | The deterministic reset state and the inventory that slots bind against. |
| `manifest.yaml` | Pack identity and version, languages, the frozen clock, the file map under `paths`, `primary_keys` per fixture collection, `absent_ids`, and the `assistant_turn_templates` used by non-tool milestones (`ask_for_slot`, `ask_confirm`, `decline`, `final_answer`). |
| `task_templates.yaml` | One entry per conversation shape: intent, category, difficulty, `turn_policy`, `call_order`, `required_tools`, `tools_present`, slots with their sources and `visible_in_first_turn`, assistant milestones, and `success_assertions`. |
| `assertions.py` | The functions named by `success_assertions`, plus the `ASSERTIONS` mapping and the `ASSERTION_CAPABILITIES` declarations that state whether each one applies to trace or executable evaluation and which category it checks. |
| `validation_cases.yaml` | At least one success probe and one negative probe per tool. A negative probe is one whose result is a structured error or an awaiting-confirmation response. |
| `held_out.yaml` | Optional existing fixture ids and template ids reserved from ordinary generation, plus backend-state visibility policy. |

:::{important}
Every template must declare at least one `success_assertions` entry. A template that names none has no statement of what success means, so replay could only confirm that its trace ran. Validation refuses Gold for that pack. A declining template can still assert that no tool was called.
:::

## Step 3: Set the Category Budget

`task_generation.tasks_per_category` is the default expansion budget for a whole category and, when deduplication and balancing run, the publication cap over unique bindings. It may not fall below the number of templates in the widest category, because the budget is shared across every template in that category and one of them would lose its only instance.

Validation refuses Gold for such a pack rather than letting generation publish a silently narrower set. Count the templates in your largest category and set the budget to at least that number.

:::{note}
Difficulty, conversation turns, and tool-call depth are independent dimensions. A dependent two-call chain can still contain exactly one user turn, so do not treat call depth as a proxy for a multi-turn conversation.
:::

## Step 4: Validate Without Generating

Run the standalone validator for a fast authoring loop. It normalizes the pack, executes the validation cases, checks reset and replay behavior, and derives the tier, all without producing benchmark rows.

```bash
python -m nemotron.steps.byob.scripts.validate_oracle_pack \
  --config /srv/bfcl/packs/warehouse_assets/validate.yaml \
  --output-dir /tmp/bfcl-warehouse-validation
```

`--config` is required. `--output-dir` is optional and defaults to a temporary directory, which is convenient while iterating but discards the artifacts. The command prints the report to standard output, so it is usable directly in a script:

| Exit code | Meaning |
| --- | --- |
| `0` | The pack validated and is Gold-eligible. |
| `2` | The pack validated and was refused Gold. Read the `checks` in the report. |
| `1` | The validator could not reach a verdict at all, for example an unreadable or malformed config. Standard error carries a JSON envelope with `status`, `error_type`, and `reason`, leaving standard output empty so a caller can parse it unconditionally. |

The distinction between `1` and `2` matters in automation: a crash is worth retrying, whereas a verdict is worth reporting to a person.

The same report is what `stage=prepare` writes inside a pipeline run:

```bash
nemotron steps run byob/bfcl \
  -c /srv/bfcl/packs/warehouse_assets/validate.yaml \
  stage=prepare \
  family=bfcl
```

## Step 5: Read the Validation Report

The report is written to `<output_dir>/<expt_name>/stage_cache/oracle_validation_report.json` and carries `tier`, `gold_eligible`, `pack_fingerprint`, the per-check results, and pack statistics. Seven named checks decide the tier: `template_tool_names` and `template_slot_sources` confirm that templates reference only declared tools and that every slot source resolves to matching fixture rows; `backend_schema_alignment` confirms that `list_tools()` and `tools.json` agree and that every parameter schema is one the pipeline can enforce; `assertions_importable` and `declared_validation_cases` confirm the assertion and probe coverage described above; `confirmation_policy` confirms that an unconfirmed call yields `awaiting_confirmation` and leaves state untouched; and `representative_generation_contract` proves that the first deterministic instance of every template expands, binds an expected trace, passes its schemas, renders without breaking a surface guard, replays twice identically, and passes its assertions. Additional checks cover mutation declaration, determinism, structured error shape, timeout enforcement, and process isolation.

Gold requires every check to pass. Two rules surprise people most often:

- A check whose preconditions failed is recorded as `skipped`, and a skipped check is not a pass, so it keeps the pack below Gold. Backend and schema disagreement in particular causes later probe checks to skip, because probing a backend that disagrees with its own catalog would only report noise.
- `oracle_runtime.worker: thread` can never reach Gold. It runs pack code in the caller's process and cannot always stop a tool that hangs. Keep `worker: process`; thread mode exists for debugging.

`stage=generate` derives the verdict from the individual checks rather than from the summary flag, and never trusts a report written by an earlier run, so editing the report on disk accomplishes nothing.

## Step 6: Smoke-Run the Pack

Once the pack is Gold-eligible, copy `smoke.example.yaml` and repoint it. The smoke profile generates every declared category at a small budget, so a pack defect surfaces in minutes rather than hours.

```bash
mkdir -p /srv/bfcl/runs && \
  cp src/nemotron/steps/byob/bfcl/config/smoke.example.yaml \
    /srv/bfcl/runs/warehouse-smoke.yaml
```

Change these fields in the copy, then run the full slice:

```yaml
expt_name: bfcl_warehouse_smoke
output_dir: /srv/bfcl/runs/warehouse-smoke-output
oracle_pack:
  manifest_path: /srv/bfcl/packs/warehouse_assets/manifest.yaml
oracle_runtime:
  allowed_roots:
    - /srv/bfcl/packs/warehouse_assets
task_generation:
  tasks_per_category: 10
```

```bash
nemotron steps run byob/bfcl \
  -c /srv/bfcl/runs/warehouse-smoke.yaml \
  stage=all \
  family=bfcl
```

:::{note}
Use absolute paths for an external pack. A relative path in a generation config resolves from the checked-in `src/nemotron/steps/byob/` root, not from your shell working directory or from the config file's own directory. Pack code must also sit under an `oracle_runtime.allowed_roots` entry, and `output_dir` must stay outside the pack root so generated artifacts cannot become pack inputs.
:::

The smoke profile pins `lineage.policy: smoke_no_publication`, which makes its output deliberately unpublishable: rows keep the pack's validation tier but carry `gold_eligible: false`. That is the point of a smoke run. Move to {doc}`publish-a-release` when you want a releasable benchmark.

## Verify Success

A successful smoke run leaves these files under `output_dir/expt_name/`:

- `benchmark_raw.parquet` and `benchmark.parquet`.
- `run_manifest.json`, written last as the publication commit marker. If it is absent, treat the Parquet files beside it as unpublished.
- `stage_cache/`, holding one table per generation stage keyed by `task_id`, so joining them shows which stage dropped a task.

## Common Failures

| Symptom | What it means |
| --- | --- |
| A category budget error | `tasks_per_category` is below the template count of the widest category. Raise it so no template loses its instances. |
| Generation refuses a non-Gold pack | Fix the reported check failures and re-run `stage=prepare`, keeping `worker: process`. Editing the report cannot raise a tier. |
| Every task was dropped before export | Read `stage_cache/replay_validated_tasks.parquet` for nondeterministic replay or assertion failures, and `stage_cache/rendered_conversations.parquet` for surface-guard violations. |
| The pack is outside the allowed roots | Move the pack under an `oracle_runtime.allowed_roots` entry, or extend the list explicitly. |

{doc}`../reference/troubleshooting` indexes the full error taxonomy.

## Next Steps

- Copy a publication config, set budgets for your reviewed pack, and archive the completed release with `archive_bfcl_release`.
