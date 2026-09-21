---
name: nemotron-customize
description: "Configure and chain repo-native Nemotron steps for curation, tokenizer extension, SDG/BYOB, training, conversion, optimization, and evaluation."
license: Apache-2.0
metadata:
  version: 0.1.1
  author: NVIDIA Nemotron Team <noreply@nvidia.com>
  tags:
    - nemotron
    - customization
    - training
    - pipelines
---

# nemotron-customize

IMPORTANT: Read this file before answering any `nemotron-customize`, Nemotron
customization, governed Curator flow, translation, tokenizer extension,
synthetic-data, BYOB benchmark, SFT, PEFT, RL, conversion, optimization,
checkpoint or existing/hosted-endpoint evaluation, or multi-step pipeline
request. This applies whether the user names one step or asks you to compose
several steps into a pipeline.

Evaluation requests count even when no training is involved: "evaluate",
"benchmark", "smoke test", or "score" an existing/hosted endpoint, an API/model
ID, or a deployed model all route to `eval/model_eval`. Read this skill for
those too.

## Purpose

Turn a model-customization request into a repo-native Nemotron step pipeline.
Plan the DAG, validate artifact wiring, and create only the YAML/config files
needed to run existing steps.

Use this skill only for inspecting, configuring, validating, running, or
submitting existing Nemotron steps or multi-step training/customization
pipelines. For frontend, dashboard, visualization, generic ML advice,
billing/access, or unrelated coding tasks, stop with a short scope note and do
not inspect the step catalog or edit files in that turn.

## Inputs

Required:

- The requested outcome and enough concrete values to run the selected steps:
  input data, model/checkpoint or endpoint, and output location.

Optional when the selected route needs them:

- Language, backend, hardware/GPU count, metrics, task IDs, tokenizer, and
  step/config preferences.
- Auth environment-variable names for hosted services. Accept names only;
  never request, inline, or commit secret values.

Resolve omitted values by consulting, in order: selected state/config files,
explicit invocation arguments, established agent context, then the current user
prompt. An explicit user correction overrides an older source. If a required
value remains unknown, ask for it or return `Blocked`; do not guess it.

## Prerequisites

- A checkout of the Nemotron repo with `src/nemotron/steps/` present; run from
  the repo root with `uv` available.
- For remote execution, an env profile TOML (`NEMOTRON_ENV_FILE` or
  `env*.toml`) with a section matching the selected step.
- Hosted-service credentials must already be exported through the environment
  variable expected by the selected step.
- Persona MCQ also needs its configured endpoint variables and may require
  `NGC_API_KEY` for managed persona assets and `HF_TOKEN` for gated models.

## Limitations

- Does not invent new catalog steps. When no existing step, runner, recipe, CLI,
  or config can satisfy the request, it names the gap (Explorer mode) instead of
  fabricating a step.
- Produces YAML/config for existing steps; new Python/shell is out of scope
  except in Explorer mode after the gap is approved.
- Not for deployment-only/serving, frontend, dashboards, generic ML advice, or
  non-Nemotron tasks.
- Does not guess concrete values (paths, model IDs, GPU counts, profiles); it
  asks or returns `Blocked` when they are missing.

## Core Rule

Use bundled references first. The `references/` folder is the first decision
surface for routing, artifacts, patterns, hardware heuristics, and command
shape. Use `src/nemotron/steps/...` only as a live verification/fallback source
when you need exact current config fields, manifests, runner imports, or details
missing from bundled references.

If sources disagree:

1. Checked live repo files win for exact execution.
2. Bundled references win for initial routing and planning.
3. Upstream docs/context packs are used only for exceptional code generation
   or library API details.

## Before You Begin

- Read this `SKILL.md` workflow and the relevant bundled reference before
  opening repo source files.
- Route from `references/CATALOG.md` and `references/ARTIFACTS.md` before any
  broad repo exploration. Once a route is determined, verify only the selected
  live step/config/env files needed for the answer.
- Do not emit commands with fake paths, placeholder model IDs, guessed task IDs,
  guessed batch profiles, or default auth variable names presented as facts.
  Ask for missing concrete values or return a `Blocked` handoff.
- Use `references/COMMANDS.md` as the authoritative checklist before
  finalizing configs or execution commands.
- For pipeline requests, plan before editing. Do not create or modify files
  until the DAG, artifact edges, required inputs, and validation checks are
  stated and approved.
- For one-shot command requests, prefer a complete parameterized command in one
  response over exploratory prose, but only after required inputs are known.
  If the user already provides the needed values and asks for only a command,
  answer with the command first and keep explanation minimal.
- Output discipline (keeps responses tight): emit one command block per step,
  include only flags the step actually defines, and add no speculative or
  invented flags. Keep narrative to a few lines — the command plus the required
  safety/profile callouts, not a tutorial. Do not restate reference content the
  user did not ask for.
- Do not spawn subagents for one-shot command lookup. Use the bundled command
  reference directly; verify only the selected step if needed.

## Safety

Keep Bash scoped to repo-safe commands such as `uv run nemotron steps ...`,
targeted tests, `git status/diff`, and config validation. Never run environment
dumps (`env`, `printenv`, broad `export`) or commands that expose secret values.
For remote submissions, destructive changes, or expensive launches, confirm
before execution.

When inspecting env/config files, avoid printing whole files that may contain
secrets. Use targeted reads, report only section names and env-var names, and
redact values for fields containing `token`, `key`, `secret`, `password`,
`credential`, or `auth`.

## Reference Map

| Question | Read first | Live fallback / verification |
|---|---|---|
| Which step or category fits? | `references/CATALOG.md` | `uv run nemotron steps list/show`, then selected `step.toml` |
| Do artifacts chain? | `references/ARTIFACTS.md` | `src/nemotron/steps/types.toml` |
| What run shape should I emit? | `references/COMMANDS.md` | checked-in config YAML plus active profile TOML |
| Remote profile generation or selection | `references/COMMANDS.md` | active `NEMOTRON_ENV_FILE`, `env.toml`, or `env.*.toml` |
| What hardware/backend should I recommend? | `references/HARDWARE.md` | selected step `[[models]]` and `[[strategies]]` |
| Which cross-step guardrails apply? | `references/PATTERNS.md` | `src/nemotron/steps/patterns/<id>.md` |
| How do I run the full workflow? | `references/WORKFLOW.md` | selected step configs, `step.py`, and runners |
| Which upstream library API should generated code use? | `references/context/index.toml` -> matching pack | selected `step.py`, `_runners/`, upstream docs |
| New project scaffold, only when existing repo code cannot support the request | `references/act/PROJECT.md` | existing repo project/recipe shape |
| Per-stage code rules, only when existing repo code cannot support the request | `references/act/STAGE.md` | selected `step.py` and shared runner |

Do not start by reading category READMEs or `step.toml` for ordinary decisions.
Select candidates from bundled references, then verify exact live details before
writing configs or final commands.

## Routing

Use `references/CATALOG.md` as the authoritative home for step selection and
route-specific fast paths. Use `ARTIFACTS.md`, `PATTERNS.md`, and `HARDWARE.md`
only to resolve artifact, cross-step, or hardware constraints after the catalog
narrows the route.

The catalog includes four specialized routing families that must not be
collapsed into generic steps:

- Use the six-step governed Curator flow when the request needs stable document
  identity, measured/approved filtering, audit evidence, holdout
  decontamination, or nested fixed-budget subsets.
- Use `sdg/persona_mcq` for persona-grounded MCQ-shaped **training data** and
  `byob/mcq` for held-out MCQ benchmarks. Use `byob/bfcl` for executable
  function-calling benchmarks and their translation/evaluation lifecycle.
- Use `tokenizer_extension/*` for tokenizer construction, fertility, embedding
  initialization, and cross-vocabulary BPB evaluation. Continue pretraining the
  resulting resized HF checkpoint with a pretraining step; tokenizer evaluation
  does not replace downstream model evaluation.
- For `eval/model_eval`, choose `launcher` when NeMo Evaluator Launcher owns the
  deployment/executor flow and `direct` when the endpoint is already hosted and
  the selected Nemotron env profile should own scheduling.

Each step is independent and stitching steps together is your job. Compose any
pipeline by artifact matching from the user's end goal: chain a step only when
the next step consumes an artifact type nothing upstream already produces. Do
not rely on fixed, named step combinations.

## Instructions

Follow the flow that matches the request: a recommendation/plan, a single-step
command, or a multi-step pipeline. In all cases, route from the bundled
references first, gather required inputs, and verify the selected live step
before presenting anything as runnable.

### Recommendation Response

Use this shape for planning answers:

`Decision`, `Why`, `Required inputs`, `Config/command`, `Avoid`, and `Next step`.
Call out the stack to avoid when the user's constraints make it a poor fit.

Whenever the answer includes a command that touches a hosted service or remote
execution, also state, in the answer:

- The auth env-var name and that its value must be exported in the environment,
  never inlined or committed (never print the value).
- For `--batch`/`--run`, the env TOML profile prerequisite; if no profile
  exists, mark the command `Blocked` or give the local `--dry-run` shape.

### Single-Step Command Flow

1. Confirm repo root has `pyproject.toml` and `src/nemotron/steps/`.
2. Read `references/CATALOG.md` and the selected section of
   `references/COMMANDS.md`.
3. Verify the selected live step with `uv run nemotron steps show <step_id>`
   when available, or the selected `step.toml` when the CLI is unavailable.
4. Read the requested checked-in config or user overlay before emitting the
   command.
5. For remote execution, read `NEMOTRON_ENV_FILE` or repo-root `env*.toml` and
   pick an actual section whose profile matches the step.
6. Emit the full command in one reply with the source tier:
   `Verified`, `Repo-grounded`, `Reference-grounded`, or `Blocked`.

Canonical command shapes live in `references/COMMANDS.md`.

### Pipeline Workflow

For pipelines with two or more stages, use **Orient -> Plan -> Act -> Verify**.
Read `references/WORKFLOW.md` for the phase checklist.

- Orient from bundled references and user constraints.
- Plan a DAG with artifact types, configs, patterns, and validation checks.
- Wait for approval before writing configs or code.
- Act with YAML/config-only changes whenever an existing step can satisfy the
  request.
- Verify every generated YAML, artifact edge, command, and README command
  before reporting completion.

### Catalog Mode

Use when the request maps to existing steps. Fast path:

`references/CATALOG.md` -> `references/ARTIFACTS.md` ->
`references/COMMANDS.md` -> verify selected live manifest/config/profile ->
add a new named config under the selected step's `config/` directory.

## Customization Surface

- Always customize through the step catalog under `src/nemotron/steps/`. Never
  divert to alternate recipe CLIs such as `src/nemotron/cli/commands/super3/` or
  `.../nano3/`, even for Super3/Nano3 work. If a request seems to need those,
  map it back to the equivalent catalog step (e.g. `sft/megatron_bridge`).
- Make customizations as NEW config files inside the selected step's
  `src/nemotron/steps/<cat>/<step>/config/` directory, for example
  `src/nemotron/steps/sft/megatron_bridge/config/my_super3.yaml`.
- Never edit the checked-in `default.yaml`, `tiny.yaml`, other shipped configs,
  `step.toml`, `step.py`, or shared runners. Adding a new config file beside
  them is the expected and only customization write.
- Base new configs on the checked-in `default.yaml` schema (read it, copy the
  needed fields), then override only what the request requires.

### Explorer Mode

Use only after confirming no existing step, runner, recipe, CLI, or YAML config
surface can satisfy the request. Full procedure lives in
`references/WORKFLOW.md`.

## Configuration Alignment

Surface these constraints before commands or config writes:

- SFT packing `pack_size`, Megatron-Bridge `seq_length`, packed sequence size,
  tokenizer, and chat template must match.
- The 128K/256K Super3 configs are dry-run topology references, not launch-ready
  stock configs; they require externally prepared alignment-aware packed shards
  and mixed-precision validation.
- Prepared `packed_parquet` and `binidx` are tokenizer-locked; rebuild after
  tokenizer, chat-template, sequence-length, split, or blend changes.
- Tokenizer-extension comparisons must keep corpus and vocabulary budget fixed.
  Use `max_docs`, not `max_tokens`, for BPB comparisons across vocabularies, and
  compare BPB rather than per-token perplexity.
- Curator policy candidates are not approvals. Profile unfiltered input, record
  an explicit approval, then apply the policy; audit completeness only against
  a producer-emitted manifest and attribute loss only when a ledger exists.
- Decontamination removes from training only and detects whole-document
  near-duplicates; do not claim it catches embedded benchmark questions.
- Megatron-Bridge global batch size must be divisible by data-parallel size;
  start distributed validation with micro batch size 1.
- TP/PP/CP/EP choices must fit GPU count, memory, topology, and model divisibility.
- LoRA merge requires the exact base checkpoint/model and tokenizer used during
  adapter training.
- Conversion/eval of Megatron checkpoints should point at a concrete `iter_*`
  checkpoint, not a parent run directory.
- Hosted eval and translation configs store auth env-var names only, not values.

## Operational Nuances

- Smoke configs (`tiny.yaml`, `tiny_chat.yaml`) are wiring tests, not quality
  evidence.
- `${art:...}` references belong in recipe-backed configs; standalone YAML uses
  plain paths.
- Keep pretraining `bin/idx` data and `blend.json` from the same run/release.
- Write customized configs as new files in the step's
  `src/nemotron/steps/<cat>/<step>/config/` directory; never modify the
  checked-in `default.yaml` or other shipped configs.
- For LoRA, preserve the exact base checkpoint and tokenizer/template metadata
  needed by later merge/eval.
- For translation and hosted eval, mention auth environment variable names only,
  never values.
- Direct evaluation runs to completion and writes per-task artifacts; launcher
  mode returns after acceptance, so poll its invocation to a terminal state
  before treating the evaluation as passed.

## Boundaries

Do:

- Always route through the step catalog under `src/nemotron/steps/`; never use
  alternate recipe CLIs (`src/nemotron/cli/commands/super3|nano3/...`).
- Reuse repo CLIs, runners, recipes, steps, and checked-in configs first.
- Customize by adding a new config under the step's `config/` directory; base it
  on `default.yaml` rather than copying it blindly.
- Validate artifact edges and cite patterns that changed the plan.
- Ask about hardware/data/backend/output path when missing.
- Surface tradeoffs such as AutoModel vs Megatron-Bridge and full SFT vs LoRA.

Do not:

- Invent steps when a catalog step fits.
- Skip Plan for pipelines with two or more stages.
- Generate Python or shell when YAML is enough.
- Add monitoring/W&B unless asked.
- Assume GPU count, env profile, endpoint type, task ID, or auth value.
- Generate Slurm/Airflow/Kubeflow wrappers unless the request explicitly needs
  deployment scaffolding.
- Edit checked-in step files (`default.yaml`/`tiny.yaml`, other shipped configs,
  `step.toml`, `step.py`, runners); only add a new config beside them.
- Restate all per-step rules in `SKILL.md`; use bundled references and source
  fallback.

## Examples

**Single-step routing (LoRA on a small box).** User: "LoRA fine-tune a HF model
on 2 GPUs." Route per `CATALOG.md` -> `peft/automodel` (HF base + small GPU
count); do not offer Megatron-Bridge. Collect base model, JSONL data path,
output dir, LoRA rank/alpha, then emit one `uv run nemotron steps run
peft/automodel -c <config> --dry-run ...` command.

**Multi-step pipeline (Super3 SFT).** User: "data prep + SFT for Super3." This is
two stages, so plan first: SFT on Super3 -> Megatron-Bridge, which consumes
`packed_parquet`, so `data_prep/sft_packing` is required upstream. Present the
DAG (`sft_packing -> sft/megatron_bridge`), align `pack_size`/`seq_length`/
tokenizer, wait for approval, then add new configs under
`src/nemotron/steps/<step>/config/<name>.yaml`. Super3 needs a remote profile;
state the env TOML prerequisite or mark `Blocked`.

**Hosted-endpoint evaluation (no training).** User: "benchmark my hosted model
endpoint." Route to `eval/model_eval`. Use `-c tiny_chat` for a one-sample
launcher smoke test, or `-c direct` when the endpoint is already hosted and the
Nemotron profile must own the backend. Collect endpoint URL, model id, task IDs,
matching tokenizer, durable output directory, and the auth env-var name (value
exported, never inlined). See `references/COMMANDS.md` Evaluation Examples.

## Troubleshooting

| Situation | Action |
|---|---|
| Artifact types do not chain | Recheck `references/ARTIFACTS.md`; insert a converter or change the DAG before writing configs. |
| Remote profile or `--batch` is unclear | Read active env TOML; do not guess profile names. |
| Config key is unclear | Verify selected checked-in config, `step.py`, and shared runner before editing. |
| Strategy points to a missing context pack | Skip the pack, use catalog/pattern text, and flag the plan with `WARNING: <topic> docs unavailable`. |
| Hardware looks too small | Use `references/HARDWARE.md`; suggest smaller model, AutoModel, then LoRA before full Megatron-Bridge. |
| Two Act attempts fail | Stop, explain what was tried and failed, and ask how to proceed. |
| No existing repo path matches | Check `references/context/index.toml` and selected source fallback; use Explorer mode only after naming the gap. |
