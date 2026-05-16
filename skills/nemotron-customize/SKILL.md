---
name: nemotron-customize
description: Use when building runnable Nemotron model-customization pipelines from existing repo steps and artifact contracts.
version: 0.1.0
metadata:
  author: NVIDIA NeMo Team <nemo-toolkit@nvidia.com>
  tags:
    - nemotron
    - customization
    - training
    - pipelines
---

# nemotron-customize

## Purpose

Use this skill to turn a model-customization request into a repo-native Nemotron step pipeline. It plans the step DAG, validates artifact wiring, and creates only the YAML configs needed to run existing steps.

Use it only for inspecting, configuring, validating, running, or submitting
existing Nemotron steps or multi-step training/customization pipelines. If the
request is a frontend, dashboard, visualization, generic ML-advice,
billing/access, or unrelated coding task, stop with a short scope note and do
not inspect the step catalog or edit files in that turn.

## Requirements

- A checkout of this Nemotron repo with `src/nemotron/steps/` available.
- User-provided model, data, hardware, backend, and output constraints before writing configs.
- Backend credentials only when the selected step requires them, such as translation or W&B-enabled training.

## Limitations

- This skill does not invent new catalog steps when an existing step can satisfy the request.
- New Python or shell code is allowed only in Explorer mode after the repo capability gap is explicit.
- Post-training deployment-only requests are out of scope unless they are part of a model-customization pipeline.

Invocation: `/nemotron-customize`.

You compose **steps** from [src/nemotron/steps/](../../src/nemotron/steps/)
into repo-native runnable configs. **The current codebase is the source of
truth.** This skill orchestrates — it does not duplicate per-step knowledge.

Priority order:

1. Use the current repo's available code, CLIs, recipes, steps, runners, and
   config conventions.
2. Create only new YAML config files needed to serve the user's request.
3. Generate new Python or shell code only when the current codebase cannot
   support the request, and explain the gap before doing so.

When you need to know what a step does, read its `step.toml` and `SKILL.md`.
When you need to know whether a chain is sound, read the patterns it cites.
When you need to configure a stage, read `step.py` + the runner + existing
configs to learn the supported YAML shape. Read context packs only if new code
is unavoidable.

For a command request, the fast path is: verify repo root, run or read the step
catalog, read the selected `step.toml`, verify the requested config exists,
read the active env TOML for any remote profile, then emit the complete command.
Do not guess `--batch` profiles from examples or naming conventions.

## How information is split (and where to find it)

| Question | Look here |
|---|---|
| What does step X consume / produce / parameterize? | `src/nemotron/steps/<cat>/<X>/step.toml` |
| When/why pick step X over its siblings? | `src/nemotron/steps/<cat>/<X>/SKILL.md` |
| Which step in category C should I pick? | `src/nemotron/steps/<cat>/SKILL.md` |
| What runner code does step X use? | `src/nemotron/steps/<cat>/<X>/step.py` → [_runners/](../../src/nemotron/steps/_runners/) |
| Cross-step constraint (tokenizer lock, sequence packing, data quality, ...) | `src/nemotron/steps/patterns/<id>.md` |
| Artifact compatibility / `is_a` hierarchy | [src/nemotron/steps/types.toml](../../src/nemotron/steps/types.toml) |
| GPU memory / parallelism heuristics | [src/nemotron/steps/hardware.md](../../src/nemotron/steps/hardware.md) |
| Library API extracts for exceptional code generation | [references/context/index.toml](references/context/index.toml) → `references/context/<pack>.txt` |
| Project scaffold rules, only when repo code cannot support the request | [references/act/PROJECT.md](references/act/PROJECT.md) |
| Per-stage code rules, only when repo code cannot support the request | [references/act/STAGE.md](references/act/STAGE.md) |

If two sources say the same thing, the **deeper, more specific** one wins
(`step.toml` > category `SKILL.md` > this file).

---

## Instructions

Use this skill when the user asks for an end-to-end Nemotron-stack pipeline:
fine-tuning, continued pretraining, alignment training, data curation,
translation for training data, or other data preprocessing for model training.
Follow the workflow below in order:

1. **Orient**: discover candidate steps, read the catalog and compatibility
   sources, and ask for missing hardware/data/backend constraints.
2. **Plan**: propose a stage DAG, validate artifact wiring, cite matched
   patterns, and wait for user approval before changing files.
3. **Act**: create the minimal YAML configs for the selected repo steps.
   Generate code only if no current repo path can satisfy the request.
4. **Verify**: check generated configs, artifact edges, and command
   consistency; fix issues before reporting completion.

Do not treat this skill as general ML advice. The step library under
[src/nemotron/steps/](../../src/nemotron/steps/) is the source of truth.

For single-step command questions, use this shorter flow instead of the full
pipeline workflow:

1. Confirm the repo root has `pyproject.toml` and `src/nemotron/steps/`.
2. Run `uv run nemotron steps list --json` when available; otherwise read
   [STEPS.md](../../src/nemotron/steps/STEPS.md).
3. Read the selected step's `step.toml` and the requested checked-in config.
4. For remote execution, read `NEMOTRON_ENV_FILE` or a repo-root `env*.toml`
   and choose an actual section name whose profile matches the step.
5. Return the command first, followed by only the rationale needed to explain
   config/profile choices.

For translation-only command requests, also read
[src/nemotron/steps/translate/SKILL.md](../../src/nemotron/steps/translate/SKILL.md)
and return `Decision`, `Config`, `Run`, `Output`, and `Env`. Do not continue
broad repository exploration once those fields are execution-ready.

Source tiers for command answers:

- **Verified**: CLI, manifest, config, env profile, and dry-run all succeeded.
- **Repo-grounded**: manifest, config, and env profile were read, but dry-run
  could not be executed.
- **Blocked**: a required repo file or env TOML is missing; name it and stop
  before emitting a guessed remote command.

Canonical commands:

```bash
uv run nemotron steps run <step_id> -c <config-or-path> --dry-run
uv run nemotron steps run <step_id> -c <config-or-path> --dry-run --batch <profile>
uv run nemotron steps run <step_id> -c <config-or-path> --batch <profile>
```

---

## Workflow

Four phases, in order: **Orient → Plan → Act → Verify.** Never skip Verify.

---

### Phase 1 — Orient

Goal: enumerate candidate steps and gather the user's constraints in one pass.

**Step 1.1 — Discover via the CLI, not by grep.** The catalog is
machine-readable:

```bash
nemotron steps list --json                                 # all steps
nemotron steps list --json --category sft                  # by category
nemotron steps list --json --consumes training_jsonl       # by input type
nemotron steps list --json --produces checkpoint_megatron  # by output type
nemotron steps show <step_id>                              # full manifest
```

**Step 1.2 — Read these in parallel** (small files, all cheap):

- [src/nemotron/steps/STEPS.md](../../src/nemotron/steps/STEPS.md) — auto-generated catalog (always read first).
- [src/nemotron/steps/PATTERNS.md](../../src/nemotron/steps/PATTERNS.md) — auto-generated pattern index.
- [src/nemotron/steps/types.toml](../../src/nemotron/steps/types.toml) — artifact compatibility graph (`is_a` hierarchy).
- [src/nemotron/steps/hardware.md](../../src/nemotron/steps/hardware.md) — GPU heuristics if hardware is in scope.

**Step 1.3 — For each candidate category, descend one level**:

- `src/nemotron/steps/<cat>/SKILL.md` — when a category has multiple options
  ([sft/](../../src/nemotron/steps/sft/SKILL.md),
  [pretrain/](../../src/nemotron/steps/pretrain/SKILL.md),
  [peft/](../../src/nemotron/steps/peft/SKILL.md),
  [rl/nemo_rl/](../../src/nemotron/steps/rl/nemo_rl/SKILL.md)).

**Step 1.4 — For each candidate step, read its `step.toml`** end-to-end.
You're after: `[[consumes]]`, `[[produces]]`, `[[parameters]]`,
`[[strategies]]`, `[[errors]]`, `[reference]`. Don't read `step.py` yet —
that's Act.

**Step 1.5 — Match patterns.** Skim `src/nemotron/steps/patterns/*.md`
frontmatter (`triggers:` field). Note matching pattern IDs for the plan.

**Step 1.6 — Ask the user any of the following that aren't already known.**
Present as a numbered list, replies as numbers or Enter for `[defaults]`:

1. Model: `[Nano3]` / Super3 / other (HF id)
2. Data: have it / acquire / synthesize / translate
3. Data size (rough): \_\_\_ examples
4. GPUs: count + type + nodes (e.g. `8x H100, 1 node`)
5. Backend preference: `[nemo-run]` / plain Python
6. W&B: `[off]` / on (project name?)
7. Output: `[./<project-name>/]` / current dir

**Never assume hardware, data availability, or framework. Ask.**

---

### Phase 2 — Plan

Goal: produce a markdown plan the user reviews before any code is written.

**Step 2.1 — Draft the stage DAG.** One stage per step. Number stages
`NN_<name>`. Use a Mermaid graph for the artifact flow.

**Step 2.2 — For each stage, list:**
- Step id (e.g. `sft/megatron_bridge`).
- `consumes` from `<stage NN | user>`.
- `produces`.
- 2–3 key parameters being set.
- Strategies fired (the `when:` clauses from `step.toml` that match).
- Patterns cited (from `src/nemotron/steps/patterns/`).

**Step 2.3 — Run preflight validation.** Hard checks: artifact types chain via [types.toml](../../src/nemotron/steps/types.toml); tokenizer/template/sequence length align across prep and train; RL warm-starts from SFT; GPU count satisfies the selected model; applicable patterns are cited. When a check fails: surface it as a `WARNING:` warning in the plan and propose a
fix. When the user can't satisfy it (e.g. hardware), propose alternatives in
descending preference: smaller model → AutoModel instead of Megatron-Bridge →
LoRA instead of full FT.

**Step 2.4 — Plan format.** Include `Intent`, `Stages`, `Validation`, and `Infrastructure`. Use a Mermaid graph for artifact flow, one short stage block per step, and explicit `PASS:` / `WARNING:` validation lines.

**Step 2.5 — Present the plan and wait.** Don't proceed to Act until the
user approves or requests changes. If new code appears necessary, name the
missing repo capability and get approval for that code path.

---

### Phase 3 — Act

Goal: produce the smallest runnable change, preferably YAML config only. No
placeholders. No TODOs.

**Step 3.1 — Prefer the existing repo execution path.**

Before creating any code, identify how the existing repo can run each stage:

- CLI commands under [src/nemotron/cli/](../../src/nemotron/cli/).
- Step entrypoints in `src/nemotron/steps/<cat>/<step>/step.py`.
- Shared runners in [src/nemotron/steps/_runners/](../../src/nemotron/steps/_runners/).
- Existing configs under the selected step, recipe, or runner directory.

**Step 3.2 — Generate only YAML configs when the repo supports the request.**

```
<project-name>/
├── configs/
│   └── <stage-name>.yaml        # user-specific config for an existing step
└── README.md                    # optional: only if the user asks for run docs
```

Naming: `<project-name>` is kebab-case. YAML filenames should match approved
stage names.

Each YAML config must:

- Match keys read by the existing `step.py` and runner code.
- Adapt existing default/tiny configs instead of inventing a schema.
- Use user-provided paths, model IDs, hardware, backend, and W&B settings.
- Preserve artifact compatibility from the approved plan.

**Step 3.3 — Only use codegen when YAML cannot satisfy the request.**

If the repo lacks a callable step, runner, CLI, or config surface for the
requested behavior, load codegen rules:

- Main agent reads [references/act/PROJECT.md](references/act/PROJECT.md) (project scaffold rules).
- Each per-stage sub-agent reads [references/act/STAGE.md](references/act/STAGE.md) (R1–R5 +
  code-quality + dry-run + W&B).

Then implement the missing stage with the narrowest possible code change:

```
You are implementing stage <NN>_<name> = <step_id>.

Load:
  - skills/nemotron-customize/references/act/STAGE.md
  - <context_pack_path>                       # from context/index.toml; OPTIONAL — skip if not mapped
  - src/nemotron/steps/<cat>/<step>/step.py   # primary code shape
  - src/nemotron/steps/_runners/<runner>.py   # if step.py imports a shared runner

Plan inputs:
  - Model: <model>
  - Hardware: <gpus>
  - Key params: <from approved plan>

Output path: <project_name>/stages/<NN>_<name>/

Deliverables (exactly these):
  - run.py
  - __init__.py
  - config/default.yaml
  - config/tiny.yaml, or the step's checked-in smoke config name such as config/tiny_chat.yaml for eval/model_eval

Report back: files written, knobs exposed, UPSTREAM notes, strategies followed.
```

If sub-agents aren't available, do stages sequentially: load one context pack,
write that stage, drop pack, move on.

**Step 3.4 — Step.py + the runner are the reference.** Don't invent YAML keys
or library APIs from memory. Mirror what the in-repo code does:

- [steps/_runners/megatron_bridge.py](../../src/nemotron/steps/_runners/megatron_bridge.py) — used by sft/peft/pretrain Megatron-Bridge steps.
- [steps/_runners/automodel.py](../../src/nemotron/steps/_runners/automodel.py) — used by AutoModel steps.
- [steps/_runners/nemo_rl.py](../../src/nemotron/steps/_runners/nemo_rl.py) — used by all NeMo-RL alignment steps.

When a step has no context pack, the agent combines: per-step `SKILL.md` + `step.toml [[strategies]]` + `step.py` + the URLs in `[reference]`. That is enough.

---

### Phase 4 — Verify

Goal: every preflight check holds against the generated YAML configs and any
exceptional code, not just the plan.

Run through:

- [ ] Every generated `*.yaml` is valid; keys match the existing step/runner code.
- [ ] Artifact wiring is consistent (stage N output type = stage N+1 input type).
- [ ] Existing CLI or runner commands can consume the generated configs.
- [ ] If exceptional code was generated, every stage script has valid Python syntax.
- [ ] If exceptional code was generated, every import references a real module from the step's reference code.
- [ ] If a README was generated, its commands match the actual configs.
- [ ] Smoke-test YAML configs use reduced iters, batch sizes, max_steps.
- [ ] Tokenizer + seq_length aligned across prep ↔ train YAMLs.
- [ ] No `${art:...}` references leaked into generated configs unless the existing recipe path explicitly requires them.

If verification finds issues, fix them silently. Don't say "I noticed an issue."

---

## Operational Nuances

- Smoke configs such as `tiny.yaml` or eval/model_eval's `tiny_chat.yaml` are for wiring tests, not model-quality evidence.
- If a `step.toml` strategy points to unavailable upstream docs, use its `then:` text and mark the plan for manual review.
- Preserve `${art:...}` only in recipe-backed configs; standalone YAML should use plain paths.
- Keep pretraining `bin/idx` data and `blend.json` from the same Nemotron release.

## Examples

- Single step: read the manifest/config/env profile, then return a complete
  `uv run nemotron steps run <step_id> -c <config> --dry-run` command.
- Pipeline: plan the step DAG first, validate artifact edges, then create only
  the project YAML overlays needed for the approved stages.

---

## Two modes

### Catalog mode — a step exists

Fast path. Levels 0 → 2 in Orient, then Plan → Act.

`STEPS.md → category/SKILL.md → step.toml → step.py → adapt YAML config`

Use whenever the user's request maps to a step in the catalog.

### Explorer mode — no repo path supports it

1. Confirm no existing step, runner, recipe, CLI, or YAML config surface can
   satisfy the request.
2. Look at libraries cited in nearby `step.toml [reference]` URLs.
3. Read the relevant library docs / examples.
4. Use [types.toml](../../src/nemotron/steps/types.toml) to type the new
   stage's consumes/produces.
5. Write the narrowest missing stage from scratch, mirroring an existing
   `step.py` as a template.

Tell the user: "This use case doesn't have a pre-built step. I'll build it
from `<library>` docs — the output will need more validation than a
catalog-based stage."

If the same Explorer build keeps appearing across projects, suggest the user
run `/nemotron-add-step` to land it in the catalog.

### Choosing a mode

| User says | Mode |
|---|---|
| "SFT with Megatron-Bridge / AutoModel" | Catalog |
| "DPO / RLVR / GRPO / RLHF" | Catalog ([rl/nemo_rl/*](../../src/nemotron/steps/rl/nemo_rl/)) |
| "Synthesize preference / SFT data" | Catalog ([sdg/data_designer](../../src/nemotron/steps/sdg/data_designer/)) |
| "Translate EN → \<lang\> for training data" | Catalog ([translate/translation](../../src/nemotron/steps/translate/translation/)) |
| "Curate web text" | Catalog ([curate/nemo_curator](../../src/nemotron/steps/curate/nemo_curator/)) |
| "Train with X exotic backend" | Explorer or **ask** |
| Post-training-only request | Out of scope for this skill; ask the user to use a more appropriate workflow. |
| Ambiguous | **Ask** |

---

## Boundaries

### Do

- Build pipelines from steps that exist; cite step.toml fields directly.
- Reuse the current repo's CLIs, recipes, runners, and step implementations first.
- Adapt configs to the user's hardware and dataset (don't blindly copy `default.yaml`).
- Fire strategies and follow `skill:` pointers when perf-tuning.
- Ask about hardware, data, backend, and output path — never assume.
- Generate only the YAML configs needed for the approved request.
- Surface tradeoffs (Megatron-Bridge vs AutoModel, full FT vs LoRA) as tables.
- Present the plan and wait for approval.

### Don't

- Invent steps. Use Explorer mode or ask.
- Skip Plan for any pipeline ≥2 stages.
- Generate new Python, shell scripts, scaffolds, or wrappers when existing repo code can already serve the request with YAML.
- Import from modules not present in the step's reference code.
- Add monitoring / logging / W&B unless the user asks.
- Tune parallelism beyond what `hardware.md` and `[[strategies]]` advise.
- Assume GPU count, type, or interconnect.
- Generate Slurm/Airflow/Kubeflow wrappers.
- Handle requests outside training and training-data preparation in this skill.
- Modify [src/nemotron/steps/](../../src/nemotron/steps/). To extend the catalog, route the user to `/nemotron-add-step`.
- Restate per-step rules in this skill — link to the step's `SKILL.md` instead.

---

## When stuck

| Situation | Action |
|---|---|
| No existing repo path matches the user's request | Check libraries cited in nearby `step.toml [reference]`. If supported, use Explorer mode. Otherwise ask. |
| Artifact types won't chain | Explain the gap and ask the user whether to change the training/data-prep plan. Do not add post-training work here. |
| Strategy points to a missing skill file | Skip the load. Use the `then:` text as guidance. Note in plan: "WARNING: Could not read perf-tuning docs for `<topic>` — config may need manual review." |
| User's hardware is too small | Show the relevant `[[models]]` `min_gpus` table. Suggest in order: smaller model → AutoModel → LoRA. |
| Two failed Act attempts | Stop. Explain what was tried, what failed, ask the user how to proceed. |
| User wants a feature that crosses 3+ projects | Confirm YAML and existing repo code cannot serve it. If not, build it Explorer-mode for them now, then suggest `/nemotron-add-step` to land it in the catalog. |

## Troubleshooting

| Symptom | Action |
|---|---|
| Artifact types do not chain | Recheck `types.toml` and change the DAG before writing configs |
| Remote profile is unclear | Read the active env TOML; do not guess `--batch` |
| Config key is unclear | Read the step config, `step.py`, and shared runner before editing |
