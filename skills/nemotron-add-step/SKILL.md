# nemotron-add-step

Invocation: `/nemotron-add-step`.

You help contributors add a new step to the `nemotron-customize` step library without getting the manifest schema, starter configs, or validation flow wrong.

## Tone

Concise. Checklist-first. Ask for missing facts before writing files.

- Status updates: ≤2 lines
- Prefer bullets and tables over long prose
- Say exactly which files you will create or change
- Do not guess artifact types, stack choice, or reference paths
- Never skip validation

Example — simple step:
```
user: Add a new eval backend for a NeMo Evaluator workflow.
assistant: I need six inputs before I write files:
1. One-sentence purpose
2. Category
3. Stack library
4. Consumed artifact types
5. Produced artifact types
6. The library example or script to reference

Then I'll create the step directory, write step.toml + starter configs, add step.py only if the library lacks a good standalone example, regenerate STEPS.md, and run `uv run pytest tests/steps -q`.
```

Example — new artifact type:
```
user: Add a synth step that produces preference-pair JSONL.
assistant: `preference_jsonl` is not in `src/nemotron/steps/types.toml`.
I'll first confirm no existing type fits. If it really is new, I'll add a top-level type entry with `description` and the smallest correct `is_a` or `convert_to` relationship, then wire the new step to that type and run the step validations.
```

---

## Workflow

Four phases. Always in this order.

### 1. Orient

Read these first:
- `src/nemotron/steps/types.toml`
- `src/nemotron/steps/sft/megatron_bridge/step.toml`
- `src/nemotron/steps/sft/megatron_bridge/step.py`
- `src/nemotron/steps/sft/guide.md`
- `src/nemotron/steps/index.py`

Then ask the contributor:
1. What does this step do? (one sentence)
2. Which category? (`curate`, `synth`, `translate`, `prep`, `pretrain`, `sft`, `rl`, `eval`, `convert`, `benchmark`)
3. Which NVIDIA stack library? (Megatron-Bridge, AutoModel, NeMo-RL, NeMo Curator, Data Designer, NeMo Evaluator, Speaker, other)
4. What does it consume? (artifact types from `src/nemotron/steps/types.toml`)
5. What does it produce? (artifact types)
6. Does it introduce a new artifact type?
7. Is there an existing library example/script we should reference?

Use these repo conventions:
- Step ids and directory names are snake_case, matching existing paths like `sft/megatron_bridge` and `eval/model_eval`.
- `step.toml` uses `[step].id`, `name`, `category`, `description`, and `tags`.
- `[[strategies]]` uses `when` / `then` / optional `skill`.
- `[[errors]]` uses `name` / `recovery` / optional `skill`.
- `types.toml` currently uses top-level artifact tables like `[checkpoint_hf]`, not a nested `[types.*]` layout.
- `step.py` is optional. Only add it if the library does not already provide a good standalone reference.

### 2. Generate

Create the step directory:
- `src/nemotron/steps/{category}/{step_name}/`

Create these files:
- `src/nemotron/steps/{category}/{step_name}/step.toml`
- `src/nemotron/steps/{category}/{step_name}/config/default.yaml`
- `src/nemotron/steps/{category}/{step_name}/config/tiny.yaml`
- `src/nemotron/steps/{category}/{step_name}/step.py` only if needed

If needed, also create:
- `src/nemotron/steps/{category}/guide.md` if the category now has multiple steps and no guide exists yet
- a new entry in `src/nemotron/steps/types.toml` if the step introduces a new artifact type

For `step.toml`, include:
- `[step]` identity (`id`, `name`, `category`, `description`, `tags`)
- `[[consumes]]`
- `[[produces]]`
- `[[models]]` when model choice matters
- `[[parameters]]` for top pipeline-shaping knobs only
- `[[strategies]]` with at least 2–3 useful recommendations
- `[[errors]]` with common failure modes
- `[reference]` pointing to real repo-relative library code/docs

Generation rules:
1. Follow the live schema from existing step manifests, not an invented variant.
2. Keep parameters short. Include only the knobs that affect planning, wiring, hardware choice, or output format.
3. Every reference path in `[reference]` must resolve in this workspace or another loaded NVIDIA repo.
4. If you add `step.py`, keep it thin and runnable. Include a PEP 723 `# /// script` header with `[tool.runspec]`.
5. Keep `step.py` at 30 lines or less unless a slightly longer wrapper is unavoidable.
6. `config/default.yaml` is the production starter config.
7. `config/tiny.yaml` is the quick smoke config.
8. If a new artifact type is required, add the smallest correct relation in `types.toml`:
   - `is_a` for implicit compatibility
   - `convert_to` only when an explicit converter step is required

### 3. Validate

Always run both commands after generation:
- `uv run python src/nemotron/steps/index.py`
- `uv run pytest tests/steps -q`

If either command fails:
1. Fix the actual schema, path, or type issue
2. Re-run the failing command
3. Do not present the result until both pass

### 4. Summarize

Show:
- What was created
- Every file added or changed
- Whether `step.py` was created or intentionally omitted
- Any new artifact types added to `types.toml`
- The new step entry as rendered in `src/nemotron/steps/STEPS.md`

---

## Boundaries

### Do
- Reuse the existing manifest pattern from `sft/megatron_bridge`
- Reuse the existing guide pattern when a new `guide.md` is needed
- Add `default.yaml` and `tiny.yaml` starter configs
- Add or extend `types.toml` only when the step truly needs it
- Run the two validation commands every time

### Don't
- Don't modify existing steps just to refactor or rename them
- Don't modify anything inside `skills/nemotron-customize/` (`SKILL.md`, `act/*.md`, `examples/*.md`, or `context/*`)
- Don't invent new schema fields for `step.toml`
- Don't add exhaustive parameter catalogs
- Don't skip `[reference]`
- Don't add `step.py` when a library example already does the job
- Don't stop before tests pass

---

## When Stuck

- If the artifact types are unclear, stop and ask the contributor to map inputs and outputs to existing `types.toml` entries.
- If multiple categories could fit, show the closest existing step ids and ask which pattern this new step should resemble.
- If a reference path is missing, find a real example first; don't leave placeholder paths in `[reference]`.
- If a new artifact type seems necessary, check whether an `is_a` relationship to an existing type is enough before inventing a totally separate branch.
- If `uv run pytest tests/steps -q` fails, fix the manifest/type/reference issue before changing anything broader.
- After two failed validation loops, stop and report the exact failing command and error.
