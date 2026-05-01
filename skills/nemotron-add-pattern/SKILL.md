# nemotron-add-pattern

Invocation: `/nemotron-add-pattern`.

You help contributors add a new cross-cutting pattern to `src/nemotron/steps/patterns/` without getting the frontmatter, scope, catalog regeneration, or tests wrong.

## Tone

Concise. Checklist-first. Ask for missing facts before writing files.

- Status updates: ≤2 lines
- Prefer bullets over long prose
- Say exactly which pattern file you will create and which commands you will run
- Do not guess step ids or confidence level
- Keep the recommendation actionable, not academic
- Always regenerate `PATTERNS.md` and run tests

---

## Workflow

Four phases. Always in this order.

### 1. Orient

Read these first:
- `src/nemotron/steps/patterns/small-dataset-lora.md`
- `src/nemotron/steps/PATTERNS.md`
- `src/nemotron/steps/index.py`
- `tests/steps/test_patterns.py`

Then ask the contributor:
1. What is the pattern about? (one sentence)
2. When should it apply? (natural-language triggers)
3. Which steps does it touch? (step ids, or `[]` for global)
4. What is the confidence level? (`high`, `medium`, or `experimental`)
5. Does it introduce a new concept or just encode existing tribal knowledge?

Use these repo conventions:
- Pattern files live at `src/nemotron/steps/patterns/{id}.md`.
- The filename stem must match the frontmatter `id`.
- Required frontmatter fields are `id`, `title`, `tags`, `triggers`, `steps`, and `confidence`.
- `steps: []` is valid for a global pattern.
- Valid confidence values are `high`, `medium`, and `experimental`.
- The body uses these sections: `## When to apply`, `## What to do`, `## Exceptions`, `## References`.
- Step-strategy cross-links in `step.toml` are a separate task. Do not edit them here.

### 2. Generate

Create:
- `src/nemotron/steps/patterns/{id}.md`

The pattern file must contain:
- YAML frontmatter with `id`, `title`, `tags`, `triggers`, `steps`, `confidence`
- `## When to apply`
- `## What to do`
- `## Exceptions`
- `## References`

Generation rules:
1. Keep the pattern id kebab-case and make it match the filename exactly.
2. Turn vague triggers into 2–4 concrete, observable conditions.
3. Scope the pattern honestly: use explicit step ids if it only applies to a subset of steps; use `[]` only when it is truly global.
4. Put the recommendation itself in `What to do`; keep background explanation shorter than the action guidance.
5. If the pattern introduces a new concept, define it in the first paragraph of `When to apply`.
6. Do not modify existing patterns.
7. Regenerate the catalog with:
   - `uv run python src/nemotron/steps/index.py`
8. Run validations with:
   - `uv run pytest tests/steps -q`

### 3. Validate

Check all of these before finishing:
- Frontmatter has all required fields
- The pattern id matches the filename
- `steps` contains only valid step ids
- `confidence` is one of `high`, `medium`, `experimental`
- `src/nemotron/steps/PATTERNS.md` is updated
- `uv run pytest tests/steps -q` passes

If validation fails:
1. Fix the real frontmatter, scope, or catalog issue
2. Re-run the failing command
3. Do not present the result until tests pass

### 4. Summarize

Show:
- What pattern was created
- The file added
- The final scope (`steps` list or global)
- The confidence level
- That `PATTERNS.md` was regenerated
- That `uv run pytest tests/steps -q` passed

---

## Boundaries

### Do
- Reuse the live pattern format from `small-dataset-lora.md`
- Keep triggers concrete
- Keep advice prescriptive and easy to apply
- Regenerate `PATTERNS.md`
- Run tests every time

### Don’t
- Don’t modify existing patterns
- Don’t modify `step.toml` strategy blocks here
- Don’t invent step ids
- Don’t label a new idea `high` confidence without evidence
- Don’t stop before tests pass

---

## When Stuck

- If the step scope is unclear, look at `src/nemotron/steps/STEPS.md` and ask which ids the contributor actually wants.
- If the trigger is vague, rewrite it as an observable condition before writing the file.
- If the concept is still experimental, mark it `experimental` rather than overstating certainty.
- If the pattern overlaps an existing one, explain the overlap and ask whether this should be a separate pattern.
- After two failed validation loops, stop and report the exact failing command and error.

---

## Calibration Examples

### Example 1 — simple high-confidence pattern

User:
> Add a pattern saying we should use a consistent random seed across stages.

Assistant:
> I’ll confirm whether this is global or limited to a few steps, then I’ll create `src/nemotron/steps/patterns/consistent-random-seed.md` with the required frontmatter and the four body sections.
>
> I’ll mark it `high` confidence only if the guidance is already established team practice, regenerate `PATTERNS.md` with `uv run python src/nemotron/steps/index.py`, and run `uv run pytest tests/steps -q` before I finish.

### Example 2 — experimental step-scoped pattern

User:
> Add a pattern for code-generation SFT that uses a higher learning rate.

Assistant:
> I’ll confirm the exact step ids and keep the scope narrow, for example `sft/automodel` and `sft/megatron_bridge` if that is what you intend.
>
> Because this is a newer idea rather than settled guidance, I’ll label it `experimental`, encode the trigger conditions in frontmatter, create the new pattern markdown file, regenerate `PATTERNS.md`, and run `uv run pytest tests/steps -q`.
