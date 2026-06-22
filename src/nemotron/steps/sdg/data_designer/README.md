# Data Designer SDG

Use `sdg/data_designer` to generate synthetic JSONL from declarative YAML column specifications.

Use this README for workflow and pitfalls; use `step.toml` for the exact artifact, parameter, strategy, and error manifest before editing configs or code.

## Modes

- SFT SDG: use `config/default.yaml` or `config/customer_support_tools.yaml`.
- RL preference SDG: use `config/rl_pref.yaml` for chosen and rejected preference pairs.
- Tiny validation: use `config/tiny.yaml` or preview mode while editing columns.
- Custom endpoint example: see the commented `providers:` block in
  `config/customer_support_tools.yaml`.

## CLI And Overlay Knobs

Start from `config/tiny.yaml` for preview and the mode-specific configs for real
generation. In a project overlay, developers usually change:

- `num_records`: keep small until preview output is correct.
- `seed_dataset.path`: seed JSONL for the target domain or capability.
- `providers` and `models.provider`: route generation to the intended endpoint.
- `columns`: prompt, transform, seed, and model-generated column specs.
- `output_projection.type`: `openai_messages`, `structured_messages`, or
  `dpo_preference`.
- Output path and generation parameters such as temperature or model alias.

Example shape:

```bash
uv run nemotron steps run sdg/data_designer \
  -c <project>/config/data_designer.yaml \
  num_records=<count> \
  seed_dataset.path=<project>/data/seeds.jsonl
```

Related patterns:

- Check `src/nemotron/steps/patterns/sdg-pipeline-versioning.md` before changing SDG design or scaling generation.

## Run It

Smoke first to validate wiring, imports, data access, and output paths:

```bash
uv run nemotron steps run sdg/data_designer -c tiny --dry-run
```

Then run the real job from a project overlay:

```bash
uv run nemotron steps run sdg/data_designer \
  -c <project>/config/sdg_data_designer.yaml
```

## Repository Layout

- Manifest: `src/nemotron/steps/sdg/data_designer/step.toml`
- Runner: `src/nemotron/steps/sdg/data_designer/step.py`
- Configs: `config/default.yaml`, `config/customer_support_tools.yaml`, `config/rl_pref.yaml`, `config/tiny.yaml`
- Seeds: `data/sft_topic_seeds.jsonl`, `data/customer_support_tool_seeds.jsonl`, `data/rl_pref_prompt_seeds.jsonl`

## Guardrails

- Keep generated schema explicit in YAML; avoid hidden assumptions in `step.py`.
- Inspect a sample of generated records before running prep or training.
- Do not put resolved API keys in YAML; provider `api_key` values are env-var
  names.
- Version prompts, seed data, model aliases, inference parameters, and projections.
