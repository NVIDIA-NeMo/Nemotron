---
name: nemotron-sdg-data-designer
description: Configure Nemotron sdg/data_designer for NeMo Data Designer synthetic data generation. Use for SFT SDG chat or tool-call data, RL preference SDG for DPO, seed datasets, column specs, preview runs, output projections, and generated JSONL validation.
---

# Data Designer SDG

Use `sdg/data_designer` to generate synthetic JSONL from declarative YAML column specifications.

Before changing configs or code, read `step.toml` to understand the step flow, consumed and produced artifacts, important parameters, strategies, failure modes, and upstream references.

## Modes

- SFT SDG: use `config/default.yaml` or `config/customer_support_tools.yaml`.
- RL preference SDG: use `config/rl_pref.yaml` for chosen and rejected preference pairs.
- Tiny validation: use `config/tiny.yaml` or preview mode while editing columns.
- Custom endpoint example: see the commented `providers:` block in
  `config/customer_support_tools.yaml`.

## Configure

- Set `num_records` to the target generated count only after preview output looks correct.
- Set `seed_dataset.path` for seed-typed columns.
- For custom inference endpoints, add `providers:` and point each
  `models[].provider` at a declared provider name.
- In `providers[].api_key`, write the environment variable name such as
  `OPENAI_API_KEY`; do not resolve the secret into YAML with `${oc.env:...}`.
- Add post-processing or projection columns so downstream steps receive the expected schema.
- Use SFT output with AutoModel directly only after it is projected to chat `messages`.
- Use preference output with `rl/nemo_rl/dpo` only after prompt, chosen, and rejected fields are present.
- Check `src/nemotron/steps/patterns/sdg-pipeline-versioning.md` before changing SDG design or scaling generation.

## Local Files

- Contract: `src/nemotron/steps/sdg/data_designer/step.toml`
- Runner: `src/nemotron/steps/sdg/data_designer/step.py`
- Configs: `config/default.yaml`, `config/customer_support_tools.yaml`, `config/rl_pref.yaml`, `config/tiny.yaml`
- Seeds: `data/sft_topic_seeds.jsonl`, `data/customer_support_tool_seeds.jsonl`, `data/rl_pref_prompt_seeds.jsonl`

## Guardrails

- Keep generated schema explicit in YAML; avoid hidden assumptions in `step.py`.
- Inspect a sample of generated records before running prep or training.
- Version prompts, seed data, model aliases, inference parameters, and projections.
