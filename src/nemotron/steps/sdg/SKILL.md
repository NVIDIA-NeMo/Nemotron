---
name: nemotron-sdg
description: Configure Nemotron synthetic data generation with NeMo Data Designer for SFT, tool-call, and RL preference data. Use when synthetic data should be generated declaratively before prep, SFT, DPO, RLVR, or RLHF stages, including sovereign cases where domain or language coverage is data-limited.
---

# Nemotron SDG

Synthetic data generation. Build a Data Designer pipeline declaratively from
seeds, columns, models, and an output projection. SDG is most valuable for
sovereign customizations where target-language, domain, or tool-call data is
data-limited and must be expanded carefully.

## Steps

The catalog ships one step under this category:
[`sdg/data_designer`](data_designer/SKILL.md).

## Configurations

| Need | Config | Output shape |
|---|---|---|
| SFT chat data | `config/default.yaml` | `{messages: [...]}` (`openai_messages` projection) |
| Tool-call SFT data | `config/customer_support_tools.yaml` | `{messages, tools}` (`structured_messages` projection) |
| RL preference pairs (DPO) | `config/rl_pref.yaml` | `{prompt, chosen, rejected}` (`dpo_preference` projection) |
| Smoke / preview | `config/tiny.yaml` or preview mode | small batch via `client.preview()` |

## Decision tree

- Need chat-format SFT data → `default.yaml`.
- Need tool-call SFT data (assistant tool_calls + matching tool responses) → `customer_support_tools.yaml`.
- Need DPO preference pairs → `rl_pref.yaml`.
- Iterating on column specs / seeds → preview mode or `tiny.yaml`. **Don't
  scale to `num_records`-thousands until preview output looks right.**

## Pipeline placement

```
sdg/data_designer (default.yaml)              → prep/sft_packing → sft/megatron_bridge
sdg/data_designer (default.yaml)              →                    sft/automodel
sdg/data_designer (customer_support_tools.yaml) → prep/sft_packing → sft/* (tool-call SFT)
sdg/data_designer (rl_pref.yaml)              → prep/rl_prep      → rl/nemo_rl/dpo
```

## Workflow

1. **Env profile first** — verify the env profile for Lepton/Slurm/Ray runs
   (`env.toml` by default, or `NEMOTRON_ENV_FILE` for backend-specific files).
2. Pick the config by output shape (table above).
3. Iterate in preview / `tiny.yaml` until column specs and seed sampling
   produce the right shape.
4. Set `num_records` only after preview is right.
5. Project the output explicitly to the schema the next stage expects:
   - SFT (Megatron-Bridge): `openai_messages` → `prep/sft_packing`.
   - SFT (AutoModel): `openai_messages` directly (no packing).
   - DPO: `dpo_preference` → `prep/rl_prep`.
6. Validate generated records before training — see
   [../patterns/sdg-pipeline-versioning.md](../patterns/sdg-pipeline-versioning.md).
7. For sovereign deployments, mix synthetic with non-synthetic data and
   keep a non-synthetic eval slice — see
   [../patterns/sft-data-blending.md](../patterns/sft-data-blending.md) and
   [../patterns/byob-benchmark-design.md](../patterns/byob-benchmark-design.md).

## Smoke commands

```bash
nemotron steps run sdg/data_designer -c tiny
nemotron steps run sdg/data_designer -c default --extra-args=--preview
```

## Patterns to cite

- [../patterns/sdg-pipeline-versioning.md](../patterns/sdg-pipeline-versioning.md) — version seeds, prompts, models, projections, outputs together.
- [../patterns/data-quality-before-quantity.md](../patterns/data-quality-before-quantity.md) — small + clean beats large + noisy.
- [../patterns/sft-data-blending.md](../patterns/sft-data-blending.md) — synthetic must be blended with human-written data, not used pure.
- [../patterns/byob-benchmark-design.md](../patterns/byob-benchmark-design.md) — keep training data and held-out eval set genuinely separate; never let synthetic items leak into the BYOB.

## Guardrails

- Keep seed files small, high-quality, licensed for the intended use,
  schema-consistent, and balanced across target capabilities.
- Validate generated records (schema, language, safety, duplication, length,
  contamination) before feeding training.
- Pure-synthetic SFT produces stylistically narrow models. Mix with at
  least 20–30% human-written data when available.
- Drive diversity with structured columns (persona / task / difficulty), not
  with high temperature.
- Version seeds + config + outputs together. SDG is a pipeline, not a prompt.
