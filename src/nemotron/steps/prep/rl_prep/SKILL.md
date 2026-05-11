---
name: nemotron-prep-rl
description: Configure the Nemotron prep/rl_prep step that resolves RL data blends and shards prompt, preference, or reward-model JSONL for rl/nemo_rl DPO, RLVR, and RLHF steps. Use before NeMo-RL training when data references need materialization, canonical split layout, or schema checks.
---

# RL Prep

Use `prep/rl_prep` before NeMo-RL when prompt or preference data needs HF resolution, local materialization, or split sharding.

Before changing configs or code, read `step.toml` to understand the step flow, consumed and produced artifacts, important parameters, strategies, failure modes, and upstream references.

## Inputs And Outputs

- Consume `training_jsonl` through an RL data blend.
- Produce sharded `training_jsonl` ready for `rl/nemo_rl/dpo`, `rl/nemo_rl/rlvr`, or `rl/nemo_rl/rlhf`.
- Smoke with `nemotron steps run prep/rl_prep -c tiny`.

## Configure

- Keep `resolve_hf_placeholders=true` for closed-network or production clusters.
- Set `num_shards_per_split` to match dataset size and filesystem throughput.
- For DPO, ensure records include prompt, chosen, and rejected responses.
- For RLVR, ensure each prompt carries verifier fields such as ground-truth answers.
- For RLHF, ensure prompt data and reward-model references are handled separately.
- Check `src/nemotron/steps/patterns/prep-data-is-tokenizer-locked.md` before changing RL data layout.
- Check `src/nemotron/steps/patterns/rl-validate-rewards-before-scale.md` before scaling RL jobs from prepared data.

## Local Files

- Contract: `src/nemotron/steps/prep/rl_prep/step.toml`
- Runner: `src/nemotron/steps/prep/rl_prep/step.py`
- Configs: `src/nemotron/steps/prep/rl_prep/config/default.yaml`, `src/nemotron/steps/prep/rl_prep/config/tiny.yaml`
- Sample blend: `src/nemotron/steps/prep/rl_prep/data/blend_tiny.json`

## Guardrails

- Validate output JSONL records from every split before launching RL.
- Preserve split names expected by the RL config.
- Keep DPO preference ordering and RLVR verifier fields explicit.
