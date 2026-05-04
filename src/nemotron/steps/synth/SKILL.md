---
name: nemotron-synth
description: Configure Nemotron synthetic data generation with NeMo Data Designer for SFT SDG and RL preference SDG. Use when generating chat, tool-use, prompt, or chosen and rejected preference data for downstream prep, SFT, DPO, RLVR, or RLHF stages.
---

# Nemotron Synth

Use this skill when synthetic data should be generated declaratively before prep, SFT, DPO, or RL stages.

## Route

| Need | Config | Output |
| --- | --- | --- |
| SFT synthetic chat data | `synth/data_designer/config/default.yaml` | chat-style `synthetic_jsonl` |
| SFT tool-use synthetic data | `synth/data_designer/config/customer_support_tools.yaml` | tool-call `synthetic_jsonl` |
| RL preference data for DPO | `synth/data_designer/config/rl_pref.yaml` | prompt/chosen/rejected `synthetic_jsonl` |

## Workflow

1. For Lepton, Slurm, Ray, or batch execution, verify the env profile file first. Default lookup uses repository-root `env.toml`; generated backend files such as `env.lepton.toml` or `env.slurm.toml` require `NEMOTRON_ENV_FILE`.
2. Use `synth/data_designer` for both SFT SDG and RL preference SDG.
3. Start with preview mode or `config/tiny.yaml` while editing columns.
4. Project SFT output into OpenAI-style `messages` before `prep/sft_packing` or AutoModel SFT.
5. Project preference output into prompt, chosen, and rejected fields before DPO.
6. Check `src/nemotron/steps/patterns/version-sdg-pipeline.md` before scaling generated datasets.
7. Check `src/nemotron/steps/patterns/data-quality-before-quantity.md` before increasing synthetic volume.

## Guardrails

- Keep seed files small, high quality, licensed for the intended use, and schema-consistent.
- Validate generated records before feeding downstream training.
- Version prompts, model aliases, inference parameters, and projection rules.
