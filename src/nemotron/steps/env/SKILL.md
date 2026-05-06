---
name: nemotron-env
description: Work with Nemotron execution environment profiles, especially env.toml generation, Lepton executor settings, RayCluster resources, mounts, container images, and profile inheritance. Use when selecting, creating, or debugging run profiles.
---

# Env Steps

Use this category for execution-profile setup under `src/nemotron/steps/env/`.

## Route

- `env/env_toml`: generate and validate starter env profile examples for Lepton or Slurm.

## Guardrails

- Read the specific step `SKILL.md` and `step.toml` before editing env profiles.
- Keep env profile files at the repository root. Default profile discovery uses `env.toml`; generated backend examples use `env.lepton.toml` or `env.slurm.toml` and require `export NEMOTRON_ENV_FILE=<file>`.
- If the target env file exists, inspect and extend it rather than overwriting; only use `force=true` when the user intentionally asks to replace it.
- Keep site logistics in env profiles and step runtime flags in the step YAML unless the flag is truly site-wide.
- Keep data-prep step profiles CPU-only unless the step explicitly needs GPUs. Slurm prep profiles should override GPU bases with CPU partitions and `gpus_per_node = 0`; Lepton prep profiles should use a CPU resource shape.
- Use the NeMo-RL v0.6.0 image for DPO/RLVR/RLHF profiles on Lepton and Slurm. On Lepton, keep `ray_version` on the latest version supported by the workspace rather than blindly matching the upstream NeMo-RL Ray pin.
- Compile one small run after profile changes and inspect `run.env` before submitting.
