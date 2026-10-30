# Env Steps

Use this category for execution-profile setup under `src/nemotron/steps/env/`.

## Developer Journey

Env profiles are the bridge between a step config and the machine that runs it.
Start here before remote submission, not after a launch fails.

1. Decide the backend: local, Lepton, Slurm, or DGX Cloud.
2. Generate or extend the repository-root env file for that backend.
3. Keep site-level details in env TOML and step-specific runtime flags in the
   step YAML.
4. Compile one tiny run and inspect the rendered `run.env`.
5. Export `NEMOTRON_ENV_FILE` when using a backend-specific env file.

## Data And Artifact Flow

```text
env/env_toml config
  -> env.<backend>.toml
  -> NEMOTRON_ENV_FILE
  -> --run / --batch profile used by any step
```

## Route

- `env/env_toml`: generate and validate starter env profile examples for Lepton, Slurm, or DGX Cloud.

## Guardrails

- Read the specific step `README.md` and `step.toml` before editing env profiles.
- Keep env profile files at the repository root. Default profile discovery uses `env.toml`; generated backend examples use `env.lepton.toml`, `env.slurm.toml`, or `env.dgxcloud.toml` and require `export NEMOTRON_ENV_FILE=<file>`.
- If the target env file exists, inspect and extend it rather than overwriting; only use `force=true` when the user intentionally asks to replace it.
- Keep site logistics in env profiles and step runtime flags in the step YAML unless the flag is truly site-wide.
- Keep data-prep step profiles CPU-only unless the step explicitly needs GPUs. Slurm prep profiles should override GPU bases with CPU partitions and `gpus_per_node = 0`; Lepton prep profiles should use a CPU resource shape; DGX Cloud prep profiles should keep `gpus_per_node = 0` and stage source on `PYTHONPATH` for Ray workers.
- Use the NeMo-RL v0.6.0 image for DPO/RLVR/RLHF profiles on Lepton, Slurm, and DGX Cloud. On Lepton, keep `ray_version` on the latest version supported by the workspace rather than blindly matching the upstream NeMo-RL Ray pin.
- Compile one small run after profile changes and inspect `run.env` before submitting.
