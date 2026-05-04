---
name: nemotron-env-toml
description: Create, validate, and adjust Nemotron env.toml profiles for Lepton, Ray, NeMo-RL, NeMo-Gym, AutoModel, ModelOpt, and functional step runs. Use when a user needs an env.toml from scratch, profile inheritance fixes, executor/resource profile advice, or debugging Ray runtime-env and placement issues.
---

# Env TOML

Use `env/env_toml` to generate starter env profile files and to preserve hard-earned profile conventions.

Before editing profiles, read `step.toml` for the contract, then choose `config/lepton.yaml` or `config/slurm.yaml`.

## Workflow

1. From the repository root, check whether the target env file already exists. If it does, inspect and extend the existing profiles instead of regenerating the file.

2. When no backend env file exists, generate one:

```bash
uv run nemotron step run env/env_toml -c lepton
export NEMOTRON_ENV_FILE=env.lepton.toml
```

For Slurm:

```bash
uv run nemotron step run env/env_toml -c slurm
export NEMOTRON_ENV_FILE=env.slurm.toml
```

The loader in `src/nemo_runspec/env.py` searches for repository-root `env.toml` by default. To use a generated backend file, set `NEMOTRON_ENV_FILE` to `env.lepton.toml` or `env.slurm.toml` before running `--run` or `--batch` commands.

3. Set site-specific values in config overrides or edit the generated file:
   - Lepton: `node_group`, `resource_shape`, `nemo_run_dir`, mount `path`/`from`
   - Slurm: `host`, `user`, `account`, `partition`, `remote_job_dir`, mounts
   - Shared: container image, `HF_HOME`, output directories, W&B project/entity

4. Validate by compiling one small case before submitting.

## Nuances

- Prefer one backend base and concrete profiles named for individual steps, such as `lepton_prep_sft_packing`, `lepton_pretrain_megatron_bridge`, or `slurm_optimize_modelopt_quantize`.
- Env profiles are inherited with `extends`; child profiles should override only what the step needs, such as image, node count, startup commands, or output path.
- Keep secrets as `${oc.env:...}` placeholders. Do not write tokens directly into env files.
- Keep `[wandb]` for run metadata and pass `WANDB_API_KEY`/`WANDB_PROJECT` through profile `env_vars` so subprocess-heavy steps such as ModelOpt pruning/quantization inherit logging settings.
- Do not put every NeMo-RL runtime flag in env files. Step YAML `run.env.env_vars` carries runtime-specific flags; the config loader deep-merges those with the selected env profile.
- If a profile defines `env_vars`, it should usually contain only site/output variables such as `RL_OUTPUT_DIR`, `HF_HOME`, `WANDB_PROJECT`, or `OPTIM_OUTPUT_DIR`.
- For Ray jobs, avoid job `runtime_env` workdirs when vLLM or NeMo-RL starts nested Ray actors. Use staged source plus `PYTHONPATH` and keep source-transport cleanup in the runner, not in env.toml profiles.
- For RLHF with GenRM, budget physical Ray nodes for policy/generation, NeMo-Gym GPU servers, and extra placement headroom. For example, a small logical `cluster.num_nodes=2` plus `env.nemo_gym.num_gpu_nodes=1` should use a 4x8-GPU Lepton profile until proven stable.
- Use separate image bases: NeMo for Megatron Bridge, NeMo-RL for DPO/RLVR/RLHF, NeMo-AutoModel for AutoModel, and NeMo 26.02 for ModelOpt.
- Keep functional runner `gpu_count` aligned with the env profile, not only the step config.

## Local Files

- Contract: `src/nemotron/steps/env/env_toml/step.toml`
- Runner: `src/nemotron/steps/env/env_toml/step.py`
- Configs: `src/nemotron/steps/env/env_toml/config/lepton.yaml`, `src/nemotron/steps/env/env_toml/config/slurm.yaml`
- Loader behavior: `src/nemo_runspec/env.py`, `src/nemo_runspec/config/loader.py`

## Guardrails

- Keep env profile files at the repository root. The generator uses `force=false`, so it will not overwrite an existing private env file unless explicitly requested.
- Do not use `force=true` as a convenience. Treat it as a deliberate replacement of an existing environment file.
- Prefer adding a new step profile over changing the shared backend base when only one step needs extra nodes or a different image.
- If `extends` resolution is confusing, compile with `--dry-run` and inspect the rendered `run.env`.
