# Env TOML

Use `env/env_toml` to generate starter env profile files and to preserve hard-earned profile conventions.

Use this README for profile workflow and site-level guardrails; use `step.toml` for generator parameters, strategies, and failure modes, then choose `config/lepton.yaml`, `config/slurm.yaml`, or `config/dgxcloud.yaml`.

## CLI And Overlay Knobs

Start from the backend config closest to your target executor. Developers
usually change:

- `output_path`: `env.lepton.toml`, `env.slurm.toml`, or `env.dgxcloud.toml`.
- `force`: keep `false` unless replacing an env file deliberately.
- `sections`: generated backend base profiles and step-specific profiles.
- Backend site fields such as account, partition, node group, mounts, PVC, image,
  and workspace paths.
- Environment variables for `HF_HOME`, `WANDB_PROJECT`, output roots, and Ray or
  NeMo-RL runtime behavior.

## Run It

From the repository root, generate the backend env file if it does not exist:

```bash
uv run nemotron steps run env/env_toml -c lepton      # or slurm, dgxcloud
export NEMOTRON_ENV_FILE=env.lepton.toml               # match the chosen backend
```

The loader in `src/nemo_runspec/env.py` searches for `env.toml` by default;
set `NEMOTRON_ENV_FILE` to point at the generated backend file before any
`--run` or `--batch` command.

Then fill site-specific values directly in the generated file (or via overrides):

- Lepton: `node_group`, `resource_shape`, `nemo_run_dir`, mount `path`/`from`.
- Slurm: `host`, `user`, `account`, `partition`, `remote_job_dir`, mounts.
- DGX Cloud: `base_url`, `kube_apiserver_url`, `client_id`, `project_name`,
  PVC name/claim/path, `pvc_nemo_run_dir`.
- Shared: container image, `HF_HOME`, output directories, W&B project/entity.

Validate the result by compiling one small step run with `--dry-run` and
inspecting the rendered `run.env`.

## Nuances

- Prefer one backend base and concrete profiles named for individual steps, such as `lepton_prep_sft_packing`, `lepton_pretrain_megatron_bridge`, `lepton_sdg_data_designer_tiny`, `slurm_optimize_modelopt_quantize`, or `dgxcloud_pretrain_megatron_bridge`.
- Env profiles are inherited with `extends`; child profiles should override only what the step needs, such as image, node count, startup commands, or output path.
- Data-prep profiles should be CPU-only by default. For Slurm prep profiles, override the GPU base with CPU partitions, `gpus_per_node = 0`, `build_include_gpus = false`, and enough `cpus_per_task` for Ray/Xenna. For Lepton prep profiles, use a CPU `resource_shape` and `gpus_per_node = 0`. For DGX Cloud prep profiles, keep `gpus_per_node = 0` and provide `RAY_RUNTIME_ENV_PYTHONPATH` for staged source.
- Keep secrets as `${oc.env:...}` placeholders. Do not write tokens directly into env files.
- Keep `[wandb]` for run metadata and pass `WANDB_API_KEY`/`WANDB_PROJECT` through profile `env_vars` so subprocess-heavy steps such as ModelOpt pruning/quantization inherit logging settings.
- Do not put every NeMo-RL runtime flag in env files. Step YAML `run.env.env_vars` carries runtime-specific flags; the config loader deep-merges those with the selected env profile.
- If a profile defines `env_vars`, it should usually contain only site/output variables such as `RL_OUTPUT_DIR`, `HF_HOME`, `WANDB_PROJECT`, or `OPTIM_OUTPUT_DIR`.
- For Ray jobs, avoid job `runtime_env` workdirs when vLLM or NeMo-RL starts nested Ray actors. Use staged source plus `PYTHONPATH` and keep source-transport cleanup in the runner, not in env.toml profiles.
- For RLHF with GenRM, budget physical Ray nodes for policy/generation, NeMo-Gym GPU servers, and extra placement headroom. For example, a small logical `cluster.num_nodes=2` plus `env.nemo_gym.num_gpu_nodes=1` should use a 4x8-GPU Lepton profile until proven stable.
- Use separate image bases: NeMo for Megatron Bridge, NeMo-RL `nvcr.io/nvidia/nemo-rl:v0.6.0` for DPO/RLVR/RLHF, NeMo-AutoModel for AutoModel, and NeMo 26.02 for ModelOpt.
- For DGX Cloud profiles, keep Run:AI credentials and client secrets as `${oc.env:...}` placeholders, use existing PVC declarations for shared storage, and keep `pvc_nemo_run_dir` on the mounted workspace path.
- Use Curator image profiles for `byob/mcq`, `translate/nemo_curator`, and `curate/nemo_curator`; use the normal NeMo image with `data-designer==0.5.5` for `sdg/data_designer`.
- For Lepton NeMo-RL profiles, keep `ray_version` on the latest workspace-supported Ray version. NeMo-RL v0.6.0 pins Ray 2.54 upstream, but some Lepton workspaces may only accept older Ray versions such as 2.48.0.
- Keep functional runner `gpu_count` aligned with the env profile, not only the step config.

## Repository Layout

- Manifest: `src/nemotron/steps/env/env_toml/step.toml`
- Runner: `src/nemotron/steps/env/env_toml/step.py`
- Configs: `src/nemotron/steps/env/env_toml/config/lepton.yaml`, `src/nemotron/steps/env/env_toml/config/slurm.yaml`, `src/nemotron/steps/env/env_toml/config/dgxcloud.yaml`
- Loader behavior: `src/nemo_runspec/env.py`, `src/nemo_runspec/config/loader.py`

## Guardrails

- Keep env profile files at the repository root. The generator uses `force=false`, so it will not overwrite an existing private env file unless explicitly requested.
- Do not use `force=true` as a convenience. Treat it as a deliberate replacement of an existing environment file.
- Prefer adding a new step profile over changing the shared backend base when only one step needs extra nodes or a different image.
- If `extends` resolution is confusing, compile with `--dry-run` and inspect the rendered `run.env`.
