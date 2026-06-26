# Deployment configs for `--serve`

Each YAML in this directory describes how to bring up a vLLM-served model for one or more long-document SDG stages. They're consumed by the `nemotron data sdg long-document <stage> --serve` flow — see [the recipe README](../README.md) for the user-facing docs.

The schema mirrors [Evaluator-launcher's `vllm.yaml`](../../../../../../../Evaluator/packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/deployment/vllm.yaml) so the two orgs can converge on a shared deployment orchestrator later. Validation is enforced by `DeploymentConfig` in `cli/commands/data/sdg/long_document/_deployment.py`.

## Available configs

| File | Stages that default to it |
|---|---|
| `nemotron-parse-v1.1.yaml` | `ocr` |
| `gpt-oss-120b.yaml` | `text-qa` |
| `qwen3-vl-30b.yaml` | `page-classification` |
| `qwen3-vl-235b.yaml` | `visual-qa`, `single-page-qa`, `windowed-qa`, `whole-document-qa` |

Stage → default mapping is registered in `_deployment.py:STAGE_DEFAULT_DEPLOYMENT`. Override per-invocation with `--serve-config <name>`.

## Schema

All fields validated by `DeploymentConfig` (extra fields rejected).

### Identity

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | str | ✅ | Stable identifier — used in run-dir paths and slurm job names. Should match the YAML's stem. |
| `hf_model_handle` | str | ✅ | HuggingFace model id served via vLLM (e.g. `Qwen/Qwen3-VL-235B-A22B-Thinking-FP8`). |
| `served_model_name` | str | ✅ | Name vLLM publishes on `/v1/*` endpoints. The serve task verifies this string appears in `/v1/models` before publishing the sentinel — it's the safety net against false-positive health checks on shared GPU nodes. Usually equal to `hf_model_handle`. |

### Container

| Field | Type | Required | Notes |
|---|---|---|---|
| `image` | str | ✅ | vLLM-capable container image (e.g. `vllm/vllm-openai:v0.14.1`). |

### vLLM tuning

| Field | Type | Default | Notes |
|---|---|---|---|
| `port` | int \| null | `null` | **Default null = pick a free TCP port at runtime** via an ephemeral-socket bind. Pin a specific port only on dedicated nodes where collisions are impossible — DCGM exporter / Prometheus / similar agents commonly bind well-known ports on shared GPU nodes. |
| `tensor_parallel_size` | int | `1` | Maps to `--tensor-parallel-size`. |
| `pipeline_parallel_size` | int | `1` | Maps to `--pipeline-parallel-size`. |
| `gpu_memory_utilization` | float | `0.90` | Maps to `--gpu-memory-utilization` (0 < x ≤ 1). |
| `max_model_len` | int \| null | `null` | Maps to `--max-model-len`; omit for the model's default. |
| `max_num_seqs` | int \| null | `null` | Maps to `--max-num-seqs`. |
| `trust_remote_code` | bool | `false` | Adds `--trust-remote-code` when true. |
| `extra_args` | str | `""` | Verbatim extra flags appended to `vllm serve` (e.g. `"--reasoning-parser deepseek_r1"`). |

### Container-side

| Field | Type | Default | Notes |
|---|---|---|---|
| `env_vars` | dict[str, str] | `{}` | Exported via `export NAME=VALUE` inside the container before `vllm serve`. |
| `pre_cmd` | str | `""` | Optional shell preamble run inside the container before `vllm serve` — handy for `pip install …` of model-specific runtime deps (Nemotron-Parse uses this for `open-clip-torch albumentations timm`). |
| `chat_template_jinja` | str \| null | `null` | Inline Jinja chat template. When set, written to a tmpfile inside the container and passed via `--chat-template <path>`. Required by Nemotron-Parse to inject its special tokens; not used by most models. |

### Slurm resources for the serve task

| Field | Type | Default | Notes |
|---|---|---|---|
| `nodes` | int | `1` | Slurm `--nodes`. Multi-node serving via vLLM's HAProxy mode is not currently exercised. |
| `gpus_per_node` | int | `1` | Slurm `--gpus-per-node`. Should match `tensor_parallel_size × pipeline_parallel_size`. |
| `walltime` | str | `"08:00:00"` | Slurm `--time`. Many clusters cap at 4 hours (e.g. dlw); set this no higher than your partition allows. The default `08:00:00` will fail to submit on those clusters — set to `"03:59:00"` (or your partition's cap minus 1 minute) in those cases. |
| `partition` | str \| null | `null` | Slurm `--partition`. Resolution chain when null: deployment YAML → `env.sdg_serve_partition` → `env.run_partition` → `env.partition`. |

## How the serve bash uses these fields

The `build_serve_bash` function in `_deployment.py` renders a self-contained bash script that:

1. Cleans any stale sentinel file from a previous run with the same id.
2. Exports each `env_vars` entry as a shell `export`.
3. Writes `chat_template_jinja` (if set) to a tmpfile.
4. Runs `pre_cmd` (if set).
5. Picks a free port if `port` is null; uses the pinned port otherwise.
6. Starts `vllm serve $hf_model_handle --port $PORT --tensor-parallel-size … --served-model-name … …` in the background.
7. Polls `http://$(hostname -f):$PORT/health` AND `/v1/models` (grepping for `served_model_name`) until both pass — fail-fast if vLLM dies before becoming healthy, time-out after 30 minutes.
8. Atomically publishes `http://<host>:<port>/v1` to the sentinel file (`tmp + mv`).
9. Watches for `<sentinel>.done` (touched by the client task on exit). When seen, SIGTERMs vLLM and exits cleanly.

Both the `/health=200` AND `/v1/models contains served_model_name` checks are required because some shared GPU nodes have a separate process bound to common ports (e.g. DCGM on 8000) that returns 200 on `/health` — without the model-list check we'd publish a wrong endpoint.

## Adding a new deployment

Most additions are a one-file change in this directory:

1. Copy an existing YAML close to your target (e.g. `qwen3-vl-235b.yaml` for another large VLM).
2. Update `name`, `hf_model_handle`, `served_model_name`, `image`, and tuning knobs.
3. If a new stage should default to this deployment, add an entry to `_deployment.py:STAGE_DEFAULT_DEPLOYMENT`. Otherwise operators reach it via `--serve-config <name>`.

Run `nemotron data sdg long-document <stage> --serve --serve-config <new-name> --dry-run --batch <profile>` to confirm the YAML parses and the partition resolution looks right.
