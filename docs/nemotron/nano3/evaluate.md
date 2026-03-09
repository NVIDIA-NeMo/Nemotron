# Stage 3: Evaluation

Evaluate trained Nemotron Nano 3 models against standard benchmarks using [NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator).

> **Different execution pattern**: Unlike training stages that submit Python scripts via NeMo-Run, evaluation compiles the YAML config and passes it directly to [nemo-evaluator-launcher](https://github.com/NVIDIA-NeMo/Evaluator). There is no recipe script—the CLI handles config compilation and artifact resolution, then delegates to the launcher.

---

## How Evaluation Works

The eval command resolves model artifacts from W&B lineage and uses NeMo Framework's Ray-based in-framework deployment. It defaults to evaluating the latest RL stage output.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryBorderColor': '#333333', 'lineColor': '#333333', 'primaryTextColor': '#333333'}}}%%
flowchart TB
    subgraph cli["Nemotron CLI"]
        direction LR
        yaml["YAML Config"] --> compile["Config Compilation"]
        compile --> save["Save job.yaml + eval.yaml"]
    end

    subgraph launcher["nemo-evaluator-launcher"]
        direction LR
        deploy["Deploy Model<br/>(NeMo Ray)"] --> run["Run Benchmarks"] --> export["Export Results<br/>(W&B)"]
    end

    save --> launcher

    style cli fill:#e3f2fd,stroke:#2196f3
    style launcher fill:#f3e5f5,stroke:#9c27b0
```

### Config Compilation Pipeline

The CLI performs several transformations on the YAML config before passing it to the launcher:

1. **Load** the YAML config via OmegaConf (with Hydra defaults resolution)
2. **Merge** env.toml profile values and CLI dotlist overrides
3. **Auto-inject** W&B credential mappings if W&B export is configured
4. **Auto-squash** container images for Slurm (converts Docker images to `.sqsh` files)
5. **Strip** the `run` section and resolve all `${run.*}` interpolations
6. **Resolve** artifact references (`${art:model,path}`) via W&B Artifacts
7. **Pass** the cleaned config to `nemo-evaluator-launcher`'s `run_eval()`

Two YAML files are saved for provenance:
- `job.yaml` — full config including `run` section (for reproducibility)
- `eval.yaml` — compiled config as seen by the launcher

### Deployment

The default config uses NeMo Framework's Ray-based in-framework deployment (`type: generic`) with a custom command for serving:

```yaml
deployment:
  type: generic
  multiple_instances: true
  image: nvcr.io/nvidia/nemo:25.11.nemotron_3_nano
  checkpoint_path: ${art:model,path}
  port: 1235
  command: >-
    bash -c 'python deploy_ray_inframework.py
    --megatron_checkpoint /checkpoint/
    --num_gpus 8
    --tensor_model_parallel_size 2
    --expert_model_parallel_size 8
    --port 1235'
```

Parallelism settings are tuned for the Nano3 30B MoE model:

| Setting | Value | Purpose |
|---------|-------|---------|
| `tensor_model_parallel_size` | 2 | Tensor parallelism across GPUs |
| `expert_model_parallel_size` | 8 | Expert parallelism for MoE layers |
| `num_gpus` | 8 | Total GPUs per node |
| `port` | 1235 | Ray serving port |

The model is deployed using the same NeMo Megatron container as training (`nvcr.io/nvidia/nemo:25.11.nemotron_3_nano`); nemo-evaluator-launcher pulls its own containers for evaluation tasks.

### Evaluation Tasks

Tasks are defined in the `evaluation.tasks` list. Each task maps to a benchmark supported by NeMo Evaluator:

```yaml
evaluation:
  tasks:
    - name: adlr_mmlu
      nemo_evaluator_config:       # Optional per-task overrides
        config:
          params:
            top_p: 0.0
    - name: adlr_arc_challenge_llama_25_shot
    - name: adlr_winogrande_5_shot
    - name: hellaswag
    - name: openbookqa
```

The default config includes five standard benchmarks:

| Task | Type | Description |
|------|------|-------------|
| `adlr_mmlu` | Text Generation | Massive Multitask Language Understanding |
| `adlr_arc_challenge_llama_25_shot` | Log Probability | ARC Challenge with 25-shot prompting |
| `adlr_winogrande_5_shot` | Log Probability | Winogrande commonsense reasoning |
| `hellaswag` | Log Probability | Commonsense sentence completion |
| `openbookqa` | Log Probability | Open-domain science questions |

To discover additional tasks: `nemo-evaluator-launcher ls tasks`

---

## Recipe Execution

### Quick Start

<div class="termy">

```console
// Evaluate the latest RL model from the pipeline
$ uv run nemotron nano3 eval --run YOUR-CLUSTER

// Evaluate a specific model artifact
$ uv run nemotron nano3 eval --run YOUR-CLUSTER run.model=sft:v2

// Filter to specific benchmarks
$ uv run nemotron nano3 eval --run YOUR-CLUSTER -t adlr_mmlu -t hellaswag

// Dry run: preview the resolved config without executing
$ uv run nemotron nano3 eval --dry-run
```

</div>

> **Note**: The `--run YOUR-CLUSTER` flag submits jobs via [NeMo-Run](../../nemo_runspec/nemo-run.md). See [Execution through NeMo-Run](../../nemo_runspec/nemo-run.md) for setup.

### Prerequisites

- **[NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator)**: Install with `pip install "nemotron[evaluator]"` or ensure `nemo-evaluator-launcher` is available
- **Container image**: `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano` (NeMo Megatron container for model serving)
- **[Weights & Biases](../wandb.md)**: For result export (optional but recommended)
- **Slurm cluster**: For remote execution

### Configuration

| File | Purpose |
|------|---------|
| `config/default.yaml` | Evaluation config with NeMo Ray deployment and benchmark tasks |

The config has five sections:

```yaml
# Nemotron extension (stripped before passing to launcher)
run:
  model: rl:latest                    # W&B artifact reference
  env:                                 # Populated from env.toml profile
    container: nvcr.io/nvidia/nemo:25.11.nemotron_3_nano
    executor: slurm
    host: ${oc.env:HOSTNAME,localhost}
    ...
  wandb:
    entity: null
    project: null

# Passed directly to nemo-evaluator-launcher
execution:
  type: slurm
  num_nodes: 1
  gres: gpu:8
  auto_export:
    enabled: true
    destinations: [wandb]

deployment:
  type: generic                        # NeMo Framework Ray
  checkpoint_path: ${art:model,path}   # Resolved from W&B artifact
  command: >-
    bash -c 'python deploy_ray_inframework.py
    --megatron_checkpoint /checkpoint/
    --num_gpus 8
    --tensor_model_parallel_size 2
    --expert_model_parallel_size 8
    --port 1235'

evaluation:
  nemo_evaluator_config:
    config:
      params:
        parallelism: 4
        request_timeout: 6000
  tasks:
    - name: adlr_mmlu
    - name: adlr_arc_challenge_llama_25_shot
    - name: adlr_winogrande_5_shot
    - name: hellaswag
    - name: openbookqa

export:
  wandb:
    entity: ${run.wandb.entity}
    project: ${run.wandb.project}
```

| Section | Purpose | Passed to Launcher? |
|---------|---------|:-------------------:|
| `run` | env.toml injection, artifact references | No (stripped) |
| `execution` | Where to run, auto-export, mounts | Yes |
| `deployment` | How to serve the model | Yes |
| `evaluation` | Tasks and evaluation parameters | Yes |
| `export` | Result destinations (W&B) | Yes |

### Artifact Resolution

The default config uses `${art:model,path}` for the model checkpoint:

```yaml
run:
  model: rl:latest  # Resolve latest RL artifact

deployment:
  checkpoint_path: ${art:model,path}  # Resolved at runtime
```

Override the model artifact on the command line:

```bash
# Evaluate the SFT model instead of RL
uv run nemotron nano3 eval --run YOUR-CLUSTER run.model=sft:latest

# Evaluate a specific version
uv run nemotron nano3 eval --run YOUR-CLUSTER run.model=sft:v2

# Use an explicit path (bypasses artifact resolution)
uv run nemotron nano3 eval --run YOUR-CLUSTER deployment.checkpoint_path=/path/to/checkpoint
```

### Task Filtering

Use `-t`/`--task` flags to run a subset of benchmarks:

```bash
# Single task
uv run nemotron nano3 eval --run YOUR-CLUSTER -t adlr_mmlu

# Multiple tasks
uv run nemotron nano3 eval --run YOUR-CLUSTER -t adlr_mmlu -t hellaswag -t openbookqa
```

### Override Examples

```bash
# Increase evaluation parallelism
uv run nemotron nano3 eval evaluation.nemo_evaluator_config.config.params.parallelism=16

# Change walltime
uv run nemotron nano3 eval --run YOUR-CLUSTER run.env.time=08:00:00
```

### Running with NeMo-Run

Configure execution profiles in `env.toml`:

```toml
[wandb]
project = "nemotron"
entity = "YOUR-TEAM"

[YOUR-CLUSTER]
executor = "slurm"
account = "YOUR-ACCOUNT"
partition = "batch"
nodes = 1
ntasks_per_node = 1
gpus_per_node = 8
mounts = ["/lustre:/lustre"]
```

See [Execution through NeMo-Run](../../nemo_runspec/nemo-run.md) for complete configuration options.

### W&B Integration

Results are automatically exported to W&B when configured:

1. **Auto-detection**: The CLI detects your local `wandb login` and propagates `WANDB_API_KEY` to evaluation containers
2. **env.toml config**: `WANDB_PROJECT` and `WANDB_ENTITY` are loaded from `env.toml`
3. **Auto-export**: Results are exported after evaluation completes when `execution.auto_export.destinations` includes `wandb`

See [W&B Integration](../wandb.md) for setup.

### Artifact Lineage

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryBorderColor': '#333333', 'lineColor': '#333333', 'primaryTextColor': '#333333', 'clusterBkg': '#ffffff', 'clusterBorder': '#333333'}}}%%
flowchart TB
    subgraph pipeline["Training Pipeline"]
        pretrain["ModelArtifact-pretrain"] --> sft["ModelArtifact-sft"]
        sft --> rl["ModelArtifact-rl"]
    end

    rl --> eval["nemotron nano3 eval"]
    sft -.-> eval
    eval --> results["Evaluation Results<br/>(W&B)"]

    style pipeline fill:#e1f5fe,stroke:#2196f3
    style eval fill:#f3e5f5,stroke:#9c27b0
    style results fill:#e8f5e9,stroke:#4caf50
```

> [Artifact Lineage & W&B Integration](../artifacts.md)

---

## Infrastructure

This stage uses the following components:

| Component | Role | Documentation |
|-----------|------|---------------|
| [NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator) | Benchmark evaluation framework and launcher | [GitHub](https://github.com/NVIDIA-NeMo/Evaluator) |
| [NeMo Framework](../nvidia-stack.md) | Ray-based in-framework model deployment | [Docs](https://docs.nvidia.com/nemo/) |

### Container

```
nvcr.io/nvidia/nemo:25.11.nemotron_3_nano
```

The NeMo Megatron container is used for model serving. The nemo-evaluator-launcher pulls its own containers for running evaluation tasks.

## CLI Reference

<div class="termy">

```console
$ uv run nemotron nano3 eval --help
Usage: nemotron nano3 eval [OPTIONS]

 Run evaluation with NeMo-Evaluator (stage3).

╭─ Options ────────────────────────────────────────────────────────────────╮
│  -c, --config NAME       Config name or path                             │
│  -r, --run PROFILE       Submit to cluster (attached)                    │
│  -b, --batch PROFILE     Submit to cluster (detached)                    │
│  -d, --dry-run           Preview config without execution                │
│  -t, --task NAME         Filter to specific task (repeatable)            │
╰──────────────────────────────────────────────────────────────────────────╯
╭─ Artifact Overrides ─────────────────────────────────────────────────────╮
│  run.model     Model checkpoint artifact (default: rl:latest)            │
╰──────────────────────────────────────────────────────────────────────────╯
```

</div>

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `nemo-evaluator-launcher` not found | Install with `pip install "nemotron[evaluator]"` |
| W&B authentication fails | Run `wandb login`. See [W&B Integration](../wandb.md) |
| Model deployment fails | Check parallelism settings match GPU config (TP=2, EP=8 for Nano3) |
| Artifact resolution fails | Verify artifact exists in W&B. Use `deployment.checkpoint_path=/explicit/path` to bypass |
| Task not found | List available tasks with `nemo-evaluator-launcher ls tasks` |

---

## Previous Stage

After RL completes in [Stage 2: RL](./rl.md), evaluation is the final step in the pipeline.

## Reference

- [NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator) -- Upstream evaluation framework
- [Artifact Lineage](../artifacts.md) -- W&B artifact system
- [Execution through NeMo-Run](../../nemo_runspec/nemo-run.md) -- Cluster configuration
- [W&B Integration](../wandb.md) -- Credentials and export setup
- **Recipe Source:** `src/nemotron/recipes/nano3/stage3_eval/`
- [Back to Overview](./README.md)
