# AGENTS.md -- Nemotron Repository Agent Context

## What This Repo Does

Nemotron is NVIDIA's open-source repository for reproducible LLM training pipelines. It provides:

1. **Training recipes** for NVIDIA model families (Nano3, Super3, Embed) -- full pretrain/SFT/RL pipelines
2. **Customization recipes** for adapting models to new languages, domains, and use cases (Sovereign AI Playbook)
3. **Data preparation** infrastructure for tokenization, packing, and format conversion
4. **Evaluation** via NeMo Evaluator with benchmark suites

## Repository Layout

```
Nemotron/
  AGENTS.md                          <-- You are here
  pyproject.toml                     <-- Package config; entry point: nemotron CLI
  src/
    nemo_runspec/                    <-- Config loading, execution, PEP 723 metadata parsing
    nemotron/
      cli/
        bin/nemotron.py              <-- CLI root (Typer app)
        commands/
          nano3/                     <-- Nano3 commands: pretrain, sft, rl, eval, pipe
          super3/                    <-- Super3 commands: pretrain, sft, rl (rlhf/rlvr/swe)
          embed/                     <-- Embedding model commands: sdg, prep, finetune, eval, export, deploy
          customize/                 <-- Customization CLI (WIP)
        kit/                         <-- CLI utilities (app, squash)
      kit/                           <-- Domain toolkit: Artifact types, lineage tracking, W&B, recipe loading
      data_prep/                     <-- Distributed data prep library (bin/idx, packed parquet, JSONL)
      recipes/
        nano3/                       <-- Nano3 recipe scripts + configs
          stage0_pretrain/           <-- train.py, data_prep.py, config/
          stage1_sft/
          stage2_rl/
          stage3_eval/
        super3/                      <-- Super3 recipe scripts + configs
          stage0_pretrain/
          stage1_sft/
          stage2_rl/                 <-- Sub-stages: rlvr, swe1, swe2, rlhf
          stage3_eval/
        embed/                       <-- Embedding model recipes
          stage0_sdg/ .. stage5_deploy/
        data_curation/               <-- NeMo Curator recipes (nemotron-cc)
      customization_recipes/         <-- Sovereign AI customization pipelines
        nemotron/                    <-- Nemotron model customization (6 stages)
          SKILL.md                   <-- E2E customization pipeline skill definition
          stage0_cpt/                <-- Continued Pretraining
          stage1_sft/                <-- Supervised Fine-Tuning + SDG
          stage2_rl/                 <-- Reinforcement Learning (DPO/GRPO)
          stage3_byob/               <-- Build Your Own Benchmark
          stage4_eval/               <-- Evaluation
          stage5_quantization/       <-- Quantization for deployment
        llama/                       <-- Llama model customization (same stage structure)
        qwen/                        <-- Qwen model customization (same stage structure)
        data_prep/                   <-- Shared data prep utilities for customization
  tests/
  docs/
  deploy/                            <-- Deployment configs (Docker, Helm)
  tools/
  usage-cookbook/
  use-case-examples/
```

## Key Infrastructure

### nemotron CLI

Entry point: `nemotron` (defined in `pyproject.toml` as `nemotron.__main__:main`).

```bash
# Pattern: nemotron <model> <stage> [options] [overrides]
nemotron nano3 pretrain -c default                     # Local execution
nemotron nano3 pretrain -c default --run MY-CLUSTER    # Remote via nemo-run (attached)
nemotron nano3 pretrain -c default --batch MY-CLUSTER  # Remote via nemo-run (detached)
nemotron nano3 pretrain -c default --dry-run            # Preview compiled config
nemotron nano3 sft -c default --run MY-CLUSTER train.train_iters=5000  # Override params
nemotron nano3 pipe --run MY-CLUSTER                    # Compose pretrain + sft
nemotron nano3 eval --run MY-CLUSTER                    # Run evaluation suite

# Data prep (run directly, not via CLI)
python src/nemotron/recipes/nano3/stage0_pretrain/data_prep.py --config <yaml>
```

Global options: `-c/--config`, `-r/--run`, `-b/--batch`, `-d/--dry-run`, `--stage`, `--force-squash`.

### nemo_runspec

Module: `src/nemo_runspec/`

Parses PEP 723 `[tool.runspec]` metadata from recipe scripts. Provides:
- `nemo_runspec.parse(script_path)` -- returns `Runspec` with name, image, config_dir, resources
- `nemo_runspec.config` -- OmegaConf YAML loading, job config building, artifact URI resolution
- `nemo_runspec.execution` -- local (torchrun) and remote (Slurm/Lepton/Run:AI/Ray via nemo-run) execution
- `nemo_runspec.packaging` -- SelfContainedPackager for remote code shipping

Config resolution chain: script `[tool.runspec]` -> `config/<name>.yaml` -> `env.toml` profile -> CLI overrides.

### nemotron.kit

Module: `src/nemotron/kit/`

Domain-specific toolkit:
- `nemotron.kit.Artifact` -- base class for typed artifacts (pydantic)
- `nemotron.kit.ModelArtifact`, `PretrainDataArtifact`, `SFTDataArtifact` -- typed artifact classes
- `nemotron.kit.init(backend="fsspec"|"wandb", root=...)` -- initialize artifact registry
- `nemotron.kit.recipe_loader` -- `import_recipe_function(target)`, `extract_recipe_config(config)`
- `nemotron.kit.train_script` -- `parse_config_and_overrides()`, `load_omegaconf_yaml()`, `apply_hydra_overrides()`
- `nemotron.kit.wandb_kit` -- W&B initialization, monkey patches, lineage tracking

### nemotron.data_prep

Module: `src/nemotron/data_prep/`

Distributed data prep built on cosmos-xenna pipelines:
- `nemotron.data_prep.api` -- `run_pretrain_pipeline()`, `run_sft_pipeline()`
- Three-phase pattern: `setup_*_run()` -> xenna pipeline stages -> `finalize_*_run()`
- Output formats: bin/idx (pretrain), packed Parquet (SFT), JSONL (RL)
- Stages: PlanStage -> DownloadStage -> terminal stage (BinIdxTokenization / PackedSftParquet / JsonlShard)

## Task Routing

| Task | Go to |
|------|-------|
| Train Nano3 from scratch | `src/nemotron/recipes/nano3/` |
| Train Super3 from scratch | `src/nemotron/recipes/super3/` |
| Train embedding model | `src/nemotron/recipes/embed/` |
| Curate web data (CommonCrawl) | `src/nemotron/recipes/data_curation/nemotron-cc/` |
| Customize Nemotron for a language/domain | `src/nemotron/customization_recipes/nemotron/SKILL.md` |
| Customize Llama for a language/domain | `src/nemotron/customization_recipes/llama/SKILL.md` |
| Customize Qwen for a language/domain | `src/nemotron/customization_recipes/qwen/SKILL.md` |
| Prepare training data (tokenize, pack) | `src/nemotron/data_prep/` |
| Add a new CLI command | `src/nemotron/cli/commands/` + register in `cli/bin/nemotron.py` |
| Add a new recipe | Create `<stage>/train.py` with `[tool.runspec]` + `<stage>/config/default.yaml` |
| Modify execution backend | Edit `_execute_*()` in the relevant CLI command module |
| Evaluate a model | `src/nemotron/recipes/<model>/stage*_eval/` |
| Build custom benchmarks (MCQ) | `src/nemotron/customization_recipes/nemotron/stage3_byob/SKILL.md` |
| Quantize a model | `src/nemotron/customization_recipes/nemotron/stage5_quantization/SKILL.md` |

## SKILL.md References

| Skill | Path |
|-------|------|
| E2E Nemotron Customization | `src/nemotron/customization_recipes/nemotron/SKILL.md` |
| Stage 0: Continued Pretraining | `src/nemotron/customization_recipes/nemotron/stage0_cpt/SKILL.md` |
| Stage 1: SFT + SDG | `src/nemotron/customization_recipes/nemotron/stage1_sft/SKILL.md` |
| Stage 2: RL (DPO/GRPO) | `src/nemotron/customization_recipes/nemotron/stage2_rl/SKILL.md` |
| Stage 3: BYOB Benchmarks | `src/nemotron/customization_recipes/nemotron/stage3_byob/SKILL.md` |
| Stage 4: Evaluation | `src/nemotron/customization_recipes/nemotron/stage4_eval/SKILL.md` |
| Stage 5: Quantization | `src/nemotron/customization_recipes/nemotron/stage5_quantization/SKILL.md` |
| Shared Data Prep | `src/nemotron/customization_recipes/data_prep/SKILL.md` |
| Llama Customization | `src/nemotron/customization_recipes/llama/SKILL.md` |
| Qwen Customization | `src/nemotron/customization_recipes/qwen/SKILL.md` |

## Execution Backends

| Backend | Flag | Infrastructure | Notes |
|---------|------|---------------|-------|
| Local | (default) | torchrun on local GPUs | For dev/debug; single-node |
| Docker | `--run <profile>` | nemo-run + DockerExecutor | Local GPU container execution |
| Slurm (attached) | `--run <profile>` | nemo-run + SlurmExecutor | Logs streamed to terminal |
| Slurm (detached) | `--batch <profile>` | nemo-run + SlurmExecutor | Submit and exit |
| Lepton (DGX Cloud) | `--run <profile>` | nemo-run + LeptonExecutor | DGX Cloud via Lepton API; requires `node_group` |
| Run:AI | `--run <profile>` | nemo-run + KubeflowExecutor | Kubernetes GPU orchestration via Run:AI; requires `cluster` + `project` |
| Ray | (auto for RL) | nemo-run + RayJob | Used by GRPO/RL stages |

Env profiles are stored in `env.toml` at repo root (not checked in). Examples:

```toml
# --- Slurm cluster ---
[MY-CLUSTER]
executor = "slurm"
host = "login.cluster.example.com"
user = "myuser"
account = "myaccount"
partition = "batch"
remote_job_dir = "/lustre/myuser/jobs"
container = "nvcr.io/nvidia/nemo:26.02.super.rc1"
gpus_per_node = 8
nodes = 2

[MY-CLUSTER.wandb]
entity = "my-team"
project = "my-project"

# --- Lepton (DGX Cloud) ---
[lepton-dgx]
executor = "lepton"
container_image = "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
node_group = "my-dgx-group"
resource_shape = "gpu.8xh100-80gb"
nodes = 2
gpus_per_node = 8

[[lepton-dgx.mounts]]
path = "/shared-storage/data"
mount_path = "/data"

# --- Run:AI (Kubernetes) ---
[runai-cluster]
executor = "runai"
container_image = "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
cluster = "my-runai-cluster"
project = "my-team"
nodes = 2
gpus_per_node = 8
node_pool = "h100-pool"

[[runai-cluster.pvc_mounts]]
name = "training-data-pvc"
mount_path = "/data"
```

## Config Resolution Order

1. Recipe script `[tool.runspec]` PEP 723 metadata (name, image, config_dir, default config)
2. YAML config file from `config/` directory (selected via `-c` flag)
3. `env.toml` profile (selected via `--run`/`--batch` flag) -- merged into `run.env`
4. CLI key=value overrides (Hydra-style, e.g., `train.train_iters=5000`)

Artifact URIs (`${art:data,path}`, `${art:model,path}`) are resolved at config load time via `nemo_runspec.config.resolvers`.

## Container Images

| Model | Stage | Image |
|-------|-------|-------|
| Nano3 | Pretrain/SFT | `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano` |
| Nano3 | RL | `nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano` |
| Super3 | Pretrain/SFT | `nvcr.io/nvidian/nemo:26.02.super.rc1` |
| Customization | CPT/SFT | `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano` (or model-specific) |
| Customization | SDG | Requires NeMo DataDesigner |
| Customization | Eval | NeMo Evaluator launcher pulls its own containers |
