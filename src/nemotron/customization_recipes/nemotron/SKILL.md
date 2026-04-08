# SKILL: Nemotron Model Customization Pipeline

## Purpose

End-to-end pipeline for customizing NVIDIA Nemotron models (Nano, Super, Ultra) to new languages, domains, and use cases. Implements the Sovereign AI Playbook pattern: take a base Nemotron model, adapt it with target-language/domain data, fine-tune for instruction following, align with RL, build domain-specific benchmarks, evaluate, and quantize for deployment.

## When to Use This Pipeline

Use this pipeline when you need to:
- Adapt a Nemotron model to a new natural language (e.g., Hindi, Thai, Arabic)
- Specialize a model for a domain (e.g., medical, legal, financial)
- Combine language + domain adaptation (e.g., Hindi medical)
- Build evaluation benchmarks for a target language/domain
- Produce a deployment-ready quantized model

Do NOT use this pipeline if you are:
- Training a Nemotron model from scratch (use `src/nemotron/recipes/nano3/` or `super3/`)
- Fine-tuning an embedding model (use `src/nemotron/recipes/embed/`)
- Curating web-scale pretraining data (use `src/nemotron/recipes/data_curation/`)

## Pipeline Overview

```
stage0_cpt --> stage1_sft --> stage2_rl --> stage3_byob --> [bridge] --> stage4_eval --> stage5_quantization
(data+train)  (SDG+train)  (DPO/GRPO)    (MCQ gen)     (sovereign     (benchmark)     (INT4/FP8)
                                                         benchmark)
```

| Stage | Name | Purpose | Inputs | Outputs |
|-------|------|---------|--------|---------|
| 0 | Continued Pretraining (CPT) | Inject language/domain knowledge into base model | Raw corpora + base model | CPT checkpoint |
| 1 | Supervised Fine-Tuning (SFT) | Teach instruction following in target language/domain | CPT checkpoint + SFT data (real or synthetic) | SFT checkpoint |
| 2 | Reinforcement Learning (RL) | Align model preferences and improve reasoning | SFT checkpoint + preference data | RL checkpoint |
| 3 | Build Your Own Benchmark (BYOB) | Generate MCQ evaluation sets from domain corpora | Domain text corpora | MCQ benchmark dataset |
| 4 | Evaluation | Assess data quality and model performance | Model checkpoint + benchmark data | Evaluation metrics |
| 5 | Quantization | Compress model for deployment | RL/SFT checkpoint | Quantized model (INT4/FP8) |

## Decision Tree: Which Stages to Run

```
START
  |
  v
Is target language different from English?
  YES --> Run stage0_cpt (language CPT)
  NO  --> Is target domain specialized?
            YES --> Run stage0_cpt (domain CPT) OR skip to stage1_sft
            NO  --> Skip to stage1_sft
  |
  v
Do you have supervised instruction data?
  YES --> Run stage1_sft with real data
  NO  --> Run stage1_sft with SDG (synthetic data generation)
  |
  v
Do you need preference alignment?
  YES --> Run stage2_rl (DPO for preference data, GRPO for reward-based)
  NO  --> Skip to stage3_byob or stage4_eval
  |
  v
Do you need domain-specific evaluation?
  YES --> Run stage3_byob to generate MCQ benchmarks
          --> Run bridge: create_sovereign_benchmark.py to compile for evaluator
  NO  --> Use existing benchmarks in stage4_eval
  |
  v
Run stage4_eval (always recommended -- include both standard + sovereign benchmarks)
  |
  v
Deploying to production?
  YES --> Run stage5_quantization
  NO  --> Use checkpoint directly for research
```

## Directory Structure

```
src/nemotron/customization_recipes/nemotron/
  SKILL.md                            <-- This file
  __init__.py
  stage0_cpt/
    SKILL.md                          <-- Stage-specific skill
    __init__.py
    config/
      data_prep/                      <-- Data acquisition configs
    run_cpt.py, run_data_prep.py      <-- Recipe scripts
  stage1_sft/
    SKILL.md
    __init__.py
    config/
      data_prep/                      <-- SFT data prep configs
      sdg/                            <-- Synthetic data generation configs
  stage2_rl/
    SKILL.md
    __init__.py
    config/                           <-- DPO/GRPO configs
  stage3_byob/
    SKILL.md
    __init__.py
    config/                           <-- BYOB pipeline configs
  stage4_eval/
    SKILL.md
    __init__.py
    config/                           <-- Evaluation configs
  stage5_quantization/
    SKILL.md
    __init__.py
    config/                           <-- Quantization configs
```

## Per-Stage SKILL.md References

Each stage has a detailed SKILL.md. Read the relevant stage SKILL.md before executing that stage.

| Stage | SKILL.md Path | Key Tools |
|-------|---------------|-----------|
| 0 - CPT | `stage0_cpt/SKILL.md` | NeMo Curator, nemo_automodel, Megatron-Bridge |
| 1 - SFT | `stage1_sft/SKILL.md` | DataDesigner (SDG), nemotron.data_prep, nemo_automodel |
| 2 - RL | `stage2_rl/SKILL.md` | NeMo-RL, nemo_automodel |
| 3 - BYOB | `stage3_byob/SKILL.md` | NIM API, NeMo Curator |
| 4 - Eval | `stage4_eval/SKILL.md` | NeMo Evaluator, NeMo Curator quality filters |
| 5 - Quant | `stage5_quantization/SKILL.md` | TensorRT-LLM, TensorRT Model Optimizer |

## Shared Data Prep

The `data_prep/` module provides shared utilities for all customization stages. See `src/nemotron/customization_recipes/data_prep/SKILL.md`.

Key capabilities:
- `nemotron.data_prep.api.run_pretrain_pipeline()` -- tokenize to bin/idx for CPT
- `nemotron.data_prep.api.run_sft_pipeline()` -- pack to Parquet for SFT
- `nemotron.data_prep.recipes.rl` -- prepare JSONL for RL
- Data blending, filtering, deduplication, translation

## Multi-Container Deployment (Docker Compose)

The recommended way to run the customization pipeline is via Docker Compose.
Five services run in parallel; you interact only with the **orchestrator**,
which automatically dispatches commands to the correct container.

### Quick Start

```bash
cd deploy/nemotron/customization_recipes

# Set API keys
export NGC_API_KEY=<your-key>
export HF_TOKEN=<your-token>

# Start all services
docker compose up -d

# Run ANY customization command from the orchestrator — the dispatcher
# routes to the right container automatically:
docker compose exec nemotron-orchestrator nemotron customize data-prep -c default
docker compose exec nemotron-orchestrator nemotron customize sft -c default
docker compose exec nemotron-orchestrator nemotron customize eval -c default

# Or enter the orchestrator and run interactively:
docker compose exec nemotron-orchestrator bash
nemotron customize sft -c default --run MY-CLUSTER
```

### Command Routing

The dispatcher maps each subcommand to the container that has the right
dependencies installed:

| Subcommand | Container | Reason |
|------------|-----------|--------|
| `data-prep` | `nemotron-curator` | NeMo Curator for data processing |
| `sdg` | `nemotron-curator` | DataDesigner for synthetic generation |
| `byob` | `nemotron-curator` | BYOB MCQ pipeline uses NeMo Curator |
| `cpt` | `nemotron-trainer` | CPT needs NeMo + Megatron-Bridge |
| `sft` | `nemotron-trainer` | SFT needs NeMo + Megatron-Bridge |
| `rl` | `nemotron-trainer` | RL needs NeMo + Ray |
| `eval` | `nemotron-evaluator` | Uses nemo-evaluator-launcher |
| `quantize` | `nemotron-trainer` | Needs model loading + TensorRT |

You never need to remember which container to exec into -- just run the
command and the dispatcher handles it.

### Start with Local NIM (optional)

```bash
docker compose --profile with-nim up -d
```

## E2E Example: Customize Nemotron Nano for Hindi Medical Domain

This walkthrough shows the complete pipeline for adapting Nemotron-3-Nano to Hindi medical text.

### Prerequisites

```bash
cd deploy/nemotron/customization_recipes

# Set environment variables
export NGC_API_KEY=<your-key>
export HF_TOKEN=<your-token>
export NVIDIA_API_KEY=<your-nim-key>   # for SDG/BYOB

# Start the multi-container stack
docker compose up -d

# All commands below are run from the orchestrator:
docker compose exec nemotron-orchestrator bash
```

### Execution Backends

All stages support multiple execution backends via env.toml profiles.
The dispatcher forwards all flags and overrides to the target container:

```bash
# Local (default) -- dispatched to the right container automatically
nemotron customize cpt -c default

# Slurm cluster
nemotron customize cpt -c default --run MY-CLUSTER

# Lepton (DGX Cloud)
nemotron customize cpt -c default --run lepton-dgx

# Run:AI (Kubernetes GPU orchestration)
nemotron customize cpt -c default --run runai-cluster
```

Example env.toml profiles for each backend:

```toml
# --- Slurm ---
[MY-CLUSTER]
executor = "slurm"
host = "login.cluster.example.com"
user = "myuser"
account = "myaccount"
partition = "batch"
remote_job_dir = "/lustre/myuser/jobs"
container_image = "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
gpus_per_node = 8
nodes = 2

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

### Stage 0: Continued Pretraining on Hindi Medical Data

Goal: Inject Hindi language + medical domain knowledge into the base Nemotron Nano model.

```bash
# All commands run from the orchestrator. The dispatcher routes
# data-prep -> nemotron-curator, cpt -> nemotron-trainer.

# 1. Acquire and prepare data (routed to nemotron-curator)
nemotron customize data-prep -c default \
  data.sources.0.dataset=ai4bharat/sangraha \
  data.sources.0.language=hi \
  output_dir=/workspace/data/cpt_prepared

# 2. Run CPT training (routed to nemotron-trainer)
nemotron customize cpt -c default \
  --run MY-CLUSTER \
  train.train_iters=10000 \
  checkpoint.save=/workspace/results/hindi_medical_cpt
```

**Key decisions:**
- Data blend: 70% target language, 20% English (knowledge retention), 10% code
- Learning rate: 1e-5 (lower than pretrain to avoid catastrophic forgetting)
- Train iterations: 5000-20000 depending on data volume (target ~10B tokens)

**Artifacts produced:**
- CPT model checkpoint at `checkpoint.save` path
- Data preparation artifacts (bin/idx blends) at `output_dir`

### Stage 1: SFT with Synthetic Data Generation

Goal: Fine-tune the CPT model for instruction following in Hindi medical domain.

```bash
# 1. Generate synthetic instruction data (routed to nemotron-curator)
nemotron customize sdg -c default \
  domain=medical \
  language=hi \
  num_samples=50000 \
  output_dir=/workspace/data/sdg_output

# 2. Prepare SFT data (routed to nemotron-curator)
nemotron customize data-prep -c default \
  --mode sft \
  data.sources.0.path=/workspace/data/sdg_output \
  output_dir=/workspace/data/sft_prepared

# 3. Run SFT training (routed to nemotron-trainer)
nemotron customize sft -c default \
  --run MY-CLUSTER \
  checkpoint.pretrained_checkpoint=/workspace/results/hindi_medical_cpt \
  train.train_iters=1700
```

**Key decisions:**
- SDG sample count: 50K-200K depending on domain complexity
- Data blend: 60% synthetic domain, 30% general instruction, 10% safety
- Pack size: 4096 tokens (match model context length)
- Learning rate: 5e-6 (lower than CPT)

**Artifacts produced:**
- SDG dataset at `sdg_output`
- Packed Parquet SFT data at `sft_prepared`
- SFT model checkpoint

### Stage 2: Reinforcement Learning

Goal: Align model with human preferences and improve reasoning quality.

```bash
# All RL commands are routed to nemotron-trainer automatically.

# Run DPO training (if you have preference pairs)
nemotron customize rl -c dpo \
  --run MY-CLUSTER \
  policy.model_name=/workspace/results/hindi_medical_sft \
  data.train_jsonl_fpath=/workspace/data/preferences_train.jsonl

# OR run GRPO training (reward-model-based)
nemotron customize rl -c grpo \
  --run MY-CLUSTER \
  policy.model_name=/workspace/results/hindi_medical_sft \
  data.train_jsonl_fpath=/workspace/data/prompts_train.jsonl
```

**Key decisions:**
- DPO vs GRPO: Use DPO if you have chosen/rejected pairs; GRPO if you have a reward signal
- KL penalty: 0.0-0.1 (higher = more conservative alignment)
- Clip ratio: 0.2-0.28

**Artifacts produced:**
- RL-aligned model checkpoint

### Stage 3: Build Your Own Benchmark

Goal: Generate MCQ evaluation sets from Hindi medical text corpora.

```bash
# Routed to nemotron-curator automatically
nemotron customize byob -c default \
  input_corpus=/workspace/data/hindi_medical_reference_texts \
  output_dir=/workspace/data/byob_benchmark \
  language=hi \
  domain=medical \
  num_questions=5000
```

The BYOB pipeline runs 5 sub-stages: generate → judge → expand distractors → validity check → filter. (Semantic dedup, coverage check, and outlier detection are planned but not yet wired.)

**Artifacts produced:**
- MCQ benchmark dataset in standardized format
- Quality metrics (coverage, validity scores)

### Bridge: BYOB -> Sovereign Benchmark

Goal: Convert BYOB MCQ output into a compiled NeMo Evaluator benchmark for use in stage4.

```bash
# Auto-generate and compile a sovereign benchmark from BYOB output
python src/nemotron/customization_recipes/nemotron/stage4_eval/create_sovereign_benchmark.py \
  --byob-output /data/byob_benchmark/benchmark.jsonl \
  --benchmark-name "hindi-medical-mcq" \
  --output-dir /data/eval/benchmarks/ \
  --compile
```

This creates a NeMo Evaluator BYOB benchmark definition that:
- Reads the BYOB-generated MCQ dataset
- Formats MCQ prompts for the model (supports 4-choice and 10-choice formats)
- Scores responses by extracting the predicted answer letter
- Reports per-topic and per-language accuracy breakdowns

The compiled benchmark is auto-discoverable by the evaluator and can be included alongside standard benchmarks (MMLU, ARC, HellaSwag) in the same eval run.

**Artifacts produced:**
- Compiled BYOB benchmark package (auto-installed in `~/.nemo-evaluator/byob_packages/`)
- (Optional) Docker image with benchmark baked in (with `--containerize`)

### Stage 4: Evaluation

Goal: Assess model quality on standard + sovereign benchmarks.

**Model evaluation** (uses nemo-evaluator-launcher, same as nano3/super3):
```bash
# Routed to nemotron-evaluator automatically
nemotron customize eval -c default \
  --run MY-CLUSTER \
  deployment.checkpoint_path=/workspace/results/hindi_medical_rl \
  -t adlr_mmlu \
  -t adlr_arc_challenge_llama_25_shot \
  -t hellaswag \
  -t byob_hindi_medical_mcq.hindi-medical-mcq
```

**Data quality evaluation** (uses NeMo Curator filters for quality assessment):
```bash
nemotron customize eval --mode data \
  data_eval.input_file=/workspace/data/hindi_medical_sft.jsonl \
  data_eval.output_dir=/workspace/results/data_quality \
  data_eval.recipe=/workspace/configs/quality_recipe.yaml
```

This runs filters (language, domain, perplexity, coherence, tool-calling accuracy) on your training data and produces aggregate quality metrics. Use before training to catch data issues early.

**Expected thresholds (Hindi medical):**
- MMLU (Hindi subset): >60% accuracy
- Custom BYOB MCQ: >70% accuracy
- General English retention: <5% drop from base model

### Stage 5: Quantization

Goal: Produce deployment-ready model.

```bash
# Routed to nemotron-trainer automatically (needs model loading + TensorRT)
nemotron customize quantize -c default \
  model_path=/workspace/results/hindi_medical_rl \
  output_dir=/workspace/results/hindi_medical_int4 \
  quantization=int4_awq
```

**Artifacts produced:**
- Quantized model checkpoint (INT4 AWQ or FP8)
- Calibration metadata

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `HF_TOKEN` | Yes | HuggingFace model/data downloads |
| `NVIDIA_API_KEY` | For SDG/BYOB | NIM API access for synthetic data generation |
| `WANDB_API_KEY` | Recommended | Experiment tracking and artifact lineage |
| `WANDB_PROJECT` | Recommended | W&B project name |
| `WANDB_ENTITY` | Recommended | W&B team/entity |
| `NEMO_HOME` | Optional | Cache directory for NeMo artifacts |

## Config System

All configs use OmegaConf YAML with the following resolution chain:

1. Default YAML in `stage*/config/default.yaml`
2. `env.toml` profile injected via `--run <profile>`
3. CLI overrides via `key=value` (Hydra-style, supports nested: `train.lr=1e-5`)

Artifact URIs (`${art:<name>,<field>}`) resolve model/data paths from the artifact registry (W&B or fsspec).

Common config patterns:
```yaml
run:
  data: <artifact-name>:latest    # Input data artifact
  model: <artifact-name>:latest   # Input model artifact
  env:
    container: nvcr.io/...        # Container image
    mounts: [...]                 # Volume mounts for Slurm

recipe:
  _target_: <module.path.function>  # Recipe callable
  <param>: <value>                  # Recipe-specific params

train:
  train_iters: <int>
  global_batch_size: <int>

checkpoint:
  save: /path/to/output
  pretrained_checkpoint: ${art:model,path}
```

## Artifact Lineage

The pipeline uses `nemotron.kit` for artifact tracking:

```python
import nemotron.kit as kit

# Initialize
kit.init(backend="wandb", wandb_project="my-customization")

# Save artifact
artifact = kit.ModelArtifact(path=Path("/results/checkpoint"), iteration=10000)
artifact.save(name="hindi-medical-cpt-model")

# Load artifact
loaded = kit.ModelArtifact.from_uri("art://hindi-medical-cpt-model:latest")
```

Each stage consumes artifacts from the previous stage and produces artifacts for the next. The artifact registry (W&B or fsspec) tracks the full lineage graph.

## Troubleshooting

| Issue | Likely Cause | Resolution |
|-------|-------------|------------|
| OOM during CPT | Batch size too large or model parallelism insufficient | Reduce `global_batch_size`, increase `tensor_model_parallel_size` |
| Loss not decreasing in CPT | Learning rate too high, data quality issues | Reduce LR to 5e-6, check data with stage4_eval data quality filters |
| Catastrophic forgetting | Too much target-domain data, too few train iterations | Adjust data blend (add more English), reduce LR, add replay data |
| SFT overfitting | Too many iterations on small SDG dataset | Reduce `train_iters`, increase SDG `num_samples`, add regularization |
| RL reward collapse | KL penalty too low or reward hacking | Increase `reference_policy_kl_penalty`, check reward model quality |
| BYOB low quality MCQs | Source corpus too short or low quality | Filter input corpus for length/quality, increase judge temperature |
| Eval scores below threshold | Insufficient CPT/SFT data or too few training steps | Increase data volume and training iterations, iterate |
| Quantization accuracy drop >2% | Calibration data mismatch | Use domain-representative calibration data, try FP8 instead of INT4 |
