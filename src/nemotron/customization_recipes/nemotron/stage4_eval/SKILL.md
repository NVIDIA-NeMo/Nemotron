# SKILL: Stage 4 -- Evaluation

## Purpose

Assess model quality through two complementary evaluation approaches: (1) model benchmark evaluation using NeMo Evaluator Launcher (same infrastructure as nano3/super3), and (2) data quality evaluation using NeMo Curator filters and scorers on training data.

## When to Use

Always run this stage. Evaluation is the gate between training and deployment.

- Run **model benchmark evaluation** (default) after each training stage (CPT, SFT, RL) to track progress
- Run **data quality evaluation** before and during training to catch data issues early
- Run both before stage5_quantization to establish pre-quantization baselines

## Inputs Required

Before running this stage, confirm these with the user:

| Input | Required? | Default | Notes |
|-------|-----------|---------|-------|
| Model checkpoint to evaluate | Yes | None | Ask: "Which model checkpoint? (path to CPT, SFT, or RL checkpoint)" |
| Evaluation mode | No | model (benchmark eval) | Ask: "Model benchmark evaluation, data quality evaluation, or both?" |
| Which benchmarks (model eval) | No | MMLU + ARC + HellaSwag | Ask: "Which benchmarks? (standard only, sovereign only, or both?)" |
| Sovereign benchmark name | If using BYOB benchmarks | None | Ask: "Name of the sovereign benchmark from stage 3? (e.g., hindi-medical-mcq)" |
| BYOB benchmark path | If sovereign and not yet compiled | None | Ask: "Path to the BYOB benchmark.jsonl from stage 3?" |
| Input data file (data eval) | If data quality mode | None | Ask: "Path to the training data file to assess quality? (JSONL)" |
| Quality recipe (data eval) | If data quality mode | None | Ask: "Quality recipe YAML path, or use default filters? (language, quality, repetition, word count)" |
| Compute: executor type | Yes | Slurm | Ask: "Where will this run? (local, Slurm, Lepton, Run:AI)" |
| Compute: GPUs | No | 1 node x 8 GPUs | Ask: "How many GPUs for model serving during eval?" |
| env.toml profile | If Slurm | None | Ask: "Which env.toml profile name for your cluster?" |

If any required input is missing, ask the user before proceeding.

## Architecture

Model evaluation uses **nemo-evaluator-launcher** directly -- the SAME execution pattern as `nano3 eval` and `super3 eval`. There is no recipe script and no nemo-run submission. The CLI command calls `run_eval()` from `nemo_evaluator_launcher.api.functional` after building and resolving the config.

Data quality evaluation uses NeMo Curator's filter/scorer pipeline via the `AssessmentTool` class in `data_prep/quality.py`.

## Prerequisites

| Prerequisite | Description |
|-------------|-------------|
| Model checkpoint | From any training stage (CPT, SFT, RL) |
| Benchmark data | Standard (MMLU, ARC, HellaSwag) or custom (from stage3_byob) |
| nemo-evaluator-launcher | `pip install "nemotron[evaluator]"` |
| GPU cluster | 1+ nodes x 8 GPUs (for model serving during eval) |
| Container | NeMo Framework container (model serving) + evaluator containers (auto-pulled) |
| env.toml profile | Required for `--run` mode (Slurm execution) |

## Sub-Stage 4a: Model Benchmark Evaluation (Default)

### How It Works

The eval command follows these steps (identical to nano3/super3):

1. Parse config from `stage4_eval/config/default.yaml`
2. Build job config with artifact resolution and env.toml injection
3. Auto-inject W&B env mappings if export is configured
4. Auto-squash container images for Slurm execution
5. Save configs (job.yaml for provenance, eval.yaml for launcher)
6. Resolve artifacts (`${art:model,path}`)
7. Call `nemo_evaluator_launcher.api.functional.run_eval()`

### Running Model Evaluation

```bash
# Eval on cluster (loads env.toml profile)
nemotron customize eval --run MY-CLUSTER

# Override model artifact
nemotron customize eval --run MY-CLUSTER run.model=sft:v2

# Filter specific benchmark tasks
nemotron customize eval --run MY-CLUSTER -t adlr_mmlu -t hellaswag

# Dry run (show resolved config without executing)
nemotron customize eval --run MY-CLUSTER --dry-run

# Local execution
nemotron customize eval execution.type=local
```

### Default Benchmarks

| Benchmark | Category | What It Tests |
|-----------|----------|--------------|
| MMLU (`adlr_mmlu`) | Knowledge/Multilingual | Broad knowledge retention and multilingual capability |
| ARC Challenge (`adlr_arc_challenge_llama_25_shot`) | Reasoning | Multi-step reasoning ability |
| HellaSwag (`hellaswag`) | Common sense | Natural language understanding and common sense |

### Adding Custom Benchmarks from BYOB (Legacy)

After running stage3_byob to generate a domain-specific MCQ benchmark, you can add it as a raw JSONL task in your config. However, the **recommended** approach is to use sovereign benchmarks (see next section).

```yaml
evaluation:
  tasks:
    # ... standard benchmarks ...
    - name: domain_mcq
      nemo_evaluator_config:
        config:
          params:
            top_p: 0.0
        target:
          api_endpoint:
            adapter_config:
              output_dir: /results/domain_mcq
              dataset_path: /data/byob_benchmark/benchmark.jsonl
```

## Sovereign Benchmarks

Sovereign benchmarks are domain/language-specific evaluation sets built from the stage3 BYOB pipeline and compiled into the NeMo Evaluator using its BYOB framework. This gives you proper MCQ scoring, per-topic breakdowns, and the ability to bake benchmarks into a "sovereign container" for reproducible evaluation.

### When to Use What

```
Do you need domain/language-specific evaluation?
  NO  --> Use standard benchmarks only (MMLU, ARC, HellaSwag)
  YES --> Did you run stage3_byob?
            NO  --> Run stage3_byob first (see stage3_byob/SKILL.md)
            YES --> Create a sovereign benchmark (this section)
                    |
                    v
            Do you also need standard benchmarks?
              YES --> Include BOTH in the same eval run (recommended)
              NO  --> Include only the sovereign benchmark
```

### Creating a Sovereign Benchmark

There are three paths, from easiest to most customizable:

**Path 1: Auto-generate (recommended for most cases)**

```bash
python src/nemotron/customization_recipes/nemotron/stage4_eval/create_sovereign_benchmark.py \
  --byob-output /data/byob_benchmark/benchmark.jsonl \
  --benchmark-name "hindi-medical-mcq" \
  --output-dir /data/eval/benchmarks/ \
  --compile
```

This reads the benchmark.jsonl, detects the number of choices (4 or 10), extracts topic/language metadata, and generates + compiles a benchmark definition.

**Path 2: Environment variable override (no code changes)**

```bash
export SOVEREIGN_BENCHMARK_NAME="hindi-medical-mcq"
export SOVEREIGN_DATASET_PATH="/data/byob_benchmark/benchmark.jsonl"
export SOVEREIGN_LANGUAGE="hi"

nemo-evaluator-byob \
  src/nemotron/customization_recipes/nemotron/stage4_eval/sovereign_benchmark.py
```

**Path 3: Copy and customize template (maximum control)**

```bash
# Copy template
cp src/nemotron/customization_recipes/nemotron/stage4_eval/sovereign_benchmark.py \
   /data/eval/benchmarks/my_benchmark.py

# Edit: change BENCHMARK_NAME, DATASET_PATH, LANGUAGE, prompt template, scorer logic
# Then compile
nemo-evaluator-byob /data/eval/benchmarks/my_benchmark.py
```

### Running Evaluation with Sovereign Benchmarks

After compiling, the benchmark is auto-discoverable by the evaluator. Add it to your eval run alongside standard benchmarks:

```bash
# Standard + sovereign benchmarks in one eval run
nemotron customize eval --run MY-CLUSTER \
  -t adlr_mmlu \
  -t adlr_arc_challenge_llama_25_shot \
  -t hellaswag \
  -t byob_hindi_medical_mcq.hindi-medical-mcq
```

Or add it to `config/default.yaml` permanently:

```yaml
evaluation:
  tasks:
    - name: adlr_mmlu
      nemo_evaluator_config:
        config:
          params:
            top_p: 0.0
    - name: adlr_arc_challenge_llama_25_shot
    - name: hellaswag
    # Sovereign benchmark (compiled via nemo-evaluator-byob)
    - name: byob_hindi_medical_mcq.hindi-medical-mcq
      nemo_evaluator_config:
        config:
          params:
            top_p: 0.0
```

### Containerization: Building the Sovereign Container

The "sovereign container" is an evaluator container image with sovereign benchmarks baked in. This is useful for:
- Airgapped / disconnected environments
- Reproducible evaluation across teams
- CI/CD pipelines

```bash
# Generate + compile + containerize in one step
python src/nemotron/customization_recipes/nemotron/stage4_eval/create_sovereign_benchmark.py \
  --byob-output /data/byob_benchmark/benchmark.jsonl \
  --benchmark-name "hindi-medical-mcq" \
  --output-dir /data/eval/benchmarks/ \
  --containerize

# Or containerize from the template directly
nemo-evaluator-byob \
  src/nemotron/customization_recipes/nemotron/stage4_eval/sovereign_benchmark.py \
  --containerize
```

The `--containerize` flag creates a Docker image based on the evaluator base image with the compiled benchmark installed. Push this image to your registry for use in Slurm, Kubernetes, or DGX Cloud deployments.

### Sovereign Benchmark Scoring

The sovereign benchmark scorer:
1. Extracts the predicted answer letter (A-D for 4-choice, A-J for 10-choice) from the model response
2. Handles common LLM response formats: bare letter, "The answer is X", parenthesized, etc.
3. Compares against the ground-truth answer from the BYOB dataset
4. Reports per-topic and per-language accuracy breakdowns (if metadata is present)

Metrics produced:
- `correct` (bool) -- primary accuracy metric
- `parsed` (bool) -- whether a valid answer letter was extracted
- `correct_<topic>` (bool) -- per-topic breakdown
- `correct_<language>` (bool) -- per-language breakdown

### Files Reference

| File | Purpose |
|------|---------|
| `sovereign_benchmark.py` | BYOB benchmark template (copy + customize or use with env vars) |
| `create_sovereign_benchmark.py` | Auto-generator utility (reads BYOB output, writes + compiles benchmark) |
| `config/default.yaml` | Eval config with sovereign benchmark task entries |

### Deployment Parallelism

The default config uses TP=2, EP=8 (suitable for Nemotron Nano 30B MoE). Adjust for your model:

```bash
# For a dense model with TP=8 (e.g., Super 48B)
nemotron customize eval --run MY-CLUSTER \
  'deployment.command=bash -c "python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py --megatron_checkpoint /checkpoint/ --num_gpus 8 --tensor_model_parallel_size 8 --port 1235 --num_replicas 1"'
```

### env.toml Profile Example

```toml
[MY-CLUSTER]
executor = "slurm"
host = "login.cluster.nvidia.com"
user = "<your-username>"
account = "nemo-eval"
partition = "batch"
container = "nvcr.io/nvidia/nemo:25.11.nemotron"
remote_job_dir = "/lustre/<your-username>/.nemotron"
time = "04:00:00"

[MY-CLUSTER.wandb]
entity = "nvidia-nemo"
project = "customize-eval"
```

## Sub-Stage 4b: Data Quality Assessment

### Running Data Quality Evaluation

```bash
nemotron customize eval --mode data \
  data_eval.input_file=/data/cpt_prepared/train.jsonl \
  data_eval.recipe=/path/to/quality_recipe.yaml
```

### Quality Recipe Format

The recipe YAML defines a list of NeMo Curator filters and scorers:

```yaml
- name: WordCountFilter
  alias: word_count
  parameters:
    min_words: 50
    max_words: 10000
- name: LanguageFilter
  alias: language_id
  parameters:
    language: hi
    threshold: 0.7
- name: FastTextQualityFilter
  alias: quality
  parameters:
    threshold: 0.5
- name: RepetitiousFilter
  alias: repetition
  parameters:
    max_repeat_ratio: 0.3
```

### Quality Metrics to Track

| Metric | Target | Tool |
|--------|--------|------|
| Language ID confidence | >0.7 for target language | `LanguageFilter` |
| Quality score | >0.5 | `FastTextQualityFilter` |
| Repetition ratio | <0.3 | `RepetitiousFilter` |
| Word count | 50-10000 | `WordCountFilter` |

## Evaluation Thresholds

### Language Adaptation Targets

| Metric | Target | Red Flag |
|--------|--------|----------|
| Target-language MMLU | >55% (Nano), >70% (Super) | <40% |
| English MMLU retention | <5% drop from base | >10% drop |
| Custom BYOB MCQ | >65% | <50% |
| HellaSwag | <3% drop from base | >8% drop |

### Domain Adaptation Targets

| Metric | Target | Red Flag |
|--------|--------|----------|
| Domain BYOB MCQ | >70% | <55% |
| General knowledge retention | <3% drop | >8% drop |
| Domain perplexity | >30% reduction from base | No reduction |

### Post-RL Targets

| Metric | Target | Red Flag |
|--------|--------|----------|
| Reward (GRPO) | Increasing trend | Flat or decreasing |
| DPO preference accuracy | >70% | <55% |
| Safety benchmark | >90% safe responses | <80% |

## How to Verify Success

1. **All standard benchmarks pass retention thresholds**: English performance does not degrade significantly.
2. **Target domain/language benchmarks show improvement**: Clear improvement over the base model.
3. **Custom BYOB benchmark scores**: Above domain-specific thresholds.
4. **Data quality report**: No anomalies flagged in training data.
5. **Results exported to W&B**: Check the eval dashboard for trends across iterations.

## Troubleshooting

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| Evaluation hangs | Model deployment failed (OOM or container issue) | Check deployment logs, reduce model parallelism, verify container image |
| All benchmarks score 0% | API endpoint not reachable | Check `deployment.port`, verify health check passes |
| English scores dropped significantly | Catastrophic forgetting in CPT/SFT | Return to stage0/stage1, adjust data blend (more English) |
| Custom benchmark scores low | Insufficient training on domain | Increase CPT data volume, add more domain SFT data |
| Evaluation timeout | Request timeout too low or model too slow | Increase `request_timeout`, check GPU utilization during inference |
| W&B export fails | Missing API key or wrong project | Set `WANDB_API_KEY`, verify `export.wandb.project` |
| Tokenizer error during eval | Tokenizer path mismatch | Verify `tokenizer` path in `nemo_evaluator_config` points to checkpoint's tokenizer |
| `nemo-evaluator-launcher not found` | Missing dependency | `pip install "nemotron[evaluator]"` |

## Artifacts Produced

| Artifact | Type | Path | Consumed By |
|----------|------|------|-------------|
| Evaluation results | JSON | `execution.output_dir/results/` | Decision to proceed to stage5 |
| Data quality report | JSON | `data_eval.output_dir/` | Data iteration feedback |
| W&B eval dashboard | W&B artifact | W&B project | Stakeholder review |
| Per-task scores | JSON | `execution.output_dir/results/<task>/` | Detailed analysis |
