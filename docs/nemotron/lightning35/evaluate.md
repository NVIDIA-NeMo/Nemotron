# Evaluate Nemotron 3.5 Lightning

Evaluate trained Nemotron 3.5 Lightning checkpoints with [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym).

> **Note**: NeMo Evaluator is deprecated for the instruct-model benchmark
> suite in favor of NeMo Gym. Base-model tasks (the lm-evaluation-harness
> short-context suite and RULER long-context) are not Gym environments and
> still run via nemo-evaluator-launcher.

## How it works

The eval stage is a regular recipe script
(`src/nemotron/recipes/lightning35/stage3_eval/eval.py`) submitted like any
other training stage. Inside the job it:

1. **Serves the checkpoint** with vLLM using the Lightning family settings:
   `nemotron_v3` reasoning parser, `qwen3_coder` tool parser, fp32 Mamba SSM
   cache, expert parallelism, and optional MTP speculative decoding
   (the released checkpoint ships a repeated-layer MTP module; speculation depth is configurable).
2. **Waits** for the OpenAI-compatible endpoint to become healthy.
3. **Runs each benchmark** with `gym eval prepare` + `gym eval run
   --model-type vllm_model` against the endpoint.
4. **Aggregates** each benchmark's `*_aggregate_metrics.json` into a
   stage-level `summary.json`.

This reproduces the Gym-native flow used for the published Lightning
evaluation numbers (see Gym `scripts/more/reproducibility.md`).

## Quick start

```bash
# Evaluate the RL stage output (default: run.model=lightning35-rl-model:latest)
uv run nemotron lightning35 eval --run YOUR-CLUSTER

# Evaluate a specific model artifact from W&B lineage
uv run nemotron lightning35 eval --run YOUR-CLUSTER run.model=sft-model:v2

# Evaluate an explicit HF checkpoint path
uv run nemotron lightning35 eval --run YOUR-CLUSTER serving.model_path=/path/to/hf_ckpt

# Filter benchmarks
uv run nemotron lightning35 eval --run YOUR-CLUSTER -t gpqa -t scicode

# Smoke run: 5 rows per benchmark
uv run nemotron lightning35 eval --run YOUR-CLUSTER gym.limit=5 gym.concurrency=8

# Preview the compiled config
uv run nemotron lightning35 eval --dry-run
```

## Configuration

`src/nemotron/recipes/lightning35/stage3_eval/config/default.yaml` has three
sections:

| Section | Purpose | Key fields |
|---------|---------|------------|
| `run` | Artifact resolution + cluster profile | `model` (W&B artifact), `env.container` |
| `serving` | vLLM serving of the checkpoint | `model_path`, `tensor_parallel_size` (4), `enable_expert_parallel`, `reasoning_parser`, `speculative_config` |
| `gym` | Benchmark suite and Gym invocation | `benchmarks`, `concurrency`, `limit`, `num_repeats`, `output_dir`, `extra_overrides` |

The model checkpoint must be in HF format (the RL stage output already is;
Megatron checkpoints must be exported first — see
[import/export](import.md)).

## Benchmark suite

The published Lightning instruct evaluation uses the Gym-native reference
suite:

| Benchmark | Self-contained | Notes |
|-----------|----------------|-------|
| `gpqa` | ✅ | GPQA Diamond, enabled by default (needs `HF_TOKEN` for the gated dataset) |
| `scicode` | ⚠️ | Needs SciCode's ~1GB `test_data.h5` (per-benchmark override) |
| `hle` | ❌ | Requires a judge model |
| `aalcr` | ❌ | AA-LCR, requires a judge model |
| `omniscience` | ❌ | AA-Omniscience, requires a judge model |
| `browsecomp` | ❌ | Requires a search API key |
| `tau2` | ❌ | Requires environment/tool servers |
| `critpt` | ❌ | Requires a judge model |
| `gdpval` | ❌ | Requires a judge model |
| `pinchbench` | ❌ | Requires a judge model |

Enable additional benchmarks by extending `gym.benchmarks` once the required
endpoints/keys are configured, and pass judge/API settings through
`gym.extra_overrides`.

## Outputs

Results land under `gym.output_dir` (default
`/nemo_run/lightning35-eval-results`):

```text
gpqa.jsonl                        # one verified rollout per task/repeat
gpqa_materialized_inputs.jsonl
gpqa_aggregate_metrics.json       # per-benchmark scores (mean/reward, pass@k, ...)
vllm_serve.log
summary.json                      # stage-level roll-up across benchmarks
```

The stage exits non-zero if any configured benchmark fails, and logs a
per-benchmark status table.

## Base-model evaluation

The base checkpoint
(`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16`) is evaluated with
21 short-context lm-evaluation-harness tasks plus RULER (64K–1M context) via
nemo-evaluator-launcher, not Gym. Those configs are maintained in the Gym
repo under `scripts/more/base/`.
