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

The published Lightning instruct evaluation (tech report §4.8, Table 21)
was run with NeMo Gym. The recipe has entries for the following Gym
benchmarks; only `gpqa` is enabled by default:

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

This is a subset of the report's suite. Table 21 also covers SWE-Bench
Verified, SWE-Bench Multilingual, Tau3 Banking, Terminal-Bench v2.1,
RULER-v1 (1M), IFBench, and LM Arena Proxy, which this recipe does not
configure.

### Reported results (post-trained model)

For reference, Table 21 of the tech report (*Nemotron 3.5 Super and
Lightning*, NVIDIA, 2026; link pending publication) gives these scores for the
post-trained Lightning checkpoints. Both were served with FP8 KV cache and
FP32 Mamba SSM cache (§4.8). The report notes that scores may differ from those
on the model cards because of run-to-run sampling and judge variability.

| Benchmark | BF16 | W4A16 NVFP4 |
|-----------|------|-------------|
| SWE-Bench Verified | 52.20 | 52.20 |
| SWE-Bench Multilingual | 37.13 | 37.00 |
| Tau2 Telecom | 59.65 | 60.96 |
| Tau3 Banking | 9.90 | 9.48 |
| Terminal-Bench v2.1 | 24.44 | 22.05 |
| GDPval | 17.19 | 19.09 |
| PinchBench | 83.86 | 84.78 |
| BrowseComp | 39.50 | 39.50 |
| SciCode (subtask) | 32.74 | 31.53 |
| GPQA Diamond (no tools) | 76.45 | 75.38 |
| HLE (no tools) | 10.80 | 10.70 |
| AA-Omniscience Accuracy | 17.47 | 16.35 |
| AA-Omni Non-Hallucination | 68.88 | 69.20 |
| CritPt | 0.00 | 0.00 |
| AA-LCR | 52.38 | 50.00 |
| RULER-v1 (1M) | 91.52 | 90.83 |
| IFBench (loose) | 72.00 | 74.04 |
| LM Arena Proxy | 12.06 | 11.10 |

NVFP4 vs BF16: median accuracy recovery 99.24%, verbosity change (ratio of
total tokens) −3.63%. See [Quantization](quantization.md).

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
a short-context lm-evaluation-harness suite plus RULER long-context tasks via
nemo-evaluator-launcher, not Gym. Those configs are maintained in the Gym
repo under `scripts/more/base/`.

The tech report's base results (§2.6, Table 6) cover 24 short-context tasks,
collected with NeMo Evaluator SDK and NVIDIA's LM Evaluation Harness
container. Table 6 has no RULER or other long-context scores for the base
model; the [base model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16)
reports RULER at 256K and 1M, collected with the NeMo Skills harness.

| Category | Task | Setting | Lightning 3.5 30B-A3B Base |
|----------|------|---------|----------------------------|
| General knowledge | MMLU | exact match | 78.63 |
| | MMLU-Pro | exact match | 67.50 |
| | AGIEval-En | CoT exact match | 68.62 |
| | SuperGPQA | CoT | 33.21 |
| | GPQA Diamond | 5-shot, CoT exact match | 33.33 |
| Math | GSM8K | 8-shot, CoT exact match | 91.74 |
| | Minerva MATH | exact match | 83.10 |
| | MATH-500 | 4-shot, sampled pass@1 (n=32) | 79.53 |
| Code | HumanEval | 0-shot, sampled pass@1 (n=32), EvalPlus | 76.91 |
| | MBPP-Sanitized | 3-shot, sampled pass@1 (n=32), EvalPlus | 76.98 |
| | LiveCodeBench CodeGen | pass@1 | 29.77 |
| | BigCodeBench Complete | greedy | 50.09 |
| | BigCodeBench Instruct | greedy | 51.75 |
| Commonsense | CommonsenseQA | 7-shot, acc | 81.08 |
| | ARC-Challenge | 25-shot, acc_norm | 93.17 |
| | HellaSwag | 10-shot, acc_norm | 85.25 |
| | OpenBookQA | 0-shot, acc_norm | 47.80 |
| | PIQA | 0-shot, acc_norm | 85.09 |
| | TruthfulQA MC2 | 0-shot, acc | 55.53 |
| | WinoGrande | 5-shot, acc | 78.06 |
| | SocialIQA | 0-shot, acc | 47.80 |
| Reading comprehension | RACE | 0-shot, acc | 87.85 |
| Multilingual | Global MMLU | average of 8 languages | 75.09 |
| | MGSM | 8-shot native CoT, average of 6 languages | 82.27 |
