# Stage 3: Evaluation (NeMo Gym)

Evaluate trained Nemotron 3.5 Lightning models using [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym).

> **Note**: NeMo Evaluator is deprecated for the instruct-model benchmark suite in
> favor of NeMo Gym. Base-model tasks (lm-evaluation-harness short-context suite,
> RULER long context) are not Gym environments and still run via
> nemo-evaluator-launcher.

## Overview

This stage reproduces the Gym-native flow used for the published Lightning
evaluation results (see Gym `scripts/more/reproducibility.md`):

1. Serve the HF checkpoint with vLLM in-job, using the Lightning family
   serving settings (`nemotron_v3` reasoning parser, `qwen3_coder` tool
   parser, fp32 Mamba SSM cache, optional MTP speculative decoding).
2. Wait for the OpenAI-compatible endpoint to become healthy.
3. For each configured benchmark, run `gym eval prepare` followed by
   `gym eval run --model-type vllm_model` against the endpoint.
4. Collect each benchmark's `*_aggregate_metrics.json` into `summary.json`.

The config supports `${art:model,path}` to automatically resolve model
artifacts from W&B lineage.

| Component | Description |
|-----------|-------------|
| `eval.py` | Recipe script: vLLM serving + Gym benchmark driver |
| `config/default.yaml` | Serving settings and Gym benchmark suite |

## Quick Start

```bash
# Evaluate the RL stage output (default: run.model=lightning35-rl-model:latest)
uv run nemotron lightning35 eval --run YOUR-CLUSTER

# Evaluate a specific model artifact
uv run nemotron lightning35 eval --run YOUR-CLUSTER run.model=sft-model:v2

# Filter specific benchmarks
uv run nemotron lightning35 eval --run YOUR-CLUSTER -t gpqa -t scicode

# Smoke run: 5 rows per benchmark
uv run nemotron lightning35 eval --run YOUR-CLUSTER gym.limit=5

# Preview the compiled config without submitting
uv run nemotron lightning35 eval --dry-run
```

## Benchmark suite

The published Lightning instruct evaluation uses the Gym-native reference
suite: GPQA Diamond, HLE, AA-LCR, AA-Omniscience, SciCode, BrowseComp, Tau3,
CritPt, GDPval, and PinchBench.

`config/default.yaml` enables `gpqa` by default (the only turnkey
benchmark; needs `HF_TOKEN` for the gated dataset). SciCode additionally
needs its ~1GB `test_data.h5`; the remaining benchmarks require judge
models or external API keys — add them to `gym.benchmarks` (with
per-benchmark `overrides`) once those assets are configured. The
reference-standard `++` overrides from Gym's published scripts are applied
to every run via `gym.common_overrides`.

## Outputs

Results land under `gym.output_dir` (default
`/nemo_run/lightning35-eval-results`):

```text
<benchmark>.jsonl                     # verified rollouts
<benchmark>_materialized_inputs.jsonl
<benchmark>_aggregate_metrics.json    # per-benchmark scores
vllm_serve.log                        # serving log
summary.json                          # stage-level roll-up
```

## Forking the execution

The vLLM command line, health check, and Gym invocation all live in
`eval.py` (`build_vllm_command`, `wait_for_endpoint`, `run_benchmark`).
Submission logic lives in `src/nemotron/cli/commands/lightning35/eval.py`.
Fork either file to change the behavior.
