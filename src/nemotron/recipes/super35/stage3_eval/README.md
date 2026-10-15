# Stage 3: Evaluation

Evaluate Nemotron 3.5 Super VL checkpoints with NeMo Evaluator.

## Overview

This stage evaluates a Hugging Face checkpoint (for example the exported MOPD phase 2 checkpoint)
by serving it with vLLM and running NeMo Evaluator tasks through `nemo-evaluator-launcher`. Task names and
per-task settings follow NeMo Evaluator's Nemotron 3 Super example
(`packages/nemo-evaluator-launcher/examples/nemotron/nemotron-3-super/` in
[NVIDIA-NeMo/Evaluator](https://github.com/NVIDIA-NeMo/Evaluator)).

| Component | Description |
|-----------|-------------|
| `config/default.yaml` | vLLM deployment + NeMo Evaluator task list |

> The tech report's numbers (Table 10) were collected with
> [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym), not NeMo Evaluator. To reproduce them, use
> NeMo Gym's [Super 3.5 reproducibility guide](https://github.com/NVIDIA-NeMo/Gym/blob/jakub/marlin/nemotron_recipes/super-3.5/reproducibility.md)
> (one recipe per benchmark; on the `jakub/marlin` branch until merged to `main`), summarized in
> [docs/nemotron/super35/evaluate.md](../../../../../docs/nemotron/super35/evaluate.md#reproducing-the-published-results-nemo-gym).
> Scores from this stage are therefore not expected to match Table 10 exactly.

## Benchmarks

Coverage of the tech report's Table 10 by NeMo Evaluator tasks:

| Table 10 benchmark | NeMo Evaluator task | Default | Needs |
|--------------------|---------------------|---------|-------|
| GPQA-Diamond | `nemo_skills.ns_gpqa` | yes | — |
| SciCode | `ns_scicode` | yes | — |
| LiveCodeBench v6 | `ns_livecodebench` (`test_v6_2408_2505`) | yes | — |
| IFBench | `ns_ifbench` | yes | — |
| MMLU-ProX | `ns_mmlu_prox` | yes | — |
| HLE (text) | `ns_hle` | no | judge model (`JUDGE_API_KEY`) |
| AA-LCR | `ns_aa_lcr` | no | judge model |
| AA-Omniscience | `ns_omniscience` | no | judge model |
| CritPt | `nemo_skills.ns_critpt` | no | Artificial Analysis API key |
| Tau2 Retail / Airline / Telecom | `tau2_bench_{retail,airline,telecom}` | no | user-simulator model |
| WMT24pp COMET | `ns_wmt24pp_comet` | no | local XCOMET-XXL checkpoint |

The judge-dependent tasks are listed (commented) in `config/default.yaml`; the Evaluator's
Nemotron 3 Super example shows the full judge / user-simulator settings.

Not covered here:

- **Agentic** — SWE-bench Verified / Multilingual / Pro (OpenCode harness), Terminal-Bench 2.1,
  Tau3 Banking, GDPVal-AA, BrowseComp: run with NeMo Gym.
- **Other** — Apex-Shortlist, GraphWalks, OSWorld.
- **Multimodal** — BabyVision, CharXiv-RQ, MathVision, MMLongBench-Doc, OCR-Reasoning,
  OCRBench-v2, V*, Video-MME-v2: the report lists these as not yet onboarded in open-source tools.
- **NV Chat Benchmark** — internal.

## Quick Start

```bash
# Evaluate an exported Hugging Face checkpoint (see "Checkpoint format" below)
uv run nemotron super35 eval --run YOUR-CLUSTER run.model=super35-rl-mopd-phase2-hf-model:latest

# Select tasks
uv run nemotron super35 eval --run YOUR-CLUSTER -t nemo_skills.ns_gpqa -t ns_ifbench

# Quick test with limited samples
uv run nemotron super35 eval --run YOUR-CLUSTER \
    evaluation.nemo_evaluator_config.config.params.limit_samples=10

# Preview without executing
uv run nemotron super35 eval --dry-run
```

## Configuration

| Section | Notes |
|---------|-------|
| `run.model` | Default `super35-rl-mopd-phase2-model:latest`; override with an HF-format artifact (see below) |
| `run.env.container` | `vllm/vllm-openai:v0.30.0`, the vLLM release the published NeMo Gym runs used |
| `deployment` | `type: vllm`, TP8 on one node, `--trust-remote-code`, Mamba SSM cache in FP32; check the model card for the recommended reasoning/tool-call parsers and MTP speculative decoding |
| `evaluation.nemo_evaluator_config` | Reasoning on (`enable_thinking: true`), temperature 1.0, top-p 0.95, up to 131,072 new tokens |
| `execution.mounts` | Add your shared filesystem so the deployment container can read the checkpoint |
| `export.wandb` | Results exported to the W&B project from `env.toml` |

## Checkpoint format

vLLM serves Hugging Face checkpoints only. Checkpoints produced by Megatron-Bridge (stages 0-1)
are Megatron format; export them with Megatron-Bridge's `convert.sh export` first.

RL stages save NeMo-RL checkpoints (`step_N/policy/weights/...`), which vLLM cannot serve.
Export the selected step to Hugging Face with NeMo-RL's converter (inside the RL container,
from `/opt/nemo-rl`), register the result, and evaluate that artifact:

```bash
uv run --extra mcore python examples/converters/convert_megatron_to_hf.py \
    --config /path/to/step_N/config.yaml \
    --hf-model-name nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 \
    --megatron-ckpt-path /path/to/step_N/policy/weights/iter_XXXXXXX \
    --hf-ckpt-path /path/to/super35-hf
uv run nemotron super35 model import rl /path/to/super35-hf --name super35-rl-mopd-phase2-hf-model
uv run nemotron super35 eval --run YOUR-CLUSTER run.model=super35-rl-mopd-phase2-hf-model:latest
```

## Previous Stages

- [Stage 0: Pretraining](../stage0_pretrain/README.md)
- [Stage 1: SFT](../stage1_sft/README.md)
- [Stage 2: RL](../stage2_rl/README.md)
