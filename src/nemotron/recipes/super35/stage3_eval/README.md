# Stage 3: Evaluation

Evaluate Nemotron 3.5 Super VL checkpoints with NeMo Evaluator.

## Overview

This stage evaluates the final RL checkpoint (MOPD phase 2, Hugging Face format) by serving it
with vLLM and running NeMo Evaluator tasks through `nemo-evaluator-launcher`. Task names and
per-task settings follow NeMo Evaluator's Nemotron 3 Super example
(`packages/nemo-evaluator-launcher/examples/nemotron/nemotron-3-super/` in
[NVIDIA-NeMo/Evaluator](https://github.com/NVIDIA-NeMo/Evaluator)).

| Component | Description |
|-----------|-------------|
| `config/default.yaml` | vLLM deployment + NeMo Evaluator task list |

> The tech report's numbers (Table 10) were collected with
> [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym), not NeMo Evaluator, and the report points to a
> Super 3.5 reproducibility guide in the Gym repository
> (`nemotron_recipes/super-3.5/reproducibility.md`).
> TODO(open-source): that guide is not published yet; link it here once it is.
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
# Evaluate the final RL checkpoint
uv run nemotron super35 eval --run YOUR-CLUSTER

# Evaluate another stage (must be a Hugging Face checkpoint)
uv run nemotron super35 eval --run YOUR-CLUSTER run.model=super35-rl-rlvr-model:latest

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
| `run.model` | Default `super35-rl-mopd-phase2-model:latest` (final checkpoint) |
| `run.env.container` | vLLM image; TODO(open-source): pin the first public vLLM release that serves Super 3.5 VL |
| `deployment` | `type: vllm`, TP8 on one node, `--trust-remote-code`, Mamba SSM cache in FP32; check the model card for the recommended reasoning/tool-call parsers and MTP speculative decoding |
| `evaluation.nemo_evaluator_config` | Reasoning on (`enable_thinking: true`), temperature 1.0, top-p 0.95, up to 131,072 new tokens |
| `execution.mounts` | Add your shared filesystem so the deployment container can read the checkpoint |
| `export.wandb` | Results exported to the W&B project from `env.toml` |

Checkpoints produced by Megatron-Bridge (stages 0-1) are Megatron format; export them to
Hugging Face with Megatron-Bridge before evaluating them here.

## Previous Stages

- [Stage 0: Pretraining](../stage0_pretrain/README.md)
- [Stage 1: SFT](../stage1_sft/README.md)
- [Stage 2: RL](../stage2_rl/README.md)
