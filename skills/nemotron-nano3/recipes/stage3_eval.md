# Stage 3 Recipe Bridge: Evaluation

This file connects the paper’s evaluation sections to `src/nemotron/recipes/nano3/stage3_eval/`.

## What Exists Publicly

Public files:

- `src/nemotron/recipes/nano3/stage3_eval/config/default.yaml`
- `src/nemotron/recipes/nano3/stage3_eval/README.md`
- `docs/nemotron/nano3/evaluate.md`

Unlike the training stages, stage3 is mainly a **compiled evaluator config path**.
There is no stage-local `train.py`.
The CLI merges config, resolves artifacts, strips the `run` section, and calls NeMo Evaluator.

## What It Maps To In The Paper

| Paper section | Public stage3 element |
|---|---|
| §2.6 base model evaluations | NeMo Evaluator style benchmark execution |
| §3.4 post-trained evaluations | evaluator deployment + tasks |
| §4.3 quantization accuracy | paper result only; public stage can evaluate released checkpoints |
| Appendix E prompt sensitivity | paper methodology only; not a default stage3 workflow |

## Public Defaults

From `config/default.yaml`:

| Setting | Value |
|---|---|
| default model artifact | `nano3-rl-model:latest` |
| serving container | `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano` |
| deployment mode | generic NeMo Framework Ray |
| GPUs | 8 |
| tensor parallel | 2 |
| expert parallel | 8 |
| port | 1235 |
| default tasks | MMLU, ARC Challenge 25-shot, Winogrande 5-shot, HellaSwag, OpenBookQA |
| evaluator parallelism | 4 |

## Public Evaluation Flow

The CLI flow is:

1. load YAML
2. merge env.toml profile + CLI overrides
3. resolve W&B artifact references
4. compile an evaluator-facing config
5. deploy the model
6. run benchmark tasks
7. export results to W&B

That makes stage3 more of an **evaluation launcher surface** than a custom model-training recipe.

## Why Stage3 Is Narrower Than The Paper

The paper reports a large evaluation program including:

- base-model comparisons
- post-trained model comparisons
- long-context evaluations
- tool/no-tool splits
- multilingual evaluations
- quantization comparisons
- prompt sensitivity studies

The public default stage3 config is intentionally narrower.
It gives a practical benchmark starter pack rather than the full internal evaluation matrix.

## What Stage3 Is Best At

Use stage3 when the goal is:

- evaluate a checkpoint produced by public stage1 or stage2
- resolve model artifacts from W&B lineage
- benchmark with the NeMo Evaluator ecosystem
- keep evaluation inside the same public artifact pipeline

## Reproduce with nemotron-customize

This stage maps directly to:

- `eval/model_eval`

Common adjacent steps:

- `convert/megatron_to_hf` if a different deployment backend prefers HF checkpoints
- `convert/hf_to_megatron` when the evaluation path needs a Megatron-native checkpoint

## Good Handoff Pattern

> “For public Nano3 evaluation, use `eval/model_eval`. It reproduces the evaluator-style deployment and benchmark flow, but the default task set is much smaller than the paper’s full reported evaluation matrix.”
