# Specialist Teachers (Stage 2.2)

Each Nemotron 3.5 Super teacher continues RL from a shared checkpoint on a narrow, domain-specific mix, so it can push one capability further than the generalist student. [MOPD](mopd.md) then distills the teachers back into a single model. All teachers use asynchronous GRPO in NeMo-RL. Each section below separates what the report states from what the config sets; where the report is silent, values come from the reference config.

---

## Overview

| Teacher | CLI | Starts from | Nodes (`run.env.nodes`) | Selected step | Used in |
|---------|-----|-------------|------------------------:|---------------|---------|
| [Agentic](#agentic-teacher) | `rl teacher agentic` | RLVR step 120 | 128 | 14 | MOPD phase 1 |
| [Reasoning](#reasoning-teacher) | `rl teacher reasoning` | RLVR step 120 | 64 | 80 | MOPD phase 1 |
| [Vision tool use](#vision-tool-use-teacher) | `rl teacher vision` | RLVR step 120 | 16 | 40 | MOPD phases 1 and 2 |
| [Video reasoning and tracking](#video-teacher) | `rl teacher video` | RLVR step 120 | 32 | 60 | MOPD phase 2 |
| [Vision reasoning](#vision-reasoning-teacher) | `rl teacher vision-reasoning [-c stage2]` | RLVR step 120 → stage 1 | 32 / 33 | 114 → 20 | MOPD phase 2 |
| [RLHF teacher 1](#alignment-rlhf-teachers) | `rl teacher rlhf [-c teacher1_stage2]` | SFT | 32 / 32 | 120 → 500 | MOPD phase 1 |
| [RLHF teacher 2](#alignment-rlhf-teachers) | `rl teacher rlhf -c teacher2_stage1 \| teacher2_stage2` | RLVR step 120 | 64 / 32 | 200 → 120 | MOPD phase 2 |

Nodes are GB200 nodes with 4 GPUs. `run.env.nodes` covers training and generation; external GenRM and judge servers are extra (see each teacher). Every teacher writes `super35-rl-teacher-<name>-model`; the two-stage teachers also write an intermediate artifact.

## Data Artifacts

Each config reads its own data artifact (`run.data`); register your NeMo Gym JSONL under exactly that name:

| Config | Data: import as |
|--------|-----------------|
| `rl teacher reasoning` | `super35-rl-teacher-reasoning-data` |
| `rl teacher vision` | `super35-rl-teacher-vision-data` |
| `rl teacher video` | `super35-rl-teacher-video-data` |
| `rl teacher vision-reasoning -c stage1` | `super35-rl-teacher-vision-reasoning-stage1-data` |
| `rl teacher vision-reasoning -c stage2` | `super35-rl-teacher-vision-reasoning-data` |
| `rl teacher rlhf -c teacher1_stage1` / `teacher1_stage2` | `super35-rl-teacher-rlhf1-data` |
| `rl teacher rlhf -c teacher2_stage1` | `super35-rl-teacher-rlhf2-stage1-data` |
| `rl teacher rlhf -c teacher2_stage2` | `super35-rl-teacher-rlhf2-data` |
| `rl teacher agentic` (blocked) | `super35-rl-teacher-agentic-data` |

```bash
uv run nemotron super35 data import rl /data/reasoning/train.jsonl --val /data/reasoning/val.jsonl --name super35-rl-teacher-reasoning-data
uv run nemotron super35 data import rl /data/vision/train.jsonl --val /data/vision/val.jsonl --name super35-rl-teacher-vision-data
uv run nemotron super35 data import rl /data/video/train.jsonl --val /data/video/val.jsonl --name super35-rl-teacher-video-data
uv run nemotron super35 data import rl /data/vr1/train.jsonl --val /data/vr1/val.jsonl --name super35-rl-teacher-vision-reasoning-stage1-data
uv run nemotron super35 data import rl /data/vr2/train.jsonl --val /data/vr2/val.jsonl --name super35-rl-teacher-vision-reasoning-data
uv run nemotron super35 data import rl /data/rlhf1/train.jsonl --val /data/rlhf1/val.jsonl --name super35-rl-teacher-rlhf1-data
uv run nemotron super35 data import rl /data/rlhf2-stage1/train.jsonl --val /data/rlhf2-stage1/val.jsonl --name super35-rl-teacher-rlhf2-stage1-data
uv run nemotron super35 data import rl /data/rlhf2-stage2/train.jsonl --val /data/rlhf2-stage2/val.jsonl --name super35-rl-teacher-rlhf2-data
uv run nemotron super35 data import rl /data/agentic/train.jsonl --val /data/agentic/val.jsonl --name super35-rl-teacher-agentic-data
```

Image and video paths inside the rows must resolve inside the RL container; mount the media store through your `env.toml` profile.

---

## Agentic Teacher

> **Blocked**: the Terminal-Bench half of this teacher needs an agent environment that is not in public NeMo Gym, and no validated replacement exists. `agentic/config/default.yaml` documents the setup for reference; the recipe does not silently substitute a different Terminal-Bench implementation.

The report (§3.3.1) trains this teacher with full end-to-end GRPO on sandboxed, fully executable Terminal, SWE, and Search tasks, with no single-step environments, profiled for difficulty against the RLVR step-120 checkpoint.

| Parameter | Report | Config |
|-----------|--------|--------|
| Prompts × generations | 64 × 16 (batch 1,024) | 64 × 16 |
| Max trajectory age | 1 | 1 |
| Rollout timeout | 2 hours per rollout | — |
| Context | — | 192K |
| Learning rate | — | 3e-6 |
| Nodes | — | 128 (32 training, TP8 / CP8 / EP8; 96 generation) |
| Selected checkpoint | — | Step 14 (the config is recovered from the neighboring step 15) |

The config mixes Terminal-Bench, SWE-bench, and BrowseComp prompts (the report's Terminal, SWE, and Search environments). Differences from the reference run: that run started from a text-only RLVR export whose identity with the selected step-120 weights is unconfirmed (the dashed edge in the [lineage diagram](index.md#why-three-sub-stages)), and it trained with MTP loss off. Artifact: `super35-rl-teacher-agentic-model`.

## Reasoning Teacher

The report (§3.3.2) replaces the broad RLVR mixture with a reasoning-focused curriculum and a longer generation budget.

| Parameter | Value |
|-----------|-------|
| Data | 25,600 examples; each 256-prompt batch is 90% science, 5% math, 5% competitive coding |
| Prompts × generations | 256 × 16 (batch 4,096) |
| Max generation length | 131,072 tokens (262,144 total context in the config) |
| Max trajectory age | 2 (vs. 1 for RLVR) |
| Learning rate | 4e-6 |
| Rewards | Answer verification with [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) as LLM judge (math, STEM); execution-based tests (coding) |
| Selected checkpoint | Step 80 (the config stops there) |
| Config layout | 64 nodes: 16 training (TP4 / CP16 / EP32) + 48 generation |

Differences from the reference run:

- DeepSeek-V4-Flash is the judge throughout (the reference run switched judges at step 30).
- Reasoning stays on for coding prompts.
- Five MTP heads at coefficient 0.3 (the reference run: one head, coefficient 1); generation keeps one MTP speculative token.

Point `DSV4_JUDGE_BASE_URL` (and `DSV4_JUDGE_MODEL`, default `deepseek-ai/DeepSeek-V4-Flash`) at your endpoint and export `NVIDIA_API_KEY`; a hosted judge needs no cluster nodes. Coding rewards need a NeMo-Skills sandbox (`run.env.sandbox.container`).

```bash
uv run nemotron super35 rl teacher reasoning --run YOUR-CLUSTER run.env.sandbox.container=<sandbox-image>
```

Artifact: `super35-rl-teacher-reasoning-model`.

## Vision Tool-Use Teacher

The report (§3.4.2) trains effective tool use on complex vision tasks. Tools (crop, zoom, rotate, flip, difference detection, side-by-side comparison, overlays, object counting, color analysis) are called as structured XML. The training set is synthetic multiple-choice and numerical questions that need one or more tools, filtered to samples that are nearly impossible to solve without them. Rewards are outcome correctness plus an auxiliary tool-use reward that penalizes duplicate and malformed calls. Checkpoints are selected on V\*, HRBench8K, and BabyVision accuracy.

The report does not give batch or schedule settings. The config's values (from the reference config):

| Parameter | Config |
|-----------|--------|
| Prompts × generations | 16 × 8 |
| Context | 32K; image agent limited to 8 steps / 6 tool calls |
| Learning rate | 3e-6 |
| Trainable | Language model, vision encoder, and projector |
| Nodes | 16: 8 training (TP8 / CP2 / EP16) + 8 generation |
| Selected checkpoint | Step 40 (the config runs to 200) |

The reference run trained one MTP head at coefficient 0.1. Artifact: `super35-rl-teacher-vision-model`.

```bash
uv run nemotron super35 rl teacher vision --run YOUR-CLUSTER grpo.max_num_steps=40
```

## Video Teacher

A video reasoning and object-tracking teacher (report §3.4.1).

| Parameter | Value |
|-----------|-------|
| Data | Video-QA multiple-choice questions over 64 ordered frames (processed as one video) curated from the Nemotron 3 Nano Omni video training subset, plus SA-V box, point, and referring-object tracking (frames processed as ordered multi-image inputs) |
| Rewards | Binary multiple-choice correctness (Video-QA); continuous track-quality reward covering localization, visibility accuracy, and track validity and completeness (SA-V); no LLM judge |
| Shaping | Responses without exactly one well-formed `<think>…</think>` block get zero reward and advantage −5; correct responses get a soft length preference (full reward up to 1,024 reasoning tokens, linearly down to 0.95 at 4,096) |
| Prompts × generations | 128 × 16 (batch 2,048) |
| Learning rate | 3e-6, constant |
| Frozen | MoE router and vision modules; router replay on |
| Hardware | 128 GB200 GPUs (config: 32 nodes, 16 training at TP2 / EP16 + 16 generation) |
| Selected checkpoint | Step 60: Video-MME 74.90 (256-frame reasoning), Video-MME-v2 19.10, GOT-10k 52.59, Molmo2-VideoTrack 60.00 |

Frozen vision weights are never refitted into vLLM, so keep `policy.generation.vllm_cfg.load_format=auto`. Preserve native-video metadata for Video-QA samples and ordered images for SA-V samples when preparing data. The config has no step limit; stop it at the selected step. The reference run trained with MTP loss off.

```bash
uv run nemotron super35 rl teacher video --run YOUR-CLUSTER grpo.max_num_steps=60
```

Artifact: `super35-rl-teacher-video-model`.

## Vision-Reasoning Teacher

Improves reasoning over charts, documents, tables, OCR, geometry, and visual puzzles, with outcome rewards from math, multiple-choice, and string-matching verifiers (report §3.4.3).

> **Stage 2 is blocked**: its structured-answer verifier is not yet in public NeMo Gym.

It runs in two stages:

| | Stage 1 (`-c stage1`, default) | Stage 2 (`-c stage2`) |
|---|---|---|
| Data (report) | 16,090 profiled examples | 13,848 examples: 8,366 chart-reasoning + 5,482 document/table/OCR; exact duplicates removed |
| Starts from | RLVR step 120, as in the reference run (the report: "an existing multimodal RLVR checkpoint") | Stage 1 step 114, **policy weights only** (fresh optimizer) |
| Prompts × generations | 256 × 16 | 256 × 16 |
| Context, LR (config) | 32K, 3e-6 | 32K, 3e-6 |
| Training parallelism (config) | TP2 / CP1 / EP16, no router replay | TP2 / CP2 / EP16, router replay on |
| Nodes | 32 (report: 128 GPUs on 32 nodes) | 33 (the reference run added a dedicated Ray head) |
| Selected checkpoint | Step 114 (config runs to 125) | Step 20 (config runs to 54) |

Both stages freeze the vision encoder and projector. Stage 2 keeps its structured-answer verifier for set, ordered-list, and interval answers. The reference runs trained five MTP heads at coefficient 0.1 (stage 1) and no MTP loss (stage 2); the stage-1 config is ported from the reference run's saved step-110 config.

```bash
uv run nemotron super35 rl teacher vision-reasoning --run YOUR-CLUSTER            # stage 1
uv run nemotron super35 rl teacher vision-reasoning -c stage2 --run YOUR-CLUSTER  # stage 2
```

Artifacts: `super35-rl-teacher-vision-reasoning-stage1-model` (stage 1), `super35-rl-teacher-vision-reasoning-model` (stage 2, used by MOPD).

## Alignment (RLHF) Teachers

The report (§3.5) trains an alignment teacher primarily with RLHF against a GenRM: a new variant of the Nemotron 3 Ultra GenRM trained with a reasoning-length penalty, which together with MTP inference makes training substantially faster. Instead of cyclic self-comparison, every policy response is compared with a fixed response from Gemma 4 31B-it, scored by the mean of per-rubric scores. The blend adds new preference data, updated model identity and capability data, and more emphasis on language mixing, and is mixed with Nemotron 3 Ultra's abstention training and a small reasoning-length penalty. Policy improvements are evaluated on NVIDIA's internal chat benchmark.

The report describes one alignment teacher, started from the RL trunk (Figure 4; NeMo-RL's guide does the same). The reference run, and this recipe, train two, one per MOPD phase, each in two stages; RLHF teacher 1 starts from SFT. The stage structure, blends, and penalties below come from the reference config (Blends 1–3 are its three alignment blends; `src/nemotron/recipes/super35/stage2_rl/stage2_teachers/rlhf/README.md` lists their sizes):

| Config | Initialization | Data | Reasoning-length penalty | Selected step | Artifact |
|--------|----------------|------|--------------------------|---------------|----------|
| `teacher1_stage1` (default) | SFT checkpoint | Blend 1 | 0.12 | 120 | `super35-rl-teacher-rlhf1-stage1-model` |
| `teacher1_stage2` | **Full-state resume** of `teacher1_stage1` step 120 | Blend 1 | 0.06 | 500 (500 RLHF steps in total) | `super35-rl-teacher-rlhf1-model` |
| `teacher2_stage1` | RLVR step 120 | Blend 2 | 0.06 | 200 | `super35-rl-teacher-rlhf2-stage1-model` |
| `teacher2_stage2` | `teacher2_stage1` step 200, **weights only** (fresh optimizer, restarted step counter) | Blend 3 (Blend 2 + abstention prompts), abstention reward 0.2 | 0.06 | 120 (320 RLHF steps in total) | `super35-rl-teacher-rlhf2-model` |

All stages use 32 prompts × 32 generations, a 48K context with up to 16K generated tokens, LR 2e-6, text-only data, and frozen vision towers. Each config stops at its selected step.

| Config | Nodes (`run.env.nodes`) | External servers in the reference run |
|--------|------------------------:|---------------------------------------|
| `teacher1_stage1`, `teacher1_stage2` | 32 (16 training + 16 generation) | GenRM, 16 TP8 replicas (32 nodes) |
| `teacher2_stage1` | 64 (32 + 32) | GenRM, 32 TP8 replicas (64 nodes) |
| `teacher2_stage2` | 32 (16 + 16) | GenRM (32 nodes) + abstention judge, Qwen3-235B-A22B-Instruct-2507-FP8, 4 TP4 replicas (4 nodes) |

- **`teacher1_stage2` must resume `teacher1_stage1`'s full training state** (optimizer, scheduler, step counter, dataloader position), not just its weights. Both stages share one checkpoint directory.
- **`teacher2_stage2` is a fresh run** warm-started from `teacher2_stage1`'s exported weights.
- Set the GenRM and abstention judge endpoints with `GENRM_BASE_URL`, `GENRM_MODEL`, `ABSTENTION_JUDGE_BASE_URL`, and `ABSTENTION_JUDGE_MODEL` (see [External Services](index.md#external-services)).
- Differences from the reference run: training uses TP8 / CP8 / EP16 (the reference launch scripts used TP4 / CP2 / EP16), the reference teachers loaded a text-only architecture rather than the full vision-language checkpoint with frozen towers, and MTP training follows the recipe-wide setting.

```bash
uv run nemotron super35 rl teacher rlhf --run YOUR-CLUSTER
uv run nemotron super35 rl teacher rlhf -c teacher1_stage2 --run YOUR-CLUSTER
uv run nemotron super35 rl teacher rlhf -c teacher2_stage1 --run YOUR-CLUSTER
uv run nemotron super35 rl teacher rlhf -c teacher2_stage2 --run YOUR-CLUSTER
```

## Next Steps

Distill the teachers with [MOPD](mopd.md).

## Reference

- Nemotron 3.5 tech report, §3.3 (agentic and reasoning teachers), §3.4 (vision teachers), §3.5 (alignment teacher)
- [RL overview](index.md) · [MOPD](mopd.md)
- **Recipe Source**: `src/nemotron/recipes/super35/stage2_rl/stage2_teachers/` (the RLHF details are in `rlhf/README.md`)
- [Back to Overview](../README.md)
