# MOPD (Stage 2.3)

Multi-teacher On-Policy Distillation (MOPD) merges the Nemotron 3.5 Super specialist teachers into a single student. The student generates a response, and the teacher that owns the prompt's domain scores every token: tokens the teacher rates higher than the student are reinforced, and tokens it rates lower are suppressed. MOPD runs in two phases; phase 2's step-30 checkpoint, after MTP boosting, is the released model.

> **Tip**: MOPD needs a checkpoint for every teacher it distills, and the teacher checkpoints are not published. Phase 1 needs the agentic teacher and phase 2 the vision-reasoning teacher, and neither can be trained from public components yet (see [Teachers](teachers.md#agentic-teacher)). Until they can, run the [single-stage variant](#single-stage-variant), which needs no agentic teacher.

---

## Method

The student minimizes the reverse KL to the domain teacher at every position of its own responses, using a token-local update with a dense per-token advantage:

$$
\hat{A}_t = \operatorname{sg}\left[\log \pi_T(y_t \mid c_t) - \log \bar{\pi}_S(y_t \mid c_t)\right]
$$

where $\pi_T$ is the domain teacher, $\bar{\pi}_S$ the frozen pre-update snapshot of the student, and $\operatorname{sg}$ stop-gradient. Each generated token gets its advantage from a single teacher forward pass — no verifier and no sampling from the teacher — and the signal is zero wherever student and teacher already agree.

- **Asynchronous, one step stale**: following the Nemotron 3 Ultra recipe, tokens are importance-weighted by the ratio of the current and previous student snapshots, and tokens with weights outside [0.2, 5] are masked. Only student-generated tokens enter the loss.
- **Routing**: prompts come from the environments used to train each teacher and are routed to that teacher. Environments no specialist owns go to a **self-anchor** (`general` in the configs) — a frozen copy of the student's starting checkpoint — which gives no signal while the student still matches it and regularizes departures from it.
- **Teacher fleets**: teachers run as independent, non-colocated inference fleets (TP8, EP8) sized to each domain's share and sequence length. Rollout generation, teacher scoring, and student optimization form a pipeline.
- **MTP**: the MTP heads are trained jointly with the language model throughout. (The configs use the recipe-wide [MTP setting](index.md#shared-settings).)

### Why Two Phases

Benchmarks improve at different rates under distillation, and some peak within a few steps and then decline. Phase 1 is weighted toward domains that improve quickly and tend to regress later (agentic, reasoning); phase 2 targets domains that improve slowly (chat quality and vision).

## Phases

| | Phase 1 (`rl mopd`) | Phase 2 (`rl mopd -c phase2`) |
|---|---|---|
| Student | RLVR step 120 | MOPD phase 1 step 30 |
| Self-anchor | RLVR step 120 | MOPD phase 1 step 30 |
| Teachers and prompt shares | Agentic 35%, reasoning 25%, alignment (RLHF teacher 1) 20%, vision tool use 15%, self-anchor 5% | Alignment (RLHF teacher 2) 30%, vision tool use 20%, vision reasoning 20%, video 20%, self-anchor 10% |
| Prompts × rollouts per step | 512 × 1 | 1,024 × 1 |
| Generation | Up to 48K tokens in a 128K context | Full 128K context |
| Learning rate | 2e-6 | 3e-6 |
| Teacher target | Teacher distribution (α = 1) | Proximal teacher, α = 0.2 |
| Run length | Report: checkpoint after 30 steps. Config: 200-step budget | Report: 40 steps, one pass over the prompts. Config: stops at 40; the reference run selected step 30 |
| Nodes (GB200, 4 GPUs) | 128 (report and config) | 128 (report and config) |
| Config node split: training / generation / teachers | 64 / 30 / 34 | 64 / 44 / 20 |
| Teacher fleets (config, nodes) | Agentic 16, reasoning 8, alignment 4, vision 4, self-anchor 2 | Alignment 4, video 4, vision 4, vision reasoning 4, self-anchor 4 |
| Output artifact | `super35-rl-mopd-phase1-model` | `super35-rl-mopd-phase2-model` |

In phase 2 the agentic and reasoning teachers are retired and their environments folded into the self-anchor stream, so the student is regularized toward the phase-1 model on them instead of being distilled again.

### Proximal teacher (phase 2)

Phase 2 smooths the target with the proximal-teacher construction from trust-region policy distillation: the teacher in the advantage above is replaced by the frozen mixture $\alpha\,\pi_T + (1-\alpha)\,\bar{\pi}_S$ with $\alpha = 0.2$ (`proximal_teacher_alpha: 0.2`). This bounds the advantage below by $\log(1-\alpha) \approx -0.22$, so tokens the teacher considers very unlikely get a bounded rather than unbounded negative signal — which stabilizes training once the student has already absorbed one round of teachers.

## Configuration

Teacher checkpoints are artifacts in the `run:` block, one `run.teacher_<role>` entry per teacher. The self-anchor teacher always resolves to the student's starting checkpoint (`run.model`). The config maps every NeMo Gym agent to a teacher alias; aliases with no explicit teacher fall back to the self-anchor.

### Differences from the Reference Run

- **Phase 1**: student TP4 / CP4 / EP16 with five-head MTP training. The reference run used TP8 / CP8 / EP16 with MTP loss off, unlike the report's joint MTP training. Reward verification and replay-buffer restoration are disabled.
- **Phase 2**: student TP8 / CP8 / EP16 with five-head MTP training, as in the reference run. That run's prompt blend used the report's shares with upsampled model-identity prompts, laid out as 44 ordered steps of 1,024 prompts. Keep `data.shuffle=false` and 1,024 prompts per step, and start a fresh experiment when you change datasets. The config saves every two steps and stops at step 40 (`grpo.max_num_steps=<limit>` changes that).
- **Phase 2's self-anchor** is the phase-1 step-30 checkpoint, the same checkpoint as the student. (The reference run used a different phase-1 checkpoint as its self-anchor; the recipe uses one checkpoint for both roles.)
- Policy MTP inference is off in both phases.

## Usage

```bash
# Phase 1: student = RLVR, teachers = agentic, reasoning, RLHF teacher 1, vision
uv run nemotron super35 rl mopd --run YOUR-CLUSTER

# Phase 2: student = phase 1, teachers = RLHF teacher 2, vision, vision reasoning, video.
# Phase 1 runs past step 30, so pin the artifact version saved at step 30.
uv run nemotron super35 rl mopd -c phase2 --run YOUR-CLUSTER \
  run.model=super35-rl-mopd-phase1-model:vN

# Swap a teacher for a different artifact version
uv run nemotron super35 rl mopd -c phase2 --run YOUR-CLUSTER \
  run.teacher_video=super35-rl-teacher-video-model:v1

# Single-stage variant (NeMo-RL's mopd.yaml): phase 1 without the agentic teacher
uv run nemotron super35 rl mopd -c single_stage --run YOUR-CLUSTER

# Preview the resolved config, including all teacher paths and routing
uv run nemotron super35 rl mopd -c phase2 --dry-run
```

The teacher role names (`run.teacher_<role>`) are listed at the top of each config.

### Single-Stage Variant

`-c single_stage` ports NeMo-RL's [`mopd.yaml`](https://github.com/NVIDIA-NeMo/RL/blob/super-v3.5-posttraining/examples/nemo_gym/nemotron-3.5-super/mopd.yaml): phase 1 without the agentic teacher. Every teacher it uses can be trained from public code with your own data, so it is the MOPD to run while the agentic teacher is blocked. It is not the report's pipeline.

- **Routing**: agent names that phase 1 sends to the agentic teacher go to the self-anchor (the RLVR student). The other teachers and all training settings are phase 1's.
- **Allocation**: 112 nodes instead of 128.
- **RLHF teacher**: this repository's RLHF teacher 1 (`rlhf1`, from SFT). NeMo-RL's `rlhf_teacher.yaml` instead starts from Student RLVR, adds abstention, and uses a 0.12 length penalty; no config here reproduces it.
- **Data and output**: reads `super35-rl-mopd-phase1-data`, writes `super35-rl-mopd-single-stage-model`, and has a 200-step budget with no selected step. To evaluate it, export your chosen step to Hugging Face and pass it to `super35 eval` (see [Evaluation](../evaluate.md#exporting-checkpoints)).

## Next Steps

[Evaluate](../evaluate.md) the phase-2 step-30 checkpoint. The report then applies [MTP boosting](index.md#mtp-boosting) (backbone frozen, MTP head trained with knowledge distillation from the backbone), which this recipe does not include, and [quantizes](../quantization.md) the result to NVFP4.

## Reference

- Nemotron 3.5 tech report, §3.6 (MOPD), §3.7 (MTP Boosting)
- [Nemotron 3 Ultra MOPD](../../ultra3/mopd.md) — the MOPD recipe Super 3.5 builds on
- [Teachers](teachers.md) · [RL overview](index.md)
- **Recipe Source**: `src/nemotron/recipes/super35/stage2_rl/stage3_mopd/`
- [Back to Overview](../README.md)
