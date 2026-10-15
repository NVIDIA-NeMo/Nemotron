# MOPD (Stage 2.3)

Multi-teacher On-Policy Distillation (MOPD) merges the specialist teachers into a single student. The student generates a response; for every token, the teacher that owns the prompt's domain scores it, and the student is pushed toward tokens the teacher rates higher than it does and away from tokens it rates lower. MOPD runs in two phases. In the reference run, phase 2's step 30 is the post-trained backbone of the released model (the report then applies MTP boosting).

:::{admonition} Pending open-source release
:class: warning

MOPD needs the MOPD prompt data and every teacher checkpoint, which are not yet published (the NeMo-RL `super-v3.5-posttraining` @ `f299847f1` stack is public). Phase 1 also needs the **agentic teacher**, which cannot be trained from public components yet (see [Teachers](teachers.md#agentic-teacher)). The [single-stage variant](#single-stage-variant) (`-c single_stage`, ported from NeMo-RL's `mopd.yaml`) needs no agentic teacher.
:::

---

## Method

The student minimizes the reverse KL to the domain teacher at every position of its own responses, using a token-local update with a dense per-token advantage:

$$
\hat{A}_t = \operatorname{sg}\left[\log \pi_T(y_t \mid c_t) - \log \bar{\pi}_S(y_t \mid c_t)\right]
$$

where $\pi_T$ is the domain teacher, $\bar{\pi}_S$ the frozen pre-update snapshot of the student, and $\operatorname{sg}$ stop-gradient. Each generated token gets its advantage from a single teacher forward pass — no verifier and no sampling from the teacher — and the signal is zero wherever student and teacher already agree.

- **Asynchronous, one step stale**: following the Nemotron 3 Ultra recipe, tokens are importance-weighted by the ratio of the current and previous student snapshots, and tokens with weights outside [0.2, 5] are masked. Only student-generated tokens enter the loss.
- **Routing**: prompts come from the environments used to train each teacher and are routed to that teacher. Environments no specialist owns go to a **self-anchor** — a frozen copy of the student's starting checkpoint — which gives no signal while the student still matches it and regularizes departures from it.
- **Teacher fleets**: teachers run as independent, non-colocated inference fleets (TP8, EP8) sized to each domain's share and sequence length. Rollout generation, teacher scoring, and student optimization form a pipeline.
- **MTP**: the MTP heads are trained jointly with the language model throughout (five repeated detached heads, coefficient 0.3).

### Why two phases

Benchmarks improve at different rates under distillation, and some peak within a few steps and then decline. Phase 1 is weighted toward domains that improve quickly and tend to regress later (agentic, reasoning); phase 2 targets domains that improve slowly (chat quality and vision).

## Phases

| | Phase 1 (`rl mopd`) | Phase 2 (`rl mopd -c phase2`) |
|---|---|---|
| Student | RLVR step 120 | MOPD phase 1 step 30 |
| Self-anchor (general teacher) | RLVR step 120 | MOPD phase 1 step 30 |
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

Recipe notes:

- **Phase 1** uses TP4 / CP4 / EP16 for the student and the common five-head MTP training. The run that produced the selected checkpoint used TP8 / CP8 / EP16 with MTP loss disabled. Reward verification and replay-buffer restoration are disabled.
- **Phase 2** uses student TP8 / CP8 / EP16 and five-head MTP training, as in the run that produced the selected checkpoint. That run's prompt blend used the report's shares (alignment 30%, vision 20%, vision reasoning 20%, video 20%, self-anchor 10%) with upsampled model-identity prompts, laid out as 44 ordered steps of 1,024 prompts. Keep `data.shuffle=false` and 1,024 prompts per step, and start a fresh experiment when changing datasets. The config saves every two steps and stops at step 40; append `grpo.max_num_steps=<limit>` to change that.
- **Phase 2's self-anchor** is the phase-1 step-30 checkpoint, the same checkpoint as the student. (The historical run used a different phase-1 checkpoint as its general teacher; the recipe uses one checkpoint for both roles.)
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

### Single-stage variant

`-c single_stage` ports NeMo-RL's [`mopd.yaml`](https://github.com/NVIDIA-NeMo/RL/blob/super-v3.5-posttraining/examples/nemo_gym/nemotron-3.5-super/mopd.yaml): phase 1 with the agentic teacher removed. The agent names phase 1 routes to the agentic teacher go to the self-anchor (the RLVR student) instead, and the allocation drops from 128 to 112 nodes; the other teachers and all training settings are phase 1's. Every teacher it uses can be trained from public code, so it is the MOPD to run while the agentic teacher is blocked. It is not the report's pipeline. It reads `super35-rl-mopd-phase1-data`, writes `super35-rl-mopd-single-stage-model`, and has a 200-step budget with no selected step. To evaluate it, pass `run.model=super35-rl-mopd-single-stage-model:<version>` to `super35 eval`.

## After MOPD

The report applies **MTP boosting** to the MOPD output (backbone frozen, MTP head trained with knowledge distillation from the backbone) before release, and then quantizes the result to NVFP4. MTP boosting is not part of this recipe; see [RL overview](index.md#mtp-boosting-not-included) and [Quantization](../quantization.md).

## Reference

- Nemotron 3.5 tech report, §3.6 (MOPD), §3.7 (MTP Boosting)
- [Nemotron 3 Ultra MOPD](../../ultra3/mopd.md) — the MOPD recipe Super 3.5 builds on
- [Teachers](teachers.md) · [RL overview](index.md)
- **Recipe Source**: `src/nemotron/recipes/super35/stage2_rl/stage3_mopd/`
- [Back to Overview](../README.md)
