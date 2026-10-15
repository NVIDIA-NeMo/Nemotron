# Stage 2.3: Multi-Teacher On-Policy Distillation (MOPD)

MOPD merges the specialist teachers from [Stage 2.2](../stage2_teachers/README.md)
into one Nemotron 3.5 Super student, in two phases on 128 GB200 nodes. Phase 2
produces the final backbone, which goes to [evaluation](../../stage3_eval/README.md);
the report applies MTP boosting first, which is not included here.

## How It Works

MOPD runs the same NeMo-RL GRPO entry point as the other RL stages, with an
`on_policy_distillation` block and the `opd` advantage estimator
(tech report §3.6).

1. **On-policy rollouts.** The student generates one response per prompt
   (`num_generations_per_prompt: 1`). Prompts come from the environments that
   were used to train each teacher.
2. **Token-level reverse KL.** The teacher that *owns* the prompt's environment
   scores every student token. The advantage of token `y_t` is the
   stop-gradient log-prob gap

   ```text
   A_t = log π_T(y_t | c_t) − log π̄_S(y_t | c_t)
   ```

   This minimizes the reverse KL `D_KL(π_S ‖ π_T)` at every position of the
   student's own responses. The signal is dense and needs one teacher forward
   pass. There is no verifier and no sampling from the teacher
   (`skip_verification: true`). The advantage is zero wherever the student
   and the teacher agree.
3. **Off-policy correction.** Training is asynchronous, so a response can be
   one step stale. Tokens are importance-weighted by the ratio of the current
   to the previous student, and tokens with weights outside [0.2, 5] are
   masked (`truncated_importance_sampling_type: icepop`). Only
   student-generated tokens enter the loss.
4. **Self-anchor routing.** Prompts from environments that no specialist owns
   go to the self-anchor (teacher alias `general`). This is a frozen copy of the student's
   *starting* checkpoint (`${art:model,path}`). It gives no signal while the
   student still matches its start, and it pulls the student back when the
   student drifts. Agent names that are missing from the routing table also
   fall back to `general` (`default_teacher_alias`).
5. **Proximal teacher (phase 2).** The distillation target becomes the
   mixture `α·π_T + (1−α)·π̄_S` (`grpo.adv_estimator.proximal_teacher_alpha`).
   With α = 0.2 the negative advantage is bounded at `log(1−α) ≈ −0.22`. A token
   that the teacher considers very unlikely therefore gets a bounded penalty,
   not an unbounded one. This stabilizes training once the student has
   already absorbed one round of teachers. Phase 1 uses α = 1, which is the
   plain teacher.

The teachers run as non-colocated inference fleets (TP8 / EP8 by default).
Each fleet is sized to its domain's share of the prompts and its sequence
lengths. The fleets score trajectories as they arrive, so rollout generation,
teacher scoring and student optimization form a pipeline. The MTP heads are
trained jointly throughout.

## Phases

Some benchmarks improve quickly under distillation and then decline, while
others improve slowly. Phase 1 is therefore weighted toward the fast domains,
and phase 2 toward chat quality and vision.

| | Phase 1 (`-c phase1`, default) | Phase 2 (`-c phase2`) |
|---|---|---|
| Student | RLVR step 120 (`super35-rl-rlvr-model`) | MOPD phase-1 step 30 (`super35-rl-mopd-phase1-model`) |
| Self-anchor (`general`) | Same checkpoint as the student | Same checkpoint as the student |
| Prompt shares | Agentic 35%, Reasoning 25%, RLHF 20%, Vision tool-use 15%, self-anchor 5% | RLHF 30%, Vision tool-use 20%, Vision-Reasoning 20%, Video 20%, self-anchor 10% (identity prompts upsampled) |
| Proximal-teacher α | 1 | 0.2 (`subtract_global_baseline: true`) |
| Prompts / step | 512 × 1 rollout | 1,024 × 1 rollout, `data.shuffle: false` |
| Generation | 48K new tokens, 128K context | 128K new tokens, 128K context |
| LR | 2e-6 | 3e-6 |
| Training parallelism | TP4 / CP4 / EP16 (see [Differences from the Reference Run](#differences-from-the-reference-run)) | TP8 / CP8 / EP16 |
| Nodes: training / generation / teachers | 64 / 30 / 34 | 64 / 44 / 20 |
| Checkpointing | every 5 steps, budget 200 steps | every 2 steps, stops at step 40 |
| Selected checkpoint | **step 30** | **step 30** (the report gives 40 steps) |
| Output artifact | `super35-rl-mopd-phase1-model` | `super35-rl-mopd-phase2-model` |

In phase 2 the agentic and reasoning teachers are retired. Their environments
are routed to the self-anchor, so the student is held near phase 1 on those
domains rather than distilled again.

### Phase → Teacher Mapping

Each canonical alias in `on_policy_distillation.teacher_model_by_agent_name`
resolves to a `run:` artifact key. Every NeMo Gym agent name in the routing table
then points at one canonical alias.

| Alias | `run:` key | Artifact | Phase 1 fleet | Phase 2 fleet |
|---|---|---|---:|---:|
| `Agentic` | `teacher_agentic` | `super35-rl-teacher-agentic-model` | 16 nodes | — |
| `Reasoning` | `teacher_reasoning` | `super35-rl-teacher-reasoning-model` | 8 nodes | — |
| `RLHF` | `teacher_rlhf` | `super35-rl-teacher-rlhf1-model` (phase 1) / `super35-rl-teacher-rlhf2-model` (phase 2) | 4 nodes | 4 nodes |
| `Vision` | `teacher_vision` | `super35-rl-teacher-vision-model` | 4 nodes | 4 nodes |
| `VisionReasoning` | `teacher_vision_reasoning` | `super35-rl-teacher-vision-reasoning-model` | — | 4 nodes |
| `Video` | `teacher_video` | `super35-rl-teacher-video-model` | — | 4 nodes |
| `general` | `model` (the student start) | phase 1: `super35-rl-rlvr-model`; phase 2: `super35-rl-mopd-phase1-model` | 2 nodes | 4 nodes |

**Canonical teacher aliases must come before the task aliases.** NeMo-RL
deduplicates teachers by checkpoint path
(`deduplicate_shared_teacher_checkpoints: true`) and takes each fleet's
settings from the first alias that maps to that path. The task aliases
interpolate the canonical entries
(`${on_policy_distillation.teacher_model_by_agent_name.RLHF}`), so changing a
teacher changes all of its routes. Fleet sizes are set in
`non_colocated_teachers.teacher_overrides`, keyed by the canonical alias.

## Usage

```bash
# Phase 1: RLVR student; agentic, reasoning, RLHF-1 and vision teachers
nemotron super35 rl mopd --run <profile> \
    run.env.sandbox.container=<nemo-skills-sandbox-image> \
    run.env.persistent_cache=/path/to/cache

# Phase 2: phase-1 student; RLHF-2, vision, vision-reasoning and video teachers
nemotron super35 rl mopd -c phase2 --run <profile> \
    run.env.sandbox.container=<nemo-skills-sandbox-image>
```

Every `run:` key resolves `:latest`, and the latest artifact is the last saved
checkpoint. Neither phase stops at its selected step: phase 1 has a 200-step
budget and phase 2 stops at step 40. Before you launch phase 2, pin the
phase-1 student to step 30, for example
`run.model=super35-rl-mopd-phase1-model:v<N>` (the version saved at step 30).
Likewise, take the final model from phase 2's step-30 checkpoint, not from
`:latest`. To swap a teacher, override its key, for example
`run.teacher_rlhf=<artifact>:<version>`.

Keep `data.shuffle: false` and 1,024 prompts per step in phase 2. The blend
is laid out as ordered batches. If you change the dataset, use a new
`checkpointing.checkpoint_dir` so that the run does not resume a different
blend's dataloader state.

## Single-Stage Variant (No Agentic Teacher)

`-c single_stage` ports NeMo-RL's own [`mopd.yaml`](https://github.com/NVIDIA-NeMo/RL/blob/super-v3.5-posttraining/examples/nemo_gym/nemotron-3.5-super/mopd.yaml). It is phase 1
with the agentic teacher removed: the agent names phase 1 sends to the agentic
teacher go to the `general` self-anchor (the RLVR student) instead, and the
allocation drops from 128 to 112 nodes. The other teachers (reasoning, RLHF
teacher 1, vision) and all training settings are phase 1's. Its RLHF teacher
is this repository's RLHF teacher 1 (from SFT), not NeMo-RL's `rlhf_teacher.yaml` (from
Student RLVR, NeMo-RL's name for its RLVR stage, with abstention and a 0.12 length penalty). Every teacher it
uses can be trained from public code with your own data, so this is the MOPD to run until the
agentic teacher's environment is public. It is not the report's pipeline,
which has the agentic teacher and a second phase.

```bash
nemotron super35 rl mopd -c single_stage --run <profile> \
    run.env.sandbox.container=<nemo-skills-sandbox-image>
```

It reads the phase-1 prompt blend (`super35-rl-mopd-phase1-data`) and writes
`super35-rl-mopd-single-stage-model`. Like NeMo-RL's config it has a 200-step
budget and no selected step; pick the checkpoint by evaluation.

## Runtime Environment

MOPD does not call any judge or GenRM servers, because verification is
skipped. The datasets mix many NeMo Gym environments, so some resource servers
still read data files when they start. Set these through
`run.env.env_vars`:

| Env var | Purpose |
|---|---|
| `NEMO_RL_VIDEO_MEDIA_ROOT` | Root directory for relative image and video paths in the MOPD data. Mount it into the container. |
| `CCC_TEST_METADATA` | Competitive-coding test metadata. The reference run used an empty placeholder file, because verification is skipped. |
| `FINANCE_SEC_CACHE_DIR` | SEC filing cache for `finance_sec_search` (`<dir>/finance_sec_search`, `<dir>/data`). |
| `SCICODE_TEST_DATA` | SciCode `combined_test_data.h5`. |

Code-execution environments need the NeMo-Skills sandbox
(`run.env.sandbox.container`). The reference run wrapped `/start-with-nginx.sh` in a
retry loop with up to six attempts, which is preserved in `run.env.sandbox.command`.

## Differences from the Reference Run

- **Phase-1 parallelism.** The selected phase-1 step-30 checkpoint came from
  a run whose saved config used **TP8 / CP8 / EP16 and zero MTP loss**. This
  recipe uses **TP4 / CP4 / EP16** with the common detached MTP5
  training (5 repeated heads, coefficient 0.3). The LR is still 2e-6, and
  policy MTP inference is off.
- **Phase-2 self-anchor.** The reference run's phase 2 used the original
  phase-1 run's checkpoint as its `general` teacher. This recipe uses the
  selected phase-1 step 30 as both the student and the self-anchor. Start a
  fresh experiment if you change either.
- **Checkpoint budget.** Phase 1's `checkpoint_must_save_by` (11h30m) assumes
  a 12-hour walltime, and phase 2's (3h30m) assumes 4 hours.
- Reward verification and replay restoration are disabled in both phases.

## Data

The MOPD prompt blends are not public yet. Register your own NeMo Gym JSONL as
`super35-rl-mopd-phase1-data` and `super35-rl-mopd-phase2-data` with
`nemotron super35 data import rl ...`. Each row's NeMo Gym agent name must appear
in the routing table, or the row falls back to the self-anchor.

## Requirements

- The Super 3.5 RL container (see [RL overview](../README.md#container)). The
  SAV tracking agents and the `bc_v4_8` tool-use config used in phase 2 come
  from its NeMo Gym pin.
- All teacher artifacts for the phase you run.
- 128 nodes × 4 GPUs (GB200).

## References

- [RL overview](../README.md) · [Teachers](../stage2_teachers/README.md) · [RLHF teachers](../stage2_teachers/rlhf/README.md)
- [NeMo RL](https://github.com/NVIDIA-NeMo/RL) · [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)
