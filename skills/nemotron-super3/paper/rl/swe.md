---
paper: "arxiv:2512.20856"
model: "nemotron-super3"
section: "swe-rl"
paper_sections: ["3.2.2"]
title: "Stages 2.1 and 2.2: Software-Engineering RL"
summary: |
  Super3 isolates software-engineering reinforcement learning into two stages:
  a pivot SWE stage and a full SWE-bench/OpenHands stage. The report treats SWE
  separately because rollouts are longer, contexts are larger, and every episode
  requires sandboxed repository execution and test-based rewards.
key_facts:
  - "SWE-RL is separated from RLVR because of rollout length, context size, and sandbox cost."
  - "Stage 2.1 uses a SWE pivot setting; stage 2.2 runs full repository-agent loops."
  - "Stage 2.2 uses OpenHands plus OpenCode/Codex-style agent variants in isolated containers."
  - "Released configs push sequence length to 131,072 for SWE1 and 196,608 for SWE2."
related_steps:
  - "stage2_rl/swe1"
  - "stage2_rl/swe2"
currency: "frozen"
---

# Scope

Use this file for questions about:

- why SWE-RL is separate from RLVR
- the difference between SWE1 and SWE2
- how the repository-agent environment works
- what OpenHands, OpenCode, and Codex-style agents are doing here
- how the released SWE configs map to the paper

---

# Why software engineering gets its own RL track

The report says SWE rollouts are operationally unlike the rest of RLVR.

| Difference | Consequence |
|---|---|
| Longer rollouts | much lower throughput per environment |
| Longer contexts | more memory pressure and different parallelism needs |
| Real repository execution | requires sandboxed filesystem/process isolation |
| Binary reward from tests | environment execution is part of the reward pipeline |

That is why SWE is not treated as just another RLVR environment family.

---

# Stage split

The report describes two SWE-oriented stages.

| Stage | Main purpose |
|---|---|
| SWE stage 1 | a pivot stage to adapt the policy toward software-engineering rollouts |
| SWE stage 2 | full end-to-end SWE-bench / OpenHands-style repository-agent training |

This ordering is practical: the policy first adapts to SWE-like interaction, then scales into the full repository harness.

---

# Stage 2.1: SWE pivot

The pivot stage is a lighter SWE-specific RL stage with shorter interaction structure than the full harness.

## Main features

| Feature | Reported/released behavior |
|---|---|
| Focus | software-engineering reward shaping before full SWE-bench loops |
| Overlong filtering | enabled |
| Learning rate | 1e-6 |
| Prompts per step | 64 |
| Generations per prompt | 16 |
| Train global batch size | 1024 |
| Max sequence length | 131,072 |
| TP / CP / EP | 8 / 8 / 8 |
| Prefix caching | enabled |

The most important change relative to RLVR is not only scale-down, but a **systems retuning** for long-horizon software tasks.

---

# Stage 2.2: full SWE-bench RL

This is the stage most users mean when they ask about “Super3 SWE training.”

## Core loop

For each episode, the environment:

1. launches an isolated repository environment,
2. presents the issue/problem statement,
3. runs an OpenHands-style agent loop,
4. extracts the resulting patch,
5. executes the ground-truth tests,
6. emits a binary reward.

The agent is therefore rewarded on **actual repository outcomes**, not just rubric judgments.

---

# OpenHands, OpenCode, and Codex-style agents

The report emphasizes tool diversity during SWE training.

## Why vary the agent interface?

A single repository harness can host multiple tool dialects and prompting styles. Super3 uses this to expose the policy to different coding-assistant interaction patterns.

## Reported agent variants

| Agent flavor | Purpose |
|---|---|
| OpenHands baseline loop | core SWE orchestration |
| OpenCode-style agent | exposes a Claude Code-like tool format |
| Codex-style agent | exposes a Codex CLI-like tool format |

The paper claims this multi-harness training improves the policy’s ability to generalize across software-engineering tool environments.

---

# Why container isolation matters

SWE training cannot safely share one global workspace across concurrent episodes. Each task needs:

- its own repository checkout,
- its own test execution context,
- protection against destructive shell actions,
- cleanup after the rollout ends.

The paper uses **Apptainer** because the target HPC setting lacks Docker-style root access.

## Practical implication

The SWE environment images exist as `.sif` files and are mapped into episodes through instance-specific format strings. This is why the open recipe has extra prerequisites beyond the base NeMo-RL container.

---

# Extra SWE-specific safeguards

The report and repo notes describe extra controls that are not central in the non-SWE RL stages.

| Safeguard | Why it exists |
|---|---|
| Memory watchdog | Apptainer shares host memory, so runaway subprocess trees can impact the node |
| Command blocklist | prevents commands like `killall` / `pkill` from killing shared infrastructure |
| Writable overlay / isolated filesystem | allows repo edits per task without contaminating other episodes |
| Faster serialization (`orjson`) | reduces overhead in environment/model-server communication |

These details are easy to overlook, but they explain why SWE-RL is a serious infrastructure project rather than just another benchmark.

---

# Released SWE1 operating point

From `stage2_rl/stage2_swe1/config/default.yaml`:

| Setting | Value |
|---|---|
| Nodes | 64 |
| Prompts per step | 64 |
| Generations per prompt | 16 |
| Train global batch size | 1024 |
| Max sequence length | 131,072 |
| Learning rate | 1e-6 |
| KL penalty | 0 |
| Overlong filtering | true |
| Prefix caching | enabled |
| Container | `nemo-rl:v0.5.0.nemotron_3_super_swe` |

This config is the clearest open-source reflection of the paper’s “SWE needs different system settings” claim.

---

# Released SWE2 operating point

From `stage2_rl/stage2_swe2/config/default.yaml`:

| Setting | Value |
|---|---|
| Nodes | 64 |
| Prompts per step | 16 |
| Generations per prompt | 32 |
| Train global batch size | 512 |
| Max sequence length | 196,608 |
| Learning rate | 1e-6 |
| KL penalty | 0 |
| Overlong filtering | true |
| Agent max turns | 200 |
| Agent concurrency | 768 |
| Agent timeout | 3600s |
| Thinking mode | enabled |

This shows how much more expensive the full SWE stage is: fewer prompts per step, larger contexts, and a heavier per-episode harness.

---

# How the repo maps the paper

| Paper concept | Open file |
|---|---|
| SWE stage 1 summary | `../../recipes/stage2_rl_swe1.md` |
| SWE stage 2 summary | `../../recipes/stage2_rl_swe2.md` |
| SWE1 config | `src/nemotron/recipes/super3/stage2_rl/stage2_swe1/config/default.yaml` |
| SWE2 config | `src/nemotron/recipes/super3/stage2_rl/stage2_swe2/config/default.yaml` |
| SWE1 README | `src/nemotron/recipes/super3/stage2_rl/stage2_swe1/README.md` |
| SWE2 README | `src/nemotron/recipes/super3/stage2_rl/stage2_swe2/README.md` |

The open recipe also spells out prerequisites the paper only summarizes:

- custom SWE container with prefetched venvs,
- sandbox container for code execution,
- Apptainer `.sif` images for full SWE-bench environments.

---

# Common answer patterns

## “Why is SWE not part of RLVR?”

Because SWE rollouts are much slower, use much longer contexts, and require isolated repository execution. Mixing them into RLVR would bottleneck the broader multi-environment run.

## “What is the difference between SWE1 and SWE2?”

SWE1 is a pivot stage with cheaper SWE-oriented rollouts; SWE2 is the full repository-agent harness with OpenHands-style loops and ground-truth test execution.

## “Does Super3 train on actual codebases?”

Yes, in the SWE-RL stages the environment launches repository sandboxes, applies patches, and scores them by running tests.

---

# Caveats

1. **Do not describe SWE-RL as just another benchmark fine-tune.** It is environment-based RL.
2. **Do not ignore the agent harness.** OpenHands and the tool dialects are part of the training story.
3. **Do not imply Docker is required.** The paper emphasizes Apptainer because of HPC constraints.

---

# Related files

- `overview.md`
- `rlvr.md`
- `rlhf.md`
- `../../recipes/stage2_rl_swe1.md`
- `../../recipes/stage2_rl_swe2.md`
