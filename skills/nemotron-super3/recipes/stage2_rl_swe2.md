# Stage 2.3 Recipe Summary — SWE2 Full SWE-Bench RL

This file maps the full software-engineering RL stage to:

- `src/nemotron/recipes/super3/stage2_rl/stage2_swe2/`

---

## What this sub-stage is

SWE2 is the full repository-agent RL harness. It runs OpenHands-style episodes inside isolated containers, produces code patches, and scores them with ground-truth test execution.

---

## Main source files

| Path | Role |
|---|---|
| `.../stage2_swe2/README.md` | human overview |
| `.../stage2_swe2/config/default.yaml` | production config |

---

## Key production config values

| Setting | Value |
|---|---|
| Input data artifact | `super3/rl/swe2/data:latest` |
| Input model artifact | `super3-sft-model:latest` (template default) |
| Container | `nemo-rl:v0.5.0.nemotron_3_super_swe` |
| Nodes | 64 |
| Prompts per step | 16 |
| Generations per prompt | 32 |
| Train batch size | 512 |
| Max sequence length | 196,608 |
| TP / CP / EP | 8 / 8 / 8 |
| Learning rate | 1e-6 |
| KL penalty | 0 |
| Overlong filtering | true |
| Agent max turns | 200 |
| Agent concurrency | 768 |
| Agent timeout | 3600s |
| Thinking mode | enabled |

---

## What makes SWE2 special

The config and README expose several details the paper emphasizes:

- Apptainer-based `.sif` container execution
- OpenHands loop orchestration
- OpenCode and Codex-style tool formats
- instance-specific container format strings
- shared-memory safeguards such as watchdog and command blocklists

This is the best released source for “how does the full SWE harness actually run?”

**Chaining note:** in the documented sequential RL pipeline, SWE2 should start from the SWE1 output checkpoint. The checked-in YAML uses `super3-sft-model:latest` as a reusable template default and should be overridden in a real chained run.

---

## Required extra infrastructure

Beyond the base RL stack, SWE2 needs:

- the special SWE container,
- a sandbox container,
- a directory of Apptainer `.sif` images for SWE task environments.

The README even includes the `download_swe_images.py` path for building those `.sif` assets.

---

## Best next file

- `stage2_rl_rlhf.md` for the next stage in the RL chain.
- `../paper/rl/swe.md` for the research framing of why SWE is separate.
