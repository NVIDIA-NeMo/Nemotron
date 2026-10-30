# Stage 2.2 Recipe Summary — SWE1 Pivot RL

This file maps the first software-engineering RL stage to:

- `src/nemotron/recipes/super3/stage2_rl/stage2_swe1/`

---

## What this sub-stage is

SWE1 is the pivot stage between broad RLVR and the full SWE-bench harness. It keeps the same NeMo-RL / async-GRPO backbone but retunes the system for much longer software-engineering rollouts.

---

## Main source files

| Path | Role |
|---|---|
| `.../stage2_swe1/README.md` | human overview |
| `.../stage2_swe1/config/default.yaml` | production SWE1 config |
| `.../stage2_swe1/config/small.yaml` | reduced-scale config |

---

## Key production config values

| Setting | Value |
|---|---|
| Input data artifact | `super3/rl/swe1/data:latest` |
| Input model artifact | `super3-sft-model:latest` (template default) |
| Container | `nemo-rl:v0.5.0.nemotron_3_super_swe` |
| Nodes | 64 |
| Prompts per step | 64 |
| Generations per prompt | 16 |
| Train batch size | 1024 |
| Max sequence length | 131,072 |
| TP / CP / EP | 8 / 8 / 8 |
| Learning rate | 1e-6 |
| KL penalty | 0 |
| Overlong filtering | true |
| Prefix caching | enabled |

---

## Operational differences from RLVR

Compared with RLVR, SWE1:

- lowers throughput targets,
- doubles tensor parallelism,
- increases sequence length,
- turns on overlong filtering,
- requires the special SWE container with prefetched venvs.

Those differences are exactly what the paper means when it says SWE needs separate infrastructure.

**Chaining note:** in the documented end-to-end sequence, SWE1 should consume the output of RLVR3. The checked-in YAML keeps `super3-sft-model:latest` as a base template and should be overridden for a true sequential run.

---

## Prerequisites surfaced by the repo

The README calls out:

- NeMo-RL `super-v3` branch,
- a sandbox container,
- a custom SWE container built from the base NeMo-RL image.

That makes this file the right citation when the user asks “what extra infra do I need for SWE?”

---

## Best next file

- `stage2_rl_swe2.md` for the full SWE-bench harness.
- `../paper/rl/swe.md` for the research-level explanation.
