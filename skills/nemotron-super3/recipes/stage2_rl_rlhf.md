# Stage 2.4 Recipe Summary — RLHF

This file maps the final RL stage to:

- `src/nemotron/recipes/super3/stage2_rl/stage3_rlhf/`

---

## What this sub-stage is

RLHF is the final preference-shaping stage after RLVR and SWE-RL. The released config exposes the GenRM-based comparison environment, the nonzero KL penalty, and the scaled-down sequence-length setting relative to SWE.

---

## Main source files

| Path | Role |
|---|---|
| `.../stage3_rlhf/README.md` | human overview |
| `.../stage3_rlhf/config/default.yaml` | production RLHF config |
| `.../stage3_rlhf/config/small.yaml` | reduced-scale config |

---

## Key production config values

| Setting | Value |
|---|---|
| Input data artifact | `super3/rl/rlhf/data:latest` |
| Input model artifact | `super3-sft-model:latest` (template default) |
| Container | `nemo-rl:v0.5.0.nemotron_3_super` |
| Nodes | 72 |
| Prompts per step | 128 |
| Generations per prompt | 16 |
| Train batch size | 2048 |
| Max sequence length | 49,152 |
| TP / CP / EP | 4 / 4 / 8 |
| Learning rate | 1e-6 |
| KL penalty | 1e-4 |
| Async GRPO | enabled |
| GenRM compare | enabled |

---

## Repo-specific observations

**Chaining note:** in the documented end-to-end sequence, RLHF should consume the SWE2 output checkpoint. The checked-in YAML keeps `super3-sft-model:latest` as a base template and should be overridden for a real sequential run.

The config makes the paper’s RLHF story concrete by showing:

- the `genrm_compare` environment,
- the principle-following comparison prompt,
- an additional tool-use comparison environment,
- the GenRM serving configuration and router DP size.

If the user wants the implementation proof that RLHF is different from RLVR, cite this config.

---

## What the repo does not surface as strongly

The paper explicitly mentions a final **MTP-healing** pass after RLHF. The released RLHF recipe does not expose that as an equally visible standalone stage, so use the paper chunk if the question is specifically about healing.

---

## Best next file

- `stage3_eval.md` for what happens after the final RL model is produced.
- `../paper/rl/rlhf.md` for the research rationale and MTP-healing context.
