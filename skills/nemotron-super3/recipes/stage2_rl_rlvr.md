# Stage 2.1 Recipe Summary — RLVR

This file maps the paper’s RLVR stage to:

- `src/nemotron/recipes/super3/stage2_rl/stage1_rlvr/`

---

## What this sub-stage is

RLVR is the first RL stage after SFT. In the paper it is the broad multi-environment capability RL phase; in the repo it is implemented as a NeMo-RL / Ray training setup plus resolved JSONL data blends.

---

## Main source files

| Path | Role |
|---|---|
| `.../stage1_rlvr/README.md` | human overview |
| `.../stage1_rlvr/train.py` | RLVR training entrypoint |
| `.../stage1_rlvr/config/default.yaml` | production config |
| `.../stage1_rlvr/config/small.yaml` | reduced-scale variant |
| `src/nemotron/recipes/super3/stage2_rl/data_prep.py` | resolves released RL blends |

---

## Key production config values

From `stage1_rlvr/config/default.yaml`:

| Setting | Value |
|---|---|
| Input data artifact | `super3/rl/rlvr1/data:latest` |
| Input model artifact | `super3-sft-model:latest` (template default) |
| Container | `nvcr.io/nvidia/nemo-rl:v0.5.0.nemotron_3_super` |
| Nodes | 109 |
| GPUs per node | 8 |
| Prompts per step | 256 |
| Generations per prompt | 16 |
| Train batch size | 4096 |
| Max sequence length | 65,536 |
| TP / CP / EP | 4 / 8 / 8 |
| Learning rate | 3e-6 |
| KL penalty | 0 |
| Async GRPO | enabled |
| In-flight weight updates | enabled |


**Chaining note:** the checked-in YAML is a base template. In the documented sequential RL pipeline, RLVR1 starts from the SFT checkpoint, then RLVR2 and RLVR3 should be pointed at the previous RLVR stage output.

---

## Training script behavior

`train.py` is a runspec-style Ray job that:

- loads the RL config,
- sets up an initial checkpoint structure if needed,
- can convert Megatron checkpoints to HF format,
- launches GRPO training inside the NeMo-RL environment.

This is the file to cite when the user wants the actual implementation entrypoint rather than only the YAML.

---

## Data path

The RLVR stage depends on the shared RL data-prep pipeline in:

- `src/nemotron/recipes/super3/stage2_rl/data_prep.py`

That script resolves placeholder rows from the released HF RL blends and writes `train-split.jsonl` / `val-split.jsonl` for each RLVR stage.

---

## Repo-specific observations

- The released config hardcodes a rich NeMo-Gym environment stack, including math, code, terminal, safety, structured outputs, and GenRM compare servers.
- The config also exposes the judge-model stack and sandbox-container hooks the paper only summarizes conceptually.
- The RLVR README is the clearest released source for the “21 environments / 37 datasets” explanation. The same README also makes clear that the paper’s single conceptual RLVR stage is implemented as three sequential runs (`rlvr1`, `rlvr2`, `rlvr3`) in the released pipeline.

---

## Best next file

- `stage2_rl_swe1.md` if the user wants to follow the actual stage chain.
- `../paper/rl/rlvr.md` if the user wants the research-level rationale.
