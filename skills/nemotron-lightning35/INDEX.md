# Nemotron 3.5 Lightning Knowledge Map

Four source layers:

1. **Skill routing**: `SKILL.md`
2. **Model card**: `model-card.md`
3. **Paper chunks**: `paper/` (the joint Nemotron 3.5 Super and Lightning report, plus labelled repo-docs facts)
4. **Recipe tracker**: `recipes/` (what `nemotron lightning35` runs)

---

## Question → file map

| Question type | Primary file(s) | Notes |
|---|---|---|
| What is Lightning 3.5? | `model-card.md`, `paper/_overview.md` | identity, lineage, checkpoints |
| Architecture / dims / MTP head | `paper/architecture.md`, `paper/mtp.md` | report dims vs docs dims |
| How was the base built? | `paper/pretraining.md` | Nano 3 Base → MTP warm-up → distillation from Super 3 Base → long context |
| Data | `paper/data.md` | distillation data, Table 3 Indic ablations, open recipe blends |
| Base benchmarks | `paper/evaluation-base.md` | Table 6 |
| SFT | `paper/sft.md` | single text-only stage; details not reported |
| RL | `paper/rl.md` | RLVR; recipe config vs docs |
| MTP boosting / speculative decoding | `paper/mtp.md` | 48K steps, no warm start |
| Post-trained benchmarks | `paper/evaluation.md` | Table 21 (BF16 vs NVFP4) |
| NVFP4 recipe and QAD | `paper/quantization.md` | W4A16 on every SKU; report vs docs disagreements |
| Safety | `paper/safety.md` | what is and is not reported |
| Run it in this repo | `recipes/overview.md` | stage map, CLI, artifacts, gaps |
| Pretraining recipe | `recipes/stage0_pretrain.md` | open data, not the report's distillation |
| SFT recipe | `recipes/stage1_sft.md` | maps to `sft/megatron_bridge` |
| RL recipe | `recipes/stage2_rl.md` | GRPO config values |
| Evaluation recipe | `recipes/stage3_eval.md` | NeMo Gym reference flow |
| Build with `/nemotron-customize` | `context/quick-reference.md` → step map | only `sft/megatron_bridge` lists Lightning |

---

## Important distinctions

1. **Lightning vs Super 3.5**: same report, different models. Lightning: text-only, SFT + RLVR, no MOPD, W4A16 everywhere. Most report tables are Super-only.
2. **Lightning vs Nano 3**: Nano 3 Base is Lightning's starting checkpoint. Earlier repo docs versions contained Nano 3 text; it was removed, and the `paper/` files note it where users may quote it.
3. **Report vs model cards vs recipe**: the report describes the released model; the Base model card describes the base differently (>20T tokens, NVFP4, no distillation); the docs follow the report and note the card; the recipes train on open data only.
4. **Base vs post-trained BF16 vs NVFP4**: Table 6 vs Table 21 (BF16 and NVFP4 columns).
