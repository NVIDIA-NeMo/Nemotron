# Nemotron 3.5 Lightning Quick Reference

Use this for compact answers or a `/nemotron-customize` handoff. Figures come from the `paper/`
chunks (joint Nemotron 3.5 Super and Lightning report) unless labelled **docs** (repo docs) or
**card** (Hugging Face model cards). Never borrow Super 3.5 numbers.

---

## 1. Identity

| Item | Value |
|---|---|
| Model | Nemotron 3.5 Lightning 30B-A3B |
| Params | 30B total / 3B active |
| Modality | text only |
| Starting point | Nemotron 3 Nano Base |
| Context | up to 1M tokens (card; report gives no context length but Table 21 has RULER-v1 at 1M) |
| License | OpenMDW-1.1 (docs) |

## 2. Checkpoints

`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-{Base-BF16,BF16,NVFP4}`

## 3. Architecture

- Hybrid Mamba-Transformer MoE; hidden 2688, MoE intermediate 1856, 128 routed experts (report §2.1).
- Docs: 52 layers, top-6 routing, 1 shared expert, GQA 32 heads / 2 KV groups.
- MTP head initialized from Super 3 Base's head: latent projections fused into the experts, then pruned (hidden 4096 → 2688, MoE intermediate 2688 → 1856, experts 512 → 128).

## 4. How the base was built (§2)

| Step | Detail |
|---|---|
| MTP warm-up | 100B tokens, backbone frozen, standard 2-step recursive MTP (no hidden-state mixing) |
| Distillation | 1T tokens of Ultra 3 Phase 2 data; teacher Super 3 Base, top-128 logits + sink token; KL/CE 0.9/0.1; LR 1e-5 → 1e-4 (200B) → 1e-5 (800B); MTP loss scale 0.1; load-balancing coef 1e-4 |
| Long context | "undergoes long-context training"; no Lightning-specific settings reported |
| Indic | not in the released model; Table 3 Indic-share ablations are research runs |

## 5. Post-training

- Single-stage text-only SFT → RLVR (no teachers, no MOPD). SFT and RL data and hyperparameters: not reported.
- MTP boosting: 48K steps against the final backbone, no warm start, LK loss. Acceptance lengths: not reported.

## 6. Results

- **Base** (Table 6): e.g. MMLU 78.63, MMLU-Pro 67.50, MATH-500 79.53, HumanEval 76.91 → `paper/evaluation-base.md`.
- **Post-trained BF16 → NVFP4** (Table 21): e.g. SWE-Bench Verified 52.20 → 52.20, Terminal-Bench 2.1 24.44 → 22.05; median recovery **99.24%**, verbosity −3.63% → `paper/evaluation.md`, `paper/quantization.md`.

## 7. Quantization

W4A16 on every SKU; MTP not quantized; FP8 KV cache and FP32 Mamba SSM cache in the report's evaluation. Mamba projections FP8 (report §4.8; confirmed by the NVFP4 card). Calibration four-over-six, 1000 samples of a Nemotron Ultra validation subset at 32K (card; the report names no method). The docs' ModelOpt recipe uses W4A16 NVFP4 Mamba linears instead; the docs flag it. Checkpoint 22 GB vs 66 GB BF16.

---

## 8. Recipe map (`nemotron lightning35`)

| Stage | CLI | Notes |
|---|---|---|
| `stage0_pretrain` | `data prep pretrain`, `pretrain` | open data; not the report's distillation |
| `stage1_sft` | `data prep sft`, `sft` | open SFT blend, pack size 4,096 |
| `stage2_rl` | `data prep rl`, `rl` | GRPO, Lightning RL training blend, 32 nodes |
| `stage3_eval` | `eval` | NeMo Gym reference flow; `gpqa` by default |
| — | `pipe` | pretrain → SFT |

## 9. `/nemotron-customize` step map

| Need | Step / mode |
|---|---|
| SFT on Megatron-Bridge | `data_prep/sft_packing` + `sft/megatron_bridge` (lists Lightning) |
| pretraining | `pretrain/megatron_bridge` shape (does not list Lightning) → ground on `recipes/lightning35/stage0_pretrain/` |
| RLVR | `rl/nemo_rl/rlvr` shape (does not list Lightning) → ground on `recipes/lightning35/stage2_rl/` |
| evaluation | `recipes/lightning35/stage3_eval/` (NeMo Gym); `eval/model_eval` only for a quick endpoint check |
| quantization | `optimize/modelopt/quantize` shape; follow `docs/nemotron/lightning35/quantization.md` |

## 10. Caveats

- The Base model card (>20T tokens, NVFP4 recipe, no distillation) conflicts with the report (Nano 3 Base + 1T-token distillation from Super 3 Base); the docs follow the report. Give both.
- Earlier repo docs versions contained Nano 3 text (25T curriculum, GenRM/DPO, 3.3x throughput, RULER comparison); it was removed. See `SKILL.md → Known caveats`.
