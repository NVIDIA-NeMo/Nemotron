# Nemotron 3.5 Super Quick Reference

Use this file for fast recall. All figures come from the `paper/` chunks of the Nemotron 3.5 Super
and Lightning Tech Report (2026). The report covers both models; everything here is **Super** unless
labeled. Throughput numbers are condition-specific.

---

## 1. Identity

- **Model**: Nemotron 3.5 Super 120B-A12B (NVIDIA); the released checkpoints are named Nemotron-3.5-Super-VL
- **Params**: 120B total / 12B active (124.453B counting vision encoder, projector, MTP; §4.1)
- **Initialized from**: Nemotron 3 Super Base
- **Modalities**: text, image, video in; text out; no audio
- **Context**: up to 1M tokens
- **Checkpoints**: Base BF16, post-trained BF16 (`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`), NVFP4
- **Sibling in the same report**: Nemotron 3.5 Lightning 30B-A3B (text-only; separate pipeline)

---

## 2. Architecture

- Language backbone inherited from Nemotron 3 Super: hybrid Mamba-Transformer LatentMoE, 512 routed experts per MoE layer. The report has **no dims table**; see `skills/nemotron-super3/paper/architecture.md` and the HF config.
- **MTP**: one recursively shared head, initialized from the first of Super 3 Base's two heads; hidden-state mixing during MTP training. The report's own wording varies (see `paper/architecture.md`).
- **Vision**: C-RADIO encoder + vision projector ("VL connector"). The temporal video embedder comes from the Megatron-Bridge provider, not the report.

---

## 3. Pretraining (text-only)

| Stage | Tokens | Notes |
|---|---:|---|
| MTP warm-up | 100B | backbone frozen, MTP head only, hidden-state mixing |
| Distillation from Ultra 3 Base | 1T | Skew Reverse KL (α 0.1) + CE; LR 4.5e-6 → 4.5e-5 → 4.5e-6 |
| Indic CPT | 100B | 10B Indic + 90B preceding mixture |
| 1M-context phase | 35B | constant LR 4.5e-6; CP16 / TP8 / EP64 on GB300 |
| Agentic SWE mid-training | 65.4B | 256K sequence length; LR warm-up to 1e-5 |
| LC recovery | ~9B | RULER-1M 79.9 → 33.5 (after mid-training) → 81.0 |

- Nemotron-Pretraining-Indic-v1: ~2.83T synthetic tokens, 22 languages, 34 language-script configurations, 4-stage NeMo Data Designer pipeline (Gemma-4-31B-it).
- Indic CPT gains: +4.98 to +9.82 points on five Indic benchmarks (Table 2).

---

## 4. SFT (adds vision)

- Vision-projector warm-up, then **two SFT stages**: stage 1 general mix (vision, reasoning, chat, long-context, safety, agentic); stage 2 small re-weighted blend.
- Packed sequences up to **524,288** tokens; **GBS 32**; LR warm-up to **1e-5**, then constant; Adam (0.9, 0.95), wd 0.1, clip 1.0; MTP aux loss scale 0.3.
- Reasoning traces regenerated with Kimi K3 (math/STEM), DeepSeek V4 Pro (proofs), GLM-5.2 (competitive programming).
- Medium-effort mode introduced in SFT (Kimi K3 low-effort data).

---

## 5. RL and MOPD

**Order:** SFT → RLVR trunk → specialist teachers → MOPD phase 1 → MOPD phase 2 → MTP boosting.

- **RLVR trunk (§3.2)**: joint RLVR + RLHF; GBS 8,192, 16 rollouts per prompt; max generation 80K → 128K; router replay + FP32 LM head. New environments: competitive programming (2,847 train problems) and a finance agent. **Step 120** is the MOPD phase-1 student and the starting point of every teacher except RLHF teacher 1.
- **Teachers (§3.3–3.5)**: agentic (end-to-end GRPO on Terminal/SWE/Search), reasoning (90% science; 131,072-token generation), video reasoning + tracking (step 60), vision tool use, vision reasoning (two stages), alignment (RLHF with a length-penalized Ultra GenRM variant).
- **MOPD (§3.6)**: per-token reverse KL to the teacher that owns each prompt's environment; frozen self-anchor for unowned environments; 128 GB200 nodes.
  - Phase 1: agentic 35%, reasoning 25%, alignment 20%, vision tool use 15%, self-anchor 5%; 512 prompts; LR 2e-6; 30 steps.
  - Phase 2: agentic and reasoning retired; alignment 30%, vision tool use 20%, vision reasoning 20%, video 20%, self-anchor 10%; 1,024 prompts; proximal teacher α = 0.2.
- **MTP boosting (§3.7)**: backbone frozen; LK loss against the backbone; 4K steps on SFT checkpoint, then 25K steps on the final backbone. Avg acceptance length (draft length 7) **4.287 → 4.718** greedy, **3.787 → 4.612** sampled.

---

## 6. Evaluation (post-trained BF16 unless labeled)

| Benchmark | Score |
|---|---:|
| SWE-bench Pro (OpenCode) | 51.78 (Ultra 3: 49.66) |
| SWE-bench Verified | 71.10 |
| Terminal-Bench 2.1 (Terminus 2) | 60.67 |
| GPQA-Diamond | 89.52 |
| LiveCodeBench v6 | 88.22 |
| HLE text, no tools / with tools | 31.79 / 46.01 |
| CharXiv-RQ | 80.82 |
| AA-LCR | 68.63 (Table 10) / 68.62 (Table 22) |
| IFBench | 70.86 |

- Test-time scaling (§3.11): Terminal-Bench 2.1 62.7 → 71.9 (5 parallel candidates, self-verification selects one via LLM-as-a-Verifier; not oracle best-of-5).
- Reasoning modes: reasoning-off, regular, medium-effort (~2.2× fewer tokens, ~9% accuracy drop).

---

## 7. Quantization (NVFP4 checkpoint)

- Released: **Aggressive** recipe, **4.82 bits per element**, + **200-iteration SFT-based QAD** (ModelOpt).
- **99.52% median accuracy recovery** over 35 benchmarks; verbosity +0.49%.
- Deploys **W4A4** on Blackwell, **W4A16** elsewhere. FP8 KV cache (NVFP4 KV: 99.36%).
- MTP: experts NVFP4, shared lm_head FP8 → AL 4.520 (draft length 7, temp 1), 1.127× projected speedup (simulated, draft length 5; Table 20).
- Reproduce: Model Optimizer's `examples/megatron_bridge/tutorials/NemotronH-Omni-W4A4-QAD` (recipe `model_type/nemotron_h/ptq/nvfp4-aggressive-mse` = released Aggressive tier; tutorial QAD is 600 iters on 32 GB300 with public data, 70 GB checkpoint — not the released run). Steps in `docs/nemotron/super35/quantization.md` → Recipe Execution.

---

## 8. Inference

- Workload: 20-turn agentic proxy, 2K input / 1K output per turn; "usable" = ≥ 50 tokens/s/user.
- NVFP4 vs BF16 max throughput: H100 522 → 970 (1.86×), GB200 1,778 → 4,324 (2.43×) tokens/s/GPU (Figure 10).
- vs Qwen on GB200 (Figure 11): Super 3.5 4,331.88, Qwen3.8 Flash-Next 2,282.73, Qwen3.5-122B-A10B 1,854.35.
- **Contradiction**: the Abstract and §1 claim 4,632 tokens/s/GPU (it matches a Table 23 ReplaySSM-off point) and §1/§6 claim 2.88× Qwen3.8 Flash-Next; Figure 11 shows 4,331.88 and implies 1.90×. Quote both.

---

## 9. Recipe map (`nemotron super35`)

| Paper stage | Repo stage | CLI |
|---|---|---|
| text-only pretraining (§2) | not reproduced | — |
| — (vision-language image-caption pretraining) | `stage0_pretrain` | `super35 pretrain` |
| SFT stage 1 / 2 (§3.1) | `stage1_sft` | `super35 sft`, `super35 sft -c stage2` (also `-c peft`, `-c long_context`) |
| RLVR (§3.2) | `stage2_rl/stage1_rlvr` | `super35 rl rlvr` |
| teachers (§3.3–3.5) | `stage2_rl/stage2_teachers/*` | `super35 rl teacher <name> [-c variant]` (agentic: blocked) |
| MOPD (§3.6) | `stage2_rl/stage3_mopd` | `super35 rl mopd [-c phase2 \| single_stage]` (`single_stage`: NeMo-RL's variant, no agentic teacher) |
| MTP boosting (§3.7), quantization (§4) | not included | — |
| evaluation (§3.8) | `stage3_eval` | `super35 eval` (five Table 10 benchmarks by default) |

The data and the report's length-penalized GenRM variant are not public (NeMo-RL's public recipe uses `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-GenRM`); users register their own (`super35 data import {sft,rl}`). The SFT output feeds RL directly (the RL runner converts Megatron-Bridge checkpoints to Hugging Face); only evaluation needs an explicit Hugging Face export.

---

## 10. `/nemotron-customize` step map

| Need | Step / mode |
|---|---|
| SFT on Megatron-Bridge | `sft/megatron_bridge` shape, but Super 3.5 is not in its `[[models]]` and the data path is text-only → ground on `recipes/super35/stage1_sft/` (Explorer mode) |
| LoRA | `peft/megatron_bridge` shape, same caveat → `stage1_sft` `peft` config |
| RLVR with GRPO | `rl/nemo_rl/rlvr` shape; environments and container come from `recipes/super35/stage2_rl/` |
| specialist teachers, MOPD | no catalog step → Explorer mode on `recipes/super35/stage2_rl/` |
| evaluate a served checkpoint | `eval/model_eval` |
| pretraining | no catalog step for this model → `recipes/super35/stage0_pretrain/` |

Adding Super 3.5 `[[models]]` entries to step manifests is a separate task; do not claim catalog support before then.

---

## 11. Best next files

| Need | File |
|---|---|
| exact numbers with conditions | `paper/evaluation.md`, `paper/quantization.md`, `paper/inference.md` |
| MOPD mechanics | `paper/rl/mopd.md` |
| running it in this repo | `recipes/overview.md` |
