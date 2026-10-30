# Nemotron 3 Ultra Quick Reference

Use this file for fast recall. All figures are from the audited `paper/` chunks of the
Nemotron 3 Ultra v3 Tech Report (2026-06-03). Throughput numbers are condition-specific.

---

## 1. Identity

- **Model**: Nemotron 3 Ultra 550B-A55B
- **Developer**: NVIDIA
- **Family**: largest / most capable Nemotron 3 model
- **Active / total params**: 55B active / 550B total (MoE)
- **Architecture**: hybrid Mamba-Attention MoE (Mamba-2 + Attention + LatentMoE)
- **Context length**: up to 1M (1,048,576) tokens
- **Pretraining**: 20T text tokens (NVFP4)
- **Post-training**: SFT → RLVR → MOPD warmup → MOPD (×2) → MTP Boosting
- **Announced**: GTC San Jose 2026; full release expected 1H 2026 (base-only first)
- **Pretraining languages**: AR, ZH, FR, DE, HE, HI, IT, JA, KO, PT, ES (11)
- **Best for**: agentic workflows, tool use, software engineering, long-context reasoning, code, math

---

## 2. Architecture (Table 1)

- Same hybrid Mamba-Attention MoE as Nemotron 3 Super, scaled to 550B/55B
- **108 layers**, **model dim 8192**
- **64 Q-heads / 2 KV-heads**, head dim 128
- Mamba: state dim 128, 8 groups, 256 heads, head dim 64
- Expert hidden dim 5120; shared-expert intermediate size 10240
- **512 experts per layer**, **top-k = 22**, **MoE latent size 2048**
- **2 MTP layers, shared weight** (each MTP head = 1 attention layer + 1 MoE layer)

### Why LatentMoE / MTP
- LatentMoE: routes in a low-rank latent space → more experts per inference cost (same as Super3)
- MTP: native speculative decoding; shared-weight head applied recursively for the draft horizon

---

## 3. Headline claims

- 550B total / 55B active MoE hybrid Mamba-Attention with LatentMoE + MTP
- Pretrained in **NVFP4** on **20T** tokens; context extended to 1M
- Post-trained with SFT + RLVR + MOPD; claimed largest stable+accurate NVFP4 training to date
- Base model claims higher accuracy than DeepSeek-V3.2, Mistral-Large-3, Kimi-K2, GLM-4.5 base

### Throughput framing (condition-specific)
- Abstract: up to **~6x** vs SOTA public LLMs at on-par accuracy
- Conclusion: **~5x** vs SOTA open LLMs at on-par accuracy
- Decode-heavy 8K/64K, GB200, NVFP4: **5.9x** vs GLM-5.1, **3.9x** vs Kimi-K2.6, **1.3x** vs Qwen-3.5
- Prefill-heavy 50K/2K: Ultra trails Qwen-3.5 (3.9 vs 4.6 relative)

---

## 4. Pretraining

### Schedule (WSD, 20T horizon)
- **Phase 1**: 15T tokens (diversity / broad coverage)
- **Phase 2**: 5T tokens (high-quality refinement); transition after ~15T (~75%)
- Warmup **200B tokens** to peak LR **2.5e-4**; final 5T = minus-sqrt decay to min LR **2.5e-6**
- MTP loss scaling **0.1** (0.05 per MTP block)
- Offline checkpoint merging: sliding window 500B tokens, interval 25B; selection windows 125B–1T

### NVFP4 recipe
- E2M1, 2D block quant on weights, Random Hadamard Transforms on wgrad inputs, stochastic rounding
- Kept higher precision: final 15% of network (**16 layers**), Mamba output projections, latent
  projections, QKV/attention projections, MTP layers, embeddings
- BF16 health-check ablations (5T/10T/16T branches, +60B tokens): avg rel train-loss gap **< 0.4%**

### Two divergences (§2.7)
- **~8T tokens**: output-layer gradient accumulation FP32→BF16 lost MTP wgrad → fixed by rollback + FP32 reduction
- **~16T tokens**: undetermined; mitigated by early LR annealing from ~15T checkpoint → horizon cut to 20T
- MaxVio_max = E/k = **23.27** (Ultra & Super), 21.33 (Nano)

---

## 5. Long-context (LC-Phase)

- CPT at end of pretraining; constant LR **2.5e-6**
- Parallelism: CP=32, TP=8, EP=128, PP=2 on GB200
- Blend: **46% long-context** data, **54% Phase 2** data; no RULER-style data
- **92%** iterations at 1M (1,048,576), **8%** at 4K (4,096); 4K iters use only math/code SFT-style data
- Tokens/iteration **25,165,824**; LC-Phase total **33B tokens**

---

## 6. Pretraining data

### New datasets (since Super)
- **Nemotron-Pretraining-Code-v3**: 173B new GitHub tokens, cutoff Sept 30 2025
- **Multiple-Choice / Generative**: task-seeded synthetic Q&A (held-out tests excluded)
- **Fact-Seeking**: from Finewiki via Qwen3-30B-A3B-Instruct-2507
- **Moral-Scenarios**: Moral Stories + Social Chemistry; CoT via Qwen3-235B-A22B-Thinking-2507
- **Legal**: many sub-datasets; Case-Law-Summary = 5.4M summaries

### Ablation gains (on Nano / Nemotron-family checkpoints, NOT Ultra)
- Synthetic Q&A: MMLU-Pro 64.8→66.6, code 73.2→75.1, GPQA 30.8→41.9
- Fact-seeking (MC SimpleQA): 40.24→50.16
- Legal (>100 LegalBench subtasks): 64.6→74.7

### Mixture
- Two-phase curriculum, 19 high-level categories
- Largest = quality-filtered + synthetic web crawl: ~49% phase-1, ~42% phase-2 tokens

---

## 7. SFT (§3.1)

- Packed sequence length **294,912**; global batch **64**; **204,800 samples**
- Cosine LR: peak **1.5e-5**, min **1e-6**, **9,600** warmup samples
- Shared-weight MTP retained (2 layers, aux-loss scaling 0.1)
- Domains: long-context, efficiency/control, safety, search, terminal-use, conversational tool-use,
  software issue resolution, math/proof, science, chat, code, CUDA, RTL, multilingual
- Safety blend ~135K (~45K EN + ~15K each DE/ES/FR/JA/IT/ZH); <0.8 similarity filtered
- Terminal-use ~370K conversations; code = 1.2M Python + 1.0M C++14 + 1.3M Python tool-calling
- Length-aware best-fit data packing (no truncation/splitting, in-pack dedup)

---

## 8. MOPD pipeline

Pipeline (Figure 9): **Base → SFT → RLVR → MOPD Warmup → MOPD (×N) → MTP Boosting**

### RLVR (§3.2)
- Unified RL with verifiable reward across all environments (async GRPO + stability opts)
- Global batch **8192**, **16** rollouts/sample; max gen length 48K → 64K

### MOPD (§3.3)
- **>10** domain-specialized teachers distilled into one student via async on-policy distillation
- Dense token-level reverse-KL on student rollouts; lambda_i weights each domain
- Async surrogate decouples behavior / proximal policies; IcePop token masking
- Max gen length **192K**; **1,024** prompts/batch; **1** rollout/prompt; **2 iterations** for Ultra

### Warmup (§3.3.3)
- Light SFT on teacher-distribution data to fix student/teacher distribution mismatch
- Helps agentic (GDPVal 35.3→46.7 no-warmup→warmup), negligible on HLE

### Results (Table 5, recovery = (MOPD2−RLVR)/(Teacher−RLVR))
- Strong agentic recovery: Terminal Bench 2.0 172.7%, TauBench Telecom 90.3%, SWE-Bench Verified 88.1%
- Weak self-contained reasoning: HLE 16.9%, LiveCodeBench 32.0%

### MTP Boosting (§3.4)
- Head-only continued training from MOPD checkpoint; backbone frozen
- Temperature-scaled forward-KL vs backbone logits (T=2, N_mtp=7, scaled by T²); gold-token CE disabled
- Raises SPEED-Bench avg acceptance length 4.387→4.584 (greedy); speedup gains 3.15%–5.82%

---

## 9. Reasoning control (§3.5)

- Three modes: **reasoning-off**, **regular**, **medium-effort**
- Regular + medium-effort combine with inference-time budget control
- Medium-effort introduced in SFT, optimized in RLVR (~2.5% of RLVR prompts; math/STEM/coding)
- Medium-effort: ~**2.5x fewer tokens** at ~**7% accuracy** cost (Figure 10, AA Index V4)

---

## 10. Evaluation

### Base (550B-A55B-Base)
- MMLU **89.08**, MMLU-Pro **79.07**, GPQA **50.00**, MATH **82.00**, HumanEval **83.84**
- RULER 1M **76.83**

### Post-trained BF16 (Table 10)
- MMLU-Pro 86.8, GPQA (no tools) 87.0, LiveCodeBench v6 89.0, IMOAnswerBench 88.6 / 92.3 (tools)
- SWE-Bench Verified 71.9, Terminal Bench 2.1 56.4, TauBench V3 avg 70.9, BrowseComp 44.4
- PinchBench 90.0, ProfBench 56.0 (held-out gates); AA-Omniscience non-hallucination 78.7 (highest)
- Arena-Hard-V2 88.1, Multi-Challenge 63.8, IFBench 81.7; RULER 1M 94.7

### Test-time scaling (Table 11, 128 attempts, 512k context)
- IMO 2025 83.3% (35/42), Putnam 2025 97.5% (117/120), USAMO 2026 97.6% (41/42)

---

## 11. Quantization (§4)

- PTQ to **NVFP4** for Blackwell via Model-Optimizer; operating point **5.03 BPE** (NVFP4 + mixed-FP8)
- Per-op: routed experts NVFP4; shared experts + Mamba linears FP8 per-tensor; attention/LatentMoE/embed/MTP BF16
- KV cache FP8; Mamba SSM cache FP16 with stochastic rounding (FP16 SR: −0.32% acc / 1.13% verbosity)
- Weight scale rule: **Four-Over-Six** (−16.4% median weight-recon MSE vs max calibration)
- Quantized with Megatron-LM (45 min total) vs HuggingFace layerwise (~2 hr)
- One NVFP4 checkpoint: W4A4 on Blackwell, W4A16 on Hopper; W4A8 not shipped

---

## 12. Inference (§5)

- LatentMoE + hybrid Mamba-2/Attention + MTP for native speculative decoding
- Decode-heavy lead (Mamba-2 per-step cost constant in seq length); prefill trails (compute-bound, 55B active)
- MTP draft length = deployment knob; peaks at DL=6 → 2.89x speedup (single-user 10K/16K/1, TP=4, GB200)
- Parallelism: wide TP for low-latency (weight-read bound), wide EP for high-throughput (comm bound)
- GB200 NVL72 single NVLink domain over 72 GPUs; prefill-decode disaggregation (~10% gain, transfers KV + Mamba state)
- FlashInfer NVLinkOneSided all-to-all (~5% gain); routed-expert all-to-all ~15–20% of prefill-heavy runtime

---

## 13. Best next file

| Need | Open |
|---|---|
| authoritative release facts | `../model-card.md` |
| architecture deep dive | `../paper/architecture.md` |
| pretraining / data | `../paper/pretraining.md`, `../paper/data.md` |
| post-training | `../paper/sft.md`, `../paper/mopd/overview.md`, `../paper/mopd/teachers.md` |
| reasoning / MTP boosting | `../paper/mopd/mtp-reasoning.md` |
| quantization / inference | `../paper/quantization.md`, `../paper/inference.md` |
| evaluation | `../paper/evaluation.md` |
| safety | `../paper/safety.md` |
