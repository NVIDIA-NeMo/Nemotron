# Nemotron 3 Ultra — Model Card

> Release identity and deployment facts for the NVIDIA Nemotron 3 Ultra 550B-A55B model family.
> Numbers are drawn from the Nemotron 3 Ultra v3 Tech Report (2026-06-03) and the
> `Nemotron-3-Ultra-Base` repo model card. This is a knowledge-base summary, not the official card.

## 1. Identity

| Field | Value |
|---|---|
| Model | Nemotron 3 Ultra 550B-A55B |
| Developer | NVIDIA |
| Family | Largest / most capable model in the Nemotron 3 family |
| Architecture | Mixture-of-Experts hybrid Mamba-Attention (Mamba-2 + Attention + LatentMoE) |
| Total parameters | 550B |
| Active parameters / token | 55B |
| Layers | 108 (model dim 8192; see `paper/architecture.md`) |
| Context length | up to 1M (1,048,576) tokens |
| Pretraining | 20T text tokens (NVFP4), then long-context extension to 1M |
| Post-training | SFT → RLVR → MOPD warmup → MOPD (×2) → MTP Boosting |
| Announced | GTC San Jose 2026 |
| Repo / weights | github.com/NVIDIA-NeMo/Nemotron + HuggingFace |

Key technologies: LatentMoE, Multi-Token Prediction (MTP), NVFP4 pre-training, multi-environment RLVR,
Multi-teacher On-Policy Distillation (MOPD), reasoning effort/budget control.

## 2. Variants / Checkpoints

The tech report (§1) names four released checkpoints:

| Checkpoint | Precision | Description | Purpose |
|---|---|---|---|
| Nemotron 3 Ultra 550B-A55B Base BF16 | BF16 | Pretrained base model (long-context extended) | Customization / further training |
| Nemotron 3 Ultra 550B-A55B BF16 | BF16 | Post-trained model | Reference high-precision deployment |
| Nemotron 3 Ultra 550B-A55B NVFP4 | NVFP4 | Post-trained + NVFP4-quantized | Blackwell inference (W4A4) / Hopper (W4A16) |
| Nemotron 3 Ultra 550B-A55B GenRM | — | Generative Reward Model used for RLHF | Alignment / reward modeling |

Also open-sourced: training recipes, training data, and RL environments.

### NVFP4 checkpoint deployment
A single NVFP4 checkpoint serves two regimes (§4, Table 17): native FP4 (W4A4) on Blackwell, and
W4A16 (NVFP4 weights, BF16 activations) on Hopper (which lacks native FP4 tensor cores). KV cache in
FP8; Mamba SSM cache in FP16 with stochastic rounding. Operating point is 5.03 bits-per-element (BPE).
W4A8 was not shipped (strictly worse than W4A16, no accuracy upside).

## 3. Intended Use

- Agentic workflows, tool use, software engineering, search, terminal use, long-context reasoning,
  reasoning/STEM, code, math, chat, multilingual tasks.
- Base BF16 checkpoint is a **pretraining base** — not instruction-tuned or aligned; per the repo card it
  is "not meant to be used out of the box as an assistant or in a production pipeline" and is best as a
  **starting point for customization** (fine-tuning, RL post-training, custom instruction tuning).
- For direct deployment, use a post-trained checkpoint (BF16 or NVFP4).

### Reasoning control
Three reasoning modes: reasoning-off, regular, and medium-effort. Regular and medium-effort can be
combined with inference-time budget control. Medium-effort mode uses on average ~2.5x fewer tokens than
regular at ~7% accuracy cost (paper claim, Figure 10, AA Intelligence Index V4).

## 4. Supported Context

- Up to **1M tokens** (1,048,576), enabled by Mamba-2 linear-time-in-sequence-length layers plus sparse
  global attention anchors.
- Base RULER (0-shot): 64K 95.30, 128K 92.49, 256K 86.22, 512K 84.54, 1M 76.83.
- Post-trained RULER (1M) 94.7; LongBench v2 (≤1M) 61.9; AA-LCR 65.4 (see `paper/evaluation.md`).

### Supported languages
Pretraining multilingual data spans 11 languages: Arabic, Chinese, French, German, Hebrew, Hindi,
Italian, Japanese, Korean, Portuguese, Spanish. (Post-training/eval multilingual coverage may differ;
see `paper/evaluation.md` MMLU-ProX / WMT24++.)

## 5. Throughput Framing

Throughput claims are **measurement-condition-specific** — do not generalize a single multiplier.

- Tech report abstract: up to **~6x** higher inference throughput vs SOTA public LLMs at on-par accuracy.
- Tech report conclusion: **~5x** higher inference throughput than other SOTA open LLMs at on-par accuracy.
- Repo (base) card: up to **5x** higher TPS at max throughput vs GLM-4.5-355B-A32B / Kimi-K2-1026B-A33B
  on GB200 NVL72.

Specific decode-heavy figures (8K input / 64K output, GB200, NVFP4, max-throughput, TRT-LLM for Ultra
vs vLLM for others, best of with/without spec decoding):

| Comparison model | Ultra speedup |
|---|---|
| GLM-5.1-754B-A40B | 5.9x |
| Kimi-K2.6-1T-A32B | 3.9x |
| Qwen-3.5-397B-17B | 1.3x |

On prefill-heavy 50K/2K workloads Ultra trails Qwen-3.5 (3.9 vs 4.6 relative throughput; see
`paper/inference.md`). MTP speculative decoding adds a deployment-time draft-length knob.

## 6. Availability / Release Status

As of the report, the public release is **staged**:

- The `Nemotron-3-Ultra-Base` repo model card describes a **base-only initial release** and states
  "Weights will become available with the full release of Nemotron 3 Ultra, expected to release in 1H 2026."
- The tech report (§1) lists four checkpoints (Base BF16, post-trained BF16, NVFP4, GenRM) plus recipes,
  data, and RL environments as open-sourced on HuggingFace + github.com/NVIDIA-NeMo/Nemotron.

Treat the base checkpoint as the first staged release and the full (post-trained + NVFP4 + GenRM) release
as expected 1H 2026.

## 7. Safety

- SFT retains a ~135K-sample safety blend (~45K English from the Nemotron 3 Super blend + ~15K each in
  German, Spanish, French, Japanese, Italian, Chinese).
- RLVR includes a `safety` environment among its unified domains; an Agentic Safety teacher targets
  indirect prompt-injection robustness; RLHF uses an Ultra-based principle-following GenRM.
- See `paper/safety.md` for the full safety synthesis. Note: the tech report does **not** include a
  dedicated responsible-use / license / over-refusal section, so downstream-use guardrail guidance is
  **not reported** in the report itself.

## 8. Caveats

- Throughput multipliers (5x / 6x / 5.9x) are condition-specific; cite the setting.
- Base BF16 is not aligned; do not present it as an assistant-ready model.
- Repo-card benchmark comparisons (GLM-4.5, Kimi-K2) differ from the tech-report Table 10 comparison set
  (GLM-5.1, Kimi-K2.6, etc.); keep them separate.
- "Full release expected 1H 2026" is from the repo card; the report itself lists checkpoints as
  open-sourced. Reconcile by treating the release as staged.
