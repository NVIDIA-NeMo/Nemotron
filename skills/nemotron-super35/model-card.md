# Nemotron 3.5 Super VL — Model Card

> Release identity and deployment facts for the NVIDIA Nemotron 3.5 Super VL 120B-A12B model family.
> Numbers are drawn from the Nemotron 3.5 Super and Lightning Tech Report (2026). This is a
> knowledge-base summary, not the official Hugging Face model card.

## 1. Identity

| Field | Value |
|---|---|
| Model | Nemotron 3.5 Super VL 120B-A12B |
| Developer | NVIDIA |
| Family | Nemotron 3.5 (with Nemotron 3.5 Lightning, 30B-A3B, covered by the same report) |
| Architecture | Hybrid Mamba-Transformer LatentMoE language backbone + C-RADIO vision encoder + vision-language projector (the report's names). The Megatron-Bridge model provider also has a temporal video embedder, which the report does not describe |
| Total parameters | 120B (124.453B counting the vision encoder, projector, and MTP module, §4.1) |
| Active parameters / token | 12B |
| Initialization | Nemotron 3 Super Base |
| Modalities | Text, image, video in; text out. No audio |
| Context length | Up to 1M tokens |
| MTP | One recursively shared MTP head (speculative decoding) |
| Pretraining | Text-only continued pretraining (MTP warm-up, distillation from Nemotron 3 Ultra Base, Indic CPT, 1M context, agentic mid-training, LC recovery) |
| Post-training | SFT (2 stages, adds vision) → RLVR → specialist teachers → MOPD (2 phases) → MTP boosting |
| Reasoning modes | Reasoning-off, regular, medium-effort; budget control at inference time for regular and medium-effort (§3.9) |
| Repo | github.com/NVIDIA-NeMo/Nemotron (`docs/nemotron/super35/`) |

## 2. Checkpoints

| Checkpoint | Precision | Description | Hugging Face |
|---|---|---|---|
| Nemotron 3.5 Super VL 120B-A12B Base BF16 | BF16 | Text-only base after the full pretraining pipeline | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16` |
| Nemotron 3.5 Super VL 120B-A12B BF16 | BF16 | Post-trained VLM (MOPD phase 2 output + MTP boosting) | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` |
| Nemotron 3.5 Super VL 120B-A12B NVFP4 | NVFP4 | Post-trained, quantized with the aggressive PTQ recipe + QAD | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` |

Not released: intermediate SFT/RLVR/teacher/MOPD checkpoints and the length-penalized GenRM variant used for RL (NeMo-RL's public recipe uses `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-GenRM`).

### NVFP4 deployment

One NVFP4 checkpoint serves two modes: **W4A4** on Blackwell (B200, GB200, GB300) and **W4A16** (NVFP4 weights, 16-bit activations) elsewhere, e.g. H100. The released recipe is the predefined **aggressive** recipe at 4.82 bits per element plus 200 iterations of SFT-based QAD, with **99.52% median accuracy recovery** over 35 benchmarks. Details: `paper/quantization.md`.

## 3. Intended use

- Agentic workflows: software engineering, terminal use, search, tool calling, long-horizon agents.
- Multimodal understanding: charts, documents and OCR, GUI, visual reasoning, video.
- Long-context reasoning up to 1M tokens.
- Multilingual use, strengthened for Indian languages by Indic CPT (evaluated on up to twelve languages; the synthetic corpus covers 22).
- The Base checkpoint is text-only and not instruction-tuned.

## 4. Open data (per the report)

| Data | Status |
|---|---|
| Nemotron-Pretraining-Indic-v1 (~2.83T synthetic Indic tokens, 22 languages, 34 language-script configurations) | Announced |
| Human-verified Indic evaluation suite (with AI4Bharat) | Announced |
| Super 3.5 SFT and RL blends | Not named or released |
| Nemotron-Post-Training-Dataset-v2, Nemotron-RL-Super-Training-Blends | Public; used as seed prompts for MTP boosting (§3.7) |

## 5. Caveats

- The report states some headline numbers inconsistently (throughput multiplier, MTP acceptance length, AA-LCR); see `SKILL.md → Known caveats`.
- Benchmark numbers in the report are for the post-trained BF16 model unless stated; several multimodal benchmarks are internal or "to be open-sourced" (`paper/evaluation.md`).
