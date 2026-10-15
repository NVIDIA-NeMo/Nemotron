# Nemotron 3.5 Lightning — Model Card

> Release identity and deployment facts for NVIDIA Nemotron 3.5 Lightning 30B-A3B. Drawn from the
> Nemotron 3.5 Super and Lightning Tech Report (2026), the Hugging Face model cards, and the repo docs
> (labelled). This is a knowledge-base summary, not the official model card.

## 1. Identity

| Field | Value | Source |
|---|---|---|
| Model | Nemotron 3.5 Lightning 30B-A3B | report §1 |
| Developer | NVIDIA | |
| Parameters | 30B total / 3B active | report §1 |
| Architecture | Hybrid Mamba-Transformer MoE with MTP | report §1, repo docs |
| Modalities | Text only | report §3.1 |
| Initialization | Nemotron 3 Nano Base, then 100B-token MTP warm-up and 1T-token distillation from Super 3 Base | report §1, §2.1, §2.2 |
| Base pretraining (model card version) | "pre-trained with over 20T tokens" using "an NVFP4 recipe", MTP continued pretraining, "no ... distillation" | Base model card; **conflicts with the report**, see §5 |
| Layers / hidden | 52 / 2688 | repo docs (hidden 2688 also in report §2.1) |
| Experts | 128 routed (top-6) + 1 shared | checkpoint `config.json` via repo docs; 128 routed also in report §2.1 |
| Context | Up to 1M tokens | model cards (not stated for Lightning in the report, though Table 21 evaluates it on RULER-v1 at 1M) |
| Languages (post-trained) | English, Spanish, French, German, Italian, Japanese | BF16 model card |
| Reasoning control | on/off via chat template (`enable_thinking`) | BF16 model card |
| Post-training | Single-stage text-only SFT → RLVR → MTP boosting (48K steps) | report §1, §3, §3.7 |
| License | OpenMDW-1.1 | repo docs |

## 2. Checkpoints

| Checkpoint | Hugging Face |
|---|---|
| Base BF16 | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16` |
| Post-trained BF16 | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` |
| Post-trained NVFP4 | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` |

## 3. NVFP4 deployment

- **W4A16 on every GPU SKU** (report §4, §4.8): experts, shared experts, and output head in 4-bit NVFP4 weights with 16-bit activations; attention projections BF16.
- **Mamba in/out projections FP8** (report §4.8). The NVFP4 model card confirms: "FP8 per-tensor dynamic scales on mamba in_proj/out_proj and KV cache". The ModelOpt recipe in the repo docs (`four_over_six`) puts them in W4A16 NVFP4 instead (see `paper/quantization.md`).
- **Calibration**: the NVFP4 model card names "Four Over Six NVFP4 (a variant of static MSE calibration) W4A16 on routed and shared experts", 1000 samples of a Nemotron Ultra validation subset at 32K tokens. The report names no Lightning calibration method.
- MTP module not quantized (report §4.7).
- **99.24% median accuracy recovery** vs BF16, verbosity −3.63% (Table 21).
- Checkpoint size: 66 GB BF16 → 22 GB NVFP4 (repo docs; matches the HF safetensors sizes). Neither the report nor the model cards give a Lightning throughput multiplier.

## 4. Intended use

Agentic and reasoning workloads where efficiency matters. The report says Lightning is designed for "local, agentic use cases as well as datacenter deployments" (§4) and notes the FP32 SSM cache is not a limiter on DGX Spark at low batch (§4.8); the repo adds DGX Station recipes (`usage-cookbook/Nemotron-3.5-Lightning/`).

## 5. Caveats

- The joint report has no Lightning comparison against other models; post-trained Lightning numbers in the report are Table 21 (BF16 vs NVFP4).
- **Base lineage conflict.** The report: Nano 3 Base + MTP warm-up + 1T-token distillation from Super 3 Base + long context. The Base model card: >20T tokens with an NVFP4 recipe, MTP continued pretraining, and "no supervised fine-tuning, reinforcement learning, or distillation has been applied". The repo docs follow the report and note the card. Give both and attribute each.
- Earlier repo docs versions contained Nano 3 text (23.5T + 1.5T from scratch, GenRM/DPO, "up to 3.3x"); it was removed. See `SKILL.md → Known caveats`.
