---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-super35"
section: "overview"
paper_sections: ["abstract", "1", "6"]
title: "Nemotron 3.5 Super Overview: Abstract, Introduction, Released Checkpoints, Conclusion"
summary: |
  Nemotron 3.5 Super is a 120B-total / 12B-active Mixture-of-Experts hybrid Mamba-Transformer
  multimodal (text, image, video) model initialized from Nemotron 3 Super. Its text-only base is built
  through a 100B-token MTP warm-up, 1T-token distillation from Ultra 3 Base, 100B-token Indic continued
  pretraining, 35B-token 1M-context extension, 65.4B-token agentic SWE mid-training at 256K, and a ~9B-token
  long-context recovery. Post-training (two-stage SFT that adds vision, RLVR + RLHF, specialist teachers,
  MOPD) turns it into a multimodal agent. The same report covers the smaller text-only Nemotron 3.5
  Lightning (30B-A3B); this chunk focuses on Super.
key_facts:
  - "Super: 120B total parameters, 12B active; multimodal (text, images, video). Lightning (same report): 30B total, 3B active, text-only."
  - "Super is initialized from Nemotron 3 Super; Lightning from Nano 3 Base."
  - "MTP: Super consolidates two non-shared MTP heads into a single recursively shared head; 100B-token MTP warm-up with the backbone frozen; hidden-state mixing (Super only)."
  - "Distillation: 1T tokens with Ultra 3 Base as teacher for Super (Super 3 Base for Lightning)."
  - "Indic CPT: 100B tokens (10B Indic + 90B preceding mixture); ~2.83T synthetic output tokens generated across 22 Indian languages / 34 language-script configurations; gains up to 9.82 points on five Indic benchmarks."
  - "1M-context extension: 35B tokens; agentic SWE mid-training: 65.4B tokens at 256K; 9B-token recovery restores RULER-1M from 33.5 to 81.0."
  - "Mid-training effect (identical experimental SFT): DeepSWE pass@1 6.4% -> 10.4%; SWE-bench Verified under Codex harness 48.4% -> 61.8%."
  - "Post-trained Super: 51.78% on SWE-bench Pro, surpassing Nemotron 3 Ultra with ~4.6x fewer active parameters."
  - "Test-time scaling (parallel agents + self-verification): Terminal-Bench 2.1 62.7% -> 71.9%; DeepSWE 24.6% -> 34.5%."
  - "MTP boosting on SPEED-Bench: average acceptance length 4.287 -> 4.718 (greedy, +10.1%) and 3.787 -> 4.612 (probabilistic draft sampling, +21.8%)."
  - "NVFP4: 99.52% median accuracy recovery vs BF16 across 35 benchmarks for the released quantized Super checkpoint."
  - "GB200 serving: 4,632 output tokens/s/GPU at >= 50 tokens/s/user; 2.88x the throughput of Qwen3.8 Flash-Next under the evaluated interactive agentic workload."
  - "Released Super checkpoints: Nemotron 3.5 Super VL 120B-A12B Base BF16, BF16 (post-trained), NVFP4 (post-trained + NVFP4-quantized)."
related_steps: []
currency: "frozen"
---

# Scope
Answers:
- What is Nemotron 3.5 Super (size, modalities, lineage) and how does it differ from Lightning?
- What are the headline claims, and under which exact conditions were they measured?
- What is the stage order of pretraining and post-training at a high level?
- Which checkpoints and artifacts are released?
- How is the report organized?

# Headline Specs (abstract + §1)

| Item | Nemotron 3.5 Super | Nemotron 3.5 Lightning (for disambiguation only) |
|---|---|---|
| Total parameters | 120B | 30B |
| Active parameters | 12B | 3B |
| Modalities | Multimodal: text, images, video, interactive tool environments | Text-only |
| Initialized from | Nemotron 3 Super (NVIDIA, 2026b) | Nano 3 Base (NVIDIA, 2025c) |
| Architecture | MoE hybrid Mamba-Transformer with native MTP | same family |
| Distillation teacher (1T tokens) | Ultra 3 Base | Super 3 Base |
| Extra base-model stages | Indic CPT, 1M long context, agentic SWE mid-training (256K), LC recovery | long-context training only |
| Post-training | Two-stage SFT (adds vision), RLVR + RLHF, specialist teachers, MOPD | Single-stage text-only SFT + RLVR |

Parameter accounting: the 120B / 12B headline is the report's rounding. §4.1 counts N = 124.453 billion parameters for the released NVFP4 checkpoint "including vision, MTP, and unquantized modules" (see architecture.md).

# Pretraining Sequence for Super (§1, §2)

| Order | Stage | Tokens | Context | Chunk |
|---|---|---|---|---|
| 1 | MTP init (first of two Super 3 Base heads) + warm-up, backbone frozen, hidden-state mixing | 100B | — | pretraining.md |
| 2 | Distillation from Ultra 3 Base | 1T | — | pretraining.md |
| 3 | Indic continued pretraining (10B Indic + 90B preceding mixture) | 100B | — | data.md |
| 4 | Long-context phase (LC-Phase) | 35B | 1M | pretraining.md |
| 5 | Agentic software-engineering mid-training | 65.4B | 256K | pretraining.md / data.md |
| 6 | Long-context recovery | 9B (§2.4; "about 9B" in §2.5) | 1M | pretraining.md |

The base is trained on **text only**; vision is introduced in SFT (§3.1).

# Headline Claims with Conditions

| Claim | Value | Exact condition / source |
|---|---|---|
| Indic gains | up to +9.82 points | Indic GSM8K strict (11 languages), before vs after Indic CPT, Table 2; all five Indic benchmarks improve (+4.98 to +9.82) |
| RULER-1M recovery | 33.5 -> 81.0 | After 256K mid-training dropped it from 79.9; 9B-token recovery stage (§2.4) |
| DeepSWE (mid-training effect) | 6.4% -> 10.4% pass@1 | Same experimental ~33B-token (2K-step) agentic SFT on base vs mid-trained checkpoint; not the §3 SFT; averaged over 4 runs (Table 5) |
| SWE-bench Verified, Codex harness | 48.4% -> 61.8% | Same Table 5 setup |
| SWE-bench Pro | 51.78% | Post-trained Super, OpenCode harness per Table 10 (see the evaluation chunk); "surpassing Nemotron 3 Ultra" with ~4.6x fewer active parameters |
| Test-time scaling | Terminal-Bench 2.1 62.7% -> 71.9%; DeepSWE 24.6% -> 34.5% | Parallel agent execution + self-verification (§3.11) |
| MTP boosting | AL 4.287 -> 4.718 greedy; 3.787 -> 4.612 probabilistic draft sampling | SPEED-Bench; conclusion phrases this as +10.1% (greedy) and +21.8% (sampling); abstract says "up to 21.8%" |
| NVFP4 | 99.52% median accuracy recovery | Released quantized Super checkpoint vs BF16, across 35 benchmarks (§4.1: aggressive recipe + QAD, 4.82 bits per element) |
| Serving throughput | 4,632 output tokens/s/GPU | GB200, at >= 50 tokens/s/user; 2.88x Qwen3.8 Flash-Next "under the evaluated interactive agentic workload" (§5 tuning workload: 20 turns, 2,048 input / 1,024 output tokens per turn) |

Figure 1 compares Super (120B-A12B) with six leading open models across agentic, code, math/science, multimodal, long-context, and chat-preference benchmarks; full results are in §3.8 / Table 10.

# Released Checkpoints (§1)

| Checkpoint | Description |
|---|---|
| Nemotron 3.5 Super VL 120B-A12B Base BF16 | base model |
| Nemotron 3.5 Super VL 120B-A12B BF16 | post-trained model |
| Nemotron 3.5 Super VL 120B-A12B NVFP4 | post-trained and NVFP4-quantized model |
| Nemotron 3.5 Lightning 30B-A3B Base BF16 / BF16 / NVFP4 | Lightning equivalents (not Super) |

Also released per abstract/§1/§6: accompanying model cards, pretraining and post-training datasets, cookbooks, and training recipes.

Indic-specific release commitments (§2.3, future tense in the report): the broader synthetic Indic corpus (Nemotron-Pretraining-Indic-v1), open-sourced generation pipelines, a companion technical report ("Scaling Synthetic Pre-training Data and Evaluation for Indic LLMs"), and, with AI4Bharat, a human-verified Indic LLM evaluation suite.

# Post-training (§1, high level)
- Two-stage SFT turns the text-only base into a vision-language model while developing reasoning, chat, safety, long-context, and agentic capabilities (after a vision-projector warm-up, §3.1).
- RL spans diverse environments, including Olympiad-level competitive programming and finance-agent tasks (SEC filing search, document parsing, evidence retrieval, tool use).
- Specialized teachers for video reasoning and tracking, vision tool use, and alignment (plus agentic and reasoning teachers in §3.3), consolidated through MOPD.
- MTP boosting after backbone post-training; NVFP4 PTQ + QAD; inference work on hybrid-state caching, speculative decoding, GPU kernels.

(Post-training, quantization, and inference are owned by other chunks.)

# Report Structure
Pretraining (§2), Post-training (§3), Quantization (§4), Inference (§5), Conclusion (§6).

# Conclusion (§6)
Restates the claims: Indic CPT up to +9.82 points; mid-training improves long-horizon SWE and harness robustness; 51.78% SWE-bench Pro, above Nemotron 3 Ultra with ~4.6x fewer active parameters; test-time scaling Terminal-Bench 2.1 62.7% -> 71.9%; MTP boosting +10.1% (greedy) / +21.8% (probabilistic sampling) acceptance length; NVFP4 99.52% median recovery; 2.88x Qwen3.8 Flash-Next serving throughput on GB200 under the evaluated interactive agentic workload. Releases checkpoints, datasets, recipes, and cookbooks.

# Caveats
- Several numbers in the abstract and intro are Super-only even though the report covers both models; never attribute them to Lightning, and do not attribute Lightning's 30B-A3B figures to Super.
- The DeepSWE 6.4 -> 10.4 and Codex 48.4 -> 61.8 gains come from an experimental SFT probe, not the released model. The post-trained model's §3.11 DeepSWE baseline (before selection) is 24.6%.
- The test-time-scaling baseline for Terminal-Bench 2.1 (62.7%, §3.11) is not the Table 10 Terminal-Bench 2.1 score (Terminus 2 harness). Check the evaluation chunk before mixing them.
- "2.88x" and "4,632 tokens/s/GPU" depend on the GB200 interactive agentic workload and the >= 50 tokens/s/user constraint; do not generalize them.
- "Surpassing Nemotron 3 Ultra" refers to SWE-bench Pro specifically.
- The intro says Super "consolidates two non-shared MTP heads"; §2.1 says Super 3 Base "originally contained two distinct MTP heads". The Nemotron 3 Super report itself describes shared-weight MTP layers. See architecture.md for this and the §3.1 "two shared-weight MTP layers" wording.
- Citation keys are inconsistent: the intro cites Ultra 3 Base as "NVIDIA, 2026" (as do the §3.1 reasoning-data and long-context paragraphs for Nemotron Ultra), while §2.2, the §3 intro, §3.2 and §3.6 cite Nemotron 3 Ultra as "NVIDIA, 2026c".
