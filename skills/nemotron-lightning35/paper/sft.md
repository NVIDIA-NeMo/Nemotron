---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-lightning35"
section: "sft"
paper_sections: ["1", "3", "3.1", "3.6", "6"]
title: "Nemotron 3.5 Lightning Post-Training: Single-Stage Text-Only SFT"
summary: |
  What the Nemotron 3.5 Super and Lightning report says about Lightning's supervised fine-tuning, and
  what the BF16 model card and the repo docs and recipe configs add. The report is brief: Lightning is a
  text-only model whose post-training is a single text-only SFT stage followed by RLVR, with no MOPD. It
  reports no Lightning SFT data sizes, teachers, or hyperparameters. The model card lists SFT data
  categories and three post-training data filters. The repo's SFT page supplies an open-data reference
  recipe (Megatron-Bridge, 4096-token packs, LR 5e-6, GBS 128) on 12 open blend entries.
key_facts:
  - "§1 / §3 / §6: Lightning follows 'a lighter post-training pipeline consisting of single-stage text-only SFT and RLVR'; §6 calls it 'a lighter text-only post-training pipeline optimized for efficient deployment'."
  - "§3.1: 'We train Nemotron 3.5 Super to be a VLM and Nemotron 3.5 Lightning to be a text only model... Nemotron 3.5 Lightning is trained with single SFT stage with the text only SFT data.'"
  - "§3.6: Lightning 'did not have an MOPD stage'. The report also describes no specialist teachers for Lightning (§3.3-§3.5 are Super-only)."
  - "Starting checkpoint: Lightning 3.5 Base, which the report says was initialized from Nemotron 3 Nano Base and distilled for 1T tokens from Super 3 Base, then given long-context training (§1, §2.2)."
  - "Not reported for Lightning SFT: data blend, sample/token counts, teacher models, sequence length, batch size, LR schedule, optimizer, number of steps, whether MTP layers are trained during SFT, reasoning-mode (on/off/medium/budget) preparation."
  - "The 'text only SFT data' is not defined further. The report does not say whether it is the text portion of Super's SFT blend (§3.1: Kimi K3 / DeepSeek V4 Pro / GLM-5.2 reasoning traces, Ultra chat and safety recipe, long-context data) or a separate blend; do not attribute Super's SFT data or config to Lightning."
  - "Repo docs (docs/nemotron/lightning35/sft.md, recipe config): open-data reference recipe using Megatron-Bridge `finetune()` with recipe `nemotron_3_5_lightning_sft_config`; LR 5e-6 (min 0.0, 10 warm-up iters), GBS 128, micro-batch 1, pack size 4096, TP=2 with sequence parallel, EP=8, PP=1, AdamW, per-token loss, role-based loss masking. Default config runs 100 iterations (train_iters: 100)."
  - "Repo recipe open data: 12 blend entries from Nemotron-Instruction-Following-Chat-v1 (chat_if, structured_outputs), Nemotron-Math-Proofs-v1 (lean), Nemotron-SWE-v1, Nemotron-Agentic-v1 (interactive_agent, tool_calling), and Nemotron-Competitive-Programming-v1 (C++ and Python parts 00/01, InfiniByte parts 00/01). Science MCQ/RQA entries are disabled (corrupted parquet), so no science dataset is included. Data prep keeps records whose used_in contains nano_v3 and splits packed output 98/1/1."
  - "BF16 model card: SFT on synthetic code, math, science, tool calling, instruction following, structured outputs, and general knowledge data, plus data for long-range retrieval and multi-document aggregation. Post-training filters: structural checks, pathological-repetition filtering, keyword/regex narrative filtering."
related_steps:
  - "sft/megatron_bridge"
currency: "frozen"
---

# Scope

- Where does SFT sit in Lightning's post-training, and how does it differ from Super's?
- What SFT data and hyperparameters were used for the released Lightning model?
- What does the repo's Lightning SFT recipe do, and how far can it be trusted as a description of the release?

# Pipeline position (report)

| Order | Stage | Source |
| --- | --- | --- |
| 0 | Lightning 3.5 Base: init from Nano 3 Base, 100B-token MTP warm-up (backbone frozen), 1T-token distillation from Super 3 Base, long-context training | §1, §2.1, §2.2 |
| 1 | **Single-stage, text-only SFT** | §3, §3.1 |
| 2 | RLVR | §1, §3 (see rl.md) |
| 3 | MTP boosting, 48k steps, no warm start | §3.7 (see mtp.md) |
| 4 | NVFP4 PTQ/QAD, W4A16 | §4, §4.8 (see quantization.md) |

Not in Lightning's pipeline, according to the report: vision-projector warm-up and second SFT stage (§3.1),
specialist teachers (§3.3-§3.5 describe Super only), MOPD (§3.6: "Nemotron 3.5 Lightning, however did not
have an MOPD stage"), and the 4k-step MTP warm start after SFT (§3.7).

# What the report says about Lightning SFT

The whole Lightning-specific text is one sentence in §3.1:

> "Nemotron 3.5 Lightning is trained with single SFT stage with the text only SFT data."

plus "We train ... Nemotron 3.5 Lightning to be a text only model" (§3.1) and "Lightning goes through a
lighter pipeline, with text-only SFT, and RLVR" (§3).

Everything else in §3.1 (vision and agentic-vision data, Kimi K3 / DeepSeek V4 Pro / GLM-5.2 reasoning
traces, the <5,000-token filter, 524,288-token packs, GBS 32, LR 1e-5, MTP aux loss 0.3) is described for
Super. The report does not say whether any of it applies to Lightning.

# Repo reference recipe (model card / repo docs, not the report)

Source: `docs/nemotron/lightning35/sft.md` and `src/nemotron/recipes/lightning35/stage1_sft/config/default.yaml`.
The docs state that these settings come from the recipe configs, not from the report.

| Setting | Repo value |
| --- | --- |
| Framework | Megatron-Bridge `finetune()`; recipe `nemotron_3_5_lightning_sft_config` (docs say it mirrors the verified packed recipe `nemotron_3_5_lightning_sft_openmathinstruct2_packed_config`) |
| LR | 5e-6, min 0.0, 10 warm-up iterations, decay over 100 iterations |
| Global / micro batch | 128 / 1 |
| Pack size / sequence length | 4096 tokens |
| Parallelism | TP=2 + sequence parallel, EP=8, PP=1; selective recompute (moe, layernorm, core_attn, mlp) |
| Optimizer | AdamW |
| Loss | role-based mask (assistant tokens only), mask rolled by 1, per-token normalization |
| Default length | `train_iters: 100` (a reference config, not the production run) |
| Chat template | Lightning35 Jinja template with `<\|im_start\|>` / `<\|im_end\|>`; reasoning from earlier turns dropped at each new user turn |

Open data blend in the repo config (12 entries): Nemotron-Instruction-Following-Chat-v1 (chat_if,
structured_outputs), Nemotron-Math-Proofs-v1 (lean), Nemotron-SWE-v1, Nemotron-Agentic-v1
(interactive_agent, tool_calling), and Nemotron-Competitive-Programming-v1 (C++, Python, InfiniByte parts
00/01). The science MCQ/RQA entries are disabled in the blend file, so no science dataset is included.
Data prep filters to records whose `used_in` contains `nano_v3` and splits packed output 98% / 1% / 1%.

## Released-model SFT data (BF16 model card, not the report)

- Data categories: synthetic code, math, science, tool calling, instruction following, structured outputs,
  and general knowledge, plus data designed for long-range retrieval and multi-document aggregation.
- Filters: structural checks (e.g. tool calls without tool definitions), pathological-repetition filtering
  of reasoning traces (repeated n-grams), and keyword/regex removal of trajectories aligned with specific
  political entities or nationalistic narratives.
- No sizes, teachers per domain, or hyperparameters are given.

(Earlier docs versions had a 14-domain "SFT Data Domains" table and a "Data Filtering" list copied from the
Nano 3 page, e.g. "Formal Proofs: 300k examples", "Long Context: 128k mean, 256k hard limit". They were
removed; those values are Nano 3's.)

# Contradictions and differences

| Topic | Report | Model card / repo |
| --- | --- | --- |
| SFT stages | One text-only stage (§3.1) | One SFT stage (consistent) |
| SFT data | "text only SFT data", not described further | Model card: data categories and filters; recipe: 12 open blend entries |
| Hyperparameters | Not reported for Lightning | Reference recipe: LR 5e-6, GBS 128, 4096-token packs |
| Sequence length | Not reported for Lightning SFT | 4096-token packs in the recipe; separately, docs/nemotron/lightning35/quantization.md says ablations indicated QAD should run at "the post-training SFT sequence length (roughly 522K tokens)". The two repo statements describe different things (reference recipe vs production SFT), and neither is in the report. |

# Caveats

- The repo recipe is a reference implementation on open data. Its hyperparameters are not claimed to match the production SFT run, and the report gives none for comparison.
- The SFT step catalog lists Lightning only for `sft/megatron_bridge`; no RL, eval or quantization step in `src/nemotron/steps` lists Lightning as a model.
- Never answer a Lightning SFT question with Super's §3.1 numbers.
