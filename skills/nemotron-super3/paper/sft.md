---
paper: "arxiv:2512.20856"
model: "nemotron-super3"
section: "sft"
paper_sections: ["3.1"]
title: "Supervised Fine-Tuning"
summary: |
  Super3 uses a large SFT program with more than 7M examples and a two-stage
  loss designed to preserve long-input/short-output behavior. The SFT mix also
  encodes reasoning-control modes, large-scale tool use, software engineering,
  long-context tasks, and safety behavior before reinforcement learning begins.
key_facts:
  - ">7M SFT samples and roughly 80B post-training tokens in the pipeline figure."
  - "Two-stage SFT loss prevents long responses from dominating optimization."
  - "Three reasoning modes are represented: reasoning-off, regular, and low-effort."
  - "Low-effort and budget-control data are explicit parts of the SFT design."
related_steps:
  - "stage1_sft/default"
  - "stage1_sft/data_prep"
  - "stage2_rl/rlvr"
currency: "frozen"
---

# Scope

Use this file when the question is about:

- how Super3’s SFT stage is designed
- why the loss is two-stage
- how reasoning control is trained
- what domains dominate the SFT mixture
- how the released SFT recipe maps to the paper

---

# What SFT is doing in Super3

Super3 does not rely on RL to create all of its assistant behavior. The report gives SFT a large role in shaping:

- instruction following,
- reasoning behavior,
- tool use,
- software-engineering priors,
- long-context response behavior,
- multilingual and safety behavior.

That is why users asking “when does Super3 become an assistant?” should usually be pointed first to SFT, not directly to RL.

---

# SFT scale

| Item | Reported value |
|---|---|
| Example count | >7M |
| Pipeline-scale token figure | ~80B |
| Main output | aligned pre-RL assistant checkpoint |

The main point is scale: this is not a small polish step on top of pretraining. It is a large, curated post-training program in its own right.

---

# The two-stage SFT loss

## Why a standard SFT loss was not enough

The report says a single token-average SFT objective hurt tasks with **long inputs and short outputs**. In such settings, examples with very long responses can dominate optimization if each token contributes equally.

## Stage 1 loss

Stage 1 is the conventional token-level averaging over assistant/output tokens.

## Stage 2 loss

Stage 2 normalizes at the **conversation level** so that a few very long assistant completions do not overwhelm shorter but important examples.

## Why this matters

| Without two-stage balancing | With two-stage balancing |
|---|---|
| Long outputs dominate the objective | Example-level balance is better preserved |
| Long-input/short-output behavior can regress | These tasks remain better represented |
| SFT favors verbosity more strongly | SFT better matches mixed task formats |

This is one of the most important details to mention when someone asks how Super3 preserves long-context or tool-use behavior after SFT.

---

# Reasoning modes are trained, not just decoded

The report describes three reasoning modes in the SFT data.

| Mode | Meaning |
|---|---|
| Reasoning-off | reasoning traces are removed or disabled |
| Regular reasoning | default thought-rich responses |
| Low-effort reasoning | shorter and cheaper reasoning traces |

This means reasoning control is partly learned through supervision, not only imposed at inference time through a chat-template switch.

---

# Low-effort reasoning

Low-effort mode is a specific SFT design choice rather than a generic decoding trick.

| Detail | Reported value |
|---|---|
| Share of SFT data | ~2% by sample count |
| Generation source | GPT-OSS-120B low-effort mode |
| Training objective | encourage shorter, efficient reasoning trajectories |

This is useful for users who ask whether low-effort reasoning was bolted on after training. The answer is no: it is represented directly in the SFT mixture.

---

# Reasoning-off and budget control

The report also describes explicit supervision for constraining or suppressing reasoning traces.

| Mechanism | Reported detail |
|---|---|
| Reasoning-off data | reasoning stripped from ~3% of samples |
| Semi-on-policy budget-control SFT | 350 training steps |
| Budget truncation rate | 12% of traces receive random truncation |

The design logic is that a deployed reasoning model should not only be able to reason deeply, but also to:

- suppress traces when requested,
- shorten them when latency matters,
- and remain useful after truncation.

---

# Core SFT domains

The SFT mixture is deliberately broad and agent-oriented.

| Domain | Why it matters |
|---|---|
| Competition math | structured reasoning and stepwise problem solving |
| Competition code | algorithmic programming competence |
| Software engineering | repository-level bug fixing and code modification priors |
| Agentic programming | iterative tool-using workflows |
| General tool use | assistant behavior with tool schemas and tool arguments |
| Long context | reliable behavior on long prompts and retrieval-heavy tasks |
| Financial reasoning | structured numerical/analytic tasks |
| CUDA | NVIDIA-relevant technical problem solving |
| Safety | boundaries, refusals, prompt injection, misuse handling |
| Search | browsing-like task decomposition |
| Terminal use | shell-oriented planning and execution behavior |
| SQL | database and table reasoning |
| Multilingual | released language footprint |

The paper’s post-training story is therefore agentic even before RL begins.

---

# Tool-use scale in SFT

The report highlights two large-scale tool-related pipelines.

| Pipeline | Reported scale |
|---|---|
| Specialized tool/customer-service conversations | 279,116 conversations across 838 domains |
| General-purpose tool-calling pipeline | 1.5M trajectories |

These are important because they explain why Super3 already knows how to structure tool interactions before it enters RLVR or SWE-RL.

---

# Safety in SFT

SFT is one of the first places safety enters the training program.

The report and release docs indicate that the SFT mixture includes examples for:

- content safety,
- jailbreak resistance,
- over-safety / over-refusal balancing,
- bias and harmful behavior boundaries,
- prompt injection robustness,
- copyright-sensitive behavior.

That means later RL safety environments build on an already safety-conditioned assistant.

---

# Why SFT comes before RL

The post-training order is intentional.

| Stage | Primary function |
|---|---|
| SFT | establish assistant behavior, formatting, reasoning controls, tool priors |
| RLVR | sharpen performance in verifiable environments |
| SWE-RL | specialize long-horizon software-engineering behavior |
| RLHF | refine preferences, identity, and safety behavior |

SFT gives RL a strong initial policy. Without that base, the RL stages would spend too much capacity learning formatting and role behavior instead of improving task reward.

---

# How the repo maps the paper

The released SFT implementation lives under `src/nemotron/recipes/super3/stage1_sft/`.

## Data preparation path

The open code uses:

- `data_prep.py`
- a Ray pipeline of **SftPlanStage → DownloadStage → PackedSftParquetStage**
- packed Parquet shards with `input_ids`, `loss_mask`, and `seq_start_id`

The data prep config shows the practical translation of the paper’s SFT design into a runnable input format:

| Setting | Released default |
|---|---|
| Tokenizer model | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` |
| Pack size | 4096 |
| Packing algorithm | `first_fit_shuffle` |
| Split ratios | 0.98 / 0.01 / 0.01 |
| Chat template | `super3` |
| Filter tag | `used_in_filter: super_v3` |

## Training path

The main recipe config `stage1_sft/config/default.yaml` sets:

| Setting | Released value |
|---|---|
| Input artifact | `super3-sft-data:latest` |
| Base model artifact | `super3-pretrain-model:latest` |
| Megatron-Bridge recipe target | `nemotron_3_super_sft_config` |
| Packed sequence | enabled |
| PEFT | null (full SFT) |
| Train iterations | 1700 |
| Global batch size | 4 |
| Checkpoint save path | `/nemo_run/super3-sft-model` |

A notable repo detail is that the SFT script optionally converts the final Megatron checkpoint into a Hugging Face checkpoint and logs it as a separate artifact.

---

# Reproduction caveats

1. **The repo gives the SFT mechanics, not the full internal SFT mixture.**
   The data-prep and training pipeline are open, but the exact paper blend is broader than a single public JSON file conveys.

2. **Two-stage-loss rationale is in the paper, not fully encoded in a prose comment in the config.**
   If a user asks “why is it built this way?”, prefer this paper file over the raw YAML.

3. **Reasoning controls are partly data-level.**
   Do not answer as if `enable_thinking=True/False` were the whole story.

---

# Common answer patterns

## “Why did Super3 need a different SFT loss?”

Because plain token averaging over-emphasized long outputs and hurt long-input/short-output tasks. The two-stage loss rebalances optimization at the conversation level.

## “Is low-effort reasoning only a runtime option?”

No. It is represented directly in the SFT data and training objective.

## “Does tool use start in RL?”

No. Large-scale tool-use supervision is already present in SFT; RL mainly sharpens verifiable tool behavior.

---

# Related files

- `data.md`
- `rl/overview.md`
- `rl/rlvr.md`
- `safety.md`
- `../recipes/stage1_sft.md`
