---
paper: "arxiv:2512.20856"
model: "nemotron-super3"
section: "data"
paper_sections: ["2.2.1", "3.1", "3.2.1", "3.2.2", "3.2.3"]
title: "Data: Pretraining, SFT, RL, and Release Caveats"
summary: |
  Super3 uses a broad mixture of pretraining web, code, STEM, multilingual, PDF,
  and synthetic sources; a 7M+-sample SFT blend focused on agentic reasoning and
  tool use; and separate RL datasets for RLVR, SWE-RL, and RLHF. The open release
  exposes major pieces of the data program but not the full internal 25T corpus.
key_facts:
  - "Pretraining spans web, code, math, STEM, multilingual, PDF, and synthetic datasets."
  - "The paper names new synthetic releases for code concepts, algorithms, economics, formal logic, and multiple choice."
  - "SFT exceeds 7M examples and includes specialized tool-use and software-engineering data."
  - "RLVR trains across 21 environments and 37 datasets; SWE-RL and RLHF use separate data paths."
related_steps:
  - "stage0_pretrain/data_prep"
  - "stage1_sft/data_prep"
  - "stage2_rl/data_prep"
currency: "frozen"
---

# Scope

This file is for questions like:

- What kinds of data were used to build Super3?
- Which synthetic datasets are new in this report?
- What domains dominate SFT?
- What RL data is separate from SFT?
- How much of the paper’s data pipeline is open?

---

# Data is a staged program, not one pool

Super3’s data story is easiest to understand by stage.

| Stage | Data role |
|---|---|
| Pretraining | Broad world knowledge, code, math, STEM, multilingual, synthetic reasoning priors |
| SFT | Instruction following, tool use, software engineering, long context, safety, multilingual behavior |
| RLVR | Verifiable-reward environments across 21 domains / 37 datasets |
| SWE-RL | Software-engineering agent rollouts in repository sandboxes |
| RLHF | Human-preference and GenRM-judged comparison data |

This matters because many users ask for “the dataset” singular, but Super3 is trained with different data regimes serving different goals at different stages.

---

# Pretraining data categories

The report describes a large corpus assembled from multiple source families.

## Core categories called out in the paper

| Category family | Role in training |
|---|---|
| Web crawl and filtered crawl | General knowledge and language coverage |
| Synthetic crawl variants | Higher-quality synthetic web-derived text |
| Code | Programming knowledge and code reasoning |
| Nemotron-CC-Code | Higher-quality code-focused continuation data |
| Math | Competition and reasoning-oriented mathematics |
| STEM / academic sources | Technical and scientific knowledge |
| Wikipedia | Factual background and encyclopedic style |
| FinePDFs / FinePDFs-high | Long-form structured documents |
| Multilingual | Support for the released multilingual footprint |
| Crawl++ | Higher-quality curated crawl sources |

The paper’s point is not just diversity, but **quality stratification**. Some source families are explicitly subdivided by quality level and reweighted between phase 1 and phase 2.

---

# Synthetic datasets introduced or highlighted for Super3

The report explicitly names several synthetic training sets that help fill gaps where raw web text is a weak teacher.

| Synthetic set | What it targets |
|---|---|
| Synthetic Code Concepts | programming abstractions, APIs, and implementation concepts |
| Synthetic Algorithmic | algorithmic reasoning and procedural problem solving |
| Synthetic Economics | financially oriented reasoning and structured analysis |
| Synthetic Formal Logic | symbolic logic and proof-like reasoning |
| Synthetic Multiple Choice | structured benchmark-style reasoning and knowledge application |

The released docs also attach some scale cues:

| Synthetic set | Released scale cue |
|---|---|
| Code Concepts | ~15M problem/solution pairs |
| Synthetic Algorithmic | ~0.2B tokens |
| Synthetic Multiple Choice | ~3.5M samples / ~1.6B tokens |

These are useful when users ask whether the paper relies purely on naturally collected text. It does not; Super3 deliberately uses synthetic data to shape reasoning and domain competence.

---

# Representative pretraining blend emphasis

The exact full internal blend is larger than the public release, but the released docs give representative high-level weights for major categories.

## Phase 1 highlights

| Category | Approx. share |
|---|---|
| syn-crawl-high | 22.4% |
| code | 14.0% |
| syn-crawl-medium | 11.3% |
| stem-sft | 11.1% |
| math | 6.4% |
| finepdfs | 6.1% |
| multilingual | 5.0% |

## Phase 2 highlights

| Category | Approx. share |
|---|---|
| syn-crawl-high | 22.4% |
| finepdfs-high | 14.3% |
| code | 14.0% |
| stem-sft | 11.8% |
| crawl-high | 6.5% |
| math | 6.4% |
| multilingual | 5.0% |

The easiest interpretation is:

- phase 1 broadens coverage,
- phase 2 sharpens quality.

---

# Long-context pretraining data

The long-context continuation stages do not abandon the base distribution. The released docs summarize the LC blend as:

- **80%** downscaled phase-2-style mixture,
- **20%** long-context document QA.

That matches the report’s story that long-context ability should be added without destroying short-context competence.

---

# SFT data

## Scale

The report describes the Super3 SFT blend as:

- **more than 7 million samples**,
- on the order of **80B tokens** in the post-training pipeline figure.

## Major SFT domains

| Domain | Why it is included |
|---|---|
| Competition math | mathematical reasoning and solution formatting |
| Competition code | code synthesis and algorithmic problem solving |
| Software engineering | repo-level debugging and patch generation priors |
| Agentic programming | tool use and iterative problem solving |
| General-purpose tool use | assistant behavior in tool-rich environments |
| Long context | behavior over long prompts and retrieval-heavy tasks |
| Financial reasoning | domain competence for structured analysis |
| CUDA | GPU-programming and NVIDIA-specific technical queries |
| Safety | refusal boundaries, harmful content handling, prompt injection |
| Search | browsing and search-style task decomposition |
| Terminal use | shell-centric action planning |
| SQL | data-access and structured query reasoning |
| Multilingual | support for the released language set |

This is one reason the model card positions Super3 for agents and enterprise workflows rather than only generic chat.

---

# Tool-use data in SFT

The report highlights two especially large tool-use pipelines.

| Pipeline | Reported scale |
|---|---|
| Specialized customer-service / structured tool-use pipeline | 279,116 conversations across 838 domains |
| General-purpose tool-calling pipeline | 1.5M trajectories |

These numbers matter because they explain why tool use is not confined to RL. Super3 learns a substantial amount of tool and agent behavior already during SFT.

---

# Reasoning-control data in SFT

The SFT data also encodes reasoning style control rather than a single behavior mode.

| Mode | Data treatment |
|---|---|
| Regular reasoning | standard reasoning-rich responses |
| Low-effort reasoning | shorter budgeted traces, about 2% of SFT by sample count |
| Reasoning-off | reasoning stripped from about 3% of samples |

That means reasoning control is partly a **data design choice**, not just a runtime decoding feature.

---

# RLVR data

The RLVR stage is described as a unified multi-environment program over **21 environments** and **37 datasets**.

## Environment families named in the paper

| RLVR family | Notes |
|---|---|
| Math | includes tool and non-tool variants |
| Code | competition and verifiable coding tasks |
| STEM | science-focused problem solving |
| Instruction following | rubric-based constraint following |
| Safety | over-refusal reduction and jailbreak robustness |
| Long context | long-context reasoning environment reused from Nano |
| Agentic tool use | conversational tool use and terminal use |
| Reasoning Gym | diverse reasoning task suite |

A key detail: prompts the SFT model already solves reliably are filtered out, and the remaining tasks are curriculum-ordered by difficulty.

---

# SWE-RL data

SWE-RL is separated because the data is not just a static prompt/answer corpus. Each rollout is built from:

1. a repository task instance,
2. a sandboxed environment image,
3. an agent loop that edits files and runs commands,
4. a reward produced by ground-truth test execution.

That makes SWE-RL data fundamentally different from both SFT data and RLVR JSONL prompts. It is better thought of as **interactive environment data** than as a fixed preference dataset.

---

# RLHF data

The RLHF stage uses pairwise comparison and principle-following reward modeling.

The report and recipe docs describe the GenRM as being trained from sources including:

- HelpSteer 3,
- commercially friendly arena-preference subsets,
- newer human preference data,
- principle-following prompts used during comparison.

This is why the final RLHF stage is more about behavioral preference shaping than factual capability acquisition.

---

# What is open vs not open

This distinction is critical.

## Publicly exposed or referenced in the release

- major portions of the **Nemotron pre-training datasets** collection,
- major portions of the **Nemotron post-training v3** collection,
- RL environments and datasets through **NeMo Gym**,
- open repo data-prep code for pretrain, SFT, and RL stages.

## Not fully public in the same way

- the complete internal 25T pretraining mixture,
- all internal-quality slices and proprietary filtering artifacts,
- exact internal benchmark-tuning subsets used for final paper parity.

The docs therefore recommend treating the released data/recipes as strong references, not as a guarantee of exact score reproduction.

---

# How the repo represents data by stage

| Stage | Main data-prep entrypoint | Output form |
|---|---|---|
| Pretrain | `src/nemotron/recipes/super3/stage0_pretrain/data_prep.py` | Megatron `bin/idx` shards + `blend.json` |
| SFT | `src/nemotron/recipes/super3/stage1_sft/data_prep.py` | packed Parquet splits + `blend.json` |
| RL | `src/nemotron/recipes/super3/stage2_rl/data_prep.py` | resolved JSONL train/val splits |

That makes it easy to answer “where is the data pipeline implemented?” with precise file paths instead of generalities.

---

# Common answer patterns

## “Is Super3 mostly trained on synthetic data?”

No. Synthetic datasets are important, especially for reasoning and domain shaping, but they sit alongside large web, code, math, PDF, and multilingual corpora.

## “Does SFT teach tool use or only RL?”

Both. SFT already contains large-scale tool-use data; RL then sharpens behavior in verifiable environments.

## “Can I fully reproduce the paper data mix?”

Not exactly from the open release. You can reproduce the methodology and stage boundaries, but not the full internal corpus composition.

---

# Related files

- `pretraining.md`
- `sft.md`
- `rl/rlvr.md`
- `rl/swe.md`
- `rl/rlhf.md`
- `../recipes/stage0_pretrain.md`
- `../recipes/stage1_sft.md`
- `../recipes/stage2_rl.md`
