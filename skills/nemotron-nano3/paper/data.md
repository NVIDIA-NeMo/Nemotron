---
paper: "arxiv:2512.20848"
model: "nemotron-nano3"
section: "data"
paper_sections: ["2.2", "2.2.1", "2.2.2", "2.2.3", "2.2.4", "3.1.2", "Appendix D"]
title: "Pretraining and Post-Training Data"
summary: |
  The paper says Nano3 adds four major new pretraining collections over Nemotron 2: Common Crawl code, refreshed GitHub/code data with synthetic augmentation, Common Crawl v2.1 with 2.1T rephrased tokens and translated-to-English augmentation, and specialized synthetic data for STEM, textbooks, scientific coding, InfiniByte cross-domain code, and reasoning QA. Post-training then adds a broad SFT mixture with competition math/code, tool use, long-context, multilingual, theorem proving, science, safety, SWE, and instruction-following data, plus RL datasets spanning math, coding, structured outputs, instruction following, long context, and agentic tool use.
key_facts:
  - "Nemotron-CC-Code-v1 is a 427.92B-token Common Crawl code corpus built with Lynx rendering, Phi-4 cleaning, and code relevance filtering."
  - "Nemotron-CC-v2.1 adds three recent Common Crawl snapshots, 2.1T rephrased medium-high-quality tokens, and translated-to-English augmentation from 9 languages."
  - "The paper says over 2.5T new tokens were curated or generated from Common Crawl data."
  - "The Specialized-v1 collection includes synthetic Wikipedia, synthetic math textbook text, scientific coding data, InfiniByte cross-domain code, and RQA/DQA STEM reasoning data."
  - "The SFT mixture spans competition math/code, tool use, long-context data, Lean proofs, multilingual data, terminal use, general chat, instruction following, safety, SWE, science, GenSelect, and CUDA pairs."
related_steps:
  - "curate/nemo_curator"
  - "prep/sft_packing"
  - "translate/nemo_skills"
  - "sft/megatron_bridge"
currency: "frozen"
---

# Pretraining and Post-Training Data

## Scope

This file is the data-centric chunk for Nano3.
It combines:

- the four major new pretraining collections in §2.2
- the high-level pretraining mixture context from §2.3
- the rich SFT-domain inventory in §3.1.2
- the RLHF safety-pair construction in Appendix D

## High-level picture

Nano3’s data story has three layers:

1. **new pretraining corpora** added over Nemotron 2
2. **data-mixture design** that balances quality and diversity
3. **post-training mixtures** that shift toward agentic, reasoning, tool-use, and safety behavior

## New pretraining collections released with Nano3

The paper says the vast majority of the new data is released on Hugging Face and grouped into four main datasets.

| Dataset | Role |
|---|---|
| Nemotron-CC-Code-v1 | Common Crawl code extraction |
| Nemotron-Pretraining-Code-v2 | refreshed GitHub/code plus synthetic augmentation |
| Nemotron-CC-v2.1 | newer Common Crawl + rephrasing + translation-to-English augmentation |
| Nemotron-Pretraining-Specialized-v1 | specialized synthetic content for STEM, textbooks, scientific coding, and reasoning |

## 2.2.1 Nemotron-CC-Code-v1

### Pipeline summary

The paper describes this dataset as a code-focused adaptation of the Nemotron-CC-Math pipeline.
The processing stages are explicit:

1. identify Common Crawl pages that contain code
2. render raw HTML through **Lynx** to preserve formatting
3. clean with an LLM stage using **Phi-4**
4. preserve code, config blocks, API references, and math expressions
5. filter with a lightweight relevance classifier to keep programming-heavy pages

### Key properties

| Property | Value / note |
|---|---|
| corpus size | 427.92B tokens |
| render stage | Lynx |
| cleaning model | Phi-4 |
| quality filter | programming relevance classifier |
| formatting behavior | equations standardized to LaTeX; code layout preserved |

### Why it matters

The paper explicitly claims this pipeline is better than prior extraction approaches that often corrupt or truncate code examples.
The goal is not just more code tokens, but cleaner technical context around the code.

## 2.2.2 Nemotron-Pretraining-Code-v2

### Data sources

This collection extends existing code corpora with:

- additional GitHub repositories missing from the prior corpus
- recent data up to **April 15, 2025**
- deduplicated raw source code
- synthetic mixed natural-language / source-code documents

### Synthetic augmentation methods called out in the paper

The report names several synthetic code augmentations:

- question-answer generation grounded in new source code
- student-teacher dialogue for Python
- code-review dialogue for Python and C++
- code rewriting using SGCR and SCOR-style prompts
- Python-to-C++ transpilation augmentation

### Quality controls

| Control | Purpose |
|---|---|
| exact and near dedup | avoid duplication against the prior corpus |
| syntax checks | validate rewritten code |
| Pylint checks | assess quality of rewritten Python |
| transpilation augmentation | improve downstream C++ code-generation accuracy |

### Key teacher model

- Qwen3 32B is the teacher named for several of these synthetic code transformations

## 2.2.3 Nemotron-CC-v2.1

This is the general English web-crawl refresh.

### What changed

The paper says NVIDIA added:

- three newer Common Crawl snapshots:
  - `CC-MAIN-2025-18`
  - `CC-MAIN-2025-21`
  - `CC-MAIN-2025-26`
- synthetic rephrasing on medium-high-quality data across **110** Common Crawl snapshots
- translated-to-English augmentation from nine languages

### Quality buckets kept for training

The paper says Nano3 trains only on:

- Medium-Quality
- Medium-High-Quality
- High-Quality

### New token counts and transformations

| Transformation | Reported value |
|---|---:|
| rephrased medium-high-quality Common Crawl tokens | 2.1T |
| total newly curated/generated Common Crawl tokens | 2.5T+ |

### Translation-to-English augmentation

The translated-source pipeline used:

- Qwen3-30B-A3B for translation
- three recent Common Crawl snapshots at the time of translation
- nine source languages:
  - Chinese
  - French
  - German
  - Italian
  - Japanese
  - Polish
  - Portuguese
  - Russian
  - Spanish

### Extra filtering note

The paper says that after training had already started, some low-value translated documents still scored highly under the quality classifiers.
For the released dataset version, NVIDIA applied an additional LLM-based quality filter that removed about **10.6%** of tokens.

## 2.2.4 Nemotron-Pretraining-Specialized-v1

This collection is a bundle of specialized synthetic subsets.

### Synthetic Wikipedia data

The pipeline:

- revises English Wikipedia articles with Qwen3-30B-A3B-Instruct-2507
- removes disambiguation and redirect pages
- drops sections like References, See also, Notes, and External Links
- removes irrelevant or dirty HTML artifacts

### Synthetic math textbook data

The pipeline:

- starts from Nemotron-CC-Math
- classifies mathematical documents by educational level
- keeps undergraduate-level and above
- rewrites them into textbook-style sections with definitions and examples

### Synthetic scientific coding data

The pipeline builds two formats:

1. code-embedded research-style articles
2. advanced computational coding problems with Python solutions

The paper says the problems are decomposed into **5 to 15** logically ordered non-trivial substeps, each represented as a function.

### Synthetic cross-domain code: InfiniByte

InfiniByte is one of the most distinctive data-generation ideas in the paper.
It creates new programming problems by combining concepts from multiple domains.

Seed materials include:

- competitive coding
- math
- physics
- chemistry
- other sciences

The paper says it uses two cross-breeding strategies:

1. **obfuscate** — keep the underlying problem but alter the presentation
2. **complicate** — make the problem materially more complex by requiring reasoning across domains

### Synthetic STEM reasoning: RQA and DQA

This is another major contribution.

#### Seed selection

The pipeline starts from the STEM subset of Essential-Web and filters for documents that are:

- undergraduate or graduate level
- free of extraction artifacts
- technically correct
- reasoning-rich
- aligned with Bloom’s higher-order processes
- English
- over 1000 characters

That leaves about **14M** seed documents.

#### Stratified selection

The paper says documents are stratified using the Free Decimal Correspondence taxonomy to preserve topic diversity across scales.

#### RQA generation

| Item | Value |
|---|---:|
| generated seed-doc question pool | first 9M samples |
| chosen training subset | first 4.5M |
| final RQA examples | 4.3M |
| total unique RQA tokens | about 31.7B |
| teacher model | Qwen3-235B-A22B-Thinking-2507 |
| reasoning limit during question creation | 8192 reasoning tokens |

#### DQA generation

| Item | Value |
|---|---:|
| seed documents used | first 9M |
| total DQA tokens | about 8B |
| teacher mentioned | Qwen3-30B-A3B |

### SFT-style data inside pretraining

The paper also says Nano3 continues the Nemotron practice of mixing some SFT-style data into pretraining, especially for:

- math
- code
- STEM

Additional math and code SFT samples are added from **AceReason-Nemotron-1.1**, with DeepSeek-R1 used for responses.

## Pretraining mixture notes from §2.3

### Fifteen categories

The pretraining corpus spans **15** categories.
The largest category is web crawl, subdivided into:

- crawl-medium
- crawl-medium-high
- syn-crawl-medium-high
- crawl-high
- syn-crawl-high

Additional named groups include:

- math
- Wikipedia
- code
- nemotron-cc-code
- academic text
- Crawl
- multilingual data
- general-sft
- stem-sft
- code-sft

### Multilingual pretraining languages

The multilingual pretraining pool covers **19** languages:

- Arabic
- Chinese
- Czech
- Danish
- Dutch
- Finnish
- French
- German
- Hebrew
- Hindi
- Italian
- Japanese
- Korean
- Portuguese
- Polish
- Russian
- Spanish
- Swedish
- Thai

## Post-training SFT domains from §3.1.2

The SFT mixture is intentionally much more agentic and interaction-heavy than the pretraining mixture.

### Domains explicitly named in the paper

| Domain | Key idea |
|---|---|
| Competition Math | refreshed with GPT-OSS 120B responses; tool-integrated Python reasoning |
| Competition Code | Nemotron Nano 2 prompt base + DeepSeek-R1-0528 responses |
| Conversational Tool Use | synthetic multi-turn agent trajectories judged for consistency |
| Long Context | synthetic data with mean length 128k and max 256k |
| Formal Proofs | Lean 4 theorem/proof traces |
| Multilingual | translated post-training data into FR/ES/IT/DE/JA |
| Terminal Use | verifiable terminal tasks from Terminal Bench and related sources |
| General Chat | LMSYS + WildChat response generation and multi-turn extension |
| Instruction Following | Tülu 3-style methodology with verifier filtering |
| Safety | safe/unsafe prompts and refusal shaping |
| Software Engineering | SWE-Gym and R2E-Gym issue-solving traces |
| Science | synthetic + real + document-retrieved question generation |
| GenSelect | candidate-solution selection reasoning |
| CUDA | verified PyTorch ↔ CUDA-C pairs |

### Concrete SFT data facts called out in the paper

| Fact | Value |
|---|---|
| long-context SFT data mean length | 128k tokens |
| long-context SFT max length | 256k tokens |
| Lean source natural-language theorems | 580k |
| Lean statements produced | 550k |
| proof traces generated | 920k |
| final filtered proof dataset | 300k |
| CUDA pairs | 21k |

### Multilingual SFT languages

The paper names five target languages for post-training translation data:

- French
- Spanish
- Italian
- German
- Japanese

## Unified SFT filtering rules from §3.1.3

Across domains, the paper says the SFT mixture is filtered for:

- structural correctness
- license compliance
- verifiability
- repetition/pathological traces
- politically or nationalistically slanted synthetic artifacts

Specific examples include:

- missing tool-definition rejection when tool calls appear
- repeated n-gram filtering
- regex/keyword filters for “our nation/party” style narratives

## Safety preference data in Appendix D

Appendix D explains how RLHF preference pairs are built.

### Harmful prompts

Chosen responses:

- safe responses generated with the desired strategy

Rejected responses:

- unsafe outputs from jailbreak-style prompting
- unsafe outputs surfaced by direct prompting and moderation filtering

### Safe prompts

Chosen responses:

- safe non-refusal responses

Rejected responses:

- over-refusals created by refusal-template prompting

### Pairing policy

The paper frames the final pair construction as:

- harmful prompt → `<safe, unsafe>` as `<chosen, rejected>`
- safe prompt → `<safe, over-refusal>` as `<chosen, rejected>`

The paper also says five open models are used to generate candidate chosen/rejected responses, then one pair per prompt is sampled after filtering.

## How to answer data questions safely

### “What new pretraining data was added?”

Use the four main §2.2 collections first.

### “What’s in the SFT mixture?”

List the major domains from §3.1.2, especially:

- math
- code
- tool use
- long context
- formal proofs
- multilingual
- terminal use
- instruction following
- safety
- SWE
- science

### “What safety data was used?”

Split the answer:

- SFT safety prompts/refusal shaping in §3.1.2
- RLHF preference-pair construction in Appendix D

## Cross-links

- `pretraining.md` for two-phase schedule and optimizer framing
- `sft.md` for SFT mixture size, reasoning control, and hyperparameters
- `rl.md` for RL task families and curriculum
- `safety.md` for the alignment-specific data interpretation
