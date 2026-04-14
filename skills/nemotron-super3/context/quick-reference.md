# Nemotron 3 Super Quick Reference

Use this file for fast recall.

---

## 1. Identity

- **Model**: Nemotron 3 Super 120B-A12B
- **Developer**: NVIDIA
- **Release date**: March 11, 2026
- **Format family**: Base BF16, aligned BF16, FP8, NVFP4
- **Active / total params**: 12.7B active / 120.6B total
- **Context length**: up to 1M tokens
- **Supported languages**: EN, FR, DE, IT, JA, ES, ZH
- **Best for**: agentic workflows, high-volume workloads, long-context reasoning, tool use, RAG

---

## 2. Architecture

- Hybrid **Mamba-2 + attention + LatentMoE**
- **88 layers**
- **4096 hidden size**
- **32 Q heads / 2 KV heads**
- **512 experts per layer**
- **top-k = 22**
- **MoE latent size = 1024**
- **2 MTP layers with shared weights**

### Why LatentMoE matters

- routes tokens in a lower-dimensional latent space
- cuts routed parameter bandwidth and all-to-all communication
- reinvests savings into more experts and more active experts
- target: better **accuracy per byte** and **accuracy per FLOP**

### Why MTP matters

- improves training signal
- enables native speculative decoding
- shared-weight head design supports longer recursive drafting
- reported overall SPEED-Bench average acceptance length: **3.45**

---

## 3. Headline claims

- first Nemotron 3 model to:
  - pretrain in **NVFP4**
  - use **LatentMoE**
  - include **MTP** layers
- pretrained on **25T tokens**
- post-trained with **SFT + RL**
- quantized for deployment to **FP8** and **NVFP4**

### Throughput framing from the report

- up to **2.2×** higher throughput than GPT-OSS-120B *(long-generation workloads; see paper for measurement conditions)*
- up to **7.5×** higher throughput than Qwen3.5-122B-A10B *(long-generation workloads; see paper for measurement conditions)*

---

## 4. Pretraining

### Core schedule

- **Phase 1**: 20T tokens (80%)
  - emphasis: diversity and broad coverage
- **Phase 2**: 5T tokens (20%)
  - emphasis: high-quality sources and refined benchmark quality

### Hyperparameters

- sequence length: **8192**
- batch size: **3072 sequences**
- tokens per batch: ~**25.17M**
- optimizer: **AdamW**
- betas: **0.9 / 0.95**
- weight decay: **0.1**
- peak LR: **4.5e-4**
- min LR: **4.5e-6**
- schedule: **Warmup-Stable-Decay**
- warmup: **200B tokens**
- final decay: **5T tokens**
- MTP loss scaling: **0.3**

### NVFP4 training recipe

Default:

- most linear layers in **NVFP4**

Kept higher precision:

- final 15% of the network → **BF16**
- latent projections → **BF16**
- MTP layers → **BF16**
- QKV and attention projections → **BF16**
- Mamba output projection → **MXFP8**
- embeddings → **BF16**

### Checkpoint merging

- merge windows evaluated: **125B / 250B / 500B**
- average improvement during stable-LR phase: **2–4 points** over raw checkpoints
- estimated compute saved vs repeated decay-readout runs: ~**16%**
- final base checkpoint chosen for alignment: **500B merge**

---

## 5. Long-context extension

Two-stage LC phase after main pretraining:

### LC Stage 1

- context length: **1,048,576**
- duration: **34B tokens**
- LR: **4.5e-6 constant**
- global batch size: **16**
- parallelism: **CP=64, TP=2, EP=64**
- hardware: **GB200**

### LC Stage 2

- alternating **1M** and **4K** sequences
- duration: **17B tokens**
- purpose: recover small math/regression hit after pure-1M CPT

### LC data blend

- **20%** long-context document QA
- **80%** downscaled Phase 2 blend

---

## 6. Pretraining data

### New synthetic sets called out in the report

- Synthetic Code Concepts
- Synthetic Algorithmic
- Synthetic Economics
- Synthetic Formal Logic
- Synthetic Multiple Choice

### Notable released scales

- Code concepts: ~**15M** problem-solution pairs
- Synthetic algorithmic: ~**0.2B tokens**
- Synthetic MCQ: ~**3.5M** samples / ~**1.6B tokens**

### Phase 1 blend highlights

- syn-crawl-high: **22.4%**
- code: **14.0%**
- syn-crawl-medium: **11.3%**
- stem-sft: **11.1%**
- finepdfs: **6.1%**
- math: **6.4%**
- multilingual: **5.0%**

### Phase 2 blend highlights

- syn-crawl-high: **22.4%**
- finepdfs-high: **14.3%**
- code: **14.0%**
- stem-sft: **11.8%**
- math: **6.4%**
- crawl-high: **6.5%**
- multilingual: **5.0%**

### Open-data caveat

The released/open subset covers only part of the internal 25T blend. The docs explicitly say the recipes should be treated as **reference implementations**, not exact benchmark-matching reproductions.

---

## 7. SFT

### Scale and purpose

- over **7M samples**
- about **80B tokens** in the post-training pipeline figure
- stronger emphasis on agentic data than Nano

### Two-stage loss

- **Stage 1**: token-level average across assistant/output tokens
- **Stage 2**: per-conversation normalization to avoid long outputs dominating the loss

This was introduced because one-stage SFT hurt long-input / short-output behavior.

### Reasoning modes

- reasoning-off
- regular reasoning
- low-effort reasoning

Low-effort mode:

- introduced in SFT
- low-effort samples are about **2%** of SFT by sample count
- generated using **GPT-OSS-120B** low-effort mode

Reasoning-off and budget control:

- reasoning traces stripped from **3%** of samples
- semi-on-policy budget-control SFT stage: **350 steps**
- truncates **12%** of reasoning traces to random budgets

### Domain highlights

- competition math
- competition code
- software engineering
- agentic programming
- general-purpose tool use
- long context
- financial reasoning
- CUDA
- safety
- search
- terminal use
- SQL
- multilingual

### Tool-use scale-up

- specialized customer-service/tool-use pipeline yielded **279,116 conversations across 838 domains**
- general-purpose tool-calling pipeline yielded **1.5M trajectories**

---

## 8. RL pipeline

The paper’s stage order:

1. **RLVR** — multi-environment RL from verifiable rewards
2. **SWE-RL** — software engineering RL
3. **RLHF** — principle-following GenRM alignment
4. **MTP healing** — train MTP heads while backbone is frozen

### RLVR

- 21 environments
- 37 datasets
- domains: math, code, STEM, safety, chat, IF, long context, puzzles, agentic tasks
- low-effort prompts start at **2%**, later reduced to **1%**
- training setup: async **GRPO**

Released-recipe operating point:

- nodes: **109**
- prompts/step: **256**
- generations/prompt: **16**
- batch size: **4096**
- max sequence length: **65,536**
- TP=4, CP=8, EP=8

### SWE-RL

Why separate?

- much longer rollouts
- much larger contexts
- slower environments
- needs sandbox isolation

#### SWE 1

- SWE-pivot style
- nodes: **64**
- max sequence length: **131,072**
- prompts/step: **64**
- generations/prompt: **16**

#### SWE 2

- full SWE-bench/OpenHands loop
- nodes: **64**
- max sequence length: **196,608**
- prompts/step: **16**
- generations/prompt: **32**
- agent max turns: **200**
- agent concurrency: **768**
- timeout: **3600s**

### RLHF

- principle-following **GenRM**
- init: **Qwen3-235B-A22B-Thinking-2507**
- data: HelpSteer 3 + commercial-friendly Arena subsets + newer human preferences
- KL penalty: **1e-4**
- nodes: **72**
- max sequence length: **49,152**

### MTP healing

- final stage after RLHF
- freezes backbone
- retrains MTP heads on RLVR prompts/responses

---

## 9. Safety

Safety appears in multiple places:

### SFT

- dedicated safety data domain
- includes content safety, jailbreaks, over-safety, bias, prompt injection, copyright

### RLVR

- over-refusal reduction environment
- jailbreak robustness environment
- iterative PAIR-style adversarial prompt generation

### RLHF

- principle-following GenRM
- explicitly used to guide identity and safety behavior

### Model-card guidance

- do not remove safety guardrails without equivalent replacements
- downstream developers must validate domain fitness and misuse risk

---

## 10. Evaluation

### Base-model highlights

- MMLU: **86.01**
- MMLU-Pro: **75.65**
- AIME 2024 pass@32: **53.33**
- RULER 1M: **71.00**

### Post-trained BF16 highlights

- MMLU-Pro: **83.73**
- AIME25: **90.21**
- HMMT with tools: **94.73**
- SWE-Bench OpenHands: **60.47**
- BIRD Bench: **41.80**
- RULER 1M: **91.64** (paper) / **91.75** (model card rounding)

### Quantized results framing

- FP8 and NVFP4 stay close to BF16 on the paper’s evaluation suite
- NVFP4 is the Blackwell-first deployment target

### Open evaluation stack

- **NeMo Evaluator SDK**
- mostly **NeMo Skills Harness**
- some benchmarks still depend on dedicated/open official harnesses

---

## 11. Quantization

### FP8

- calibration: **256** SFT samples at **65,536** context
- quantizes MoE GEMMs and Mamba GEMMs
- KV cache in FP8
- Mamba state cache in FP16

### NVFP4

- deployment target: Blackwell
- improved PTQ recipe + AutoQuantize
- candidate precisions for searched ops: **{NVFP4, FP8, BF16}**
- effective precision budget: **4.75 bits**
- search run: under **2 hours** on **1 B200 node / 8 GPUs**
- eval calibration: **512** samples from Super3 SFT data
- reported quality: **99.8% median accuracy vs BF16**

### QAD

- teacher: BF16 checkpoint
- student: NVFP4 checkpoint
- data: SFT + RL rollouts

### Mamba state quantization

- direct FP16 storage increases verbosity
- selected recipe: **FP16 + stochastic rounding**
- chosen PRNG setting: **Philox<5>**

---

## 12. Released recipe structure

Top-level repo stages:

1. `stage0_pretrain`
2. `stage1_sft`
3. `stage2_rl`
4. `stage3_eval`

Important released configs:

- pretrain: `phase1`, `phase2`, `long_context_1m`, `long_context_mixed`
- SFT: `default`, `tiny`
- RLVR: `default`, `small`
- SWE1: `default`, `small`
- SWE2: `default`
- RLHF: `default`, `small`
- eval: `default`

---

## 13. Best next file

| Need | Open |
|---|---|
| fast orientation | `../INDEX.md` |
| authoritative release facts | `../model-card.md` |
| architecture deep dive | `../paper/architecture.md` |
| training methodology | `../paper/pretraining.md`, `../paper/sft.md`, `../paper/rl/overview.md` |
| runnable commands | `../recipes/overview.md` |

