# Nemotron 3 Nano Quick Reference

Use this when you need a compact answer or a `/nemotron-customize` handoff.

## 1. Model Identity

| Item | Value |
|---|---|
| Family | NVIDIA Nemotron |
| Model | Nemotron 3 Nano 30B-A3B |
| Developer | NVIDIA |
| Architecture | Hybrid Mamba-Transformer with sparse MoE |
| Total params | 31.6B |
| Active params | 3.2B per forward pass; 3.6B including embeddings |
| Pretraining tokens | 25T |
| Public context claim | up to 1M tokens |
| HF default context | 256k in public config examples |
| License | NVIDIA Nemotron Open Model License |

## 2. Public Checkpoints

| Checkpoint | Purpose |
|---|---|
| `NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` | base pretrained model |
| `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | post-trained model |
| `NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` | quantized post-trained deployment model |

## 3. Architecture at a Glance

| Component | Value |
|---|---:|
| Layers | 52 |
| Model dimension | 2688 |
| Q heads | 32 |
| KV heads | 2 |
| Head dimension | 128 |
| Mamba state dimension | 128 |
| Mamba groups | 8 |
| Mamba heads | 64 |
| Mamba head dimension | 64 |
| Expert dimension | 1856 |
| Total routable experts | 128 |
| Activated experts | 6 |
| Shared experts | 2 |

## 4. Pretraining Summary

- Warmup-Stable-Decay schedule
- 25T tokens total
- two phases:
  - Phase 1: 23.5T tokens, diversity-oriented
  - Phase 2: 1.5T tokens, higher-quality emphasis
- phase switch at 94% of training
- sequence length 8192
- batch size 3072 sequences
- fifteen pretraining categories
- nineteen natural languages in multilingual data
- long-context CPT added at the end
- long-context phase uses 121B tokens

## 5. New Data Families Added Over Nemotron 2

| Dataset | What it adds |
|---|---|
| Nemotron-CC-Code-v1 | Common Crawl code pages cleaned with Lynx + LLM pipeline |
| Nemotron-Pretraining-Code-v2 | refreshed GitHub/code corpus plus synthetic code |
| Nemotron-CC-v2.1 | newer Common Crawl, rephrasing, translation-to-English augmentation |
| Nemotron-Pretraining-Specialized-v1 | synthetic STEM reasoning, textbook, scientific coding, InfiniByte cross-domain code |

## 6. SFT Summary

- Section: paper §3.1
- trained on **18M+ samples**
- 13,000 steps
- batch size 64
- sequence packing to 256k
- reasoning on/off control:
  - strip reasoning on 10% of samples
- token budget control:
  - truncate 3% of reasoning traces to alternate budgets
- training mixture spans chat, tool use, reasoning, STEM, coding, multilingual, long context, theorem proving, safety

## 7. RL Summary

- Section: paper §3.2 and §3.3
- RLVR uses **six** simultaneous environments
- GRPO is synchronous
- 128 prompts per step
- 16 generations per prompt
- effective batch size 2048
- MoE router weights frozen during RL
- expert bias continues updating via aux-loss-free balancing
- RLHF uses a large generative reward model (GenRM)
- DPO is used separately for reducing hallucinated tool use

## 8. RL Environments

| Environment family | Example reward signal |
|---|---|
| Math | answer correctness |
| Code generation | unit-test execution |
| STEM MCQA | option correctness |
| Instruction following | constraint satisfaction |
| Workplace assistant | task completion / tool trajectory quality |
| Structured outputs JSON | schema adherence |

## 9. Headline Public Comparison Claims

From the paper, in the **single-H200, 8K input / 16K output** comparison setup:

- up to **3.3×** higher throughput than Qwen3-30B-A3B-Thinking-2507
- about **2.2×** higher throughput than GPT-OSS-20B
- competitive or better accuracy on several reasoning/agentic benchmarks
- stronger long-context results than competitors on RULER at large contexts

## 10. Benchmark Anchors

### Base model vs Qwen3-30B-A3B-Base

Nano3 Base is stronger on:

- MMLU-Pro
- AGIEval-En
- HumanEval
- MBPP-Sanitized
- GSM8K
- MATH
- MATH-500
- HellaSwag
- OpenBookQA
- PIQA
- WinoGrande
- RULER 64K / 128K / 256K

Qwen3 Base is stronger on:

- MMLU
- ARC-Challenge
- RACE
- MMLU Global Lite
- MGSM

### Post-trained model vs Qwen3 / GPT-OSS

Use the full table in `paper/evaluation.md`, but the important anchors are:

- AIME25 (with tools): 99.17
- GPQA (no tools): 73.04
- LiveCodeBench: 68.25
- SWE-Bench (OpenHands): 38.76
- IFBench (prompt): 71.51
- RULER-100 @ 1M: 86.34

## 11. Quantization Notes

- public FP8 checkpoint exists
- selective PTQ keeps self-attention layers and their preceding Mamba layers in BF16
- quantizes remaining layers and KV cache to FP8
- goal: preserve accuracy while improving throughput

Public before/after FP8 anchors:

| Metric | BF16 | FP8 |
|---|---:|---:|
| MMLU-Pro | 78.30 | 78.10 |
| AIME25 (no tools) | 89.06 | 87.71 |
| TauBench Avg | 49.04 | 47.04 |
| IFBench | 71.51 | 72.19 |

## 12. Safety / Alignment Notes

- RLHF reward-model data reuses the same underlying prompt distribution as SFT safety data
- for harmful prompts, preference pairs are `<safe, unsafe>`
- for safe prompts, preference pairs are `<safe, over-refusal>`
- chosen/rejected candidates were generated from multiple open models
- DPO appendix shows reduced hallucinated tool use with small additional preference training

## 13. Deployment Defaults

### Transformers

- `trust_remote_code=True`
- HF config defaults to 256k context
- reasoning can be disabled with `enable_thinking=False`

### vLLM

- needs Nano3 reasoning parser
- supports auto tool choice
- 256k example in public card
- 1M serving is enabled with:
  - `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`

### TRT-LLM

- documented with `trtllm-serve`
- uses Nano3 reasoning and tool parsers

### SGLang

- public launch example exists
- uses `flashinfer`

## 14. Reasoning Controls

| Control | Where it appears |
|---|---|
| reasoning on/off | SFT data construction and model-card deployment API |
| thinking budget / reasoning budget | model-card deployment examples |
| tool-integrated reasoning | SFT data + deployment parsers |

## 15. Public Recipe Map

| Stage | Repo path | What it covers |
|---|---|---|
| 0 | `src/nemotron/recipes/nano3/stage0_pretrain/` | pretraining data prep + Megatron-Bridge pretraining |
| 1 | `src/nemotron/recipes/nano3/stage1_sft/` | chat-template prep + packed Parquet + Megatron-Bridge SFT |
| 2 | `src/nemotron/recipes/nano3/stage2_rl/` | RL JSONL prep + NeMo-RL / GRPO |
| 3 | `src/nemotron/recipes/nano3/stage3_eval/` | NeMo Evaluator config/deploy/run |

## 16. What the Public Recipes Do *Not* Promise

They explicitly say:

- they use **open-source subsets**
- they are **reference implementations**
- results will differ from the paper because the full internal/proprietary training mix is not public

So if asked “can I match the paper numbers?” the safe answer is:

- **not exactly with the public repo alone**
- you can reproduce the **pipeline shape and public subset workflow**

## 17. `/nemotron-customize` Step Map

| Need | Step / mode |
|---|---|
| curate or filter new corpus | `curate/nemo_curator` |
| pack Nano3 SFT JSONL | `prep/sft_packing` |
| SFT with Megatron-Bridge | `sft/megatron_bridge` |
| SFT with smaller GPU counts / LoRA | `sft/automodel` |
| RL with GRPO | `rl/nemo_rl_grpo` *(present in the step catalog as planned; ground it on `src/nemotron/recipes/nano3/stage2_rl/`)* |
| evaluate model | `eval/model_eval` |
| convert HF → Megatron | `convert/hf_to_megatron` |
| convert Megatron → HF | `convert/megatron_to_hf` |

## 18. Important Handoff Caveat

There is **currently no public catalog step** for Nano3 pretraining in `src/nemotron/steps/STEPS.md`.

Also note that `rl/nemo_rl_grpo` is listed in the step catalog but its manifest marks it as **planned**, so RL builds should still be grounded carefully on the stage2 recipe rather than treated as a fully mature catalog path.

If the user asks to build stage0 pretraining via `/nemotron-customize`:

- treat it as an **Explorer-mode** or direct recipe task
- ground on `src/nemotron/recipes/nano3/stage0_pretrain/`
- do not claim a catalog `pretrain/*` step exists unless one is later added

If the user asks for Nano3 RL generation via `/nemotron-customize`:

- mention `rl/nemo_rl_grpo` as the closest step surface
- note that it is still marked **planned**
- ground specifics on `src/nemotron/recipes/nano3/stage2_rl/`

## 19. Quick Answer Templates

### “What is Nano3?”

Use:

- 31.6B total / 3.2B active / 3.6B incl. embeddings
- hybrid Mamba-Transformer + sparse MoE
- trained on 25T tokens
- post-trained with SFT + RLVR + RLHF

### “Can I reproduce it?”

Use:

- public repo reproduces the **workflow**
- public repo does **not** include all paper data/conditions
- open-source subsets only

### “How do I deploy it?”

Use:

- model-card deployment table
- Transformers / vLLM / TRT-LLM / SGLang
- mention reasoning on/off and budget control if relevant

## 20. Best File by Topic

| Topic | File |
|---|---|
| architecture | `paper/architecture.md` |
| data | `paper/data.md` |
| pretraining schedule | `paper/pretraining.md` |
| SFT | `paper/sft.md` |
| RL | `paper/rl.md` |
| benchmarks | `paper/evaluation.md` |
| safety | `paper/safety.md` |
| public usage/deployment | `model-card.md` |
| public reproduction | `recipes/overview.md` |
