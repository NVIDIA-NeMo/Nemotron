# Nemotron 3 Nano Model Card Notes

This file condenses the public Hugging Face model cards and the Nano3 paper into one operator-facing summary.

Primary sources:

- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16`
- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`
- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8/safety.md`
- the Nano3 tech report

## Identity

| Item | Value |
|---|---|
| Family | NVIDIA Nemotron |
| Model | Nemotron 3 Nano 30B-A3B |
| Developer | NVIDIA Corporation |
| Architecture family | Hybrid Mamba-Transformer with sparse MoE |
| Total params | 31.6B |
| Active params | 3.2B per forward pass; 3.6B including embeddings |
| Base checkpoint | `NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` |
| Post-trained checkpoint | `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| Quantized checkpoint | `NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` |
| License | NVIDIA Nemotron Open Model License |

## Release Window and Data Freshness

| Item | Value |
|---|---|
| Model dates | September 2025 – December 2025 |
| Pretraining data freshness | cutoff June 25, 2025 |
| Post-training data freshness | cutoff November 28, 2025 |

## Checkpoint Variants

| Checkpoint | What it is | Best use |
|---|---|---|
| Base BF16 | pretrained next-token-prediction model | continued pretraining, SFT, conversion, ablations |
| BF16 | post-trained instruct/reasoning/chat model | highest-fidelity reference for public post-trained behavior |
| FP8 | selectively quantized post-trained deployment model | fastest inference/deployment path with minimal accuracy loss |

## High-Level Capability Positioning

The public model cards present Nano3 as:

- a unified model for **reasoning and non-reasoning tasks**
- a model that can emit a **reasoning trace first** and a final answer after
- a model with **reasoning on/off control**
- a model with **thinking budget / reasoning budget** controls in API-compatible serving paths
- a model intended for **agent systems, chatbots, RAG systems, coding, and general instruction following**

## Context Length Notes

| Surface | Context note |
|---|---|
| Paper | supports up to **1M tokens** |
| HF Transformers default config | **256k** default because higher context requires more VRAM |
| vLLM example | shows **256k** by default and **1M** with `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` |

Interpretation:

- **1M** is the model capability claim
- **256k** is the common public runtime default in Hugging Face/vLLM examples

## Headline Benchmark Notes

The paper’s headline claims, repeated in the cards, are:

- up to **3.3×** higher inference throughput than Qwen3-30B-A3B-Thinking-2507
- about **2.2×** higher inference throughput than GPT-OSS-20B
- strong long-context performance up to **1M tokens**
- competitive or better accuracy on multiple reasoning, agentic, and instruction-following benchmarks

The FP8 card reproduces the public before/after quantization table:

| Benchmark | BF16 | FP8 |
|---|---:|---:|
| MMLU-Pro | 78.30 | 78.10 |
| AIME25 (no tools) | 89.06 | 87.71 |
| AIME25 (with tools) | 99.17 | 98.80 |
| GPQA (no tools) | 73.04 | 72.47 |
| GPQA (with tools) | 75.00 | 73.40 |
| LiveCodeBench | 68.25 | 67.62 |
| SciCode (subtask) | 33.28 | 31.88 |
| TauBench V2 Avg | 49.04 | 47.04 |
| BFCL v4 | 53.76 | 53.15 |
| IFBench (prompt) | 71.51 | 72.19 |
| AA-LCR | 35.85 | 36.06 |
| MMLU-ProX | 59.50 | 59.63 |

Use `paper/evaluation.md` for the fuller benchmark comparison tables.

## Public Deployment Paths

The FP8 model card gives concrete public deployment guidance for:

| Runtime | Notes |
|---|---|
| Hugging Face Transformers | `trust_remote_code=True`; BF16 load path shown; reasoning enabled by default |
| vLLM | custom Nano3 reasoning parser; auto tool choice; FP8 KV cache; 256k example with optional 1M serving |
| TensorRT-LLM | uses `trtllm-serve` with Nano3 reasoning/tool parsers |
| SGLang | `flashinfer` attention backend; Nano3 reasoning parser |

## Public Runtime Defaults and Recommendations

From the FP8 card:

| Scenario | Recommendation |
|---|---|
| Reasoning enabled | keep `max_tokens` high; example says `10,000` is a useful value |
| Reasoning tasks | `temperature=1.0`, `top_p=1.0` |
| Tool calling | `temperature=0.6`, `top_p=0.95` |
| Reasoning off in Transformers | set `enable_thinking=False` in `apply_chat_template()` |
| Reasoning off in vLLM | pass `chat_template_kwargs: {"enable_thinking": false}` |

## Budget Control

The FP8 card documents **thinking budget** support for OpenAI-compatible deployments.

Key behavior:

- `reasoning_budget` limits internal reasoning
- the server tries to stop at the next newline
- if no newline appears soon enough, it truncates after a small spillover window

This is especially relevant for:

- latency-sensitive agent steps
- customer support
- edge inference
- throughput-constrained deployments

## Safety Notes from Public Model Materials

The public `safety.md` for the FP8 release says:

- a guard model was used to exclude potentially harmful content from training
- a Gemma-3 4B-based guard model trained on **Nemotron Content Safety Dataset v2** was used for harmful/illegal content filtering
- use cases are restricted by the NVIDIA Open Model License
- training data handling followed least-privilege controls
- model safety evaluation used **Nemotron Content Safety Dataset V2** and an internal minority-sexuality-focused safety dataset

This should be read together with `paper/safety.md`, which explains the RLHF preference-pair construction for harmful prompts, safe prompts, unsafe outputs, and over-refusals.

## What the Model Card Is Good For

Use this file when the question is:

- “Which Nano3 checkpoint should I use?”
- “Is FP8 public?”
- “What license is this under?”
- “How do I deploy it with vLLM / Transformers / TRT-LLM?”
- “How do I turn reasoning off?”
- “What’s the public data freshness / model date?”

Use the paper chunks when the question is:

- “How was it trained?”
- “What exactly is the RL algorithm?”
- “What data was added over Nemotron 2?”
- “What do the benchmark tables say?”

## Reproduction Caveat

The model cards describe the released checkpoints and supported deployment paths.
They do **not** imply that the public repo recipes exactly recreate NVIDIA’s full internal training runs.

For public reproducibility paths, use:

- `recipes/overview.md`
- `recipes/stage0_pretrain.md`
- `recipes/stage1_sft.md`
- `recipes/stage2_rl.md`
- `recipes/stage3_eval.md`
