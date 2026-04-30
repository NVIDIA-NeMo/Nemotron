<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
<!--
Card footer badges

Add a badge only when the attribute applies. Omit it (and do not substitute its opposite) when it does not.
Absence is the signal for the simpler case.

{bdg-success}`Beginner`  — no GPU required and no fine-tuning step. Uses an API or cloud endpoint.
                           Add when no other attribute badges apply.

{bdg-success}`Notebook`  — self-contained notebook and no environment beyond pip install.
                           Omit for multi-file projects.

{bdg-secondary}`Local GPU` — requires a local GPU or managed hardware.
                           Omit for API/cloud-only examples.

{bdg-info}`Fine-tuning`  — example includes a training or fine-tuning step.
                           Omit for inference-only examples.
-->

(deployment-guides)=
# Deployment Guides

Deployment guides, fine-tuning recipes, and agentic usage examples for Nemotron models. Each card links to its directory in the [Nemotron GitHub repository](https://github.com/NVIDIA-NeMo/nemotron).

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Nemotron 3 Super
:link: https://github.com/NVIDIA-NeMo/nemotron/tree/main/usage-cookbook/Nemotron-3-Super
:link-type: url

Notebooks for deploying the 120B/12B-active hybrid Mamba-Transformer MoE model with vLLM, SGLang, and TensorRT-LLM.
+++
{bdg-success}`Notebook` {bdg-secondary}`Local GPU` {bdg-muted-line}`Apr 28, 2026`
:::

:::{grid-item-card} Nemotron 3 Super — LoRA Text2SQL
:link: https://github.com/NVIDIA-NeMo/nemotron/tree/main/usage-cookbook/Nemotron-3-Super/lora-text2sql
:link-type: url

Supervised fine-tuning with LoRA for Text2SQL using the BIRD SQL benchmark. Includes recipes for both NeMo AutoModel and Megatron Bridge.
+++
{bdg-secondary}`Local GPU` {bdg-info}`Fine-tuning` {bdg-muted-line}`Apr 28, 2026`
:::

:::{grid-item-card} Nemotron 3 Super on DGX Spark
:link: https://github.com/NVIDIA-NeMo/nemotron/tree/main/usage-cookbook/Nemotron-3-Super/SparkDeploymentGuide
:link-type: url

Deploy on a single DGX Spark with 128 GB unified memory using vLLM (nightly) and TensorRT-LLM, including NVFP4 quantization and MTP speculative decoding.
+++
{bdg-secondary}`Local GPU` {bdg-muted-line}`Apr 10, 2026`
:::

:::{grid-item-card} Nemotron 3 Ultra Base
:link: https://github.com/NVIDIA-NeMo/nemotron/tree/main/usage-cookbook/Nemotron-3-Ultra-Base
:link-type: url

550B total / 55B active parameter base model checkpoint announced at GTC 2026. A starting point for custom fine-tuning and RL post-training pipelines — not yet instruction-tuned.
+++
{bdg-secondary}`Local GPU` {bdg-info}`Fine-tuning` {bdg-muted-line}`Mar 23, 2026`
:::

:::{grid-item-card} Nemotron 3 Super on GRPO/DAPO RL Training
:link: https://github.com/NVIDIA-NeMo/nemotron/tree/main/usage-cookbook/Nemotron-3-Super/grpo-dapo
:link-type: url

Full-weight RL training from a base model using the GRPO/DAPO algorithm to reproduce emergent math reasoning. Requires 5× GB200 or 3× B200 nodes.
+++
{bdg-secondary}`Local GPU` {bdg-info}`Fine-tuning` {bdg-muted-line}`Mar 11, 2026`
:::

:::{grid-item-card} Nemotron 3 Super on Agentic Coding
:link: https://github.com/NVIDIA-NeMo/nemotron/tree/main/usage-cookbook/Nemotron-3-Super/OpenScaffoldingResources
:link-type: url

Use Nemotron 3 Super with OpenCode, OpenClaw, Kilo Code CLI, and OpenHands via OpenRouter and build.nvidia.com.
+++
{bdg-success}`Beginner` {bdg-muted-line}`Mar 11, 2026`
:::

:::{grid-item-card} Nemotron Nano 2 VL
:link: https://github.com/NVIDIA-NeMo/nemotron/tree/main/usage-cookbook/Nemotron-Nano2-VL
:link-type: url

Notebooks for the 12B multimodal model that unifies visual and textual understanding. Covers NIM inference via build.nvidia.com and local Hugging Face deployment.
+++
{bdg-success}`Notebook` {bdg-secondary}`Local GPU` {bdg-muted-line}`Oct 28, 2025`
:::

:::{grid-item-card} Nemotron Parse v1.1
:link: https://github.com/NVIDIA-NeMo/nemotron/tree/main/usage-cookbook/Nemotron-Parse-v1.1
:link-type: url

Notebook for the document-parsing VLM that converts PDFs and unstructured documents into structured JSON, LaTeX, and Markdown. Available via NIM at build.nvidia.com.
+++
{bdg-success}`Beginner` {bdg-success}`Notebook` {bdg-muted-line}`Oct 28, 2025`
:::

::::
