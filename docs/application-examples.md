<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
<!--
Card footer badges
==================
Add a badge only when the attribute applies. Omit it (and do not substitute its opposite) when it does not.
Absence is the signal for the simpler case.

{bdg-success}`Notebook`  — self-contained notebook; no environment beyond pip install.
                           Omit for multi-file projects and anything that isn't a notebook.

{bdg-secondary}`Local GPU` — requires a local GPU or hardware beyond a laptop or desktop.
                           Omit for API/cloud-only examples. Do not add an "API" badge.

{bdg-info}`Fine-tuning`  — example includes a training/fine-tuning step.
                           Omit for inference-only examples. Do not add an "Inference" badge.
-->

(application-examples)=
# Application Examples

End-to-end applications built on Nemotron models, including agentic workflows, RAG systems, and fine-tuning pipelines. Each card links to its directory in the [Nemotron GitHub repository](https://github.com/NVIDIA-NeMo/nemotron).

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`mark-github;1.5em;sd-mr-1` Nemotron 3 Super Getting Started Guide
:link: https://github.com/NVIDIA-NeMo/nemotron/tree/main/use-case-examples/Nemotron-3-Super-Getting-Started-Guide
:link-type: url

Introductory notebook covering Nemotron 3 Super's reasoning features: thinking, reasoning budget, low effort mode, streaming responses, tool-call streaming, and Perplexity Search integration using the OpenAI-compatible API.
+++
{bdg-success}`Notebook` {bdg-muted-line}`Mar 11, 2026`
:::

:::{grid-item-card} {octicon}`mark-github;1.5em;sd-mr-1` SQL LoRA Fine-tuning and Deployment
:link: https://github.com/NVIDIA-NeMo/nemotron/tree/main/use-case-examples/sql-lora-finetuning-and-deployment
:link-type: url

End-to-end LoRA fine-tuning of Nemotron 3 Nano on Text2SQL (BIRD SQL) with deployment via NVIDIA NIM or vLLM using NeMo AutoModel or Megatron Bridge.
+++
{bdg-secondary}`Local GPU` {bdg-info}`Fine-tuning` {bdg-muted-line}`Mar 11, 2026`
:::

:::{grid-item-card} {octicon}`mark-github;1.5em;sd-mr-1` Intelligent Document Processing
:link: https://github.com/NVIDIA-NeMo/nemotron/tree/main/use-case-examples/Intelligent%20Document%20Processing%20with%20Nemotron%20RAG
:link-type: url

IDP pipeline that extracts and queries complex enterprise documents — financial reports, charts, and tables — using NeMo Retriever and multimodal Nemotron models.
+++
{bdg-secondary}`Local GPU` {bdg-muted-line}`Feb 09, 2026`
:::

:::{grid-item-card} {octicon}`mark-github;1.5em;sd-mr-1` Voice RAG Agent
:link: https://github.com/NVIDIA-NeMo/nemotron/tree/main/use-case-examples/nemotron-voice-rag-agent-example
:link-type: url

End-to-end voice-driven RAG agent combining speech-to-text, multimodal retrieval, 1M-token reasoning, and safety guardrails using open Nemotron models.
+++
{bdg-secondary}`Local GPU` {bdg-muted-line}`Jan 07, 2026`
:::

:::{grid-item-card} {octicon}`mark-github;1.5em;sd-mr-1` Simple Nemotron 3 Nano Usage
:link: https://github.com/NVIDIA-NeMo/nemotron/tree/main/use-case-examples/Simple%20Nemotron-3-Nano%20Usage%20Example
:link-type: url

Introductory notebook covering basic inference, reasoning mode toggling, and multi-agent systems using the OpenAI-compatible API via OpenRouter and LangChain.
+++
{bdg-success}`Notebook` {bdg-muted-line}`Dec 15, 2025`
:::

:::{grid-item-card} {octicon}`mark-github;1.5em;sd-mr-1` Data Science ML Agent
:link: https://github.com/NVIDIA-NeMo/nemotron/tree/main/use-case-examples/Data%20Science%20ML%20Agent
:link-type: url

Natural language-driven ML agent built on Nemotron Nano 9B with GPU-accelerated data exploration and model training using RAPIDS cuDF and cuML.
+++
{bdg-secondary}`Local GPU` {bdg-muted-line}`Nov 06, 2025`
:::

:::{grid-item-card} {octicon}`mark-github;1.5em;sd-mr-1` RAG Agent
:link: https://github.com/NVIDIA-NeMo/nemotron/tree/main/use-case-examples/RAG%20Agent%20with%20Nemotron%20RAG%20Models
:link-type: url

Production-ready RAG agent using local Hugging Face embedding and reranking models with NVIDIA AI Endpoints for LLM inference, built on LangGraph.
+++
{bdg-secondary}`Local GPU` {bdg-muted-line}`Oct 28, 2025`
:::

::::
