# Nemotron-3.5-Content-Safety Notebooks

A collection of notebooks demonstrating deployment and application cookbooks for **NVIDIA Nemotron 3.5 Content Safety**.

## Overview

Nemotron 3.5 Content Safety is a multimodal safety classifier that labels prompts and AI responses against a built-in taxonomy, supports optional reasoning traces, and accepts bring-your-own custom policies via the model's native chat-template interface. These notebooks cover serving the model on each major backend and applying it to real moderation use cases.

## What's Inside

### Deployment

- **[vllm_cookbook.ipynb](vllm_cookbook.ipynb)** — Deploy Nemotron 3.5 Content Safety with vLLM.
- **[sglang_cookbook.ipynb](sglang_cookbook.ipynb)** — Deploy Nemotron 3.5 Content Safety with SGLang.
- **[trtllm_cookbook.ipynb](trtllm_cookbook.ipynb)** — Deploy Nemotron 3.5 Content Safety with TensorRT-LLM.
- **[nim_cookbook.ipynb](nim_cookbook.ipynb)** — Deploy Nemotron 3.5 Content Safety as a prebuilt NVIDIA NIM container, the recommended path for production serving.

### Applications

- **[custom_policy_cookbook.ipynb](custom_policy_cookbook.ipynb)** — Bring your own safety policy: define custom categories and allow-lists and inject them natively via the `custom_taxonomy` chat-template kwarg.
- **[customer_service_cookbook.ipynb](customer_service_cookbook.ipynb)** — Moderate an e-commerce customer-support chatbot, catching threats, PII exposure, and social engineering while allowing frustrated-but-harmless language.

## Model Resources

- **build.nvidia.com:** [nvidia/nemotron-3.5-content-safety](https://build.nvidia.com/nvidia/nemotron-3.5-content-safety)
- **NVIDIA NIM:** [nim/nvidia/nemotron-3.5-content-safety](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/nemotron-3.5-content-safety)
- **Hugging Face:** [nvidia/Nemotron-3.5-Content-Safety](https://huggingface.co/nvidia/Nemotron-3.5-Content-Safety)
