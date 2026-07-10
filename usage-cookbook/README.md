# Usage Cookbook

Examples on how to get started with Nemotron models

---

## Fine-Tuning: How Many GPUs Do I Need?

Start with the **[post-training sizing quick reference](nemotron3-post-training-sizing-quickref.md)**
— one page: model × method → GPU count, with copy-paste entry points.
Detailed per-model guides (memory math, parallelism, caveats):
[Nano 30B](Nemotron-3-Nano/nemotron3-nano-post-training-sizing.md) ·
[Super 120B](Nemotron-3-Super/nemotron3-super-post-training-sizing.md) ·
[Ultra 550B](Nemotron-3-Ultra/nemotron3-ultra-post-training-sizing.md)

## What's Inside

This directory contains cookbook-style guides showing how to deploy and use the models directly:

- **TensorRT-LLM Launch Guide** - Running Nemotron models efficiently with TensorRT-LLM
- **vLLM Integration** - Steps for fast inference and scalable serving of Nemotron models with vLLM.
- **SGLang Deployment** - Tutorials on serving and interacting with Nemotron via SGLang
- **NIM Microservice** - Guide to deploying Nemotron as scalable, production-ready endpoints using NVIDIA Inference Microservices (NIM).
- **Hugging Face Transformers** - Direct loading and inference of Nemotron models with Hugging Face Transformers


