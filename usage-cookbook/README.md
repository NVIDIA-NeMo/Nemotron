# Usage Cookbook

Examples on how to get started with Nemotron models

---

## Model Specific Cookbooks

| Model | Best For | Key Features | Trade-offs | Resources |
|-------|----------|--------------|------------|-----------|
| **Llama-3.3-Nemotron-Super-49B-v1.5** | Production deployments needing strong reasoning with efficiency | • 128K context<br>• Single H200 GPU<br>• RAG & tool calling<br>• Optimized via NAS | Balances accuracy & throughput | [📁 Cookbooks](./Llama-Nemotron-Super-49B-v1.5/) |
| **NVIDIA-Nemotron-Nano-9B-v2** | Resource-constrained environments needing flexible reasoning | • 9B params<br>• Hybrid Mamba-2 architecture<br>• Controllable reasoning traces<br>• Unified reasoning/non-reasoning | Smaller model with configurable reasoning | [📁 Cookbooks](./Nemotron-Nano-9B-v2/) |


---

## What's Inside

This directory contains cookbook-style guides showing how to deploy and use the models directly:

- **TensorRT-LLM Launch Guide** - Running Nemotron models efficiently with TensorRT-LLM
- **vLLM Integration** - Steps for fast inference and scalable serving of Nemotron models with vLLM.
- **SGLang Deployment** - Tutorials on serving and interacting with Nemotron via SGLang
- **NIM Microservice** - Guide to deploying Nemotron as scalable, production-ready endpoints using NVIDIA Inference Microservices (NIM).
- **Hugging Face Transformers** - Direct loading and inference of Nemotron models with Hugging Face Transformers


