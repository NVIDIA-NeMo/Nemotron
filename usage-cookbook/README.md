# Usage Cookbook

Examples on how to get started with Nemotron models

---

## What's Inside

This directory contains cookbook-style guides showing how to deploy and use the models directly:

- **TensorRT-LLM Launch Guide** - Running Nemotron models efficiently with TensorRT-LLM
- **vLLM Integration** - Steps for fast inference and scalable serving of Nemotron models with vLLM.
- **SGLang Deployment** - Tutorials on serving and interacting with Nemotron via SGLang
- **NIM Microservice** - Guide to deploying Nemotron as scalable, production-ready endpoints using NVIDIA Inference Microservices (NIM).
- **Hugging Face Transformers** - Direct loading and inference of Nemotron models with Hugging Face Transformers
- **[Sports Intelligence Playbooks](https://github.com/NVIDIA/sports-intelligence-playbooks)** - End-to-end guidance for customizing Nemotron 3 Nano Omni for sports video with audio, including data curation and preparation, NeMo AutoModel/Megatron SFT/LoRA, inference, and evaluation. ([Documentation](https://nvidia.github.io/sports-intelligence-playbooks/) · [Single-node SFT configuration](https://github.com/NVIDIA/sports-intelligence-playbooks/blob/main/avlm/training/automodel/sft/configs/nemotron_omni_automodel_sft_singlenode.yaml))
- **[DGX Station Post-Training](Nemotron-3.5-Lightning/dgx-station-recipes/README.md)** - Customize Nemotron 3.5 Lightning on one or two DGX Station GB300 systems with LoRA, full-weight SFT, or GRPO.
- **OCI OKE Private Deployment** - A Phoenix-only private deployment guide for `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` using OKE, OCI Bastion service, and `vLLM`, providing a reproducible OCI path comparable to common AWS GPU/Kubernetes deployment patterns.
