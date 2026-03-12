# LoRA Fine-Tuning & Deployment of Nemotron 3 Nano for Text2SQL

End-to-end guide for LoRA fine-tuning **NVIDIA Nemotron-3-Nano-30B** on a text-to-SQL task (BIRD SQL) and deploying with NVIDIA NIM or vLLM. Two notebook options:

- **AutoModel** – Uses the [NeMo AutoModel](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/containers/nemo-automodel) container for training and checkpointing.
- **Megatron Bridge** – Uses [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) for training.

## What's Inside

| File | Description |
|------|-------------|
| `finetuning_deployment_guide_automodel.ipynb` | Full walkthrough (NeMo AutoModel) |
| `finetuning_deployment_guide_mbridge.ipynb` | Full walkthrough (Megatron Bridge) |
| `bird_sql/` | Dataset preparation helpers (BIRD SQL → chat-templated JSONL) |

## Quick Start

Open either notebook to get started. A **Deploy on Brev** one-click option is also available for the Megatron Bridge example:

[![Deploy on Brev](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg)](https://brev.nvidia.com/launchable/deploy?launchableID=env-39PnUMhHmxbMcHKO61iQ8O5F7ZL)