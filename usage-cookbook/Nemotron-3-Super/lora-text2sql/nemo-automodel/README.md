# Nemotron-3-Super Fine-tuning with NeMo AutoModel

This directory contains a Jupyter notebook and PEFT config for fine-tuning Nemotron-3-Super with NVIDIA NeMo AutoModel.

## What's included

- **`automodel_lora_cookbook.ipynb`** — End-to-end recipe: dataset creation, config setup, finetuning, and optional export.
- **`base-peft-config-cookbook.yaml`** — PEFT (LoRA) config. Copy this into your AutoModel clone at `examples/llm_finetuning/nemotron`
- **`text2sql.py`** — Dataset Target. Copy this into your AutoModel clone at `nemo_automodel/components/datasets/llm`

## Prerequisites

- Multi-GPU machine (e.g. 8× H100 for Nemotron-Super; fewer GPUs for smaller models).
- [uv](https://github.com/astral-sh/uv) (for AutoModel env), or use `pip`/`venv` as in the AutoModel repo docs.
- Hugging Face token for gated models (e.g. Nemotron-Super): run `huggingface-cli login`.

## Setup

### 1. Clone the AutoModel repo

Clone NVIDIA NeMo AutoModel and set up its environment (this repo does **not** include AutoModel):

```bash
git clone https://github.com/NVIDIA-NeMo/Automodel.git <AUTOMODEL_DIR> 
cd <AUTOMODEL_DIR>
uv venv && source .venv/bin/activate
uv sync --frozen
```

Optional (e.g. for Nemotron-Super):
`pip install mamba-ssm --no-build-isolation`
Optional: `pip install transformer-engine-torch==2.11.0`

### 2. Download the base model

Obtain the base model (e.g. from Hugging Face or NVIDIA NGC) and place it on disk. You will point the PEFT config and/or notebook to this path (`model.pretrained_model_name_or_path`).

For gated models (e.g. `[nvidia/nemotron-super-*](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16)`), ensure you are logged in:
`huggingface-cli login`

### 3. Copy the PEFT config into AutoModel

Copy this repo’s **`base-peft-config-cookbook.yaml`** into the AutoModel tree. 

```bash
cp base-peft-config-cookbook.yaml <AUTOMODEL_DIR>/examples/llm_finetune/nemotron/
```

### 4. Set paths in the notebook

Open `automodel_lora_cookbook.ipynb` and set in the first cell:

- **`WORKSPACE_ROOT`** — Your workspace root (e.g. the directory that contains this repo and/or your data).
- **`AUTOMODEL_DIR`** — Path to your AutoModel clone (where you ran `uv sync`).
- **`DATASET_DIR`** — Directory for `training.jsonl`, `validation.jsonl`, and optionally `test.jsonl`. The notebook can generate these from BIRD-SQL, or you can place your own JSONL files there.

## Running the notebook

1. Use the kernel that points to the AutoModel `.venv` (or install the same deps in your Jupyter env).
2. Dataset creation (Section 2) writes `training.jsonl`, `validation.jsonl`, and optionally `test.jsonl` into `DATASET_DIR`. Adjust the config so those paths match what you set in the PEFT config.
3. Finetuning (Section 3) runs `torchrun` with the cookbook config. Ensure `N_DEVICES` matches your GPU count.
