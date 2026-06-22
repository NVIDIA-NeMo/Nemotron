# Nemotron-3 Ultra Text2SQL LoRA Fine-Tuning with Megatron Bridge

This example walks through LoRA fine-tuning of **Nemotron-3 Ultra** for the Text2SQL use case
using [NeMo Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge). Everything is
driven from a single notebook: [`mbridge_lora_cookbook.ipynb`](mbridge_lora_cookbook.ipynb).

If you've seen the [Nemotron-3 Super](../../Nemotron-3-Super/lora-text2sql) version, this follows
the same spirit — prepare data, convert the checkpoint, fine-tune with LoRA — with one important
difference: **Ultra is a 550B-total / A55B-active hybrid Mamba-Transformer MoE, which doesn't fit
on a single node.** So instead of running steps inline with `torchrun`, each step here is submitted
as a **multi-node SLURM job** (via Pyxis/enroot). The notebook is meant to be run from a cluster
**login node**.

## What you'll do

1. **Set up your environment** — one cell defines all SLURM settings (node counts, account,
   partition, QOS) and paths, and writes `config.env`. This is the only place you should need to edit.
2. **Prepare data** — build a [BIRD](https://huggingface.co/datasets/xu3kev/BIRD-SQL-data-train)
   Text2SQL `training.jsonl` from both the no-reasoning and reasoning splits (the full set by
   default; set `MAX_TRAIN_SAMPLES` for a smaller smoke run) using Ultra's tokenizer/chat template.
3. **Convert** the Hugging Face checkpoint to Megatron-Bridge format (distributed import — CPU import
   isn't feasible at 550B).
4. **Fine-tune** with LoRA on the prepared data (packed sequences) and save an adapter.

Each step follows the same rhythm: a cell to **launch** the SLURM job, a cell you can re-run to
**check** its status, and a **sanity-check** cell that confirms you got the expected output before
moving on.

## Prerequisites

- A SLURM multi-node cluster with at least 48 GPUs (H100 and above) with Pyxis/enroot (`srun --container-image=...`). **This notebook was tested on GB200 nodes (4 GPUs/node).**
- The **Nemotron-3 Ultra checkpont** downloaded to a shared path (accessible to all nodes, such as lustre) — set as `HF_MODEL_PATH`.
- A **Hugging Face token** placed at `${HF_HOME}/token`.
- This notebook directory on a **shared filesystem** the compute nodes can mount.
- A working **container image**. The notebook ships with a placeholder
  (`<<REPLACE_WITH_OFFICIAL_CONTAINER>>`) — point `CONTAINER_IMAGE` at the official Ultra container
  once it's available.

## Files

| File | What it is |
| --- | --- |
| `mbridge_lora_cookbook.ipynb` | The main notebook — start here. |
| `config.env` | Single config file (written by the notebook's setup cell; sourced by every step). |
| `dataprep.py`, `dataset_bird.py`, `dataset_bird_reasoning.py`, `base_sft_dataset.py` | Build the BIRD Text2SQL `training.jsonl` (both splits). |
| `train_lora.py` | Packed-sequence LoRA training entrypoint (run by `slurm/train_lora.sbatch`). |
| `slurm/` | The SLURM scripts the notebook submits: `dataprep.sbatch`, `convert.sbatch`, `train_lora.sbatch`. |
| `SKILL.md` | Agent skill — lets a coding agent run this tutorial for you (see below). |

## Run this notebook with a coding agent

`SKILL.md` is an agent skill that teaches a coding agent how to run this cookbook end-to-end: what
each step does, which environment details to ask you for (account, partition/QOS, paths, container,
HF token), how to launch and monitor the SLURM jobs, and how to confirm each step succeeded. Point
your agent at it and it will gather what it needs and drive the run on your behalf.

