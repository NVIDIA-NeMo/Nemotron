# Nemotron-3.5 Lightning Text2SQL LoRA Fine-Tuning with Megatron Bridge

LoRA fine-tuning of Nemotron-3.5 Lightning for Text2SQL, using
[NeMo Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge). Everything runs from
[`mbridge_lora_cookbook.ipynb`](mbridge_lora_cookbook.ipynb).

This model fits on a single node, and only the training step needs a GPU.

## Steps

1. **Prepare data** — build a [BIRD](https://huggingface.co/datasets/xu3kev/BIRD-SQL-data-train)
   `training.jsonl` from the no-reasoning and reasoning splits. *(CPU)*
2. **Convert** the Hugging Face checkpoint to Megatron-Bridge format. *(CPU)*
3. **Fine-tune** with LoRA on packed sequences. *(GPU)*
4. **Merge** the adapter and export back to Hugging Face format. *(CPU)*

## Hardware

Set `n_devices` to the number of GPUs you have; expert parallelism follows it. Measured on 80GB
H100s, training one epoch at the notebook's defaults:

| GPUs | Peak memory/GPU | One epoch |
| --- | --- | --- |
| 1 | ~79 GB | ~60 min |
| 2 | ~51 GB | ~34 min |
| 4 | ~35 GB | ~18 min |
| 8 | ~27 GB | ~8 min |

On one GPU the notebook sets `REDUCE_MTP_HEADS=1` for you: the model only fits with a single
multi-token-prediction head instead of the recipe's two, and under 1 GB to spare. Use two GPUs if
you have them.

You also need about 130 GB of disk for the downloaded and converted checkpoints, plus room for the
merged export.

## Files

| File | What it is |
| --- | --- |
| `mbridge_lora_cookbook.ipynb` | The main notebook — start here. |
| `dataprep.py`, `dataset_bird.py`, `dataset_bird_reasoning.py`, `base_sft_dataset.py` | Build the BIRD `training.jsonl`. |
| `convert.py` | One-time Hugging Face → Megatron-Bridge import (CPU). |
| `train.py` | LoRA training; points the shipped recipe at your paths and GPUs. |
| `SKILL.md` | Agent skill — lets a coding agent run this for you. |

## Running with a coding agent

[`SKILL.md`](SKILL.md) tells a coding agent how to run this end to end: what to ask you for, how to
pick a GPU count, and how to check each step worked.
