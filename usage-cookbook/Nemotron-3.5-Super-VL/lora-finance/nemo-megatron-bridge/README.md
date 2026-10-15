# Nemotron-3.5 Super VL Finance LoRA Fine-Tuning with Megatron Bridge

LoRA fine-tuning of Nemotron-3.5 Super VL on financial documents, text and images together, using
[NeMo Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge). One adapter learns to read
values off document images, calculate from extracted evidence, and write small Python programs for
financial questions. Everything runs from [`mbridge_lora_cookbook.ipynb`](mbridge_lora_cookbook.ipynb)
on one 8-GPU node, or as Slurm jobs.

## Files

| File | What it is |
| --- | --- |
| `mbridge_lora_cookbook.ipynb` | The main notebook — start here. |
| `config.env` | The one file to edit: container, model, `WORKSPACE`, and training layout. |
| `steps/` | One script per step; the notebook and the Slurm jobs both run these. |
| `slurm/step.sbatch` | Runs a step as a Slurm job in the container. |
| `dataprep.py`, `dataset_tatdqa.py`, `dataset_bizbench.py`, `base_sft_dataset.py` | Build the finance chat data: one class per source dataset, sharing one chat format. |
| `train.py` | LoRA training; points the shipped recipe at your data, checkpoint, and GPUs. |
| `try_model.py` | Answers held-out questions with vLLM. |
| `SKILL.md` | Agent skill — lets a coding agent run this for you. |

## Hardware

**Minimum:** one node with 8 GPUs of 80 GB or more, and about 800 GB of disk. Training alone also
runs on 4 H200s.

Pick the training layout with `GPUS_PER_NODE`, `TRAIN_TP`, `TRAIN_PP`, and `TRAIN_EP` in
`config.env`, and request the same GPUs per node on Slurm. Measured on H200 (141 GB), training one
epoch (1,389 steps):

| GPUs | TP / PP / EP | Peak memory/GPU | One epoch |
| --- | --- | --- | --- |
| 4 | 4 / 1 / 4 | ~95 GB | ~7.5 h |
| 8 (default) | 2 / 1 / 8 | ~98 GB | ~1.75 h |
| 8 | 4 / 1 / 8 | ~66 GB | ~3.5 h |
| 16, two nodes | 4 / 2 / 8 | ~40 GB | ~2–2.5 h (estimated) |

Memory figures are PyTorch allocations; `nvidia-smi` includes cached memory and can show much higher
usage. On 80 GB GPUs use TP 4. The try-out step was verified on 8 GPUs.

Disk: about 250 GB each for the download, the converted checkpoint, and each merged export, 1 GB
for data, and 12 GB per training run (adapter plus optimizer state).

## Prerequisites

- **Model:** [`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16),
  with a Hugging Face token in `HF_TOKEN`.
- **Container:** `nvcr.io/nvidian/nemo:26.10.rc4` (Megatron-Bridge 0.7.0), run with Docker for the
  notebook or with Pyxis on Slurm.
- **Settings:** all in [`config.env`](config.env): the model, the container, `WORKSPACE` (where
  everything is written; the only value you must set), the GPU layout, and the run length.

The notebook's first cell has the details and the command that launches the container.

## Steps

1. **Download** the checkpoint. *(CPU)*
2. **Prepare data**: build chat-format training data from
   [TAT-DQA](https://huggingface.co/datasets/next-tat/TAT-DQA) document images and
   [BizBench](https://huggingface.co/datasets/kensho/bizbench) questions. *(CPU)*
3. **Convert** the Hugging Face checkpoint to Megatron-Bridge format. *(4 GPUs)*
4. **Fine-tune** with LoRA. *(8 GPUs)*
5. **Merge** the adapter and export back to Hugging Face format. *(4 GPUs)*
6. **Try** the fine-tuned and base models side by side with vLLM. *(8 GPUs)*

Each step is one script in `steps/`, run inside the container, and writes under `WORKSPACE`.

## Running on Slurm

From a login node with Pyxis, submit the steps from this directory one at a time, each after the
previous job finishes. `WORKSPACE` and this directory must be on a filesystem every node can see.
Add your `--account`, `--partition`, and `--time` ([step times](SKILL.md#what-the-cookbook-does)).
Each job reads `config.env` when it starts, not when you submit it.

```bash
export HF_TOKEN=...
sbatch -c 16 --mem=64G slurm/step.sbatch download
sbatch -c 16 --mem=64G slurm/step.sbatch dataprep
sbatch --exclusive --gpus-per-node=8 slurm/step.sbatch convert   # uses 4 of the 8 GPUs
sbatch --exclusive --gpus-per-node=8 slurm/step.sbatch train     # --nodes=2 for the two-node layout
sbatch --exclusive --gpus-per-node=8 slurm/step.sbatch merge     # uses 4 of the 8 GPUs
sbatch --exclusive --gpus-per-node=8 slurm/step.sbatch try
sbatch --exclusive --gpus-per-node=8 slurm/step.sbatch try base   # the base model, for comparison
```

Each job logs to `slurm-<job id>.log`. The notebook's bash sanity checks work from the login node
too.

## Use your own data

Each source dataset is one small class that turns its rows into `Example`s (prompt, reasoning,
answer, optional image); `BaseSFTDataset` turns every `Example` into the same chat format and holds
out one in ten source documents for testing. To add a dataset:

1. Copy `dataset_bizbench.py` (text) or `dataset_tatdqa.py` (images), and rewrite `_load_examples()`
   and the prompt template for your data. Set `split_key` so that questions about the same document share
   it (a document ID, or the question itself for self-contained questions); never the answer.
2. Add an instance of your class to the `datasets` list in `dataprep.py`, and rerun data prep.

The instruction shared by every prompt is `INSTRUCTION` in `base_sft_dataset.py`. Training
hyperparameters come from the Megatron-Bridge recipe; override them in `build_config()` in
`train.py`.

## Running with a coding agent

[`SKILL.md`](SKILL.md) tells a coding agent how to run this end to end: what to ask you for, how to
pick a GPU layout, and how to check each step worked.
