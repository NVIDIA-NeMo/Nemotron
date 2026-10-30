# Nemotron-3-Ultra — LoRA Fine-Tuning on Text2SQL with NeMo AutoModel

An end-to-end cookbook for parameter-efficient (LoRA) fine-tuning of
Nemotron-3-Ultra-550B-A55B-BF16 — a 550B-parameter Mixture-of-Experts model with a
Multi-Token-Prediction (MTP) head — on the BIRD-SQL
text-to-SQL task. One notebook drives two supercomputers (GB200 and H100) from a single
switch, and walks all the way to evaluation and vLLM deployment.

## What's included


| File                                             | Purpose                                                                | Copy into AutoModel at                    |
| ------------------------------------------------ | ---------------------------------------------------------------------- | ----------------------------------------- |
| `**automodel_ultra_lora_cookbook.ipynb`**        | The end-to-end notebook (config → env → data → train → eval → deploy). | — (run it from this directory)            |
| `**nemotron_ultra_v3_text2sql_peft_gb200.yaml**` | LoRA recipe, **GB200 (primary / P0)**.                                 | `examples/llm_finetune/nemotron/`         |
| `**nemotron_ultra_v3_text2sql_peft_h100.yaml`**  | LoRA recipe, **H100 (secondary / P1)**.                                | `examples/llm_finetune/nemotron/`         |
| `**text2sql.py`**                                | Dataset target (`make_text2sql_dataset`).                              | `nemo_automodel/components/datasets/llm/` |


> The notebook's Section 2 copies the two YAMLs and `text2sql.py` into these locations for you
> (it finds the installed `nemo_automodel` package automatically).

## Minimum hardware & storage

Both targets are **multi-node** runs (`ep_size` **must equal the world size**,
`N_NODES × N_PROC_PER_NODE`, which spans several nodes). Training is launched **directly from the
notebook** — no Slurm — by SSHing `torchrun` to each node of your allocation, so the only extra
requirement is **passwordless SSH between the allocated nodes** (the notebook's §5.2 verifies it).


| Target         | Minimum allocation                          | `ep_size` | Launch geometry                 |
| -------------- | ------------------------------------------- | --------- | ------------------------------- |
| **GB200** (P0) | **4 × GB200 nodes — 16 GPUs (184 GB each)** | `16`      | `--nnodes=4 --nproc-per-node=4` |
| **H100** (P1)  | **32 × H100 80 GB** (e.g. 4 × 8)            | `32`      | `--nnodes=4 --nproc-per-node=8` |


Also required:

- **Base checkpoint from NGC: ~1.1 TB on disk (BF16).**
- CUDA 12.1+, Python 3.10+, `[uv](https://github.com/astral-sh/uv)`, `mamba-ssm`, `causal-conv1d`.
- A Hugging Face token (gated tokenizer) and an NGC API key (model download).
