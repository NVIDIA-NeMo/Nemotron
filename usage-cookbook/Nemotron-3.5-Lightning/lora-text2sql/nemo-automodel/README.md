# Nemotron-3.5-Lightning — LoRA Fine-Tuning on Text2SQL with NeMo AutoModel

LoRA fine-tuning of Nemotron-3.5-Lightning-30B-A3B-BF16 — a 30B-parameter hybrid
Mamba-Transformer Mixture-of-Experts with ~3B active parameters per token and a
Multi-Token Prediction (MTP) head — on the BIRD-SQL text-to-SQL task, using
[NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel). Everything runs from
`[automodel_lightning35_lora_cookbook.ipynb](automodel_lightning35_lora_cookbook.ipynb)`.

## Steps

1. **Copy artifacts** — wire the YAML recipe and `text2sql.py` into the AutoModel tree. *(CPU)*
2. **Prepare data** — build a [BIRD](https://huggingface.co/datasets/xu3kev/BIRD-SQL-data-train)
  `training.jsonl` / `validation.jsonl` / `test.jsonl` from the no-reasoning and reasoning splits. *(CPU)*
3. **Fine-tune** with LoRA via the `automodel` CLI on a single H100 node. *(GPU)*
4. **Evaluate** — before/after generation + BIRD SQL execution accuracy. *(GPU)*
5. **Serve** with vLLM (adapter-swap or merged checkpoint). *(GPU)*

## Hardware

Single-node run — `ep_size: 8` must equal `nproc-per-node`:


| Target   | Minimum allocation                      | `ep_size` | Launch               |
| -------- | --------------------------------------- | --------- | -------------------- |
| **H100** | **1 × H100 node — 8 GPUs (80 GB each)** | `8`       | `--nproc-per-node=8` |


Plus the base checkpoint from HF Hub: **~62 GB on disk (BF16)**. CUDA 12.1+, Python 3.10+,
`uv`, `mamba-ssm`, `causal-conv1d`, and an HF token (gated model + BIRD dataset).

## Files


| File                                           | What it is                                            | Copy into AutoModel at                    |
| ---------------------------------------------- | ----------------------------------------------------- | ----------------------------------------- |
| `automodel_lightning35_lora_cookbook.ipynb`    | The main notebook — start here.                       | — (run from this directory)               |
| `nemotron_mtp_lightning35_hellaswag_peft.yaml` | LoRA recipe — MTP + deepep + FSDP2, H100 single node. | `examples/llm_finetune/nemotron/`         |
| `text2sql.py`                                  | Dataset target (`make_text2sql_dataset`).             | `nemo_automodel/components/datasets/llm/` |


> The notebook's Section 2 copies the YAML and `text2sql.py` into the AutoModel tree automatically.

