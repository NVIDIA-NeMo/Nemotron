# Quantization (PTQ)

This stage quantizes the pretrained Nemotron 3 Super model using [Megatron-Bridge](../nvidia-stack.md#megatron-bridge)'s post-training quantization (PTQ) pipeline.

---

## Quantization Configurations

Nemotron 3 Super supports four quantization configurations tailored for the Mamba-MoE architecture:

| Config Name | Format | Description |
|---|---|---|
| `mamba_moe_fp8_aggressive` | FP8 | Aggressive FP8 quantization for Mamba-MoE |
| `mamba_moe_fp8_conservative` | FP8 | Conservative FP8 quantization for Mamba-MoE |
| `mamba_moe_nvfp4_aggressive` | NVFP4 | Aggressive NVFP4 quantization for Mamba-MoE |
| `mamba_moe_nvfp4_conservative` | NVFP4 | Conservative NVFP4 quantization for Mamba-MoE |

Pass the desired config name via `--export-quant-cfg` to `quantize.py`.

---

## Recipe Execution

### Direct Script Execution (Megatron-Bridge)

For direct execution outside this CLI, use the scripts in the [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) repository:

```bash
# Clone the repository and checkout the super-v3 branch
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge
git checkout super-v3
```

### Quantize

```bash
export HF_MODEL=/path/to/hf/model
export MEGATRON_SAVE_PATH=/path/to/quantized/megatron/ckpt

torchrun --nproc_per_node=16 examples/quantization/quantize.py \
    --hf-model-id $HF_MODEL \
    --export-quant-cfg mamba_moe_nvfp4_conservative \
    --megatron-save-path $MEGATRON_SAVE_PATH \
    --pp 2 \
    --tp 8 \
    --ep 8 \
    --trust-remote-code
```

### Resume Quantized Megatron Checkpoint and Generate

```bash
torchrun --nproc_per_node=16 examples/quantization/ptq_generate.py \
    --hf-model-id $HF_MODEL \
    --megatron-load-path $MEGATRON_SAVE_PATH \
    --pp 2 \
    --tp 8 \
    --ep 8 \
    --trust-remote-code
```

### Export Quantized Megatron Checkpoint to Huggingface Checkpoint

After quantization, export the Megatron checkpoint back to Hugging Face format:

```bash
export EXPORT_DIR=/path/to/output/hf/ckpt

torchrun --nproc_per_node=16 examples/quantization/export.py \
    --hf-model-id $HF_MODEL \
    --megatron-load-path $MEGATRON_SAVE_PATH \
    --export-dir $EXPORT_DIR \
    --pp 8 \
    --dtype bfloat16 \
    --trust-remote-code
```

Notes:
- For multi-node setups (e.g. 2 nodes with 8× H100), increase `--pp` accordingly (e.g. `--pp 2`) and use a job scheduler like SLURM to launch across nodes.

---

## Infrastructure

This stage uses the following components from the [NVIDIA AI Stack](../nvidia-stack.md):

| Component | Role | Documentation |
|-----------|------|---------------|
| [Megatron-Core](../nvidia-stack.md#megatron-core) | Distributed training primitives (TP, PP, EP) | [GitHub](https://github.com/NVIDIA/Megatron-LM) |
| [Megatron-Bridge](../nvidia-stack.md#megatron-bridge) | PTQ quantization, checkpoint export | [Docs](https://docs.nvidia.com/nemo/megatron-bridge/latest/) |
| [Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) | Quantization algorithms (FP8, NVFP4) | [GitHub](https://github.com/NVIDIA/TensorRT-Model-Optimizer) |

### Parallelism Configuration

| Parallelism | Default | Flag |
|-------------|---------|------|
| Tensor (TP) | 8 | `--tp` |
| Pipeline (PP) | 2 | `--pp` |
| Expert (EP) | 8 | `--ep` |

**Minimum resources:** 2 nodes with 8× H100 GPUs.

---

## Reference

- [Megatron-Bridge Nemotron 3 Super](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/super-v3/docs/models/llm/nemotron3-super.md) — MB documentation and examples
- [NVIDIA AI Stack](../nvidia-stack.md) — Megatron-Core, Megatron-Bridge documentation
- [Stage 0: Pretraining](./pretrain.md) — Pretrain the base model
- [Stage 1: SFT](./sft.md) — Supervised fine-tuning
- [Back to Overview](./README.md)
