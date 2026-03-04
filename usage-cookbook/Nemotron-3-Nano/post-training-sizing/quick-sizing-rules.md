# Post-Training GPU Memory Estimation: Nemotron 3 Nano 30B-A3B

> GPU sizing for LoRA fine-tuning, full SFT, and GRPO of
> [NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
> across three NeMo training pathways: **Megatron-Bridge**, **Automodel**, and
> **NeMo RL**.
>
> For detailed memory breakdowns, derivations, and activation scaling see
> [Training Memory Estimation Details](./training-memory-details.md).

---

## Quick Sizing Rules

### Heuristic for rough estimation

`NOTE`: Numbers below use H100 80 GiB as a reference. GPU counts for MoE models are **not**
simply total bytes ÷ device memory — parallel topology (EP, TP, DP divisibility) and
recipe defaults set the floor. See the [Memory Floor](#memory-floor-analytical-estimates)
table for minimums.

- **Full SFT** — **~16 bytes/param** static memory (2B BF16 weights + 2B BF16 gradients + 12B Adam FP32),
  plus activation memory that scales with sequence length and micro-batch size.

  For 31.6B params in Nemotron 3 Nano → **~471 GiB** static, plus activations. Per-GPU memory is dominated
  by how [EP](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-reference/distributed_checkpointing.html#expert-parallelism)
  (Expert Parallelism) distributes the 128 routed experts across GPUs:
  - With FSDP2 (Automodel): fits on **1 node (8 GPUs)** at EP=8
  - Without FSDP2 (Megatron-Bridge): needs DP≥2 to shard optimizer states → **16 GPUs** (EP=8, DP=2)
. Recommended recipes ship with [16 GPUs (2 nodes)](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/nemotron3.html) for headroom

- **LoRA** — **~2 bytes/param** static memory (frozen BF16 weights; adapter and optimizer
  overhead is negligible at low ranks), plus activations.

  For 31.6B params → **~59 GiB** static + activations. Fits on 1 GPU at EP=1 (tight),
  2 GPUs at EP=2 (comfortable), 4–8 GPUs with larger EP for headroom.

- **GRPO / RL** — training side follows the Full SFT or LoRA rule above, **plus** additional
  GPUs for inference/generation if training and generation are not colocated.

> These are **memory-floor estimates** at seq_len=2048 and BF16 precision. Longer
> sequences, larger micro-batch sizes, higher LoRA ranks, or mixed-precision
> choices will shift the numbers — see [Activation Memory & Sequence Length](./training-memory-details.md#activation-memory--sequence-length) for details.

### Memory Floor (Analytical Lower Bound)

Minimum number of GPUs where the model training fits in memory.

> Reference GPU: **H100 80 GiB**. Scale GPU counts proportionally for other devices and their memory capacity.

| Framework | Training Mode | GPUs | EP | TP | PP | DP | Seq Len |
|-----------|---------------|:----:|:--:|:--:|:--:|:--:|--------:|
| **Megatron-Bridge** | LoRA | 1 | 1 | 1 | 1 | 1 | 2048 |
| **Megatron-Bridge** | Full SFT | 16 | 8 | 1 | 1 | 2 | 2048 |
| **Automodel** | LoRA | 1 | 1 | — | — | 1 | 2048 |
| **Automodel** | Full SFT | 8 | 8 | — | — | 8 | 2048 |
| **NeMo RL** | Full SFT | 16 | 8 | — | — | 16 | 2048 |
| **NeMo RL** | GRPO† | 16 | 8 | 4* | — | 16 | 2048 |

> Automodel and NeMo RL use DTensor (FSDP2) for weight sharding. NeMo RL also
> supports Megatron-Bridge as an alternative training backend (pick one, not both).
> †NeMo RL GRPO recipe uses LoRA rank 128 for the training side, which increases optimizer memory; GRPO also
> needs GPUs for both training and generation (vLLM) simultaneously, driving the floor to 16.
> *TP=4 in GRPO is for the vLLM generation engine, not the training side.

### Tested / Recommended Recipes

Shipped tested recipe defaults with headroom.

> Reference GPU: **H100 80 GiB**.

| Framework | Training Mode | GPUs | EP | TP | PP | DP | Seq Len | Recipe Config |
|-----------|---------------|:----:|:--:|:--:|:--:|:--:|--------:|--------|
| [**Megatron-Bridge**](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/nemotron3.html#lora-fine-tuning) | LoRA | 8 | 8 | 1 | 1 | 1 | 2048 | [Config](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/nemotronh/nemotron_3_nano.py) |
| [**Megatron-Bridge**](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/nemotron3.html) | Full SFT | 16 | 8 | 1 | 1 | 2 | 2048 | [Config](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/nemotronh/nemotron_3_nano.py) |
| [**Automodel**](https://docs.nvidia.com/nemo/automodel/latest/guides/llm-finetuning.html) | LoRA | 8 | 8 | — | — | 8 | 2048 | [Config](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/nemotron/nemotron_nano_v3_hellaswag_peft.yaml) |
| [**Automodel**](https://docs.nvidia.com/nemo/automodel/latest/guides/llm-finetuning.html) | Full SFT | 8 | 8 | — | — | 8 | 2048 | [Config](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/nemotron/nemotron_nano_v3_hellaswag_peft.yaml) |
| [**NeMo RL**](https://docs.nvidia.com/nemo/rl/latest/guides/sft.html) | Full SFT | 16 (2×8) | 8 | — | — | 16 | 2048 | [Config](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/recipes/llm/sft-nanov3-30BA3B-2n8g-fsdp2.yaml) |
| [**NeMo RL**](https://docs.nvidia.com/nemo/rl/latest/guides/grpo.html) | GRPO | 16 (2×8) | 8 | 4* | — | 16 | 2048 | [Full](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-fsdp2.yaml), [LoRA](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron-lora.yaml) |

> *TP=4 in GRPO is for the vLLM generation engine, not the training side.

### Notes

- **Target hardware:** The data above assumes **H100 80 GiB SXM** with **BF16 precision**.
   A B200 (192 GiB) could hold the full model at EP=1; an A100 40 GiB would need
   more GPUs. All memory figures use binary GiB (1 GiB = 2^30), matching `nvidia-smi`.
