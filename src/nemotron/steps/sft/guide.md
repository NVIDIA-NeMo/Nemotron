# SFT — Choosing a Backend

| Backend | Best for | Min GPUs | Data format | Checkpoint format |
|---------|----------|----------|-------------|-------------------|
| **Megatron-Bridge** | Large-scale distributed training with TP / PP / CP control | 8 | packed_parquet (needs `prep/sft_packing`) | checkpoint_megatron |
| **AutoModel** | Simpler setup, fewer GPUs, LoRA / PEFT, quick iteration | 4 | training_jsonl (no packing) | checkpoint_hf |

## Quick decision tree

- Need TP / PP / CP parallelism or official Nano3 / Super3 recipe patterns? → **Megatron-Bridge**
- Have fewer than 8 GPUs? → **AutoModel**
- Want LoRA with minimal setup? → **AutoModel**
- Need the highest-throughput multi-node path? → **Megatron-Bridge**
- Just want to get SFT running fast on existing JSONL? → **AutoModel**

## Impact on the pipeline

### If you choose Megatron-Bridge
- Add `prep/sft_packing` upstream.
- Input artifact becomes `packed_parquet`.
- Output artifact is `checkpoint_megatron`.
- If you later need HuggingFace format, add a conversion step.

### If you choose AutoModel
- No packing step is required.
- The step reads `training_jsonl` directly.
- Output artifact is `checkpoint_hf`.
- LoRA / PEFT is the default starting point for small GPU counts.

## Rule of thumb

Use **Megatron-Bridge** when the training problem is large enough that distributed parallelism strategy is the main design decision.

Use **AutoModel** when the user is GPU-constrained, wants HuggingFace-native outputs, or needs the simplest path from JSONL to an SFT checkpoint.
