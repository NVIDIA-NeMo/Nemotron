# Choose an SFT Backend

Supervised fine tuning (SFT) is implemented by two interchangeable steps. Pick one step based on data format, checkpoint format, and scale.

## Options

| Step id | Best when | Primary input artifact | Primary output artifact |
|---------|-----------|------------------------|-------------------------|
| `sft/automodel` | You have OpenAI chat-formatted JSON Lines (JSONL), you want Hugging Face style checkpoints, or you want the smallest cluster footprint for iteration | `training_jsonl` | `checkpoint_hf` |
| `sft/megatron_bridge` | You need distributed Megatron Bridge training with packed sequences and an Apache Parquet pipeline | `packed_parquet` | `checkpoint_megatron` |

## Decision Flow

1. If your data is already chat-formatted JSON Lines (JSONL) and downstream tools expect Hugging Face safetensors, start with `sft/automodel`.
2. If your data is packed Parquet produced by the packing prep step, or you require Megatron distributed checkpoints without an export round trip, use `sft/megatron_bridge`.
3. If you start on one backend and later need the other output format, plan an explicit conversion step in your pipeline. Do not switch backends silently without conversion.

## Prerequisites for Megatron Bridge

Megatron Bridge SFT expects packed Parquet that is compatible with the tokenizer and sequence length you will use in training. The pack size in prep must match the training sequence length. If they diverge, you risk shape errors mid-run.

## Sample Commands

```console
$ uv run nemotron steps run sft/automodel -c tiny
$ uv run nemotron steps run sft/megatron_bridge -c tiny
```

## Success Criteria

- The commands `nemotron steps show sft/automodel` and `nemotron steps show sft/megatron_bridge` list the `consumes` types your workspace must provide.
- Loss decreases on a small slice before you scale data or learning rate.
- Tokenizer, chat template, and sequence length stay aligned with evaluation and with any later reinforcement learning (RL) step that reuses the policy.

## Related Reading

- [Data and Checkpoint Formats](data-and-checkpoint-formats.md)
- [Artifact Graph](../explanation/artifact-graph.md)
