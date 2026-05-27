# Choose a PEFT Backend

Parameter-efficient fine tuning (PEFT) in Nemotron is implemented as dedicated steps that emit adapter checkpoints. Pick the backend that matches your base checkpoint format and your data path.

## Options

| Step id | Best when | Primary inputs | Primary output artifact |
|---------|------------|------------------|---------------------------|
| `peft/automodel` | You have a Hugging Face base, chat-formatted JSON Lines (JSONL), and a small GPU count | `training_jsonl` | `checkpoint_lora` |
| `peft/megatron_bridge` | You have a Megatron base checkpoint and packed Apache Parquet at scale | `packed_parquet`, `checkpoint_megatron` | `checkpoint_lora` |

## Decision Flow

1. If you have one to four graphics processing units (GPUs) and JSON Lines (JSONL) chat data, use `peft/automodel`.
2. If you have eight or more GPUs, you already run Megatron packing, and you train adapters on a Megatron base, use `peft/megatron_bridge`.
3. If deployment requires a merged Hugging Face model, plan `convert/merge_lora` after training. Add any Megatron to Hugging Face conversion step that your pipeline needs before merge. Adapter evaluation scores are not identical to merged model scores.

## Sample Commands

```console
$ uv run nemotron steps run peft/automodel -c tiny
$ uv run nemotron steps run peft/megatron_bridge -c tiny
```

The Megatron Bridge path needs compatible packed Parquet and a base `checkpoint_megatron` path that you set in training configuration.

## Success Criteria

- You version adapter artifacts with base model id, data blend, rank, alpha, and target module set so you can reproduce runs.
- You re-evaluate after merge when production uses merged weights.

## Related Reading

- [Choose an SFT Backend](choose-sft-backend.md)
- [Data and Checkpoint Formats](data-and-checkpoint-formats.md)
