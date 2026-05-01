# Data Prep — When You Need `sft_packing`

| Downstream trainer | Need `prep/sft_packing`? | Why |
|---|---|---|
| `sft/megatron_bridge` | **Yes** | Megatron-Bridge SFT expects `packed_parquet`, not raw JSONL. |
| AutoModel / HuggingFace-style SFT | **No** | These workflows can read `training_jsonl` directly. |

## Use `sft_packing` when

- your next step is `sft/megatron_bridge`
- you need packed Parquet shards for high-throughput Megatron-Bridge loading
- you want chat-template application, tokenization, and packing materialized as a reusable converter stage

## Skip `sft_packing` when

- your trainer already consumes chat-format JSONL directly
- you are using AutoModel or another HF-native fine-tuning path
- you are still iterating on data shape and do not want to commit to a tokenizer/chat-template/pack-size yet

## Keep these settings aligned

- `tokenizer` must match the downstream training and log-prob evaluation tokenizer
- `pack_size` should match downstream `seq_length`
- `chat_template` should match the target model family

If you already have matching `packed_parquet`, reuse it and do not repack unless one of those three settings changes.
