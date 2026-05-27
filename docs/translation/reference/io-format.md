---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Input and output expectations for nemotron steps run translate/nemo_curator."
topics: ["Translation", "Schema"]
tags: ["Reference", "JSONL"]
content:
  type: "Reference"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

# Input and Output Format

Expectations for `input_path` layouts, shard layout under `output_dir`, `output_mode`, chat reconstruction, and FAITH columns.

## Inputs

Supported layouts include JSON Lines (JSONL), with one JSON object per line, or Apache Parquet columnar files.

- JSONL records contain arbitrary JSON objects per line. The `text_field` setting selects which strings `TranslationStage` visits. Wildcards expand across arrays such as `messages` items.
- Parquet inputs share the same logical schema. The Curator `ParquetReader` partitions row groups according to `files_per_partition` and `blocksize` overrides.

Avoid pointing `input_path` at directories that mix `.jsonl` and `.parquet` shards when `input_format=auto`. The reader raises an error instead of guessing.

## Outputs

The `JsonlWriter` and `ParquetWriter` emit multiple shards under `output_dir` because Curator pipelines partition work for parallelism. Downstream steps should treat `output_dir` as the artifact, not individual filenames.

### `output_mode`

| Mode | Producer behavior |
|------|-------------------|
| `replaced` | Source strings at `text_field` are overwritten with translations suitable for immediate training. |
| `raw` | Original strings remain. Translations and metadata appear in auxiliary columns defined by `TranslationStage`. |
| `both` | Emits replaced views and retains intermediate structures. This is the starter default for auditing. |

### Chat Reconstruction

When `reconstruct_messages=true`, expect parallel arrays such as `translated_messages` mirroring the `messages` layout but with translated `content` entries. This simplifies quality assurance review without sacrificing structured tool-call payloads.

### FAITH Annotations

When `faith_eval.enabled=true` and `merge_scores=true`, each record carries score blobs aligned with segment boundaries. When `filter_enabled=true`, low-trust rows disappear entirely from `output_dir` shards.

## Sampling Outputs

Use `find`, `head`, and `python3 -m json.tool` on any emitted shard to inspect one translated row:

```bash
find ./output_dir -name '*.jsonl' | head -n 1 | xargs head -n 1 | python3 -m json.tool --no-ensure-ascii
```

## Related Pages

- Field tuning guide: {doc}`../how-to/configure-fields-and-output`
- YAML defaults: {doc}`translate-config`
