---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Switch segmentation_mode between coarse and fine modes for nemotron steps run translate/nemo_curator."
topics: ["Translation", "Segmentation"]
tags: ["How-To", "Translation"]
content:
  type: "How-To"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

# Use Fine Segmentation

Use this guide when `segmentation_mode=coarse` mishandles long prose and you want to try `fine` on a sample before scaling.

Conceptual background lives in {doc}`../explanation/segmentation`.

## Procedure

1. Reproduce an issue with `segmentation_mode=coarse`, which is the default in `default.yaml`.
2. Retry on a slice with `segmentation_mode=fine`:

```bash
uv run nemotron steps run translate/nemo_curator -c default \
  segmentation_mode=fine \
  input_path=/path/to/sample.jsonl \
  output_dir=/path/to/out-fine \
  source_language=en \
  target_language=de \
  server.model=YOUR_LLM_MODEL_ID
```

3. Compare reconstruction fidelity against throughput cost. Fine mode increases segment counts.

## Pair With `min_segment_chars`

Raise `min_segment_chars` modestly to skip trivial whitespace-only fragments if fine mode becomes noisy.

## Related Pages

- Conceptual background: {doc}`../explanation/segmentation`
- YAML keys: {doc}`../reference/translate-config`
