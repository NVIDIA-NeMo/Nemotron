---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Troubleshooting reference for curate/nemo_curator."
topics: ["Curation", "Reference", "Troubleshooting"]
tags: ["Troubleshooting", "Curation", "NeMo Curator"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Data Scientist"]
---

# Curation Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `input_glob` matches no files | Path does not exist in the current host, container, or shared mount | Use an absolute path or verify the mount. For local tiny runs, override the packaged `/nemo_run/code/...` path with `${PWD}/src/nemotron/steps/curate/nemo_curator/data/tiny.jsonl`. |
| Missing FastText model | `language_codes` is non-empty but `models.fasttext_langid` is missing or invalid | Set `language_codes=[]` to disable language filtering, or provide the FastText `lid.176.bin` path. |
| `quality_filters` error about `min_words` and `max_words` | Only one word-count bound was set | Set both `quality_filters.min_words` and `quality_filters.max_words`, or set `quality_filters={}`. |
| Output is empty or much smaller than expected | Filters are too strict or applied before corpus shape is understood | Re-run with `language_codes=[]`, `domains=[]`, and `quality_filters={}`. Add filters back one at a time. |
| Ray worker starts a new `.venv` or cannot import dependencies | Local `uv run` and the Ray runtime environment are both attempting to manage dependency setup | Export `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` and run with `uv run --no-sync` after `uv sync --extra curate`. |
| Large local file causes memory pressure | Input shard is too large for the available Ray worker memory | Split large JSONL files into smaller shards before running Curator. |
| Domain classifier downloads repeatedly | Hugging Face cache path is not persistent | Set `models.hf_cache_dir` to a persistent cache location and mount it in remote profiles. |

## Debug Checklist

1. Run the local tiny command with filters disabled.
2. Confirm `uv sync --extra curate` completed in the same repository clone.
3. Confirm the input path exists where the command runs.
4. Confirm `output_dir` is writable.
5. Add language, word-count, and domain filters one at a time.
