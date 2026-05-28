---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Input and output format for curate/nemo_curator."
topics: ["Curation", "Reference", "IO"]
tags: ["Reference", "JSONL", "Curation"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Data Scientist"]
---

# Curation Input and Output Format

## Input

Input files must be JSON Lines.
Each line is a JSON object.
The configured `text_field` must exist on each record.

```json
{"id": "doc-001", "text": "The text to curate."}
```

By default, `text_field` is `text`.

## Output

The step writes JSONL shards under `output_dir` using NeMo Curator `JsonlWriter`.
The output contains the fields read by `JsonlReader`, plus filter or classifier fields when those stages are enabled.

Typical no-filter output:

```json
{"text": "The text to curate."}
```

When language filtering is enabled, the pipeline adds a language score field used by the filter.
When domain classification is enabled, classifier output fields depend on the installed NeMo Curator classifier implementation.

## Downstream Use

Use the output as `filtered_jsonl`.
Common downstream paths are:

- Use `translate/nemo_curator` for corpus translation.
- Use `data_prep/pretrain_prep` for pretraining data preparation.
- Use `data_prep/sft_packing` when the curated records are already in the required supervised fine-tuning (SFT) format.

If a downstream step needs fields beyond `text`, verify that the curation reader/writer path preserves those fields before scaling the run.
