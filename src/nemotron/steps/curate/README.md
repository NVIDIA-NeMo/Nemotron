# Nemotron Curation

Use this category to turn raw or third-party JSONL into a filtered corpus that
can feed translation, pretraining prep, or SFT prep.

## Developer Journey

1. Identify the raw source: local JSONL or a Hugging Face snapshot.
2. Run the curate step with all optional filters disabled to verify the
   reader/writer path.
3. Add language, word-count, or domain filters one at a time.
4. Inspect intermediate shards after each filter change — empty output usually
   means a filter is too aggressive.
5. Hand the filtered JSONL to translation or data prep.

## Steps

| Need | Step | Input | Output |
|---|---|---|---|
| Lightweight JSONL filtering with optional language/word-count/domain gates | [`curate/nemo_curator`](nemo_curator/README.md) | `raw_jsonl` (or HF snapshot) | `filtered_jsonl` |

## Data And Artifact Flow

```text
raw_jsonl / HF snapshot
  -> curate/nemo_curator (JsonlReader -> optional filters -> JsonlWriter)
  -> filtered_jsonl
  -> translate/* or data_prep/*
```

This category is intentionally lightweight. Deduplication, crawling, and full
web extraction belong in dedicated NeMo Curator recipes, not this step.

## Guardrails

- Don't enable every filter on the first run.
- Inspect intermediate JSONL before tightening filters.
- Split very large input files before reading; OOMs usually come from
  oversized partitions.
