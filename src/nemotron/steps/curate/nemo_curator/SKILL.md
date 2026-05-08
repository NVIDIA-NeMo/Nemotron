---
name: nemotron-curate-nemo-curator
description: Configure Nemotron curate/nemo_curator to acquire and filter web/public text data into filtered_jsonl with language and domain annotations. Use for Common Crawl, Wikipedia, custom URL lists, or local JSONL/Parquet inputs that need quality, language, or domain gating before training.
---

# Data Acquisition & Curation (NeMo Curator)

Use `curate/nemo_curator` to turn raw or remote text into `filtered_jsonl`
with language and domain annotations downstream training can rely on.

Read `step.toml` for the full strategy/error matrix.

## Inputs and outputs

- Consume: external sources (Common Crawl, Wikipedia, URL lists) or local
  JSONL/Parquet. The step doesn't declare an explicit `[[consumes]]` because
  acquisition is the first step in many pipelines.
- Produce: `filtered_jsonl` (with language/domain annotations).

## Configure

- **Pick the entry stage based on data location:**
  - Remote public sources → Curator download/extract composite stages.
  - Local JSONL/Parquet → `JsonlReader` / `ParquetReader`, skip acquisition.
- **Stack filters in this order**: heuristic ScoreFilter → QualityClassifier
  → deduplication. Don't reverse the order — quality classifiers are
  expensive and benefit from heuristic prefiltering.
- **Multilingual gating** needs FastText `lid.176` (the `missing_language_model`
  error recovery is "download lid.176 and wire its path"). Have the path ready
  before enabling language filters.
- **Domain gating**: FastText language ID first, then `DomainClassifier` /
  `MultilingualDomainClassifier`.
- Reference [src/nemotron/steps/patterns/data-quality-before-quantity.md](../../patterns/data-quality-before-quantity.md)
  before tuning `num_records` upward.

## Local files

- Contract: [step.toml](step.toml)
- Runner: [step.py](step.py)
- Configs: `config/default.yaml`, `config/tiny.yaml`

## Guardrails

- Don't enable everything at once. Filter with heuristics first; classifiers
  and dedupe come after the corpus is small enough to iterate on.
- Inspect intermediate JSONL when output is empty or tiny — usually a filter
  is set too aggressively.
- Split very large input files before reading; OOMs come from oversized
  partitions, not Curator itself.
