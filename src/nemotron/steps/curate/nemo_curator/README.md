---
name: nemotron-curate-nemo-curator
description: Configure Nemotron curate/nemo_curator to read JSONL text, optionally hydrate a Hugging Face dataset snapshot, apply light NeMo Curator language, word-count, and domain filters, and write filtered_jsonl for translate or data_prep steps.
---

# Lightweight Text Curation (NeMo Curator)

Use `curate/nemo_curator` to turn JSONL text into `filtered_jsonl` that can
feed translation, pretraining prep, or SFT prep.

Read `step.toml` for the full strategy/error matrix.

## Current runner

The step is intentionally small:

`JsonlReader -> optional FastText language filter -> optional WordCountFilter -> optional MultilingualDomainClassifier -> JsonlWriter`

It can call `huggingface_hub.snapshot_download` before reading if `dataset` is
set in YAML. It does not implement Common Crawl extraction, URL crawling, or
deduplication itself; use a dedicated Curator recipe for those before this step
or add them as a separate step.

## Inputs and outputs

- Consume: `raw_jsonl` files matched by `input_glob`. If `dataset` is set, the
  Hugging Face snapshot is downloaded first and `input_glob` should point into
  that local snapshot.
- Produce: JSONL shards under `output_dir`. Language/domain fields appear only
  when the corresponding filters are enabled.

## Configure

- Set `input_glob`, `output_dir`, and `text_field` first.
- Set `dataset: null` for local files. Set `dataset.repo_id`,
  `dataset.repo_type`, `dataset.local_dir`, and optional `allow_patterns` for a
  Hugging Face snapshot.
- Set `language_codes: []` to skip FastText language filtering. If non-empty,
  provide `models.fasttext_langid`.
- Set `quality_filters: {}` to skip word-count filters. If either `min_words`
  or `max_words` is set, set both.
- Set `domains: []` to skip domain classification. If non-empty, provide
  `models.hf_cache_dir` when you need a persistent model cache.
- On small CPU Lepton runs, use the Curator container as-is and set
  `NEMOTRON_CURATOR_RAY_NUM_CPUS=4` through the env profile when the YAML does
  not include `ray.num_cpus`.
- Reference [src/nemotron/steps/patterns/data-quality-before-quantity.md](../../patterns/data-quality-before-quantity.md)
  before scaling corpus size or tightening filters.

## Smoke commands

```bash
uv run nemotron steps run curate/nemo_curator -c tiny -r lepton_curate
```

```bash
uv run lep log get -j curate-nemo-curator-step-xxxx --limit 300
```

## Local files

- Contract: [step.toml](step.toml)
- Runner: [step.py](step.py)
- Configs: `config/default.yaml`, `config/tiny.yaml`

## Guardrails

- Don't enable every optional filter on the first run. Start with `tiny` or
  local JSONL plus no filters, then add language, word-count, and domain gates.
- Inspect intermediate JSONL when output is empty or tiny — usually a filter
  is set too aggressively.
- Split very large input files before reading; OOMs usually come from oversized
  partitions.
