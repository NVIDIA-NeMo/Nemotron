---
id: translate-training-corpus
title: "Translate corpora before multilingual training"
tags: [translate, multilingual, data-prep]
triggers:
  - "The user wants to translate a corpus, dataset, or chat records before CPT or SFT."
  - "Training data must be produced in a target language from source-language examples."
  - "A multilingual fine-tuning pipeline needs translated JSONL or Parquet artifacts."
steps: [translate/translation]
confidence: high
---

## When to apply

Use this when translation is part of the data pipeline for training or curation. The output should be row-oriented corpus data that can feed preparation, packing, SFT, CPT, or downstream review.

Do not apply it to benchmark-only translation. Benchmark translation has different schema and scoring requirements and should not be mixed with training-corpus translation.

## What to do

Insert `translate/translation` before packing or training. Ask for explicit source and target language codes, the input format, and the field path to translate.

Prefer Curator-native reader -> `TranslationStage` -> writer flow. Do not generate custom pandas chunking unless the user has one huge single file and Curator file partitioning is not enough.

Use `output_mode=both` when auditability matters, because it preserves translated fields, metadata, and optional quality scores.

After translation, apply `multilingual-tokenizer-check` before SFT so sequence length and packing assumptions reflect the translated language.

## Exceptions

Skip this pattern when the user already has target-language training data or only needs model evaluation.

If translation is a one-off data-inspection task with no downstream training artifact, keep the pipeline small and do not force SFT/prep stages.
