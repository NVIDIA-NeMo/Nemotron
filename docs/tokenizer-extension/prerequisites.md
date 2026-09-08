---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Software, hardware, data, and access requirements for the tokenizer_extension steps."
topics: ["Tokenizer Extension", "Setup"]
tags: ["Reference", "Prerequisites"]
content:
  type: "Reference"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

# Prerequisites

## Software

- A Nemotron checkout with the project environment synchronized with `uv sync`; see {doc}`../steps/getting-started`.
- The optional `tokenizer-extension` extra:

  ```console
  $ uv pip install -e '.[tokenizer-extension]'
  ```

  The extra installs two packages.
  `indic-nlp-library` supplies the Devanagari normalizer; a language whose profile uses that normalizer raises an `ImportError` without it.
  `fasttext-wheel` is required for `method: focus` in `init_embeddings` and is not imported by the other methods.
  The generated Lepton, Slurm, and DGX Cloud profiles install these packages at job start-up.

## Hardware

| Step | Resources |
|------|-----------|
| `extend` | CPU only. The BPE trainer holds the word-count table in host memory; the corpus limits in `corpus.*` control its size. |
| `init_embeddings` | One GPU for every method except `subword.gemma_weighted` on a large Gemma model, which may require several. The base model is loaded on the host. |
| `evaluate` | CPU only. |
| `eval_init` | GPU; each model is scored in turn. |

## Data and Model Access

- A base checkpoint that Hugging Face `transformers` can load. `extend` and `evaluate` read only the tokenizer; `init_embeddings` and `eval_init` load the weights. The shipped defaults use `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` and its `-Base-` variant with `trust_remote_code: true`; replace them with the checkpoint you extend.
- A target-language corpus, as a Hugging Face dataset or as local Parquet or JSON Lines files with a text column. The guidebook's measurements used 10,000 to 100,000 diverse documents.
- A held-out evaluation corpus in the target language that is distinct from the training corpus, for `evaluate` and `eval_init`.
- Network access to Hugging Face, or pre-downloaded models and datasets in the cache for an air-gapped site; see {doc}`../steps/airgap`.

## Language Coverage

The `language` value must be a registered profile.
The registered names are listed in {doc}`how-to/add-a-language`, which also describes how to register a new one.
