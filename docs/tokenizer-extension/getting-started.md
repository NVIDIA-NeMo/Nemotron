---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Run tokenizer_extension/extend, init_embeddings, and evaluate end-to-end for one target language with the shipped default.yaml configurations."
topics: ["Tokenizer Extension", "Tutorial"]
tags: ["Tutorial", "Tokenizer Extension"]
content:
  type: "Tutorial"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

(getting-started-tokenizer-extension)=
# Getting Started With Tokenizer Extension

You will build a tokenizer that adds 30,000 target-language subwords to the Nemotron 3 Nano tokenizer, initialize the matching embedding rows in the base model, and measure the fertility of the result.

This tutorial runs three of the four `tokenizer_extension` steps in order.
Each step uses its `config/default.yaml` and CLI dotlist overrides.

:::{card}
What You Will Produce

An extended tokenizer under `./output/tokenizer_extension/add/`, a resized Hugging Face checkpoint under `./output/resized_checkpoint/`, and a fertility report JSON.

^^^

Overview

1. Install the `tokenizer-extension` extra.
2. Run `tokenizer_extension/extend` with `method=add`.
3. Run `tokenizer_extension/init_embeddings` with `arm=add`.
4. Run `tokenizer_extension/evaluate` on the extended tokenizer.

{octicon}`clock;1em;sd-mr-1` The `extend` and `evaluate` steps run on CPU and stream their corpora.
The `init_embeddings` step loads the base model and needs a GPU node.
:::

## Prerequisites

See {doc}`prerequisites` for the full requirements. For this tutorial you need:

- A Hugging Face dataset, or a local Parquet or JSON Lines corpus, in the target language.
  The `extend` default configuration points at `ai4bharat/sangraha` with `hf_name: verified` and `hf_split: hin`.
- A GPU node for `init_embeddings`. The base model, `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` by default, is loaded host-resident for embedding surgery; the GPU serves the auxiliary encoders, and the generated environment profiles request one GPU for this step.
- A `language` value from the registry in `languages.py`. The registered values are `hindi`, `marathi`, `nepali`, `sanskrit`, `bengali`, `punjabi`, `gujarati`, `odia`, `tamil`, `telugu`, `kannada`, `malayalam`, `urdu`, and `vietnamese`.

## Procedure

1. Clone the repository, if you have not already:

   ```console
   $ git clone https://github.com/NVIDIA-NeMo/Nemotron && cd Nemotron
   ```

1. Install the optional dependencies.

   `indic-nlp-library` is required when `language` selects the Devanagari normalizer (`hindi`, `marathi`, `nepali`, or `sanskrit`).
   `fasttext-wheel` is required only for `method: focus` in `init_embeddings`.

   ```console
   $ uv pip install -e '.[tokenizer-extension]'
   ```

1. Set the values that the following commands share.

   ```console
   $ L=hindi
   $ OUT=./output/tokenizer_extension
   ```

   Substitute a registered `language` value and a corpus that matches it.
   The commands below keep the corpus from `extend/config/default.yaml`; to use a different Hugging Face dataset, add `corpus.hf_dataset=<id> corpus.hf_name=<config> corpus.hf_split=<split>` overrides.

1. Build the extended tokenizer.

   ```console
   $ uv run nemotron steps run tokenizer_extension/extend -c default \
       language=$L method=add extension_size=30000 output_dir=$OUT
   ```

   The step writes `$OUT/add/`, which contains the tokenizer files and `summary.json`.
   Confirm that `tokens_spliced` in `summary.json` equals `extension_size`.
   The step logs a warning when the two values differ, because fertility and BPB comparisons between arms are valid only at a matched budget.

1. Initialize the new embedding rows and write the resized checkpoint.

   ```console
   $ uv run nemotron steps run tokenizer_extension/init_embeddings -c default \
       language=$L arm=add extended_tokenizer=$OUT/add \
       output_dir=./output/resized_checkpoint
   ```

   The default `method` is `subword` with `input_averaging: uniform`, which initializes each new row from the mean of its base-vocabulary subword rows.
   The `arm` value must match how `extend` built the tokenizer: `add` for `method=add` or `method=expand`, and `replace` for `method=replace`.

   The output directory is a Hugging Face checkpoint with weights and tokenizer.
   Pass it to `pretrain/megatron_bridge` as `hf_model_path` when you start continued pretraining.

1. Measure fertility.

   ```console
   $ uv run nemotron steps run tokenizer_extension/evaluate -c default \
       tokenizer=$OUT/add output=./output/eval/fertility_add.json
   ```

   The default evaluation corpus is `ai4bharat/samanantar` with `hf_config: hi` and `text_field: tgt`.
   Change the `corpus` block to match your target language.
   The report records `fertility`, `chars_per_token`, `unique_tokens_used`, and `vocab_coverage`.
   Lower fertility is better.
   To compare tokenizers, run the step once per tokenizer with the same `corpus` block.

1. Optional: Print the merged configuration without running a step.

   Pass `--dry-run` or `-d` to any of the commands above.

   ```console
   $ uv run nemotron steps run tokenizer_extension/extend -d -c default language=$L method=add
   ```

## Next Steps

- Compare the three extension methods: {doc}`explanation/extension-methods`
- Choose an initialization engine: {doc}`how-to/initialize-embeddings`
- Score the resized checkpoint with bits per byte: {doc}`how-to/evaluate-a-tokenizer`
- Run the steps on Lepton, DGX Cloud, or Slurm: {doc}`how-to/run-on-a-cluster`
- Field meanings: {doc}`reference/extend-config`
