---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Run tokenizer_extension/init_embeddings with the correct arm and engine to produce a resized Hugging Face checkpoint for continued pretraining."
topics: ["Tokenizer Extension", "Embeddings"]
tags: ["How-To", "Tokenizer Extension"]
content:
  type: "How-To"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Initialize Embeddings

Use `tokenizer_extension/init_embeddings` to attach an extended tokenizer to the base model, initialize the new embedding and language-model-head rows, and save a resized Hugging Face checkpoint.
The step needs a GPU node.

## Before You Begin

- Finish {doc}`extend-a-tokenizer`; you need the `output_dir/<method>/` directory it wrote.
- Confirm that `base_model` shares the tokenizer that `extend` started from. The `row_count_mismatch` error means the model's embedding rows do not equal the base tokenizer size.
- For `method: focus`, install `fasttext-wheel` and download a fastText `.bin` for the language.

## Match `arm` to the Extension Method

| `extend` `method` | Set `arm` to | `extended_tokenizer` |
|-------------------|--------------|----------------------|
| `add` | `add` | `output_dir/add` |
| `expand` | `add` | `output_dir/expand` |
| `replace` | `replace` | `output_dir/replace` |

`arm=replace` reads `id_remap.json` from the tokenizer directory and permutes the survivor rows through it.
Pointing `arm=add` at a Replace tokenizer raises `replace_tokenizer_rejected`.

## Run With the Default Engine

The default is `method: subword` with `input_averaging: uniform` and `output_averaging: uniform`, which needs no auxiliary model.

```console
$ uv run nemotron steps run tokenizer_extension/init_embeddings -c default \
    language=<lang> arm=add extended_tokenizer=./output/tokenizer_extension/add \
    output_dir=./output/resized_checkpoint
```

For a Replace tokenizer:

```console
$ uv run nemotron steps run tokenizer_extension/init_embeddings -c default \
    language=<lang> arm=replace extended_tokenizer=./output/tokenizer_extension/replace \
    output_dir=./output/resized_checkpoint_replace
```

## Select Another Engine

Baseline floor:

```console
$ uv run nemotron steps run tokenizer_extension/init_embeddings -c default \
    language=<lang> arm=add method=baseline baseline.mode=mean_target
```

Encoder-weighted subword averaging:

```console
$ uv run nemotron steps run tokenizer_extension/init_embeddings -c default \
    language=<lang> arm=add method=subword \
    subword.input_averaging=bert_weighted subword.output_averaging=bert_weighted
```

Leave `subword.bert_model` unset so that the language profile supplies an encoder that covers the language.
An explicit `bert_model` overrides the profile; `default.yaml` records that this is how a Vietnamese run once weighted with MuRIL, which does not cover Vietnamese.

FOCUS:

```console
$ uv run nemotron steps run tokenizer_extension/init_embeddings -c default \
    language=<lang> arm=add method=focus focus.fasttext_model=<path/to/cc.xx.300.bin>
```

Not every engine is available for `arm=replace`; see the support table in {doc}`../explanation/embedding-initialization`.
An unsupported combination fails with an explicit message rather than substituting another engine.

## Iterate Before the Full Run

For a low-cost verification run, override `base_model` with a smaller model that shares the tokenizer and use a small `extension_size` in `extend`, so that the configuration is validated before the full-size job.

## Verify the Result

`output_dir` contains the weights, `config.json`, and the tokenizer files.
Load it with `AutoTokenizer` and `AutoModelForCausalLM` from `transformers` to confirm that the vocabulary size equals the embedding row count, then pass the path to `pretrain/megatron_bridge` as `hf_model_path`.

## Next Steps

- Score the checkpoint with BPB: {doc}`evaluate-a-tokenizer`
- Field reference: {doc}`../reference/init-embeddings-config`
