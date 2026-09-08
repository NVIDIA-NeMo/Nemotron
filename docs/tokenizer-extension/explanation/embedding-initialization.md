---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "How tokenizer_extension/init_embeddings places new embedding rows: the baseline, subword, and FOCUS engines, norm correction, and Add versus Replace surgery."
topics: ["Tokenizer Extension", "Embeddings"]
tags: ["Explanation", "Tokenizer Extension"]
content:
  type: "Explanation"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Embedding Initialization

When a tokenizer gains new tokens, the model needs new rows in its input embedding matrix and in its language-model head.
The `init_embeddings` step decides where those rows start in embedding space.
The three engines differ in what information they use to place a new token.

## Engines

`embeddings.py` dispatches on `method` to a separate engine script.
`METHODS` is `("baseline", "subword", "focus")`.

### `baseline`: No Per-Token Information

| `baseline.mode` | New row is initialized from |
|-----------------|-----------------------------|
| `hf_default` | The Hugging Face resize: a multivariate normal fitted to the mean and covariance of the existing embeddings |
| `mean_all` | The mean of every existing embedding |
| `mean_target` | The mean of the base model's existing target-language rows only |

Every new token receives essentially the same vector, so the model must learn the distinctions during continued pretraining.
The engine is inexpensive and provides a floor for comparison.
`mean_hindi` is accepted as a legacy alias for `mean_target`.

### `subword`: Compose From Known Pieces

Each new token is decomposed into subwords of the original vocabulary and initialized from a weighted average of their rows.
Input and output sides are weighted independently through `subword.input_averaging` and `subword.output_averaging`.

| Value | Weighting |
|-------|-----------|
| `uniform` | Plain mean of the subword rows |
| `char_weighted` | Weighted by each subword's character length |
| `max_char` | The longest subword's row alone; ties resolve to the first occurrence |
| `bert_weighted` | `softmax(cos(subword, full token) / temperature)` over mean-pooled encoder hidden states |
| `gemma_weighted` | The same weighting computed from a decoder's input embedding table instead of a forward pass |

`uniform` is the default.
The guidebook labels this configuration *meanconst* and reports it as the strongest initialization in its final-horizon Hindi and Malayalam measurements, with the reservation that initializer differences are small and horizon-dependent.
The encoder-weighted variants require an auxiliary model that covers the target language; set `language` and let the profile choose the encoder rather than setting `subword.bert_model` directly.

### `focus`: Borrow From Similar Tokens

FOCUS (Dobler and de Melo, EMNLP 2023) initializes each new token as a Sparsemax-weighted combination of the base tokens closest to it in a fastText space.
fastText uses character n-grams, so it can embed strings that never appeared in its own vocabulary.
Cosine similarities are divided by `focus.sparsemax_temperature` before Sparsemax, which zeroes most weights so that only genuinely similar tokens contribute.

fastText tokenizes on whitespace, so it suits languages with space-separated words.
For a syllable-spaced language such as Vietnamese, the `cc.vi.300` vectors are largely syllable vectors, and `LANGUAGES.md` notes that tokens spanning several syllables fall back to subword composition.
This engine requires `fasttext-wheel` and a `focus.fasttext_model` file.

## Norm Correction

`subword.input_norm_correction` rescales new input rows so that their L2 norm matches the median norm of the existing rows.
Output rows are not rescaled by default: `output_norm_correction` is `false` in `default.yaml` because inflating the language-model-head norm makes the model over-confidently predict the new tokens and the training loss diverges.

## Add Versus Replace Surgery

The initialization arithmetic is arm-independent.
What the arm changes is the surgery around it:

- `arm: add` resizes, keeps the base rows, and initializes the appended rows `[old_n, new_n)`.
- `arm: replace` resizes, permutes the survivor rows through `id_remap.json`, then initializes the new rows `[pruned_size, final_n)`.

The Replace arm runs through `replace_init.py`, which does not implement every method.
The step fails rather than substituting a near-equivalent, because a silent substitution would report one initialization while running another.

| Method | `add` | `replace` |
|--------|:-----:|:---------:|
| `baseline.hf_default` | Supported | Supported |
| `baseline.mean_all` | Supported | Supported |
| `baseline.mean_target` | Supported | Not supported |
| `subword.uniform`, `char_weighted`, `max_char` | Supported | Supported |
| `subword.bert_weighted` | Supported | Supported |
| `subword.gemma_weighted` | Supported | Not supported |
| `focus` | Supported | Supported |

For a Replace run that needs target-script-mean behavior, the closest supported option is `subword` with `input_averaging: uniform`.

## Hardware

The base model is loaded host-resident for embedding surgery and performs no forward pass.
A GPU is used only by the auxiliary encoders for `bert_weighted` and `gemma_weighted`.
The generated environment profiles request one GPU, which suffices for every method except `gemma_weighted` on a large Gemma model, which shards with `device_map='auto'` and needs a multi-GPU profile.

## Related Pages

- Procedure: {doc}`../how-to/initialize-embeddings`
- Fields: {doc}`../reference/init-embeddings-config`
- Source contract: [`src/nemotron/steps/tokenizer_extension/init_embeddings/README.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/tokenizer_extension/init_embeddings/README.md)
