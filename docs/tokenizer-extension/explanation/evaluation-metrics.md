---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Fertility and bits per byte: what the tokenizer_extension evaluate and eval_init steps measure and why per-token perplexity cannot compare tokenizers."
topics: ["Tokenizer Extension", "Evaluation"]
tags: ["Explanation", "Tokenizer Extension"]
content:
  type: "Explanation"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Evaluation Metrics

The category provides two evaluation steps with different scopes.
`evaluate` measures a tokenizer.
`eval_init` measures a model that carries an extended tokenizer.
Neither step evaluates downstream task quality after continued pretraining; that belongs to the `eval` step catalog.

## Fertility

*Fertility* is corpus-level tokens per word: `sum(tokens) / sum(words)` over an evaluation corpus.
Lower is better.
The `evaluate` step streams the corpus so that it remains memory-safe on corpora of ten million or more rows, and reports:

| Field | Meaning |
|-------|---------|
| `fertility` | Tokens per word |
| `chars_per_token` | Characters per token |
| `unique_tokens_used` | Distinct token identifiers that appeared |
| `vocab_coverage` | `unique_tokens_used` divided by vocabulary size |

Fertility numbers are comparable only when every tokenizer is scored on the same `corpus` block.
Keep the evaluation corpus disjoint from the corpus that trained the extension; `corpus.skip_docs` skips a leading slice when the two share a source.

Fertility is also comparable between arms only at a matched budget.
`extend` records `tokens_spliced` in `summary.json` and logs a warning when the value differs from `extension_size`.

## Bits per Byte

Per-token perplexity drops mechanically when a tokenizer covers more bytes per token, so a larger vocabulary appears better without being better.
*Bits per byte* (BPB) divides the cross-entropy by UTF-8 bytes instead of by tokens, which makes it comparable across vocabularies.
Lower is better, and comparisons are meaningful within one language.

The `eval_init` step scores one or more checkpoints with a sliding window (`max_length` and `stride`) and, when `base_model` is set, scores the unextended base first as a reference and reports the delta.

### Every Model Must Score the Same Bytes

The budget that limits the run must be tokenizer-independent:

- `max_docs: N`, or `-1` for the whole corpus, caps the document stream. Document counts do not depend on the tokenizer.
- `max_tokens: N` stops each tokenizer after a different amount of text, so the resulting BPB values describe different corpora.

When more than one model would be scored under `max_tokens` with no `max_docs`, `bpb.py` refuses to run.
Because `base_model` is scored alongside `models`, one extended checkpoint plus the base already constitutes a two-tokenizer comparison.
The `allow_token_cap_comparison` configuration key forwards `--allow-token-cap-comparison` and exists only to reproduce a historical run.

## Recommended Evaluation Bundle

The guidebook recommends measuring, in order:

1. Fertility or characters per token on held-out target text.
2. Target-language BPB and English BPB on identical fixed-byte corpora.
3. Target downstream evaluation.
4. English or general retention.
5. Matched serving throughput and memory on the intended deployment shape.

Items 3 through 5 are outside this step category.
The guidebook also reports that vocabulary rows impose a serving cost and that the net throughput result is not hardware-independent; re-measure on the production tensor-parallel shape.

## Related Pages

- Procedure: {doc}`../how-to/evaluate-a-tokenizer`
- Fields: {doc}`../reference/evaluate-config` and {doc}`../reference/eval-init-config`
- Source contracts: [`src/nemotron/steps/tokenizer_extension/evaluate/README.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/tokenizer_extension/evaluate/README.md) and [`src/nemotron/steps/tokenizer_extension/eval_init/README.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/tokenizer_extension/eval_init/README.md)
