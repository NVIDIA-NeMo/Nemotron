---
id: multilingual-tokenizer-check
title: "Check tokenizer coverage for multilingual training"
tags: [tokenizer, multilingual, validation]
triggers:
  - "Training data includes non-English text or mixed-language prompts."
  - "You are adapting a mostly English base model to another language."
  - "The target language uses scripts, spacing rules, or morphology unlike English."
steps: [prep/sft_packing, sft/megatron_bridge, sft/automodel]
confidence: high
---

## When to apply

Use this before multilingual or non-English training, especially for Thai, Arabic, Hindi, Japanese, or code-switched data. Tokenizer quality can become the hidden bottleneck even when the model architecture and dataset are otherwise reasonable.

Apply it whenever the base model was primarily trained on English and you are asking it to learn a different script, segmentation pattern, or domain-specific vocabulary.

This also matters for mixed-language chat data. A model may appear multilingual on paper yet still tokenize your target corpus inefficiently or inconsistently.

## What to do

Sample real training rows and inspect tokenization directly. Look for excessive token counts, broken segmentation, and unstable handling of punctuation, whitespace, or script boundaries.

Compare average characters-per-token or tokens-per-example between English and the target language. If the target language explodes in token count, your effective context length and throughput assumptions may be wrong.

Verify that the chat template and role formatting do not introduce accidental corruption for the target language. Special tokens, separators, and newline conventions can interact badly with scripts that use different spacing norms.

For packing pipelines, re-check length distributions after tokenization, not just at the raw text level. The tokenized distribution is what drives truncation risk and packing efficiency.

Audit important domain terms, product names, abbreviations, and transliterated entities. If they fragment badly, consider whether prompt formatting or terminology normalization can help.

Run a tiny preflight generation test before full training. Ask the base model to produce short responses in the target language and inspect whether detokenized text looks stable and legible.

If coverage is poor, adjust expectations early: shorter usable context windows, different sequence lengths, more careful curation, or a base model with better multilingual prior may matter more than hyperparameter tuning.

## Exceptions

If the target language is already known to be well covered by the base model and you have prior tokenizer audits for the same data shape, you may only need a quick spot check instead of a full analysis.

Tokenizer issues are not always fatal. A model can still improve on a target language with imperfect segmentation, but you should expect worse efficiency and possibly weaker fluency.

Do not blame the tokenizer for every multilingual failure. Data quality, translation noise, and evaluation mismatch can dominate too.

## References

- Cross-check before `prep/sft_packing` so sequence-length and pack-size choices reflect real tokenized lengths.
- Revisit during `sft/megatron_bridge` or `sft/automodel` planning if throughput or truncation looks worse than expected.
- For sovereign or regional use cases, tokenizer validation is often as important as model-size selection.
