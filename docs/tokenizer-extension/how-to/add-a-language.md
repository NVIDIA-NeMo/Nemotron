---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Register a new target language for the tokenizer_extension steps by adding a LanguageProfile and, if needed, Unicode script ranges."
topics: ["Tokenizer Extension", "Multilingual"]
tags: ["How-To", "Tokenizer Extension"]
content:
  type: "How-To"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Developer"]
---

# Add a Language

The `language` key in the `extend` and `init_embeddings` configurations resolves to a *language profile*: one registry entry that supplies the corpus normalizer, the script whose residual tokens Replace prunes, the script used to locate the base model's existing target rows, the auxiliary encoder for `bert_weighted`, and the fastText code for `focus`.
Registering a language is a small source change in `src/nemotron/steps/tokenizer_extension/`; no step needs editing.

## Before You Begin

Check whether the language is already registered.
The current registry contains `hindi`, `marathi`, `nepali`, `sanskrit`, `bengali`, `punjabi`, `gujarati`, `odia`, `tamil`, `telugu`, `kannada`, `malayalam`, `urdu`, and `vietnamese`.

## Procedure

1. Add Unicode ranges in `script_ranges.py` if the script is not already present in `SCRIPT_UNICODE_RANGES`.

   A token belongs to the language when any character of its decoded surface falls in one of the ranges.
   For a script with its own Unicode block, that block is the range:

   ```python
   "thai": [(0x0E00, 0x0E7F)],
   ```

   For a Latin-script language, enumerate only the codepoints that the language uses, as the `vietnamese` entry does with precomposed vowel-tone letters, horned o and u, and the combining horn.
   Never include plain ASCII Latin: it is shared with every other Latin-script language, and pruning it under `method=replace` would remove the byte alphabet that new merges are built from.

1. Add a `LanguageProfile` in `languages.py`.

   ```python
   "thai": LanguageProfile("none", "thai", "th", _XLMR),
   ```

   The four positional fields are:

   | Field | Meaning |
   |-------|---------|
   | `normalizer` | Key into `NORMALIZERS`; `"none"` applies NFKC only |
   | `script` | Key into `SCRIPT_UNICODE_RANGES` |
   | `fasttext` | Language code of the `cc.<code>.300` fastText vectors used by `focus` |
   | `encoder` | Hugging Face identifier of an encoder that covers the language, used by `bert_weighted` |

   Choose the encoder deliberately.
   MuRIL (`google/muril-base-cased`) covers 17 Indian languages and nothing else, so `_MURIL` is correct for the Indic entries and `_XLMR` (`FacebookAI/xlm-roberta-base`) is the general fallback.
   An encoder that does not cover the target language produces a `bert_weighted` initialization that `LANGUAGES.md` describes as no better than noise.

1. Add a normalizer only when the script needs more than NFKC.

   NFKC is always applied.
   The `devanagari` normalizer additionally runs `DevanagariNormalizer` from `indic-nlp-library`.
   New normalizers are registered in `languages.get_normalizer()`.

1. Run `extend` with the new value.

   ```console
   $ uv run nemotron steps run tokenizer_extension/extend -c default language=thai method=add \
       corpus.hf_dataset=<org/dataset> corpus.hf_name=<config> corpus.hf_split=<split>
   ```

   The step logs the resolved `script_normalizer` and `remove_script` at start-up; confirm they match the profile.

## Backward Compatibility

- A configuration with no `language` key keeps the historical Devanagari defaults: Devanagari normalizer, Devanagari prune set, MuRIL, and `cc.hi.300`.
- The language-neutral key names have aliases for the earlier Hindi-specific names:

  | Earlier name | Current name |
  |--------------|--------------|
  | `subword.input_hindi_norm` | `subword.input_target_norm` |
  | `subword.output_hindi_norm` | `subword.output_target_norm` |
  | `baseline.mode: mean_hindi` | `baseline.mode: mean_target` |
  | `focus.candidate_pool: hindi` | `focus.candidate_pool: target` |

## Considerations

- `focus` on a syllable-spaced language: fastText tokenizes on whitespace, so for Vietnamese the `cc.vi.300` vectors are largely syllable vectors, and tokens that span several syllables fall back to subword composition.
- `method=replace` on a Latin-script language frees far fewer rows than on a language with its own block, because only tokens that carry a diacritic qualify.

## Related Pages

- Source guide: [`src/nemotron/steps/tokenizer_extension/LANGUAGES.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/tokenizer_extension/LANGUAGES.md)
- Method selection: {doc}`../explanation/extension-methods`
