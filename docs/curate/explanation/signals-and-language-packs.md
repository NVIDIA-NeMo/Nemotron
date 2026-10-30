---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "How curation quality signals are organized, why most of them depend on the language, and how language packs supply what the runtime does not know."
topics: ["Curation", "Explanation", "Language Packs"]
tags: ["Explanation", "Curation", "Multilingual"]
content:
  type: "Explanation"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Signals and Language Packs

A *signal* is a per-document measurement with a declared direction: a document is retained when the value is below a maximum, above a minimum, or inside an interval.
The `curate/profile` step measures signals; an approved policy names them with thresholds; the `curate/nemo_curator` step applies them.
This page explains how the signal set is organized and why language packs exist.

## A Closed Registry

The 24 signals live in a closed registry (`runtime/registry.py`).
A configuration or policy names a signal by its registered name and never by an import path.
The registry records each signal's direction, units, and *requirements*: the capabilities a language pack must declare before the signal can be computed.
A policy threshold whose direction disagrees with the registry is refused.

Two shipped documents describe how to extend the set:

- [`ADDING_A_SIGNAL.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/curate/nemo_curator/ADDING_A_SIGNAL.md): subclass the local base, implement scoring and retention, register the signal.
- [`data/langpacks/ADDING_A_LANGUAGE.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/curate/nemo_curator/data/langpacks/ADDING_A_LANGUAGE.md): add a pack directory for a language.

## Three Groups of Language Dependence

### Language-agnostic

Six signals are defined over Unicode categories or over characters that mean the same thing in every script: `bullet_ratio`, `ellipsis`, `parentheses_ratio`, `unicode_alpha_numeric`, `urls_ratio`, and `white_space`.
They can be computed on any corpus with nothing supplied.

Transferring is not the same as being comparable.
`white_space` is well defined everywhere, but its ordinary range differs between a language that separates words with spaces and one that does not.
Agnostic means the number is meaningful on any corpus, not that a threshold chosen on one corpus belongs on another.
Choosing the threshold is what profiling is for.

### Pack-backed

Eight signals need data that only a language pack can provide: a character set for `script_ratio`, `latin_ratio`, and `foreign_script_ratio`; a character set and fold map for `diacritic_ratio`; a stopword list for `stopword_ratio` and, with a fold map, `stopword_ratio_folded`; boilerplate patterns for `boilerplate_hits`; sentence terminators for `sentence_end_ratio`.
When the pack does not declare the corresponding capability, the profile skips the signal with a named warning rather than computing it on a false premise.

### Writing-system assertions

Ten signals wrap NeMo Curator heuristic filters that assume a property of the writing system: that words are delimited by whitespace (`word_count`, `mean_word_length`, `max_word_length`, `symbol_to_word`, `words_with_alphabets`, `repeating_duplicate_ngrams`), that digits are ASCII (`numbers_ratio`), that punctuation is ASCII (`punctuation`), or that the alphabet is ASCII (`non_alpha_numeric`).
`token_count` requires a tokenizer instead, because a token budget counted by a different tokenizer is a different budget.

The loader cannot verify these assertions from files, so a pack declares them explicitly (`word_segmentation`, `ascii_digits`, `ascii_punctuation`, `ascii_alphabet`).
A pack that withholds one has those signals skipped during profiling, and a configuration or policy that names one is refused.
This is why `quality_filters.min_words` and `max_words` ship without defaults: a whitespace word count applied to a script without word boundaries removes most of the corpus for reasons unrelated to quality.

## Why Packs Are Data

Nothing in the curation runtime knows that any particular language exists.
Character sets, word lists, boilerplate patterns, fold maps, and sentence terminators arrive from a *language pack*: a directory named by BCP-47 tag that contains `pack.toml` and the text resources it declares.
Adding a language means adding a directory, not editing code, and reviewing a language means reviewing a small set of text files with recorded provenance.

Every source entry in `pack.toml` must record an `origin` and a `license`; the loader rejects an asset whose provenance is absent even when the file is present.
Declaring a capability without the data behind it is rejected at load, because the alternative is a report full of zeroes that reads as a finding about the corpus.

## No Implicit Pack

Nemotron ships three example packs, `en`, `vi`, and `hi`, chosen because they break different assumptions: unmarked Latin script, Latin script whose decomposed form produces only nonspacing marks, and an abugida whose vowel signs span two Unicode categories.
None of them is a default.
`language` has no default value, `langpack_dir` has no default value, and the packs under `tests/` are private-use fixtures.
For a production corpus, a pack is a reviewed, versioned input owned by the workflow that uses it, and `langpack_dir` points at that external directory.

Two checks keep a pack and a policy aligned:

- The pack directory name and the `language_tag` inside `pack.toml` must match (`langpack_tag_mismatch`), so one language's word lists cannot be filed under another language's tag.
- The policy records the pack's `content_hash`. A run that loads a pack with a different hash is refused (`policy_langpack_mismatch`), because a stopword ratio measured against one list is a different quantity against another.

## Packs Do Not Carry Thresholds

A pack states what can be measured for a language, never what a good value is.
Thresholds depend on the corpus, and the same language drawn from a different source yields different distributions.
Finding the value is the work of {doc}`two-run-curation`.

## Related Pages

- {doc}`../reference/profile`
- {doc}`../reference/policy-file`
- [`data/langpacks/SPEC.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/curate/nemo_curator/data/langpacks/SPEC.md)
