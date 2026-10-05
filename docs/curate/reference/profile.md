---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Reference for the curate/profile step: parameters, artifacts, signals, strategies, and errors."
topics: ["Curation", "Reference", "Profiling"]
tags: ["Reference", "Curation", "Steps"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Data Scientist"]
---

# curate/profile

The `curate/profile` step measures quality-signal distributions on an unfiltered corpus and reports what candidate thresholds would remove from it.
It writes a profile report, a human-readable summary, and a set of *candidate policies* that carry `approved: false`.
The step never approves a policy; approval is a separate, recorded act described in {doc}`policy-file`.

## Syntax

```bash
uv run --no-sync nemotron steps run curate/profile \
    [-c <config-name-or-path>] \
    [-d] \
    [<dotlist-overrides>...]
```

Refer to the [Nemotron Steps CLI Reference](../../train-models/reference/cli-reference.md) for the shared flag set.

## Configuration Files

The step ships two configuration files under `src/nemotron/steps/curate/nemo_curator/profile/config/`.

| File | Purpose |
| --- | --- |
| `default.yaml` | Profiles `./output/filtered_jsonl/**/*.jsonl` with `language: null` and `langpack_dir: null`. Both must be set before the step runs. Samples up to 200,000 documents. |
| `en.yaml` | Profiles the packaged tiny fixture with the packaged English reference pack and four pack-backed signals. `max_total_docs: 0` profiles the whole fixture. |

## Inputs and Outputs

| Direction | Artifact type | Content |
| --- | --- | --- |
| Consumes | `raw_jsonl` | The corpus to profile, before any filtering. `prepared_jsonl` from `curate/ingest` is accepted. |
| Produces | `profile_report` | `profile_report.json`: per-signal quantiles and retention surfaces, macro and micro views, gate co-occurrence, and a producer block. |
| Produces | `filter_policy` | `candidate_policies.yaml` with `approved: false`. |

The step also writes `profile_summary.md`, the human-readable summary, and `sample_manifest.json`, which records the sample that was measured.

The report carries a `profile_digest` taken over everything except the producer block, so re-profiling the same corpus under the same configuration does not make an approved policy appear stale.

## Parameters

```{option} input_glob

JSONL files to profile.
Profile the unfiltered corpus; profiling filtered output measures the gates after they have already removed what they remove.
```

```{option} output_dir

Directory for `profile_report.json`, `profile_summary.md`, `candidate_policies.yaml`, and `sample_manifest.json`.
```

```{option} language

BCP-47 tag that selects the language pack.
There is no default.
```

```{option} langpack_dir

Root directory of the language packs.
There is no default; a pack root must be named explicitly.
The `en` configuration names the packaged reference root, `src/nemotron/steps/curate/nemo_curator/data/langpacks`.

Default: `null`.
```

```{option} text_field

Record field holding the text to score.

Default: `text`.
```

```{option} source_field

Field identifying the source of each record.
When records do not carry it, each shard path is treated as a source and the report states so.

Default: `source`.
```

```{option} id_field

Field holding a stable document identifier, used to make sampling reproducible.
Falls back to a content hash.

Default: `null`.
```

```{option} signals

Signals to profile, from the closed registry in `src/nemotron/steps/curate/nemo_curator/runtime/registry.py`.
An empty list profiles every signal whose requirements the language pack meets.
A named signal whose requirements are unmet is an error, not a skip.

Default: `[]`.
```

```{option} max_total_docs

Cap on sampled documents.
Per-source counts are derived from it proportionally.
`0` profiles the whole corpus.

Default: `200000`.
```

```{option} seed

Seed for hash-bottom-k sampling.
The same seed and cap reproduce the same sample.

Default: `0`.
```

```{option} band_search

Retention range reported as a candidate band, as `min_keep_rate` and `max_keep_rate`.
These bounds are analysis choices, not properties discovered in the data.

Default: `{min_keep_rate: 0.80, max_keep_rate: 0.995}`.
```

```{option} models

Optional models.
`models.fasttext_langid` is a path to a FastText language-identification model and enables the language-composition table; a configured path that is not a file is refused.
`models.tokenizer` is a `{name, revision}` mapping and enables the `token_count` signal; `revision` is required because counts from two revisions are not comparable.
```

## Signals

Every signal declares a bound (`min`, `max`, or `interval`) and, where applicable, the language-pack capability it requires.
Six signals are language-agnostic and require nothing from the pack.

| Signal | Bound | Units |
| --- | --- | --- |
| `bullet_ratio` | `max` | ratio |
| `ellipsis` | `max` | ratio |
| `parentheses_ratio` | `max` | ratio |
| `unicode_alpha_numeric` | `max` | ratio |
| `urls_ratio` | `max` | ratio |
| `white_space` | `max` | ratio |

The remaining signals declare a pack capability and are skipped, with a named warning, when the pack does not declare it.

| Signal | Bound | Units | Requires |
| --- | --- | --- | --- |
| `boilerplate_hits` | `max` | patterns | `boilerplate_hits` |
| `diacritic_ratio` | `min` | ratio | `diacritic_ratio` |
| `foreign_script_ratio` | `max` | ratio | `script_ratio` |
| `latin_ratio` | `max` | ratio | `script_ratio` |
| `script_ratio` | `min` | ratio | `script_ratio` |
| `sentence_end_ratio` | `min` | ratio | `sentence_end_ratio` |
| `stopword_ratio` | `min` | ratio | `stopword_ratio` |
| `stopword_ratio_folded` | `min` | ratio | `stopword_ratio_folded` |
| `max_word_length` | `max` | characters | `word_segmentation` |
| `mean_word_length` | `interval` | characters | `word_segmentation` |
| `non_alpha_numeric` | `max` | ratio | `ascii_alphabet` |
| `numbers_ratio` | `max` | ratio | `ascii_digits` |
| `punctuation` | `max` | ratio | `ascii_punctuation` |
| `repeating_duplicate_ngrams` | `max` | ratio | `word_segmentation` |
| `symbol_to_word` | `max` | ratio | `word_segmentation` |
| `token_count` | `interval` | tokens | `tokenizer` |
| `word_count` | `interval` | words | `word_segmentation` |
| `words_with_alphabets` | `min` | ratio | `word_segmentation` |

Refer to {doc}`../explanation/signals-and-language-packs` for what the capability requirements mean.

## Strategies

| When | Then |
| --- | --- |
| You are curating a language for which NeMo Curator publishes no thresholds | Run with the upstream filters disabled and read the retention curves. Promote a candidate only after validating it. |
| `curate/nemo_curator` produced far less output than expected | Profile the unfiltered input. The retention curves and the co-occurrence table identify the responsible gate. |
| You want to compare two filtering policies fairly | Profile first, then hold the token budget fixed with `curate/subset`. |

## Common Errors

```{option} language_pack_not_found

Set `langpack_dir` explicitly and confirm that it contains a directory for the requested BCP-47 tag.
```

```{option} language_pack_invalid

The pack exists but breaks its contract, usually a capability declared without the word list or character set behind it.
The message names what is missing.
```

```{option} unknown_signal

The configuration named a signal that is not in the registry.
```

```{option} signal_requirements_unmet

A named signal needs something this run does not have, such as a tokenizer for `token_count`.
Supply it or remove the signal from the list.
```

```{option} direction_mismatch

A registered signal's declared direction disagrees with the underlying filter's `keep_document` behavior.
The run stops because the retention figures would be wrong.
Report the defect rather than overriding it.
```

```{option} empty_corpus

`input_glob` matched files but no parsable records.
Check `text_field` and confirm that the files are JSONL.
```

```{option} config_error

The configuration requests a figure that cannot be reported, usually `models.tokenizer` without a `revision`.
The message names the field.
```

## Examples

Profile the packaged fixture with the English reference pack:

```bash
uv run --no-sync nemotron steps run curate/profile -c en
```

Profile an ingested Vietnamese corpus with every supported signal:

```bash
uv run --no-sync nemotron steps run curate/profile -c default \
    input_glob=./output/vi/ingested/*.jsonl \
    output_dir=./output/vi/profile \
    language=vi \
    langpack_dir=./src/nemotron/steps/curate/nemo_curator/data/langpacks \
    id_field=id
```

## Related Pages

- [`src/nemotron/steps/curate/nemo_curator/profile/README.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/curate/nemo_curator/profile/README.md)
- {doc}`policy-file`
- {doc}`../how-to/run-the-measure-apply-flow`
