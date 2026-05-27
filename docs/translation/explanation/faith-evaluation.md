---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "How FAITH integrates into nemotron steps run translate/nemo_curator and interacts with translation backends."
topics: ["Translation", "FAITH"]
tags: ["Explanation", "Quality"]
content:
  type: "Explanation"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

# FAITH Evaluation Inside Translation

This page explains how optional FAITH scoring behaves when `faith_eval.enabled` is `true` inside `nemotron steps run translate/nemo_curator`.
FAITH stands for the five quality dimensions the judge scores against each translated segment: *Fluency*, *Accuracy*, *Idiomaticity*, *Terminology*, and *Handling of Format*.

FAITH runs in the same `TranslationStage` invocation as translation. There is no separate CLI only for FAITH scoring.

## What FAITH Adds

FAITH scores translation quality segment-by-segment using a large language model (LLM) judge configured alongside your translation backend:

- `faith_eval.threshold` defines the minimum acceptable average score. The starter default is `2.5`, which you should tune per model. See the next section for what the scale means.
- FAITH scoring follows the translated segment pairs produced by Curator's translation stage for long inputs.
- `faith_eval.filter_enabled` drops failing rows when `true`, which lets you keep high-confidence shards only.

## Score Scale

The FAITH judge scores each of the five dimensions on a one-to-five scale, where one is poor and five is excellent.
The full per-dimension rubric lives in the upstream NeMo Curator prompt at `nemo_curator/stages/text/experimental/translation/prompts/faith_eval.yaml`.
The Fluency band quoted below is representative; the other four dimensions use the same one-to-five shape with dimension-specific wording.

```text
1. **Fluency (1-5)**: Does the translation read naturally in the target language, free from grammar or syntax errors?
   - 1: Very poor fluency, difficult to understand.
   - 2: Somewhat fluent but with major grammatical issues.
   - 3: Generally fluent with a few errors.
   - 4: Mostly fluent but may have minor grammatical issues.
   - 5: Perfect grammar, native-like fluency.
```

The judge also emits two sentinel values that sit outside the one-to-five scale, quoted here from the same prompt:

```text
In case there is no translation provided, give -1 to all the categories!
If case of non-applicable score, make the score=0
```

A score of `0` means the dimension does not apply to the row, for example Terminology on a translation that contains no specialized terms.
A score of `-1` means the judge received no translation to evaluate.

The `faith_avg` column is the mean of the dimensions that scored above zero.
Dimensions marked `0` for "not applicable" are excluded from the average, so a translation with no specialized terminology can still earn a perfect `faith_avg` of `5.0`.
If every dimension is `0` or `-1`, `faith_avg` is `0.0`.

Filtering keeps a row when `faith_avg >= faith_eval.threshold`, with parse failures preserved so reviewers can audit them.
The starter default `2.5` sits between band two, "major grammatical issues," and band three, "generally fluent with a few errors."
Treat `2.5` as a noisy-data floor rather than a quality bar.
Raise the threshold when you want a tighter quality gate, for example to `3.5` when you are building a high-confidence parallel corpus.

## Why an LLM Client Is Always Required

Whenever `faith_eval.enabled` is `true`, the stage instantiates an OpenAI-compatible client using `server.url`, `server.api_key` or `server.api_key_env`, and `faith_eval.model_name`. If you omit `faith_eval.model_name`, the stage falls back to `server.model`.

Even if `backend` is `nmt`, `google`, or `aws`, FAITH still issues LLM calls. Plan keys and quotas accordingly.

## Merge Semantics

`merge_scores: true` is the default. It attaches FAITH outputs alongside translated columns so reviewers can audit scores without losing structured chat payloads.

## Related YAML Surface

See `faith_eval` in {doc}`../reference/translate-config` and operational recipes in {doc}`../how-to/run-faith-evaluation`.

## Related Pages

- Where FAITH sits in the pipeline: {doc}`pipeline-overview`
- Input and output shapes when scores merge or filter: {doc}`../reference/io-format`
