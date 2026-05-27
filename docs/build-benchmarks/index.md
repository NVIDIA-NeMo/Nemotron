<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(build-benchmarks-index)=
# About Building Multiple-Choice Question Benchmarks

<!-- Explanation and navigation hub for the bring your own benchmark (BYOB) MCQ series. -->

This section describes how to build a custom multiple-choice question (MCQ) benchmark as Apache Parquet files with the `nemotron steps run byob/mcq` command.
You supply domain text files under `input_dir`, and the pipeline samples few-shot exemplars from a Hugging Face benchmark named in your configuration, such as `cais/mmlu`.
The configuration specifies subject filters such as `high_school_mathematics`.

The benchmark step prepares seed rows, generates and judges questions, runs optional deduplication and distractor stages, and writes `benchmark.parquet`.
An optional translation stage reads an existing benchmark and writes another `benchmark.parquet` with the same column layout.

:::{tip}
New to this flow? Follow {doc}`getting-started` once, then use the grids and tables below to jump to how-to guides, concepts, or reference pages.
:::

## When to Use

The `nemotron steps run byob/mcq` command enables the following outcomes.

- Questions grounded in your own documents, paired with few-shot items from a public benchmark subject you declare in configuration.
- A repeatable Parquet artifact, one experiment folder under your configured `output_dir`, plus intermediate caches when you iterate.
- Optional translation with forward passes, backtranslation, and metric thresholds before you export another Parquet benchmark.

## Pipeline Summary

At a high level, the benchmark step performs the following work.

1. Prepare: sample few-shot examples and align them with chunks from your corpus into a seed dataset.
2. Generate: run the staged MCQ pipeline from generation through filtering into `benchmark_raw.parquet` and `benchmark.parquet`.
3. Translate, optional: translate questions and options, score backtranslation quality, and export a new `benchmark.parquet`.

## Documentation Series

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Tutorial
:link: getting-started
:link-type: doc
Install the `byob` extra, run the sample `tiny` configuration with local paths, and inspect Parquet outputs.
The `tiny` fixture pairs `cais/mmlu` high school mathematics few-shots with a one-line input file related to algebra.
+++
{bdg-secondary}`hands-on`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` How-To Guides
:link: how-to/index
:link-type: doc
Prepare data, tune models in YAML, customize prompts, and resume with `skip_until`.
+++
{bdg-secondary}`task-based`
:::

:::{grid-item-card} {octicon}`light-bulb;1.5em;sd-mr-1` Concepts
:link: explanation/index
:link-type: doc
How prepare, generate, and translate stages fit together and what each configuration block does.
+++
{bdg-secondary}`concept-focused`
:::

:::{grid-item-card} {octicon}`list-unordered;1.5em;sd-mr-1` Reference
:link: reference/index
:link-type: doc
Supported Hugging Face datasets, Parquet outputs, and YAML fields.
+++
{bdg-secondary}`specification`
:::

::::

## All Documentation

````{tab-set}

```{tab-item} Tutorial

| Guide | What you will do |
| --- | --- |
| {doc}`./getting-started` | Run `nemotron steps run byob/mcq` with `tiny` and inspect outputs |

```

```{tab-item} How-To Guides

| Guide | What you will do |
| --- | --- |
| {doc}`how-to/prepare-data` | Lay out `input_dir` and `target_source_mapping` |
| {doc}`how-to/domain-data` | Lay out per-target `.txt` corpora under `input_dir` |
| {doc}`how-to/custom-model-endpoints` | Point generation, judgement, and filter models at your endpoints |
| {doc}`how-to/prompt-tuning` | Override prompts with a YAML file |
| {doc}`how-to/skip-stages` | Resume after intermediate Parquet caches |

```

```{tab-item} Concepts

| Guide | What you will learn |
| --- | --- |
| {doc}`explanation/pipeline-overview` | Stage order for prepare, generate, and translate |
| {doc}`explanation/data-preparation` | Seeds, chunking, and the prepare step |
| {doc}`explanation/get-right-questions` | `source_subjects` and `target_source_mapping` |
| {doc}`explanation/question-generation` | Data Designer batched generation |
| {doc}`explanation/quality-validation` | Judgement, deduplication, distractors, coverage, outliers |
| {doc}`explanation/filtering` | Easiness and hallucination filters |
| {doc}`explanation/translation` | Curator translation and backtranslation metrics |

```

```{tab-item} Reference

| Guide | What you will find |
| --- | --- |
| {doc}`reference/output-files` | Paths under `output_dir` / `expt_name` |
| {doc}`reference/troubleshooting` | Symptom-to-fix index for BYOB runs |
| {doc}`reference/benchmarks` | Allowed `hf_dataset` values and default subsets |
| {doc}`reference/generate-config` | Generation YAML keys |
| {doc}`reference/translation-config` | Translation YAML keys |

```

````

## What You Need

- A Nemotron clone with dependencies installed, including the `byob` extra from `uv sync --extra byob`.
- Model credentials and endpoints that match the `generation_model_config`, `judge_model_config`, and related blocks in your YAML, as described in {doc}`how-to/custom-model-endpoints`.
- Network access to download the configured Hugging Face benchmark split unless it is already cached on disk.

## Quick Start

1. Follow {doc}`getting-started` if you have not run the step yet.
2. Read {doc}`how-to/prepare-data` when you are ready to point the pipeline at your own corpus and mapping.
3. Open {doc}`reference/generate-config` or {doc}`reference/translation-config` when you need field-level YAML detail.

## Limitations and Considerations

- Cost: generation, judgement, expansion, validity checks, and filters call remote models whenever you configure them to do so.
- Time: full runs depend on corpus size, model latency, and which optional stages stay enabled.
- Rate limits: hosted APIs may throttle parallel requests that you set under `inference_parameters`.
- Curator mount: checked-in configurations mount NeMo Curator from Git for translation and deduplication-related paths, so remote profiles must expose that tree the same way your environment expects.
