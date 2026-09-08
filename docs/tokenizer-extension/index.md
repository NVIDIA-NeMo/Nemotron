---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Extend a Nemotron tokenizer with target-language subwords, initialize the new embedding rows, and measure fertility and bits per byte with the tokenizer_extension steps."
topics: ["Tokenizer Extension", "Continued Pretraining", "Multilingual"]
tags: ["Tokenizer Extension", "Documentation"]
content:
  type: "Explanation"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

(tokenizer-extension-index)=
# Tokenizer Extension With Nemotron

The `tokenizer_extension` step category extends a base-model tokenizer with subwords for a target language, initializes the model embedding rows for the new vocabulary, and measures the result.
The output is a resized Hugging Face checkpoint that you pass to the `pretrain/megatron_bridge` step as `hf_model_path` for continued pretraining (CPT).

:::{tip}
New here? Start with {doc}`getting-started`, then use this page as the map to deeper topics.
:::

## When to Use

Use the `tokenizer_extension` steps when you need:

- Lower token *fertility*, the number of tokens per word, for a language that the base tokenizer represents inefficiently. Fewer tokens per word reduce context consumption and serving cost.
- A resized checkpoint whose new embedding rows start from an informed initialization rather than random values, so that CPT converges from a better starting point.
- Comparable measurements across tokenizers with different vocabularies, by using bits per byte (BPB) rather than per-token perplexity.

Tokenizer extension is primarily an efficiency intervention.
The [Tokenizer Extension Guidebook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/tokenizer_extension/guidebook/README.md) reports that an independent contribution to downstream model quality has not been established.

## Pipeline Summary

```{mermaid}
flowchart LR
    A[Base checkpoint_hf] --> B[extend]
    C[Target-language corpus] --> B
    B --> D[tokenizer]
    D --> E[init_embeddings]
    A --> E
    E --> F[Resized checkpoint_hf]
    F --> G[pretrain/megatron_bridge]
    D --> H[evaluate: fertility]
    F --> I[eval_init: BPB]
```

The category contains four steps.
Each step reads what the previous one wrote.

| Step | Compute | Consumes | Produces |
|------|---------|----------|----------|
| `tokenizer_extension/extend` | CPU | Base tokenizer and a text corpus | Extended tokenizer directory and `summary.json` |
| `tokenizer_extension/init_embeddings` | GPU | Extended tokenizer and base checkpoint | Resized Hugging Face checkpoint |
| `tokenizer_extension/evaluate` | CPU | Any tokenizer | Fertility report JSON |
| `tokenizer_extension/eval_init` | GPU | One or more resized checkpoints | BPB and perplexity report JSON |

## Documentation Series

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Tutorial
:link: getting-started
:link-type: doc
Build an extended tokenizer, initialize embeddings, and measure fertility with the shipped `default.yaml` configurations.
+++
{bdg-secondary}`hands-on`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` How-to guides
:link: how-to/index
:link-type: doc
Choose an extension method, initialize embeddings, evaluate, register a new language, and run on a cluster.
+++
{bdg-secondary}`task-based`
:::

:::{grid-item-card} {octicon}`light-bulb;1.5em;sd-mr-1` Concepts
:link: explanation/index
:link-type: doc
Pipeline architecture, Add versus Replace versus Expand, embedding initialization, and evaluation metrics.
+++
{bdg-secondary}`learn`
:::

:::{grid-item-card} {octicon}`list-unordered;1.5em;sd-mr-1` Reference
:link: reference/index
:link-type: doc
YAML parameters for each step, CLI syntax, output files, and troubleshooting.
+++
{bdg-secondary}`lookup`
:::

::::

## All Documentation

````{tab-set}

```{tab-item} Tutorial

| Guide | What you do |
|-------|-------------|
| {doc}`prerequisites` | Software, hardware, data, and access requirements |
| {doc}`getting-started` | Run `extend`, `init_embeddings`, and `evaluate` for one language |

```

```{tab-item} How-to guides

| Guide | Focus |
|-------|-------|
| {doc}`how-to/extend-a-tokenizer` | `method: add`, `replace`, or `expand` |
| {doc}`how-to/initialize-embeddings` | `method: baseline`, `subword`, or `focus` and the `arm` setting |
| {doc}`how-to/evaluate-a-tokenizer` | Fertility and BPB comparisons |
| {doc}`how-to/add-a-language` | `languages.py` and `script_ranges.py` entries |
| {doc}`how-to/run-on-a-cluster` | Environment profiles and optional dependencies |

```

```{tab-item} Concepts

| Guide | Topic |
|-------|-------|
| {doc}`explanation/pipeline-overview` | Steps, artifacts, and the hand-off to CPT |
| {doc}`explanation/extension-methods` | Add, Replace, and Expand |
| {doc}`explanation/embedding-initialization` | Baseline, subword, and FOCUS engines |
| {doc}`explanation/evaluation-metrics` | Fertility and bits per byte |

```

```{tab-item} Reference

| Guide | Content |
|-------|---------|
| {doc}`reference/extend-config` | `extend/config/default.yaml` fields |
| {doc}`reference/init-embeddings-config` | `init_embeddings/config/default.yaml` fields |
| {doc}`reference/evaluate-config` | `evaluate/config/default.yaml` fields |
| {doc}`reference/eval-init-config` | `eval_init/config/default.yaml` fields |
| {doc}`reference/cli` | `nemotron steps run tokenizer_extension/<step>` syntax |
| {doc}`reference/outputs` | Files each step writes |
| {doc}`reference/troubleshooting` | Symptoms, causes, and corrections |

```

````

## Limitations and Considerations

- Scope: the `evaluate` step measures tokenizer-level fertility only. Model and downstream evaluation after CPT belongs to the `eval` step catalog; see {doc}`../model-eval/index`.
- Optional dependencies: `indic-nlp-library` and `fasttext-wheel` are declared as the `tokenizer-extension` extra in `pyproject.toml` and are not part of the base install. A run that needs one of them and cannot import it fails with the install command rather than producing a quietly different tokenizer.
- Learning-rate policy: the guidebook reports differential learning-rate results that were measured outside this repository. The `pretrain/megatron_bridge` step does not ship a differential learning-rate option.
- Remote execution: use `--run <profile>` or `--batch <profile>` with the `*_tokenizer_extend`, `*_tokenizer_init_embeddings`, `*_tokenizer_evaluate`, and `*_tokenizer_eval_init` profiles that `env/env_toml` generates.

## Quick Paths

1. First run: {doc}`getting-started`
2. Choose a method: {doc}`explanation/extension-methods`
3. Register a language: {doc}`how-to/add-a-language`
4. Look up a field: {doc}`reference/index`
