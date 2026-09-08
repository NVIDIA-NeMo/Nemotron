---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "How the Nemotron step categories combine into a language-adaptation workflow: tokenizer extension, curation, continued pretraining, synthetic SFT data, fine-tuning, benchmarks, and evaluation."
topics: ["Sovereign AI", "Continued Pretraining", "Multilingual"]
tags: ["Sovereign AI", "Documentation"]
content:
  type: "Explanation"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

(sovereign-ai-index)=
# Adapting Nemotron to a New Language

This page maps the Nemotron step categories onto a single workflow: adapting an English-centric Nemotron base model to a target language, region, or domain, from tokenizer through evaluation.
Each stage is documented in its own section; this page explains how the stages connect and which artifact each one hands to the next.

## Workflow at a Glance

```{mermaid}
flowchart LR
    tok["tokenizer_extension/*<br/>resized HF checkpoint"] --> cpt
    cur["curate/nemo_curator<br/>filtered_jsonl"] --> prep["data_prep/pretrain_prep<br/>binidx"]
    prep --> cpt["pretrain/megatron_bridge<br/>checkpoint_megatron"]
    cpt --> sft["sft/megatron_bridge or sft/automodel<br/>fine-tuned checkpoint"]
    sdg["sdg/persona_mcq<br/>training_jsonl"] --> sft
    sft --> ev["eval/model_eval<br/>eval_results"]
    byob["byob/mcq, translate/nemo_curator<br/>target-language benchmarks"] --> ev
```

Steps communicate through typed artifacts declared in each `step.toml`; the label under each node is the artifact the step produces.
A stage can be skipped when its input already exists: for example, start at `pretrain/megatron_bridge` when the base tokenizer already covers the target script, or start at `sft/*` when a continued-pretraining checkpoint is already available.

## Stages

| Stage | Step category | Consumes | Produces | Documentation |
|---|---|---|---|---|
| Extend the tokenizer | `tokenizer_extension/extend`, `init_embeddings`, `evaluate`, `eval_init` | Base tokenizer and Hugging Face checkpoint, target-language corpus | Extended tokenizer, resized Hugging Face checkpoint, fertility and bits-per-byte reports | {doc}`tokenizer-extension/index` |
| Curate the corpus | `curate/nemo_curator` | JSONL text or a Hugging Face dataset snapshot | `filtered_jsonl` | {doc}`curate/index` |
| Prepare pretraining data | `data_prep/pretrain_prep` | `filtered_jsonl` | `binidx` | {doc}`nemotron/data-prep` |
| Continue pretraining | `pretrain/megatron_bridge` | `binidx`, resized checkpoint as `hf_model_path` | `checkpoint_megatron` | {doc}`train-models/index` |
| Generate SFT data | `sdg/persona_mcq`, `sdg/data_designer` | Persona locales and teacher-model endpoints | Aligned `train.jsonl` and `blend.json` per teacher | {doc}`sdg/how-to/persona-mcq-data`, {doc}`sdg/index` |
| Fine-tune | `sft/megatron_bridge`, `sft/automodel` | `training_jsonl` or a packed blend, the continued-pretraining checkpoint | Fine-tuned checkpoint | {doc}`train-models/how-to/choose-sft-backend`, {doc}`train-models/how-to/run-sft-automodel` |
| Build benchmarks | `byob/mcq`, `translate/nemo_curator` | Source corpus or an existing benchmark | Target-language MCQ benchmarks | {doc}`build-benchmarks/index`, {doc}`translation/index` |
| Evaluate | `eval/model_eval` | Checkpoint or hosted endpoint, benchmark tasks | `eval_results` | {doc}`model-eval/index` |

## How the Stages Connect

**Tokenizer extension precedes continued pretraining.**
`tokenizer_extension/extend` adds target-script subwords to the base tokenizer and `init_embeddings` writes a resized Hugging Face checkpoint whose new embedding rows are initialized from existing subwords.
`pretrain/megatron_bridge` loads that checkpoint through `hf_model_path` and trains the new rows on the curated corpus.
`tokenizer_extension/evaluate` reports fertility (tokens per word) before any GPU is used, and `eval_init` reports bits per byte on the resized checkpoint, which is the metric that remains comparable across vocabularies.

**Curation and data preparation feed continued pretraining.**
`curate/nemo_curator` applies language identification, word-count, and domain filters to JSONL shards and writes `filtered_jsonl`.
`data_prep/pretrain_prep` tokenizes that output into Megatron `binidx` files with the extended tokenizer, so the pretraining corpus is tokenized with the same vocabulary the model will train on.

**Synthetic data supplies the target-language SFT set.**
`sdg/persona_mcq` authors multiple-choice questions grounded in regional personas, deduplicates them lexically and semantically, and keeps only questions on which a panel of three or more teacher models agrees.
Agreement is a consistency filter, not factual verification; validate a sample before training.
The step writes one aligned `train.jsonl` and `blend.json` per teacher, which `sft/automodel` reads directly and `data_prep/sft_packing` packs for `sft/megatron_bridge`.

**Benchmarks are built independently of training.**
`byob/mcq` generates multiple-choice benchmarks from a source corpus and can translate existing benchmarks; `translate/nemo_curator` translates JSONL or Parquet corpora with NeMo Curator backends.
Keep benchmark sources disjoint from the SFT data so that `eval/model_eval` measures generalization rather than memorization.

## Choosing an Entry Point

| Situation | Start at |
|---|---|
| The base tokenizer produces high fertility on the target language | {doc}`tokenizer-extension/getting-started` |
| The tokenizer is adequate but the model lacks target-language knowledge | {doc}`curate/index`, then {doc}`train-models/index` |
| A continued-pretraining checkpoint exists and needs instruction data | {doc}`sdg/how-to/persona-mcq-data` |
| A fine-tuned model needs target-language evaluation | {doc}`build-benchmarks/index`, then {doc}`model-eval/index` |

## Limitations

- The workflow is assembled from independently documented steps; there is no single command that runs all stages.
- `sdg/persona_mcq` depends on managed Data Designer persona locales; languages without a managed locale reuse a related locale, as the shipped Malayalam configuration reuses `en_IN`.
- Cluster environment profiles (`env.toml`) are per step; refer to each section's cluster guide for the profile names.

## Related Documentation

- {doc}`steps/index` — the Nemotron Steps model and CLI
- {doc}`nemotron/data-prep` — pretraining, SFT, and RL data preparation
- {doc}`train-models/explanation/artifact-graph` — the typed artifact graph that connects steps
