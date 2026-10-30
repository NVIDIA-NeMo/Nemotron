<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

<!-- Explanation: what operators configure for MCQ generation and what the model receives from seed.parquet. -->

# Question Generation

Understand how the generate stage turns prepared seeds into new multiple-choice question (MCQ) rows, and which YAML settings you control.

Generation reads `seed.parquet` from the prepare step, batches work through Data Designer using `generation_model_config` and your prompt templates, and writes `stage_cache/generated_questions.parquet` under `output_dir`/`expt_name`.

For YAML field names and defaults, use {doc}`../reference/generate-config`.
To override packaged prompts, use {doc}`../how-to/prompt-tuning`.

## Overview

Question generation uses few-shot prompting so a configured model can draft MCQs grounded in your domain text.
Each call combines the following inputs:

- Few-shot exemplars that show the answer format and tone you want, drawn from the Hugging Face benchmark you configured during prepare.
- Domain passages taken from your `input_dir` corpora and paired in `seed.parquet`.
- Model parameters under `generation_model_config`, including provider, model id, and `inference_parameters` such as temperature and token limits.

## Few-Shot Prompting

### How It Works

Templates package exemplars and domain text into a single prompt.
The exact layout comes from the packaged defaults when `prompt_config` is `null`, or from your override file when you set `prompt_config` to a path.

Illustrative shape:

```text
[Few-shot example 1]
Question: [example question]
Choices: A. [...], B. [...], C. [...], D. [...]
Answer: [correct label]

[Few-shot example 2]
...

[Domain passage]
[Your domain-specific text here]

Generate a question based on the content above, following the format of the examples.
```

What the model sees:

1. Example MCQs in the target shape.
2. The domain passage the prepare step attached to that row.
3. Instructions to match the exemplar format.

What the model returns:

Structured fields for a four-option item: question text, choice list, and the correct answer key, aligned with the response schema Data Designer expects for this family.

## Language

Benchmark few-shots are often English, and many teams pair them with an English-capable model, then run the separate translation stage when they need another locale.

To skip translation, you need benchmark rows and a model that both support your target language.

## Row Inputs

After prepare formats the seed for generation, each batch row carries at least:

- `text`: domain passage the question must ground on.
- `few_shot_examples`: formatted benchmark exemplars for the prompt.
- `target_subject`: which corpus folder or Parquet target produced the row.

Additional columns such as `source_subject`, `tags`, and `language` are carried through for tracing and templating.
Refer to {doc}`../reference/output-files` for the evolving Parquet layout.

## Model Configuration

You control cost, latency, and creativity through `generation_model_config`.
A minimal pattern matches the sample configuration files:

```yaml
generation_model_config:
  alias: gpt-oss-120b
  model: openai/gpt-oss-120b
  provider: nvidia
  inference_parameters:
    max_tokens: 1024
    max_parallel_requests: 1
    temperature: 0.0
    top_p: 1.0
```

### Temperature and Sampling

`temperature` controls randomness in the provider request.

- Values closer to zero usually produce steadier, more repeatable wording.
- Higher values increase variety but can also increase refusals or off-format answers, so raise them gradually while you watch `generated_questions.parquet`.

`max_parallel_requests` caps concurrent calls per batch; balance it against provider rate limits.

## After Generation

New rows move through validation and cleanup stages before they become `benchmark.parquet`, including judgement, optional semantic deduplication, distractor expansion when enabled, optional coverage scoring, semantic outlier scoring, and easiness or hallucination filtering.

Read {doc}`quality-validation` for the full stage list and artifacts, {doc}`filtering` for threshold behavior, and {doc}`../reference/output-files` for filenames under `stage_cache/`.

## Next Steps

- {doc}`quality-validation` for the checks that run after `generated_questions.parquet`.
- {doc}`filtering` for how easy or hallucinated items are removed or retained.
- {doc}`pipeline-overview` for where generate sits in the `mcq` family.
- {doc}`data-preparation` for how `seed.parquet` is built upstream.
