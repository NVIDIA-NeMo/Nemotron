<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

<!-- How-to: point `prompt_config` at a YAML override for BYOB MCQ stages, keep placeholders valid, and run the step. Use when packaged prompts are close but you need different wording or guardrails. -->

(byob-mcq-prompt-tuning)=
# Prompt Tuning for Benchmarks

The build-your-own benchmark (BYOB) pipeline for multiple-choice questions (MCQs) runs several stages that each call a large language model (LLM).
Each stage reads prompt text from your configuration.
You can replace the packaged wording to change tone, difficulty, guardrails, and output shape while keeping the same pipeline contract.

Use this guide when packaged prompts are almost right and you want different instructions, tone, or few-shot examples without changing stage order or the expected output shape.

For every field on the main experiment YAML file, see {doc}`../reference/generate-config`.
For load-time validation errors, see {doc}`../reference/troubleshooting`.

## Point the experiment file at your prompts

Leave `prompt_config` set to `null` when the packaged prompts are acceptable.

Set `prompt_config` to a filesystem path when you want a full override file for all six stages.
The path is read when the configuration loads, so the file must exist and be readable at that moment.
Relative paths resolve from the shell working directory where you launch `nemotron`, not from the directory that contains your main configuration file.
Use an absolute path when you want the same resolution no matter which directory you run from.

```yaml
prompt_config: /path/to/my_prompts.yaml
```

Run the step as usual.

```console
$ uv run nemotron steps run byob/mcq -c /path/to/config.yaml
```

If any required stage is missing, if either string under a stage is absent, or if a value is not a string, validation stops before the pipeline starts and reports the failing field.

The samples that you can copy from are in `src/nemotron/steps/byob/runtime/benchmark_families/mcq/prompts/` in the Nemotron repository, one module per stage.

## Structure of the Prompt YAML File

Create one YAML file that defines all six stages.
Each stage is a mapping with exactly two string keys, `system_prompt` and `prompt`.
Multi-line strings are fine when you use YAML block scalars.

The loader treats a custom file as all-or-nothing.
You must include every stage even when you only intend to change one of them.

### Required Stages

| Stage key | Role in the pipeline |
| --- | --- |
| `qa_generation` | Creates new questions with four choices from the seed passage and few-shot examples. |
| `question_judge` | Scores clarity, validity, and knowledge versus reasoning style. |
| `hallucination_filter` | Asks filter models to answer with the passage present to catch unsupported content. |
| `easiness_filter` | Asks filter models to answer without the passage to catch questions that are too easy. |
| `distractor_expansion` | Adds six distractors when expansion is enabled in the main configuration file. |
| `distractor_validity` | Checks that only the labeled answer is correct given the passage. |

## How Placeholders Work

Prompt strings are not plain static prose.
They carry two different kinds of markers, and some stages use both kinds in the same string.

*NeMo Data Designer* substitutes *seed column* markers.
After any earlier formatting, those markers look like `{{column_name}}` and pull values from the CSV row the stage is processing.

Some stages run an earlier pass that substitutes *run-parameter* markers written with single braces, such as `{num_questions}`.
Those values come from your main experiment configuration file or from small integers the pipeline derives before Data Designer runs.

Packaged templates sometimes show column names with four braces, for example `{{{{target_subject}}}}`.
That pattern survives the run-parameter pass and leaves a normal `{{target_subject}}` marker for Data Designer.

If a stage forwards both strings as written, the pipeline does not apply the single-brace pass.
Every `{{column_name}}` token you leave in YAML must match a real column in that stage’s seed data.

### Common Run-Parameter Tokens

- `{num_few_shot_samples}` is the few-shot count from the main configuration file.
- `{num_questions}` is the questions per query from the main configuration file.
- `{num_choices}` either rewritten for filter stages so Data Designer can inject `{{num_choices}}`, or replaced with the integer `4` or `10` for distractor validity’s system prompt.
- `{choices}` rewritten during filtering so Data Designer receives `{{choices_text}}` for the rendered choice list on each row.

### Column Markers Copied from Packaged Templates

The list below shows fields that appear in the sample files.
Quadruple-brace forms are common in `qa_generation` and in some filter user prompts.
Judging prompts use ordinary double braces because that stage is forwarded as written.

- `{{{{target_subject}}}}` — subject label for the target passage.
- `{{{{text}}}}` — source passage body.
- `{{{{few_shot_examples}}}}` — serialized few-shot examples.
- `{{{{language}}}}` — target locale string for generation.
- `{{{{question_generated_formatted}}}}` — question plus choices as produced upstream.
- `{{question}}` — generated question text for the judge stage.

## Example: QA Generation Block

Use the layout from `src/nemotron/steps/byob/runtime/benchmark_families/mcq/prompts/qa_generation.py` as a starting point when you author your own file.

```{code-block} yaml
:class: scrollable

qa_generation:
  system_prompt: |
    You are an expert in creating questions from a description of a given topic.
    You will be given {num_few_shot_samples} example questions and answers unrelated to the topic.

    You should
    1. Create {num_questions} questions, four choices and answers similar to the example question answer.
    2. Follow the question style of the example question answers (Direct WH question/Completion/Best explanation/Best action/Equivalence/Other).
    3. Try to create questions that are higher in the cognitive level scale, by mostly using concepts from the text passage.
    4. The questions should not explicitly refer to the passage or example question answer.
    5. Assume that the person reading the questions does not have access to the passage or example question answer. So make the question clear as to what the topic is.
  prompt: |
    Definition of cognitive level (higher is better):
    1: Recall
    The question asks for recall of isolated facts, definitions, or simple formulas (e.g., “What is the full form of SEBI?”).

    2: Understanding
    The question checks comprehension of concepts, classifications, or simple explanations (e.g., interpret what an LTV ratio means for a borrower).

    3: Application
    The question requires using a concept, rule, or formula in a straightforward, familiar situation (e.g., compute post‑tax return on a fixed deposit given basic data).

    4: Analysis
    The question involves breaking down information, comparing alternatives, or identifying relationships/causes (e.g., infer the impact of an RBI rate change on bond prices or bank NIMs).

    5: Evaluate
    The question requires judgment among alternatives based on criteria, or synthesizing information to choose the best course of action or construct a plan (e.g., select the most appropriate investment strategy for a given Indian retail investor scenario).


    Create {num_questions} questions, four choices and answers for the given topic:
    Topic: {{{{target_subject}}}}
    <start of topic>
    {{{{text}}}}
    <end of topic>

    Now make {num_questions} questions, four choices and answers for the topic similar to the example question answer. Don't make the answer too obvious.

    Example questions and answers:-
    {{{{few_shot_examples}}}}

    Write the questions and options in the language: {{{{language}}}}
    Return the questions in JSON format with each question having the following fields:
    - question: The question
    - choice_a: The first choice (A)
    - choice_b: The second choice (B)
    - choice_c: The third choice (C)
    - choice_d: The fourth choice (D)
    - answer: The answer (A/B/C/D)
```

## Shorter Excerpts for Other Stages

Use these fragments when you want a quick visual check that a stage matches your expectations.

### Judging

Forwarded as written, so only double-brace column names appear.

```text
Here is the question:
Question: {{question}}
```

### Filtering

The system line keeps `{num_choices}` in YAML.
The pipeline rewrites it so Data Designer later fills `{{num_choices}}` per row.

```text
You are answering a multiple-choice question with {num_choices} choices.
You will be given a passage on a topic and a question and a list of choices.
```

The hallucination user prompt carries passage markers, the formatted question, and `{choices}`, which becomes `{{choices_text}}` for Data Designer.

```text
Topic: {{{{target_subject}}}}
<start of topic>
{{{{text}}}}
<end of topic>

Answer the following question:

{{{{question_generated_formatted}}}}

The answer should be one of {choices}. Think step by step and then finish your answer with "The answer is (X)" where X is the correct letter choice.
```

The easiness filter reuses the same `{num_choices}` and `{choices}` contract on the system and user strings but omits the topic and passage block.
Copy each filter stage from its own packaged default instead of swapping strings between them.

### Distractor Validity

Only the system line runs the single-brace pass, replacing `{num_choices}` with the integer `4` or `10`.

```text
The question claims that there is only one correct answer among the {num_choices} choices.
```

The user `prompt` is forwarded as written; keep its `{{column}}` names aligned with the packaged file.

## Stage-by-Stage Formatting Checklist

Use this table when you need a compact reminder of where single-brace substitution runs.

| Stage | Single-brace pass on `system_prompt` | Single-brace pass on `prompt` |
| --- | --- | --- |
| `qa_generation` | Yes, `{num_few_shot_samples}` and `{num_questions}` | Yes, `{num_questions}` only |
| `question_judge` | No | No |
| `hallucination_filter` | Yes, `{num_choices}` becomes `{{num_choices}}` | Yes, `{choices}` becomes `{{choices_text}}` |
| `easiness_filter` | Same as hallucination | Same as hallucination |
| `distractor_expansion` | No | No |
| `distractor_validity` | Yes, `{num_choices}` becomes `4` or `10` | No |

`distractor_expansion` is fully forwarded, so treat it like judging for placeholder editing rules.

## Practical Tips

Duplicate an entire packaged stage before you delete lines.
Partial copies are the most common source of missing `{{column}}` names.

When you only need small wording edits, change sentences around the placeholder lines first and leave the markers alone until you must move them.

You must still ship `hallucination_filter` and `easiness_filter` blocks in your override file even when your edits focus on question generation.
