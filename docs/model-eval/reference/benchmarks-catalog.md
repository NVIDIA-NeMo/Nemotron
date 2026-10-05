<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(model-eval-benchmarks-catalog)=
# Tasks Catalog

This page catalogs the suites and task identifiers used by `eval/model_eval`.
Use it as a map, then verify exact names against the authority for your mode.

## Where Task Names Come From

| Mode | Authority | Command |
| --- | --- | --- |
| launcher | The NeMo Evaluator Launcher registry installed with `uv sync --extra evaluator`. | `uv run nemo-evaluator-launcher ls tasks` |
| direct | The harness image named by `harness_image`. Task sets differ between image tags, and some benchmarks exist only in an image. | `docker run --rm <harness-image> nemo-evaluator ls` |

```bash
uv run nemo-evaluator-launcher ls tasks
uv run nemo-evaluator-launcher ls task <task-id> --json
docker run --rm nvcr.io/nvidia/eval-factory/lm-evaluation-harness:26.03 nemo-evaluator ls
```

`ls task <task-id> --json` prints the task's defaults, including its `task` subset default, `request_timeout`, and `container_digest`.

## Naming Rule

Use the exact task id listed by the authority for your mode.
Do not prepend a harness name unless the launcher lists that exact dotted id.
In direct mode, task names are also used as directory names under `output_dir`, so a name containing a path separator is rejected.

## Direct-Mode Suites

Each suite is a config that inherits from `direct.yaml`.
Select one with `-c <suite>`.

| Suite | Endpoint type | Tasks | Image | Notes |
| --- | --- | --- | --- | --- |
| `direct` | completions | `adlr_arc_challenge_llama_25_shot`, `hellaswag` | lm-evaluation-harness | Base config; also the default task list when `-t` is not given. |
| `base_en` | completions | `adlr_arc_challenge_llama_25_shot`, `hellaswag`, `gsm8k` | lm-evaluation-harness | English capability of a base model. |
| `mmlu_prox` | completions | `mmlu_prox_completions` | lm-evaluation-harness | Pinned to one language by `config.params.task=mmlu_prox_${MMLU_PROX_LANG}`; default `en`. |
| `milu` | completions | `milu_${MILU_LANG}` (default `milu_Hindi`), `milu_English` | `nvcr.io/nvidia/eval-factory/sovereign:latest` | Indic-language knowledge; `milu_English` is included as a forgetting check. Overrides `harness_image`. |
| `instruct_en` | chat | `ifeval`, `mmlu_instruct`, `gsm8k_cot_instruct` | lm-evaluation-harness | English capability of an instruct model, with `temperature: 0.0`, `top_p: 1.0`, `max_new_tokens: 2048`. `humaneval_instruct` is shipped commented out because it executes model-generated code. |
| `mmlu_prox_chat` | chat | `mmlu_prox_chat` | lm-evaluation-harness | Chat counterpart of `mmlu_prox`, pinned the same way. |

The default lm-evaluation-harness image is `nvcr.io/nvidia/eval-factory/lm-evaluation-harness:26.03`.
`EVAL_HARNESS_IMAGE` overrides it for every suite; for published results, pin the image by digest.

## Launcher-Mode Starting Points

| Config | Task entries | When to use |
| --- | --- | --- |
| `tiny_chat.yaml` | `mmlu_instruct` | Hosted chat verification run. |
| `default.yaml` | `adlr_mmlu`, `hellaswag` | Launcher-managed Megatron checkpoint evaluation. |

## Chat And Instruction Tasks

These tasks use a chat endpoint and score generated answers.
They suit instruct models.

| Identifier | Used by | Notes |
| --- | --- | --- |
| `mmlu_instruct` | `tiny_chat.yaml`, `instruct_en` | Multiple-choice knowledge, answered in chat form. |
| `ifeval` | `instruct_en` | Instruction following. |
| `gsm8k_cot_instruct` | `instruct_en` | Grade-school mathematics with chain-of-thought prompting. |
| `humaneval_instruct` | `instruct_en` (commented out) | Code generation. Executes generated code; requires `extra.args: '--confirm_run_unsafe_code'`. |
| `mmlu_prox_chat` | `mmlu_prox_chat` | Multilingual MMLU-ProX; multi-subset, pin one language. |
| `adlr_mmlu` | `default.yaml` | Verify endpoint requirements in the installed launcher. |

## Completions Tasks

These tasks use a completions endpoint with few-shot prompts; the multiple-choice tasks among them score log probabilities, so the endpoint must return logprobs.
They suit base models, including continued-pretraining checkpoints.

| Identifier | Used by | Notes |
| --- | --- | --- |
| `hellaswag` | `default.yaml`, `direct`, `base_en` | Commonsense completion. |
| `adlr_arc_challenge_llama_25_shot` | `direct`, `base_en` | ARC-Challenge, 25-shot. |
| `gsm8k` | `base_en` | Grade-school mathematics, generative, completions endpoint. |
| `mmlu_prox_completions` | `mmlu_prox` | Multilingual MMLU-ProX; multi-subset, pin one language. |
| `milu_<Language>`, `milu_English` | `milu` | MILU Indic benchmark; exists only in the sovereign image. |

If the served model name is its Hugging Face model ID, the evaluator loads that tokenizer automatically.
If you used Tokenizer Extension, set the generated tokenizer under:

```text
evaluation.nemo_evaluator_config.config.params.extra.tokenizer
evaluation.nemo_evaluator_config.config.params.extra.tokenizer_backend
```

## Multi-Subset Tasks

`mmlu_prox_completions`, `mmlu_prox_chat`, `global_mmlu`, and `global_mmlu_full` expand to every language subset unless `config.params.task` pins one.
An unpinned MMLU-ProX run does not warn; it covers all 29 languages, roughly 350 000 requests.
The shipped `mmlu_prox*` suites pin the subset; when requesting such a task with `-t`, pin it yourself as described in {doc}`../how-to/run-a-benchmark-suite`.

## Choosing Tasks

Ask four questions before changing the task list.

1. Does the authority for your mode list the task id exactly?
1. Does the endpoint type match the task family?
1. Is the task multi-subset, and if so, is the subset pinned?
1. Is this a verification run or a production comparison?

For production comparisons, keep the same task list, endpoint type, tokenizer, harness image, and generation parameters across baseline and post-training runs.

## Related

- {doc}`config-schema` for the `tasks` section and evaluator params.
- {doc}`../how-to/run-a-benchmark-suite` for selecting and pinning suites.
- {doc}`output-artifacts` for result layout expectations.
- {doc}`../explanation/endpoint-types-and-benchmarks` for endpoint/task pairing.
- {ref}`model-eval-comparing-runs` for before-and-after evaluation framing.
