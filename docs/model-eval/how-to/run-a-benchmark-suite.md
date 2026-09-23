<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(model-eval-run-a-benchmark-suite)=
# Run A Benchmark Suite

This guide selects one of the *benchmark suites* that ship with `eval/model_eval`, pins a language or subset where a task requires it, and narrows the task list with `-t`.
Every shipped suite inherits from `direct.yaml`, so the endpoint, tokenizer, and results settings from {doc}`run-direct-mode-evaluation` apply unchanged.

## Prerequisites

- The required direct-mode environment variables are exported: `EVAL_ENDPOINT_URL`, `EVAL_MODEL_HANDLE`, `EVAL_RESULTS_DIR`, and, if the endpoint requires authentication, the credential named by `EVAL_API_KEY_NAME`.
- `EVAL_TOKENIZER` is set if the served model name is an alias, the tokenizer is at a custom path, or you used Tokenizer Extension.
- The `<backend>_eval_direct` profile is present in `env.toml`.
- You know whether the served model is a base model or an instruct model.

## Choose A Suite

The endpoint type follows the model: a base model, including a continued-pretraining checkpoint, is scored with completions tasks; an instruct model is scored with chat tasks.
The chat suites set `target.api_endpoint.type: chat` in their config; the completions suites use the `direct.yaml` default.

| Suite | Endpoint type | Tasks | Use for |
| --- | --- | --- | --- |
| `base_en` | completions | `adlr_arc_challenge_llama_25_shot`, `hellaswag`, `gsm8k` | English capability of a base model. |
| `mmlu_prox` | completions | `mmlu_prox_completions`, one language per run | Multilingual knowledge of a base model. |
| `milu` | completions | `milu_<MILU_LANG>`, `milu_English` | Indic-language knowledge; uses the sovereign harness image. |
| `instruct_en` | chat | `ifeval`, `mmlu_instruct`, `gsm8k_cot_instruct` | English capability of an instruct model. |
| `mmlu_prox_chat` | chat | `mmlu_prox_chat`, one language per run | Multilingual knowledge of an instruct model. |

`tiny_chat` is a launcher-mode verification config, not a direct-mode suite; refer to {doc}`run-hosted-evaluation`.

Point `EVAL_ENDPOINT_URL` at the path that matches the suite: `/v1/completions` for the completions suites and `/v1/chat/completions` for the chat suites.

```bash
uv run nemotron steps run eval/model_eval -c base_en --batch slurm_eval_direct
```

Scores are comparable only between runs that share an endpoint type, tokenizer, harness image, and generation parameters.
Do not compare a completions score with a chat score for the same benchmark.

## Pin The Language For MMLU-ProX And MILU

`mmlu_prox` and `mmlu_prox_chat` read `MMLU_PROX_LANG` and default to `en`.
The value becomes `config.params.task=mmlu_prox_<lang>`, which selects one language subset.

```bash
MMLU_PROX_LANG=hi uv run nemotron steps run eval/model_eval \
  -c mmlu_prox --batch slurm_eval_direct
```

`milu` reads `MILU_LANG`, defaults to `Hindi`, and always adds `milu_English` as a forgetting check.

```bash
MILU_LANG=Punjabi uv run nemotron steps run eval/model_eval \
  -c milu --batch slurm_eval_direct
```

Give each language its own `EVAL_RESULTS_DIR`; direct mode refuses to write into a task directory that already holds results.

Direct mode resolves `${oc.env:...}` inside the job, and the shipped `*_eval_direct` profiles forward the endpoint, tokenizer, result, limit, and credential variables plus `ENDPOINT_TOKEN`.
They intentionally omit `EVAL_HARNESS_IMAGE` because `direct.yaml` resolves and forwards the selected image itself; they also omit `MMLU_PROX_LANG` and `MILU_LANG`.
If your profile does not forward the language variable, pin the subset with a dotlist override instead, which travels with the config: `evaluation.nemo_evaluator_config.config.params.task=mmlu_prox_hi` for the `mmlu_prox*` suites, or `-t milu_Punjabi -t milu_English` for `milu`.
Confirm the resolved task in `run_manifest.json` before relying on a run.

## Pin Multi-Subset Tasks

Some task names cover a whole family.
`mmlu_prox_completions` and `mmlu_prox_chat` default to all 29 languages, and `global_mmlu` and `global_mmlu_full` behave the same way.
An unpinned run does not fail and does not warn; it issues roughly 350 000 requests and does not finish in a practical time.
The shipped `mmlu_prox*` suites pin the subset for you.
When you request such a task with `-t` instead, pin it on the command line.

```bash
uv run nemotron steps run eval/model_eval -c direct --batch slurm_eval_direct \
  -t mmlu_prox_completions \
  evaluation.nemo_evaluator_config.config.params.task=mmlu_prox_hi
```

`config.params.task` is global to the run, so give a multi-subset task its own invocation and its own `EVAL_RESULTS_DIR`.
To combine it with other tasks in one config, pin the subset per task instead.

```yaml
evaluation:
  tasks:
    - name: mmlu_prox_completions
      nemo_evaluator_config: {config: {params: {task: mmlu_prox_hi}}}
```

Before running any task for the first time, read its defaults; the `task` default, the `request_timeout`, and the `container_digest` are the fields that most often need attention.

```bash
uv run nemo-evaluator-launcher ls task mmlu_prox_completions --json
```

## Narrow Or Extend The Task List

`-t <task>` is repeatable.
In direct mode the list is used exactly as supplied, so it can both narrow a suite and add a task that the harness image ships but the suite does not list.

```bash
uv run nemotron steps run eval/model_eval -c base_en --batch slurm_eval_direct \
  -t hellaswag -t gsm8k
```

The launcher registry is not the authority for direct mode.
List the tasks in the harness image the suite selects, because task sets differ between image tags, and some benchmarks, such as MILU, exist only in an image.

```bash
docker run --rm nvcr.io/nvidia/eval-factory/lm-evaluation-harness:26.03 nemo-evaluator ls
```

## Enable Code Benchmarks

Code benchmarks such as `humaneval_instruct` execute model-generated code inside the evaluation job.
lm-evaluation-harness refuses to run them unless you confirm, so `instruct_en` ships with `humaneval_instruct` commented out.
To enable it, add the task with the harness flag under `extra.args` in a copy of the suite config.

```yaml
evaluation:
  tasks:
    - name: humaneval_instruct
      nemo_evaluator_config:
        config:
          params:
            extra:
              args: '--confirm_run_unsafe_code'
```

Run the copy with `-c /path/to/your-config.yaml`.
Enable this only in an environment where executing untrusted code is acceptable.

## Adjust Generation And Request Parameters

Direct mode accepts these top-level keys under `evaluation.nemo_evaluator_config.config.params`: `task`, `limit_samples`, `parallelism`, `request_timeout`, `max_retries`, `temperature`, `top_p`, `max_new_tokens`, and `extra`.
Any other top-level key stops the run with `unsupported nemo_evaluator_config params`; harness-specific arguments belong under `extra`.
Per-task values in `evaluation.tasks[].nemo_evaluator_config` override the global ones, and the merged result is recorded per task in `run_manifest.json`.

```bash
uv run nemotron steps run eval/model_eval -c instruct_en --batch slurm_eval_direct \
  evaluation.nemo_evaluator_config.config.params.parallelism=8 \
  evaluation.nemo_evaluator_config.config.params.request_timeout=7200
```

## Related

- {doc}`run-direct-mode-evaluation` for the environment variables and results layout.
- {doc}`../explanation/endpoint-types-and-benchmarks` for why the endpoint type follows the model.
- {doc}`../reference/benchmarks-catalog` for the task identifiers in each suite.
- {doc}`../reference/config-schema` for every key the suite configs set.
