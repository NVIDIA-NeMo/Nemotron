---
name: nemotron-eval-model-eval
description: Configure Nemotron eval/model_eval to run NeMo Evaluator Launcher benchmark suites for hosted endpoints or Megatron Bridge checkpoints. Use for MMLU, HellaSwag, standard English benchmarks, container-backed tasks, sovereign or multilingual benchmark containers, and consolidated eval_results.
---

# Model Evaluation (NeMo Evaluator)

Use `eval/model_eval` to run NeMo Evaluator Launcher. This generic step follows
the same launcher pattern as `nano3 eval` and `super3 eval`: compile YAML,
strip Nemotron-only run metadata, then call `run_eval()`.

Before changing configs, read `step.toml` end-to-end for the full
strategies/errors/parameters list.

## Inputs and outputs

- Consume a hosted endpoint config or `checkpoint_megatron`.
- Produce `eval_results`: benchmark metrics, artifacts, logs, and optional W&B export.

## Configure

- For hosted endpoints, use `config/tiny_chat.yaml`; it sets `deployment.type=none` and configures `target.api_endpoint`.
- For Megatron checkpoint evaluation, start from `config/default.yaml`; it follows the Nano3/Super3 pattern where NeMo Evaluator Launcher owns deployment and benchmark execution.
- Store only the API key environment variable name in YAML, usually `NVIDIA_API_KEY`.
- Use exact model IDs returned by the serving provider when available.
- Use exact task IDs from `nemo-evaluator-launcher ls tasks` or `nemo-evaluator-launcher ls task <task_id>`. Packaged tasks use IDs such as `mmlu_instruct`, `adlr_mmlu`, and `hellaswag`; do not prepend the harness name unless the task container exposes that exact dotted ID.
- Match task choice to endpoint capability. Hosted NVIDIA Integrate smoke tests use `tiny_chat.yaml` with a chat-compatible task. Logprob benchmarks such as HellaSwag need a completions endpoint with logprobs and are not part of the hosted QA smoke path.
- Start smoke tests with `limit_samples=1` before full MMLU or HellaSwag runs.
- Megatron deployments need the concrete `iter_*` path, not the parent output dir.
- Reasoning models often need higher `max_new_tokens` and model-card sampling defaults.
- Reference [src/nemotron/steps/patterns/eval-before-and-after-training.md](../../patterns/eval-before-and-after-training.md)
  before treating any single eval as a result.

## Local files

- Contract: [step.toml](step.toml)
- Runner: [step.py](step.py)
- Runtime helpers: [runtime.py](runtime.py)
- Configs: `config/default.yaml` for Megatron checkpoint evaluation, `config/tiny_chat.yaml` for hosted chat smoke tests

## Common Commands

Hosted chat MMLU smoke:

```bash
export NEMO_EVALUATOR_MODEL_ID=<exact-model-id>
export NEMO_EVALUATOR_MODEL_URL=<openai-compatible-chat-completions-endpoint-url>
export NEMO_EVALUATOR_API_KEY_NAME=NVIDIA_API_KEY
export NEMO_EVALUATOR_ENDPOINT_TYPE=chat

uv run nemotron steps run eval/model_eval -c tiny_chat
```

Preview compiled launcher config without running:

```bash
uv run nemotron steps run eval/model_eval -c tiny_chat dry_run=true
```

## Guardrails

- Don't compare scores across different endpoint types or different
  generation settings.
- Don't add checkpoint conversion "just in case"; pick the artifact format and configure the matching deployment path.
- Don't use the direct NeMo Evaluator API path for this step; use NeMo Evaluator Launcher.
- Don't put raw API keys in YAML or command output.
- Inspect a handful of generations before trusting aggregate metrics.
