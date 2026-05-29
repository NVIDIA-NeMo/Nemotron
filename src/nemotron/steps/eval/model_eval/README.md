# Model Evaluation (NeMo Evaluator)

Use `eval/model_eval` to run NeMo Evaluator Launcher. This generic step follows
the same launcher pattern as `nano3 eval` and `super3 eval`: compile YAML,
strip Nemotron-only run metadata, then call `run_eval()`.

Use this README for the hosted-vs-checkpoint workflow and task-selection rules; use `step.toml` for the full strategies, errors, and parameter list.

## Inputs and outputs

- Consume a hosted endpoint config or `checkpoint_megatron`.
- Produce `eval_results`: benchmark metrics, artifacts, logs, and optional W&B export.

## CLI And Overlay Knobs

Use `config/tiny_chat.yaml` for hosted endpoint smoke tests and `config/default.yaml`
for Megatron checkpoint evaluation. In a project overlay, developers usually
change:

- `evaluation.tasks`: task IDs from `nemo-evaluator-launcher ls tasks`.
- `target.api_endpoint.url`, `target.api_endpoint.model_id`, and
  `target.api_endpoint.type` for hosted endpoints.
- `target.api_endpoint.api_key_name`: environment variable name, not the key.
- `deployment.checkpoint_path`: concrete Megatron `iter_*` path.
- `evaluation.nemo_evaluator_config.config.params.limit_samples`: smoke before full runs.
- `dry_run`: preview compiled launcher config.

Example shape:

```bash
uv run nemotron steps run eval/model_eval \
  -c tiny_chat \
  target.api_endpoint.model_id=<served-model-id> \
  evaluation.nemo_evaluator_config.config.params.limit_samples=1
```

Related patterns:

- Reference [src/nemotron/steps/patterns/eval-before-and-after-training.md](../../patterns/eval-before-and-after-training.md)

## Repository Layout

- Manifest: [step.toml](step.toml)
- Runner: [step.py](step.py)
- Runtime helpers: [runtime.py](runtime.py)
- Configs: `config/default.yaml` for Megatron checkpoint evaluation, `config/tiny_chat.yaml` for hosted chat smoke tests

## Run It

Set the hosted-endpoint environment variables once:

```bash
export NEMO_EVALUATOR_MODEL_ID=<exact-model-id>
export NEMO_EVALUATOR_MODEL_URL=<openai-compatible-chat-completions-endpoint-url>
export NEMO_EVALUATOR_API_KEY_NAME=NVIDIA_API_KEY
export NEMO_EVALUATOR_ENDPOINT_TYPE=chat
```

Preview the compiled launcher config first, then smoke with `limit_samples=1`:

```bash
uv run nemotron steps run eval/model_eval -c tiny_chat dry_run=true
uv run nemotron steps run eval/model_eval -c tiny_chat \
  evaluation.nemo_evaluator_config.config.params.limit_samples=1
```

Run the full evaluation once the smoke looks right:

```bash
uv run nemotron steps run eval/model_eval -c tiny_chat
```

For Megatron checkpoint evaluation, use `-c default` and set
`deployment.checkpoint_path` to a concrete `iter_*` directory.

## Guardrails

- Don't compare scores across different endpoint types or different
  generation settings.
- Don't add checkpoint conversion "just in case"; pick the artifact format and configure the matching deployment path.
- Don't use the direct NeMo Evaluator API path for this step; use NeMo Evaluator Launcher.
- Don't put raw API keys in YAML or command output.
- Inspect a handful of generations before trusting aggregate metrics.
