# Nemotron Evaluation

Use this category before and after any quality-changing stage. Evaluation is
how you know SFT, RL, optimization, or conversion actually helped.

## Developer Journey

1. Decide the target: a hosted endpoint or a Megatron checkpoint.
2. Pick task IDs your endpoint can actually serve (chat vs completions).
3. Smoke with `limit_samples=1` before any full run.
4. Re-evaluate against the same task set after each pipeline change.
5. Inspect a few generations, not only aggregate scores.

## Steps

| Need | Step | Input | Output |
|---|---|---|---|
| Run a benchmark suite via NeMo Evaluator Launcher | [`eval/model_eval`](model_eval/README.md) | hosted endpoint or `checkpoint_megatron` | `eval_results` |

## Data And Artifact Flow

```text
hosted endpoint
  -> eval/model_eval -> eval_results

checkpoint_megatron (concrete iter_*)
  -> eval/model_eval (NeMo Evaluator Launcher deploys + runs)
  -> eval_results
```

If a Megatron-Bridge step produced the checkpoint, point at the concrete
`iter_*` directory, not the parent run dir. If the consumer is HF-native,
convert with `convert/megatron_to_hf` first instead of pointing eval at the
Megatron path.

## Guardrails

- Don't compare scores across endpoint types or generation settings.
- Don't ship hosted API keys in YAML; store the variable name only.
- Don't treat a `limit_samples=1` smoke as evidence of model quality.
