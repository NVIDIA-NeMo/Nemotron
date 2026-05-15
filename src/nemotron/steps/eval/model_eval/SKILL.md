---
name: nemotron-eval-model-eval
description: Configure Nemotron eval/model_eval to deploy a trained checkpoint behind an OpenAI-compatible endpoint and run NeMo Evaluator benchmark suites. Use for chat/instruction benchmarks, log-probability tasks, reasoning-model evaluation, and producing consolidated eval_results from checkpoint_hf or checkpoint_megatron inputs.
---

# Model Evaluation (NeMo Evaluator)

Use `eval/model_eval` to score a checkpoint on standard benchmarks. The step
deploys the model behind an OpenAI-compatible endpoint, then NeMo Evaluator
hits that endpoint with the benchmark suite.

Before changing configs, read `step.toml` end-to-end for the full
strategies/errors/parameters list.

## Inputs and outputs

- Consume **either** `checkpoint_megatron` (iter_* dir from Megatron-Bridge)
  or `checkpoint_hf` (HF safetensors). Both are optional in the manifest, but
  exactly one must be present at run time.
- Produce `eval_results` — benchmark metrics + artifacts + run summary.

## Configure

- **Match endpoint type to benchmark family.** Chat/instruction → chat
  endpoint. Log-probability (arc_challenge, hellaswag, piqa, etc.) → completions
  endpoint with `logprobs` support.
- Set `deployment.url` to the OpenAI-compatible serving endpoint.
- **Tokenizer is required for log-probability tasks.** For `checkpoint_megatron`,
  point at `checkpoint/tokenizer`. For `checkpoint_hf`, use the HF handle or path.
- **Megatron deployments need the iter_* path**, not the parent output dir.
- **Reasoning models** need higher `max_new_tokens`, reasoning-trace
  processing on, and the temperature/top_p from the model card.
- Reference [src/nemotron/steps/patterns/eval-before-and-after-training.md](../../patterns/eval-before-and-after-training.md)
  before treating any single eval as a result.

## Local files

- Contract: [step.toml](step.toml)
- Runner: [step.py](step.py)
- Configs: `config/default.yaml`, `config/tiny.yaml`

## Guardrails

- Don't compare scores across different endpoint types or different
  generation settings.
- Don't add `convert/megatron_to_hf` "just in case" — pick one input artifact
  and configure the matching deployment path.
- Inspect a handful of generations before trusting aggregate metrics.
