# Model Evaluation (NeMo Evaluator)

```{step-toml} src/nemotron/steps/eval/model_eval/step.toml
```

## Hugging Face Access

`eval/model_eval` may run benchmark containers that download Hugging Face
datasets, tokenizers, or model assets. Export `HF_TOKEN` before running the
step, even for public assets, so Hugging Face does not serve the job as an
unauthenticated client and rate-limit it with `429 Too Many Requests`.

```bash
export HF_TOKEN=<your-hf-token>
```

For remote or container-backed runs, make sure your selected env profile passes
`HF_TOKEN` through to the job environment. See [Train Models getting
started](/train-models/getting-started.html) for the shared token setup.

## Reference Implementation

```{literalinclude} ../../../../src/nemotron/steps/eval/model_eval/step.py
:language: python
:caption: step.py
```

## Starter Configs

```{literalinclude} ../../../../src/nemotron/steps/eval/model_eval/config/default.yaml
:language: yaml
:caption: config/default.yaml
```

```{literalinclude} ../../../../src/nemotron/steps/eval/model_eval/config/tiny.yaml
:language: yaml
:caption: config/tiny.yaml
```
