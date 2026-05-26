# Pretrain Data Prep

```{step-toml} src/nemotron/steps/data_prep/pretrain_prep/step.toml
```

## Hugging Face Access

`data_prep/pretrain_prep` may download Hugging Face datasets and tokenizers
from the configured blend and `tokenizer.model`. Export `HF_TOKEN` before
running the step, even for public assets, so Hugging Face does not serve the job
as an unauthenticated client and rate-limit it with `429 Too Many Requests`.

```bash
export HF_TOKEN=<your-hf-token>
```

See [Train Models getting started](/train-models/getting-started.html) for the
shared token setup.

## Reference Implementation

```{literalinclude} ../../../../src/nemotron/steps/data_prep/pretrain_prep/step.py
:language: python
:caption: step.py
```

## Starter Configs

```{literalinclude} ../../../../src/nemotron/steps/data_prep/pretrain_prep/config/default.yaml
:language: yaml
:caption: config/default.yaml
```

```{literalinclude} ../../../../src/nemotron/steps/data_prep/pretrain_prep/config/tiny.yaml
:language: yaml
:caption: config/tiny.yaml
```
