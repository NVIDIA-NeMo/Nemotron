# Data Preparation

```{include} ../../../../src/nemotron/steps/data_prep/guide.md
```

## Hugging Face Access

All `data_prep/*` steps may read Hugging Face datasets, model tokenizers, or
placeholder records through the Hub. Set `HF_TOKEN` before running these steps,
even for public assets, to avoid unauthenticated Hub rate limits such as `429
Too Many Requests`.

```bash
export HF_TOKEN=<your-hf-token>
```

See [Train Models getting started](/train-models/getting-started.html) for the
shared token setup.

```{toctree}
:maxdepth: 1

pretrain-prep
rl-prep
sft-packing
```
