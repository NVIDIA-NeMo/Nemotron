---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Reference for the tokenizer_extension steps: YAML fields per step, CLI syntax, output files, and troubleshooting."
topics: ["Tokenizer Extension", "Reference"]
tags: ["Reference", "Tokenizer Extension"]
content:
  type: "Reference"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Reference

Lookup material for the four `tokenizer_extension` steps.
Every field on these pages traces to the step's `config/default.yaml`, `step.toml`, or executor source under `src/nemotron/steps/tokenizer_extension/`.

```{toctree}
:maxdepth: 1
:hidden:

extend-config
init-embeddings-config
evaluate-config
eval-init-config
cli
outputs
troubleshooting
```

| Page | Content |
|------|---------|
| {doc}`extend-config` | `extend/config/default.yaml`: method, language, corpus, and output fields |
| {doc}`init-embeddings-config` | `init_embeddings/config/default.yaml`: arm, engine, and per-engine fields |
| {doc}`evaluate-config` | `evaluate/config/default.yaml`: tokenizer and evaluation corpus fields |
| {doc}`eval-init-config` | `eval_init/config/default.yaml`: models, corpus, and budget fields |
| {doc}`cli` | `nemotron steps run tokenizer_extension/<step>` syntax and global options |
| {doc}`outputs` | Directories and JSON files each step writes |
| {doc}`troubleshooting` | Named errors from the `step.toml` files with their recoveries |

## Where to Read More

| Document | Content |
|----------|---------|
| [`src/nemotron/steps/tokenizer_extension/guide.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/tokenizer_extension/guide.md) | Category overview and quickstart |
| [`src/nemotron/steps/tokenizer_extension/LANGUAGES.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/tokenizer_extension/LANGUAGES.md) | Language profiles and registration |
| [`src/nemotron/steps/tokenizer_extension/extend/README.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/tokenizer_extension/extend/README.md) | Extension methods and the constructive splice |
| [`src/nemotron/steps/tokenizer_extension/init_embeddings/README.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/tokenizer_extension/init_embeddings/README.md) | Initialization engines and the Add versus Replace support table |
| [`src/nemotron/steps/tokenizer_extension/evaluate/README.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/tokenizer_extension/evaluate/README.md) | Fertility evaluation |
| [`src/nemotron/steps/tokenizer_extension/eval_init/README.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/tokenizer_extension/eval_init/README.md) | BPB evaluation |
| [`src/nemotron/steps/tokenizer_extension/guidebook/README.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/tokenizer_extension/guidebook/README.md) | Measured results and decision rules |
