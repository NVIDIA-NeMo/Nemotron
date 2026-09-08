---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Run the tokenizer_extension steps on Lepton, Slurm, or DGX Cloud with the generated *_tokenizer_* environment profiles."
topics: ["Tokenizer Extension", "Infrastructure"]
tags: ["How-To", "Tokenizer Extension"]
content:
  type: "How-To"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "DevOps"]
---

# Run on a Cluster

The bundled environment templates for Lepton, Slurm, and DGX Cloud each define four profiles for the `tokenizer_extension` steps, because the steps have different resource shapes.
Pass a profile with `--run` for attached execution or `--batch` for detached execution.

## Before You Begin

Generate and edit an environment profile file as described in {doc}`../../steps/basics`:

```console
$ uv run nemotron steps run env/env_toml -c slurm
$ export NEMOTRON_ENV_FILE=env.slurm.toml
```

Substitute `-c lepton` or `-c dgxcloud` for the other targets.
Replace the site-specific values in the generated file before use.

## Profiles

The generated profiles follow the pattern `<target>_tokenizer_<step>`.

| Profile suffix | Step | Shape | Optional dependencies installed at start-up |
|----------------|------|-------|---------------------------------------------|
| `_tokenizer_extend` | `extend` | CPU only, 1 node | `indic-nlp-library` |
| `_tokenizer_init_embeddings` | `init_embeddings` | 1 GPU, 1 node | `fasttext-wheel`, `indic-nlp-library` |
| `_tokenizer_evaluate` | `evaluate` | CPU only, 1 node | none |
| `_tokenizer_eval_init` | `eval_init` | GPU, 1 node | none |

The CPU profiles select a CPU partition or CPU resource shape; the `extend` trainer holds the whole word-count table in memory, so choose a large-memory shape for a large corpus.
The `init_embeddings` profile requests a single GPU, which is enough for every engine except `subword.gemma_weighted` on a large Gemma model.

Each profile also sets `TOKEXT_OUTPUT_DIR` to a shared workspace path.
The `init_embeddings` profile sets `FASTTEXT_CACHE_DIR`, the directory where `focus` caches downloaded `cc.<code>.300.bin` vectors so that a second run reuses them.

## Run the Pipeline Detached

The following commands use the Slurm profile names; replace the `slurm_` prefix with `lepton_` or `dgxcloud_` for the other targets.

```console
$ L=hindi
$ OUT=$TOKEXT_OUTPUT_DIR

$ uv run nemotron steps run tokenizer_extension/extend \
    -b slurm_tokenizer_extend -c default \
    language=$L method=add extension_size=30000 output_dir=$OUT

$ uv run nemotron steps run tokenizer_extension/init_embeddings \
    -b slurm_tokenizer_init_embeddings -c default \
    language=$L arm=add extended_tokenizer=$OUT/add \
    output_dir=$OUT/resized_checkpoint

$ uv run nemotron steps run tokenizer_extension/evaluate \
    -b slurm_tokenizer_evaluate -c default tokenizer=$OUT/add

$ uv run nemotron steps run tokenizer_extension/eval_init \
    -b slurm_tokenizer_eval_init -c default \
    models=[$OUT/resized_checkpoint] max_docs=2000
```

Paths passed as overrides must be visible to the compute node.
Use the shared workspace that the profile's `TOKEXT_OUTPUT_DIR` points at rather than a path on the submitting host.

## Confirm the Merged Configuration First

Add `--dry-run` to compile the configuration without submitting a job:

```console
$ uv run nemotron steps run tokenizer_extension/extend -d -c default language=$L method=add
```

## Related Pages

- Profile generation and guardrails: {doc}`../../steps/basics`
- CLI options: {doc}`../reference/cli`
