---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "CLI reference for nemotron steps run tokenizer_extension/{extend,init_embeddings,evaluate,eval_init}."
topics: ["Tokenizer Extension", "CLI"]
tags: ["Reference", "CLI"]
content:
  type: "Reference"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

# CLI Reference

Syntax, global options, and override rules for the four `tokenizer_extension` steps.
Pair this page with the per-step YAML references for field meanings.

## Synopsis

```bash
uv run nemotron steps run tokenizer_extension/<step> [GLOBAL OPTIONS] [-c CONFIG] [DOTLIST_OVERRIDES...]
```

`<step>` is one of `extend`, `init_embeddings`, `evaluate`, or `eval_init`.

## Global Options

| Option | Purpose |
|--------|---------|
| `-c NAME`, `--config NAME` | Select `NAME.yaml` inside the step's `config/` directory, or pass an explicit `*.yaml` path. Each step ships `default`. |
| `-d`, `--dry-run` | Print the merged configuration and exit without running the step. |
| `-r PROFILE`, `--run PROFILE` | Attached execution through an environment profile such as `slurm_tokenizer_extend`. |
| `-b PROFILE`, `--batch PROFILE` | Detached execution through an environment profile. |

Without `-r` or `-b`, the step runs on the current host.
Environment profiles come from the file named by `NEMOTRON_ENV_FILE`, which defaults to `env.toml`; see {doc}`../how-to/run-on-a-cluster`.

## Dotlist Overrides

Arguments of the form `key=value` after the options merge into the loaded YAML.
Nested keys use dotted paths, and list values use bracket syntax:

```bash
uv run nemotron steps run tokenizer_extension/extend -c default \
    language=vietnamese method=replace extension_size=30000 \
    corpus.hf_dataset=null corpus.path=./data/vi corpus.glob='*.parquet'

uv run nemotron steps run tokenizer_extension/eval_init -c default \
    models=[./output/resized_checkpoint,./output/resized_checkpoint_replace] max_docs=2000
```

Setting a key to `null` clears the default; `extend` and `evaluate` require exactly one of `corpus.hf_dataset` or `corpus.path` to be set.

## Step-Specific Notes

| Step | Note |
|------|------|
| `extend` | One `method` per invocation. Run the step once per arm you intend to compare. |
| `init_embeddings` | `arm` must match the tokenizer directory: `add` for `add/` or `expand/`, `replace` for `replace/`. |
| `evaluate` | Use the same corpus block for every tokenizer in a comparison. |
| `eval_init` | Use `max_docs`, not `max_tokens`, when scoring more than one model. |

## Inspect a Step

```bash
uv run nemotron steps show tokenizer_extension/extend
```

Prints the step's `step.toml` metadata, including its declared configuration fields and errors.

## Related Pages

- {doc}`extend-config`
- {doc}`init-embeddings-config`
- {doc}`evaluate-config`
- {doc}`eval-init-config`
- {doc}`../../steps/basics`
