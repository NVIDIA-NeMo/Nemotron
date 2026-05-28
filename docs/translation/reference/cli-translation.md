---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "CLI reference for nemotron steps run translate/nemo_curator."
topics: ["Translation", "CLI"]
tags: ["Reference", "CLI"]
content:
  type: "Reference"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

# CLI Reference for Translation

Syntax, global flags, and merge rules for `nemotron steps run translate/nemo_curator`. Pair this page with {doc}`translate-config` for YAML field meanings.

## Synopsis

```bash
uv run nemotron steps run translate/nemo_curator [GLOBAL OPTIONS] [-c CONFIG] [DOTLIST_OVERRIDES...]
```

## Global Options

These flags mirror other Nemotron commands:

| Flag | Purpose |
|------|---------|
| `-c NAME`, `--config NAME` | Select `NAME.yaml` inside `src/nemotron/steps/translate/nemo_curator/config/` or pass an explicit `*.yaml` path. |
| `-d`, `--dry-run` | Print the merged OmegaConf YAML without executing `TranslationStage`. |
| `-r`, `--run PROFILE` | Run attached through an environment profile such as `lepton_translate` or a Slurm profile. |
| `-b`, `--batch PROFILE` | Submit detached through an environment profile such as `lepton_translate` or a Slurm profile. |

Invocation without `-c` loads `default` automatically through `parse_config`.

For local Curator execution through `uv run`, set:

```bash
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
```

This keeps Ray workers on the synchronized project environment instead of
letting Ray ask uv to create a separate worker environment.

## Dotlist Overrides

Anything after the global flags that matches `key=value` merges into the YAML dictionary loaded from `default.yaml`. Nested keys use dotted paths:

```bash
uv run nemotron steps run translate/nemo_curator -c default \
  faith_eval.threshold=3.1 \
  server.model=YOUR_MODEL \
  input_path=/data/text.jsonl \
  output_dir=/data/translated \
  source_language=en \
  target_language=ar
```

## Restrictions Specific to Translation

- Passthrough arguments are not supported.
- For remote execution, use a normal Nemotron env profile with `-r` or `-b`, such as `lepton_translate`.

## Artifact Overrides

The current translation step reads `input_path` directly from YAML and dotlist overrides.

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Dry-run printed or translation finished successfully. |
| `1` | Validation failures such as missing languages, illegal backend configuration, unsupported CLI mode, or missing API keys. |

## Related Pages

- YAML keys: {doc}`translate-config`
- Tutorial invocation: {doc}`../getting-started`
