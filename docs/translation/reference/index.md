---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Reference index for nemotron steps run translate/nemo_curator YAML, CLI, and input and output shapes."
topics: ["Translation", "Reference"]
tags: ["Reference", "Translation"]
content:
  type: "Reference"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

# Reference for Translation

Lookup pages for YAML keys, CLI flags, and file shapes. Prefer the tutorial and how-to guides for procedures; use this section when you need exact parameters or syntax.

```{toctree}
:maxdepth: 1
:hidden:

translate-config
cli-translation
io-format
troubleshooting
```

Specifications for `nemotron steps run translate/nemo_curator`.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Translation YAML
:link: translate-config
:link-type: doc
Anchored on `config/default.yaml` plus FAITH semantics.
+++
{bdg-secondary}`yaml`
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` CLI syntax
:link: cli-translation
:link-type: doc
Global recipe flags and translation-specific constraints.
+++
{bdg-secondary}`cli`
:::

:::{grid-item-card} {octicon}`file-code;1.5em;sd-mr-1` I/O format
:link: io-format
:link-type: doc
How `input_path` layouts map to `output_dir` shards.
+++
{bdg-secondary}`jsonl`
:::

::::
