<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(model-eval-reference-index)=
# Model Evaluation Reference

Lookup pages for `eval/model_eval`.
For the section overview, refer to {doc}`../index`.
For procedural walk-throughs, refer to {doc}`../how-to/index`.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`file-code;1.5em;sd-mr-1` Configuration Reference
:link: config-schema
:link-type: doc
YAML schema for `default.yaml` and `tiny_chat.yaml`, field by field.
+++
{bdg-secondary}`yaml`
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` CLI Reference
:link: cli-reference
:link-type: doc
`nemotron steps run eval/model_eval` flags and Hydra overrides.
+++
{bdg-secondary}`cli`
:::

:::{grid-item-card} {octicon}`archive;1.5em;sd-mr-1` Output Artifacts
:link: output-artifacts
:link-type: doc
The `eval_results` contract and the on-disk directory layout.
+++
{bdg-secondary}`artifacts`
:::

:::{grid-item-card} {octicon}`list-unordered;1.5em;sd-mr-1` Benchmarks Catalog
:link: benchmarks-catalog
:link-type: doc
Benchmark identifiers grouped by family, with endpoint-type guidance.
+++
{bdg-secondary}`tasks`
:::

:::{grid-item-card} {octicon}`alert;1.5em;sd-mr-1` Troubleshooting
:link: troubleshooting
:link-type: doc
Named error modes from `step.toml`, with the most common cause and the recovery for each.
+++
{bdg-secondary}`errors`
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

config-schema
cli-reference
output-artifacts
benchmarks-catalog
troubleshooting
```
