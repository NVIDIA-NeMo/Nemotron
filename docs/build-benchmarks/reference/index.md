<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Reference

Specifications grounded in `src/nemotron/steps/byob`.

```{toctree}
:maxdepth: 1
:hidden:

benchmarks
output-files
generate-config
translation-config
troubleshooting
```

## Outputs

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`file-code;1.5em;sd-mr-1` Output files
:link: output-files
:link-type: doc
Seed, stage cache, raw and final Parquet paths.
+++
{bdg-secondary}`parquet`
:::

:::{grid-item-card} {octicon}`alert;1.5em;sd-mr-1` Troubleshooting
:link: troubleshooting
:link-type: doc
Common configuration errors, missing caches, filtering, and endpoint issues.
+++
{bdg-secondary}`faq`
:::

::::

## Configuration

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Generation YAML
:link: generate-config
:link-type: doc
Required keys for `ByobConfig.from_yaml`.
+++
{bdg-secondary}`yaml`
:::

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Translation YAML
:link: translation-config
:link-type: doc
`ByobTranslationConfig.from_yaml` requirements.
+++
{bdg-secondary}`yaml`
:::

::::

## Source benchmarks

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Allowed Hugging Face datasets
:link: benchmarks
:link-type: doc
Identifiers and default subsets from `runtime/constants.py`.
+++
{bdg-secondary}`hf_dataset`
:::

::::

## Related documentation

- Tutorial: {doc}`../getting-started`
- Concepts: {doc}`../explanation/index`
