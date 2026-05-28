<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

(sdg-reference-index)=
# SDG Reference

Complete specifications for the SDG pipeline. For pipeline overview and when to use it, refer to {doc}`../index`.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`file-code;1.5em;sd-mr-1` Config Schema
:link: config-schema
:link-type: doc
All YAML fields: top-level settings, seed dataset, model aliases, column types, and output projections.
+++
{bdg-secondary}`lookup`
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` CLI Reference
:link: cli-reference
:link-type: doc
`nemotron steps run sdg/data_designer` flags and hydra override syntax.
+++
{bdg-secondary}`lookup`
:::

:::{grid-item-card} {octicon}`arrow-switch;1.5em;sd-mr-1` Output Projections
:link: output-projections
:link-type: doc
The three projection shapes with annotated JSONL examples.
+++
{bdg-secondary}`lookup`
:::

:::{grid-item-card} {octicon}`alert;1.5em;sd-mr-1` Troubleshooting
:link: troubleshooting
:link-type: doc
Failure modes for local runs and cluster dispatch.
+++
{bdg-secondary}`lookup`
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

config-schema
cli-reference
output-projections
troubleshooting
```
