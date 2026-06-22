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

(sdg-how-to-index)=
# Synthetic Data How-To Guides

This section provides task-focused guides for common SDG workflows.
For your first run, start with {doc}`../getting-started`.

If you are new to model training or want a calmer on-ramp before tasks, read {doc}`../using-skills` for how to run a productive session with a coding agent.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Run the Pipeline
:link: run
:link-type: doc
Preview, generate, and customize output path and projection.
+++
{bdg-success}`10 min` {bdg-secondary}`intermediate`
:::

:::{grid-item-card} {octicon}`briefcase;1.5em;sd-mr-1` Create a Domain Dataset
:link: create-domain-dataset
:link-type: doc
Adapt the pipeline to a custom domain with a seed file and multiple category dimensions.
+++
{bdg-success}`20 min` {bdg-secondary}`intermediate`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Generate Tool-Call Data
:link: tool-call-data
:link-type: doc
Generate multi-turn conversations with OpenAI-style tool calls for tool-use SFT.
+++
{bdg-success}`15 min` {bdg-secondary}`intermediate`
:::

:::{grid-item-card} {octicon}`git-compare;1.5em;sd-mr-1` Generate Preference Data
:link: preference-data
:link-type: doc
Generate DPO preference pairs (prompt / chosen / rejected) from `rl_pref.yaml`.
+++
{bdg-success}`15 min` {bdg-secondary}`intermediate`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Dispatch to a Cluster
:link: dispatch-to-cluster
:link-type: doc
Configure an env.toml profile and run SDG on Lepton or Slurm.
+++
{bdg-success}`30 min` {bdg-secondary}`intermediate`
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

Create a Domain Dataset <create-domain-dataset>
Create Tool-Calling Dataset <tool-call-data>
preference-data
dispatch-to-cluster
run
```
