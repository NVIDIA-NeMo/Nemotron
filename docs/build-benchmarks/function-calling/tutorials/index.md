<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(bfcl-tutorials-index)=
# Function-Calling Benchmark Tutorials

These tutorials follow complete BFCL journeys. Start with the bundled tiny pack to
learn the artifact flow, run the credential-free assisted-authoring demonstration to
observe every review and publication gate, then use the manual lifecycle when you are
ready to publish and evaluate a pack of your own.

::::{grid} 1 1 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Build Your First Benchmark
:link: ../getting-started
:link-type: doc
Run the bundled `tiny_oracle_pack`, inspect its validation report, benchmark rows, and
publication manifest, without a model endpoint.
+++
{bdg-success}`5 min` {bdg-secondary}`beginner`
:::

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Run Assisted Authoring End to End
:link: assisted-authoring-demo
:link-type: doc
Watch source intake, certification, drafting, human-review boundaries, Gold validation,
publication, and evaluation in one credential-free demonstration.
+++
{bdg-success}`15–30 min` {bdg-secondary}`intermediate`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Publish and Evaluate a Manual Pack
:link: manual-pack-lifecycle
:link-type: doc
Take a reviewed Oracle Pack through validation, publication, independent-candidate
preflight, live evaluation, and artifact inspection.
+++
{bdg-secondary}`advanced` {bdg-secondary}`production-shaped`
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

Build Your First Benchmark <../getting-started>
Run Assisted Authoring End to End <assisted-authoring-demo>
Publish and Evaluate a Manual Pack <manual-pack-lifecycle>
```
