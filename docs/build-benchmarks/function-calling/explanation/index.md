<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Concepts

Background for `nemotron steps run byob/bfcl` with the `bfcl` family: what the pieces are, and why they are arranged this way.

Start with {doc}`../getting-started` if you have not produced a `benchmark.parquet` yet. These pages assume you have seen the pipeline run once.

```{toctree}
:maxdepth: 1
:hidden:

Pipeline Overview <pipeline-overview>
Pipeline Worked Example <pipeline-worked-example>
The Oracle Pack <oracle-pack>
Authoring Flows <authoring-flows>
Evaluation <evaluation>
```

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Pipeline overview
:link: pipeline-overview
:link-type: doc
Stage order from prepare through publication, what is checkpointed, and why the pipeline refuses configuration it will not honor.
+++
{bdg-secondary}`stages`
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Pipeline worked example
:link: pipeline-worked-example
:link-type: doc
Follow one missing-slot transfer-fee task from an abstract template through binding,
conversation planning, executable replay, selection, and publication.
+++
{bdg-secondary}`worked example`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` The Oracle Pack
:link: oracle-pack
:link-type: doc
The unit of truth: its file layout, the certification tiers, the Gold gate, and why pack code runs in its own process.
+++
{bdg-secondary}`oracle_pack`
:::

:::{grid-item-card} {octicon}`git-branch;1.5em;sd-mr-1` Authoring flows
:link: authoring-flows
:link-type: doc
Three ways to obtain a pack, one generation pipeline, and the two human authorization boundaries that a model cannot cross.
+++
{bdg-secondary}`lineage`
:::

:::{grid-item-card} {octicon}`check-circle;1.5em;sd-mr-1` Evaluation
:link: evaluation
:link-type: doc
Trace and executable scoring, the gates that run before any inference, and why the evaluator never sees the gold trace as a prompt.
+++
{bdg-secondary}`eval`
:::

::::

## Where to Go Next

- To do something rather than understand it, refer to {doc}`../how-to/index`.
- For field-level detail on configuration and artifacts, refer to {doc}`../reference/index`.
- The normative contracts live in the source tree next to the code they govern: [`src/nemotron/steps/byob/references/bfcl-oracle-pack.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/byob/references/bfcl-oracle-pack.md) for the pack, and [`src/nemotron/steps/byob/references/bfcl-eval-scoring-contract.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/byob/references/bfcl-eval-scoring-contract.md) for scoring.
