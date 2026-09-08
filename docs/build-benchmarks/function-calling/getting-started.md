<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(getting-started-byob-bfcl)=
# Getting Started with Building Function-Calling Benchmarks

<!-- Tutorial: end-to-end `tiny` BFCL run; about 5 minutes; requires a Nemotron clone, uv, and the BYOB extra. No model endpoint needed. -->

::::{grid} 2

:::{grid-item-card}
:columns: 8

**What You'll Build**: A small function-calling benchmark generated from the bundled `tiny_oracle_pack`, an executable library-catalog pack that exists to exercise the pipeline quickly.

^^^

**In this tutorial, you will**:

1. Install Python dependencies.
2. Run the `tiny` configuration from the repository root.
3. Read the oracle validation report to see which certification tier the pack earned.
4. Inspect `benchmark.parquet` and the `run_manifest.json` commit marker.

{octicon}`clock;1.5em;sd-mr-1` This tutorial requires about 5 minutes to complete.
:::

:::{grid-item-card}
:columns: 4

{octicon}`flame;1.5em;sd-mr-1` **Sample Prompt**

^^^

Run the `tiny` BFCL configuration from my Nemotron clone, then show me the benchmark rows it produced and which oracle tier the pack was awarded.

:::
::::

## Start Here

- Run all commands from the repository root so the pack paths in the configuration resolve.
- No model endpoint or API key is needed. BFCL generation renders conversations from the pack's templates; it does not ask a model to write them.
- The configuration reads the pack at `src/nemotron/steps/byob/data/tiny_oracle_pack` and writes outputs under `/tmp/bfcl/tiny_out`.

:::{note}
Paths in a BFCL configuration resolve relative to the BYOB skill root, not your shell's working directory, unless they are absolute. See {doc}`reference/generate-config`.
:::

## Prerequisites

- A host with the `uv` tool available in your shell.
- A local clone of the repository.

## Procedure

1. Clone the repository:

   ```console
   git clone https://github.com/NVIDIA-NeMo/Nemotron && cd Nemotron
   ```

1. From the repository root, add the dependencies for building benchmarks:

   ```console
   uv sync --extra byob
   ```

1. Run every stage against the bundled tiny pack:

   ```console
   uv run nemotron steps run byob/bfcl \
     -c src/nemotron/steps/byob/bfcl/config/tiny.yaml \
     stage=all \
     family=bfcl
   ```

   The command echoes the resolved configuration, validates the pack, then runs generation. On success the last line it prints is the path to the benchmark it wrote:

   ```text
   /tmp/bfcl/tiny_out/bfcl_tiny_library_validation/benchmark.parquet
   ```

1. Confirm the pack passed validation and see the tier it earned:

   ```console
   uv run python -c "
   import json
   report = json.load(open('/tmp/bfcl/tiny_out/bfcl_tiny_library_validation/stage_cache/oracle_validation_report.json'))
   print('tier:', report['tier'])
   print('gold_eligible:', report['gold_eligible'])
   "
   ```

   The tier is awarded by validation, never by configuration. Generation refuses a
   pack that does not qualify for Gold. This bundled pack is Gold, but its configuration
   deliberately uses `lineage.policy: smoke_no_publication`, so the run exercises the
   complete pipeline while its manifest remains ineligible for release.
   {doc}`explanation/oracle-pack` explains the pack tier, and
   {doc}`how-to/publish-a-release` explains publication eligibility.

1. Inspect the benchmark:

   ```console
   uv run python -c "
   import pandas as pd
   df = pd.read_parquet('/tmp/bfcl/tiny_out/bfcl_tiny_library_validation/benchmark.parquet')
   print('rows:', len(df))
   print(df[['task_id', 'category', 'turn_policy', 'num_tool_calls', 'tier']].to_string(index=False))
   "
   ```

   The tiny pack declares four templates in one category and the configuration budgets four tasks per category, so this run publishes four rows. Each row carries the rendered conversation in `messages`, the tool catalog the candidate is allowed to see in `tools`, and the calls the oracle proved correct in `expected_tool_calls`.

1. Read the commit marker:

   ```console
   uv run python -c "
   import json
   m = json.load(open('/tmp/bfcl/tiny_out/bfcl_tiny_library_validation/run_manifest.json'))
   print('run_id:', m['run_id'])
   print('tier:', m['tier'], '| gold_eligible:', m['gold_eligible'])
   print('pack:', m['pack']['pack_id'], m['pack']['version'], m['pack']['content_hash'])
   print('stage counts:', m['stage_counts'])
   "
   ```

   `run_manifest.json` is what makes a run reproducible: it pins the pack's content hash, the resolved configuration hash, the seeds, and how many tasks each stage admitted or dropped. {doc}`reference/output-files` documents every field and every other file the run wrote.

## Understanding the Result

The intermediate tables under `stage_cache/` are worth a look, because they are how you diagnose a pack. Each generation stage writes one Parquet table keyed by `task_id`, so joining two adjacent tables shows exactly which stage dropped a task and therefore which part of the pack to fix.

| File | Written by | Contains |
| --- | --- | --- |
| `task_instances.parquet` | expand | Template expansions with their slot bindings |
| `conversation_plans.parquet` | state_machine | The planned turn sequence per task |
| `rendered_conversations.parquet` | render | The user and assistant turns as text |
| `expected_traces.parquet` | expected_trace | The tool calls each task should produce |
| `schema_validated_traces.parquet` | schema_validation | Traces that match the declared tool schemas |
| `replay_validated_tasks.parquet` | executable_replay | Tasks the oracle reproduced and whose assertions held |

A task that appears in `expected_traces.parquet` but not in `replay_validated_tasks.parquet` is the most informative failure: the pack claimed a behavior that its own backend did not reproduce.

## Next Steps

- Read {doc}`explanation/oracle-pack` before authoring a pack of your own. The file layout and the tier rules are the parts worth understanding first.
- If you are starting with your own tool interface, records, and business behavior,
  follow {doc}`how-to/start-from-domain-data` to choose between the manual and
  model-assisted developer paths.
- Follow {doc}`how-to/author-a-pack` to scaffold and validate your own pack, or {doc}`how-to/assisted-authoring` to draft one from an existing Python package or HTTP service.
- Run a domain-sized generation by copying `src/nemotron/steps/byob/bfcl/config/smoke.example.yaml` and repointing `oracle_pack.manifest_path` at your pack.
- When you have a benchmark you trust, evaluate a candidate model against it with {doc}`how-to/run-evaluation`.
- To localize a completed, verified generation run, follow {doc}`how-to/translate`.
- If a run fails, start at {doc}`reference/troubleshooting`.
