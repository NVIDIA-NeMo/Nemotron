<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Publish a Benchmark Release

Use this guide to take a Gold-eligible Oracle Pack to a publication-scale benchmark:
choose the size target, declare the challenge mix, run the full pipeline, and produce
the immutable publication consumed by evaluation.

The handoff between generation and evaluation is:

```text
Gold-eligible Oracle Pack
  → publication configuration
  → benchmark.parquet + benchmark_raw.parquet + run_manifest.json
  → run-evaluation.md
  → eval_report.json + eval_task_results.parquet + eval_manifest.json
```

## Before You Start

- A pack that validates as Gold-eligible, and a completed smoke run against it. Generation refuses a pack that is not Gold-eligible; see {doc}`author-a-pack`.
- A fresh `output_dir` and a unique `expt_name`, which is a single directory name rather than a path. `output_dir/expt_name` must sit outside the pack root.
- A decision about whether the release needs model-authored wording. See [Choose a Profile](#choose-a-profile).

:::{warning}
Do not edit the pack between validation and publication. The pack fingerprint is checked before validation, after validation, and again before final output, and any drift aborts publication rather than stamping a report and a benchmark that came from different bytes.
:::

(choose-a-profile)=
## Step 1: Copy a Publication Profile

Two bundled examples run as written against the bundled reference pack. Copy the one that matches your intent, then repoint it at your own pack.

| Profile | What it does |
| --- | --- |
| `publication.example.yaml` | A template-only Gold run. Every published surface is rendered from the pack's own templates. |
| `publication.paraphrase.example.yaml` | The same run with one difference: a model rewords the prompts under deterministic guards, preserving the same executable cases. |

The paraphrase profile is the only one that calls a model during generation, and those
calls are routed by NeMo Data Designer. Choosing it therefore adds a dependency: the
provider named by the enabled role must exist in your Data Designer installation, and
the model identity must be pinned. Template-only generation has no such prerequisite.
See {doc}`../reference/data-designer-provider` for the provider registry, endpoint,
model identity, and credential environment-variable setup shared by paraphrasing and
assisted authoring.

```bash
mkdir -p /srv/bfcl/runs && \
  cp src/nemotron/steps/byob/bfcl/config/publication.example.yaml \
    /srv/bfcl/runs/warehouse-gold.yaml
```

Change `expt_name`, `output_dir`, `oracle_pack.manifest_path`, `oracle_runtime.allowed_roots`, and `surface_generation.language` to match your own pack. Template-only generation is fully Gold-eligible, so reach for the paraphrase profile only when you need more wording inventory than your templates provide.

:::{note}
Every number in the bundled profiles is a worked example for the bundled pack's inventory, not a framework default. Copying another pack's task counts and diversity limits is the most common way to make the balancing stage infeasible.
:::

## Step 2: Choose the Size Target and the Category Budget

The size target drives everything else: divide it by the number of categories your pack declares, and the mixes further down stay meaningful.

```yaml
task_generation:
  candidate_tasks_per_category: 480  # expansion ceiling, >= tasks_per_category
  tasks_per_category: 232            # size target divided by your category count
  target_published_tasks: 1392       # tasks_per_category x your category count
  max_turns: 5
  max_tool_calls: 3
```

Those numbers are one worked example, not defaults to copy: they read `232 x 6` for a pack
that declares six categories. Recompute both against your own category count, because
`target_published_tasks` is an exact count and a value that does not match your inventory
aborts the run rather than publishing short.

`tasks_per_category` is the default expansion budget for a category and the publication cap over unique bindings, and it may not fall below the template count of your widest category. `candidate_tasks_per_category` is an optional, larger expansion ceiling so that balancing has inventory to select from; it defaults to, and cannot be smaller than, `tasks_per_category`. `target_published_tasks` is the optional exact run-wide publication count, and declaring it is what lets the pipeline abort instead of quietly publishing short. `max_turns` and `max_tool_calls` are publication hard limits, and a task that exceeds either is dropped with its own reason.

`tasks_per_category` is a maximum, not a replication instruction. Every published row still carries a unique deterministic binding: over-generating candidates gives the balancing stage a choice set, and it never authorizes copying a row or treating a paraphrase as new task semantics. Over-generation also helps only when deduplication and balancing are enabled and the pack holds enough distinct bindings in the buckets you asked for. It does not guarantee the ceiling will be reached, because deduplication, the hard limits, coverage locks, a held-out policy, and incompatible cross-dimensional targets can all reduce the feasible set.

## Step 3: Declare the Challenge Mix

The mixes are normalized balancing targets over generic task dimensions. Fractional targets are allocated by deterministic largest remainder.

```yaml
task_generation:
  difficulty_mix: {easy: 0.25, medium: 0.30, hard: 0.45}
  turn_mix: {single_turn: 0.70, multi_turn: 0.30}
  tool_call_count_mix: {"1": 0.60, "2": 0.30, "3+": 0.10}
  max_intent_share: 0.50
  policy_mix:
    {single_turn: 0.19, missing_slot: 0.10, dependent_call: 0.15,
     multi_tool: 0.07, confirmation: 0.07, correction: 0.08,
     negative_path: 0.07, clarify_only: 0.1033, irrelevant: 0.1667}
```

`policy_mix` keys are `turn_policy` values, which makes it the knob that states how much of the release must exercise clarification, correction, confirmation, and documented-failure behavior instead of plain lookups. Conversation policy is the axis a candidate is most likely to fail on, so a release is better off declaring it than letting inventory decide.

Keep the dimensions conceptually separate. `turn_class` is derived from the number of rendered user turns and `tool_call_count` from the executable plan, so a dependent two-call chain remains a single-turn task. Weight `tool_call_count_mix` toward multi-call paths only when the domain has real tool chains; a catalog of independent lookups is better served by a flatter mix.

## Step 4: Set the Deduplication and Balancing Targets

Enabling this stage requires the surface-quality stage, so deduplication never admits an unvalidated run.

```yaml
surface_quality_validation: {contract_version: "1.1", enabled: true, drop_authority: false}

semantic_deduplication_config:
  contract_version: "1.0"
  enabled: true
  model_identifier: sentence-transformers/all-MiniLM-L6-v2
  n_clusters: 64
  eps: 0.08
  remove_duplicates: false
  max_execution_case_reuse: 1
  max_rows_per_intent: 120
  representative_source_preference: [template, model]
  unmet_target_policy: abort
```

The optional repetition caps are one mechanism applied to different projections of a row, and each has its own shortfall reason so a report can say which kind of repetition ran out. `max_execution_case_reuse: 1` is the strongest statement a release can make: no two published rows call the same tools with the same arguments against the same state, so a candidate cannot earn credit twice for one behavior. `max_rows_per_intent` stops one broad intent, typically a refusal or out-of-scope intent with cheap inventory, from owning a disproportionate share of the benchmark. `max_exact_surface_reuse` and `min_exact_surface_ratio` bound and floor exact wording diversity, which matters for a paraphrase profile whose whole point is more distinct wording.

`remove_duplicates: false` keeps distinct fixture-bound evaluation cases even when their masked surface forms cluster together; similarity remains recorded evidence, and the hard limits and quotas still apply with their own drop reasons.

:::{important}
Keep `unmet_target_policy: abort`, which is the default. Because the diversity constraints count masked surfaces, wording inventory rather than binding count is what limits how many rows a pack can publish: one template renders one canonical wording per language, so a thousand fixture bindings of it still contribute a single masked surface. When a declared target cannot be met, `abort` leaves the diagnostic artifacts and stops before publication. The alternative, `publish_non_gold`, publishes only after setting both the manifest and every row's `gold_eligible` to false and recording the unmet targets as the reason. A publication run should not silently publish short, because the manifest would then report a balance the release does not have.
:::

For a paraphrase profile, the surface stage assigns each binding one structural style axis from the framework catalog, chosen from the task seed, so repeated bindings are asked for different sentence forms rather than the same rewrite. `surface_generation.paraphrases_per_template` may not exceed the axis count, and a pack whose domain or language needs different registers declares its own list in `surface_generation.surface_style_axes`.

## Step 5: Decide on Exports

Both export flags, `exports.bfcl_json` and `exports.nemo_evaluator_bundle`, default to `false`. Enabling either makes export read-back validation part of the publication transaction, so the flag is never silently ignored: the writer output is read back from disk and checked for tree hash, row count, task order, canonical truth fields, and format envelopes against the single published projection, and any mismatch aborts publication.

Direct BFCL evaluation does not need a compatibility export: it reads the published
tables through `run_manifest.json`. If the release will be submitted through the NeMo
Evaluator Launcher native adapter, enable its bundle before publication:

```yaml
exports:
  bfcl_json: false
  nemo_evaluator_bundle: true
```

A successful publication then contains:

```text
<publication>/
├── benchmark.parquet
├── benchmark_raw.parquet
├── run_manifest.json
└── exports/
    ├── export_validation_report.json
    └── nemo_evaluator_bundle/
        ├── bundle.json
        ├── dataset.jsonl
        ├── dataset.schema.json
        ├── evaluator.yaml
        ├── metadata.json
        └── system_prompts.json
```

The commit marker is `run_manifest.json`; BFCL does not produce a
`manifest.parquet`. Evaluation imports the publication through that JSON manifest,
not by loading `benchmark.parquet` as an unverified standalone table.

The bundle is verified adapter input, not a standalone Launcher configuration. Keep
it beside the publication it came from and follow {doc}`run-evaluation` to create the
Launcher envelope. Do not add files inside the bundle: its exact file set is hashed.

## Step 6: Run the Full Pipeline

```bash
uv run nemotron steps run byob/bfcl \
  -c /srv/bfcl/runs/warehouse-gold.yaml \
  stage=all \
  family=bfcl
```

`stage=all` runs prepare followed by generate; it does not translate or evaluate, because those are separate post-publication runs.

If a stage fails, preserve the experiment directory and resume with `skip_until=<stage>` only when the predecessor checkpoint is intact and the pack, configuration, and pipeline identities have not changed. Resume recursively verifies that chain and fails closed on any drift, and restoration keeps the append-only model input/output caches so a re-run stage replays recorded responses instead of paying for new ones that would render different surfaces. Never patch a generated Parquet file, export, manifest, or cache record.

## Step 7: Verify the Publication

`run_manifest.json` is moved last as the commit marker. If it is absent, any adjacent Parquet or export is unpublished, whatever the file names suggest.

```bash
PUB=/srv/bfcl/runs/warehouse-gold-output/bfcl_warehouse_gold
test -f "$PUB/run_manifest.json" && test -f "$PUB/benchmark.parquet"
test -f "$PUB/benchmark_raw.parquet"
test -f "$PUB/exports/export_validation_report.json"   # exports enabled only
```

Then read these fields from the manifest: `tier` and `gold_eligible` to confirm the release claims the tier you expect and is publishable; `publication` for both row counts, both content hashes, which surface gate decided, and which ordering applies; `stage_counts` for the full generation funnel, which is where a surprising row count is explained; `semantic_deduplication.report.actual_counts` for the realized category, difficulty, turn, and policy mixes to compare against what you declared; and `models` for every model role that read a published row, because the evaluation contamination gate reads that block and refuses a publication that has a gap in it.

`benchmark_raw.parquet` holds every schema-valid, replay-valid row, and `benchmark.parquet` holds the published selection. Both carry the same schema, and the difference between them is a selection and never a rewrite: the published rows are byte-identical to their raw counterparts across every column. `held_out_hit` is `false` on every published row once a held-out policy has been evaluated, and null when no policy was declared.

When exports are enabled, `exports/bfcl_json/` holds the question and answer JSONL pair, and `exports/nemo_evaluator_bundle/` holds the six-file native adapter input bundle. The bundle is adapter input, not a standalone NeMo Evaluator Launcher run configuration; it declares that an adapter must supply a registered environment, a candidate endpoint, and a tool resource service.

This verified publication directory is the evaluation input. Keep it unchanged and
set the following field in the separate evaluation configuration:

```yaml
source_run_manifest: /srv/bfcl/runs/warehouse-gold-output/bfcl_warehouse_gold/run_manifest.json
```

For direct trace evaluation, the three top-level publication files are the complete
benchmark handoff. Executable mode additionally requires the exact Oracle Pack. For
NeMo Evaluator Launcher, the publication must also contain the
`exports/nemo_evaluator_bundle/` tree shown in Step 5. Continue with
{doc}`run-evaluation` for import checks, candidate configuration, execution, result
inspection, and result export.

## Step 8: Audit the Release

The bias audit is a read-only post-release check. It recomputes one primary metric per audit dimension from frozen evidence, fails closed on missing applicable evidence or hash drift, and writes a content-addressed JSON report plus a deterministic Markdown rendering, without modifying any source artifact.

```bash
uv run python -m nemotron.steps.byob.scripts.audit_bfcl_bias \
  --run-manifest "$PUB/run_manifest.json" \
  --output-dir /srv/bfcl/audits/warehouse-gold \
  --raw "$PUB/benchmark_raw.parquet" \
  --expanded "$PUB/stage_cache" \
  --pack-manifest /srv/bfcl/packs/warehouse_assets/manifest.yaml
```

`--run-manifest` and `--output-dir` are required; the remaining inputs are supplied when the evidence applies. `--contamination-report` may be repeated once per evaluation run, and `--published`, `--distractor-evidence`, `--judge-evidence`, `--portability-evidence`, and `--exceptions` cover the remaining evidence classes.

## Step 9: Bundle the Release for Handoff

To hand the release to someone else, or to keep it as evidence, bundle it rather than copying directories. The bundle is deterministic and content-addressed, so the recipient can verify they have the same bytes you produced.

```bash
uv run python -m nemotron.steps.byob.scripts.archive_bfcl_release \
  --release-dir "$PUB" \
  --output-dir /srv/bfcl/bundles \
  --bundle-name warehouse-gold-v1
```

`--release-dir` and `--output-dir` are required. Add `--evaluation-dir` to include an evaluation run's artifacts and `--evidence-dir` to include audit or review evidence, so that the score and the artifact it describes travel together.

## Common Failures

| Symptom | What it means |
| --- | --- |
| The run stopped with unmet balancing targets | The declared mix was not reachable. Read `stage_cache/dedup_balancing_report.json`, which names the bound that caused it: candidate inventory, a category cap, surface diversity, a declared mix, or coverage. |
| A held-out row was bound or reached publication | Read `stage_cache/held_out_scan.json` for the task identifiers, then fix the pack sources or the held-out policy and re-run generation. Enforcement is abort-only, because the publication set is already fixed and dropping a row would break the balance the manifest reports. |
| An export failed read-back equivalence | Inspect `exports/export_validation_report.json` when present, discard every final payload, and re-run. Never repair one JSONL or bundle file in place, because the manifest pins the whole tree. |
| Every replay survivor was dropped | Final output refuses to stamp an empty benchmark as Gold. Loosen the surface policy or fix the templates failing their guards. |

## Next Steps

- Score a model against the release: {doc}`run-evaluation`.
- Localize approved model-facing text: {doc}`translate`.
- Look up any field you changed in {doc}`../reference/generate-config`, or any artifact it wrote in {doc}`../reference/output-files`.
