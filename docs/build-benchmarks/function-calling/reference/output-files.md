<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Output Files

Every path on this page is relative to `output_dir/expt_name` from the generation
config. The groups appear in the order a run produces them. For the stages behind each group, see
{doc}`../explanation/pipeline-overview`. To see representative values change across
these files, follow the transfer-fee task in
{doc}`../explanation/pipeline-worked-example`.

## Pack Preparation

`stage=prepare` normalizes the oracle pack into `stage_cache/` so that no later stage
reads pack files directly.

| File | Description |
| --- | --- |
| `stage_cache/pack_manifest.json` | The resolved pack manifest. |
| `stage_cache/pack_paths.json` | Which file supplied each pack artifact. |
| `stage_cache/tools_normalized.json` | Model-facing tool schemas. |
| `stage_cache/tools_normalized_internal.json` | Tool schemas including fields withheld from the model. |
| `stage_cache/fixtures_normalized.json` | Fixture collections with resolved primary keys. |
| `stage_cache/task_templates_normalized.yaml` | Conversation templates. |
| `stage_cache/validation_cases_normalized.yaml` | Pack validation cases. |
| `stage_cache/held_out_normalized.json` | The held-out policy, when the pack declares one. It is removed when no policy is declared. |
| `stage_cache/oracle_validation_report.json` | The Gold gate verdict. `stage=generate` refuses a pack whose report is not `gold_eligible`. |

## Generation Stage Cache

One table per canonical stage, keyed by `task_id`, plus the reports and model I/O caches
each stage produces. The canonical stage order is `reference_profile`, `expand`,
`state_machine`, `render`, `expected_trace`, `schema_validation`, `executable_replay`,
`surface_quality`, `dedup_balancing`, `final_output`. The last two run only when their
config section is enabled.

| File | Written by |
| --- | --- |
| `stage_cache/reference_samples.parquet` | `reference_profile`, alongside `reference_profile.json` and the append-only `reference_profile_io_cache.jsonl`. |
| `stage_cache/task_instances.parquet` | `expand`, alongside `held_out_bindings.json`. |
| `stage_cache/conversation_plans.parquet` | `state_machine`. |
| `stage_cache/rendered_conversations.parquet` | `render`, alongside `paraphrase_rejections.json` and the append-only `paraphrase_io_cache.jsonl`. |
| `stage_cache/expected_traces.parquet` | `expected_trace`. Drop reasons for instances whose bindings failed are recorded here. |
| `stage_cache/schema_validated_traces.parquet` | `schema_validation`. |
| `stage_cache/replay_validated_tasks.parquet` | `executable_replay`. Nondeterministic replays and failed assertions are recorded here. |
| `stage_cache/surface_validated_tasks.parquet` | `surface_quality`, alongside `surface_quality_rejections.json`, `surface_judge_cache_usage.json`, and the append-only `surface_judge_io_cache.jsonl`. |
| `stage_cache/balanced_tasks.parquet` | `dedup_balancing`, alongside `dedup_balancing_report.json`. |
| `stage_cache/held_out_scan.json` | `final_output`, naming any publication candidate that bound a reserved template or fixture row. |

:::{note}
The three `*_io_cache.jsonl` files are append-only and shared across runs. A resumed
stage replays the recorded responses instead of paying for new ones, so a resume renders
the same surfaces as the original run.
:::

## Checkpoint Chain

Each enabled stage writes one verified checkpoint under
`stage_cache/checkpoints/<stage>/`, forming a parent chain that `skip_until` walks
backwards to restore the immediate enabled predecessor.

| File | Description |
| --- | --- |
| `stage_cache/checkpoints/<stage>/manifest.json` | Contract `bfcl-generation-checkpoint/1.0`. Records the stage, its contract version, its declared parent and that parent's checkpoint id, the run identity, the task id list with its set hash, and the inherited and produced artifact metadata. Its own `checkpoint_id` is a hash of the rest, so an edited manifest fails to verify. |
| `stage_cache/checkpoints/<stage>/state.json` | Contract `bfcl-generation-state/1.0`. The canonical stage state, with the task ids and task-set hash restated. |
| `stage_cache/checkpoints/<stage>/artifacts/` | Immutable copies of the artifacts the stage may change, each carrying its content hash, size, row count, and schema fingerprint. |

Restore is fail-closed. A checkpoint whose parent chain, run identity, artifact hashes,
or task set no longer verify is refused rather than partially reused, so do not edit
checkpoint manifests, state, or snapshots.

## Published Benchmark

| File | Description | Content-addressed |
| --- | --- | --- |
| `benchmark_raw.parquet` | Every schema-valid, replay-valid row, before surface-quality drops and before deduplication and balancing. This is the audit table that explains a score. | Yes, as `artifacts.benchmark_raw_parquet.content_hash`. |
| `benchmark.parquet` | The published rows. Same schema as the raw table; the difference is a selection, never a rewrite. | Yes, as `artifacts.benchmark_parquet.content_hash`. |
| `run_manifest.json` | The commit marker. Written last. | It is the record; the artifacts it names carry the hashes. |

The publication contract re-derives the published set from the stage decisions, reads
both files back from disk, and refuses the run unless every published row is
byte-identical to its raw counterpart across every column.

`run_manifest.json` is what makes the tree published, and an evaluation reads it rather
than a bare table. These are the fields that give it that role:

| Field | Meaning |
| --- | --- |
| `run_id` | Identity of this run, combining `expt_name`, the creation timestamp, a prompt-bundle prefix, and a unique suffix, so two runs of one config are distinguishable. |
| `schema_version` | The benchmark row schema the tables were written with. A consumer selects its adapter by this value. |
| `artifacts` | One content hash per stage artifact plus `benchmark_raw_parquet`, `benchmark_parquet`, and, when exports ran, `export_validation_report`. |
| `publication` | Both row counts, both table content hashes, which surface gate decided, and which ordering applies. |
| `pack` | `pack_id`, `version`, `fingerprint_contract`, a `content_hash` fingerprint of the whole pack tree, and a per-file hash map, so a later run can name which file moved. Contract `bfcl-pack-fingerprint-v2` length-prefixes logical names and hashes one fixed-width SHA-256 digest per file. |
| `oracle` | `kind` (`python` or `endpoint`) and, for an endpoint pack, the verified endpoint metadata. |
| `oracle_clock`, `seeds` | The frozen clock and the seed derivation, which together make the bindings reproducible. |
| `tier`, `gold_eligible`, `gold_ineligibility_reasons` | The publication verdict and, when it is negative, why. |
| `lineage_policy`, `models`, `judge_advisory`, `profile_influenced_surface` | Which model roles read published rows, which is the inventory the contamination gate compares candidates against. |
| `generation_config_hash`, `resolved_config_hash` | Config identity, computed with the eval reference keys excluded so an eval edit cannot move a benchmark's identity. |
| `stage_counts` | The full generation funnel, from expansion to published rows. |
| `exports` | Enabled and disabled formats, their schema versions, row counts, paths, and tree hashes. |

:::{important}
If `run_manifest.json` is absent, any benchmark table or export beside it is not
published, even though the files exist. Rerun `stage=generate`; startup removes
abandoned staging trees and the final stage replaces all payloads before writing the
manifest last.
:::

## Compatibility Exports

Written only when the matching `exports` flag is on. Both formats derive from one
canonical projection of `benchmark.parquet` and are read back for equivalence before the
manifest is written.

| Path | Description |
| --- | --- |
| `exports/bfcl_json/BFCL_v4_multi_turn.jsonl` | The question records in upstream BFCL layout. |
| `exports/bfcl_json/possible_answer/BFCL_v4_multi_turn.jsonl` | The matching answer records. |
| `exports/nemo_evaluator_bundle/bundle.json` | The native adapter descriptor. |
| `exports/nemo_evaluator_bundle/dataset.jsonl` | One record per published task. |
| `exports/nemo_evaluator_bundle/dataset.schema.json` | The schema `dataset.jsonl` conforms to. |
| `exports/nemo_evaluator_bundle/metadata.json` | Provenance of the bundle. |
| `exports/nemo_evaluator_bundle/evaluator.yaml` | The native adapter contract. It is not a standalone NeMo Evaluator Launcher run config. |
| `exports/nemo_evaluator_bundle/system_prompts.json` | The distinct system prompts the dataset references. |
| `exports/export_validation_report.json` | Read-back evidence that every enabled export matches `benchmark.parquet` on row count, order, truth fields, and hashes. |

Each export is content-addressed as a tree hash over its complete file set, recorded in
the manifest's `exports` section. The bundle is verified by exact file set, so one extra
file in that directory fails the next verification. Never repair one file in place.

## Related Pages

- Generation YAML fields: {doc}`generate-config`
- Stage order: {doc}`../explanation/pipeline-overview`
- Troubleshooting: {doc}`troubleshooting`
