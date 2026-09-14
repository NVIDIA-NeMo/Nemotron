<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Generation Configuration Reference

This page lists the YAML keys accepted by the BFCL generation config, as validated by
`BfclConfig.from_yaml` in
`src/nemotron/steps/byob/runtime/benchmark_families/bfcl/config.py`. The bundled
configurations named below all live in `src/nemotron/steps/byob/bfcl/config/`. Start from
`default.yaml`, which carries the same keys with inline commentary, or from `tiny.yaml`,
`smoke.example.yaml`, `publication.example.yaml`, or `publication.paraphrase.example.yaml`
for worked examples at different scales.
For how the fields fit together, see {doc}`../explanation/pipeline-overview`.

:::{important}
Every section is closed, and the pipeline refuses a config it will not honor rather
than ignoring the key. An unknown key inside any block is an error naming the block
and the key. Separately, a correctly spelled key that no enabled stage would apply is
also refused: enabling `semantic_deduplication_config` is what makes the balancing
targets in `task_generation` legal, and `input_dir`, `generation_model_config`,
`judge_model_config`, `translation_config_path`, `eval_config_path`, and an inline
`eval` block are all refused by `stage=generate`. See
{doc}`troubleshooting` for the refusal messages.
:::

## Path Resolution

Relative paths in this file resolve against the BYOB root, `src/nemotron/steps/byob`, not
against the config file's own directory. Absolute paths are used as written, and every
resolved path is normalized before validation. Two placement rules apply:
`output_dir/expt_name` must resolve outside the oracle pack root, so generated artifacts
cannot change the pack fingerprint, and any pack or reference file must sit under an
`oracle_runtime.allowed_roots` entry.

## Top-Level Identity

| Field | Type | Default | Controls |
| --- | --- | --- | --- |
| `family` | string | none, required | Must be `bfcl`. The shared dispatcher otherwise selects the MCQ family. |
| `expt_name` | string | none, required | The run directory name under `output_dir`. Must be a single path component, so no separators, no whitespace padding, and not `.` or `..`. |
| `output_dir` | path string | none, required | The parent directory of the run tree. |
| `stage` | string | `all` | Which generation action to run: `prepare`, `generate`, or `all`. |
| `random_seed` | integer | `null` | Seeds every deterministic binding. Recorded in the manifest as `seeds.global`, where an absent seed is stored as `0`. |
| `ndd_batch_size` | integer ≥ 1 | `32` | Batch size for model-facing generation calls. |
| `schema_version` | string | `null` | The benchmark row schema the run promises to write. This build writes only `"1.1"`; any other value is refused. |
| `config_status` | `template` or `resolved` | `null` | Declares whether the file is finished. `resolved` additionally requires that no value anywhere in the file still contains a `REPLACE_ME_` placeholder. |

`oracle_pack`, `oracle_runtime`, and `lineage` are the three required mappings. Every
other section may be omitted, in which case it is treated as empty.

## `oracle_pack`

Paths into the executable oracle pack. Only `manifest_path` is required; the pack's own
manifest names the rest, and these keys exist to override individual files.

| Field | Type | Default | Controls |
| --- | --- | --- | --- |
| `manifest_path` | path string | none, required | The pack's `manifest.yaml`. Its parent directory is the pack root. |
| `backend_path` | path string | `null` | The Python backend module for a `python`-transport pack. |
| `endpoint_config_path` | path string | `null` | The immutable endpoint configuration for an `endpoint`-transport pack. |
| `fixtures_path`, `task_templates_path`, `assertions_path` | path string | `null` | Fixture collections, conversation templates, and success assertions. |
| `validation_cases_path` | path string | `null` | Pack validation cases used by `stage=prepare`. |

## `oracle_runtime`

Process-worker bounds, the frozen clock, and the pack trust roots.

| Field | Type | Default | Controls |
| --- | --- | --- | --- |
| `clock` | quoted ISO-8601 string | none, required | The frozen clock every oracle session receives. It must parse as ISO-8601 and must carry a UTC offset, so a run stays reproducible. Quote it, or YAML delivers a `datetime` and the loader refuses it. |
| `tool_timeout_s` | number > 0 | `5.0` | Per tool call. |
| `assertion_timeout_s` | number > 0 | `5.0` | Per assertion evaluation. |
| `import_timeout_s` | number > 0 | `10.0` | Importing pack code inside the worker. |
| `reset_timeout_s` | number > 0 | `5.0` | Establishing clean state for a task. |
| `episode_timeout_s` | number > 0 | `60.0` | A whole replay episode. |
| `worker` | `process` or `thread` | `process` | Isolation for pack code. Gold eligibility requires `process`; `thread` exists for debugging. |
| `allowed_roots` | list of path strings | `[<byob>/data]` | Trust roots for pack and reference files. An absent or empty list resolves to the checked-in BYOB `data` directory. |

## `lineage`

| Field | Type | Default | Controls |
| --- | --- | --- | --- |
| `policy` | `strict_separation` or `smoke_no_publication` | none, required | `smoke_no_publication` marks the run permanently ineligible for publication. `strict_separation` additionally requires enabled roles to use pairwise-distinct canonical identities. |
| `profile_influenced_surface` | bool | `false` | Declares that an enabled profile role shaped the published surface text. It may be true only when the `profile` role is enabled and model paraphrasing is on. |
| `judge_advisory` | bool or `null` | `null` | Must be `null` while the surface judge is disabled, and otherwise exactly the inverse of `surface_quality_validation.drop_authority`. |
| `roles` | mapping | `{}` | Accepts only `profile`, `paraphrase`, and `surface_judge`. |

Each role accepts exactly `enabled` (bool, default `false`) and `model_config`
(mapping or `null`). Pinning a model identity means the `model_config` of an enabled
role carries non-empty strings for all four of `alias`, `provider`, `model`, and
`canonical_id`; a role that is enabled without them is refused. Credentials are refused
anywhere inside `model_config`: any key named like `api_key`, `token`, `password`,
`secret`, or `authorization`, or ending in one of those suffixes, is rejected before the
config is hashed or logged. Name the environment variable instead. Enabling the
`profile` role also requires a `reference_benchmark` block.

`reference_benchmark` accepts `name`, `samples_path`, and `content_hash`, and requires
all three as non-empty strings when it is present. The samples file must sit under an
allowed root, must exist, and its bytes must hash to the declared
`sha256:<64 lowercase hex>` value. Each line is validated as a style-only sample:
a unique `sample_id`, a `language`, a non-empty `messages` list, and no oracle-truth
fields such as `tools`, `tool_calls`, `expected_tool_calls`, or `oracle_state`.

## `surface_generation`

| Field | Type | Default | Controls |
| --- | --- | --- | --- |
| `language` | non-empty string | pack's `default_language`, then its single available language | The language the pack's templates are rendered in. A language the pack does not offer is refused. |
| `model_paraphrase_enabled` | bool | `false` | Must equal `lineage.roles.paraphrase.enabled`. |
| `paraphrases_per_template` | integer ≥ 0 | `0` | Guarded model variants per eligible binding. Must be positive when paraphrasing is enabled and zero when it is not, and may not exceed the number of declared style axes. |
| `preserve_slot_values`, `prevent_tool_name_leakage` | bool | type-checked only | Declare the `must_preserve` slot-value guard and that tool names are withheld from the paraphrase model. |
| `surface_style_axes` | non-empty list of unique strings | the 20 framework axes | Replaces the framework register list with axes suited to the domain or language. Each axis must select phrasing only. |

## `surface_quality_validation`

| Field | Type | Default | Controls |
| --- | --- | --- | --- |
| `contract_version` | string | `"1.1"` | Must equal the contract version this build implements. |
| `enabled` | bool | `false` | Enables the `surface_quality` stage. Required when the `surface_judge` role is enabled, and required when semantic deduplication is enabled. |
| `drop_authority` | bool | `false` | Grants the surface judge authority to drop rows. Requires an enabled `surface_judge` role. |

## `task_generation`

Budgets are per category, and a category budget is shared by every template in that
category, so it may not fall below the template count of the widest category.

| Field | Type | Default | Controls |
| --- | --- | --- | --- |
| `tasks_per_category` | integer ≥ 1 | `1` | The maximum selected rows per category. |
| `candidate_tasks_per_category` | integer ≥ 1 | `tasks_per_category` | The over-generated candidate pool per category. Must be greater than or equal to `tasks_per_category`. |
| `target_published_tasks` | integer ≥ 1 | none | The total row count the run promises. Declaring it lets balancing abort instead of publishing short. |
| `max_turns` | integer ≥ 1 | none | Turn ceiling for selection. Honored only when semantic deduplication is enabled. |
| `max_tool_calls` | integer ≥ 1 | none | Tool-call ceiling for selection. Honored only when semantic deduplication is enabled. |
| `max_intent_share` | number in (0, 1] | none | The within-category intent target. Honored only when semantic deduplication is enabled. |

The four mixes below are probability mappings: every value must be a number between 0 and
1, and the values must sum to 1. Each is honored only when deduplication is enabled.

| Mix | Accepted keys |
| --- | --- |
| `difficulty_mix` | any non-empty string keys |
| `turn_mix` | `single_turn`, `multi_turn` |
| `tool_call_count_mix` | `"1"`, `"2"`, `"3+"` (quote them, or YAML makes them integers) |
| `policy_mix` | `single_turn`, `missing_slot`, `confirmation`, `correction`, `multi_tool`, `dependent_call`, `negative_path`, `clarify_only`, `irrelevant` |

## `semantic_deduplication_config`

| Field | Type | Default | Controls |
| --- | --- | --- | --- |
| `contract_version` | string | `"1.0"` | Must equal the contract version this build implements. |
| `enabled` | bool | `false` | Enables the `dedup_balancing` stage. Requires `surface_quality_validation.enabled`, and makes `model_identifier`, `n_clusters`, `eps`, and `remove_duplicates` required. |
| `model_identifier` | non-empty string | none | The embedding model used for clustering. |
| `n_clusters` | integer ≥ 1 | none | Cluster count. |
| `eps` | number in (0, 1) | none | Similarity threshold. |
| `remove_duplicates` | bool | none | Whether semantic duplicates are dropped or only recorded. |
| `max_exact_surface_reuse` | integer ≥ 1 | none | How often one exact surface string may repeat. |
| `min_exact_surface_ratio` | number in (0, 1] | none | Floor on the share of rows keeping exact template wording. |
| `max_rows_per_intent`, `max_execution_case_reuse` | integer ≥ 1 | none | Hard caps on rows owned by a single intent, and on repeats of one executable case. |
| `representative_source_preference` | list of `template` and/or `model` | none | Cluster representative preference order. Sources may not repeat. |
| `unmet_target_policy` | `abort` or `publish_non_gold` | `abort` | What happens when a declared balancing target cannot be met. |

## `exports`

| Field | Type | Default | Controls |
| --- | --- | --- | --- |
| `bfcl_json` | bool | `false` | Writes the BFCL question and answer JSONL pair. |
| `nemo_evaluator_bundle` | bool | `false` | Writes the six-file NeMo Evaluator input bundle. |

Both writers derive from one canonical projection of the published table, are read back
for equivalence, and are recorded in the run manifest. An export name that no writer
implements is refused.

## Related Pages

- Evaluation YAML fields: {doc}`eval-config`
- Artifacts a run writes: {doc}`output-files`
- Refusal messages and their fixes: {doc}`troubleshooting`
- Authoring a pack: {doc}`../how-to/author-a-pack`, {doc}`../explanation/oracle-pack`
- Publishing a release: {doc}`../how-to/publish-a-release`
