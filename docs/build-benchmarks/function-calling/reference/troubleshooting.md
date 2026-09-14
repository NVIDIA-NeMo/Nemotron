<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Troubleshooting

This page indexes the registered failures of `nemotron steps run byob/bfcl` by the phase
that raises them. Each row names the code the step reports, what usually causes it, and
the fix the code itself suggests. The codes come from the error taxonomy declared in
`src/nemotron/steps/byob/bfcl/step.toml` and, for evaluation, from
`src/nemotron/steps/byob/runtime/benchmark_families/bfcl/eval/error_taxonomy.py`.

:::{note}
The pipeline refuses rather than degrades. Most rows below describe a config or a pack
that asked for something no stage would honor, or evidence that no longer identifies one
run. In both cases the fix is to correct the input or start from a clean tree, not to
retry the same command.
:::

## Configuration

| Code | Likely cause | Fix |
| --- | --- | --- |
| `missing_family_pin` | The config omits `family`. | Set `family: bfcl`. Without it the shared dispatcher defaults to the MCQ family. |
| `unsupported_config_feature` | The config asks for work no stage performs: an unknown export name, balancing controls with deduplication disabled, eval or translation orchestration keys, leftover shared model fields, or an unrecognized `surface_generation` key. | Disable or remove the listed setting, or register its owning stage. See {doc}`generate-config`. |
| `pack_outside_allowlist` | The oracle pack sits outside every trust root. | Move the pack under an `oracle_runtime.allowed_roots` entry, or extend `allowed_roots` explicitly. |
| `model_lineage_invalid` | An enabled `profile`, `paraphrase`, or `surface_judge` role has no canonical identity, or two enabled roles collide under `strict_separation`. | Give every enabled role a non-secret `canonical_id`, keep enabled identities pairwise distinct, and supply credentials through the provider environment. |
| `reference_benchmark_invalid` | The reference JSONL is outside the allowlist, its hash does not match, or a sample carries oracle truth. | Place it under an allowed root, pin its exact `sha256` `content_hash`, and keep the samples style-only. |
| `category_budget_too_small` | `task_generation.tasks_per_category` is below the template count of a category. | Raise it to at least the number of templates in the largest category, so no template loses its instances. |
| `stage_resume_invalid` | `skip_until` names an unknown or disabled stage, or the checkpoint chain was edited. | Use one enabled canonical stage name, and restore the untouched parent chain or run a full generation without `skip_until`. Do not edit checkpoint manifests, state, or snapshots. |
| `byob_stage_unsupported` | The requested stage is not implemented by this family. | BFCL supports `prepare`, `generate`, `translate`, `eval`, and `all`. Note that `translate` and `eval` do not accept generation resume controls. |
| `bfcl_translation_invalid` | The translate config names a bare table, reuses an output directory, or enables quality filtering. | Start from `bfcl/config/translate.yaml`, set `config_status: resolved`, name the source release's `run_manifest.json`, use a distinct empty output directory, and leave `remove_low_quality` off, because task identity and publication order cannot change. |

## Pack Validation and the Gold Gate

| Code | Likely cause | Fix |
| --- | --- | --- |
| `non_gold_pack` | `stage=generate` was given a pack whose `oracle_validation_report.json` is not `gold_eligible`. | Fix the reported check failures, keep `oracle_runtime.worker: process`, then re-run `stage=prepare`. |
| `endpoint_contract_invalid` | The endpoint does not satisfy the BFCL Oracle HTTP v1 contract. | Serve HTTPS routes for metadata, tools, isolated sessions, calls, state, and session deletion. Store only environment variable names in the endpoint config, and pin the expected oracle id, version, and content digest. |
| `endpoint_identity_changed` | The endpoint metadata no longer matches its config, or changed mid-run. | Deploy or select the intended immutable oracle revision, update the expected digest, and re-run `stage=prepare`. |
| `held_out_contract_invalid` | The held-out policy names missing ids, overlaps `absent_ids`, or uses wrong types. | Reference `held_out.yaml` from the pack manifest, list existing fixture primary ids and template ids, keep them disjoint from `absent_ids`, and use a boolean `fixtures_in_backend_state` with an integer `seed`. |
| `template_without_success_assertion` | A template declares no `success_assertions`. | Add an assertion describing what success means for that shape. A decline template can assert that no tool was called. |
| `slot_missing_visibility_flag` | A template slot omits `visible_in_first_turn`. | Declare the flag on every slot. It decides whether the value must appear in the opening request or must stay out of it, so a slot without it is guarded by neither rule. |
| `unknown_turn_policy` | `turn_policy` is misspelled. | Use one of `single_turn`, `missing_slot`, `confirmation`, `correction`, `multi_tool`, `dependent_call`, `negative_path`, `clarify_only`, or `irrelevant`. A typo would silently skip that policy's gates. |
| `slot_correction_declaration_invalid` | A `slot_updates` entry is not a well-formed correction. | Give it `turn_policy: correction`, a slot the user already stated, a source of the same kind as the original, and a replacement value that differs from it. |
| `primary_key_ambiguous` | A fixture collection carries both its own id and a foreign key. | Declare `manifest.primary_keys.<collection>`. The naming convention only covers `<singular>_id` and `id`, and guessing would attribute a task to a record it merely references. |
| `ask_for_slot_without_a_named_slot` | An assistant question uses `{slot_name}` while the template withholds several slots. | Add `slot: <name>` to the milestone so the question names the slot it asks about. |
| `missing_assistant_turn_templates` | A non-tool milestone has no phrasing. | Declare `assistant_turn_templates` for the milestones the templates use — `ask_for_slot`, `ask_confirm`, `decline`, `final_answer` — on the pack manifest or the template. |

## Generation Stages

| Code | Likely cause | Fix |
| --- | --- | --- |
| `held_out_binding_starved` | The held-out reservations and the category budget cannot both be met: a slot had every matching row reserved, a category ran short, or every template was reserved. | Add fixture rows or templates, lower the budget, or release the reserved ids. |
| `held_out_leak_detected` | A publication candidate bound a reserved template or fixture row, so the run aborted before writing the published table. | Read `stage_cache/held_out_scan.json` for the offending task ids, fix the pack sources or the held-out policy, and re-run `stage=generate`. |
| `superseded_slot_value_in_trace` | A call read a corrected slot while it still held the replaced value. | Move the correction turn before the call that reads the slot, and re-confirm after it when the tool requires confirmation. |
| `confirmed_mutation_without_user_confirmation` | A call carrying `confirm: true` sits in an unconfirmed assistant turn. | Put an `ask_confirm` milestone and its user reply before the call, and re-confirm after any correction turn. |
| `dependent_call_binding_failed` | A `from_result` argument does not resolve. | Name an earlier `tool_call` id in a strictly lower `call_group`, with a path resolving to a scalar in that call's result. A path that misses drops only that instance, with its reason in `stage_cache/expected_traces.parquet`; narrow the template's slot filter if too many drop. |
| `no_replay_validated_rows` | Every task was dropped before export. | Read `stage_cache/replay_validated_tasks.parquet` for nondeterministic replays and assertion failures, and `stage_cache/rendered_conversations.parquet` for guard violations. |

## Publication and Balancing

| Code | Likely cause | Fix |
| --- | --- | --- |
| `export_schema_mismatch` | The published table or an export record does not match the pinned publication schema. | Do not edit generated files. Remove `benchmark_raw.parquet`, `benchmark.parquet`, `exports/`, and `run_manifest.json`, then rerun `stage=generate` on one code revision. Consumers must select adapters by the manifest's `schema_version`. |
| `unsupported_export_call_layout` | A canonical row cannot be represented by the selected compatibility format, for example a function or argument name that is not a Python identifier, or an ambiguous evaluator record layout. | Read the task id in the exception, fix the pack tool schema or the conversation template, and regenerate. The final stage leaves no manifest behind. |
| `export_hash_or_equivalence_mismatch` | An export changed after encoding, or failed read-back equivalence against the published table. | Inspect `exports/export_validation_report.json` when present, discard all final payloads, and rerun. Never repair one export file in place; the manifest pins the whole tree. |
| `export_publication_interrupted` | A run died before the commit marker. | Rerun `stage=generate`. If `run_manifest.json` is absent, the parquet and exports beside it are not published even though the files exist. |
| `nemo_evaluator_adapter_required` | The bundle directory was handed straight to the Launcher. | `evaluator.yaml` inside the bundle is a native adapter contract. Native function calling needs an installed harness with task registration and a tool resource service. |

Balancing that cannot meet a declared target is governed by
`semantic_deduplication_config.unmet_target_policy`: `abort` refuses the run, while
`publish_non_gold` publishes and records the run as ineligible. See
{doc}`generate-config`.

## Related Pages

- {doc}`generate-config` for generation configuration.
- {doc}`output-files` for generated artifacts.
- {doc}`../explanation/pipeline-overview` for stage order and checkpoints.
