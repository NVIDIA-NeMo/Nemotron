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
| `unsupported_config_feature` | The config asks for work no stage performs: an unknown export name, balancing controls with deduplication disabled, eval or translation orchestration keys, leftover shared model fields, or an unrecognized `surface_generation` key. | Disable or remove the listed setting, or register its owning stage. Refer to {doc}`generate-config`. |
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

## Evaluation Configuration

| Code | Likely cause | Fix |
| --- | --- | --- |
| `eval_config_schema_invalid` | A section is open, a setting is missing, a boolean or number is quoted, `eval.mode` is empty or repeated, or a `REPLACE_ME_*` value remains. | Declare `schema_version: "1.2"` (or `"1.1"` under its stricter candidate rules) with `config_status: resolved`, and fix the field named in the error. Nothing here defaults. |
| `eval_config_path_invalid` | `source_run_manifest` names a table, a translation manifest names another run, executable paths disagree with the source, or the output directory overlaps the publication tree. | Point `source_run_manifest` at `run_manifest.json` from a completed `stage=generate` run, match `source_oracle` kind and pack identity to that run, and choose an output directory outside the generation tree. |
| `candidate_identity_invalid` | Two candidates share an alias or resolve to the same weights, or a candidate declares no serving route. | Give each candidate a unique filesystem-safe alias and a distinct canonical identity, one candidate per set of weights. Run `resolve_bfcl_model_identity` to fill the block in. |
| `candidate_revision_mutable` | The revision names a moving pointer such as `main`, a tag, or `refs/heads/*`. | Pin a full 40 to 64 character commit id, or set `model_identity.weights_digest`. A provider that publishes neither is declared by leaving both null, which records the candidate as `provider_managed` and makes the run non-publishable. |
| `secret_in_eval_config` | A literal credential was written into the config or embedded in `base_url`. | Name the environment variable with `candidates[].api.api_key_env` and export the value in the runner environment. Rotate the key that reached the file. |
| `eval_publication_policy_violation` | Publication was requested with a weakened gate or an unpinned candidate. | Restore every locked gate and pin each candidate, or set `publication.requested: false` and read `non_publication_reasons`. Refer to {doc}`eval-config`. |
| `unsupported_eval_mode` | `eval.mode` is empty, repeated, or names an unknown mode. | Write a non-empty list with no repeats. Executable modes additionally need a source run whose manifest declares an oracle. |
| `eval_cli_invalid` | The envelope changed a value the runner owns. | Start from `bfcl/config/eval.cli.yaml` or `bfcl/config/eval.launcher.yaml`, keep `stage: eval` and `family: bfcl`, and point `eval_config_path` at one resolved eval config. |

## Evaluation Source and Contamination

| Code | Likely cause | Fix |
| --- | --- | --- |
| `eval_source_manifest_invalid` | The named file is not a commit marker: absent, unreadable, misnamed, missing a publication field, or declaring an unreadable schema. | Rerun `stage=generate`, or evaluate with the pipeline revision that published the run. A parquet without its manifest is an unpublished artifact. |
| `eval_source_manifest_drift` | The manifest, its run id, or its schema version changed between config resolution and verification. | Re-resolve the eval config against the current publication and verify again. |
| `eval_source_benchmark_hash_mismatch` | A table's bytes do not match the declared hash, or the file is a symlink or missing. | Evaluate the tree the manifest describes, or regenerate. A table that changed after publication is a different benchmark whatever its name says. |
| `eval_source_benchmark_schema_mismatch` | The parquet was not written with the schema this build reads, or a row will not decode. | Regenerate with this pipeline revision. Skipping the row would silently change the task set. |
| `eval_source_publication_invalid` | The published table is not a selection of raw rows, a row restates a truth field, a held-out row shipped, or the declared counts disagree. | Regenerate the benchmark; the audit table can no longer explain a score. |
| `eval_source_task_index_invalid` | A duplicate task id, or an id that cannot be a path component or a log token. | Task ids derive from `pack_id` and `template_id`, so fix those in the pack and regenerate. |
| `eval_source_oracle_pack_drift` | The pack manifest, the execution resource, or any file in the pack tree no longer matches the recorded fingerprint. | Restore the pack revision the benchmark was generated from. A helper module the backend imports changes what the oracle does, so the whole tree counts. |
| `eval_source_oracle_resource_mismatch` | The declared oracle is not the one the source run executed, or the backend does not expose the required interface. | Declare the executed resource in the pack manifest. A resource chosen only by the eval config could execute code the source run never ran. |
| `eval_source_translation_lineage_invalid` | A localized benchmark does not satisfy the content-addressed translation contract. | Regenerate with the translation adapter and restore the complete output as one immutable tree. |
| `eval_source_model_exposure_invalid` | The manifest cannot say which models read its rows. | Regenerate with `lineage.roles` configured, or evaluate with the pipeline revision that published the run. A gap here would read as "no contamination found". |
| `eval_source_changed_during_eval` | The source moved after verification, usually a regeneration into the same directory. | Stop writing into the publication tree during an evaluation, then verify again from a clean tree. |
| `eval_source_invalid` | Source verification refused the run for a reason with no narrower code. | Read the subject and problem fields of the report; they name the specific check that failed. |
| `eval_contamination_candidate_exposed` | A candidate is the model that profiled, paraphrased, judged, or translated rows it would be scored on. | Evaluate a different candidate, regenerate with surface models that are not under evaluation, or accept a non-publishable score over the remaining rows with `on_violation: exclude_row`. The subject names every contaminated candidate, so fix them together. |
| `eval_contamination_unresolved` | A candidate and an exposed model cannot be told apart, and publication was requested. | Pin `weights_digest` and record the same identity for the generation role, or pin an immutable revision on both sides. Otherwise set `publication.requested: false` and read the recorded evidence. |
| `eval_contamination_empty_task_set` | Exclusion leaves nothing to score, or no row is answerable by every candidate. | Evaluate the candidate against a benchmark it did not help build, or regenerate with a different surface model. This is a benchmark-and-candidate mismatch, not a tuning problem. |
| `eval_contamination_task_set_inconsistent` | Eligible and excluded rows overlap, a candidate lacks the common set in publication order, or two candidates share an alias. | Re-verify the source and re-run the gate. A hand-edited plan is not an authorization. |
| `eval_contamination_plan_drift` | The plan was gated against another config or source, or no longer resolves to the current decision. | Verify the source and gate the candidates together, immediately before execution. |

## Evaluation Execution

| Code | Likely cause | Fix |
| --- | --- | --- |
| `eval_candidate_credentials_missing` | The variable named by `api_key_env` is unset or empty. | Export it in the eval runner environment. Never put the key in the config; replay from a completed cache needs no credential. |
| `eval_candidate_authentication_failed` | The endpoint answered HTTP 401 or 403. | Every task presents the same credential, so this is a configuration fault and the run stops on the first rejection. Check the key, then re-run into a new output directory. |
| `eval_candidate_retry_exhausted` | Every allowed transient attempt failed before the deadline. | Check endpoint health and rate limits, then start or resume. Authentication failures and malformed output never retry. |
| `eval_candidate_request_invalid` | The request is empty, belongs to another candidate, or an extension replaces a pinned field. | Rebuild it from the same candidate, authorized task, ordered messages, and model-facing tools. |
| `eval_candidate_provider_extension_invalid` | An extension sits outside the exact `<provider>.<api version>` namespace, or conflicts with a standard field. | Remove the unknown namespace or the conflicting key. |
| `eval_candidate_response_invalid` | HTTP 200 carried no OpenAI-compatible chat completion choice. | Fix endpoint compatibility. This is recorded as malformed model output and is never retried or repaired; malformed argument JSON is preserved separately for deterministic scoring. |
| `eval_conversation_script_invalid` | A published row is not a replayable conversation. | Do not edit the parquet. Regenerate from a pipeline run whose replay produced the conversation. |
| `eval_conversation_unauthorized` | The task was not assigned to this candidate, the alias now names different weights, or the projection does not match the verified source. | Re-project the benchmark under its expected hash and task ids, and drive the exact candidate contract passed to the gate. |
| `eval_conversation_answer_key_leak` | A prompt held an assistant or tool message the candidate did not produce and the driver did not release. | This is unrecoverable for the run: discard its episodes. Add prompt material only through the driver's append operations. |
| `eval_conversation_transition_invalid` | The driver was asked for a move its own state forbids, such as echoing an absent or ambiguous call id. | End the episode with the matching status instead of repairing the model's output. |
| `eval_executable_projection_invalid` | The clock, tool policy, milestones, projection identity, or oracle lineage is missing or inconsistent. | Restore the verified publication and pack. Do not reconstruct runner metadata from an unrelated row. |
| `eval_executable_unauthorized` | The task spec does not match the candidate, plan, source, policy, or oracle being driven. | Rebuild it from the exact projection and handles that source verification and contamination gating produced. |
| `eval_oracle_session_failed` | A task-local worker or endpoint session could not be opened or closed cleanly. | Discard the episode, verify the source resource again, and inspect worker or endpoint cleanup before retrying the complete task. |
| `eval_oracle_reset_failed` | The oracle could not establish clean state for the task. | Inspect fixtures and reset behavior, then rerun the whole task with a new session. Do not reuse a prior session. |
| `eval_oracle_call_failed` | A schema-valid call failed outside the structured business-result contract. | Record the infrastructure outcome. Never retry a mutating call in the same episode, because its commit state may be unknown. |
| `eval_oracle_state_failed` | The oracle could not return canonical JSON for final state. | Retain the earlier call evidence and rerun only after fixing the verified oracle. |
| `eval_assertion_infrastructure_failed` | A pack assertion could not be imported or executed. | Fix the assertion infrastructure and rerun the complete task. This is not a verdict and must not be counted as a candidate pass or failure. |
| `eval_candidate_cache_invalid`, `eval_tool_trace_cache_invalid` | A replay cache is truncated, hash-invalid, holds a claimed request with no completion marker, or contains a failure-only candidate completion. | Preserve the file as evidence and resume into a new output directory. Current candidate clients persist only semantically completed responses; transport, provider, malformed-envelope, authentication, and cancellation failures remain uncached and retryable. |
| `eval_candidate_cache_conflict`, `eval_tool_trace_cache_conflict` | The same immutable request key has different persisted evidence. | Never overwrite an observation. Investigate the source, plan, task-spec, or oracle identity, and use a new output directory. |

## Evaluation Scoring, Artifacts, and Orchestration

| Code | Likely cause | Fix |
| --- | --- | --- |
| `eval_trace_evidence_mismatch`, `eval_executable_evidence_mismatch` | An episode was scored against a script or task spec it did not answer. | Score each episode under the exact handles that drove it. A re-projected row is a different question, so re-drive it rather than re-pairing the evidence. |
| `eval_trace_scoring_policy_unsupported` | Trace scoring was asked for a verdict it cannot produce. | Score with `allow_llm_repair: false` and `task_success: all_applicable_gates`, or run executable evaluation. Repair would make the number a property of the repairer, and `assertions_only` needs an oracle. |
| `eval_executable_scoring_policy_unsupported` | The executable policy requests behavior the deterministic scorer cannot provide. | Restore the frozen scoring contract and disable repair or unsupported task-success semantics. |
| `eval_trace_aggregation_invalid`, `eval_executable_aggregation_invalid` | Task scores are missing, duplicated, reordered, or cross an identity boundary. | Score every authorized task exactly once in plan order before aggregation. |
| `eval_artifact_invalid` | The report, task table, manifest, aggregate hashes, or required caches do not form one immutable set. | Preserve the existing evidence and resume into a new output directory. |
| `eval_runner_invalid` | A runner precondition is unmet: an unverified config, an incomplete task set, or an empty run id. | Fix the named precondition and restart before candidate inference. |
| `eval_runner_mode_unsupported` | The runner does not publish the measurement the pinned mode declares. | Call the runner the mode declares, or let the config choose it. Each runner publishes one measurement. |
| `eval_nemo_adapter_invalid` | The bundle tree, source, candidate count, mount, or pinned versions do not match the native adapter contract. | Restore the exact six-file bundle tree and canonical source, use one candidate with the same model id, pass the mounted bundle path, keep the bundle, BFCL output, and native output trees disjoint, and pin the required evaluator and launcher versions. |
| `eval_cli_artifact_conflict` | Generated orchestration files disagree with what is on disk, or a path sits inside the immutable bundle. | Use a fresh orchestration output path, or restore the byte-identical generated files. The bundle's exact file set is re-verified every run. |
| `eval_cli_framework_not_installed`, `eval_cli_framework_version_mismatch` | The generated framework package is absent or stale in the Launcher environment. | Remove any stale distribution, install the exact package path printed as `framework_package`, then rerun the same orchestration config with submission enabled. |
| `eval_cli_runtime_failed` | An unknown runtime exception surfaced through the CLI. | Inspect the chained exception and the immutable source and contamination diagnostics. The CLI maps unrecognized failures here rather than inventing a candidate score. |

## Reading a Failure Attribution

A failed task is not automatically evidence about the model. Every terminal episode
status carries a fixed attribution, and only `candidate` statuses count against the
model under test.

| Attribution | Episode statuses |
| --- | --- |
| `success` | `completed` |
| `candidate` | `candidate_mismatch`, `malformed_response`, `unusable_tool_call_ids`, and, in executable mode, `confirmation_not_earned` |
| `infrastructure` | `candidate_call_failed`, `max_turns_exceeded`, `episode_timeout`, and, in executable mode, the oracle reset, call, timeout, malformed-result, state, and session failures plus `unknown_commit_state`, `dependency_resolution_failed`, and `assertion_infrastructure_failed` |

A metric that had nothing applicable to measure reports a `metric.*` not-applicable
code rather than a zero, so an absent gate never reads as a failed one. Setup failures
appear with `fatal_setup` attribution and one of the codes tabulated above. For how the
gates themselves are defined, refer to {doc}`../explanation/evaluation`.

## Related Pages

- Generation YAML fields: {doc}`generate-config`
- Evaluation YAML fields: {doc}`eval-config`
- Where each artifact is written: {doc}`output-files`
- Pack-input field and validation references: {doc}`oracle-pack-inputs`
- Pack structure and the Gold gate: {doc}`../explanation/oracle-pack`
