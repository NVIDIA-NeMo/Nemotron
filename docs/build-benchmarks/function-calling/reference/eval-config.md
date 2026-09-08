<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Evaluation Configuration Reference

Evaluation is a separate run over a benchmark that was already published. This page lists
the keys of the resolved scoring config, as validated by `load_eval_config` in
`src/nemotron/steps/byob/runtime/benchmark_families/bfcl/eval/config.py` and typed by the
models in the sibling `eval/schemas.py`. For what the numbers mean, see
{doc}`../explanation/evaluation`; to run one, refer to {doc}`../how-to/run-evaluation`.

## The Three Files

All three ship under `src/nemotron/steps/byob/bfcl/config/`.

| File | Role |
| --- | --- |
| `eval.default.yaml` | The scoring template. Copy it, fill in every `REPLACE_ME_*` value, set `config_status: resolved`, and keep it outside the generation output tree. Everything on this page describes this file. |
| `eval.cli.yaml` | The direct envelope. It carries `family: bfcl`, `stage: eval`, `eval_config_path` pointing at the resolved scoring config, `execution_backend: direct`, `output_format`, `probe_oracle`, and `dry_run`. |
| `eval.launcher.yaml` | The Launcher envelope. Same envelope keys with `execution_backend: nemo_launcher`, plus a `launcher` block naming the bundle root, materialized adapter, framework, task, and Launcher configs, the dataset mount, container mounts, and `submit`. |

The split is deliberate: operational CLI choices belong in an envelope so that selecting
a backend, changing output rendering, or doing a dry run never changes
`eval_config_hash`, which is the identity of the measurement itself.

:::{important}
Nothing in the scoring config falls back to a provider or pipeline default. Every section
is closed against unknown keys, and every key in a section must be present: an absent
limit or gate is an error, not a default, because a run must not be able to claim a
setting the config never stated. Quoted booleans and numbers are refused for the same
reason. Resolution reports every independent violation in one pass, with the first in
file order setting the exit status.
:::

## Identity and Path Resolution

| Field | Type | Default | Controls |
| --- | --- | --- | --- |
| `schema_version` | `"1.1"` or `"1.2"` | none, required | The contract the file is held to. `1.2` is current; a `1.1` file still loads and keeps its stricter candidate identity rules. Quote it so YAML keeps it a string. |
| `config_status` | `resolved` | none, required | Only `resolved` runs. `template` is refused as unrunnable, and any remaining `REPLACE_ME_` value anywhere in the file is refused separately. |

Relative paths resolve against the eval config's own directory. When an eval config is
inlined into a generation config as a legacy `eval` block, they resolve against the BYOB
root instead, matching how generation resolves its own paths. Credentials are refused
before anything is hashed or logged: any key that names a credential, and any string value
that looks like one, is rejected, so only variable names survive.

## Source Binding

| Field | Type | Default | Controls |
| --- | --- | --- | --- |
| `source_run_manifest` | path string | none, required | The `run_manifest.json` of a completed `stage=generate` run. A `.parquet` path is refused by name: the manifest is the commit marker and it names the table it published. |
| `source_oracle` | mapping or `null` | none, required | `null` for trace-only scoring. Required for executable modes. |
| `translation_manifest` | path string or `null` | none, required | Set only when scoring a localized benchmark derived from the same source run. |

Almost nothing about the source is restated by the operator. Which table to read, the run
id, the schema version, the lineage policy, and whether the run was gold-eligible all
come out of the manifest, so the config and the publication cannot disagree. The table's
content hash comes from `artifacts.benchmark_parquet.content_hash`: resolution pins what
the run claims, and source verification later proves the bytes.

`source_oracle` accepts exactly `kind`, `pack_manifest`, and `resource`, and requires all
three. `kind` is `python` or `endpoint` and must equal `oracle.kind` in the source
manifest. `pack_manifest` must be the `manifest.yaml` whose `pack_id` and `version` match
the pack the source run recorded. `resource` must be a `.py` file for kind `python`, and a
`.yaml`, `.yml`, or `.json` file for kind `endpoint`. A `translation_manifest` must declare
the same `source_run_id` or `source_run_manifest_content_hash` as the run being evaluated.

## `eval`

| Field | Type | Default | Controls |
| --- | --- | --- | --- |
| `mode` | list | none, required | A non-empty list with no repeats, drawn from `trace`, `executable`, and `held_out_eval`. It is reordered canonically, so two configs asking for the same work hash the same. |

`trace` scores the calls the candidate proposed against the published gold trace.
`executable` additionally replays them against the oracle pack and evaluates the pack's
assertions, so it requires a resolvable `source_oracle`. `held_out_eval` is the private
executable path; it requires a `held_out_eval` section, `contamination.comparison_set:
common_intersection`, no translation manifest, and `write_task_results`,
`cache_candidate_responses`, and `cache_tool_results` all false, so private tasks never
leave the boundary. That section accepts `policy_hash`, `fixture_refs`, `template_ids`,
`seed`, `pack_version`, `max_tasks_per_template`, and an optional `contract_version`.

## `scoring`

| Field | Type | Default | Controls |
| --- | --- | --- | --- |
| `contract` | path string | none, required | The document defining how a call is compared. It is content-hashed, so editing it changes `eval_config_hash`. Point it at `src/nemotron/steps/byob/references/bfcl-eval-scoring-contract.md`. |
| `argument_matching` | `schema_then_canonical` or `canonical_only` | none, required | Whether candidate arguments must satisfy their declared schema before canonical comparison. |
| `insert_declared_defaults` | bool | none, required | Whether declared parameter defaults are inserted before comparison. |
| `respect_call_order`, `respect_call_group` | bool | none, required | Whether call ordering and parallel call grouping are scored gates. |
| `allow_llm_repair` | bool | none, required | Whether malformed candidate output may be repaired by a model. |
| `task_success` | `all_applicable_gates` or `assertions_only` | none, required | How per-task success is derived. `assertions_only` needs an oracle, so a trace-only run refuses it. |
| `intermediate_text_matching` | `structural` or `verbatim` | `structural` | What an intermediate text-only turn must reproduce. This is the one optional key here: a config written before the policy existed means the value publication requires. |

## `limits`

Limits are part of the measurement: a model cut off at two turns did not answer the same
question as one given ten, so they enter the hash like the scoring flags do.

| Field | Type | Default | Controls |
| --- | --- | --- | --- |
| `max_turns` | integer > 0 | none, required | Assistant turns per episode. |
| `tool_timeout_s`, `candidate_timeout_s` | number > 0 | none, required | Per tool call, and per candidate request covering all attempts. |
| `episode_timeout_s` | number > 0 | none, required | Per episode. Must be at least the larger of the two per-call timeouts. |
| `max_parallel_tasks` | integer > 0 | none, required | Concurrent task-local sessions per candidate. |
| `max_retries` | integer ≥ 0 | none, required | Total HTTP attempts are `1 + max_retries`. Only transient transport failures retry. |

## `candidates`

A non-empty list. Each entry declares exactly `alias`, `model`, `provider`,
`provider_api_version`, `api`, `model_identity`, and `inference`, all required. `provider`
and `model` are the serving route; `model_identity` is the weights that answered. Keeping
them apart is what makes a score comparable after an endpoint is renamed.

| Field | Type | Controls |
| --- | --- | --- |
| `alias` | lowercase token, up to 64 characters | A unique, filesystem-safe name for the candidate's artifacts. |
| `model`, `provider`, `provider_api_version` | string, lowercase token, string | The serving route: the model id the request names, the provider, and its API version. |
| `api.base_url`, `api.api_key_env` | `http` or `https` URL, uppercase variable name | The endpoint, which may not embed credentials, a query string, or a fragment, and the variable the runner reads the key from. The key value never appears in the config. |
| `model_identity.source`, `model_identity.model` | lowercase token, string | Where the weights come from, for example `huggingface`, and their identifier at that source. |
| `model_identity.revision` | string or `null` | An immutable revision, meaning 40 to 64 hexadecimal characters. Branch and tag names such as `main` or `refs/heads/*` are refused rather than downgraded. |
| `model_identity.weights_digest` | string or `null` | `sha256:<64 hex>` over the weight bytes, or `bfcl-weight-manifest-v1:<64 hex>` over a manifest of a served directory. |
| `inference.temperature`, `inference.top_p` | number ≥ 0, number in (0, 1] or `null` | Decoding temperature and nucleus sampling. `top_p` must still be stated, but may be `null` when `temperature` is 0: the field is then omitted from the request instead of sent. Some endpoints, including every Anthropic model, refuse a request carrying both fields, and at temperature 0 the decode is greedy, so a nucleus cutoff cannot change the token chosen. Above 0 a value is required, because an omitted one would inherit a provider default that decides real sampling behaviour. |
| `inference.max_tokens`, `inference.seed` | integer > 0, integer or `null` | Response bound and optional decoding seed. |
| `inference.tool_choice` | `auto`, `required`, or `none` | Tool-choice policy. |
| `inference.provider_extensions` | mapping | Optional; keys are versioned namespaces such as `nvidia.v1`, and an extension may not replace a standard request field. |

A candidate that sets neither `revision` nor `weights_digest` is recorded as
`provider_managed`, the honest case for a hosted route that publishes neither. Its identity
must then restate its own `provider` and `model`, because the route is the only evidence of
which weights answered, and two such candidates on one route collide. Schema `1.1` has
neither an unpinned candidate nor a scheme-qualified digest; declare `1.2` to use either.
Aliases must be unique, and two candidates may not resolve to the same canonical identity.
Run `python -m nemotron.steps.byob.scripts.resolve_bfcl_model_identity` to fill the block
in rather than assembling it by hand.

## `contamination`

| Field | Type | Default | Controls |
| --- | --- | --- | --- |
| `enforce` | bool | none, required | Whether the contamination gate runs. |
| `on_violation` | `fail_run` or `exclude_row` | none, required | `fail_run` refuses the run. `exclude_row` drops only the rows the exposure covers and is debug-only. |
| `comparison_set` | `common_intersection` or `per_candidate` | none, required | Whether every candidate answers one shared task set. `per_candidate` is debug-only. |

Overlap is a validity precondition, not a scoring dimension. A comparison that cannot
settle whether a candidate is the model that profiled, paraphrased, judged, or translated
the rows is recorded as unresolved and never guessed either way, and unresolved evidence
blocks publication.

## `publication`

| Field | Type | Default | Controls |
| --- | --- | --- | --- |
| `requested` | bool | none, required | Whether the operator is asking for a publishable result. |
| `require_same_task_ids` | `true` | none, required | Must be `true`. Candidates scored on different task sets produce numbers that cannot be compared. |

`publication.requested: true` requires every locked gate: `argument_matching:
schema_then_canonical`, `respect_call_order` and `respect_call_group` true,
`allow_llm_repair` false, `task_success: all_applicable_gates`,
`intermediate_text_matching: structural`, contamination enforced with `fail_run` over a
`common_intersection`, all four `outputs` artifact flags true, and weights pinned on every
candidate. An executable claim also requires a gold-eligible source run. Requesting
publication with any of these weakened is refused; `requested: false` runs anyway and
reports each weakened field in `non_publication_reasons`.

## `outputs`

| Field | Type | Default | Controls |
| --- | --- | --- | --- |
| `output_dir` | path string | none, required | Where eval artifacts go. It must be a directory if it exists, must not overlap the source publication tree in either direction, and must not already hold `run_manifest.json`, `benchmark.parquet`, or `benchmark_raw.parquet`. |
| `write_task_results`, `write_eval_manifest` | bool | none, required | Write `eval_task_results.parquet` and `eval_manifest.json`. |
| `cache_candidate_responses`, `cache_tool_results` | bool | none, required | Write `candidate_io_cache.jsonl` for network-free replay and `tool_trace_cache.jsonl` for oracle-free replay. |

`output_dir` is deliberately absent from `eval_config_hash`: two operators running the same
evaluation into different directories ran the same evaluation. Which artifacts get written
is in the hash, because a run that skipped its per-task results cannot be audited.

For the resulting artifacts refer to {doc}`output-files`, for generation fields see
{doc}`generate-config`, and for error codes refer to {doc}`troubleshooting`.
