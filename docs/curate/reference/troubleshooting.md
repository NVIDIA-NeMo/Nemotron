---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Troubleshooting reference for the curation steps and the curate flow."
topics: ["Curation", "Reference", "Troubleshooting"]
tags: ["Troubleshooting", "Curation", "NeMo Curator"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Data Scientist"]
---

# Curation Troubleshooting

Error identifiers below are the ones the step manifests declare.
Each per-step reference page lists that step's identifiers with recovery guidance; this page collects the identifiers raised by `curate/nemo_curator` and by the flow driver, grouped by cause.

## Input Problems

| Identifier or symptom | Likely cause | Fix |
| --- | --- | --- |
| `input_glob_no_matches` | Path does not exist in the current host, container, or shared mount | Use an absolute path or verify the mount. For local tiny runs, override the packaged `/nemo_run/code/...` path with `${PWD}/src/nemotron/steps/curate/nemo_curator/data/tiny.jsonl`. |
| `input_is_not_readable_here` | `input_glob` resolves to Parquet, which `curate/nemo_curator` does not read | Run `curate/ingest` to normalize the corpus, or narrow `input_glob` past the Parquet files. A Parquet file beside readable shards is skipped, not refused. |
| `input_has_nothing_readable` | The glob matched files but none parse as JSONL | Check that the glob does not point at a directory of reports or archives, and that the corpus has been through `curate/ingest`. |
| `metadata_field_absent_from_some_records` | A field in `metadata_fields` is missing from some records | Curator's behavior in this case has not been verified. Treat an output that lost a metadata column as unexplained and inspect a sample before relying on it. |
| `large_file_oom` | An input shard is too large for the available Ray worker memory | Split large JSONL files into smaller shards before running Curator. |

## Gate Problems

| Identifier or symptom | Likely cause | Fix |
| --- | --- | --- |
| `missing_language_model` | `language_codes` is non-empty but `models.fasttext_langid` is missing or invalid | Set `language_codes=[]` to disable language filtering, or provide the FastText `lid.176.bin` path. |
| `language_filter_removed_everything` | Language codes did not match the labels FastText emits, or the confidence threshold is too high | Codes are matched case-insensitively against the emitted label, including the part before a script suffix. Confirm that `models.fasttext_langid` points at the expected model and lower `quality_filters.min_langid_score`. |
| `incomplete_word_filter` | Only one of `min_words` and `max_words` was set | Set both `quality_filters.min_words` and `quality_filters.max_words`, or remove both keys. |
| `empty_or_tiny_output` | Filters are too strict, or were applied before the corpus was understood | Re-run with `language_codes=[]`, `domains=[]`, and `quality_filters={}`, or profile the unfiltered corpus with `curate/profile`. Add gates back one at a time. |
| Domain classifier downloads repeatedly | Hugging Face cache path is not persistent | Set `models.hf_cache_dir` to a persistent cache location and mount it in remote profiles. |

## Policy Problems

| Identifier | Likely cause | Fix |
| --- | --- | --- |
| `policy_not_approved` | The policy carries `approved: false` or lacks a required approval field | `curate/profile` emits candidates, never approvals. Promote the policy deliberately, or set `heuristic_filters.allow_unvalidated_policy` to override with a recorded warning. Refer to {doc}`policy-file`. |
| `policy_langpack_mismatch` | The policy was derived from a different language pack than this run loads | Thresholds do not transfer between packs. Re-profile against the current pack or pin the pack version. |
| `unknown_signal_in_policy` | The policy names a signal outside the registry | The error lists the allowed names. A policy file cannot name an import path. |
| `langpack_tag_mismatch` | The pack directory name and the `language_tag` in its `pack.toml` differ | Rename the directory to match the manifest, or correct `pack.language_tag`. |
| `missing_langpack_dir` | No pack root was named | Set `langpack_dir` explicitly. Nemotron never selects a pack root implicitly. |

## Flow Problems

These identifiers are raised by the flow driver at preflight, before any step runs.
Refer to {doc}`flow-config`.

| Identifier | Likely cause | Fix |
| --- | --- | --- |
| `approve_before_profile` | `approve.from` points at a `candidate_policies.yaml` that does not exist | Run once with `steps.profile` enabled. There is nothing to approve until the corpus has been measured. |
| `profile_enabled_during_approval` | `approve` is set while `steps.profile` is enabled, even when the candidate file exists | Disable `steps.profile` to avoid replacing reviewed candidate measurements. |
| `approval_corpus_mismatch` | The candidate policy's corpus fingerprint differs from the corpus present now | Re-profile, or set `approve.verify_corpus: false` if you accept that the approval describes different data. |
| `conflicting_approved_policy` | `approve` promotes one policy while `steps.filter.heuristic_filters.approved_policy` names another | Remove the per-step path and let the flow wire the policy it promoted. |
| `score_column_never_written` | `steps.subset.quality_score_field` names a `__<signal>` column but `steps.filter.mode` is `filter` | Set `mode` to `annotate` or `both`, or unset `quality_score_field`. |
| `missing_upstream_artifact` | An enabled step needs an artifact whose producer is disabled and which is not on disk | Enable the producing step, or point `output_root` at a previous run that has it. |
| `stale_filter_output` | `filtered_jsonl/` already holds corpus shards | Delete the directory or choose a new `output_root`. The step appends rather than replaces. |
| `ingest_source_field_unmapped` | `corpus.source_field` is set but ingestion has no column or constant to write into it | Set `corpus.source_field_in_source` or `corpus.source_value`. |
| `decontamination_needs_a_holdout` | `steps.decontamination` is enabled without `holdout` | Set `steps.decontamination.holdout`. The protected split cannot be guessed. |
| `no_steps_enabled` | Every `steps.<name>.enabled` is `false` | Enable at least one step. |
| `unknown_step` | A key under `steps` is not one of `ingest`, `profile`, `filter`, `audit`, `subset`, or `decontamination` | Correct the spelling. Unknown steps are refused rather than ignored. |

## Runtime Problems

| Identifier or symptom | Likely cause | Fix |
| --- | --- | --- |
| `no_gpu_available` or the decontamination similarity pass fails to start | Flow preflight warns about missing GPU dependencies or resources, but `curate/decontamination` fails when the similarity pass starts | Run `uv sync --extra curate-gpu` and provide a GPU, or set `steps.decontamination.skip_similarity: true` to run only the exact source-identity pass. |
| `not_enough_cpu_resources` | Ray cannot schedule the requested CPUs | Set `ray.num_cpus` in YAML or `NEMOTRON_CURATOR_RAY_NUM_CPUS` in the environment profile. |
| Ray worker starts a new `.venv` or cannot import dependencies | Local `uv run` and the Ray runtime environment are both attempting to manage dependency setup | Export `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` and run with `uv run --no-sync` after `uv sync --extra curate`. |
| `ModuleNotFoundError: No module named 'cosmos_xenna'` inside a Ray worker | The flow was started without the `xenna` extra | Run the flow with `uv run --extra curate --extra xenna`. |

## Debug Checklist

1. Run the local tiny command with filters disabled.
2. Confirm `uv sync --extra curate` completed in the same repository clone.
3. Confirm the input path exists where the command runs.
4. Confirm `output_dir` is writable and, for the flow, that `filtered_jsonl/` is empty.
5. For the flow, run with `--plan` and read `flow_plan.json` before running.
6. Add language, word-count, and domain gates one at a time, or profile the corpus first.
