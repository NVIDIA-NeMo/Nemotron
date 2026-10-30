---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Symptom-to-remedy tables for nemotron steps run translate/nemo_curator across the LLM, NMT, Google, and AWS backends."
topics: ["Translation", "Reference"]
tags: ["Reference", "Translation", "Troubleshooting"]
content:
  type: "Reference"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

<!-- Reference: symptom-to-remedy tables for nemotron steps run translate/nemo_curator runs across all four backends and FAITH evaluation. -->

# Troubleshooting

This page lists common symptoms when you run `nemotron steps run translate/nemo_curator` and shows the field, flag, or environment variable to inspect first.
Each table pairs a symptom with a concrete remedy.
For stage flow and design rationale, see the explanation pages linked from {doc}`../explanation/index`.

## Authentication and Credentials

| Symptom | What to do |
| --- | --- |
| HTTP 401 or 403 from the chat-completions endpoint, or a Curator log line about a missing API key | Confirm the variable named in `server.api_key_env` is exported in the shell that launches the run. The starter `default.yaml` expects `NVIDIA_API_KEY`; export it with `export NVIDIA_API_KEY="<api-key>"` and rerun. See {doc}`../how-to/run-llm-translation`. |
| FAITH scoring fails with a credentials error even though `backend` is `nmt`, `google`, or `aws` | FAITH always uses the large language model (LLM) client under `server`. Keep `server.api_key_env` populated whenever `faith_eval.enabled` is `true`, or set `faith_eval.enabled=false` for a diagnostic run. See {doc}`../how-to/run-faith-evaluation`. |
| Google backend rejects the request with a permission or project error | Confirm application default credentials are present in the environment that runs the step. Do not paste secrets into `default.yaml`. See {doc}`../how-to/run-google-aws-translation`. |

## Model and Endpoint Configuration

| Symptom | What to do |
| --- | --- |
| HTTP 404 or a "model not found" message from the LLM endpoint | Hosted catalogs retire identifiers frequently. List the models your tenant currently exposes and pin `server.model` to one of them before large batch jobs. See {doc}`../how-to/run-llm-translation`. |
| Google translation rejects the request because `project_id` is missing | API version `v3` requires project metadata. Set both `google.project_id` and `google.api_version=v3`, or downgrade `google.api_version` to a release that does not require the project. See {doc}`../how-to/run-google-aws-translation`. |
| NMT requests time out before the service responds | Raise `nmt.timeout` to match observed server latency, lower `nmt.batch_size` so each request returns sooner, and confirm `nmt.server_url` resolves from the host that runs the step. See {doc}`../how-to/run-nmt-translation`. |

## Throttling and Concurrency

| Symptom | What to do |
| --- | --- |
| HTTP 429 responses, bursty failures, or sustained slowdowns from a hosted LLM endpoint | Lower `max_concurrent_requests` in your YAML and rerun on a smaller slice of data. Confirm your tenant quota covers the planned batch size. See {doc}`translate-config`. |
| A self-hosted NMT service returns errors under load | Reduce `nmt.max_concurrent_requests` and `nmt.batch_size` together, then raise them only after the service reports healthy throughput. See {doc}`../how-to/run-nmt-translation`. |

## Inputs and Output Layout

| Symptom | What to do |
| --- | --- |
| Reader errors about mixed file types when `input_path` points at a directory containing both JSONL and Parquet files | Curator readers expect one record format per directory. Split the inputs into separate directories for JSON Lines (JSONL) and Parquet, or set `input_path` to a single file. See {doc}`io-format`. |
| Ray worker logs show `Creating virtual environment at: .venv` followed by `ModuleNotFoundError: No module named 'ray'` | Export `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` before running local `uv run --no-sync nemotron steps run translate/nemo_curator ...`. This keeps Ray workers in the synchronized Nemotron environment. |
| Empty JSONL input fails with `No data read from files in task file_group_0` | The reader found no records. Treat the run as an empty-input validation failure, confirm the input path is correct, and rerun with a non-empty file or directory. |
| Output shards do not appear under `output_dir` after the run reports success | The writer emits partitioned files, not a single merged file. Inspect the shard pattern under `output_dir` and confirm `output_format` matches what downstream consumers expect. See {doc}`io-format`. |

## FAITH Evaluation

| Symptom | What to do |
| --- | --- |
| Every translated row is dropped after FAITH runs | The `faith_eval.threshold` value may be too strict for the chosen scorer model. Lower the threshold, set `faith_eval.filter_enabled=false` while you tune, or override the scorer with `faith_eval.model_name`. See {doc}`../how-to/run-faith-evaluation`. |
| FAITH scores look inconsistent across runs of the same data | Pin both `server.model` and `faith_eval.model_name` to specific identifiers so scorer drift does not move the threshold under you. See {doc}`../explanation/faith-evaluation`. |

## Related Reference

- Translation YAML fields: {doc}`translate-config`
- CLI syntax: {doc}`cli-translation`
- Input and output shapes: {doc}`io-format`
