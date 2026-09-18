<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

(sdg-persona-mcq-config)=
# Persona MCQ Configuration Reference

Configuration fields, output artifacts, and named errors for `sdg/persona_mcq`. For the procedure, refer to {doc}`../how-to/persona-mcq-data`.

## Command

```text
uv run nemotron steps run sdg/persona_mcq -c {default|tiny} [-d] [--run PROFILE | --batch PROFILE] [KEY=VALUE ...]
```

Overrides use OmegaConf dotlist syntax. List values are passed in brackets and quoted for the shell, for example `'pipeline.stages=[answers,build_sft,sample]'`. Shipped environment profiles are `<backend>_sdg_persona_mcq` and `<backend>_sdg_persona_mcq_tiny` for `lepton`, `slurm`, and `dgxcloud`.

## Shipped Configuration

```{literalinclude} ../../../src/nemotron/steps/sdg/persona_mcq/config/default.yaml
:language: yaml
:class: scrollable
```

`tiny.yaml` differs from `default.yaml` in `pipeline.experiment_name` (`persona-mcq-smoke`), `question_models` (`[oss]`), reduced inference parallelism and token limits, `question_generation.num_records: 8`, `semantic_dedup.device: cpu`, `answer_generation.max_retries: 3`, and `sampling.per_language: 1`.

## Fields

### `pipeline`

| Field | Default | Description |
|---|---|---|
| `experiment_name` | `persona-mcq` | Required. Artifact directory name under `output_root`; also identifies the configuration for resume checks. |
| `output_root` | `${NEMOTRON_RUN_DIR:-./outputs}/persona_mcq` | Parent directory for experiments. |
| `stages` | `[all]` | `[all]` or an ordered subset of `personas`, `questions`, `lexical_dedup`, `semantic_dedup`, `answer_seed`, `answers`, `build_sft`, `sample`. `all` cannot be combined with named stages. |
| `seed` | `42` | Seed for question authoring and answer-seed option shuffles. |
| `resume` | `true` | Skip completed question outputs and answered rows; reload the existing `summary.json`. |
| `overwrite` | `false` | Delete the experiment directory before running. Required to reuse an experiment name with a different configuration. |

### `languages.<key>`

Every entry is independent; the key names carry no language-specific behavior.

| Field | Required | Description |
|---|---|---|
| `display_name` | yes | Language name inserted into authoring and answering prompts. |
| `locale` | yes | Data Designer persona locale, for example `en_IN` or `hi_Deva_IN`. |
| `answer_label` | yes | Localized final-answer marker; the assistant's last line must be `<answer_label>: <LETTER>`. |
| `answer_label_aliases` | no | Additional accepted markers when parsing teacher answers. |
| `script_pattern` | yes | Single-character regular expression for the target script. Must not match the empty string. |
| `question_script_fraction` | yes | `{min, max}` inclusive range for the fraction of alphabetic characters in a question that match `script_pattern`. |
| `answer_script_fraction` | yes | The same range applied to the exported assistant answer. |
| `strip_latin_glosses` | no (`false`) | Remove parenthetical Latin-script glosses from questions and options before deduplication. |

### `question_models`, `answer_models`, and `models.<alias>`

`question_models` lists the aliases whose questions are pooled before deduplication. `answer_models` lists the teacher panel; at least three are required. Every alias must be defined under `models`.

| Field | Required | Description |
|---|---|---|
| `model` | yes | Model identifier sent to the endpoint. |
| `endpoint` | yes | OpenAI-compatible base URL; the shipped configurations resolve `${oc.env:*_API_BASE}`. |
| `api_key_env` | yes | Name of the environment variable holding the bearer token. The value is never written to run metadata. |
| `provider_type` | no (`openai`) | Data Designer provider type for question authoring. |
| `skip_health_check` | no (`false`) | Skip the Data Designer endpoint health check. |
| `extra_headers` | no | Additional HTTP headers for question authoring. |
| `question.inference` | no | Data Designer `ChatCompletionInferenceParams` for authoring: `temperature`, `top_p`, `max_tokens`, `max_parallel_requests`, `timeout`, `extra_body`. |
| `answer` | no | Answer request parameters: `temperature` (`1.0`), `top_p` (`1.0`), `max_tokens` (`16384`), and `extra_body` merged into the request body. |

### `question_generation`

| Field | Default | Description |
|---|---|---|
| `num_records` | `100000` | Questions requested per question model and language. |
| `max_parallel_requests` | `256` | Data Designer non-inference worker parallelism. |
| `buffer_size` | `500` | Data Designer run buffer size. |

### `lexical_dedup`

| Field | Default | Description |
|---|---|---|
| `threshold` | `0.80` | Jaccard similarity of question shingles at or above which a question is a near-duplicate. |
| `shingle_size` | `4` | Shingle length. |
| `permutations` | `128` | MinHash permutations. Must be divisible by `bands`. |
| `bands` | `16` | Locality-sensitive hashing bands. |
| `seed` | `42` | MinHash seed. |

### `semantic_dedup`

| Field | Default | Description |
|---|---|---|
| `model` | `intfloat/multilingual-e5-large` | Sentence Transformers embedding model. Questions are embedded with the `query: ` prefix. |
| `device` | `cuda` | Embedding device. `tiny.yaml` uses `cpu`. |
| `batch_size` | `256` | Embedding batch size. |
| `threshold` | `0.965` | Cosine-similarity threshold for a near-duplicate pair; must be between 0 and 1. |
| `method` | `greedy` | `greedy` visits records in a seeded random order, keeps each unvisited record, and removes its similar neighbors; `components` keeps one randomly chosen record per connected component of similar pairs. |
| `seed` | `13` | Seed for the keep order. |
| `chunk_size` | `4096` | Rows per similarity block. |

### `answer_generation`

| Field | Default | Description |
|---|---|---|
| `max_parallel_requests` | `256` | Concurrent answer requests per teacher and language. |
| `timeout` | `600` | HTTP timeout in seconds. |
| `max_retries` | `5` | Attempts per row; HTTP 429 and 5xx responses are retried with exponential back-off capped at 30 seconds. |

### `sft`

| Field | Default | Description |
|---|---|---|
| `response_teachers` | `[qwen, oss, gemma]` | Teachers whose answers are exported. Each must also appear in `answer_models`. |
| `agreement` | `unanimous` | `unanimous`: every parsed letter must match. `majority`: strictly more than half of the panel must match. The exporting teacher must match the vote. |
| `reasoning.display_name` | `English` | Language requested for the reasoning. |
| `reasoning.script_pattern` | `[A-Za-z]` | Script matcher for the reasoning. |
| `reasoning.script_fraction` | `{min: 0.85, max: 1.0}` | Inclusive script-fraction range for the retained reasoning. |

### `sampling`

| Field | Default | Description |
|---|---|---|
| `per_language` | `50000` | Records sampled per language from the questions that every response teacher exported. |
| `seed` | `42` | Sampling and shuffle seed. |
| `reasoning_off_fraction` | `0.10` | Fraction of sampled records per language that omit `reasoning_content`; the selection is deterministic in `seed` and `query_id`. |
| `answer_variant` | `full` | `full` keeps the teacher's answer text; `stripped` replaces it with `<answer_label>: <letter>`. |

## Output Artifacts

All paths are relative to `<output_root>/<experiment_name>/`.

| Path | Content |
|---|---|
| `run.json` | Redacted configuration, `config_hash`, `data_designer_version`, and `nemotron_commit`. |
| `summary.json` | Per-stage statistics: cached and downloaded locales, question counts, deduplication drop reasons, answer counts, `build_sft` rejection reasons, and sample views. Rewritten after every stage. |
| `questions/<model>/<language>/records.jsonl` | Authored question records and Data Designer artifacts. |
| `lexical/<language>.jsonl` | Pooled, lexically deduplicated questions with `query_id`, `question`, `choices`, and provenance `metadata`. |
| `semantic/<language>.jsonl` | Semantically deduplicated questions. |
| `answer_seed/<language>.jsonl` | Shuffled options with `original_choices` and `shuffle_permutation`. |
| `answers/<model>/<language>/answers.jsonl` | Teacher responses with `answer`, `reasoning`, `parsed_letter`, `finish_reason`, and `completion_tokens`. Append-safe on resume. |
| `answers/<model>/<language>/failures.jsonl` | Rows that failed after `max_retries`; retried on resume. |
| `sft/<teacher>/<language>.jsonl` | Quality-gated `{messages, metadata}` records. |
| `training/<teacher>/train.jsonl`, `blend.json` | Aligned samples across all configured languages and the corresponding SFT-packing blend manifest. |
| `training/<teacher>/english_<language>/train.jsonl`, `blend.json` | Equal-count English/target views, written when a language named `english` is configured. |

### Training Record

```json
{
  "messages": [
    {"role": "system", "content": ""},
    {"role": "user", "content": "<answer instruction>\n\n<question>\n\nA) ...\nB) ...\nC) ...\nD) ..."},
    {"role": "assistant", "reasoning_content": "<English reasoning>", "content": "<answer text ending in answer_label: LETTER>"}
  ],
  "metadata": {
    "query_id": "...",
    "language": "hindi",
    "answer_label": "उत्तर",
    "reasoning_language": "English",
    "response_model": "gemma",
    "agreement": "unanimous",
    "voted_letter": "B",
    "n_valid_votes": 3,
    "difficulty": "...",
    "topic": "...",
    "facet": "...",
    "region": "...",
    "gen_model": "oss",
    "persona_uuid": "...",
    "reasoning_mode": "on"
  }
}
```

Records with `reasoning_mode: "off"` omit `reasoning_content`.

(sdg-persona-mcq-persona-cache)=
## Persona Asset Cache

The `personas` stage looks for each configured `languages.<key>.locale` at:

```text
<managed-assets>/datasets/<locale>.parquet
```

`DATA_DESIGNER_MANAGED_ASSETS_PATH` sets `<managed-assets>` directly. If it is unset, the path defaults to `${DATA_DESIGNER_HOME:-~/.data-designer}/managed-assets`.
To populate the cache without giving the pipeline an NGC key, set the same paths in the downloading shell and download every configured locale before running:

```console
$ export DATA_DESIGNER_HOME="<shared-cache-root>"
$ export DATA_DESIGNER_MANAGED_ASSETS_PATH="$DATA_DESIGNER_HOME/managed-assets"
$ uv run data-designer download personas --locale en_US
```

Repeat the download for each configured locale, then make the cache visible to the pipeline job at the same path.
The shipped `*_sdg_persona_mcq` profiles forward both variables.

## Errors

| Error | Cause | Recovery |
|---|---|---|
| `experiment_config_mismatch` | The experiment directory was created with a different configuration (`config_hash` differs). | Choose a new `pipeline.experiment_name`, or set `pipeline.overwrite=true` intentionally. Never combine artifacts from incompatible configurations. |
| `persona_assets_missing` | A configured locale has no cached persona asset and `NGC_API_KEY` is unset or the NGC CLI is not on `PATH`. | Export `NGC_API_KEY` and install the NGC CLI, or pre-download every locale as described in {ref}`sdg-persona-mcq-persona-cache`. |
| `teacher_intersection_too_small` | Fewer questions survived every response teacher's gates than `sampling.per_language` requests. | Inspect the `build_sft` rejection reasons in `summary.json`; reduce `sampling.per_language`, relax `sft.agreement`, or resume the `answers` stage to retry failed rows. |

Configuration validation also rejects fewer than three `answer_models`, response teachers absent from `answer_models`, unknown stage names, invalid `script_pattern` values, and script-fraction ranges outside `0 <= min <= max <= 1`.
The `lexical_dedup` stage separately rejects `permutations` values that are not divisible by `bands` when that stage runs; check the pair before a full run because initial configuration validation does not inspect it.

## Related

- Step contract: [`src/nemotron/steps/sdg/persona_mcq/README.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/sdg/persona_mcq/README.md)
- Question authoring plugin: [`src/nemotron/steps/sdg/plugins/persona_mcq/`](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/src/nemotron/steps/sdg/plugins/persona_mcq)
- Cluster dispatch: {doc}`../how-to/dispatch-to-cluster`
