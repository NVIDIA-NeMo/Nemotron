# Translation Guide

## Purpose

`translate/nemo_curator` turns row-oriented corpus data into translated corpus data. It is intended for data that may later feed data prep, SFT, CPT, BYOB review, or human QA.

The step is a thin Nemotron wrapper around NeMo Curator:

```text
input files
  -> Curator JsonlReader or ParquetReader
  -> Curator TranslationStage
  -> Curator JsonlWriter or ParquetWriter
output files
```

The translation sub-stages operate on Curator `DocumentBatch` objects in memory. Output files are created only by the final writer stage.

## Installation

Install translation dependencies before running locally:

```bash
uv sync --extra translate
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
```

Set `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` for local Curator/Ray translation runs so
Ray workers use the same installed project environment as the driver process.

If a QA environment needs BYOB and SDG in the same run:

```bash
uv sync --extra translate --extra byob --extra data-sdg --group run
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
```

For package-resource validation, confirm Curator prompt files are visible from the installed environment:

```bash
uv run --no-sync python - <<'PY'
import importlib.resources as ir

root = ir.files("nemo_curator.stages.text.experimental.translation.prompts")
for name in ("translate.yaml", "faith_eval.yaml"):
    path = root.joinpath(name)
    assert path.is_file(), path
    print(name, len(path.read_text()))
PY
```

## Required Questions

Ask these before running real translation:

- What is the input path?
- Should the output be JSONL or Parquet?
- What is the source language code?
- What is the target language code?
- Which field should be translated?
- Which backend should be used?
- Should FAITH score rows, filter rows, or stay disabled?
- Is the data plain text, structured chat, code-heavy, or tool-call data?

Do not infer `source_language` or `target_language` silently. The starter config intentionally leaves them empty.

## Field Selection

Use `text_field=text` for simple row text:

```json
{"text": "The central bank raised rates."}
```

Use `text_field='messages.*.content'` for OpenAI-style chat records:

```json
{"messages": [{"role": "user", "content": "Translate this."}]}
```

For chat data, set `reconstruct_messages=true` when the user wants message structure preserved in the output.

## Output Modes

Use `output_mode=replaced` for downstream training. It replaces the selected source field with translated text and keeps the dataset simple.

Use `output_mode=both` for audit and debugging. It keeps translated fields plus helper metadata and enables score merging.

Use `output_mode=raw` only for debugging internals. It is not the usual user-facing output.

Do not set `merge_scores=true` with `output_mode=replaced`. If score fields are required, use `output_mode=both`.

## Backends

### LLM

Use `backend=llm` for hosted OpenAI-compatible translation, structured chat, code blocks, JSON-heavy rows, or tool-call data.

Required config:

```bash
backend=llm
server.url="$TRANSLATION_BASE_URL"
server.model="$TRANSLATION_MODEL"
server.api_key_env=NVIDIA_API_KEY
```

The API key should be in the named environment variable. Do not write secrets into checked-in YAML.

### NMT

Use `backend=nmt` for large plain-text corpora when a local or remote NMT service is available.

The service contract is:

```text
GET  /health
POST /translate
```

Request body:

```json
{"texts": ["Hello world"], "src_lang": "en", "tgt_lang": "hi"}
```

Response body:

```json
{"translations": ["नमस्ते दुनिया"]}
```

If using IndicTrans2, prefer the Hugging Face interface server when possible. Keep the public service API as `en` and `hi`, and map internally to IndicTrans language codes such as `eng_Latn` and `hin_Deva`.

### Google And AWS

Use `backend=google` or `backend=aws` only when provider credentials are configured in the runtime environment. Keep YAML limited to provider settings such as project, location, region, and concurrency.

## FAITH

FAITH is optional quality scoring after segment translation.

Use it when translation quality needs evidence:

```bash
faith_eval.enabled=true
faith_eval.filter_enabled=false
faith_eval.model_name="$FAITH_MODEL"
```

Set `faith_eval.filter_enabled=true` only when the user explicitly wants low-quality rows dropped.

FAITH uses an LLM client even when translation uses `nmt`, `google`, or `aws`, so the `server` block still needs a valid endpoint and API key.

If the FAITH model returns an empty or invalid response, the run should fail clearly. Treat this as a model or implementation issue, not as successful translation.

## Skip Already Translated Rows

`skip_translated=true` checks input rows for a non-empty `translation_column`, usually `translated_text`.

It is not output-directory resume.

Correct use:

```json
{"text": "The central bank raised rates.", "translated_text": "केंद्रीय बैंक ने दरें बढ़ाईं।"}
{"text": "Heavy rains caused flooding.", "translated_text": ""}
```

Expected behavior:

- Rows with non-empty `translated_text` are not sent to the backend.
- Rows with empty or missing `translated_text` are translated.
- Skipped rows are restored before writing output.
- The output directory is still overwritten by the writer.

Do not test `skip_translated=true` by pointing input at a fully translated output directory unless the expected behavior is all rows skipped.

## Input And Output Directories

Output directory names in QA are descriptive only:

| Directory | Meaning |
| --- | --- |
| `out_llm_hi` | LLM backend output translated to Hindi. |
| `out_chat_hi` | Structured chat output translated to Hindi. |
| `out_faith_annotated` | Translation with FAITH scores and no filtering. |
| `out_nmt_hi` | NMT backend output translated to Hindi. |
| `out_parquet_hi` | Parquet input and Parquet output. |
| `out_resume_hi` | Output from a partial input with `skip_translated=true`. |
| `out_mixed_should_fail` | Intentional negative test for mixed formats. |

The writer uses overwrite mode. Existing output directory contents are removed at the start of a run.

## Single Huge Files

Curator readers are file-partition oriented. Do not add generic pandas chunking to the step by default.

If the user has one huge file and Curator file partitioning is not enough, create a one-off pre-step that splits the file into homogeneous JSONL or Parquet shards, then run `translate/nemo_curator` on the shard directory.

## Validation

For every smoke run, verify:

- Command exits 0.
- Output files exist under `output_dir`.
- Row count matches input when filtering is disabled.
- Translated field exists and is non-empty.
- Chat rows preserve `messages` shape and tool-call JSON.
- Parquet outputs can be read with pandas.
- Logs do not print API keys.

For negative tests, verify:

- Mixed JSONL and Parquet directories fail before translation.
- Missing `source_language` or `target_language` fails clearly.
- Missing LLM credentials fail before backend calls.
- Missing NMT server fails with a clear health check or connection error.

## Pipeline Placement

Use translated outputs before downstream prep or training:

```text
translate/nemo_curator -> data_prep/sft_packing -> sft/megatron_bridge
translate/nemo_curator -> sft/automodel
translate/nemo_curator -> data_prep/pretrain_prep -> pretrain/*
```

After translating data for training, run a multilingual tokenizer check before packing or training so sequence length and template assumptions still hold.
