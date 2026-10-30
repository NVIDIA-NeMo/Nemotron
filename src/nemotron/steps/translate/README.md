# Nemotron Translation

Use this README when a user wants to translate corpus data, chat records, or row-oriented training artifacts. The concrete step is [`translate/nemo_curator`](nemo_curator/README.md).

## Developer Journey

Translation is usually an upstream data step. Decide whether the translated
output is training-ready data, audit data with metadata, or quality-scored data
that may be filtered before training.

1. Choose one input format and schema: JSONL or Parquet, plain text or structured
   chat.
2. Pick the text field path to translate (`text`, `messages.*.content`, or a
   project-specific field).
3. Choose backend by data shape and throughput needs.
4. Run a two-row smoke test and inspect translated fields before scaling.
5. Send the translated output to curation review, SFT prep, pretraining prep, or
   human audit depending on `output_mode` and FAITH settings.

## Default Workflow

1. Install runtime dependencies with `uv sync --extra translate`.
2. For local Curator/Ray runs, set `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` before running translation commands so Ray workers use the installed project environment.
3. Read [`nemo_curator/step.toml`](nemo_curator/step.toml) for the step contract.
4. Ask for `source_language`, `target_language`, input path, output path, backend, and field path. Do not infer source or target language silently.
5. For downstream training data, start with `output_mode=replaced`, `merge_scores=false`, and `faith_eval.enabled=false`.
6. For audit or quality review, use `output_mode=both` and enable `faith_eval`.
7. Run a two-row smoke test before a large corpus.
8. Validate row count, schema, translated field content, and that secrets were not printed.

For one-shot translation requests, do not end in exploration mode. Provide the
minimal runnable handoff first:

- `Decision`: chosen step and scope, such as `translation-only` or `translation+FAITH`.
- `Config`: key fields or config path bound to a concrete input path and format.
- `Run`: exact command.
- `Output`: expected output directory and artifacts.
- `Env`: required environment variable names only, never values.

Before finalizing, make these constraints explicit in prose even when the
command implies them:

- Input path and format selected for the run.
- Incompatible inputs excluded, such as mixed JSONL and Parquet roots.
- Observed language mismatch versus requested translation direction, with the
  assumption used.
- Exact model variant used, or the default model assumption if the user gave
  only a model family.
- Credential environment variable names used for auth.

For `translation+FAITH`, add a short `FAITH handoff` section that confirms
`faith_eval.enabled=true`, states any filter assumptions, lists expected
FAITH-related outputs, and includes the exact run command and output path.

## Data And Artifact Flow

```text
filtered_jsonl / parquet
  -> translate/nemo_curator
  -> translated_jsonl / translated parquet
  -> data_prep/sft_packing, data_prep/pretrain_prep, review, or audit
```

For chat data, translate only natural-language message content and preserve
tool-call structure. For training data, start with `output_mode=replaced` and
FAITH filtering disabled; use `output_mode=both` when scores and raw metadata
must be inspected before deciding what to train on.

## Backend Choice

| Need | Backend | Notes |
| --- | --- | --- |
| Structured chat, tool calls, JSON, code, or high formatting fidelity | `llm` | Use OpenAI-compatible endpoint settings under `server`. |
| Large plain-text corpus with local service | `nmt` | Service must expose `GET /health` and `POST /translate`. |
| Managed provider translation | `google` or `aws` | Credentials must come from environment or provider config, not YAML secrets. |
| Quality scoring for any backend | FAITH | FAITH still needs an LLM client even when translation backend is not `llm`. |

## Common Commands

Plain text JSONL through a hosted LLM:

```bash
uv run --no-sync nemotron steps run translate/nemo_curator \
  input_path="$TR_ROOT/news_en" \
  output_dir="$TR_ROOT/out_llm_hi" \
  source_language=en \
  target_language=hi \
  backend=llm \
  text_field=text \
  output_mode=replaced \
  merge_scores=false \
  reconstruct_messages=false \
  faith_eval.enabled=false \
  server.url="$TRANSLATION_BASE_URL" \
  server.model="$TRANSLATION_MODEL" \
  server.api_key_env=NVIDIA_API_KEY
```

Structured chat records:

```bash
uv run --no-sync nemotron steps run translate/nemo_curator \
  input_path="$TR_ROOT/chat_code_en.jsonl" \
  output_dir="$TR_ROOT/out_chat_hi" \
  source_language=en \
  target_language=hi \
  backend=llm \
  text_field='messages.*.content' \
  output_mode=replaced \
  merge_scores=false \
  reconstruct_messages=true \
  faith_eval.enabled=false \
  server.url="$TRANSLATION_BASE_URL" \
  server.model="$TRANSLATION_MODEL" \
  server.api_key_env=NVIDIA_API_KEY
```

NMT server:

```bash
uv run --no-sync nemotron steps run translate/nemo_curator \
  input_path="$TR_ROOT/news_en" \
  output_dir="$TR_ROOT/out_nmt_hi" \
  source_language=en \
  target_language=hi \
  backend=nmt \
  nmt.server_url="$NMT_SERVER_URL" \
  text_field=text \
  output_mode=replaced \
  merge_scores=false \
  reconstruct_messages=false \
  faith_eval.enabled=false
```

## Patterns To Cite

- [`../patterns/translate-training-corpus.md`](../patterns/translate-training-corpus.md) for inserting translation before prep or training.
- [`../patterns/prefer-llm-for-structured-chat.md`](../patterns/prefer-llm-for-structured-chat.md) for chat, tool-call, JSON, and code-heavy data.
- [`../patterns/prefer-nmt-for-large-corpora.md`](../patterns/prefer-nmt-for-large-corpora.md) for large plain-text translation.
- [`../patterns/enable-faith-for-high-value-data.md`](../patterns/enable-faith-for-high-value-data.md) for quality annotation or filtering.
- [`../patterns/multilingual-tokenizer-check.md`](../patterns/multilingual-tokenizer-check.md) before using translated data for SFT or CPT.

## Guardrails

- Do not build custom readers or writers first. Use Curator `JsonlReader` or `ParquetReader`, `TranslationStage`, and `JsonlWriter` or `ParquetWriter`.
- Do not mix JSONL and Parquet in one input directory.
- If the user provides a mixed-format root, require an explicit include/exclude decision before running.
- Do not use `merge_scores=true` with `output_mode=replaced`; use `output_mode=both` if scores must be merged.
- Do not treat `skip_translated=true` as output-directory resume. It only skips input rows that already contain a non-empty translation column.
- Do not enable FAITH filtering without telling the user that rows may be dropped.
- Keep API keys in environment variables such as `NVIDIA_API_KEY`, `NGC_API_KEY`, AWS credentials, or Google application credentials.
- Never run environment-dump commands such as `env`, `printenv`, `set`, or broad `export` listings.
- For diagnostics, mention only environment variable names and keep values redacted.

## Troubleshooting

- CLI mismatch or unexpected-argument errors: return to the documented command
  shape in this file and confirm supported flags with `--help`; do not invent
  alternate subcommands.
- Missing translation dependencies: run `uv sync --extra translate` first and
  export `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` before local Curator/Ray translation
  commands.
  If an eval/runtime environment still misses basics such as `toml` or
  `pyyaml`, report the blocker and still provide the runnable handoff.
- Mixed `.jsonl` and `.parquet` roots: bind `input_path` to one format only and
  explicitly state excluded paths or formats.
- Missing `translate/nemo_curator` metadata in a runtime workspace: treat it as
  an environment/path issue, state the blocker, and provide the canonical
  command for a complete checkout.
- Path-not-found during validation: inspect actual created paths before
  retrying; do not guess output roots.

## Further Reading

- [`guide.md`](guide.md) for detailed flow, output modes, FAITH, resume semantics, and validation.
- [`nemo_curator/README.md`](nemo_curator/README.md) for the concrete step.
- [`nemo_curator/config/default.yaml`](nemo_curator/config/default.yaml) for starter config.
- [`nemo_curator/step.py`](nemo_curator/step.py) for the reader -> translation stage -> writer implementation.
