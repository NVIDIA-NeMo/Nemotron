# Translation Step (Curator backend)

Use `translate/nemo_curator` for corpus translation. See `../README.md` for
the broader translation journey and backend choice. This README covers the
concrete step: knobs, runtime setup, validation, and pitfalls.

Use this README for workflow and pitfalls; use `step.toml` for the exact
artifact, parameter, strategy, and error manifest.

## CLI And Overlay Knobs

CLI overrides are often clearer than writing a new YAML file for one-off runs.
For repeatable jobs, put the same fields in a project overlay. Developers
usually change:

- `input_path` and `output_dir`: concrete runtime-visible paths.
- `source_language` and `target_language`: never infer silently for production.
- `text_field`: `text` for plain JSONL, `messages.*.content` for chat records.
- `backend`: `llm`, `nmt`, `google`, or `aws`.
- For `backend=llm` (and FAITH scoring): `server.url`, `server.model`, and
  `server.api_key_env`.
- For `backend=nmt`: `nmt.server_url` after verifying `/health` and `/translate`.
- `output_mode`: `replaced` for training-ready data, `both` for audit/scoring.
- `reconstruct_messages`: `true` for chat records, `false` for plain text.
- `translation_prompt_path`: optional absolute path to a custom Curator prompt YAML.
- `generation_config`: optional OpenAI-compatible translation generation settings.
- `max_concurrent_requests`, `health_check`, `dry_run`, and `dry_run_log_count`:
  optional translation execution controls.
- `faith_eval.enabled` (and `faith_eval.filter_enabled` only after confirming
  rows may be dropped).
- `faith_eval.prompt_path`, `faith_eval.generation_config`, and
  `faith_eval.max_concurrent_requests`: optional FAITH prompt, generation, and
  concurrency controls.

Related pattern: [translate-training-corpus.md](../../patterns/translate-training-corpus.md).

## Runtime Setup

- Install runtime dependencies with `uv sync --extra translate`.
- Export `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` before local Curator/Ray translation
  commands so Ray workers inherit the project environment.
- Runtime dependencies should include parser/config basics (`toml`, `pyyaml`).
- Import check before scaling:
  `uv run --no-sync python -c "from nemo_curator.stages.text.experimental.translation import TranslationStage; print(TranslationStage)"`.

## Run It

Smoke first with two rows to validate the reader/writer path:

```bash
uv run --no-sync nemotron steps run translate/nemo_curator \
  input_path=<two-row-sample> \
  output_dir=<smoke-output> \
  source_language=<src> \
  target_language=<tgt> \
  backend=llm \
  text_field='messages.*.content' \
  faith_eval.enabled=false \
  server.url="$TRANSLATION_BASE_URL" \
  server.model="$TRANSLATION_MODEL" \
  server.api_key_env=NVIDIA_API_KEY
```

Then run the real job with the production input and the desired `output_mode`
/ FAITH settings:

```bash
uv run --no-sync nemotron steps run translate/nemo_curator \
  -c <project>/config/translate.yaml \
  input_path=<input> \
  output_dir=<translated-output>
```

After a run, verify row count, translated field content, and (for chat data)
that `tool_calls[].function.arguments` remains valid JSON. For NMT, call
`GET /health` against the configured server before kicking off a long run.

## Repository Layout

- Manifest: `src/nemotron/steps/translate/nemo_curator/step.toml`
- Runner: `src/nemotron/steps/translate/nemo_curator/step.py`
- Config: `src/nemotron/steps/translate/nemo_curator/config/default.yaml`
- Guide: `src/nemotron/steps/translate/guide.md`

## Avoid

- Do not mix JSONL and Parquet in one input directory.
- Do not store API keys in config files; use environment variable names only.
- Do not use `merge_scores=true` with `output_mode=replaced`.
- Do not treat `skip_translated=true` as output-directory resume — it only
  skips input rows that already contain a non-empty translation column.
- Do not silently enable FAITH filtering for training data; rows can be
  dropped.
- Do not add custom chunking to `step.py` for normal use. Split huge single
  files before this step if needed.
