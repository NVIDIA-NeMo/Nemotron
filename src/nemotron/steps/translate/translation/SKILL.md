---
name: nemotron-translate-translation
description: Configure and run the concrete translate/translation step for JSONL or Parquet corpus translation with NeMo Curator's experimental TranslationStage.
---

# Translation Step

Use `translate/translation` for corpus translation. Before changing configs or code, read `step.toml` to understand the step contract, consumed artifacts, produced artifacts, parameters, strategies, and failure modes.

## Inputs And Outputs

- Consume homogeneous JSONL or Parquet records from `input_path`.
- Translate `text_field`, such as `text` or `messages.*.content`.
- Produce translated JSONL or Parquet shards under `output_dir`.
- Use `output_mode=replaced` for training-ready data.
- Use `output_mode=both` for audit data with metadata and scores.

## Configure

- Set `source_language` and `target_language` explicitly.
- Set `backend` to `llm`, `nmt`, `google`, or `aws`.
- For `backend=llm`, set `server.url`, `server.model`, and `server.api_key_env`.
- For `backend=nmt`, set `nmt.server_url` and verify `/health` and `/translate`.
- For structured chat, set `text_field='messages.*.content'` and `reconstruct_messages=true`.
- For plain text, set `text_field=text` and `reconstruct_messages=false`.
- For FAITH annotation, set `faith_eval.enabled=true` and usually `faith_eval.filter_enabled=false` first.
- For FAITH filtering, confirm with the user because rows can be dropped.

For one-shot or eval-style runs, emit an execution-ready handoff before deep
exploration:

- Step decision and scope: `translate/translation`, either translation-only or translation+FAITH.
- Input policy: selected JSONL or Parquet path and excluded incompatible inputs.
- Config: inline key fields or a config path.
- Run: exact command.
- Output: expected translated artifact path.
- Env: required environment variable names only.

If model variant or source-language evidence is incomplete, state explicit
assumptions and continue with a runnable default rather than stalling. Prefer
inline config in the response before optional file writes.

## Local Files

- Contract: `src/nemotron/steps/translate/translation/step.toml`
- Runner: `src/nemotron/steps/translate/translation/step.py`
- Config: `src/nemotron/steps/translate/translation/config/default.yaml`
- Guide: `src/nemotron/steps/translate/guide.md`
- Patterns: `src/nemotron/steps/patterns/translate-training-corpus.md`

## Avoid

- Do not mix JSONL and Parquet in one input directory.
- Do not store API keys in config files.
- Do not print environment values or run env-dump commands that may expose tokens or keys.
- Do not use `merge_scores=true` with `output_mode=replaced`.
- Do not treat `skip_translated=true` as output-directory resume.
- Do not add custom chunking to `step.py` for normal use. Split huge single files before this step if needed.
- Do not silently enable FAITH filtering for training data.

## Validate

- Import check: `uv run --no-sync python -c "from nemo_curator.stages.text.experimental.translation import TranslationStage; print(TranslationStage)"`.
- Prompt resource check: verify `translate.yaml` and `faith_eval.yaml` exist under `nemo_curator.stages.text.experimental.translation.prompts`.
- LLM smoke: translate two plain-text JSONL rows with `faith_eval.enabled=false`.
- NMT smoke: call `GET /health`, then translate two rows with `backend=nmt`.
- Chat smoke: translate `messages.*.content` and verify `tool_calls[].function.arguments` remains valid JSON.
- If a command fails with CLI argument errors, return to the documented step command template before retrying.
- If file validation fails with `FileNotFoundError`, re-check actual output paths and validate only existing files.

## Runtime Prerequisites

- Runtime dependencies must include parser/config basics such as `toml` and `pyyaml`.
- If an eval container misses these packages, report the environment blocker
  and still provide a complete handoff.
- Do not end with blocker-only output when a command template and expected
  output path can still be provided.

## Completion Checklist

Before ending a response, ensure these are explicitly present:

1. Runnable config evidence.
2. `Run` command and expected `Output` path.
3. Required `Env` variable names, never values.
4. For FAITH runs, `faith_eval.enabled=true` and expected FAITH outputs.
5. Any environment/runtime blocker plus the workaround-ready command.
