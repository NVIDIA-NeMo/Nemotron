# Corpus Translation + FAITH Scoring

This step wraps NeMo Curator's reader -> `TranslationStage` -> writer pipeline.
It should stay a thin wrapper around Curator; do not generate custom chunking or
pandas processing unless a single huge input file needs a one-off preprocessing
stage.

```{step-toml} src/nemotron/steps/translate/translation/step.toml
```

## Agent Checklist

- Ask for `source_language` and `target_language` explicitly.
- Ask for `input_path`, `input_format`, and the field path to translate.
- Choose one backend: `llm`, `nmt`, `google`, or `aws`.
- For `llm`, set `server.url`, `server.model`, and `server.api_key_env` or `server.api_key`.
- For `nmt`, set `nmt.server_url`; tune `batch_size`, `timeout`, and concurrency only if needed.
- For `google`, use environment credentials and set `api_version`, `project_id` for v3, and `location`.
- For `aws`, use environment credentials or an IAM role and set `region`.
- If FAITH is enabled, configure the LLM `server` fields even when translation uses a non-LLM backend.
- Keep `output_mode=both` when users need auditability of translated fields and metadata.

## CLI

Run the step directly:

```bash
nemotron steps translation \
  input_path=/path/to/source.jsonl \
  output_dir=/path/to/translated \
  source_language=en \
  target_language=hi
```

Use `-c` or `--config` to pass a config file or config name from the step's
`config/` directory. The CLI currently supports local execution only.

## Reference Implementation

```{literalinclude} ../../../../src/nemotron/steps/translate/translation/step.py
:language: python
:caption: step.py
```

## Starter Config

```{literalinclude} ../../../../src/nemotron/steps/translate/translation/config/default.yaml
:language: yaml
:caption: config/default.yaml
```
