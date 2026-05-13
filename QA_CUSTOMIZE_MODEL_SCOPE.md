# QA Runbook: Model Customization Playbook RC1

## Purpose

This runbook tells QA how to validate the Model Customization Playbook RC1 workflows as a user would use them.

For every required module, QA should run one direct CLI workflow and one agent-driven workflow. Compare the generated commands, configs, artifacts, logs, and result summaries against the expected behavior in this document.

## Scope

In scope modules:

- General setup
- Translation
- BYOB
- Training
- SDG
- Airgap
- Lepton testing
- Evaluation

Out of scope:

- Full model training to convergence.
- Public benchmark leaderboard validation.
- Production deployment hardening outside the Airgap module.
- Vendor account provisioning, cloud quota, and external service uptime.
- Curator internals, except validating that Nemotron installs and calls a Curator build with experimental translation support.

## Runbook Rules

- Run commands from the repo root.
- Use a fresh `QA_ROOT` for each QA pass.
- If a backend, credential, GPU, Lepton workspace, Docker daemon, or external model endpoint is unavailable, mark that case `BLOCKED` and record the exact missing prerequisite.
- Do not edit checked-in configs for QA. Copy configs to `QA_ROOT`, patch the copies, and run with those copies.
- Do not treat tiny runs as model-quality evidence. Tiny runs validate plumbing, schemas, artifact paths, and executor behavior.
- Capture stdout/stderr, config paths, output artifact paths, row counts, job IDs, and redacted credential evidence for every case.

## Common Setup

### Environment

```bash
export QA_RUN_ID="${QA_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
export QA_ROOT="${QA_ROOT:-/tmp/nemotron-qa-$QA_RUN_ID}"
mkdir -p "$QA_ROOT"/{logs,metadata,artifacts}

export TRANSLATION_MODEL="${TRANSLATION_MODEL:-openai/gpt-oss-120b}"
export FAITH_MODEL="${FAITH_MODEL:-$TRANSLATION_MODEL}"
export TRANSLATION_BASE_URL="${TRANSLATION_BASE_URL:-https://integrate.api.nvidia.com/v1}"
export NMT_SERVER_URL="${NMT_SERVER_URL:-http://localhost:5000}"

export BYOB_MODEL="${BYOB_MODEL:-nvidia/nemotron-3-nano-30b-a3b}"
export SDG_MODEL="${SDG_MODEL:-nvidia/nemotron-3-nano-30b-a3b}"
export EVAL_MODEL_ID="${EVAL_MODEL_ID:-megatron_model}"
export EVAL_URL="${EVAL_URL:-http://0.0.0.0:8080/v1/completions/}"
export EVAL_TOKENIZER="${EVAL_TOKENIZER:-/path/to/checkpoint/tokenizer}"

# Required on the remote QA machine used for this run. Ray 2.55 can otherwise
# propagate uv runtime state to workers and start workers in an incomplete env.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV="${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-0}"
```

Credential variables used by the runbook:

```bash
# Set these only when running real hosted workflows.
export NVIDIA_API_KEY="${NVIDIA_API_KEY:-}"
export NGC_API_KEY="${NGC_API_KEY:-$NVIDIA_API_KEY}"
export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
```

### SETUP-001 Install Base Environment And Discover CLI

Prerequisites: `uv` is available and commands are run from the repo root.

```bash
uv sync
uv run nemotron --help
uv run nemotron steps --help
uv run nemotron steps list
uv run nemotron steps show --help
uv run nemotron steps run --help
uv run nemotron data --help
uv run nemotron byob --help
```

Success criteria:

- `uv sync` exits 0.
- All listed help and discovery commands exit 0.
- Help output is captured for `nemotron`, `nemotron steps`, `nemotron data`, and `nemotron byob`.

Evidence to collect: command logs, CLI help snippets, and final exit status.

### SETUP-002 Install Optional Workflow Extras

Prerequisites: Network access to package indexes and Git dependencies.

```bash
uv sync --extra translation
uv sync --extra byob
uv sync --extra data-sdg
uv sync --group run

# Until the Curator package-data change is available from a released package,
# validate translation against an editable Curator checkout so prompt YAML files
# are available from site-packages.
export CURATOR_ROOT="${CURATOR_ROOT:-$QA_ROOT/curator-main}"
if [ ! -d "$CURATOR_ROOT/.git" ]; then
  git clone https://github.com/NVIDIA-NeMo/Curator.git "$CURATOR_ROOT"
fi
git -C "$CURATOR_ROOT" fetch origin main
git -C "$CURATOR_ROOT" checkout main
git -C "$CURATOR_ROOT" pull --ff-only origin main
uv pip install -e "$CURATOR_ROOT[translation_all]"

uv run --no-sync python - <<'PY'
import importlib.resources as ir

prompt = ir.files("nemo_curator.stages.text.experimental.translation.prompts").joinpath("translate.yaml")
assert prompt.is_file(), prompt
print(prompt)
PY
```

After this setup, use `uv run --no-sync ...` for validation commands in this
runbook. This avoids an implicit `uv` resync replacing the editable Curator
install during the same QA pass.

Success criteria:

- Translation, BYOB, SDG, and run dependencies install successfully.
- Editable Curator is installed and `translate.yaml` is importable through package resources.
- If package indexes or Git dependencies are unavailable, this test is marked `BLOCKED` with the exact failed dependency or network error.

Evidence to collect: install logs and final exit status.

### SETUP-003 Snapshot Step Metadata

Prerequisites: Base environment from `SETUP-001`.

```bash
for STEP in \
  translate/translation \
  byob \
  data_prep/sft_packing \
  data_prep/pretrain_prep \
  data_prep/rl_prep \
  sft/automodel \
  sft/megatron_bridge \
  peft/automodel \
  peft/megatron_bridge \
  pretrain/automodel \
  pretrain/megatron_bridge \
  rl/nemo_rl/dpo \
  rl/nemo_rl/rlvr \
  rl/nemo_rl/rlhf \
  optimize/modelopt/quantize \
  optimize/modelopt/prune \
  optimize/modelopt/distill \
  sdg/data_designer \
  eval/model_eval \
  env/env_toml
do
  uv run --no-sync nemotron steps show "$STEP" --json > "$QA_ROOT/metadata/${STEP//\//_}.json"
done
```

Success criteria:

- Every command exits successfully.
- Metadata files are valid JSON.
- Step IDs match the paths shown in the command.
- No command requires a model credential just to show metadata.

Evidence to collect: metadata JSON files under `$QA_ROOT/metadata` and command logs.

## Translation

Required translation QA should focus on `output_mode=replaced`, because that is the user-facing workflow where
the output dataset is ready for downstream customization. `output_mode=raw` and `output_mode=both` are useful
for audit/debug workflows, but they are optional coverage and should not be required for release signoff unless
the feature under test explicitly needs translation metadata.

### Test Data Setup

```bash
export TR_ROOT="$QA_ROOT/translation"
mkdir -p "$TR_ROOT/news_en" "$TR_ROOT/mixed_dir"

cat > "$TR_ROOT/news_en/shard_0001.jsonl" <<'EOF'
{"text":"The central bank raised its benchmark interest rate by a quarter point today."}
{"text":"Heavy monsoon rains caused widespread flooding across the western districts."}
EOF

cat > "$TR_ROOT/chat_en.jsonl" <<'EOF'
{"messages":[{"role":"user","content":"What is the capital of France?"},{"role":"assistant","content":"The capital of France is Paris."}]}
{"messages":[{"role":"user","content":"Book a flight from Boston to Seattle next Tuesday."},{"role":"assistant","content":"Searching available flights now.","tool_calls":[{"id":"call_1","type":"function","function":{"name":"search_flights","arguments":"{\"from\":\"BOS\",\"to\":\"SEA\",\"date\":\"2026-05-19\"}"}}]}]}
EOF

cat > "$TR_ROOT/chat_code_en.jsonl" <<'EOF'
{"messages":[{"role":"user","content":"Show me a Python snippet that reads a CSV."},{"role":"assistant","content":"```python\nimport pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.head())\n```"}]}
{"messages":[{"role":"user","content":"Call the weather tool for London."},{"role":"assistant","content":"Issuing the call.","tool_calls":[{"id":"call_w","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"London\",\"units\":\"metric\"}"}}]}]}
EOF

uv run --no-sync python - <<'PY'
from pathlib import Path
import os
import pandas as pd

root = Path(os.environ["TR_ROOT"])
df = pd.DataFrame(
    [
        {"text": "A solar startup announced a new battery pilot in Nevada."},
        {"text": "Researchers published a study on multilingual model evaluation."},
    ]
)
df.to_parquet(root / "news_en.parquet", index=False)
PY

cp "$TR_ROOT/news_en/shard_0001.jsonl" "$TR_ROOT/mixed_dir/shard_0001.jsonl"
cp "$TR_ROOT/news_en.parquet" "$TR_ROOT/mixed_dir/shard_0002.parquet"
```

### TR-001 Discover Translation Step

Prerequisites: Translation extra installed with `uv sync --extra translation`; test data from `Test Data Setup`.

```bash
uv run --no-sync nemotron steps show translate/translation
uv run --no-sync nemotron steps run translate/translation --help
uv run --no-sync nemotron steps translation --help
uv run --no-sync python -c "from nemo_curator.stages.text.experimental.translation import TranslationStage; print(TranslationStage)"
```

Success criteria:

- Step metadata and CLI help are available.
- `source_language` and `target_language` are explicit.
- `TranslationStage` imports from `nemo_curator.stages.text.experimental.translation`.

Evidence to collect: CLI logs, metadata/help output, and import output.

### TR-002 Translate JSONL Text Records With Hosted LLM

Prerequisites: `NVIDIA_API_KEY` or configured hosted LLM credential; live `TRANSLATION_MODEL`.

```bash
: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY for hosted LLM translation}"

uv run --no-sync nemotron steps translation \
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

Validate output:

```bash
uv run --no-sync python - <<'PY'
from pathlib import Path
import json
import os

root = Path(os.environ["TR_ROOT"]) / "out_llm_hi"
files = sorted(root.rglob("*.jsonl"))
assert files, f"No JSONL output found under {root}"
rows = []
for path in files:
    rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
assert rows, "No rows written"
assert any("translated_text" in row or "text" in row for row in rows), rows[0]
print({"files": len(files), "rows": len(rows), "sample": rows[0]})
PY
```

Success criteria:

- Command exits 0.
- Output JSONL shards exist.
- Row count matches input when FAITH filtering is disabled.
- Translated text is present.
- Logs do not print API keys.

Evidence to collect: output JSONL files, row-count validation output, sampled row, and redacted logs.

### TR-003 Translate Structured Chat Records

Prerequisites: Hosted LLM backend available; chat test data from `Test Data Setup`.

```bash
uv run --no-sync nemotron steps translation \
  input_path="$TR_ROOT/chat_code_en.jsonl" \
  output_dir="$TR_ROOT/out_chat_hi" \
  source_language=en \
  target_language=hi \
  backend=llm \
  text_field='messages.*.content' \
  output_mode=replaced \
  merge_scores=false \
  reconstruct_messages=true \
  segmentation_mode=coarse \
  faith_eval.enabled=false \
  server.url="$TRANSLATION_BASE_URL" \
  server.model="$TRANSLATION_MODEL" \
  server.api_key_env=NVIDIA_API_KEY
```

Validate output:

```bash
uv run --no-sync python - <<'PY'
from pathlib import Path
import json
import os

root = Path(os.environ["TR_ROOT"]) / "out_chat_hi"
rows = []
for path in sorted(root.rglob("*.jsonl")):
    rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
assert rows, "No rows written"
assert any("messages" in row for row in rows), rows[0]
for row in rows:
    for msg in row.get("messages", []):
        if "tool_calls" in msg:
            for call in msg["tool_calls"]:
                json.loads(call["function"]["arguments"])
print({"rows": len(rows), "tool_json_ok": True})
PY
```

Success criteria:

- Command exits 0.
- Tool-call JSON remains parseable.
- Code fences are not corrupted.
- Natural-language message content is translated in the output `messages` field.
- Output rows exist and preserve the expected chat structure.

Evidence to collect: output JSONL, JSON parse validation output, sampled translated row, and redacted logs.

### TR-004 Run FAITH Annotation Without Filtering Rows

Prerequisites: Hosted LLM backend available for `FAITH_MODEL`.

```bash
uv run --no-sync nemotron steps translation \
  input_path="$TR_ROOT/news_en" \
  output_dir="$TR_ROOT/out_faith_annotated" \
  source_language=en \
  target_language=hi \
  backend=llm \
  text_field=text \
  output_mode=replaced \
  reconstruct_messages=false \
  merge_scores=false \
  faith_eval.enabled=true \
  faith_eval.filter_enabled=false \
  faith_eval.model_name="$FAITH_MODEL" \
  server.url="$TRANSLATION_BASE_URL" \
  server.model="$TRANSLATION_MODEL" \
  server.api_key_env=NVIDIA_API_KEY
```

Success criteria:

- Command exits 0.
- Output rows are retained.
- FAITH scoring runs without row filtering.
- Logs do not print API keys.

Evidence to collect: output files, score-field inspection, row count, and redacted logs.

### TR-005 Translate With NMT Backend

Prerequisites: Reachable `NMT_SERVER_URL` implementing the expected translation endpoint.

```bash
: "${NMT_SERVER_URL:?Set NMT_SERVER_URL to an NMT service endpoint}"

uv run --no-sync nemotron steps translation \
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

Success criteria:

- Translation succeeds without an LLM API key.
- Backend traffic goes to the configured NMT server.
- Output rows exist and are readable.

Evidence to collect: command logs, output files, sampled rows, and service logs if available.

### TR-006 Translate Parquet Input And Write Parquet Output

Prerequisites: Hosted LLM backend available; `pandas` and `pyarrow` available.

```bash
uv run --no-sync nemotron steps translation \
  input_path="$TR_ROOT/news_en.parquet" \
  output_dir="$TR_ROOT/out_parquet_hi" \
  source_language=en \
  target_language=hi \
  input_format=parquet \
  output_format=parquet \
  backend=llm \
  text_field=text \
  output_mode=replaced \
  merge_scores=false \
  reconstruct_messages=false \
  faith_eval.enabled=false \
  server.url="$TRANSLATION_BASE_URL" \
  server.model="$TRANSLATION_MODEL" \
  server.api_key_env=NVIDIA_API_KEY

uv run --no-sync python - <<'PY'
from pathlib import Path
import pandas as pd
import os

root = Path(os.environ["TR_ROOT"]) / "out_parquet_hi"
files = sorted(root.rglob("*.parquet"))
assert files, f"No Parquet output found under {root}"
df = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
assert len(df) == 2, len(df)
print(df.head().to_dict(orient="records"))
PY
```

Success criteria:

- Command exits 0.
- Parquet output is readable.
- Row count is preserved when filtering is disabled.

Evidence to collect: Parquet validation output, sampled rows, and redacted logs.

### TR-007 Resume Or Skip Already Translated Records

Prerequisites: Prior translated output from `TR-002` or equivalent.

```bash
uv run --no-sync nemotron steps translation \
  input_path="$TR_ROOT/out_llm_hi" \
  output_dir="$TR_ROOT/out_resume_hi" \
  source_language=en \
  target_language=hi \
  backend=llm \
  text_field=text \
  output_mode=replaced \
  merge_scores=false \
  reconstruct_messages=false \
  skip_translated=true \
  faith_eval.enabled=false \
  server.url="$TRANSLATION_BASE_URL" \
  server.model="$TRANSLATION_MODEL" \
  server.api_key_env=NVIDIA_API_KEY
```

Success criteria:

- Command exits 0.
- Records already containing the configured translation column are not retranslated where supported by the schema.
- Output remains readable.

Evidence to collect: resume command logs and output sample.

### TR-008 Reject Mixed JSONL And Parquet Input Directory

Prerequisites: Mixed-format test directory from `Test Data Setup`.

```bash
if uv run --no-sync nemotron steps translation \
  input_path="$TR_ROOT/mixed_dir" \
  output_dir="$TR_ROOT/out_mixed_should_fail" \
  source_language=en \
  target_language=hi \
  backend=llm \
  text_field=text \
  output_mode=replaced \
  merge_scores=false \
  faith_eval.enabled=false \
  server.url="$TRANSLATION_BASE_URL" \
  server.model="$TRANSLATION_MODEL" \
  server.api_key_env=NVIDIA_API_KEY
then
  echo "FAIL: mixed-format translation unexpectedly succeeded"
else
  echo "PASS: mixed-format translation failed as expected"
fi
```

Success criteria:

- Mixed JSONL/Parquet input fails with a clear format error.
- The command does not report misleading partial success.

Evidence to collect: failure log, exit status, and error message.

### TR-009 Agent-Driven Translation Workflow

Prerequisites: Agent environment available; chat test data from `Test Data Setup`; hosted LLM credential if running for real.

Ask the agent:

```text
Translate /tmp/nemotron-qa-<run-id>/translation/chat_code_en.jsonl from English to Hindi using the Nemotron translation step. Preserve tool-call JSON, do not filter rows, and show me the exact command before running it.
```

Success criteria:

- Uses `nemotron steps translation` for local execution.
- Uses `text_field='messages.*.content'`.
- Sets `source_language=en` and `target_language=hi`.
- Keeps credentials in environment variables.
- Validates output JSON and row count.

Evidence to collect: agent transcript, generated command, validation output, and output artifact paths.

## BYOB

### Test Data And Config Setup

```bash
export BYOB_ROOT="$QA_ROOT/byob"
mkdir -p "$BYOB_ROOT/input/maths" "$BYOB_ROOT/config" "$BYOB_ROOT/output"

cat > "$BYOB_ROOT/input/maths/overview.txt" <<'EOF'
Algebra studies symbols and the rules for manipulating them. Linear equations relate variables through addition and scalar multiplication.
Finance studies how people, firms, and governments allocate money over time. It includes saving, investing, borrowing, lending, budgeting, and risk management.
EOF

cp src/nemotron/steps/byob/config/tiny.yaml "$BYOB_ROOT/config/byob_mcq.yaml"
cp src/nemotron/steps/byob/config/translate.yaml "$BYOB_ROOT/config/byob_translate.yaml"
```

Patch the BYOB generation config:

```bash
uv run --no-sync python - <<'PY'
from pathlib import Path
import os
import yaml

root = Path(os.environ["BYOB_ROOT"])
cfg_path = root / "config" / "byob_mcq.yaml"
cfg = yaml.safe_load(cfg_path.read_text())
cfg["expt_name"] = "byob_mcq_qa"
cfg["input_dir"] = str(root / "input")
cfg["output_dir"] = str(root / "output")
cfg["language"] = "en-US"
model = {
    "alias": "qa_generation_model",
    "model": os.environ.get("BYOB_MODEL", "nvidia/nemotron-3-nano-30b-a3b"),
    "provider": "nvidia",
    "inference_parameters": {
        "max_tokens": 1024,
        "max_parallel_requests": 1,
        "temperature": 0.0,
        "top_p": 1.0,
    },
}
cfg["generation_model_config"] = model
cfg["judge_model_config"] = model
cfg["distractor_expansion_model_config"] = model
cfg["distractor_validity_model_config"] = model
cfg["filtering_model_configs"] = {
    "hallucination": [{**model, "alias": "qa_hallucination"}],
    "easiness": [{**model, "alias": "qa_easiness"}],
}
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(cfg_path)
PY
```

Create a standalone benchmark for translation-only validation:

```bash
uv run --no-sync python - <<'PY'
from pathlib import Path
import os
import pandas as pd

root = Path(os.environ["BYOB_ROOT"])
rows = [
    {
        "question_id": "finance-001",
        "question": "What does budgeting help manage?",
        "options": ["Weather", "Money", "Photos", "Music"],
        "answer_index": 1,
        "answer": "Money",
        "cot_content": "Budgeting tracks income and spending.",
        "src": "maths/overview.txt",
        "category": "maths",
    }
]
path = root / "output" / "translation_input" / "benchmark.parquet"
path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_parquet(path, index=False)
print(path)
PY
```

Patch the BYOB translation config:

```bash
uv run --no-sync python - <<'PY'
from pathlib import Path
import os
import yaml

root = Path(os.environ["BYOB_ROOT"])
cfg_path = root / "config" / "byob_translate.yaml"
cfg = yaml.safe_load(cfg_path.read_text())
cfg["expt_name"] = "byob_mcq_translation_qa"
cfg["dataset_path"] = str(root / "output" / "translation_input" / "benchmark.parquet")
cfg["output_dir"] = str(root / "output")
cfg["source_language"] = "en-US"
cfg["target_language"] = "hi-IN"
cfg["remove_low_quality"] = False
cfg["translation_model_config"]["params"]["model"] = os.environ.get("TRANSLATION_MODEL", "openai/gpt-oss-120b")
cfg["translation_model_config"]["params"]["provider"] = "nvidia"
cfg["translation_model_config"]["params"]["api_key_env"] = "NGC_API_KEY"
cfg["translation_model_config"]["params"]["base_url"] = os.environ.get(
    "TRANSLATION_BASE_URL",
    "https://integrate.api.nvidia.com/v1",
)
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(cfg_path)
PY
```

### BYOB-001 Discover BYOB CLI And Validator

Prerequisites: BYOB extra installed with `uv sync --extra byob`.

```bash
uv run --no-sync nemotron byob --help
uv run --no-sync nemotron byob --list-families
uv run --no-sync python src/nemotron/steps/byob/scripts/validate.py --help
```

Success criteria:

- `mcq` is listed as a family.
- Help output shows `prepare`, `generate`, `translate`, and `all`.
- Validator help exits 0.

Evidence to collect: CLI logs and family list output.

### BYOB-002 Generate MCQ Benchmark With `--stage all`

Prerequisites: `NGC_API_KEY` or `NVIDIA_API_KEY`; live generation and judge model; copied BYOB config from setup.

```bash
: "${NGC_API_KEY:?Set NGC_API_KEY or NVIDIA_API_KEY for BYOB generation}"

uv run --no-sync nemotron byob \
  --family mcq \
  --stage all \
  --config "$BYOB_ROOT/config/byob_mcq.yaml"
```

Validate benchmark artifacts:

```bash
find "$BYOB_ROOT/output" -maxdepth 4 -type f | sort

uv run --no-sync python - <<'PY'
from pathlib import Path
import os
import pandas as pd

root = Path(os.environ["BYOB_ROOT"]) / "output"
benchmarks = sorted(root.rglob("benchmark.parquet"))
assert benchmarks, f"No benchmark.parquet under {root}"
df = pd.read_parquet(benchmarks[-1])
required = {"question_id", "question", "options", "answer_index", "answer", "src", "category"}
missing = required - set(df.columns)
assert not missing, missing
assert len(df) > 0, "benchmark is empty"
print({"path": str(benchmarks[-1]), "rows": len(df), "columns": list(df.columns)})
PY
```

Success criteria:

- Command exits 0.
- `all` runs prepare followed by generate.
- Seed, raw benchmark, and final benchmark artifacts are written.
- Final benchmark schema contains MCQ fields.
- Final `benchmark.parquet` has at least one row.

Evidence to collect: artifact listing, Parquet schema validation, row count, and sampled benchmark row.

Optional stage-by-stage commands for isolating failures:

```bash
uv run --no-sync nemotron byob \
  --family mcq \
  --stage prepare \
  --config "$BYOB_ROOT/config/byob_mcq.yaml"

uv run --no-sync nemotron byob \
  --family mcq \
  --stage generate \
  --config "$BYOB_ROOT/config/byob_mcq.yaml"
```

### BYOB-003 Resume BYOB Generation From A Named Stage

Prerequisites: Outputs from `BYOB-002` or compatible stage cache.

```bash
uv run --no-sync nemotron byob \
  --family mcq \
  --stage generate \
  --skip-until JUDGEMENT \
  --config "$BYOB_ROOT/config/byob_mcq.yaml"
```

Success criteria:

- Earlier cached stages are reused.
- Later stages regenerate or validate without corrupting final schema.
- If resume is invalid for the current cache, failure clearly states why.

Evidence to collect: resume logs and artifact timestamps or paths.

### BYOB-004 Translate BYOB Benchmark

Prerequisites: BYOB translation config; hosted translation credential; benchmark parquet exists.

```bash
: "${NGC_API_KEY:?Set NGC_API_KEY or NVIDIA_API_KEY for BYOB translation}"

uv run --no-sync nemotron byob \
  --family mcq \
  --stage translate \
  --config "$BYOB_ROOT/config/byob_translate.yaml"
```

Validate translation artifacts:

```bash
find "$BYOB_ROOT/output" -maxdepth 5 -type f | sort

uv run --no-sync python - <<'PY'
from pathlib import Path
import os
import pandas as pd

root = Path(os.environ["BYOB_ROOT"]) / "output"
benchmarks = sorted(root.rglob("benchmark.parquet"))
assert benchmarks, f"No translated benchmark.parquet under {root}"
df = pd.read_parquet(benchmarks[-1])
required = {"question_id", "question", "options", "answer_index", "answer", "src", "category"}
missing = required - set(df.columns)
assert not missing, missing
assert len(df) > 0, "translated benchmark is empty"
print({"path": str(benchmarks[-1]), "rows": len(df), "columns": list(df.columns)})
PY
```

Success criteria:

- Command exits 0.
- Translated benchmark exists.
- `question_id`, options, answer, and answer index remain structurally valid.
- If `remove_low_quality=false`, row count should match the translation input.

Evidence to collect: output Parquet schema, row count, sampled translated question/options, and redacted logs.

### BYOB-005 Agent-Driven BYOB Workflow

Prerequisites: Agent environment available; BYOB test data and credentials if running real generation.

Ask the agent:

```text
Use BYOB to create a tiny MCQ benchmark from /tmp/nemotron-qa-<run-id>/byob/input and then translate the generated benchmark to Hindi. Use copied configs under /tmp/nemotron-qa-<run-id>/byob/config and do not edit repo configs.
```

Success criteria:

- Uses `nemotron byob --family mcq --stage all` for prepare+generate, or explicitly explains why it is running `prepare` and `generate` separately.
- Uses `nemotron byob --family mcq --stage translate`.
- Patches config copies only.
- Validates final Parquet schema.

Evidence to collect: agent transcript, generated configs, command logs, validation output, and artifact paths.

## Training

### Test Data Setup

```bash
export TRN_ROOT="$QA_ROOT/training"
mkdir -p "$TRN_ROOT/data" "$TRN_ROOT/output"

cat > "$TRN_ROOT/data/sft_chat.jsonl" <<'EOF'
{"messages":[{"role":"user","content":"Summarize what a budget is."},{"role":"assistant","content":"A budget is a plan for income, spending, saving, and borrowing over time."}]}
{"messages":[{"role":"user","content":"Give one reason to diversify investments."},{"role":"assistant","content":"Diversification spreads risk across assets instead of concentrating it in one place."}]}
EOF

cat > "$TRN_ROOT/data/pretrain.jsonl" <<'EOF'
{"text":"Finance studies how people and organizations allocate resources under uncertainty."}
{"text":"Risk management identifies, measures, and mitigates sources of financial loss."}
EOF

cat > "$TRN_ROOT/data/preferences.jsonl" <<'EOF'
{"prompt":"Explain diversification in one sentence.","chosen":"Diversification spreads investment risk across multiple assets.","rejected":"Diversification means buying only one stock."}
{"prompt":"What is budgeting?","chosen":"Budgeting plans income and expenses over a period of time.","rejected":"Budgeting predicts tomorrow's weather."}
EOF

cat > "$TRN_ROOT/data/sft_blend.json" <<EOF
{"datasets":[{"name":"qa_sft_chat","path":"$TRN_ROOT/data/sft_chat.jsonl"}]}
EOF

cat > "$TRN_ROOT/data/pretrain_blend.json" <<EOF
{"datasets":[{"name":"qa_pretrain_text","path":"$TRN_ROOT/data/pretrain.jsonl","text_field":"text"}]}
EOF

cat > "$TRN_ROOT/data/rl_blend.json" <<EOF
{"datasets":[{"name":"qa_preferences","path":"$TRN_ROOT/data/preferences.jsonl"}]}
EOF
```

### TRAIN-001 Discover Training And Data-Prep Step Metadata

Prerequisites: Base environment from `SETUP-001`; training test data setup completed.

```bash
uv run --no-sync nemotron steps list --json > "$TRN_ROOT/steps.json"
uv run --no-sync nemotron steps show data_prep/sft_packing --json > "$TRN_ROOT/data_prep_sft_packing.json"
uv run --no-sync nemotron steps show data_prep/pretrain_prep --json > "$TRN_ROOT/data_prep_pretrain_prep.json"
uv run --no-sync nemotron steps show data_prep/rl_prep --json > "$TRN_ROOT/data_prep_rl_prep.json"
uv run --no-sync nemotron steps show sft/automodel --json > "$TRN_ROOT/sft_automodel.json"
uv run --no-sync nemotron steps show sft/megatron_bridge --json > "$TRN_ROOT/sft_megatron_bridge.json"
uv run --no-sync nemotron steps show peft/automodel --json > "$TRN_ROOT/peft_automodel.json"
uv run --no-sync nemotron steps show peft/megatron_bridge --json > "$TRN_ROOT/peft_megatron_bridge.json"
uv run --no-sync nemotron steps show pretrain/automodel --json > "$TRN_ROOT/pretrain_automodel.json"
uv run --no-sync nemotron steps show pretrain/megatron_bridge --json > "$TRN_ROOT/pretrain_megatron_bridge.json"
uv run --no-sync nemotron steps show rl/nemo_rl/dpo --json > "$TRN_ROOT/rl_dpo.json"
uv run --no-sync nemotron steps show rl/nemo_rl/rlvr --json > "$TRN_ROOT/rl_rlvr.json"
uv run --no-sync nemotron steps show rl/nemo_rl/rlhf --json > "$TRN_ROOT/rl_rlhf.json"
uv run --no-sync nemotron steps show optimize/modelopt/quantize --json > "$TRN_ROOT/modelopt_quantize.json"
uv run --no-sync nemotron steps show optimize/modelopt/prune --json > "$TRN_ROOT/modelopt_prune.json"
uv run --no-sync nemotron steps show optimize/modelopt/distill --json > "$TRN_ROOT/modelopt_distill.json"
```

Success criteria:

- All listed steps are discoverable.
- All metadata files are valid JSON.
- Requested step IDs match the emitted metadata.

Evidence to collect: metadata JSON files and command logs.

### TRAIN-002 Confirm Tiny Data-Prep Inputs

Prerequisites: Base environment from `SETUP-001`.

Training and data-prep runtime validation is intentionally not local. The
bundled tiny data-prep configs carry in-step `blend_tiny.json` files, so QA does
not need the driver machine to write custom blend files into Lepton shared
storage before submitting the runtime smoke. Actual data prep and training run
through the Lepton executor in `LEP-003`.

```bash
test -s src/nemotron/steps/data_prep/sft_packing/data/blend_tiny.json
test -s src/nemotron/steps/data_prep/pretrain_prep/data/blend_tiny.json
test -s src/nemotron/steps/data_prep/rl_prep/data/blend_tiny.json
```

Success criteria:

- Bundled tiny blend files exist for SFT, pretrain, and RL data prep.
- No local shared-storage mount is required for this setup case.

Evidence to collect: command logs and blend file paths.

### TRAIN-003 Agent-Driven Training Workflow

Prerequisites: Agent environment available; tiny training data.

Ask the agent:

```text
Prepare a tiny SFT workflow using Lepton. Use the bundled tiny data-prep config, run data_prep/sft_packing on Lepton first, then submit one actual one-GPU SFT training smoke on Lepton.
```

Success criteria:

- Uses `data_prep/sft_packing` before a packed-data Megatron-Bridge SFT smoke.
- Uses the one-GPU AutoModel smoke command when the selected Lepton profile has one A100.
- Uses `--batch lepton` and submits real Lepton jobs.
- Keeps models, datasets, checkpoints, and generated outputs on shared storage.

Evidence to collect: agent transcript, generated command plan, Lepton job IDs, and output artifact paths.

## SDG

### SDG-001 Create SDG Test Data And Config Copy

Prerequisites: Base environment from `SETUP-001`.

```bash
export SDG_ROOT="$QA_ROOT/sdg"
mkdir -p "$SDG_ROOT/config" "$SDG_ROOT/data" "$SDG_ROOT/output"

cat > "$SDG_ROOT/data/topic_seeds.jsonl" <<'EOF'
{"topic":"budgeting basics"}
{"topic":"investment diversification"}
EOF

cp src/nemotron/steps/sdg/data_designer/config/tiny.yaml "$SDG_ROOT/config/sdg_tiny.yaml"
cp src/nemotron/steps/sdg/data_designer/config/default.yaml "$SDG_ROOT/config/default.yaml"
```

Patch the copied SDG config:

```bash
uv run --no-sync python - <<'PY'
from pathlib import Path
import os
import yaml

root = Path(os.environ["SDG_ROOT"])
cfg_path = root / "config" / "sdg_tiny.yaml"
cfg = yaml.safe_load(cfg_path.read_text())
cfg["output_dir"] = str(root / "output")
cfg["output_path"] = str(root / "output" / "sft-tiny.jsonl")
cfg["num_records"] = 5
cfg["seed_dataset"] = {
    "path": str(root / "data" / "topic_seeds.jsonl"),
    "strategy": "ordered",
    "fields": ["topic"],
}
for model in cfg.get("models", []):
    model["model"] = os.environ.get("SDG_MODEL", "nvidia/nemotron-3-nano-30b-a3b")
    model["provider"] = "nvidia"
    model["skip_health_check"] = True
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(cfg_path)
PY
```

Success criteria:

- Test seed data is created under `$QA_ROOT`.
- SDG config is copied under `$QA_ROOT` and patched there.
- Checked-in configs are not modified.

Evidence to collect: config path, test data path, and copied config contents.

### SDG-002 Discover And Run SDG Workflow

Prerequisites: `data-sdg` dependencies; hosted model credential if running real generation.

```bash
uv run --no-sync nemotron steps list --category sdg --json
uv run --no-sync nemotron steps show sdg/data_designer --json
: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY for SDG generation}"

uv run --no-sync nemotron steps run sdg/data_designer -c "$SDG_ROOT/config/sdg_tiny.yaml"
```

Validate output:

```bash
uv run --no-sync python - <<'PY'
from pathlib import Path
import json
import os

path = Path(os.environ["SDG_ROOT"]) / "output" / "sft-tiny.jsonl"
assert path.exists(), path
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
assert rows, "No SDG rows"
assert all("messages" in row for row in rows), rows[0]
print({"rows": len(rows), "sample": rows[0]})
PY
```

Success criteria:

- Real run writes OpenAI-style `messages`.
- Invalid endpoint/key failures identify the missing credential or service.

Evidence to collect: CLI logs, generated JSONL path, row count, and sampled output row.

### SDG-003 Agent-Driven SDG Workflow

Prerequisites: Agent environment available; SDG task requirements.

Ask the agent:

```text
Generate five synthetic SFT chat examples about budgeting and diversification using sdg/data_designer. Use a copied config under /tmp/nemotron-qa-<run-id>/sdg/config and validate that the output is JSONL with messages.
```

Success criteria:

- Uses `sdg/data_designer`.
- Copies and patches config instead of editing repo files.
- Runs real generation or records a concrete endpoint/credential blocker.
- Validates output JSONL schema.

Evidence to collect: agent transcript, generated config, logs, and artifacts.

## Airgap

### AIR-001 Generate Airgap Plan

Prerequisites: Airgap runner config available; Docker is not required for metadata planning mode.

Airgap planning is the default command behavior. Pass `--execute` only when QA is intentionally building and saving images.

```bash
export AIRGAP_ROOT="$QA_ROOT/airgap"
mkdir -p "$AIRGAP_ROOT"

uv run --no-sync python deploy/nemotron-customizer/airgap/runner.py \
  --config deploy/nemotron-customizer/airgap/airgap.yaml
```

Success criteria:

- Plan command exits 0.
- Generated plan includes scoped targets and required dependency categories.
- No image build starts during metadata planning.

Evidence to collect: plan output and target list.

### AIR-002 Validate Selected Airgap Targets

Prerequisites: Required local tooling and images for validation, or a recorded blocker.

```bash

uv run --no-sync python deploy/nemotron-customizer/airgap/runner.py \
  --config deploy/nemotron-customizer/airgap/airgap.yaml \
  --stage validate \
  --target sdg/data_designer:tiny

uv run --no-sync python deploy/nemotron-customizer/airgap/runner.py \
  --config deploy/nemotron-customizer/airgap/airgap.yaml \
  --stage validate \
  --target data_prep/sft_packing:tiny \
  --target sft/megatron_bridge:tiny

uv run --no-sync python deploy/nemotron-customizer/airgap/runner.py \
  --config deploy/nemotron-customizer/airgap/airgap.yaml \
  --stage validate \
  --target sdg/data_designer:tiny \
  --target data_prep/sft_packing:tiny \
  --target sft/megatron_bridge:tiny
```

Dependency discovery plan:

```bash
uv run --no-sync python deploy/nemotron-customizer/airgap/runner.py \
  --config deploy/nemotron-customizer/airgap/airgap.yaml \
  --stage validate \
  --stage discover-execution-deps \
  --target sdg/data_designer:tiny \
  --target sft/megatron_bridge:tiny
```

Optional connected-machine image build:

```bash
uv run --no-sync python deploy/nemotron-customizer/airgap/runner.py \
  --config deploy/nemotron-customizer/airgap/airgap.yaml \
  --execute \
  --target sdg/data_designer:tiny \
  --target sft/megatron_bridge:tiny
```

Success criteria:

- Planning and validation commands validate selected targets without building images.
- Selected targets and dependencies are printed.
- Models, datasets, checkpoints, and customer files remain external to images.
- Execute mode writes outputs under `deploy/nemotron-customizer/airgap/out/` if Docker and registry access are available.
- Missing image, Docker, registry, or site tooling is recorded as `BLOCKED` with the exact missing prerequisite.

Evidence to collect: validation logs, dependency plan output, and generated bundle or metadata paths when execute mode is used.

### AIR-003 Agent-Driven Airgap Workflow

Prerequisites: Agent environment available; target list from user.

Ask the agent:

```text
Create an airgap plan for SDG plus Megatron-Bridge SFT. Run target validation and explain which assets remain outside the images.
```

Success criteria:

- Uses `deploy/nemotron-customizer/airgap/runner.py`.
- Uses `--execute` only if QA explicitly requests image builds.
- Selects scoped targets.
- States that models, datasets, checkpoints, and customer data remain on persistent storage.

Evidence to collect: agent transcript, plan output, validation logs, and selected target list.

## Lepton Testing

### LEP-001 Create Lepton Env File

Prerequisites: Lepton workspace credential available.

```bash
export LEPTON_ROOT="$QA_ROOT/lepton"
mkdir -p "$LEPTON_ROOT"

export LEPTON_WORKSPACE="${LEPTON_WORKSPACE:?Set Lepton workspace name}"
export LEPTON_API_KEY="${LEPTON_API_KEY:?Set Lepton API key}"
uvx --from leptonai lep login -c "$LEPTON_WORKSPACE:$LEPTON_API_KEY"

cat > "$LEPTON_ROOT/env.lepton.toml" <<'EOF'
[lepton]
executor = "lepton"
container_image = "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
node_group = "az-sat-lepton-001"
resource_shape = "gpu.a100-80gb"
nemo_run_dir = "/mnt/lustre-shared/nemo-run"
nodes = 1
gpus_per_node = 1
image_pull_secrets = []
pip_extras = ["cosmos-xenna"]
ray_version = "2.48.0"
# NOTE: OTEL_EXPORTER_OTLP_ENDPOINT intentionally omitted. Setting it to ""
# can crash Ray's OpenTelemetryMetricRecorder during actor init.
# Raylet-level tuning: avoid prestarting 96 workers at once on 96-CPU nodes.
env_vars = { HF_TOKEN = "${oc.env:HF_TOKEN,''}", HF_HOME = "${oc.env:HF_HOME,/mnt/lustre-shared/hf}", RAY_GRAFANA_IFRAME_HOST = "", RAY_DEDUP_LOGS = "0", RAY_worker_maximum_startup_concurrency = "16", RAY_num_prestart_python_workers = "16", RAY_worker_register_timeout_seconds = "1200", WANDB_API_KEY = "${oc.env:WANDB_API_KEY,''}" }
pre_launch_commands = [
  "export RAY_worker_maximum_startup_concurrency=16",
  "export RAY_num_prestart_python_workers=16",
]
mounts = [
  { path = "/sovereign-ai-playbook/", mount_path = "/mnt/lustre-shared", from = "node-nfs:amlfs" },
]

[lepton_sft_automodel]
extends = "lepton"
container_image = "nvcr.io/nvidia/nemo-automodel:26.04"
startup_commands = [
  "python -m pip install --quiet --break-system-packages omegaconf",
]
EOF

export NEMOTRON_ENV_FILE="$LEPTON_ROOT/env.lepton.toml"

uvx --from leptonai lep node list
uvx --from leptonai lep node storage -ng az-sat-lepton-001
uv run --no-sync python - <<'PY'
import os
import tomllib
from pathlib import Path

path = Path(os.environ["NEMOTRON_ENV_FILE"])
parsed = tomllib.loads(path.read_text())
assert "lepton" in parsed, parsed.keys()
assert "lepton_sft_automodel" in parsed, parsed.keys()
assert parsed["lepton"]["mounts"][0]["mount_path"] == "/mnt/lustre-shared"
print(path)
PY
```

Success criteria:

- `env.lepton.toml` is generated.
- Lepton login succeeds.
- Node group `az-sat-lepton-001` and storage `node-nfs:amlfs` are visible to the workspace.
- Profile `[lepton]` uses `/sovereign-ai-playbook/` mounted at `/mnt/lustre-shared`.
- Profile `[lepton_sft_automodel]` uses the AutoModel container image.
- Secrets are not written in plain text unless explicitly intended by the profile.

Evidence to collect: generated env file path and redacted content sample.

### LEP-002 Validate Lepton Profiles For Scoped Steps

Prerequisites: Env file from `LEP-001`.

```bash
test -s "$NEMOTRON_ENV_FILE"
grep -E "^\[(lepton|lepton_sft_automodel)\]" "$NEMOTRON_ENV_FILE"
uvx --from leptonai lep workspace list
```

Success criteria:

- Required Lepton profile exists in `env.lepton.toml`.
- AutoModel-specific Lepton profile exists in `env.lepton.toml`.
- Lepton workspace listing succeeds.
- This profile-validation case does not replace the actual Lepton runtime cases in `LEP-003`.

Evidence to collect: env file path, profile grep output, and Lepton workspace listing.

### LEP-003 Run Real Lepton Smoke Commands

Prerequisites: Lepton workspace, quota, credentials, images, shared storage, and reviewed env file.

Run only after the env file has valid site-specific values. These are real
Lepton job submissions, not simulations.

```bash
export NEMOTRON_ENV_FILE="$LEPTON_ROOT/env.lepton.toml"
export LEPTON_CONTAINER_MOUNT="${LEPTON_CONTAINER_MOUNT:-/mnt/lustre-shared}"
export LEPTON_OUTPUT_ROOT="${LEPTON_OUTPUT_ROOT:-$LEPTON_CONTAINER_MOUNT/output/nemotron-qa-$QA_RUN_ID}"

SFT_OUTPUT_DIR="$LEPTON_OUTPUT_ROOT/sft/packed" \
  uv run --no-sync nemotron steps run data_prep/sft_packing \
  -c tiny \
  --batch lepton \
  sample=8 \
  num_shards=1 \
  valid_shards=0 \
  test_shards=0 \
  pack_size=1024 \
  force=true \
  observability.pipeline_logging_interval_s=30

PRETRAIN_OUTPUT_DIR="$LEPTON_OUTPUT_ROOT/pretrain/prep" \
  uv run --no-sync nemotron steps run data_prep/pretrain_prep \
  -c tiny \
  --batch lepton \
  sample=8 \
  num_shards=1 \
  valid_shards=0 \
  test_shards=0 \
  force=true \
  observability.pipeline_logging_interval_s=30

RL_OUTPUT_DIR="$LEPTON_OUTPUT_ROOT/rl/prep" \
  uv run --no-sync nemotron steps run data_prep/rl_prep \
  -c tiny \
  --batch lepton \
  sample=8 \
  num_shards_per_split=1 \
  force=true \
  observability.pipeline_logging_interval_s=30

SFT_OUTPUT_DIR="$LEPTON_OUTPUT_ROOT/sft/automodel" \
  uv run --no-sync nemotron steps run sft/automodel \
  -c tiny \
  --batch lepton_sft_automodel \
  run.env.nodes=1 \
  run.env.gpus_per_node=1 \
  run.env.nprocs_per_node=1 \
  model.pretrained_model_name_or_path=Qwen/Qwen3-0.6B \
  distributed.ep_size=1 \
  step_scheduler.max_steps=1 \
  step_scheduler.global_batch_size=1 \
  dataset.split='train_sft[:8]' \
  validation_dataset.split='test_sft[:2]'

SFT_PACKED_DIR="$LEPTON_OUTPUT_ROOT/sft/packed/splits/train/*.parquet" \
SFT_OUTPUT_DIR="$LEPTON_OUTPUT_ROOT/sft/megatron_bridge" \
  uv run --no-sync nemotron steps run sft/megatron_bridge \
  -c tiny \
  --batch lepton

PRETRAIN_BLEND_PATH="$LEPTON_OUTPUT_ROOT/pretrain/prep/blend.json" \
PRETRAIN_OUTPUT_DIR="$LEPTON_OUTPUT_ROOT/pretrain/megatron_bridge" \
  uv run --no-sync nemotron steps run pretrain/megatron_bridge \
  -c tiny \
  --batch lepton
```

Success criteria:

- Real data-prep and training runs return Lepton job IDs or complete successfully depending on executor behavior.
- The one-GPU AutoModel smoke is the required training check for the provided one-A100 profile.
- Megatron-Bridge training commands are follow-on compatibility checks; if they require more GPUs than the selected profile provides, record the resource mismatch instead of treating it as a product failure.
- Remote logs show mounted paths and do not print secrets.
- If Lepton workspace, quota, credentials, images, or shared storage are unavailable, mark the specific run `BLOCKED`.

Evidence to collect: Lepton job IDs, logs, prepared data artifacts, checkpoints, and redacted secret evidence.

### LEP-004 Agent-Driven Lepton Workflow

Prerequisites: Agent environment available; generated and reviewed `env.lepton.toml`.

Ask the agent:

```text
Run the tiny training workflow on Lepton. Copy the QA training data to shared storage, run data_prep/sft_packing on Lepton, then submit one actual tiny SFT job on Lepton. Do not substitute metadata checks for the training runtime.
```

Success criteria:

- Uses `nemotron steps run data_prep/sft_packing` with `--batch lepton`.
- Uses one training step with the `--batch lepton` profile.
- Keeps input/output under shared storage.
- Returns Lepton job IDs and validates output artifact paths.

Evidence to collect: agent transcript, generated commands, Lepton job IDs, job logs, and output artifact paths.

## Evaluation

### EVAL-001 Create Evaluation Test Data

Prerequisites: Base environment from `SETUP-001`.

```bash
export EVAL_ROOT="$QA_ROOT/eval"
mkdir -p "$EVAL_ROOT/results"

cat > "$EVAL_ROOT/results/synthetic_results.jsonl" <<'EOF'
{"benchmark":"qa-smoke","sample_id":"1","prediction":"B","answer":"B","score":1.0}
{"benchmark":"qa-smoke","sample_id":"2","prediction":"A","answer":"C","score":0.0}
EOF
```

Success criteria:

- Test data files are created under `$QA_ROOT`.
- Checked-in files are not modified.

Evidence to collect: test data paths and file samples.

### EVAL-002 Discover Model Evaluation

Prerequisites: Eval step config present.

```bash
uv run --no-sync nemotron steps list --category eval --json
uv run --no-sync nemotron steps show eval/model_eval --json
```

Success criteria:

- Eval step is discoverable.
- Metadata describes required endpoint, model, tokenizer, and output parameters.

Evidence to collect: metadata JSON and CLI logs.

### EVAL-003 Run Hosted Eval Smoke

Prerequisites: Reachable evaluation endpoint and tokenizer/model identifiers.

```bash
: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY for hosted eval}"
: "${EVAL_URL:?Set EVAL_URL for the eval endpoint}"
: "${EVAL_MODEL_ID:?Set EVAL_MODEL_ID for the eval endpoint}"
: "${EVAL_TOKENIZER:?Set EVAL_TOKENIZER to a tokenizer path visible to the run}"

uv run --no-sync nemotron steps run eval/model_eval \
  -c tiny \
  output_dir="$EVAL_ROOT/results-tiny" \
  deployment.url="$EVAL_URL" \
  deployment.model_id="$EVAL_MODEL_ID" \
  deployment.api_key_name=NVIDIA_API_KEY \
  params.limit_samples=1 \
  params.extra.tokenizer="$EVAL_TOKENIZER"
```

Validate output:

```bash
find "$EVAL_ROOT/results-tiny" -maxdepth 5 -type f | sort
```

Success criteria:

- One-sample eval runs or fails with a clear endpoint/tokenizer/credential error.
- Result files are written under `EVAL_ROOT/results-tiny` when successful.
- Secrets are not printed in logs.

Evidence to collect: result files, summary logs, endpoint metadata, and redacted logs.

### EVAL-004 Agent-Driven Evaluation Workflow

Prerequisites: Agent environment available; endpoint details from user.

Ask the agent:

```text
Set up a one-sample hosted evaluation smoke run for my model endpoint. Ask me for any missing endpoint, model ID, tokenizer path, or API key environment variable before running.
```

Success criteria:

- Uses `eval/model_eval`.
- Runs a real one-sample eval or records a concrete endpoint/tokenizer/credential blocker.
- Asks for missing endpoint/model/tokenizer/key instead of inventing values.
- Validates output artifact paths after the real run.

Evidence to collect: agent transcript, generated command, and result summary.

## Follow-Up TODOs From QA Execution

- Rename or document `merge_scores` more clearly as metadata-specific behavior, because it means "merge FAITH scores into translation metadata", not "merge translated segments back into documents".
- Harden Curator FAITH response parsing so a null model response is handled as a parse failure or clear backend error instead of `TypeError: 'NoneType' object is not iterable`.
- Add dedicated Lepton CPU profiles for `data_prep/pretrain_prep` and `data_prep/rl_prep`; the current runbook reuses the existing CPU data-prep profile.

## Reporting

Use these result labels:

| Result | Meaning |
| --- | --- |
| `PASS` | Command ran and all expected artifacts or behavior were observed. |
| `FAIL` | Product behavior violated the expected result. |
| `BLOCKED` | Required credential, endpoint, GPU, Lepton workspace, Docker daemon, image, quota, or external service was unavailable. |
| `NOT RUN` | Intentionally skipped; reason must be recorded. |

For each module, report:

- Direct CLI result.
- Agent-driven result.
- Exact command used.
- Config path and relevant overrides.
- Output artifact paths.
- Row counts or job IDs where applicable.
- Any redacted credential evidence.
- Final status: `PASS`, `FAIL`, `BLOCKED`, or `NOT RUN`.

## Exit Criteria

- General setup commands pass in a clean `uv` environment.
- Translation direct local workflow passes for at least one real backend, or unavailable backend is marked `BLOCKED`.
- BYOB prepare/generate and translation workflows pass, or unavailable hosted model/backend is marked `BLOCKED`.
- Training metadata discovery passes locally; actual data prep and tiny training jobs run on Lepton and return job IDs or completed artifacts.
- SDG real generation writes JSONL or is marked `BLOCKED` for missing endpoint/key.
- Airgap plan validation passes for SDG plus SFT targets.
- Lepton profile validation passes, and real Lepton data-prep/training smoke is run or marked `BLOCKED`.
- Hosted eval smoke runs or is marked `BLOCKED`.
- Agent-driven runs choose the same module contracts and do not invent missing credentials, endpoints, paths, or model names.
