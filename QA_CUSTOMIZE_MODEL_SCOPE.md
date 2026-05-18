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
- Curator implementation details.

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
export EVAL_MODEL_ID="${EVAL_MODEL_ID:-openai/gpt-oss-120b}"
export EVAL_URL="${EVAL_URL:-https://integrate.api.nvidia.com/v1/chat/completions}"

# Required for reliable Ray worker startup in the QA environment.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV="${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-0}"
```

Credential variables used by the runbook:

```bash
# Set these only when running real hosted workflows.
export NVIDIA_API_KEY="${NVIDIA_API_KEY:-}"
export NGC_API_KEY="${NGC_API_KEY:-$NVIDIA_API_KEY}"
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HOME="${HF_HOME:-$QA_ROOT/hf-cache}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-nemotron-qa}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_NAME="${WANDB_NAME:-nemotron-qa-$QA_RUN_ID}"
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
uv run nemotron steps show byob/mcq
```

Success criteria:

- `uv sync` exits 0.
- All listed help and discovery commands exit 0.
- Help output is captured for `nemotron`, `nemotron steps`, and `nemotron steps show byob/mcq`.

Evidence to collect: command logs, CLI help snippets, and final exit status.

### SETUP-002 Install Optional Workflow Extras

Prerequisites: Network access to package indexes and Git dependencies.

```bash
uv sync --extra translation --extra byob --extra data-sdg --extra evaluator --group run

# For this RC, install Curator from source so translation prompt resources are
# available from site-packages.
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

- Translation, BYOB, SDG, evaluator, and run dependencies install successfully.
- Editable Curator is installed and `translate.yaml` is importable through package resources.
- If package indexes or Git dependencies are unavailable, this test is marked `BLOCKED` with the exact failed dependency or network error.

Evidence to collect: install logs and final exit status.

### SETUP-003 Snapshot Step Metadata

Prerequisites: Optional workflow extras from `SETUP-002`.

```bash
for STEP in \
  translate/curator \
  byob/mcq \
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
uv run --no-sync nemotron steps show translate/curator
uv run --no-sync nemotron steps run translate/curator --help
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

uv run --no-sync nemotron steps run translate/curator \
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
uv run --no-sync nemotron steps run translate/curator \
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
uv run --no-sync nemotron steps run translate/curator \
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

uv run --no-sync nemotron steps run translate/curator \
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
uv run --no-sync nemotron steps run translate/curator \
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
uv run --no-sync nemotron steps run translate/curator \
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
if uv run --no-sync nemotron steps run translate/curator \
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

- Uses `nemotron steps run translate/curator` for local execution.
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

cp src/nemotron/steps/byob/mcq/config/tiny.yaml "$BYOB_ROOT/config/byob_mcq.yaml"
cp src/nemotron/steps/byob/mcq/config/translate.yaml "$BYOB_ROOT/config/byob_translate.yaml"
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
uv run --no-sync nemotron steps show byob/mcq
uv run --no-sync python -m nemotron.steps.byob.scripts.run --list-families
uv run --no-sync python src/nemotron/steps/byob/scripts/validate.py --help
```

Success criteria:

- `mcq` is listed as a family.
- `steps show byob/mcq` lists `prepare`, `generate`, `translate`, and `all` under the `stage` parameter choices.
- Validator help exits 0.

Evidence to collect: CLI logs and family list output.

### BYOB-002 Generate MCQ Benchmark With `--stage all`

Prerequisites: `NGC_API_KEY` or `NVIDIA_API_KEY`; live generation and judge model; copied BYOB config from setup.

```bash
: "${NGC_API_KEY:?Set NGC_API_KEY or NVIDIA_API_KEY for BYOB generation}"

uv run --no-sync nemotron steps run byob/mcq \
  -c "$BYOB_ROOT/config/byob_mcq.yaml" \
  stage=all family=mcq
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
uv run --no-sync nemotron steps run byob/mcq \
  -c "$BYOB_ROOT/config/byob_mcq.yaml" \
  stage=prepare family=mcq

uv run --no-sync nemotron steps run byob/mcq \
  -c "$BYOB_ROOT/config/byob_mcq.yaml" \
  stage=generate family=mcq
```

### BYOB-003 Resume BYOB Generation From A Named Stage

Prerequisites: Outputs from `BYOB-002` or compatible stage cache.

```bash
uv run --no-sync nemotron steps run byob/mcq \
  -c "$BYOB_ROOT/config/byob_mcq.yaml" \
  stage=generate family=mcq skip_until=JUDGEMENT
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

uv run --no-sync nemotron steps run byob/mcq \
  -c "$BYOB_ROOT/config/byob_translate.yaml" \
  stage=translate family=mcq
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

- Uses `nemotron steps run byob/mcq stage=all family=mcq` for prepare+generate, or explicitly explains why it is running `prepare` and `generate` separately.
- Uses `nemotron steps run byob/mcq stage=translate family=mcq`.
- Patches config copies only.
- Validates final Parquet schema.

Evidence to collect: agent transcript, generated configs, command logs, validation output, and artifact paths.

## Training

### Training Workspace Setup

```bash
export TRN_ROOT="$QA_ROOT/training"
mkdir -p "$TRN_ROOT/output"
```

### TRAIN-001 Discover Training And Data-Prep Step Metadata

Prerequisites: Base environment from `SETUP-001`; training workspace setup completed.

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

Prerequisites: Agent environment available; Lepton env file from `LEP-001`.

Ask the agent:

```text
Prepare a tiny SFT workflow using Lepton. Use the bundled tiny data-prep config, run data_prep/sft_packing on Lepton first, then submit the packaged tiny Megatron-Bridge SFT training step with SFT_PACKED_DIR pointing at the data-prep output.
```

Success criteria:

- Uses `data_prep/sft_packing` before a packed-data Megatron-Bridge SFT smoke.
- Uses `--batch lepton_prep_sft_packing` for data prep and `--batch lepton_sft_megatron_bridge` for training.
- Does not override model names, dataset names, split sizes, or scheduler knobs from the packaged configs.
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
export LEPTON_NODE_GROUP="${LEPTON_NODE_GROUP:-az-sat-lepton-001}"
export NEMOTRON_HOST_MOUNT="${NEMOTRON_HOST_MOUNT:-/sovereign-ai-playbook/}"
export NEMOTRON_WORKSPACE="${NEMOTRON_WORKSPACE:-/mnt/lustre-shared}"
export NEMOTRON_MOUNT_FROM="${NEMOTRON_MOUNT_FROM:-node-nfs:amlfs}"
export NEMO_RUN_DIR="${NEMO_RUN_DIR:-$NEMOTRON_WORKSPACE/nemo-run}"
export HF_HOME="${HF_HOME:-$NEMOTRON_WORKSPACE/hf}"
export WANDB_PROJECT="${WANDB_PROJECT:-nemotron-qa}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_NAME="${WANDB_NAME:-nemotron-qa-$QA_RUN_ID}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY for Lepton training and data-prep telemetry}"
uvx --from leptonai lep login -c "$LEPTON_WORKSPACE:$LEPTON_API_KEY"

export LEPTON_ENV_FILE="$LEPTON_ROOT/env.lepton.toml"
uv run --no-sync nemotron steps run env/env_toml \
  -c lepton \
  output_path="$LEPTON_ENV_FILE" \
  force=true \
  sections.lepton_base.node_group="$LEPTON_NODE_GROUP" \
  sections.lepton_base.env_vars.WANDB_PROJECT="$WANDB_PROJECT" \
  sections.lepton_base.env_vars.WANDB_ENTITY="$WANDB_ENTITY" \
  sections.lepton_base.env_vars.WANDB_NAME="$WANDB_NAME"

export NEMOTRON_ENV_FILE="$LEPTON_ENV_FILE"

uvx --from leptonai lep node list
uvx --from leptonai lep node storage -ng "$LEPTON_NODE_GROUP"
uv run --no-sync python - <<'PY'
import os
from pathlib import Path

from nemo_runspec.env import load_env_profile
from omegaconf import OmegaConf

path = Path(os.environ["NEMOTRON_ENV_FILE"])
profiles = path.read_text()
for name in ("lepton_base", "lepton_prep_sft_packing", "lepton_sft_megatron_bridge", "lepton_sft_automodel"):
    assert f"[{name}]" in profiles, name
base = OmegaConf.to_container(load_env_profile("lepton_base", config_path=path), resolve=True)
assert base["mounts"][0]["mount_path"] == os.environ["NEMOTRON_WORKSPACE"]
assert base["env_vars"]["HF_HOME"] == os.environ["HF_HOME"]
assert base["env_vars"]["WANDB_API_KEY"]
assert base["env_vars"]["WANDB_PROJECT"] == os.environ["WANDB_PROJECT"]
assert "WANDB_ENTITY" in base["env_vars"]
assert "WANDB_NAME" in base["env_vars"]
print(path)
PY
```

Success criteria:

- `env.lepton.toml` is generated.
- Lepton login succeeds.
- Configured node group and storage `node-nfs:amlfs` are visible to the workspace.
- Profile `[lepton_base]` uses the configured shared mount.
- Data-prep and training profiles are present, including `[lepton_prep_sft_packing]`, `[lepton_sft_megatron_bridge]`, and `[lepton_sft_automodel]`.
- `HF_HOME`, `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`, and `WANDB_NAME` are exported before submission.
- Secrets are not written in plain text unless explicitly intended by the profile.

Evidence to collect: generated env file path and redacted content sample.

### LEP-002 Validate Lepton Profiles For Scoped Steps

Prerequisites: Env file from `LEP-001`.

```bash
test -s "$NEMOTRON_ENV_FILE"
grep -E "^\[(lepton_base|lepton_prep_sft_packing|lepton_prep_pretrain_prep|lepton_sft_megatron_bridge|lepton_pretrain_megatron_bridge|lepton_sft_automodel)\]" "$NEMOTRON_ENV_FILE"
uvx --from leptonai lep workspace list
```

Success criteria:

- Required Lepton data-prep and training profiles exist in `env.lepton.toml`.
- Lepton workspace listing succeeds.
- This profile-validation case does not replace the actual Lepton runtime cases in `LEP-003`.

Evidence to collect: env file path, profile grep output, and Lepton workspace listing.

### LEP-003 Run Real Lepton Smoke Commands

Prerequisites: Lepton workspace, quota, credentials, images, shared storage, and reviewed env file.

Run only after the env file has valid site-specific values. These are real
Lepton job submissions, not simulations.

```bash
export NEMOTRON_ENV_FILE="$LEPTON_ROOT/env.lepton.toml"
export LEPTON_NODE_GROUP="${LEPTON_NODE_GROUP:-az-sat-lepton-001}"
export NEMOTRON_HOST_MOUNT="${NEMOTRON_HOST_MOUNT:-/sovereign-ai-playbook/}"
export NEMOTRON_WORKSPACE="${NEMOTRON_WORKSPACE:-/mnt/lustre-shared}"
export NEMOTRON_MOUNT_FROM="${NEMOTRON_MOUNT_FROM:-node-nfs:amlfs}"
export NEMO_RUN_DIR="${NEMO_RUN_DIR:-$NEMOTRON_WORKSPACE/nemo-run}"
export LEPTON_CONTAINER_MOUNT="$NEMOTRON_WORKSPACE"
export LEPTON_OUTPUT_ROOT="${LEPTON_OUTPUT_ROOT:-$LEPTON_CONTAINER_MOUNT/output/nemotron-qa-$QA_RUN_ID}"
export HF_HOME="${HF_HOME:-$LEPTON_CONTAINER_MOUNT/hf}"
export WANDB_PROJECT="${WANDB_PROJECT:-nemotron-qa}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_NAME="${WANDB_NAME:-nemotron-qa-$QA_RUN_ID}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY for Lepton training and data-prep telemetry}"

SFT_PREP_DIR="$LEPTON_OUTPUT_ROOT/data_prep/sft_packing"
PRETRAIN_PREP_DIR="$LEPTON_OUTPUT_ROOT/data_prep/pretrain_prep"
RL_PREP_DIR="$LEPTON_OUTPUT_ROOT/data_prep/rl_prep"
LEPTON_JOB_POLL_SECONDS="${LEPTON_JOB_POLL_SECONDS:-60}"
LEPTON_JOB_MAX_WAIT_SECONDS="${LEPTON_JOB_MAX_WAIT_SECONDS:-14400}"

lepton_job_state() {
  uvx --from leptonai lep job get -i "$1" | uv run --no-sync python -c '
import re
import sys

text = sys.stdin.read()
match = re.search(r"\"state\":\s*\"([^\"]+)\"", text)
if not match:
    raise SystemExit("Could not parse Lepton job state")
print(match.group(1))
'
}

wait_for_lepton_job() {
  job_id="$1"
  start_time="$SECONDS"
  while true; do
    state="$(lepton_job_state "$job_id")"
    echo "$job_id: $state"
    case "$state" in
      Completed|Succeeded)
        return 0
        ;;
      Failed|Stopped|Deleted)
        uvx --from leptonai lep job log -i "$job_id"
        return 1
        ;;
    esac
    if (( SECONDS - start_time >= LEPTON_JOB_MAX_WAIT_SECONDS )); then
      echo "$job_id: timed out after ${LEPTON_JOB_MAX_WAIT_SECONDS}s"
      uvx --from leptonai lep job log -i "$job_id"
      return 1
    fi
    sleep "$LEPTON_JOB_POLL_SECONDS"
  done
}

SFT_PACKING_SUBMIT_LOG="$LEPTON_ROOT/sft_packing_submit.log"
SFT_OUTPUT_DIR="$SFT_PREP_DIR" \
  uv run --no-sync nemotron steps run data_prep/sft_packing \
  -c tiny \
  max_rows=100 \
  --batch lepton_prep_sft_packing 2>&1 | tee "$SFT_PACKING_SUBMIT_LOG"
SFT_PACKING_JOB_ID="$(grep -oE 'data-prep-sft-packing-step-[a-z0-9]+' "$SFT_PACKING_SUBMIT_LOG" | tail -1)"
test -n "$SFT_PACKING_JOB_ID"
wait_for_lepton_job "$SFT_PACKING_JOB_ID"
test -d "$SFT_PREP_DIR/splits/train"
find -L "$SFT_PREP_DIR/splits/train" -maxdepth 1 -type f -name "*.parquet" -print -quit | grep -q .

PRETRAIN_PREP_SUBMIT_LOG="$LEPTON_ROOT/pretrain_prep_submit.log"
PRETRAIN_OUTPUT_DIR="$PRETRAIN_PREP_DIR" \
  uv run --no-sync nemotron steps run data_prep/pretrain_prep \
  -c tiny \
  --batch lepton_prep_pretrain_prep 2>&1 | tee "$PRETRAIN_PREP_SUBMIT_LOG"
PRETRAIN_PREP_JOB_ID="$(grep -oE 'data-prep-pretrain-prep-step-[a-z0-9]+' "$PRETRAIN_PREP_SUBMIT_LOG" | tail -1)"
test -n "$PRETRAIN_PREP_JOB_ID"
wait_for_lepton_job "$PRETRAIN_PREP_JOB_ID"

RL_PREP_SUBMIT_LOG="$LEPTON_ROOT/rl_prep_submit.log"
RL_OUTPUT_DIR="$RL_PREP_DIR" \
  uv run --no-sync nemotron steps run data_prep/rl_prep \
  -c tiny \
  --batch lepton_prep_rl_prep 2>&1 | tee "$RL_PREP_SUBMIT_LOG"
RL_PREP_JOB_ID="$(grep -oE 'data-prep-rl-prep-step-[a-z0-9]+' "$RL_PREP_SUBMIT_LOG" | tail -1)"
test -n "$RL_PREP_JOB_ID"
wait_for_lepton_job "$RL_PREP_JOB_ID"

SFT_MB_SUBMIT_LOG="$LEPTON_ROOT/sft_megatron_bridge_submit.log"
SFT_PACKED_DIR="$SFT_PREP_DIR/splits/train/*.parquet" \
SFT_OUTPUT_DIR="$LEPTON_OUTPUT_ROOT/sft/megatron_bridge" \
  uv run --no-sync nemotron steps run sft/megatron_bridge \
  -c tiny \
  --batch lepton_sft_megatron_bridge 2>&1 | tee "$SFT_MB_SUBMIT_LOG"
SFT_MB_JOB_ID="$(grep -oE 'sft-megatron-bridge-step-[a-z0-9]+' "$SFT_MB_SUBMIT_LOG" | tail -1)"
test -n "$SFT_MB_JOB_ID"
wait_for_lepton_job "$SFT_MB_JOB_ID"

PRETRAIN_MB_SUBMIT_LOG="$LEPTON_ROOT/pretrain_megatron_bridge_submit.log"
PRETRAIN_BLEND_PATH="$PRETRAIN_PREP_DIR/blend.json" \
PRETRAIN_OUTPUT_DIR="$LEPTON_OUTPUT_ROOT/pretrain/megatron_bridge" \
  uv run --no-sync nemotron steps run pretrain/megatron_bridge \
  -c tiny \
  --batch lepton_pretrain_megatron_bridge 2>&1 | tee "$PRETRAIN_MB_SUBMIT_LOG"
PRETRAIN_MB_JOB_ID="$(grep -oE 'pretrain-megatron-bridge-step-[a-z0-9]+' "$PRETRAIN_MB_SUBMIT_LOG" | tail -1)"
test -n "$PRETRAIN_MB_JOB_ID"
wait_for_lepton_job "$PRETRAIN_MB_JOB_ID"

SFT_AM_SUBMIT_LOG="$LEPTON_ROOT/sft_automodel_submit.log"
SFT_OUTPUT_DIR="$LEPTON_OUTPUT_ROOT/sft/automodel" \
  uv run --no-sync nemotron steps run sft/automodel \
  -c tiny \
  --batch lepton_sft_automodel 2>&1 | tee "$SFT_AM_SUBMIT_LOG"
SFT_AM_JOB_ID="$(grep -oE 'sft-automodel-step-[a-z0-9]+' "$SFT_AM_SUBMIT_LOG" | tail -1)"
test -n "$SFT_AM_JOB_ID"
wait_for_lepton_job "$SFT_AM_JOB_ID"

PRETRAIN_AM_SUBMIT_LOG="$LEPTON_ROOT/pretrain_automodel_submit.log"
PRETRAIN_BLEND_PATH="$PRETRAIN_PREP_DIR/blend.json" \
PRETRAIN_OUTPUT_DIR="$LEPTON_OUTPUT_ROOT/pretrain/automodel" \
  uv run --no-sync nemotron steps run pretrain/automodel \
  -c tiny \
  --batch lepton_pretrain_automodel 2>&1 | tee "$PRETRAIN_AM_SUBMIT_LOG"
PRETRAIN_AM_JOB_ID="$(grep -oE 'pretrain-automodel-step-[a-z0-9]+' "$PRETRAIN_AM_SUBMIT_LOG" | tail -1)"
test -n "$PRETRAIN_AM_JOB_ID"
wait_for_lepton_job "$PRETRAIN_AM_JOB_ID"
```

Success criteria:

- Real data-prep and training runs submit Lepton job IDs and each submitted job reaches `Completed` or `Succeeded`.
- The required end-to-end checks use data-prep outputs as training inputs: `SFT_PACKED_DIR` points to the SFT packing output and `PRETRAIN_BLEND_PATH` points to the pretrain prep output.
- Training commands do not override model names, dataset names, split sizes, or scheduler knobs; those come from `tiny.yaml` and `default.yaml`.
- AutoModel commands use their packaged tiny configs. Do not force packed-parquet data into AutoModel SFT; the packed output is for Megatron-Bridge SFT.
- Remote logs show mounted paths and do not print secrets.
- Jobs that exceed `LEPTON_JOB_MAX_WAIT_SECONDS` are reported with the Lepton job ID and job logs.
- If Lepton workspace, quota, credentials, images, or shared storage are unavailable, mark the specific run `BLOCKED`.

Evidence to collect: submit logs, Lepton job IDs, status lines, prepared data artifacts, checkpoints, and redacted secret evidence.

### LEP-004 Agent-Driven Lepton Workflow

Prerequisites: Agent environment available; generated and reviewed `env.lepton.toml`.

Ask the agent:

```text
Run the tiny training workflow on Lepton. Generate the Lepton env file, run data_prep/sft_packing with the packaged tiny config and max_rows=100, then submit the tiny SFT Megatron-Bridge training job using SFT_PACKED_DIR from the data-prep output. Do not override model names, dataset names, split sizes, or scheduler knobs.
```

Success criteria:

- Uses `nemotron steps run data_prep/sft_packing -c tiny max_rows=100 --batch lepton_prep_sft_packing`.
- Uses `nemotron steps run sft/megatron_bridge -c tiny --batch lepton_sft_megatron_bridge`.
- Passes the SFT packing output to training through `SFT_PACKED_DIR`.
- Keeps input/output under shared storage.
- Returns Lepton job IDs and validates output artifact paths.

Evidence to collect: agent transcript, generated commands, Lepton job IDs, job logs, and output artifact paths.

## Evaluation

### EVAL-001 Verify Evaluator Dependencies

Prerequisites: Optional workflow extras from `SETUP-002`.

```bash
export EVAL_ROOT="$QA_ROOT/eval"
mkdir -p "$EVAL_ROOT"

uv run --no-sync python - <<'PY'
import importlib.util

assert importlib.util.find_spec("nemo_evaluator_launcher") is not None
PY
```

Success criteria:

- Evaluator dependencies installed in `SETUP-002` remain available.
- `nemo_evaluator_launcher` is importable in the uv environment.
- Checked-in files are not modified.

Evidence to collect: import-check output.

### EVAL-002 Discover Model Evaluation

Prerequisites: Eval step config present.

```bash
uv run --no-sync nemotron steps list --category eval --json
uv run --no-sync nemotron steps show eval/model_eval --json
```

Success criteria:

- Eval step is discoverable.
- Metadata describes required endpoint, model, API key environment variable, and output parameters.

Evidence to collect: metadata JSON and CLI logs.

### EVAL-003 Run Hosted Chat Eval Smoke

Prerequisites: `NVIDIA_API_KEY` is set and `EVAL-001` completed successfully.

```bash
: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY}"
: "${EVAL_URL:?Set EVAL_URL for the eval endpoint}"
: "${EVAL_MODEL_ID:?Set EVAL_MODEL_ID for the eval endpoint}"

export NEMO_EVALUATOR_MODEL_ID="$EVAL_MODEL_ID"
export NEMO_EVALUATOR_MODEL_URL="$EVAL_URL"
export NEMO_EVALUATOR_API_KEY_NAME=NVIDIA_API_KEY
export NEMO_EVALUATOR_ENDPOINT_TYPE=chat

set -o pipefail
uv run --no-sync nemotron steps run eval/model_eval -c tiny_chat \
  output_dir="$EVAL_ROOT/results-tiny-chat" \
  2>&1 | tee "$EVAL_ROOT/tiny-chat.log"

export EVAL_INVOCATION_ID="$(
  awk '/launcher_invocation_id:/ {print $2}' "$EVAL_ROOT/tiny-chat.log" | tail -1
)"
test -n "$EVAL_INVOCATION_ID"
```

Validate output:

```bash
uv run --no-sync nemo-evaluator-launcher status "$EVAL_INVOCATION_ID"

# Wait until the launcher status is SUCCESS, then check the artifacts.

find "$EVAL_ROOT/results-tiny-chat" -path "*/artifacts/results.yml" -o \
  -path "*/artifacts/report.json" -o \
  -path "*/artifacts/eval_factory_metrics.json" | sort
test -n "$(find "$EVAL_ROOT/results-tiny-chat" -path "*/artifacts/results.yml" -print -quit)"
```

Success criteria:

- One-sample hosted chat eval exits 0.
- Launcher status reaches `SUCCESS`.
- Result files are written under `$EVAL_ROOT/results-tiny-chat/<run>/<task>.0/artifacts/`.
- `artifacts/results.yml` exists.
- Secrets are not printed in logs.

Evidence to collect: result files, summary logs, endpoint metadata, and redacted logs.

### EVAL-004 Agent-Driven Evaluation Workflow

Prerequisites: Agent environment available; endpoint details from user.

Ask the agent:

```text
Set up a one-sample hosted chat evaluation smoke run for my model endpoint. Use eval/model_eval with the tiny_chat config. Ask me for any missing endpoint URL, model ID, or API key environment variable before running.
```

Success criteria:

- Uses `eval/model_eval`.
- Uses `tiny_chat.yaml`.
- Runs a real one-sample hosted chat eval or records a concrete endpoint/credential blocker.
- Asks for missing endpoint/model/key instead of inventing values.
- Validates output artifact paths after the real run.

Evidence to collect: agent transcript, generated command, and result summary.

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
