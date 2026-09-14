# Manual BFCL flow: Oracle Pack to evaluation

This guide runs a hand-authored Oracle Pack through the complete BFCL path:

```text
manual Oracle Pack
  -> fresh validation and Gold gate
  -> benchmark generation
  -> optional guarded LLM paraphrasing
  -> optional surface quality and deduplication/balancing
  -> atomic publication
  -> evaluation source verification and contamination gate
  -> trace and executable candidate evaluation
  -> immutable evaluation artifacts
```

The flow is manual because the user supplies the executable Oracle Pack. A
paraphrase model may change approved model-facing text, but it does not author
tools, fixtures, expected calls, assertions, or Oracle truth.

The bundled
[`banking_vn_oracle_pack`](../data/banking_vn_oracle_pack/README.md) is a
reference implementation. Banking-specific paths and scale are isolated in
[the reference example](#banking-vn-reference-example); they are not BFCL
defaults. Its file map, runnable commands, and the release it produced are in
[banking_vn pack operations](bfcl-banking-vn-pack-operations.md), which is kept
outside the pack directory because a published benchmark freezes every byte in
it.

## 1. User inputs

### Oracle Pack files

Prepare one Oracle Pack directory under an `oracle_runtime.allowed_roots`
entry. Paths declared in `manifest.yaml` are relative to the pack directory.

| File | Required | User responsibility |
| --- | --- | --- |
| `manifest.yaml` | Yes | Pack identity, version, languages, paths, clock, primary keys, absent IDs, prompts, and confirmation vocabulary. |
| `tools.json` | Yes | OpenAI-compatible function schemas and pack-local `x-mutates` / `x-requires-confirmation` declarations. |
| `backend.py` | One Oracle implementation | Local deterministic Oracle implementing `list_tools`, `reset`, `call_tool`, and `get_state`. |
| `endpoint_config.yaml` | One Oracle implementation | HTTPS BFCL Oracle HTTP v1 endpoint. Use this instead of `backend.py`, never together. |
| `fixtures.json` | Optional | Deterministic reset state and slot inventory. |
| `task_templates.yaml` | Yes | User turns, slots, milestones, call groups, policies, expected dependencies, and success-assertion references. |
| `validation_cases.yaml` | Yes | Positive, negative, confirmation, mutation, reset, and schema-alignment probes. |
| `assertions.py` | Yes | Deterministic success assertions referenced by task templates. |
| `held_out.yaml` | Optional | Reserved fixture IDs and template IDs excluded from normal generation. |
| Pack-local helper modules | Optional | Code imported by `backend.py` or `assertions.py`; these files are included in the pack fingerprint. |

Exactly one Oracle is required:

```text
backend.py XOR endpoint_config.yaml
```

For the complete schema and runtime rules, see
[BFCL Oracle Pack Contract](bfcl-oracle-pack.md).

### Generation inputs

Prepare a resolved BFCL generation YAML containing:

- absolute `oracle_pack.manifest_path` for an external pack;
- `oracle_runtime.allowed_roots`, frozen clock, process worker, and timeouts;
- unique `output_dir` and `expt_name`;
- task budgets and optional publication target;
- optional paraphrase, surface-quality, and semantic-dedup settings;
- explicit export settings.

If LLM paraphrasing is enabled, also prepare:

- a Data Designer provider entry in
  `$DATA_DESIGNER_HOME/model_providers.yaml`;
- an immutable paraphrase-model identity;
- an environment-variable name containing its credential;
- language-appropriate templates and reachable diversity constraints.

### Evaluation inputs

Prepare:

- a committed generation `run_manifest.json`;
- the exact Oracle Pack used by generation for executable mode;
- an OpenAI-compatible candidate endpoint with tool calling;
- the served model ID;
- an immutable candidate revision or SHA-256 weights digest;
- a credential environment-variable name;
- a new evaluation output directory outside the generation tree.

The candidate must not be a model exposed to benchmark rows during profiling,
paraphrasing, judging, or translation.

## 2. Work from the repository root

```bash
cd /path/to/Nemotron
export NEMOTRON_ROOT="$PWD"
```

Set pack and run paths. Use persistent storage on shared or managed hosts:

```bash
export BFCL_PACK_ROOT="/absolute/path/to/oracle_pack"
export BFCL_PACK_MANIFEST="$BFCL_PACK_ROOT/manifest.yaml"
export BFCL_RUN_ROOT="/persistent/path/to/bfcl-runs/manual-pack-v1"
export BFCL_GEN_CONFIG="$BFCL_RUN_ROOT/generation.paraphrase.yaml"
```

Relative paths in a BFCL generation config resolve from the checked-in
`src/nemotron/steps/byob/` root, not from the shell working directory or the
YAML file’s directory. Use absolute `manifest_path` and `allowed_roots` for an
external pack.

Do not edit the Oracle Pack between validation and publication. Any pack drift
invalidates validation, checkpoints, and publication.

## 3. Install the runtime

Use Python 3.11–3.13:

```bash
uv sync --extra byob
```

When Stage 11 uses local embedding or GPU deduplication dependencies:

```bash
uv sync --extra byob --extra byob-gpu
```

Confirm the entry point:

```bash
uv run nemotron steps run byob/bfcl --help
```

## 4. Create a generation config

Start from the closest checked-in example, then copy it outside the source
tree:

```bash
mkdir -p "$BFCL_RUN_ROOT"
cp \
  "$NEMOTRON_ROOT/src/nemotron/steps/byob/bfcl/config/default.yaml" \
  "$BFCL_GEN_CONFIG"
```

At minimum, resolve these fields:

```yaml
schema_version: "1.1"
config_status: resolved
family: bfcl
stage: all
expt_name: REPLACE_WITH_UNIQUE_EXPERIMENT_NAME
random_seed: 42
output_dir: <BFCL_RUN_ROOT>/generation

oracle_pack:
  manifest_path: <BFCL_PACK_MANIFEST>

oracle_runtime:
  clock: "REPLACE_WITH_FROZEN_ISO_8601_TIME"
  tool_timeout_s: 5.0
  assertion_timeout_s: 5.0
  import_timeout_s: 10.0
  reset_timeout_s: 5.0
  episode_timeout_s: 60.0
  worker: process
  allowed_roots:
    - <BFCL_PACK_ROOT>
```

Replace each angle-bracket value with the corresponding exported absolute
path. YAML does not expand shell variables automatically.

`worker: thread` is debugging-only and cannot produce a Gold release.

Task counts are pack-specific. Set them from reachable template/fixture
inventory, not from another pack:

```yaml
task_generation:
  tasks_per_category: REPLACE_WITH_CATEGORY_CAP
  # Optional larger Stage-4 inventory:
  candidate_tasks_per_category: REPLACE_WITH_CANDIDATE_CAP
  # Optional exact Stage-11 publication target:
  target_published_tasks: REPLACE_WITH_TARGET_OR_NULL
```

## 5. Optional LLM paraphrase configuration

To generate model-authored surface variants, configure the paraphrase role and
surface generation together:

```yaml
lineage:
  policy: strict_separation
  profile_influenced_surface: false
  judge_advisory: null
  roles:
    profile: {enabled: false, model_config: null}
    paraphrase:
      enabled: true
      model_config:
        alias: REPLACE_WITH_ALIAS
        model: REPLACE_WITH_MODEL_ROUTE
        provider: REPLACE_WITH_PROVIDER_NAME
        canonical_id: REPLACE_WITH_IMMUTABLE_MODEL_ID
        api_key_env: BFCL_PARAPHRASE_API_KEY
        base_url: https://provider.example.com/v1
        inference_parameters:
          temperature: 0.8
          max_tokens: 2048
          max_parallel_requests: 8
    surface_judge: {enabled: false, model_config: null}

surface_generation:
  language: REPLACE_WITH_BCP47_LANGUAGE
  model_paraphrase_enabled: true
  paraphrases_per_template: 1
  preserve_slot_values: true
  prevent_tool_name_leakage: true
```

Register the matching provider and export credentials by reference:

```bash
export DATA_DESIGNER_HOME="/path/to/data-designer-home"
test -f "$DATA_DESIGNER_HOME/model_providers.yaml"
export BFCL_PARAPHRASE_API_KEY="<secret>"
```

Never put credential values in generation or provider YAML.

Paraphrasing does not create new executable cases. Each variant retains the
canonical task’s Oracle binding and passes deterministic guards before it can
reach publication.

## 6. Optional Stage 10 and Stage 11 configuration

Stage 10 checks surface quality. Stage 11 performs semantic deduplication and
coverage-aware balancing and requires Stage 10 when enabled:

```yaml
surface_quality_validation:
  contract_version: "1.1"
  enabled: true
  drop_authority: false

semantic_deduplication_config:
  contract_version: "1.0"
  enabled: true
  model_identifier: sentence-transformers/all-MiniLM-L6-v2
  n_clusters: REPLACE_WITH_REACHABLE_CLUSTER_COUNT
  eps: 0.08
  remove_duplicates: false
  representative_source_preference: [model, template]
  unmet_target_policy: abort
```

Category caps, mix targets, cluster counts, exact-surface limits, and execution
reuse limits must be derived for the current pack. Copying them from another
domain can make Stage 11 infeasible.

## 7. Run validation before spending model tokens

For a fast authoring loop, run the standalone validator:

```bash
uv run python -m nemotron.steps.byob.scripts.validate_oracle_pack \
  --config "$BFCL_GEN_CONFIG"
```

Use the pipeline preflight before publication even if the standalone check
passes.

Run `prepare` first:

```bash
uv run nemotron steps run byob/bfcl \
  -c "$BFCL_GEN_CONFIG" \
  stage=prepare \
  family=bfcl
```

This normalizes the pack, executes validation cases, checks reset/replay
behavior, and derives the tier without requesting paraphrases.

Locate the report under:

```text
<output_dir>/<expt_name>/stage_cache/oracle_validation_report.json
```

Continue only when it reports:

```text
gold_eligible: true
```

Do not bypass the gate or edit the report.

## 8. Generate and publish the benchmark

```bash
uv run nemotron steps run byob/bfcl \
  -c "$BFCL_GEN_CONFIG" \
  stage=generate \
  family=bfcl
```

Generation performs fresh validation again, then runs:

```text
reference_profile
  -> expand
  -> state_machine
  -> render (optional paraphrase)
  -> expected_trace
  -> schema_validation
  -> executable_replay
  -> surface_quality (optional)
  -> dedup_balancing (optional)
  -> final_output
```

Set the resulting paths:

```bash
export BFCL_PUBLICATION_DIR="<output_dir>/<expt_name>"
export BFCL_RUN_MANIFEST="$BFCL_PUBLICATION_DIR/run_manifest.json"
```

Model I/O caches are append-only. On failure, preserve the experiment directory
and use `skip_until=<stage>` only when its predecessor checkpoint is intact and
the pack, config, and pipeline identities have not changed.

Never patch generated parquet, exports, manifests, or cache completion records.

## 9. Verify publication

`run_manifest.json` is written last as the Stage 12 commit marker:

```bash
test -f "$BFCL_RUN_MANIFEST"
test -f "$BFCL_PUBLICATION_DIR/benchmark_raw.parquet"
test -f "$BFCL_PUBLICATION_DIR/benchmark.parquet"
```

If compatibility exports are enabled:

```bash
test -f "$BFCL_PUBLICATION_DIR/exports/export_validation_report.json"
```

Inspect publication lineage:

```bash
uv run python - "$BFCL_RUN_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps({
    "tier": manifest.get("tier"),
    "gold_eligible": manifest.get("gold_eligible"),
    "publication": manifest.get("publication"),
    "stage_counts": manifest.get("stage_counts"),
    "models": manifest.get("models"),
}, indent=2, ensure_ascii=False))
PY
```

Do not evaluate a bare parquet or a directory without `run_manifest.json`.

### Evaluate an existing published benchmark

If generation already completed successfully, skip Sections 4–8 and bind the
evaluation to the existing publication directory. The directory must contain
the original `run_manifest.json`, `benchmark.parquet`, and
`benchmark_raw.parquet`; copying only the parquet data is insufficient.

For example, to evaluate the existing Banking VN publication
`bfcl_banking_vn_gold_v1_1392`:

```bash
cd /path/to/Nemotron
export NEMOTRON_ROOT="$PWD"

# Override BFCL_RUN_ROOT when the publication is on another persistent mount.
export BFCL_RUN_ROOT="${BFCL_RUN_ROOT:-$HOME/bfcl-runs}"
export BFCL_PUBLICATION_DIR="$BFCL_RUN_ROOT/bfcl_banking_vn_gold_v1_1392"
export BFCL_RUN_MANIFEST="$BFCL_PUBLICATION_DIR/run_manifest.json"

test -f "$BFCL_RUN_MANIFEST"
test -f "$BFCL_PUBLICATION_DIR/benchmark.parquet"
test -f "$BFCL_PUBLICATION_DIR/benchmark_raw.parquet"
```

Executable evaluation also needs the exact Oracle Pack that produced the
publication:

```bash
export BFCL_PACK_ROOT="$NEMOTRON_ROOT/src/nemotron/steps/byob/data/banking_vn_oracle_pack"
export BFCL_PACK_MANIFEST="$BFCL_PACK_ROOT/manifest.yaml"

test -f "$BFCL_PACK_MANIFEST"
test -f "$BFCL_PACK_ROOT/backend.py"
```

Do not edit or regenerate any file inside the existing publication. If it was
copied from another host, preserve the complete directory and use a checkout
whose Oracle Pack matches the identity recorded by `run_manifest.json`.
Source verification will fail closed if the benchmark, manifest, or Oracle
resource has drifted.

Continue with Section 10 to configure the independent candidate, then set these
values in Section 11:

```yaml
source_run_manifest: /absolute/path/to/bfcl_banking_vn_gold_v1_1392/run_manifest.json

source_oracle:
  kind: python
  pack_manifest: /absolute/path/to/banking_vn_oracle_pack/manifest.yaml
  resource: /absolute/path/to/banking_vn_oracle_pack/backend.py
```

Keep the evaluation output outside
`$BFCL_PUBLICATION_DIR`, for example
`$BFCL_RUN_ROOT/eval/banking-vn-candidate-v1/artifacts`. Run the preflight in
Section 12 before enabling live candidate traffic.

## 10. Prepare an independent candidate endpoint

The endpoint must support OpenAI-compatible chat completions and function/tool
calling:

```bash
export BFCL_CANDIDATE_BASE_URL="https://candidate.example.com/v1"
export BFCL_CANDIDATE_API_KEY="<secret>"

curl -sS \
  -H "Authorization: Bearer $BFCL_CANDIDATE_API_KEY" \
  "$BFCL_CANDIDATE_BASE_URL/models"
```

Record:

- the returned served model ID;
- an immutable 40–64 character model commit or
  `sha256:<64 hex>` weights digest;
- the registry/source-qualified model name.

Do not use moving revisions such as `main`, `master`, `latest`, branches, or
tags.

## 11. Create a resolved evaluation config

Create a new directory outside the generation publication tree:

```bash
export BFCL_EVAL_ROOT="$BFCL_RUN_ROOT/eval/candidate-v1"
mkdir -p "$BFCL_EVAL_ROOT"
```

Copy the current schema templates:

```bash
cp \
  "$NEMOTRON_ROOT/src/nemotron/steps/byob/bfcl/config/eval.default.yaml" \
  "$BFCL_EVAL_ROOT/eval.yaml"
cp \
  "$NEMOTRON_ROOT/src/nemotron/steps/byob/bfcl/config/eval.cli.yaml" \
  "$BFCL_EVAL_ROOT/eval.cli.yaml"
```

Resolve every `REPLACE_ME_*` value in `$BFCL_EVAL_ROOT/eval.yaml`, set
`config_status: resolved`, and use the following shape:

```yaml
schema_version: "1.1"
config_status: resolved

source_run_manifest: <BFCL_RUN_MANIFEST>

# Required for executable mode. Choose exactly one resource kind.
source_oracle:
  kind: python
  pack_manifest: <BFCL_PACK_MANIFEST>
  resource: <BFCL_PACK_ROOT>/backend.py

# Endpoint-backed alternative:
# source_oracle:
#   kind: endpoint
#   pack_manifest: <BFCL_PACK_MANIFEST>
#   resource: <BFCL_PACK_ROOT>/endpoint_config.yaml

translation_manifest: null

eval:
  mode: [trace, executable]

scoring:
  contract: <NEMOTRON_ROOT>/src/nemotron/steps/byob/references/bfcl-eval-scoring-contract.md
  argument_matching: schema_then_canonical
  insert_declared_defaults: true
  respect_call_order: true
  respect_call_group: true
  allow_llm_repair: false
  task_success: all_applicable_gates

limits:
  max_turns: 5
  tool_timeout_s: 10.0
  candidate_timeout_s: 60.0
  episode_timeout_s: 300.0
  max_parallel_tasks: 1
  max_retries: 2

candidates:
  - alias: candidate_a
    model: REPLACE_WITH_SERVED_MODEL_ID
    provider: openai_compatible
    provider_api_version: v1
    api:
      base_url: https://candidate.example.com/v1
      api_key_env: BFCL_CANDIDATE_API_KEY
    model_identity:
      source: huggingface
      model: REPLACE_WITH_REGISTRY_MODEL_NAME
      revision: REPLACE_WITH_40_TO_64_HEX_COMMIT
      weights_digest: null
    inference:
      temperature: 0.0
      top_p: 1.0
      max_tokens: 8192
      seed: 42
      tool_choice: auto
      provider_extensions: {}

contamination:
  enforce: true
  on_violation: fail_run
  comparison_set: common_intersection

publication:
  requested: true
  require_same_task_ids: true

outputs:
  output_dir: <BFCL_EVAL_ROOT>/artifacts
  write_task_results: true
  write_eval_manifest: true
  cache_candidate_responses: true
  cache_tool_results: true
```

Replace the angle-bracket values with the exported absolute paths before
running evaluation; the eval loader does not expand shell variables.

For trace-only debugging, use `eval.mode: [trace]` and
`source_oracle: null`. Publishable executable evaluation requires a
Gold-eligible source and the exact Oracle resource.

Resolve `$BFCL_EVAL_ROOT/eval.cli.yaml` as:

```yaml
schema_version: "1.0"
family: bfcl
stage: eval
eval_config_path: ./eval.yaml
execution_backend: direct
output_format: human
probe_oracle: true
dry_run: true
```

## 12. Run evaluation preflight

```bash
uv run nemotron steps run byob/bfcl \
  -c "$BFCL_EVAL_ROOT/eval.cli.yaml"
```

Expected CLI output includes:

- `status: preflight_passed`;
- `candidate_network_used: false`;
- the evaluation config hash;
- authorized task counts;
- whether the Oracle was probed.

Direct CLI preflight performs no candidate inference and does not commit
evaluation artifacts.

Resolve contamination findings before proceeding. Do not weaken the gate for a
publishable score.

## 13. Run live evaluation

Change the CLI envelope:

```yaml
dry_run: false
```

Run:

```bash
uv run nemotron steps run byob/bfcl \
  -c "$BFCL_EVAL_ROOT/eval.cli.yaml"
```

The evaluator drives candidate conversations, scores proposed traces, executes
candidate calls against task-local Oracle sessions in executable mode, runs
pack assertions, aggregates metrics, and writes the final eval manifest last.

Start with `max_parallel_tasks: 1`. Increase it only after confirming endpoint
concurrency and rate limits; any config change produces a different
`eval_config_hash`.

## 14. Inspect evaluation artifacts

A successful trace-and-executable run writes:

```text
artifacts/
├── resolved_eval_config.json
├── source_verification_report.json
├── contamination_report.json
├── candidate_io_cache.jsonl
├── tool_trace_cache.jsonl
├── eval_report.json
├── eval_task_results.parquet
└── eval_manifest.json
```

Inspect aggregate metrics:

```bash
uv run python - "$BFCL_EVAL_ROOT/artifacts/eval_report.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps(report, indent=2, ensure_ascii=False))
PY
```

Inspect task-level schema and row count:

```bash
uv run python - "$BFCL_EVAL_ROOT/artifacts/eval_task_results.parquet" <<'PY'
import sys
import pyarrow.parquet as pq

table = pq.read_table(sys.argv[1])
print(table.schema)
print(f"rows={table.num_rows}")
PY
```

Interpret trace and executable metrics separately. Trace success means the
candidate proposed the expected calls. Executable success additionally means
those calls ran against the Oracle and satisfied the pack assertions.

## 15. Safe reruns and recovery

- Preserve append-only paraphrase and candidate I/O caches.
- Use generation `skip_until` only with a verified predecessor checkpoint.
- Create a new evaluation output directory for a new candidate or changed
  configuration.
- Do not overwrite a committed evaluation artifact set.
- Regenerate after schema, hash, or publication-contract failures.
- Point `source_oracle` at the exact pack manifest and Oracle resource used by
  generation.
- Select another candidate if contamination is detected or unresolved.

## 16. Banking VN reference example

Use these files as a complete example of the generic inputs:

```text
src/nemotron/steps/byob/data/banking_vn_oracle_pack/
├── manifest.yaml
├── tools.json
├── backend.py
├── fixtures.json
├── task_templates.yaml
├── validation_cases.yaml
├── assertions.py
└── README.md
```

Reference generation profiles:

```text
src/nemotron/steps/byob/bfcl/config/smoke.example.yaml
src/nemotron/steps/byob/bfcl/config/publication.example.yaml
src/nemotron/steps/byob/bfcl/config/publication.paraphrase.example.yaml
```

- `smoke.example.yaml` is a small `smoke_no_publication` profile.
- `publication.example.yaml` is the template-only Gold profile.
- `publication.paraphrase.example.yaml` adds guarded model paraphrasing on top of
  the same executable cases.

All three run as written against the bundled reference pack. Copy the one that
matches your intent and repoint `oracle_pack.manifest_path` at your own pack.

Example paths:

```bash
export BFCL_PACK_ROOT="$NEMOTRON_ROOT/src/nemotron/steps/byob/data/banking_vn_oracle_pack"
export BFCL_PACK_MANIFEST="$BFCL_PACK_ROOT/manifest.yaml"
export BFCL_GEN_CONFIG="$NEMOTRON_ROOT/src/nemotron/steps/byob/bfcl/config/publication.paraphrase.example.yaml"
```

The bundled profile uses a provider named `nvidia_inference_api`, credential
reference `NGC_API_KEY`, Vietnamese language `vi`, Stage 10, Stage 11, and
Banking-specific balance constraints. Replace those deployment values when
using another provider, but do not copy the Banking task counts or diversity
limits into another pack without proving they are reachable.

## 17. Completion checklist

The flow is complete only when:

- every required Oracle Pack file exists and exactly one Oracle kind is used;
- fresh validation reports `gold_eligible: true`;
- Stage 12 publishes `run_manifest.json`;
- published row counts satisfy the current config, not another pack’s target;
- enabled paraphrase, surface-quality, dedup, and export reports are complete;
- evaluation source verification and contamination checks pass;
- requested trace and executable metrics are present;
- `eval_manifest.json` commits the immutable evaluation artifact set.
