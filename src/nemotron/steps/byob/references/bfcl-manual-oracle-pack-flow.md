# Manual BFCL flow: Oracle Pack to publication

This guide runs a hand-authored Oracle Pack through the complete BFCL path:

```text
manual Oracle Pack
  -> fresh validation and Gold gate
  -> benchmark generation
  -> optional guarded LLM paraphrasing
  -> optional surface quality and deduplication/balancing
  -> atomic publication
  -> immutable release archive
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

## 10. Archive and recover safely

Archive a completed publication with:

```bash
python -m nemotron.steps.byob.scripts.archive_bfcl_release \
  --run-manifest /path/to/run_manifest.json \
  --output /path/to/release.tar.gz
```

Treat `run_manifest.json` as the commit marker. Never repair one generated file
in place. Resume only from an untouched checkpoint chain; otherwise run a fresh
`prepare` and `generate` into a clean output tree.

## Completion checklist

- Oracle validation is Gold-eligible for publication configs.
- Local code runs only in a process worker, or HTTPS identity is pinned.
- Every generated row passed schema validation and deterministic replay.
- Optional surface quality, balancing, and exports have their reports.
- `benchmark.parquet` is an unchanged selection of `benchmark_raw.parquet`.
- `run_manifest.json` was written last and hashes every published artifact.
