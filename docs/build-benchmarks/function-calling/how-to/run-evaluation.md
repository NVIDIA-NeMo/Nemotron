<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Evaluate a Candidate Model

Use this guide when generation has produced `benchmark.parquet` and you want to
import that benchmark into evaluation, score one or more candidate models, and
retrieve the results. Evaluation is a separate run with its own configuration, so
changing a candidate can never change the identity of the benchmark it is scored on.

## Before You Start

- A completed generation publication containing `run_manifest.json`,
  `benchmark.parquet`, and `benchmark_raw.parquet`.
- An OpenAI-compatible candidate endpoint that supports tool calling, its served model identifier, and the name of an environment variable holding its credential. A literal key anywhere in the configuration is refused.
- For executable mode, the exact Oracle Pack the benchmark was generated from.
- An output directory outside the generation publication tree.
- For NeMo Evaluator Launcher, install this repository's `evaluator` extra in the
  environment that materializes and submits the Launcher task.

Run commands from the repository root. Install the direct evaluator dependencies:

```bash
uv sync --extra byob
```

For the NeMo Evaluator Launcher backend, also install its optional dependency:

```bash
uv sync --extra byob --extra evaluator
```

## Import the Generated Benchmark

In BFCL, importing a benchmark means pointing evaluation at an intact generation
publication. There is no upload or Parquet-ingestion command. The required handoff is:

```text
<publication>/
├── benchmark.parquet
├── benchmark_raw.parquet
├── run_manifest.json
└── exports/
    └── nemo_evaluator_bundle/    # required only for NeMo Evaluator Launcher
```

There is no BFCL artifact named `manifest.parquet`. The publication commit marker is
`run_manifest.json`. It names both Parquet tables and pins their hashes, row counts,
schema, pack identity, and lineage.

If the publication was generated on another host, transfer the whole directory
without changing its contents or layout. For example:

```bash
mkdir -p /srv/bfcl/benchmarks/warehouse-gold
rsync -a /path/to/generated-publication/ \
  /srv/bfcl/benchmarks/warehouse-gold/
```

Confirm the minimum direct-evaluation handoff:

```bash
PUB=/srv/bfcl/benchmarks/warehouse-gold
test -f "$PUB/run_manifest.json"
test -f "$PUB/benchmark.parquet"
test -f "$PUB/benchmark_raw.parquet"
```

If you received only `benchmark.parquet`, stop and obtain the complete publication or
rerun generation. BFCL intentionally cannot reconstruct trusted provenance and raw-row
equivalence from the published table alone.

Connect the imported publication to `eval.yaml` through its manifest:

```yaml
source_run_manifest: /srv/bfcl/benchmarks/warehouse-gold/run_manifest.json
```

The direct evaluator reads `benchmark.parquet` through that manifest. Source
verification also checks the adjacent `benchmark_raw.parquet`; moving only one table
to an evaluation directory is therefore insufficient.

There are two execution backends:

| Backend | Benchmark handoff |
| --- | --- |
| `direct` | Set `source_run_manifest` in `eval.yaml`. No compatibility export is required. |
| `nemo_launcher` | Set the same `source_run_manifest` and use the publication's verified `exports/nemo_evaluator_bundle/`. Generation must have enabled `exports.nemo_evaluator_bundle: true`. |

The exported bundle is adapter input, not a replacement for source verification and
not a standalone Launcher configuration. The evaluation config still binds the
original `run_manifest.json`. If the bundle is absent, rerun publication with the
export enabled; do not create or edit bundle files manually.

## Step 1: Resolve Your Own Evaluation Config

Copy the template and fill it in. Every `REPLACE_ME_*` value must be resolved and `config_status` must become `resolved`.

```bash
REPO_ROOT="$(pwd)"
mkdir -p /srv/bfcl/eval/candidate-a && \
  cp src/nemotron/steps/byob/bfcl/config/eval.default.yaml \
    /srv/bfcl/eval/candidate-a/eval.yaml
```

The template's scoring contract is repository-relative. Because the copy no
longer has the repository's directory layout, replace it with the absolute
contract path before editing any other values:

```bash
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
python - "$REPO_ROOT" /srv/bfcl/eval/candidate-a/eval.yaml <<'PY'
from pathlib import Path
import sys
import yaml

repo, destination = map(Path, sys.argv[1:])
document = yaml.safe_load(destination.read_text())
document["scoring"]["contract"] = str(
    (repo / "src/nemotron/steps/byob/references/bfcl-eval-scoring-contract.md").resolve()
)
destination.write_text(yaml.safe_dump(document, sort_keys=False))
PY
```

:::{important}
Nothing in this file falls back to a default. Every scoring gate, runtime limit, and decoding parameter is stated, because each one changes what the resulting number means: a model cut off at two turns did not answer the same question as one given ten. Quoted booleans and numbers are refused rather than coerced, since a `"false"` that became `true` would silently switch off a correctness gate.
:::

At minimum, resolve these operator-owned values:

- `source_run_manifest`: the imported publication's `run_manifest.json`.
- `eval.mode`: `[trace]` or `[trace, executable]`.
- `source_oracle`: required only for executable mode.
- `candidates`: the endpoint route, credential environment-variable name, and model
  identity.
- `outputs.output_dir`: a new directory outside the imported publication.
- `publication.requested`: keep `true` for a publishable measurement; set `false`
  only when intentionally running a debug configuration.

```yaml
schema_version: "1.2"
config_status: resolved
source_run_manifest: /srv/bfcl/benchmarks/warehouse-gold/run_manifest.json
```

Relative paths resolve from the evaluation config's own directory. Resolution ends in one `eval_config_hash` taken over the configuration's meaning: referenced files enter as content hashes, candidates are ordered by alias, and absolute paths, output locations, and secret values are excluded. Moving the checkout leaves the hash alone; changing a candidate, a revision, an inference parameter, a limit, the scoring contract, or the source run changes it.

## Step 2: Choose Trace or Trace Plus Executable

```yaml
eval:
  mode: [trace]
```

`[trace]` measures whether the candidate proposed the expected calls, with the expected arguments, grouping, and ordering, releasing the tool results the benchmark recorded; it needs the published benchmark and nothing else. `[trace, executable]` measures all of that and additionally that the candidate's calls ran against a live oracle and satisfied the pack's assertions; it needs `source_oracle` naming the exact pack manifest and its concrete `backend.py` or endpoint config.

For executable mode, add the resource explicitly, because the `oracle.kind` field in `run_manifest.json` is lineage only and does not locate a backend for a later process.

```yaml
source_oracle:
  kind: python
  pack_manifest: /srv/bfcl/packs/warehouse_assets/manifest.yaml
  resource: /srv/bfcl/packs/warehouse_assets/backend.py
```

The two modes are not redundant. A candidate can emit well-formed calls on every attempt and still fail a large share of tasks once the backend and the pack's assertions are in the loop, because a well-formed call is not necessarily the right call against the state the task established. Executable publication additionally requires a Gold-eligible source run, since only Gold rows were validated against a real oracle.

## Step 3: Point at the Candidate

A candidate separates two identities: `provider`, `model`, and `api.base_url` name the route a request takes, while `model_identity` names the weights that answered.

```yaml
candidates:
  - alias: candidate_a
    model: <SERVED_MODEL_ID>
    provider: openai_compatible
    provider_api_version: v1
    api:
      base_url: https://candidate.example.com/v1
      api_key_env: BFCL_CANDIDATE_API_KEY
    model_identity:
      source: huggingface
      model: <ORG>/<MODEL>
      revision: <40_TO_64_HEX_COMMIT>
      weights_digest: null
```

Do not fill the identity block by hand. The resolver writes it for you and records what it actually observed:

```bash
uv run python -m nemotron.steps.byob.scripts.resolve_bfcl_model_identity registry \
  --model <ORG>/<MODEL> \
  --revision main
```

`registry` resolves a reference to the commit it currently names, `local --model <NAME> --weights-dir <DIR>` digests weights on disk, and `provider-managed --source <PROVIDER> --model <MODEL>` records a hosted route that publishes neither. Each subcommand accepts `--output` to write the YAML fragment to a file as well.

A branch-style reference such as `main` is refused, because the same configuration would score different weights next month. Leaving both `revision` and `weights_digest` null is allowed and means what it says: the identity resolves as provider-managed, the run is still scored, and it may not publish.

## Step 4: Understand the Gates That Run First

Two gates run before any candidate token is spent, and both fail closed.

**Source verification** re-reads the publication from disk and holds it to what the configuration recorded. It proves that `run_manifest.json` is a commit marker whose bytes still hash to what the configuration resolved, that both tables hash to what the publication declares in all three places that declare them, that the published table selects raw rows without rewriting truth and ships no held-out row, and that every published row decodes into a unique addressable task. For executable mode it also recomputes the whole pack fingerprint, because a helper module the backend imports changes what the oracle does. It writes `source_verification_report.json` on success and `source_verification_failure.json` on refusal, using different names so no reader can mistake a diagnosis for a pass.

**The contamination gate** then decides who may answer which rows. Source verification records every model that read a published row while it was being built, together with the rows it read. Each candidate is compared against that inventory on the strongest available evidence, and the comparison returns one of three verdicts: `different` records nothing, `match` is a violation, and `unknown` means neither side pinned enough to tell. Under the locked publication setting `on_violation: fail_run`, a match refuses the run. Unknown evidence never shrinks a task set on suspicion and never aborts a debug run, but it always blocks publication. The gate writes `contamination_report.json` or `contamination_failure.json`.

Both are rechecked immediately before the first request, because verification and use are separated in time and that gap is exactly where a source or a plan gets replaced.

## Step 5: Select the CLI Envelope

The evaluation configuration carries scoring semantics; a separate envelope carries operational choices, so the envelope cannot change `eval_config_hash`.

Copy `eval.cli.yaml`, whose `execution_backend: direct` runs the evaluation in this
process on this host, beside your resolved configuration:

```bash
cp src/nemotron/steps/byob/bfcl/config/eval.cli.yaml \
  /srv/bfcl/eval/candidate-a/eval.cli.yaml
```

Point `eval_config_path` at the absolute
`/srv/bfcl/eval/candidate-a/eval.yaml`. The alternative,
`eval.launcher.yaml`, sets `execution_backend: nemo_launcher` and submits the
published `nemo_evaluator_bundle` as a native NeMo Evaluator task for exactly one
candidate.

```yaml
schema_version: "1.0"
family: bfcl
stage: eval
eval_config_path: /srv/bfcl/eval/candidate-a/eval.yaml
execution_backend: direct
output_format: human
probe_oracle: true
dry_run: true
```

`output_format: json` emits stable machine-readable run and artifact locations; `human` renders the same payload one key per line. `dry_run: true` verifies the source and the contamination decision and reports authorized task counts without any candidate inference.

For Launcher, copy and resolve the separate envelope:

```bash
cp src/nemotron/steps/byob/bfcl/config/eval.launcher.yaml \
  /srv/bfcl/eval/candidate-a/eval.launcher.yaml
```

Resolve every path in the copied Launcher envelope to an absolute path. In
particular, its shipped `launcher_base_config_path` is repository-relative and
must become `$REPO_ROOT/src/nemotron/steps/eval/model_eval/config/tiny_chat.yaml`;
do not leave that relative path in a file copied under `/srv`.

```yaml
execution_backend: nemo_launcher
eval_config_path: /srv/bfcl/eval/candidate-a/eval.yaml
launcher:
  bundle_root: /srv/bfcl/benchmarks/warehouse-gold/exports/nemo_evaluator_bundle
  native_output_dir: /srv/bfcl/eval/candidate-a/native-output
  adapter_config_path: /srv/bfcl/eval/candidate-a/adapter.yaml
  framework_build_dir: /srv/bfcl/eval/candidate-a/framework
  task_config_path: /srv/bfcl/eval/candidate-a/task.yaml
  launcher_base_config_path: /absolute/path/to/Nemotron/src/nemotron/steps/eval/model_eval/config/tiny_chat.yaml
  launcher_config_path: /srv/bfcl/eval/candidate-a/launcher.yaml
  launcher_output_dir: /srv/bfcl/eval/candidate-a/launcher-output
  submit: false
```

`bundle_root` is optional when the source publication contains the export at its
standard location; setting it explicitly makes the handoff visible. Every
orchestration path must lie outside `bundle_root`, because the bundle is verified by
exact file set and one extra file would fail its next verification.

Step 6 executes the Launcher envelope as a two-phase materialize-then-submit flow.
If the merged configuration names an evaluation container anywhere,
`launcher.evaluation_mounts` becomes mandatory and must use identity mappings
covering the adapter and evaluation configurations, verified source, executable
oracle resources, and output trees.

## Step 6: Run the Preflight, Then the Evaluation

### Direct Backend

```bash
uv run nemotron steps run byob/bfcl \
  -c /srv/bfcl/eval/candidate-a/eval.cli.yaml
```

With `dry_run: true` this performs no candidate inference and commits no artifacts. Resolve any contamination finding here rather than weakening the gate, then set `dry_run: false` and run the same command again.

### NeMo Evaluator Launcher

The Launcher path imports the generated benchmark in two parts: `eval.yaml` binds
the publication through `source_run_manifest`, while `eval.launcher.yaml` gives
NeMo Evaluator the compatible exported bundle. Confirm that both inputs exist:

```bash
test -f /srv/bfcl/benchmarks/warehouse-gold/run_manifest.json
test -f \
  /srv/bfcl/benchmarks/warehouse-gold/exports/nemo_evaluator_bundle/bundle.json
```

Then run the following sequence.

1. Keep `launcher.submit: false` and materialize the native task:

   ```bash
   uv run nemotron steps run byob/bfcl \
     -c /srv/bfcl/eval/candidate-a/eval.launcher.yaml
   ```

   The command verifies the publication and bundle, then prints
   `framework_package`, `adapter_config_path`, `task_config_path`, and
   `launcher_config_path`. It does not submit a job yet.

2. Install the printed `framework_package` in the environment that will run NeMo
   Evaluator. For a local Launcher environment:

   ```bash
   uv pip install <FRAMEWORK_PACKAGE_PRINTED_BY_STEP_1>
   ```

   If Launcher uses an evaluation container, install that package in the image
   instead. Also configure identity `launcher.evaluation_mounts` as described in
   Step 5 so the container can access all absolute paths.

3. Change only `launcher.submit` to `true` in `eval.launcher.yaml`, export the
   candidate credential, and submit:

   ```bash
   export BFCL_CANDIDATE_API_KEY='<credential>'
   uv run nemotron steps run byob/bfcl \
     -c /srv/bfcl/eval/candidate-a/eval.launcher.yaml
   ```

   A successful submission prints `status: launcher_submitted`,
   `launcher_invocation_id`, and `submitted_launcher_config_path`. Use the
   invocation ID with the Launcher deployment's normal job-status interface.
   Submission is asynchronous; `launcher_submitted` means accepted, not scored.

:::{tip}
Start with `limits.max_parallel_tasks: 1` and raise it only after confirming the endpoint's concurrency and rate limits. Any such change produces a different `eval_config_hash`, which is correct: it is a different measurement.
:::

## Step 7: Read the Artifacts

A completed run writes an immutable set directly into `outputs.output_dir`, with no
intermediate subdirectory:

```text
<outputs.output_dir>/
├── resolved_eval_config.json
├── source_verification_report.json
├── contamination_report.json
├── candidate_io_cache.jsonl
├── tool_trace_cache.jsonl        # executable mode only
├── eval_report.json
├── eval_task_results.parquet
└── eval_manifest.json
```

Confirm completion before consuming or exporting the score:

```bash
EVAL=/srv/bfcl/eval/candidate-a/output
test -f "$EVAL/eval_report.json"
test -f "$EVAL/eval_task_results.parquet"
test -f "$EVAL/eval_manifest.json"
```

`eval_manifest.json` is written last and binds the source, plan, and candidate aggregate identities to the byte hashes of the report, the task table, and the required caches. Read trace and executable numbers separately: a trace aggregate reports one pass rate per gate over tasks, while an executable aggregate reports per-call accuracies, so `arguments_pass_rate` and `argument_accuracy` are deliberately different names for incomparable measurements. Every aggregate declares the scope it measured, and a report whose aggregates mix scopes is refused. A metric no task applied is reported as not applicable with a stable reason code rather than as a vacuous zero, and a failed gate carries a failure class, so an unreachable endpoint reads as infrastructure rather than as weak tool use.

For a Launcher run, wait for the Launcher job to finish, then read the same canonical
BFCL artifacts from `outputs.output_dir` in `eval.yaml`. The native adapter also
writes these two files to `launcher.native_output_dir`:

```text
<launcher.native_output_dir>/
├── nemo_evaluator_results.json
└── nemo_native_adapter_manifest.json
```

`nemo_evaluator_results.json` is the NeMo `EvaluationResult` consumed by Launcher.
`nemo_native_adapter_manifest.json` binds it to the BFCL report hash, source,
candidate, and evaluation configuration. If the native task fails before producing
a result, inspect `nemo_native_adapter_failure.json` in the same directory and the
logs under `launcher.launcher_output_dir`.

To print the canonical BFCL aggregate metrics after either backend completes:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

report = json.loads(
    Path("/srv/bfcl/eval/candidate-a/output/eval_report.json").read_text()
)
for candidate in report["candidates"]:
    print(candidate["alias"], candidate["scope"])
    print("tasks:", candidate["task_count"])
    for name, metric in candidate["metrics"].items():
        print(f"  {name}: {metric['value']} "
              f"({metric['numerator']}/{metric['denominator']})")
PY
```

Replace the example path with the exact `outputs.output_dir` from `eval.yaml`.
For audit or publication, retain the entire artifact directory and treat
`eval_manifest.json`, not the printed summary, as the completion marker.

## Step 8: Export the Benchmark and Results

To hand off a reproducible result, export the publication and evaluation together.
Do not send only `eval_report.json`, because it does not contain the source and cache
hashes needed to verify the score:

```bash
PUB=/srv/bfcl/benchmarks/warehouse-gold
EVAL=/srv/bfcl/eval/candidate-a/output

uv run python -m nemotron.steps.byob.scripts.archive_bfcl_release \
  --release-dir "$PUB" \
  --evaluation-dir "$EVAL" \
  --output-dir /srv/bfcl/bundles \
  --bundle-name warehouse-gold-candidate-a
```

The resulting content-addressed bundle is the recommended machine-verifiable export.
For downstream analysis, read `eval_task_results.parquet`; for a concise human-facing
summary, use `eval_report.json`.

## What Makes a Run Publication-Eligible

With `publication.requested: true` the loader requires every locked gate: `scoring.argument_matching: schema_then_canonical` with call order and grouping respected and `allow_llm_repair: false`; `scoring.task_success: all_applicable_gates`; `contamination.enforce: true` with `on_violation: fail_run` and `comparison_set: common_intersection`; every artifact under `outputs` enabled; pinned weights on every candidate, since a provider-managed identity names a route the provider may re-point; and, for executable mode, a Gold-eligible source run.

Relaxing any of those is allowed only with `publication.requested: false`, and the configuration then reports each weakened field in `non_publication_reasons`.

## Common Failures

| Symptom | What it means |
| --- | --- |
| Refused before any request | Validation reports every independent violation in one pass. Fix the fields named, starting with the first in file order. |
| The endpoint answered 401 or 403 | Every task presents the same credential, so this is a configuration fault and the run stops on the first refusal. No completion is cached, so a rerun with a working key re-contacts the endpoint. |
| The source hash moved | Something wrote into the publication tree during the evaluation, usually a regeneration into the same directory. Verify again from a clean tree. |
| A finished artifact set exists | Immutable results are never overwritten. Use a new output directory. |

Every failure leaves through one published exit status: `2` for a configuration you must edit, `3` for a setup, source, scoring, or aggregation refusal, `4` for contamination or answer-key exposure, `5` for a candidate-endpoint failure, `6` for live oracle or assertion infrastructure, and `7` for an immutable artifact that already holds different evidence. Exit `7` also
covers a publication-policy refusal raised against an existing artifact set, which you
fix in the eval config rather than in the output tree.

## Next Steps

- Field-by-field details: {doc}`../reference/eval-config`, and symptom-to-fix entries in {doc}`../reference/troubleshooting`.
- What each gate and metric means: {doc}`../explanation/evaluation`, with the normative contract at [`src/nemotron/steps/byob/references/bfcl-eval-scoring-contract.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/byob/references/bfcl-eval-scoring-contract.md).
