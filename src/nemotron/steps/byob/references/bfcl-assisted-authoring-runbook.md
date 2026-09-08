# Assisted authoring runbook: source intake to evaluation

This is the step-by-step runbook for the supported assisted-authoring conventional-source flow.
It carries every command, refusal, and recovery. For a short orientation to the same flow —
what the bundled demo simulates, what it executes for real, and how to read its scores —
read [bfcl-assisted-authoring-walkthrough.md](bfcl-assisted-authoring-walkthrough.md) first.

The flow is:

```text
reviewed local Python source + domain brief + probe plan
  -> transport-neutral evidence and A2 certification
  -> model-exposure authorization
  -> evidence approval
  -> model-authored coverage/case/template/assertion proposals
  -> reviewed semantic supplement
  -> candidate Oracle Pack assembly
  -> fresh Gold validation
  -> release review, approval, and immutable freeze
  -> fresh publication
  -> candidate evaluation
```

The shipped runnable example uses a deterministic library source and the
`local_python` adapter, because local Python has a verified publication adapter.
An HTTP package can currently reach intake, drafting, review, and freeze, but
publication is intentionally refused until its publication adapter exists.

## 1. Understand what assisted authoring means

The authoring model may propose:

- a tool coverage plan;
- validation cases;
- task-template plans;
- declarative trace/path assertion specifications.

It may not:

- change `backend.py`, endpoint behavior, tool schemas, or fixtures;
- certify its own output;
- invent fixture bindings or hidden business truth;
- approve model exposure or release;
- bypass executable Gold validation;
- use target-model answers to select or repair benchmark rows.

The reviewed semantic supplement still supplies slot bindings, conversation
policies, localized user turns, and other semantics that evidence alone cannot
prove.

This differs from **model paraphrasing**. The shipped demo uses a model during
pack authoring but does not enable a second model to paraphrase the published
task surfaces. Add paraphrasing only as a separate, explicitly
configured generation role after the generated pack is known to satisfy the
required diversity and publication constraints.

## 2. Choose the demo level

The runnable script supports two authoring-model modes:

| Mode | Authoring model | Human decisions | Evaluation candidate | Purpose |
| --- | --- | --- | --- | --- |
| `scripted` | Canned structured responses | Simulated and printed | Loopback expected-reply server | Verify all plumbing without credentials |
| `live` | Real configured model endpoint | Simulated and printed | Loopback expected-reply server | Exercise real authoring prompts |

Both modes execute real source probes, certification, grounding, candidate-pack
assembly, Gold validation, freeze, publication, and BFCL trace scoring.

The built-in loopback candidate reads the benchmark’s recorded assistant turns.
A clean score of `1.0` proves the generated benchmark is passable and the scorer
is wired; it is not evidence of real model quality. Section 12 replaces it with
an independent OpenAI-compatible candidate.

## 3. Work from the repository root

```bash
cd /path/to/Nemotron
export NEMOTRON_ROOT="$PWD"
```

The main implementation and operator references are:

```text
scripts/bfcl_assisted_authoring_demo.py
src/nemotron/steps/byob/references/bfcl-authoring-user-guide.md
src/nemotron/steps/byob/references/bfcl-authoring-support-matrix.md
src/nemotron/steps/byob/references/bfcl-authoring-release-v2.md
```

## 4. Install dependencies

Use Python 3.11–3.13:

```bash
uv sync --extra byob
```

Confirm both entry points:

```bash
uv run python scripts/bfcl_assisted_authoring_demo.py --help
uv run python -m nemotron.steps.byob.scripts.bfcl_author --help
```

The local Python rollout flag is set by the demo script. A custom operator flow
must explicitly enable only the reviewed adapter policy it uses.

For the standalone guided commands shown below:

```bash
export BFCL_ENABLE_LOCAL_PYTHON=1
```

Without this flag, or an equivalent reviewed rollout policy, intake fails with
`adapter_rollout_disabled`.

## 5. Run the credential-free baseline

Choose a path that does not exist:

```bash
export BFCL_DEMO_ROOT="${TMPDIR:-/tmp}/bfcl-assisted-authoring-walkthrough"
test ! -e "$BFCL_DEMO_ROOT"
```

Run all nine stages:

```bash
uv run python scripts/bfcl_assisted_authoring_demo.py \
  --workdir "$BFCL_DEMO_ROOT"
```

The script refuses an existing work directory. This prevents old evidence,
approvals, caches, or generated files from being mistaken for a fresh run.

## 6. Step 1 — source intake and A2 certification

The demo first materializes a reviewed local Python source package:

```text
$BFCL_DEMO_ROOT/library-source/
├── backend.py
├── tools.json
├── fixtures.json
└── dependency-lock.json
```

It also creates:

- `domain-brief.txt`;
- `probe-plan.json`;
- a temporary Ed25519 certification key pair.

The probe plan covers every published tool, a structured error, a mutation, and
a controlled timeout. Intake runs those probes in isolated processes and may
attain A2 only from observed outcomes. To author these two by hand for a source
of your own, copy the shape from
[`bfcl-probe-plan.example.json`](bfcl-probe-plan.example.json) and
[`bfcl-domain-brief.example.txt`](bfcl-domain-brief.example.txt).

Equivalent guided command shape:

```bash
uv run python -m nemotron.steps.byob.scripts.bfcl_author \
  --ci author \
  --workspace "$BFCL_DEMO_ROOT/workspace" \
  --source "$BFCL_DEMO_ROOT/library-source" \
  --brief "$BFCL_DEMO_ROOT/domain-brief.txt" \
  --pack-id tiny_library \
  --pack-version 0.1.0 \
  --required-tier A2 \
  --held-out-not-applicable-reason "The catalogue is public reference data." \
  --held-out-reviewed-by reviewer@example.test \
  --certification-private-key "$BFCL_DEMO_ROOT/certification-private.pem" \
  --certification-key-id bfcl-demo \
  --probe-plan "$BFCL_DEMO_ROOT/probe-plan.json"
```

Key outputs:

```text
workspace/intake/
├── adapter_certification.json
├── evidence_bundle.json
├── source_observations.json
├── model_exposure_subject.json
├── domain_brief.source.txt
├── domain_brief_redaction.json
├── held_out_redaction.json
└── intake_provenance.json
```

Stop if `adapter_certification.json` does not report `attained_tier: A2`.
Approval cannot raise a certification tier.

## 7. Steps 2–3 — authorization and evidence approval

These are distinct trust boundaries:

1. the source owner authorizes the exact redacted evidence subject for model
   exposure;
2. a reviewer approves the exact source and normalized evidence digests for
   drafting.

The demo prints both as `[simulated human review]`. In a real run, the named
people must inspect the artifacts before supplying their identities.

Outputs:

```text
workspace/exposure_authorization.json
workspace/evidence_approval.json
```

Do not reuse either file after source, brief, redaction, observations,
certification, or resolved authoring config changes. Digest drift makes the
approval stale.

## 8. Step 4 — Model drafting

Drafting issues four bounded structured requests:

1. coverage plan;
2. validation-case proposals;
3. task-template proposals;
4. assertion specifications.

With the default `scripted` mode, local canned responses are still passed
through the real schema, grounding, blocker, compilation, provenance, and cache
logic. The expected output is:

```text
workspace/drafting/
├── draft_provenance.json
├── authoring_io_cache.jsonl
└── drafts/                 # structured stage outputs
```

Unknown tools, unsupported assertions, ungrounded arguments, malformed output,
or cache conflicts fail closed.

## 9. Run a live authoring model

Use a fresh work directory; do not rerun `all` over the scripted workspace:

```bash
export BFCL_LIVE_DEMO_ROOT="${TMPDIR:-/tmp}/bfcl-assisted-authoring-live"
test ! -e "$BFCL_LIVE_DEMO_ROOT"
```

Configure the model provider through Data Designer:

```bash
export DATA_DESIGNER_HOME="/path/to/data-designer-home"
test -f "$DATA_DESIGNER_HOME/model_providers.yaml"
export AUTHOR_MODEL_API_KEY="<secret>"
```

The provider definition must reference the environment-variable name, never the
secret value. Then run:

```bash
uv run python scripts/bfcl_assisted_authoring_demo.py \
  --workdir "$BFCL_LIVE_DEMO_ROOT" \
  --author-model live \
  --model-provider REPLACE_WITH_PROVIDER_NAME \
  --model REPLACE_WITH_MODEL_ROUTE \
  --model-canonical-id REPLACE_WITH_IMMUTABLE_CANONICAL_ID
```

`--model-provider` must match a configured Data Designer provider.
`--model-canonical-id` identifies the authoring model in provenance and must not
be a moving alias.

The demo still simulates the four human decisions. A live model does not turn
those decisions into automatic approvals.

## 10. Steps 5–8 — assemble, validate, review, freeze, publish

### Assemble

The demo adds a reviewed supplement containing the semantics the model is not
authorized to invent, then runs candidate-pack assembly.

Output:

```text
workspace/candidate/
├── pack/
│   ├── manifest.yaml
│   ├── tools.json
│   ├── backend.py
│   ├── fixtures.json
│   ├── task_templates.yaml
│   ├── validation_cases.yaml
│   └── assertions.py
└── candidate_pack_provenance.json
```

### Validate

`prepare_bfcl` performs unmocked executable validation and derives the tier:

```text
validation-out/bfcl-demo-validation/stage_cache/oracle_validation_report.json
```

The run stops unless the candidate pack is Gold-eligible.

### Review and freeze

The review packet binds certification, evidence, answers, authoring provenance,
candidate-pack bytes, fresh validation, and both approval boundaries. Release
approval records the complete checklist before freeze seals immutable bytes:

```text
workspace/review_packet.json
workspace/release_approval.json
workspace/release/
├── pack/
└── ... release provenance and reviewed sidecars
```

### Publish

Publication performs another fresh Gold validation and runs `stage=all`.
Successful Stage 12 output is:

```text
workspace/generated/bfcl-demo/
├── benchmark_raw.parquet
├── benchmark.parquet
├── run_manifest.json
└── stage_cache/
```

`run_manifest.json` is written last and is the publication commit marker.

Set convenient paths:

```bash
export BFCL_PUBLICATION="$BFCL_DEMO_ROOT/workspace/generated/bfcl-demo"
export BFCL_MANIFEST="$BFCL_PUBLICATION/run_manifest.json"
test -f "$BFCL_MANIFEST"
```

For a live-authoring run, replace `BFCL_DEMO_ROOT` above with
`BFCL_LIVE_DEMO_ROOT`.

### Evaluate an existing publication without rerunning authoring

If a benchmark was already generated and published, skip Sections 5–10. The
evaluation path is the same regardless of whether its Oracle Pack was authored
manually or through assisted authoring. A usable publication must include
the original `run_manifest.json`, `benchmark.parquet`, and
`benchmark_raw.parquet`; a standalone parquet file is not an evaluation source.

For example, bind to the existing Banking VN publication
`bfcl_banking_vn_gold_v1_1392`:

```bash
cd /path/to/Nemotron
export NEMOTRON_ROOT="$PWD"

# Override this root when the publication is on another persistent mount.
export BFCL_RUN_ROOT="${BFCL_RUN_ROOT:-$HOME/bfcl-runs}"
export BFCL_PUBLICATION="$BFCL_RUN_ROOT/bfcl_banking_vn_gold_v1_1392"
export BFCL_MANIFEST="$BFCL_PUBLICATION/run_manifest.json"

test -f "$BFCL_MANIFEST"
test -f "$BFCL_PUBLICATION/benchmark.parquet"
test -f "$BFCL_PUBLICATION/benchmark_raw.parquet"
```

Executable evaluation requires the exact Oracle Pack recorded by that
publication. For the Banking VN example:

```bash
export BFCL_EXISTING_PACK="$NEMOTRON_ROOT/src/nemotron/steps/byob/data/banking_vn_oracle_pack"

test -f "$BFCL_EXISTING_PACK/manifest.yaml"
test -f "$BFCL_EXISTING_PACK/backend.py"
```

For an existing benchmark produced by this demo, point
`BFCL_EXISTING_PACK` at the preserved frozen pack instead:

```bash
export BFCL_EXISTING_PACK="/absolute/path/to/preserved-workspace/release/pack"
```

Do not use `scripts/bfcl_assisted_authoring_demo.py --stage eval` with only a copied
publication directory. That command expects the complete demo workspace,
including its release pack and loopback-candidate state. For a standalone
existing publication, continue at Section 13 and create an independent
evaluation config with:

```yaml
source_run_manifest: /absolute/path/to/bfcl_banking_vn_gold_v1_1392/run_manifest.json

source_oracle:
  kind: python
  pack_manifest: /absolute/path/to/exact-oracle-pack/manifest.yaml
  resource: /absolute/path/to/exact-oracle-pack/backend.py

outputs:
  output_dir: /absolute/path/outside-the-publication/eval/artifacts
```

Run the preflight from the
[Manual Oracle Pack flow](bfcl-manual-oracle-pack-flow.md#12-run-evaluation-preflight)
before enabling live candidate traffic. Source verification fails closed if the
publication or Oracle Pack has drifted.

### Optional post-freeze paraphrase generation

The generated tiny-library pack does not ship with a validated paraphrase
profile. Do not point it at
`publication.paraphrase.example.yaml`: its 1,392-row target, category budgets, style
axes, and exact-surface constraints are specific to the Banking VN inventory.

To add model-authored wording, create a pack-specific publication config that:

1. points `oracle_pack.manifest_path` at the frozen generated pack;
2. enables both `lineage.roles.paraphrase` and
   `surface_generation.model_paraphrase_enabled`;
3. records a distinct immutable paraphrase-model identity and credential
   environment-variable name;
4. sets task counts and diversity limits reachable by this pack;
5. uses a new output directory and experiment name;
6. reruns fresh `stage=all` validation and publication.

Treat this as a separate generation release. The authoring model and paraphrase
model have different roles and both become contamination inputs for later
candidate evaluation.

## 11. Step 9 — built-in trace evaluation

The full demo automatically starts a loopback OpenAI-compatible candidate and
runs the real BFCL trace evaluator. Its first immutable output directory is:

```text
$BFCL_DEMO_ROOT/eval-1/
├── resolved_eval_config.json
├── source_verification_report.json
├── contamination_report.json
├── candidate_io_cache.jsonl
├── eval_report.json
├── eval_task_results.parquet
└── eval_manifest.json
```

The built-in evaluation is intentionally `mode: [trace]`; it does not execute a
live Oracle during scoring. A clean run should score `1.0` because the loopback
candidate is primed with recorded expected replies.

Inspect the report:

```bash
uv run python - "$BFCL_DEMO_ROOT/eval-1/eval_report.json" <<'PY'
import json
import sys
from pathlib import Path

print(json.dumps(
    json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")),
    indent=2,
    ensure_ascii=False,
))
PY
```

## 12. Prove that the scorer can fail

Select a task ID from `eval_task_results.parquet`, then run evaluation only:

```bash
export WRONG_TASK_ID="REPLACE_WITH_TASK_ID"

uv run python scripts/bfcl_assisted_authoring_demo.py \
  --workdir "$BFCL_DEMO_ROOT" \
  --stage eval \
  --wrong-answer-task "$WRONG_TASK_ID"
```

The script creates `eval-2/` instead of overwriting `eval-1/`. The selected task
returns text where a tool call is expected, so `task_success_rate` must drop and
the task-level result must name the failed gates.

This comparison proves the scorer is sensitive to an incorrect candidate. It
still does not measure a real model.

## 13. Evaluate an independent candidate endpoint

For a real model evaluation, reuse the resolved evaluation procedure in
[Manual Oracle Pack flow](bfcl-manual-oracle-pack-flow.md), with these source
changes. Store `eval.yaml` under
`$BFCL_DEMO_ROOT/external-eval-1/`; the paths below are relative to that
file:

```yaml
source_run_manifest: ../workspace/generated/bfcl-demo/run_manifest.json

source_oracle:
  kind: python
  pack_manifest: ../workspace/release/pack/manifest.yaml
  resource: ../workspace/release/pack/backend.py

eval:
  mode: [trace, executable]

outputs:
  output_dir: ./artifacts
```

Also replace:

- `candidates[].model` with the ID returned by the candidate `/v1/models`;
- `candidates[].api.base_url` and `api_key_env`;
- `candidates[].model_identity` with an immutable revision or weights digest.

The candidate must be independent from every model exposed during authoring,
paraphrasing, judging, or translation. Keep:

```yaml
contamination:
  enforce: true
  on_violation: fail_run
  comparison_set: common_intersection
```

First run the CLI envelope with `dry_run: true`. Proceed with
`dry_run: false` only after source verification, contamination checks, and
Oracle probing pass.

## 14. Important production differences

The one-command script is intentionally demonstrative:

- it creates a tiny library source rather than onboarding a customer package;
- its signing key is generated locally instead of coming from a trusted
  certification service;
- its human approvals and reviewed supplement are constants;
- its default authoring model is scripted;
- its candidate is a benchmark-keyed loopback server;
- its built-in evaluation is trace-only.

A production run must use a reviewed source, controlled certification keys,
real named reviewers, an independently authored semantic supplement, a live
authoring model with immutable identity, and an independent candidate endpoint.

## 15. Failure and recovery

- **Work directory exists:** choose a fresh directory for `--stage all`.
- **Certification below A2:** expand the reviewed probe plan; approval cannot
  upgrade the result.
- **Exposure authorization stale:** rerun authorization against the current
  model-exposure subject.
- **Evidence approval stale:** approve the current source and normalized bundle
  digests.
- **Draft grounding failure:** fix the source evidence, brief, or model proposal;
  do not weaken grounding.
- **Unknown supplement assertion:** add a supported drafted assertion spec and
  compile it before assembly.
- **Candidate pack not Gold:** inspect the fresh validation report and repair
  reviewed semantics or source behavior before review.
- **Release approval stale:** rebuild the review packet and approve its new
  digest.
- **Publication manifest absent:** treat adjacent files as unpublished and rerun
  through the guided publication boundary.
- **External candidate contamination:** select a distinct model identity; do not
  disable the gate for a publishable score.
- **Finished eval exists:** use a new output directory. Immutable results are not
  overwritten.

Frozen release files are read-only. To archive a completed demo:

```bash
chmod -R u+w "$BFCL_DEMO_ROOT"
mv "$BFCL_DEMO_ROOT" "${BFCL_DEMO_ROOT}.archived"
```

Verify both paths before running the archive command.

## 16. Completion checklist

The demo is complete only when:

- source certification reaches A2;
- model exposure and evidence approval bind current digests;
- all four drafting stages complete with provenance;
- the reviewed supplement is bound into the candidate pack;
- unmocked candidate-pack validation is Gold-eligible;
- review and release approvals are current;
- freeze produces an immutable release;
- fresh publication writes `run_manifest.json`;
- clean loopback trace evaluation succeeds;
- sabotaged evaluation lowers the score;
- any external candidate passes preflight before live inference.
