# QA Runbook: GA_v2 Feature Validation

This runbook defines how the features introduced on `GA_v2` are validated before
release signoff. It covers text curation, evaluation, BFCL benchmark creation,
Persona MCQ synthetic data generation, long-context chat SDG, and tokenizer
extension, together with the supporting `nemo_runspec`, environment-template,
step-catalog, and conversion changes those features depend on.

## Contents

- [Audience And Use](#audience-and-use)
- [Status Definitions](#status-definitions)
- [Purpose](#purpose)
- [Scope](#scope)
- [Change-to-Test Traceability](#change-to-test-traceability)
- [Runbook Rules](#runbook-rules)
- [Common Setup](#common-setup)
  - [Environment](#environment)
  - [Cluster Wiring](#cluster-wiring)
  - [SETUP-001 Confirm Branch And Diff](#setup-001-confirm-branch-and-diff)
  - [SETUP-002 Install Root Feature Environment](#setup-002-install-root-feature-environment)
  - [SETUP-003 Install Isolated Long-Context Environment](#setup-003-install-isolated-long-context-environment)
  - [SETUP-004 Snapshot CLI And Step Metadata](#setup-004-snapshot-cli-and-step-metadata)
  - [SETUP-005 Run New Offline Regression Suites](#setup-005-run-new-offline-regression-suites)
- [Text Curation](#text-curation)
  - [CUR-001 Discover Curate Steps, Flow, And Dependencies](#cur-001-discover-curate-steps-flow-and-dependencies)
  - [CUR-002 Run All Six CPU Smoke Paths](#cur-002-run-all-six-cpu-smoke-paths)
  - [CUR-003 Measure, Approve, Apply, And Reject Corpus Drift](#cur-003-measure-approve-apply-and-reject-corpus-drift)
  - [CUR-004 Validate Multilingual Signal And Language-Pack Safety](#cur-004-validate-multilingual-signal-and-language-pack-safety)
  - [CUR-005 Decontaminate Before Building Nested Subsets](#cur-005-decontaminate-before-building-nested-subsets)
  - [CUR-006 Real-Corpus Release Run](#cur-006-real-corpus-release-run)
- [Evaluation](#evaluation)
  - [EVAL-001 Discover Modes, Suites, Tasks, And Profiles](#eval-001-discover-modes-suites-tasks-and-profiles)
  - [EVAL-002 Direct-Mode Internal Dry Run And Provenance](#eval-002-direct-mode-internal-dry-run-and-provenance)
  - [EVAL-003 Output Safety, Collision, And Invalid Task Names](#eval-003-output-safety-collision-and-invalid-task-names)
  - [EVAL-004 Direct Base-Model Endpoint Smoke](#eval-004-direct-base-model-endpoint-smoke)
  - [EVAL-005 Direct Instruct/Chat Endpoint Smoke](#eval-005-direct-instructchat-endpoint-smoke)
  - [EVAL-006 Multilingual Suite Selection](#eval-006-multilingual-suite-selection)
  - [EVAL-007 Multi-Task Failure Diagnostics And Preemption Artifacts](#eval-007-multi-task-failure-diagnostics-and-preemption-artifacts)
  - [EVAL-008 Launcher-Mode Compile And Dry Run](#eval-008-launcher-mode-compile-and-dry-run)
  - [EVAL-009 Backend Profile Matrix](#eval-009-backend-profile-matrix)
  - [EVAL-010 Tokenizer-Extended Checkpoint Integration](#eval-010-tokenizer-extended-checkpoint-integration)
  - [EVAL-011 Agent-Driven Evaluation Workflow](#eval-011-agent-driven-evaluation-workflow)
- [BFCL Benchmark Creation](#bfcl-benchmark-creation)
  - [BFCL-001 Hotel-Domain Assisted Authoring, Publication, And Evaluation](#bfcl-001-hotel-domain-assisted-authoring-publication-and-evaluation)
- [Persona MCQ SDG For SFT](#persona-mcq-sdg-for-sft)
  - [PMC-001 Discover Step, Plugin, And Dependencies](#pmc-001-discover-step-plugin-and-dependencies)
  - [PMC-002 Persona Asset Staging And Secret Safety](#pmc-002-persona-asset-staging-and-secret-safety)
  - [PMC-003 Tiny End-To-End Generation](#pmc-003-tiny-end-to-end-generation)
  - [PMC-004 Deduplication, Voting, And Language Gates](#pmc-004-deduplication-voting-and-language-gates)
  - [PMC-005 Resume, Stage Selection, And Config Identity](#pmc-005-resume-stage-selection-and-config-identity)
  - [PMC-006 Config-Driven Language And Geography Extension](#pmc-006-config-driven-language-and-geography-extension)
  - [PMC-007 Sampling And Reasoning-On/Off Contract](#pmc-007-sampling-and-reasoning-onoff-contract)
  - [PMC-008 Downstream SFT Handoff](#pmc-008-downstream-sft-handoff)
  - [PMC-009 Remote Execution Profiles](#pmc-009-remote-execution-profiles)
  - [PMC-010 Agent-Driven Persona MCQ Workflow](#pmc-010-agent-driven-persona-mcq-workflow)
- [Long-Context Chat SDG](#long-context-chat-sdg)
  - [LCSDG-001 Validate Standalone Installation And CLI](#lcsdg-001-validate-standalone-installation-and-cli)
  - [LCSDG-002 Prepare Corpus And Query-Generation Dry Run](#lcsdg-002-prepare-corpus-and-query-generation-dry-run)
  - [LCSDG-003 Query Preparation From Bring-Your-Own Queries](#lcsdg-003-query-preparation-from-bring-your-own-queries)
  - [LCSDG-004 Retrieval Mode Is Explicit And Fail-Closed](#lcsdg-004-retrieval-mode-is-explicit-and-fail-closed)
  - [LCSDG-005 HTTP Retrieval Adapter Contract](#lcsdg-005-http-retrieval-adapter-contract)
  - [LCSDG-006 Live Five-Row Generation Smoke](#lcsdg-006-live-five-row-generation-smoke)
  - [LCSDG-007 Objective Evaluation And SFT Export](#lcsdg-007-objective-evaluation-and-sft-export)
  - [LCSDG-008 Judge, Re-Thresholding, And Reasoning Policy](#lcsdg-008-judge-re-thresholding-and-reasoning-policy)
  - [LCSDG-009 Resume And Existing Experiment Behavior](#lcsdg-009-resume-and-existing-experiment-behavior)
  - [LCSDG-010 Agent-Driven Long-Context Workflow](#lcsdg-010-agent-driven-long-context-workflow)
- [Super3 Long-Context SFT References](#super3-long-context-sft-references)
  - [LCSFT-001 Static Invariants For 128K And 256K Profiles](#lcsft-001-static-invariants-for-128k-and-256k-profiles)
  - [LCSFT-002 Generic-Step Dry-Run Guard](#lcsft-002-generic-step-dry-run-guard)
  - [LCSFT-003 Packed-Data Contract Review](#lcsft-003-packed-data-contract-review)
- [Tokenizer Extension](#tokenizer-extension)
  - [Tokenizer Group Standing Risk](#tokenizer-group-standing-risk)
  - [Tokenizer Test Data Setup](#tokenizer-test-data-setup)
  - [TOK-001 Discover Workflow, Profiles, And Languages](#tok-001-discover-workflow-profiles-and-languages)
  - [TOK-002 Extend Add Arm With Exact Budget](#tok-002-extend-add-arm-with-exact-budget)
  - [TOK-003 Replace And Expand Arms](#tok-003-replace-and-expand-arms)
  - [TOK-004 Corpus And Dependency Failure Guards](#tok-004-corpus-and-dependency-failure-guards)
  - [TOK-005 Language Profile And Override Precedence](#tok-005-language-profile-and-override-precedence)
  - [TOK-006 Initialize Add/Expand Embeddings](#tok-006-initialize-addexpand-embeddings)
  - [TOK-007 Initialize Replace Embeddings And Method Matrix](#tok-007-initialize-replace-embeddings-and-method-matrix)
  - [TOK-008 Fertility Evaluation On Independent Corpus](#tok-008-fertility-evaluation-on-independent-corpus)
  - [TOK-009 BPB Evaluation Across Vocabularies](#tok-009-bpb-evaluation-across-vocabularies)
  - [TOK-010 Conversion And Downstream CPT Handoff](#tok-010-conversion-and-downstream-cpt-handoff)
  - [TOK-011 Remote Profile Matrix And Release-Scale Smoke](#tok-011-remote-profile-matrix-and-release-scale-smoke)
  - [TOK-012 Agent-Driven Tokenizer Extension Workflow](#tok-012-agent-driven-tokenizer-extension-workflow)
  - [TOK-013 Corpus Loader Contract](#tok-013-corpus-loader-contract)
  - [LCSDG-011 Endpoint Context Budget](#lcsdg-011-endpoint-context-budget)
  - [PMC-011 Asset Staging Cost And Cache Reuse](#pmc-011-asset-staging-cost-and-cache-reuse)
- [Cross-Feature Security And Reproducibility](#cross-feature-security-and-reproducibility)
  - [SEC-001 Credential Scan](#sec-001-credential-scan)
  - [REP-001 Reproducibility Bundle](#rep-001-reproducibility-bundle)
- [Reporting](#reporting)
- [Exit Criteria](#exit-criteria)

## Audience And Use

The reader is a QA engineer executing the cases directly, or an engineer
reproducing a reported result. Cases are command-driven: each one states its
prerequisites, the exact commands to run, the criteria that decide pass or fail,
and the evidence to retain. Run the direct CLI workflow first, validate the
artifacts and the failure behaviour, then repeat the principal workflow through
an agent where an agent case is defined.

Case identifiers are stable and are referenced by defect reports and signoff
records. Do not renumber them.

## Status Definitions

| Status | Meaning |
| --- | --- |
| `PASS` | The command succeeded and every stated success criterion was met. |
| `FAIL` | Product, documentation, packaging, config, schema, safety, or artifact behaviour violates a stated contract. |
| `BLOCKED` | A named external prerequisite is unavailable. Record the owner and the next action. Never use for a reproducible product failure. |
| `NOT RUN` | Not attempted. Never equivalent to signoff. |

## Purpose

This runbook validates the features added on `GA_v2` relative to `main` as a
user would exercise them.

The scope was derived from:

```bash
git fetch origin main GA_v2
git diff --name-status origin/main...origin/GA_v2
git log --oneline --no-merges origin/main..origin/GA_v2
```

The reference comparison is:

- `main`: `f8f332a51879b1440ca823bd2d68bba0788346f5`
- `GA_v2`: `0748d20507049160c6700d56273d5567385c649b`
- Merge base: `f8f332a51879b1440ca823bd2d68bba0788346f5`

Record the actual SHAs for every QA pass. If either branch advances, regenerate
the diff and add coverage for any new files before signoff.

## Scope

In-scope feature groups:

- Text curation:
  - all six registered `curate/*` steps and the unregistered six-step flow;
  - content-derived ingest identity, profiling, approved-policy filtering,
    independent audit, holdout decontamination, and deterministic nested subsets;
  - Unicode-safe signals, explicit language-pack capabilities, corpus/policy
    fingerprints, manifests, ledgers, and cross-step artifact lineage.
- Evaluation:
  - expanded `eval/model_eval` launcher behavior;
  - direct mode on Local and Lepton profiles, executed live; Slurm and
    DGX Cloud/Run:ai profiles compiled and reviewed but not executed;
  - base, instruct, MMLU-ProX, and MILU suites;
  - durable artifacts, provenance, credential redaction, output safety, failure
    diagnostics, and preemption behavior.
- BFCL benchmark creation:
  - `byob/bfcl` discovery and the local-Python, model-assisted authoring flow;
  - source scaffolding from a reviewed tool catalog and domain brief, followed
    by human completion of executable behavior, fixtures, probes, and semantics;
  - measured A2 intake certification, Gold validation, publication, and direct
    benchmark evaluation from the published run manifest.
- SDG for SFT:
  - `sdg/persona_mcq` discovery and local/remote execution;
  - persona asset staging, multilingual question generation, deduplication,
    teacher voting, quality gates, resumability, and downstream SFT handoff.
- Long-context SDG:
  - the standalone `use-case-examples/long-context-chat-sdg` pipeline;
  - query generation/preparation, real and simulated retrieval, multi-turn tool
    trajectories, objective/judge evaluation, provenance, and SFT export;
  - the related Super3 128K/256K SFT topology reference configs, for discovery
    and dry-run validation only.
- Tokenizer extension:
  - `tokenizer_extension/extend` (`add`, `replace`, and `expand`);
  - `tokenizer_extension/init_embeddings` (`baseline`, `subword`, and `focus`);
  - tokenizer fertility evaluation;
  - BPB/perplexity initialization evaluation;
  - language profiles, conversion handoff, remote profiles, and failure guards.

Supporting changes to `nemo_runspec`, environment templates, the step catalog,
and conversion runner are in scope wherever the six feature groups depend on
them.

Out of scope:

- Full training to convergence or proof of downstream model-quality gains.
- Public leaderboard reproduction or comparison of scores produced with
  different tokenizers, endpoint types, harness images, or generation settings.
- Generating the full 100,000-question Persona MCQ default dataset during smoke
  testing.
- Building, deploying, or operating the external retriever used by long-context
  SDG.
- Production endpoint, cluster, quota, account, or credential provisioning.
- Live execution on Slurm and DGX Cloud/Run:ai. No such cluster is available to
  this effort; those profiles are covered by compile and review only, and their
  live arms are not recorded as `BLOCKED`.
- Executing model-generated code benchmarks unless the QA owner separately
  approves that risk.
- Launching `super3_128k.yaml` or `super3_256k.yaml` with the stock GA_v2 generic
  SFT runner. They are intentionally non-runnable planning references.

## Change-to-Test Traceability

| Changed area | Primary cases |
| --- | --- |
| `src/nemotron/steps/curate/**`, Curator runtime packaging, and curation artifact types | `CUR-001` through `CUR-005` |
| `src/nemotron/steps/eval/model_eval/**`, `src/nemo_runspec/**`, eval profiles | `EVAL-001` through `EVAL-011` |
| `src/nemotron/steps/byob/bfcl/**`, BFCL runtime, assisted-authoring scripts, and function-calling guides | `BFCL-001` |
| `src/nemotron/steps/sdg/persona_mcq/**`, Persona plugin and profiles | `PMC-001` through `PMC-011` |
| `use-case-examples/long-context-chat-sdg/**` | `LCSDG-001` through `LCSDG-011` |
| Super3 128K/256K SFT configs and metadata | `LCSFT-001` through `LCSFT-003` |
| `src/nemotron/steps/tokenizer_extension/**`, converter change and profiles | `TOK-001` through `TOK-013` |

## Tooling Notes

Three environment behaviours cost time during execution and can be mistaken for
product failures. None is a defect in this repository.

- **A Hugging Face streaming fetch can exit 134 after succeeding.** Interpreter
  teardown after `load_dataset(..., streaming=True)` can raise
  `PyGILState_Release: thread state must be current when releasing`, which
  aborts the process with SIGABRT once every row has already been written.
  Verify a fetch by reading back the row count and checksum of what landed, not
  by the exit status.
- **The `fasttext` Python wrapper cannot predict under NumPy 2.**
  `model.predict()` raises `ValueError: Unable to avoid copy while creating an
  array as requested` from inside `FastText.py`. This affects QA diagnostic
  scripts that call fasttext directly; the curate steps wrap the model
  differently and are unaffected. Diagnose language composition by measuring
  script share over the kept and rejected sets instead.
- **`lep job log` takes the job id via `-i`.** There is no `-n` or `--name`
  option; `lep job list` shows the id. Retrieval may still return nothing — see
  the log-collection criterion in `EVAL-009`.

## Runbook Rules

- Run root-project commands from the repository root unless a case explicitly
  changes into the long-context example.
- Use a fresh `QA_ROOT` and fresh output directory for every pass.
- Do not edit checked-in YAML or TOML. Copy configurations to `QA_ROOT` before
  modification.
- Keep the standalone long-context example in its own environment. Its
  `pyproject.toml` pins Data Designer 0.7.0; the root `data-sdg` extra pins the
  0.5.x line.
- Smoke with bounded samples before any full dataset or benchmark.
- Tiny and smoke runs validate plumbing, schemas, paths, safety, and executor
  behavior; they are not evidence of model or dataset quality.
- Use a new experiment/output name for every changed configuration. Only reuse
  a name when the case is explicitly testing resume behavior.
- Never put raw credentials in YAML, command-line overrides, logs, manifests,
  job specs, or the QA report.
- For a missing credential, backend, GPU, model endpoint, retriever, dataset,
  quota, or mount, mark the live case `BLOCKED` and record the exact prerequisite.
  Offline tests and dry runs must still be completed.
- Capture commands, stdout/stderr, exit status, resolved config, branch SHAs,
  package versions, job IDs, artifact paths, row counts, and redacted secret
  evidence for every case.
- A case passes only when both its command and all stated success criteria pass.
  A zero exit status alone is insufficient.
- Before scheduling any remote case, confirm the QA operator can READ the durable
  results path from wherever they will assess it. A remote job writes
  `summary.json`, `run_manifest.json`, and logs to a cluster mount; if that mount
  is not visible to the operator and job logs cannot be fetched, the run can be
  observed to complete but none of its criteria can be verified, and the case
  cannot be recorded as `PASS`. Arrange a mounted path, an artifact copy-back, or
  a reader job as part of setup — not after the first submission.

## Common Setup

### Environment

```bash
set -o pipefail
export QA_RUN_ID="${QA_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
export QA_ROOT="${QA_ROOT:-/tmp/nemotron-ga-v2-qa-$QA_RUN_ID}"
mkdir -p "$QA_ROOT"/{logs,metadata,artifacts,configs,data}

export NEMOTRON_RUN_DIR="$QA_ROOT/artifacts/nemotron-runs"
export HF_HOME="$QA_ROOT/artifacts/hf-cache"
export DATA_DESIGNER_MANAGED_ASSETS_PATH="$QA_ROOT/artifacts/data-designer"

export QA_BASE_SHA="$(git rev-parse origin/main)"
export QA_HEAD_SHA="$(git rev-parse HEAD)"
git diff --name-status origin/main...HEAD > "$QA_ROOT/metadata/main-to-head.files.txt"
git log --oneline --no-merges origin/main..HEAD > "$QA_ROOT/metadata/main-to-head.commits.txt"
git status --short --branch > "$QA_ROOT/metadata/git-status.txt"
```

Credential and service variables used later:

```bash
export LEPTON_API_TOKEN="${LEPTON_API_TOKEN:-}"
export NVIDIA_API_KEY="${NVIDIA_API_KEY:-$LEPTON_API_TOKEN}"
export NGC_API_KEY="${NGC_API_KEY:-}"
export HF_TOKEN="${HF_TOKEN:-}"

export QWEN_API_BASE="${QWEN_API_BASE:-https://xtcr3jss-rkalani-qwen-3-5-122b-sdg-gcp.xenon.lepton.run/v1}"
export OSS_API_BASE="${OSS_API_BASE:-https://xtcr3jss-gpt-oss-120b-rkalani-az.xenon.lepton.run/v1}"
export GEMMA_API_BASE="${GEMMA_API_BASE:-https://xtcr3jss-gemma-4-31b-it-vllm-ravirajj-az-32k.xenon.lepton.run/v1}"
export NANO_API_BASE="${NANO_API_BASE:-https://xtcr3jss-nano-30b-chat.xenon.lepton.run/v1}"
export SUPER_API_BASE="${SUPER_API_BASE:-https://xtcr3jss-super-120b-sdg.xenon.lepton.run/v1}"

export EVAL_ENDPOINT_URL="${EVAL_ENDPOINT_URL:-}"
export EVAL_MODEL_HANDLE="${EVAL_MODEL_HANDLE:-}"
export EVAL_TOKENIZER="${EVAL_TOKENIZER:-}"
export EVAL_API_KEY_NAME="${EVAL_API_KEY_NAME:-ENDPOINT_TOKEN}"
export ENDPOINT_TOKEN="${ENDPOINT_TOKEN:-}"
export EVAL_PROFILE="${EVAL_PROFILE:-lepton_eval_direct}"
# For a remote run, override this with a durable path mounted at the same path
# inside the selected worker profile.
export EVAL_RESULTS_ROOT="${EVAL_RESULTS_ROOT:-$QA_ROOT/artifacts/eval}"

export QA_BACKEND="${QA_BACKEND:-lepton}"
export QA_ENV_FILE="${QA_ENV_FILE:-$QA_ROOT/configs/env.$QA_BACKEND.toml}"

export ASSISTANT_ENDPOINT="${ASSISTANT_ENDPOINT:-$OSS_API_BASE}"
export USER_MODEL_ENDPOINT="${USER_MODEL_ENDPOINT:-$GEMMA_API_BASE}"
export EMBEDDING_ENDPOINT="${EMBEDDING_ENDPOINT:-}"
export RETRIEVAL_ENDPOINT="${RETRIEVAL_ENDPOINT:-}"
export ASSISTANT_API_KEY="${ASSISTANT_API_KEY:-$LEPTON_API_TOKEN}"
export USER_MODEL_API_KEY="${USER_MODEL_API_KEY:-$LEPTON_API_TOKEN}"
export EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-}"
```

`NVIDIA_API_KEY` is the credential variable name consumed by the checked-in
Persona MCQ model configs; for these Lepton routes it carries the same value as
`LEPTON_API_TOKEN`. `NGC_API_KEY` remains separate because it authorizes persona
asset downloads, not the Lepton model deployments.

Lepton model route snapshot used by this runbook on 2026-09-15:

| Role/name | OpenAI-compatible API root | Served model id |
| --- | --- | --- |
| Qwen author/teacher | `https://xtcr3jss-rkalani-qwen-3-5-122b-sdg-gcp.xenon.lepton.run/v1` | `Qwen/Qwen3.5-122B-A10B` |
| GPT-OSS teacher/assistant | `https://xtcr3jss-gpt-oss-120b-rkalani-az.xenon.lepton.run/v1` | `openai/gpt-oss-120b` |
| Gemma teacher/user simulator and BFCL candidate | `https://xtcr3jss-gemma-4-31b-it-vllm-ravirajj-az-32k.xenon.lepton.run/v1` | `google/gemma-4-31B-it` |
| Nemotron Nano alternate | `https://xtcr3jss-nano-30b-chat.xenon.lepton.run/v1` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| Nemotron Super alternate | `https://xtcr3jss-super-120b-sdg.xenon.lepton.run/v1` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` |

These are mutable deployment routes, not weight pins. Before any live case,
query `<API root>/models` with `LEPTON_API_TOKEN` and require the exact served
id in the third column. A route that returns `401`/`403`, times out, or reports a
different id is not ready for spend. Append no `/chat/completions` suffix to
these variables; Data Designer and the BFCL candidate client add it.

### Cluster Wiring

A generated `env.toml` is a template, not a runnable profile. Three fields are
placeholders and every remote case fails until they are filled in:

```bash
# 1. node group -- use the node group NAME, not the display ID.
#    `lep node-group list` shows both; nemo-run matches on the name.
#    Wrong value fails ONLY at submit with
#    "Could not find node group that matches requested ID".
export LEPTON_NODE_GROUP="<name, e.g. az-sat-lepton-001>"

# 2. the shared mount, copied from a working deployment on the same cluster
export NEMOTRON_HOST_MOUNT="<host path>"
export NEMOTRON_MOUNT_FROM="<mount source, e.g. node-nfs:amlfs>"
export NEMOTRON_WORKSPACE="/mnt/lustre-shared"
```

Then edit `node_group` in `$QA_ENV_FILE` to that name.

Neither `env.toml` generation nor `--dry-run` contacts the scheduler, so both
pass with an invalid profile. **A live submission is the only real check of a
profile.** Compile first to catch config errors, then submit — and treat a
submit-time rejection as a profile defect, not a transient error.

### SETUP-001 Confirm Branch And Diff

Prerequisites: Git access to the fork.

```bash
git fetch origin main GA_v2
git switch GA_v2
git merge-base --is-ancestor origin/main HEAD
git diff --check origin/main...HEAD
git diff --stat origin/main...HEAD | tee "$QA_ROOT/logs/diff-stat.txt"
```

Success criteria:

- The active branch is `GA_v2` and contains `origin/main` as an ancestor.
- `git diff --check` exits 0.
- The captured diff contains all four in-scope feature groups.
- Any difference from the authoring SHAs is recorded and reviewed for added
  test coverage.

Evidence to collect: branch status, three SHAs, diff stat, changed-file list,
and commit list.

### SETUP-002 Install Root Feature Environment

Prerequisites: `uv`, network access to package indexes, and sufficient disk.

```bash
uv sync \
  --extra evaluator \
  --extra data-sdg \
  --extra tokenizer-extension \
  --group dev

uv run python - <<'PY'
import importlib.metadata as md

for package in (
    "nemotron",
    "nemo-evaluator-launcher",
    "data-designer",
    "sentence-transformers",
    "indic-nlp-library",
    "fasttext-wheel",
    "accelerate",
):
    print(package, md.version(package))
PY
```

Success criteria:

- The sync exits 0 from a clean checkout.
- Evaluator Launcher is in the supported `>=0.2.6,<0.3` range.
- Root Data Designer is in the supported `>=0.5.9,<0.6` range.
- The tokenizer optional dependencies import in the same environment.
- `accelerate` is present. `tokenizer_extension/eval_init` sets
  `device_map="auto"` whenever CUDA is visible, so without it every BPB run
  on a GPU host fails at model load. A CPU-only host hides this.

Evidence to collect: full sync log and package-version output.

### SETUP-003 Install Isolated Long-Context Environment

Prerequisites: `SETUP-002`; Python 3.12.

```bash
export LC_ROOT="$PWD/use-case-examples/long-context-chat-sdg"
cd "$LC_ROOT"
uv sync --extra dev --extra lancedb
uv run python - <<'PY'
import importlib.metadata as md
import lancedb
import scipy
import sklearn

print("data-designer", md.version("data-designer"))
print("numpy", md.version("numpy"))
print("scikit-learn", sklearn.__version__)
print("scipy", scipy.__version__)
print("lancedb", lancedb.__version__)
from sklearn.cluster import KMeans
print(KMeans)
PY
cd -
```

Success criteria:

- The example resolves Data Designer exactly to 0.7.0.
- `scikit-learn` resolves within `>=1.8,<1.9`.
- The optional LanceDB source dependency imports in this full QA environment.
- `KMeans` imports without a missing-module or binary-symbol error.
- The example installation does not replace the root environment.
- `git status --porcelain` is empty afterwards. A lock file written by this
  sync leaves the tree dirty and every later eval manifest then records
  `git_dirty: true`, which weakens REP-001 provenance.

Evidence to collect: sync log, environment path, package versions, and import
output. An error such as a missing `sklearn` or a SciPy binary-symbol import is
an environment failure; recreate the example environment before product triage.

### SETUP-004 Snapshot CLI And Step Metadata

```bash
for BACKEND in lepton slurm dgxcloud; do
  uv run nemotron steps run env/env_toml -c "$BACKEND" \
    output_path="$QA_ROOT/configs/env.$BACKEND.toml"
done

uv run nemotron --help > "$QA_ROOT/metadata/nemotron-help.txt"
uv run nemotron steps --help > "$QA_ROOT/metadata/steps-help.txt"
uv run nemotron steps list > "$QA_ROOT/metadata/steps-list.txt"

for STEP in \
  byob/bfcl \
  eval/model_eval \
  sdg/persona_mcq \
  sft/megatron_bridge \
  tokenizer_extension/extend \
  tokenizer_extension/init_embeddings \
  tokenizer_extension/evaluate \
  tokenizer_extension/eval_init
do
  uv run nemotron steps show "$STEP" --json \
    > "$QA_ROOT/metadata/${STEP//\//_}.json"
done
```

Success criteria:

- Every command exits 0 without a model credential.
- All three backend environment files generate without overwriting a user-owned
  `env.toml`.
- All eight step IDs are discoverable and metadata output is valid JSON.
- `byob/bfcl` advertises the Oracle Pack, benchmark Parquet, run-manifest, and
  evaluation artifact contracts used by `BFCL-001`.
- `sdg/qasynth` is absent and `sdg/persona_mcq` is present.
- The tokenizer category exposes all four steps.
- `sft/megatron_bridge` advertises the long-context reference profiles as
  planning-only, not runnable production profiles.

Evidence to collect: help output, catalog listing, and metadata JSON.

### SETUP-005 Run New Offline Regression Suites

```bash
uv run pytest -q \
  tests/nemo_runspec/test_evaluator.py \
  tests/nemo_runspec/test_lepton_log_collection.py \
  tests/steps/test_model_eval_direct.py \
  tests/steps/test_model_eval_integration_contracts.py \
  tests/steps/byob/test_bfcl_authoring_gold_e2e.py \
  tests/steps/byob/test_bfcl_authoring_release.py \
  tests/steps/byob/test_bfcl_eval_cli_orchestration.py \
  tests/steps/byob/test_bfcl_input_contract_docs.py \
  tests/steps/byob/test_bfcl_pipeline_assets.py \
  tests/steps/sdg/test_persona_mcq.py \
  tests/steps/tokenizer_extension/test_extend_budget.py \
  tests/steps/test_convert_tokenizer_config.py \
  | tee "$QA_ROOT/logs/root-feature-tests.txt"

cd "$LC_ROOT"
uv run pytest -q | tee "$QA_ROOT/logs/long-context-tests.txt"
cd -
```

Success criteria:

- All selected root tests pass.
- All long-context example tests pass in the freshly synced isolated
  environment.
- No test is skipped for a missing declared dependency.
- Deprecation warnings are recorded separately and do not hide test failures.

Evidence to collect: pytest output, duration, environment/package snapshot, and
exit status.

## Text Curation

The curate category provides six independently runnable steps plus a Python
flow driver. The highest-risk contract is not whether a command exits
successfully; it is whether the measured corpus, approved policy, filtered
corpus, audit, decontaminated corpus, and subset tiers retain one verifiable
identity.

Three guards refuse a run that looks reasonable. Each is correct behaviour;
none is a defect. Expect them, and do not file them:

- **Not enough CPU.** The filter builds one Ray stage per gate, so a
  three-threshold policy needs 5.5 CPUs and Ray reserves one more. On an 8-core
  host `ray: {num_cpus: 8}` fits a small policy; `num_cpus: 6` does not. The
  error arrives from inside Curator and names no config key.
- **Parquet is not filter input.** `curate/nemo_curator` reads `.jsonl`,
  `.json` or `.ndjson` only. A real corpus arrives as parquet, so run
  `curate/ingest` first to mint ids and normalise to JSONL — the filter refuses
  the parquet and names the fix. The packaged fixtures are already JSONL, so
  this only appears the first time a case is pointed at a real corpus.
- **Stale output.** The filter does not clear its output directory, so a second
  run into a populated `filtered_jsonl/` is refused rather than appending a
  second corpus that every later step would read as one.
- **Incomparable identity spaces.** Decontamination refuses a holdout that is
  keyed off a field the training split never uses. `shared_id_space` defaults to
  true because the holdout is meant to be a split of the SAME corpus. A holdout
  taken from a separately published benchmark shares no key space with the
  training corpus, so exact identity matching would return a guaranteed zero
  rather than a measured one — near-duplicate detection across two corpora needs
  the similarity pass, not the identity pass.

Use one fresh root for these cases:

```bash
export CURATE_QA="$QA_ROOT/artifacts/curate"
export CURATE_SRC="$PWD/src/nemotron/steps/curate/nemo_curator"
mkdir -p "$CURATE_QA"/{configs,data,logs,metadata,outputs}
```

### CUR-001 Discover Curate Steps, Flow, And Dependencies

Prerequisites: Python 3.11 or newer.

```bash
uv sync --extra curate --extra xenna --group dev

uv run python - <<'PY'
import importlib.metadata as md
import sys

assert sys.version_info >= (3, 11), sys.version
for package in ("nemotron", "nemo-curator", "pyyaml"):
    print(package, md.version(package))
PY

uv run nemotron steps list --category curate \
  | tee "$CURATE_QA/metadata/steps-list.txt"

for STEP in ingest profile nemo_curator audit decontamination subset; do
  uv run nemotron steps show "curate/$STEP" --json \
    > "$CURATE_QA/metadata/curate_${STEP}.json"
done

# The flow is intentionally a Python driver, not a registered step.
! uv run nemotron steps show curate/flow
uv run python -m nemotron.steps.curate.nemo_curator.scripts.run_flow --help \
  > "$CURATE_QA/metadata/flow-help.txt"

uv run pytest -q \
  tests/steps/curate \
  tests/steps/test_curator_runtime_bootstrap.py \
  tests/steps/test_run_cmd_runtime_preflight.py \
  | tee "$CURATE_QA/logs/offline-tests.txt"
```

Success criteria:

- Exactly these six step IDs are discoverable: `curate/ingest`,
  `curate/profile`, `curate/nemo_curator`, `curate/audit`,
  `curate/decontamination`, and `curate/subset`.
- `curate/flow` is not advertised as a registered step, while the module driver
  exposes `--config` and `--plan`.
- The CPU `curate` extra installs the pinned text runtime. It does not silently
  select the CUDA decontamination stack.
- Step manifests declare the actual prepared, filtered, decontaminated,
  policy, manifest, ledger, audit, and subset artifact cascade.
- All curate-focused tests pass with no skip caused by a missing CPU dependency.
  GPU-only skips are recorded separately and never counted as CPU coverage.

Evidence to collect: PR/base SHAs, Python and package versions, step JSON,
flow help, test output, skipped-test reasons, and `uv.lock` checksum.

### CUR-002 Run All Six CPU Smoke Paths

Prerequisites: `CUR-001`. These commands override container-oriented fixture
paths with absolute checkout paths so the same cases run locally.

`curate/nemo_curator` starts a real local Ray cluster (a separate `ray start
--head` subprocess) and connects to it by address rather than running
in-process. When Ray auto-packages the driver's working directory to ship to
that cluster, it respects `.gitignore` with no override — so if this step is
run from a bare `uv run` checkout, Ray silently excludes the whole gitignored
`.venv` (where `cosmos_xenna`/`nemo_curator` actually live) and every worker
task fails with `ModuleNotFoundError: No module named 'cosmos_xenna'`, even
though the driver process itself imports fine. `ingest`, `profile`, `audit`,
`decontamination`, and `subset` are plain CPU steps with no Ray dependency and
are unaffected. Run `curate/nemo_curator` (and any `run_flow` invocation with
`filter` enabled, e.g. `CUR-003`/`CUR-005`) inside
`nvcr.io/nvidia/nemo-curator:26.07` (NeMo Curator 1.3.0) — the same container the `lepton_curate`
profile in `steps/env/env_toml/config/lepton.yaml` uses for production — where
dependencies are system site-packages outside the packaged working directory,
not a gitignored local venv Ray can silently drop.

```bash
export CURATE_FIX="$(realpath "$CURATE_SRC")"
export CURATE_SMOKE="$CURATE_QA/outputs/smoke"
mkdir -p "$CURATE_SMOKE"

uv run nemotron steps run curate/ingest -c tiny \
  input="$CURATE_FIX/profile/data/tiny/*.jsonl" \
  output_dir="$CURATE_SMOKE/ingest"

uv run nemotron steps run curate/profile -c en \
  output_dir="$CURATE_SMOKE/profile"

# The filter starts Ray, and Ray resolves its worker interpreter independently
# of the launcher. Without both extras the worker dies with
# `ModuleNotFoundError: No module named 'cosmos_xenna'` from inside a Ray task.
uv run --extra curate --extra xenna nemotron steps run curate/nemo_curator -c tiny \
  input_glob="$CURATE_FIX/profile/data/tiny/*.jsonl" \
  output_dir="$CURATE_SMOKE/filter" \
  emit_manifest="$CURATE_SMOKE/filter/run_manifest.json" \
  emit_ledger="$CURATE_SMOKE/filter/curation_ledger.json"

uv run nemotron steps run curate/audit -c tiny \
  target_glob="$CURATE_FIX/audit/data/tiny/*.jsonl" \
  declared_manifest="$CURATE_FIX/audit/data/tiny/run_manifest.json" \
  digest_root="$CURATE_FIX/audit/data/tiny" \
  output_dir="$CURATE_SMOKE/audit"

uv run nemotron steps run curate/decontamination -c tiny \
  train_glob="$CURATE_FIX/decontamination/data/tiny/train.jsonl" \
  holdout_glob="$CURATE_FIX/decontamination/data/tiny/holdout.jsonl" \
  work_dir="$CURATE_SMOKE/decontamination/cache" \
  output_dir="$CURATE_SMOKE/decontamination"

uv run nemotron steps run curate/subset -c tiny \
  input_glob="$CURATE_FIX/subset/data/tiny/corpus.jsonl" \
  output_dir="$CURATE_SMOKE/subset"

uv run python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["CURATE_SMOKE"])
required = (
    root / "ingest/ingest_report.json",
    root / "profile/profile_report.json",
    root / "profile/candidate_policies.yaml",
    root / "profile/sample_manifest.json",
    root / "filter/run_manifest.json",
    root / "filter/curation_ledger.json",
    root / "audit/audit_report.json",
    root / "decontamination/train_decontaminated.jsonl",
    root / "decontamination/decontamination_report.json",
    root / "subset/plan.json",
    root / "subset/subset_report.json",
)
missing = [str(path) for path in required if not path.is_file()]
assert not missing, missing

manifest = json.loads((root / "filter/run_manifest.json").read_text())
audit = json.loads((root / "audit/audit_report.json").read_text())
decon = json.loads((root / "decontamination/decontamination_report.json").read_text())
assert manifest["producer"]["completed_at"], manifest
assert audit.get("passed") is True, audit
assert "NOT measured" in decon["similarity"]["note"], decon
PY
```

Success criteria:

- Every registered step completes through its real runner, not only `--dry-run`.
- Ingest emits stable IDs plus `ingest_report.json`; reordering or resharding the
  input does not change content-derived IDs.
- Profile emits a report, human-readable summary, sample manifest, and a
  `candidate_policies.yaml` whose `approved` field is false.
- Filter emits a completed manifest and ledger, and preserves the source text
  byte-for-byte rather than truncating it during classification.
- Audit reports `passed: true` for the packaged declaration.
- CPU decontamination explicitly says in `similarity.note` that similarity was
  not measured; it never turns a skipped GPU comparison into a zero-overlap
  claim.
- Subset emits one corpus per requested word budget, and each report labels the
  unit as words because no tokenizer was configured.

Evidence to collect: commands, logs, artifact tree, all reports/manifests,
input/output row counts, text checksums, and selected document IDs.

### CUR-003 Measure, Approve, Apply, And Reject Corpus Drift

Prerequisites: a representative SQA corpus with stable source metadata and a
reviewed language pack. Use at least 1,000 documents; smoke data can validate
plumbing but cannot justify a production threshold.

Use the corpus fetched in `CUR-006`; nothing smaller than that satisfies the
1,000-document minimum, and the fixtures under `$CURATE_FIX` hold 40.

Copy `vi_c4_measure.yaml` and `vi_c4_apply.yaml` into `$CURATE_QA/configs`.
Edit only the copies. Point both at the same corpus, text/ID/source fields,
language, and language-pack root.

Four more settings in those copies are specific to the shipped Vietnamese
example and must be changed too. The first is the one that matters:

- **`steps.filter.language_codes`.** This is NOT the same setting as
  `corpus.language`: the first decides what to KEEP, the second selects the pack
  used to MEASURE. Leaving the example's `[VI]` while pointing the corpus at
  another language removes essentially every document — 2,000 in, 0 kept, with
  the ledger reporting `balanced: true` and the filter step exiting 0. Only the
  audit catches it, and only because it cannot key an empty corpus.
- **`steps.*.models.fasttext_langid`** is the relative path `./models/lid.176.bin`
  and will not resolve. Point it at a real copy (`CUR-006` records the URL and
  expected size) or remove the key to profile without a language breakdown.
- **`steps.profile.models.tokenizer`** pins a Hugging Face repo and revision.
  Warm it once before the run — a cold cache can fail the profile with a
  connection error even when the repo is reachable and ungated.
- **`corpus.id_prefix` and `corpus.source_value`** still say Vietnamese.
- **`steps.subset.token_budgets`** are `[100000000, 500000000]`, sized for the
  full Vietnamese corpus. On a 2,000-document sample every tier is the whole
  corpus.
- **`steps.subset.quality_score_field`** is `__script_ratio`, which only exists
  if `script_ratio` is one of the thresholds you approved. Approve that signal,
  or unset the field, or the subset step fails after the filter has already run.
- **`approve.thresholds`** must be replaced with grid points from YOUR profile,
  and `approver` plus `evidence` are required. `approve.from` alone is refused:
  "an approval that gates nothing is not one".

Size the host before running the apply pass. The filter builds one Ray stage per
gate, so a policy of three thresholds plus the language and word-count gates
needs 8.5 CPUs while an 8-core host exposes 7. Either give the run a larger host
or approve fewer thresholds; the error surfaces from inside Curator and names no
config key. Use separate output roots:
`$CURATE_QA/outputs/measure` and `$CURATE_QA/outputs/apply`. In the apply copy,
set `approve.from` to
`$CURATE_QA/outputs/measure/profile/candidate_policies.yaml`; this avoids
deleting the measurement run's artifacts. Keep profile enabled only in the
measure copy.

```bash
export CURATE_MEASURE_CFG="$CURATE_QA/configs/measure.yaml"
export CURATE_APPLY_CFG="$CURATE_QA/configs/apply.yaml"
export CURATE_MEASURE_OUT="$CURATE_QA/outputs/measure"
export CURATE_APPLY_OUT="$CURATE_QA/outputs/apply"

uv run python -m nemotron.steps.curate.nemo_curator.scripts.run_flow \
  --config "$CURATE_MEASURE_CFG" --plan
uv run python -m nemotron.steps.curate.nemo_curator.scripts.run_flow \
  --config "$CURATE_MEASURE_CFG"

uv run python - <<'PY'
import os
from pathlib import Path
import yaml

root = Path(os.environ["CURATE_MEASURE_OUT"])
candidate = yaml.safe_load((root / "profile/candidate_policies.yaml").read_text())
assert candidate["approved"] is False
assert candidate["corpus"]["fingerprint"].startswith("sha256:")
assert candidate["profile_digest"].startswith("sha256:")
print(root / "profile/profile_summary.md")
PY
```

Review `profile_summary.md`, the per-signal retention curves, rejected samples,
and the combined policy simulation. Then put the selected measured thresholds
and the reviewer evidence into the apply copy's `approve` block.

```bash
uv run python -m nemotron.steps.curate.nemo_curator.scripts.run_flow \
  --config "$CURATE_APPLY_CFG" --plan

# Planning validates the approval but must not publish it.
test ! -e "$CURATE_APPLY_OUT/policy/approved_policy.yaml"

uv run python -m nemotron.steps.curate.nemo_curator.scripts.run_flow \
  --config "$CURATE_APPLY_CFG"

uv run python - <<'PY'
import json
import os
from pathlib import Path
import yaml

root = Path(os.environ["CURATE_APPLY_OUT"])
policy = yaml.safe_load((root / "policy/approved_policy.yaml").read_text())
manifest = json.loads((root / "filtered_jsonl/run_manifest.json").read_text())
flow = json.loads((root / "flow_report.json").read_text())
assert policy["approved"] is True
assert policy["thresholds"]
assert manifest["policy"]["status"] == "approved"
assert manifest["policy"]["thresholds_applied"] == len(policy["thresholds"])
assert flow["policy_promoted"] is True
assert flow["policy_applied"] is True
assert flow["policy_status"] == "approved"
assert flow["audit_passed"] is True
PY
```

For the negative arm, copy the approved config to a new output root and point it
at a copy of the corpus with one document's text changed while retaining its ID.
The run must fail:

```bash
export CURATE_DRIFT_CFG="$CURATE_QA/configs/apply-corpus-drift.yaml"
! uv run python -m nemotron.steps.curate.nemo_curator.scripts.run_flow \
    --config "$CURATE_DRIFT_CFG"
```

Success criteria:

- The measurement flow writes only an unapproved candidate policy.
- Each approved threshold has a documented reviewer decision and comes from a
  signal actually profiled on this corpus. Off-grid choices are called out and
  are not reported with invented retention numbers.
- The approved policy preserves corpus fingerprint, profile digest, scoring
  implementation version, language-pack identity, threshold direction, and
  reviewer evidence.
- The apply manifest and flow report agree that the policy was promoted,
  approved, and applied, and the ledger/audit reconcile every removed row.
- A changed corpus is rejected before filtering applies the old policy. No
  completed filter manifest or approved-policy artifact is published for the
  drifted run.
- `allow_unvalidated_policy` is not used in the release-signoff arm. If tested
  separately, its manifest status is `override_unvalidated`, never `approved`.

Evidence to collect: both configs, plan files, profile report/summary,
candidate and approved policies, corpus fingerprints, reviewer rationale,
filter manifest, ledger, audit and flow reports, and drift-arm error text.

### CUR-004 Validate Multilingual Signal And Language-Pack Safety

Prerequisites: `CUR-001`. The `x-test-*` language packs are implementation
fixtures only; use them to exercise behavior, not as production packs.

```bash
cat > "$CURATE_QA/data/unicode.jsonl" <<'EOF'
{"id":"vi-nfc","source":"qa","text":"Tiếng Việt là ngôn ngữ chính thức của Việt Nam, được viết bằng chữ Quốc ngữ."}
{"id":"vi-nfd","source":"qa","text":"Tiếng Việt là ngôn ngữ chính thức của Việt Nam, được viết bằng chữ Quốc ngữ."}
{"id":"hi","source":"qa","text":"यह एक परीक्षण वाक्य है। भारत एक विशाल देश है जिसमें अनेक भाषाएँ बोली जाती हैं।"}
{"id":"hi-digits","source":"qa","text":"वर्ष २०२६ में कुल १५०० उदाहरण जाँचे गए।"}
EOF

export CURATE_TEST_PACKS="$PWD/tests/steps/curate/fixtures/langpacks"

uv run nemotron steps run curate/profile -c default \
  input_glob="$CURATE_QA/data/unicode.jsonl" \
  output_dir="$CURATE_QA/outputs/profile-vi" \
  language=x-test-vi \
  langpack_dir="$CURATE_TEST_PACKS" \
  'signals=[unicode_alpha_numeric,script_ratio,diacritic_ratio,sentence_end_ratio]'

uv run nemotron steps run curate/profile -c default \
  input_glob="$CURATE_QA/data/unicode.jsonl" \
  output_dir="$CURATE_QA/outputs/profile-hi" \
  language=x-test-hi \
  langpack_dir="$CURATE_TEST_PACKS" \
  'signals=[unicode_alpha_numeric,script_ratio,sentence_end_ratio]'

# Hindi deliberately does not declare diacritic_ratio support.
! uv run nemotron steps run curate/profile -c default \
    input_glob="$CURATE_QA/data/unicode.jsonl" \
    output_dir="$CURATE_QA/outputs/profile-hi-unsupported" \
    language=x-test-hi \
    langpack_dir="$CURATE_TEST_PACKS" \
    'signals=[diacritic_ratio]'

uv run pytest -q \
  tests/steps/curate/test_signals.py \
  tests/steps/curate/test_langpack.py \
  tests/steps/curate/test_registry.py \
  | tee "$CURATE_QA/logs/multilingual-contracts.txt"
```

Success criteria:

- NFC and NFD forms of equivalent Vietnamese text receive equivalent scores;
  profiling never rewrites the original text bytes. Check this by profiling the
  two Vietnamese rows ALONE: every signal must place both documents in one
  histogram bin and report an identical p1 through p99. Profiling them alongside
  the Hindi rows hides the answer, because the spread you see is then between
  languages rather than between normalisation forms.
- `unicode_alpha_numeric` accepts Unicode letter, number, all combining-mark,
  ZWJ, and ZWNJ categories required by Vietnamese and Indic text.
- Script ratio is described as a script-composition signal, not language ID;
  English and Vietnamese sharing Latin script is not reported as separation.
- A named pack-backed signal that the pack does not declare fails clearly. When
  all supported signals are auto-selected, unsupported signals are skipped with
  an explicit warning rather than fabricated scores.
- Language-pack tag, capability declarations, source files, licenses, and
  content hash are validated and recorded in the profile/policy lineage.
- The ASCII-bound content signals are capability-gated, not silently computed.
  Against a pack that does not declare the capability, the profile SKIPS the
  signal and says why, rather than scoring it on a false premise:
  `numbers_ratio` requires `ascii_digits`, `punctuation` requires
  `ascii_punctuation`, and `non_alpha_numeric` requires `ascii_alphabet`.
  Confirm each appears as a `NOTE: skipped <signal>: requires [...]` line in
  `profile_summary.md` for a non-ASCII pack, and that none of them carries a
  threshold in the approve block for that corpus. A signal that scores instead
  of skipping is a defect: `numbers_ratio` reads 0.0000 on Devanagari-digit text
  that reads non-zero in ASCII, and `punctuation` scores 1.000 on Hindi
  terminated correctly with `।`.

Evidence to collect: fixture bytes/checksums, both profile reports and warnings,
per-document scores, pack metadata/hash, unsupported-signal error, and tests.

### CUR-005 Decontaminate Before Building Nested Subsets

This CPU case proves the cross-step order and identity path. It creates a
holdout from one document in the packaged subset corpus, passes the source
corpus through the filter, removes the held-out identity, and only then builds
the tiers.

```bash
export CURATE_CASCADE="$CURATE_QA/outputs/cascade"
sed -n '1p' "$CURATE_FIX/subset/data/tiny/corpus.jsonl" \
  > "$CURATE_QA/data/holdout.jsonl"

cat > "$CURATE_QA/configs/cascade.yaml" <<EOF
corpus:
  input: $CURATE_FIX/subset/data/tiny/corpus.jsonl
  text_field: text
  id_field: id
  source_field: source
  language: en
  langpack_dir: $CURATE_FIX/data/langpacks
output_root: $CURATE_CASCADE
steps:
  ingest: {enabled: false}
  profile: {enabled: false}
  filter:
    enabled: true
    mode: filter
    language_codes: []
    quality_filters: {}
    domains: []
    models: {}
    dataset: null
  audit:
    enabled: true
    mode: all
    comparison_fields: [id]
  decontamination:
    enabled: true
    holdout: $CURATE_QA/data/holdout.jsonl
    skip_similarity: true
  subset:
    enabled: true
    token_budgets: [1000, 2500, 4500]
    tokenizer: null
    quality_score_field: null
    length_bands: [20, 60]
    seed: 0
approve: null
EOF

uv run python -m nemotron.steps.curate.nemo_curator.scripts.run_flow \
  --config "$CURATE_QA/configs/cascade.yaml" --plan

uv run python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["CURATE_CASCADE"])
plan = json.loads((root / "flow_plan.json").read_text())
steps = {entry["key"]: entry for entry in plan["steps"]}
assert steps["decontamination"]["enabled"] is True
assert steps["subset"]["enabled"] is True
assert steps["subset"]["config"]["input_glob"] == str(
    root / "decontaminated/train_decontaminated.jsonl"
)
PY

uv run python -m nemotron.steps.curate.nemo_curator.scripts.run_flow \
  --config "$CURATE_QA/configs/cascade.yaml"

uv run python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["CURATE_CASCADE"])
holdout = json.loads(Path(os.environ["CURATE_QA"] + "/data/holdout.jsonl").read_text())
blocked_id = holdout["id"]

def ids(path):
    return {
        json.loads(line)["id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

decontaminated = ids(root / "decontaminated/train_decontaminated.jsonl")
tier_paths = sorted((root / "subset").glob("budget_*_words/subset.jsonl"))
tiers = [ids(path) for path in tier_paths]
assert tier_paths, "no subset tiers"
assert blocked_id not in decontaminated
assert all(blocked_id not in tier for tier in tiers)
assert all(smaller <= larger for smaller, larger in zip(tiers, tiers[1:])), tier_paths

decon = json.loads((root / "decontaminated/decontamination_report.json").read_text())
subset = json.loads((root / "subset/subset_report.json").read_text())
flow = json.loads((root / "flow_report.json").read_text())
assert "NOT measured" in decon["similarity"]["note"]
assert flow["status"] == "ok"
assert flow["audit_passed"] is True
assert subset["tiers"]
PY
```

Success criteria:

- `flow_plan.json` schedules decontamination before subset and points subset at
  the single committed `train_decontaminated.jsonl`, not the pre-decontamination
  corpus or a directory that can also match report sidecars.
- The held-out document is never modified or emitted as output. Its matching
  training identity is removed and named in `decontamination_report.json`.
- No subset tier reintroduces a removed ID, every smaller tier is a subset of
  every larger tier, and repeated runs with the same seed select identical IDs.
- Plan/report counts, token shortfall, per-stratum deviation, manifest, ledger,
  and audit reconcile. Missing or duplicate IDs fail before publishing tiers.
- On a GPU-capable release host, repeat the case with `nemotron[curate-gpu]` and
  `skip_similarity: false`; record verified Jaccard pairs, unverifiable pairs,
  threshold, shingling, normalization, and candidate-recall evidence. If no GPU
  is available, only that similarity arm is `BLOCKED`; the CPU identity/order
  arm remains required.

Evidence to collect: cascade config, plan, flow report, filter manifest/ledger,
audit report, decontamination report and retained corpus, subset plan/report,
tier ID sets/checksums, and GPU-arm report or named blocker.

### CUR-006 Real-Corpus Release Run

Every case above runs on packaged fixtures, which proves the machinery but
cannot show how the module behaves on real text. This case runs the release
chain once, on a real multilingual web corpus, and is the only case here that
needs the network.

Keep it small. The point is that real text exercises paths synthetic fixtures do
not — a real corpus arrives as parquet, carries boilerplate and mixed scripts,
and produces retention figures a human has to read before approving anything.

Prerequisites: network access, roughly 8 CPUs, and about 250 MB of disk
(380 MB with the optional language-ID arm). No GPU. Datasets are ungated;
`HF_TOKEN` only avoids rate limits. If the environment is offline, record this
case `BLOCKED (no network)` — every other curate case still runs.

```bash
export CURATE_REAL="$CURATE_QA/real"
mkdir -p "$CURATE_REAL"/{corpora,holdout,out}

uv run python - <<'PY'
import os, pathlib, pandas as pd
from datasets import load_dataset
root = pathlib.Path(os.environ["CURATE_REAL"]) / "corpora"
# c4 carries real web noise; wikipedia is clean prose and acts as the
# false-rejection control. Both expose hi/en/vi and neither is gated.
for repo, cfg, n, name in [("allenai/c4", "hi", 2000, "c4-hi"),
                           ("wikimedia/wikipedia", "20231101.hi", 2000, "wiki-hi")]:
    dest = root / name
    if dest.exists():
        continue
    dest.mkdir(parents=True)
    rows = []
    for i, row in enumerate(load_dataset(repo, cfg, split="train", streaming=True)):
        if i >= n:
            break
        rows.append(row)
    pd.DataFrame(rows).to_parquet(dest / "part_0.parquet", index=False)
    print(f"{name}: {len(rows)} docs")
PY
```

Verify the fetch by reading back row counts and checksums, not by exit status:
interpreter teardown after a streaming download can abort with SIGABRT once
every row has already been written.

Run the two passes described in `CUR-003` against `c4-hi`, using a fixture
language pack (`x-test-hi`) and `curate/ingest` to normalise parquet to JSONL.
For decontamination, carve the holdout out of the filtered corpus so both splits
share one id space. Then apply the same approved policy to `wiki-hi`.

Success criteria:

- `curate/ingest` mints ids for a corpus that carries none, and the filter reads
  its JSONL output. Pointing the filter at the parquet directly is refused.
- The profile skips every signal the pack does not support and says why. On a
  Devanagari pack that includes `numbers_ratio`, `punctuation` and
  `non_alpha_numeric`, which are ASCII-bound.
- The release run executes ingest, filter, audit, decontamination and subset in
  that order, with every cross-step path derived from one `output_root`.
- The removed set equals the planted holdout exactly, no removed document
  appears in any subset tier, and the ledger balances.
- Every surviving document is byte-identical to its ingested form.
- **False-rejection control.** Do NOT try to apply the approved policy to
  `wiki-hi`: an approval is bound to the corpus it was measured on, so the
  fingerprint gate refuses it at the filter and every document lands in
  `n_failed` with an empty `filtered_by_reason`. That refusal is correct
  behaviour, not a finding.

  Run `curate/profile` on `wiki-hi` separately instead, and compare the
  retention each corpus reports for the SAME signal at the SAME threshold. Clean
  encyclopaedic prose should retain at least as much as noisy web text. If a
  threshold cuts Wikipedia harder than C4, it is keying on something other than
  quality, whatever its C4 retention says.

  Observed on 2,000-document slices, for orientation only:

  | signal | threshold | c4-hi retains | wiki-hi retains |
  | --- | --- | --- | --- |
  | `latin_ratio` | `max 0.7619` | 0.9900 | 0.9990 |
  | `script_ratio` | `min 0.0` | 1.0000 | 1.0000 |
  | `boilerplate_hits` | `max 1.0` | 0.9980 | 0.9980 |
- Retention is reported per gate, and a human could defend each threshold from
  the profile alone.

Optional language-ID arm — the one failure mode fixtures cannot reproduce.
Fetch `lid.176.bin` (131,266,198 bytes) from
`https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin`, record
its checksum, then filter `c4-hi` twice: once with `language_codes: [HI]` and
once with a deliberately foreign code. Both runs must exit 0 with balanced
ledgers, and `filtered_by_reason` must name `language_code`. Retention is the
only signal that separates the correct run from the catastrophic one, which is
precisely why this is worth measuring. Skip the arm if the model is unavailable
and record it unqualified.

Reference observations from a 2,000-document C4-hi slice, for orientation only —
these move with the slice and are not assertions: ingest 2,000 → filter 1,976
(24 removed, attributed to `latin_ratio` and `boilerplate_hits`) → decontaminate
1,826 → tiers cut only from the survivors. Language-ID retention was 73.7% under
`HI` against 0.1% under a foreign code; the rejected documents had a median
Devanagari share of 0.000 against 0.888 for those kept, so the gate was removing
genuinely non-Devanagari content rather than misfiring.

Evidence to collect: dataset ids, row counts and checksums, both profile
reports, the approved policy, flow report, ledger, audit report, decontamination
report, subset report, the two retention figures, and the language-ID model
checksum when that arm runs.

## Evaluation

### EVAL-001 Discover Modes, Suites, Tasks, And Profiles

```bash
uv run nemotron steps show eval/model_eval
uv run nemotron steps run eval/model_eval --help
uv run nemo-evaluator-launcher ls tasks > "$QA_ROOT/metadata/eval-launcher-tasks.txt"
uv run nemo-evaluator-launcher ls task hellaswag --json \
  > "$QA_ROOT/metadata/hellaswag.json"

for CFG in direct base_en instruct_en mmlu_prox mmlu_prox_chat milu tiny_chat default; do
  test -f "src/nemotron/steps/eval/model_eval/config/$CFG.yaml"
done

for PROFILE in lepton_eval_direct slurm_eval_direct dgxcloud_eval_direct; do
  rg -n "^\[$PROFILE\]" "$QA_ROOT"/configs/env.*.toml
done
```

Success criteria:

- Metadata clearly separates `direct` from `launcher` mode.
- Shipped suite configs are present.
- Launcher registry output identifies task image, supported endpoint type,
  tokenizer requirements, and real task/subset defaults.
- QA identifies or generates the required backend profile before submission.
- Documentation and metadata consistently state that client-side
  `EVAL_TOKENIZER` is required for the shipped base and chat suites. Treat a
  claim that chat suites do not need it as a documentation defect.

Evidence to collect: metadata, task-registry JSON, and profile source.

### EVAL-002 Direct-Mode Internal Dry Run And Provenance

This is the evaluator's internal dry run (`dry_run=true`), not the outer CLI
compile-only flag.

```bash
export EVAL_ENDPOINT_URL="https://qa.invalid/v1/completions"
export EVAL_MODEL_HANDLE="qa-served-model"
export EVAL_TOKENIZER="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
export EVAL_RESULTS_DIR="$QA_ROOT/artifacts/eval/dry-run"

uv run nemotron steps run eval/model_eval -c direct \
  dry_run=true \
  target.api_endpoint.api_key_name=null \
  -t hellaswag

python -m json.tool "$EVAL_RESULTS_DIR/summary.dry-run.json"
python -m json.tool "$EVAL_RESULTS_DIR/run_manifest.dry-run.json"
```

Success criteria:

- No network request is sent and no harness task is executed.
- The printed command selects `hellaswag`, the endpoint type, tokenizer,
  caching, request/response logging, retries, and timeout.
- Dry-run artifacts use the `.dry-run.json` names and never overwrite a real
  `summary.json` or `run_manifest.json`.
- The manifest records the model handle, redacted endpoint, merged task params,
  harness image, pin status, Nemotron version, and Git SHA.
- No secret value appears in stdout, artifacts, or the rendered command.

Evidence to collect: stdout/stderr and both JSON files.

### EVAL-003 Output Safety, Collision, And Invalid Task Names

Prerequisites: the placeholder endpoint/model/tokenizer environment from
`EVAL-002` remains set.

```bash
export EVAL_RESULTS_DIR="$QA_ROOT/artifacts/eval/safety"
mkdir -p "$EVAL_RESULTS_DIR/hellaswag"
echo keep > "$EVAL_RESULTS_DIR/hellaswag/sentinel.txt"

# Must fail without deleting the sentinel.
! uv run nemotron steps run eval/model_eval -c direct \
    dry_run=true target.api_endpoint.api_key_name=null -t hellaswag
test -f "$EVAL_RESULTS_DIR/hellaswag/sentinel.txt"

# An overwrite dry run may plan replacement but must still preserve the file.
uv run nemotron steps run eval/model_eval -c direct \
  dry_run=true overwrite=true target.api_endpoint.api_key_name=null -t hellaswag
test -f "$EVAL_RESULTS_DIR/hellaswag/sentinel.txt"

# Path-like task names must fail before creating or deleting output.
! uv run nemotron steps run eval/model_eval -c direct \
    dry_run=true target.api_endpoint.api_key_name=null -t ../escape
```

Success criteria:

- A non-empty task directory is refused unless `overwrite=true`.
- A dry run never deletes existing results, even with overwrite requested.
- Absolute paths, separators, `..`, symlink escapes, and two tasks resolving to
  one directory are rejected during preflight.
- A live output-directory claim cannot be overridden by `overwrite=true`.

Evidence to collect: exit statuses, error text, and sentinel checksum.

### EVAL-004 Direct Base-Model Endpoint Smoke

Prerequisites: reachable completions endpoint, correct served-model name,
matching tokenizer, endpoint secret, harness image, backend profile, and durable
result mount.

```bash
: "${EVAL_ENDPOINT_URL:?Set a /v1/completions URL}"
: "${EVAL_MODEL_HANDLE:?Set the exact served model name}"
: "${EVAL_TOKENIZER:?Set the matching tokenizer path or HF id}"
: "${ENDPOINT_TOKEN:?Set the endpoint token}"
: "${EVAL_PROFILE:?Set a *_eval_direct profile}"

export EVAL_ENDPOINT_TYPE=completions
export EVAL_LIMIT_SAMPLES=5
# For remote execution, EVAL_RESULTS_ROOT must be a durable worker-visible mount.
export EVAL_RESULTS_DIR="$EVAL_RESULTS_ROOT/base-smoke-$QA_RUN_ID"
export NEMOTRON_ENV_FILE="$QA_ENV_FILE"

uv run nemotron steps run eval/model_eval \
  -c direct --batch "$EVAL_PROFILE" -t hellaswag
```

Success criteria:

- The endpoint is verified with one raw request before the benchmark; its reply
  is coherent and contains no leaked reasoning trace.
- The job exits 0 and `summary.json` reports `hellaswag: ok`.
- `run_manifest.json`, per-task `harness.log`, metrics, request/response samples,
  and cache artifacts persist on durable storage.
- The manifest records `limit_samples=5`, completions endpoint type, matching
  tokenizer, and the actual harness image used.
- No `failures.txt` remains after a clean run.

Evidence to collect: endpoint probe with content redacted as needed, job ID,
summary, manifest, metrics, and log paths.

### EVAL-005 Direct Instruct/Chat Endpoint Smoke

Prerequisites: reachable chat endpoint served with the correct reasoning parser
when applicable.

```bash
export EVAL_ENDPOINT_TYPE=chat
export EVAL_LIMIT_SAMPLES=5
export EVAL_RESULTS_DIR="$EVAL_RESULTS_ROOT/instruct-smoke-$QA_RUN_ID"

uv run nemotron steps run eval/model_eval \
  -c instruct_en --batch "$EVAL_PROFILE" -t ifeval
```

Success criteria:

- The suite selects the chat endpoint and deterministic generation defaults.
- The matching client-side tokenizer is present; the run does not fall back to
  interpreting the served-model name as an HF repository.
- Raw response inspection shows the final answer in content rather than an
  unparsed reasoning trace.
- `summary.json` reports success and artifacts meet `EVAL-004` criteria.
- HumanEval remains opt-in and is not executed by this smoke test.

Evidence to collect: redacted raw response, job ID, manifest, summary, and
metrics.

### EVAL-006 Multilingual Suite Selection

Prerequisites: base or instruct endpoint as appropriate.

```bash
export EVAL_LIMIT_SAMPLES=5

export MMLU_PROX_LANG=hi
export EVAL_RESULTS_DIR="$EVAL_RESULTS_ROOT/mmlu-prox-hi-$QA_RUN_ID"
uv run nemotron steps run eval/model_eval \
  -c mmlu_prox --batch "$EVAL_PROFILE"

export MILU_LANG=Hindi
export EVAL_RESULTS_DIR="$EVAL_RESULTS_ROOT/milu-hi-$QA_RUN_ID"
uv run nemotron steps run eval/model_eval \
  -c milu --batch "$EVAL_PROFILE"
```

For an instruct model, repeat MMLU-ProX with `-c mmlu_prox_chat` and a chat
endpoint.

Success criteria:

- MMLU-ProX manifest pins `params.task=mmlu_prox_hi`; it does not silently run
  the all-29-language family.
- Each multi-subset task has its own invocation and result directory.
- MILU uses the sovereign harness image and evaluates the configured language
  plus English control.
- `EVAL_LIMIT_SAMPLES` is recorded and described in reports as a deterministic
  subset, not a full benchmark score.

Evidence to collect: manifests, task list, image identity, summaries, and
sample counts.

### EVAL-007 Multi-Task Failure Diagnostics And Preemption Artifacts

Prerequisites: a disposable endpoint or harness configuration that can make one
task succeed and a later task fail quickly.

Run two tasks in one direct invocation. Inject a deterministic second-task
failure through a copied config, then repeat on a preemptible backend and stop
the job after the first task completes.

Success criteria:

- The run manifest exists before the first task starts.
- `summary.json` is rewritten after each task and preserves completed outcomes
  if the job is preempted during a later task.
- A failed task is `failed(<exit-code>)`; the overall process exits nonzero.
- `failures.txt` contains a bounded, task-labelled tail of each failed harness
  log, not credentials.
- A subsequent clean overwrite removes stale `failures.txt`.
- The output claim is released on normal failure or exception; a stale claim
  after hard preemption reports owner and age with actionable recovery text.

Evidence to collect: job events, partial/final summaries, failure file, logs,
claim diagnostics, and rerun result.

### EVAL-008 Launcher-Mode Compile And Dry Run

Prerequisites: Evaluator Launcher installed. A real checkpoint is not required
for the outer compile-only command.

```bash
uv run nemotron steps run eval/model_eval -c default --dry-run \
  deployment.checkpoint_path="$QA_ROOT/artifacts/fake/iter_0000001" \
  run.env.launcher_executor=slurm \
  run.env.account=qa-account \
  run.env.partition=qa-partition
```

Success criteria:

- The compiled plan retains separate outer and inner executor fields.
- `run.env.launcher_executor=slurm` does not cause a nested outer submission.
- A concrete `iter_*` checkpoint path bypasses unwanted W&B artifact discovery.
- W&B credentials are neither required nor forwarded unless W&B export is
  explicitly enabled.
- Documentation warns that launcher-managed Lepton is experimental and that
  launcher mode has no Run:ai executor.

Evidence to collect: compiled config and credential-forwarding fields.

### EVAL-009 Backend Profile Matrix

Compile every backend profile. Run `EVAL-004` live on Lepton only.

| Backend | Required profile | Required result |
| --- | --- | --- |
| Local | local harness environment | direct-mode smoke succeeds |
| Lepton | `lepton_eval_direct` | logs collected and mount persists results |
| Slurm | `slurm_eval_direct` | compile only — see below |
| DGX Cloud/Run:ai | `dgxcloud_eval_direct` | compile only — see below |

Slurm and DGX Cloud/Run:ai are **compile-only**. Their live arms are out of
scope: compile and profile review is the defined coverage for those backends.
Do not record their live arms as `BLOCKED`; they are not pending work.

Success criteria:

- The YAML-selected harness container cannot be accidentally replaced by an
  unrelated batch-profile image.
- An explicit CLI harness-image override is both used and recorded.
- Lepton endpoint tokens use `secret_vars`, not plaintext `env_vars`. Assert
  this against the FRESHLY GENERATED `env.toml`, not only against a profile
  you edited: a tester who writes `secret_vars` by hand will never observe
  that the shipped default forwards the token in plaintext.
- Lepton log collection is enabled on the job spec unless explicitly set.
- Job output is recoverable after the job reaches a terminal state, for a job
  that succeeded AND for one that failed. Enabling collection on the spec is
  necessary but not sufficient. Note that `lep job log` retrieval also depends
  on the caller's network path to the log backend: on some hosts it returns
  "Connection stopped." for terminated AND running jobs even with collection
  correctly enabled, so an empty fetch is not by itself a product defect. Verify
  the spec carries log collection, then read the durable per-task `harness.log`
  and `summary.json` from the shared output directory. Record which of the two
  paths produced the evidence.
- A digest-pinned image reports `harness.image_pinned_by_digest=true`; a tag is
  correctly reported as unpinned.

Evidence to collect: one compiled job spec per backend, at least one live remote
smoke job, logs, mounts, secret mapping with values redacted, and manifests.

### EVAL-010 Tokenizer-Extended Checkpoint Integration

Prerequisites: output of `TOK-006` or a real tokenizer-extended endpoint.

Run a five-sample benchmark twice against the same endpoint and settings: once
with the matching extended tokenizer and once only as a negative control with
the base tokenizer.

Success criteria:

- The release run uses the extended tokenizer and records its immutable path or
  revision in `run_manifest.json`.
- QA never compares or publishes the negative-control score as a valid model
  score.
- The plan/report explicitly identifies endpoint model, tokenizer, harness
  image digest, merged parameters, and code SHA as a single reproducibility
  tuple.

Evidence to collect: manifests and tokenization sanity examples.

### EVAL-011 Agent-Driven Evaluation Workflow

Give the agent this intent without supplying a command:

> Evaluate my instruct endpoint on Hindi MMLU-ProX with five samples on
> Lepton. The served model is a tokenizer-extended checkpoint. Preserve enough
> provenance to compare it with another checkpoint and do not expose my token.

Success criteria:

- The agent discovers `eval/model_eval` and chooses `mmlu_prox_chat`, not the
  completions suite.
- It requires a matching extended tokenizer and exact served-model name.
- It sets `MMLU_PROX_LANG=hi`, a fresh durable `EVAL_RESULTS_DIR`, sample limit
  5, and a direct Lepton profile.
- It uses a platform secret mapping, smokes before any full run, and explains
  that direct mode does not create or destroy the endpoint.
- It retrieves `summary.json`, `run_manifest.json`, and failures/logs and does
  not claim that a limited subset is a full benchmark.

Evidence to collect: agent transcript, generated command/config, job ID, and
artifact assessment.

## BFCL Benchmark Creation

### BFCL-001 Hotel-Domain Assisted Authoring, Publication, And Evaluation

Objective: verify that an SQA engineer can start with only a reviewed domain
brief and tool catalog, follow the published local-Python model-assisted BFCL
flow, publish a Gold benchmark, and evaluate a candidate. This is an end-to-end
acceptance test of the pipeline and its documentation. Detailed component and
negative testing remains in the BFCL automated suites.

The checked-in starting inputs are deliberately not an executable source
package or an Oracle Pack:

- `src/nemotron/steps/byob/data/sqa_hotel_booking/domain-brief.txt`
- `src/nemotron/steps/byob/data/sqa_hotel_booking/tools.json`

The domain is an English hotel-reservation assistant with ten tools covering
room search and pricing, bookings, guests, confirmation-gated mutations, an
immediate contact mutation, structured refusals, and a real timeout path. SQA
must generate, review, complete, or configure every remaining artifact. Do not
add a prebuilt backend, fixtures, probe plan, supplement, Oracle Pack, or
publication config to the input directory; doing so would invalidate the
acceptance test.

Prerequisites:

- `SETUP-001`, `SETUP-002`, and the BFCL offline test suites have passed.
- The QA owner has approved execution of the reviewed model-proposed Python
  source for this case.
- The named Lepton authoring and candidate deployments below are reachable with
  `LEPTON_API_TOKEN`. If a deployment has been replaced, update its URL and
  served model id together from `/v1/models`; do not point a new URL at the old
  model name.
- SQA can create an Ed25519 certification key and make the public key/key id
  available to the BFCL trust policy. Keep private keys and provider
  credentials outside the repository and generated configs.

Follow these operator guides in order, without implementation guidance from the
pipeline developers:

1. `docs/build-benchmarks/function-calling/how-to/start-from-domain-data.md`,
   using **Path B: Author From An Executable Source With Model Assistance** and
   beginning with the optional model-assisted source scaffold.
2. `docs/build-benchmarks/function-calling/how-to/assisted-authoring.md` for
   intake, certification, authorization, drafting, assembly, review, and
   freeze.
3. `docs/build-benchmarks/function-calling/how-to/publish-a-release.md`, using
   a real publication configuration rather than a smoke-only configuration.
4. `docs/build-benchmarks/function-calling/how-to/run-evaluation.md`, consuming
   the publication through its `run_manifest.json`.

Set up a fresh run and verify that only the two intended inputs are staged:

```bash
export BFCL_INPUT_ROOT="$(pwd)/src/nemotron/steps/byob/data/sqa_hotel_booking"
export BFCL_RUN="$QA_ROOT/artifacts/bfcl-hotel"
export BFCL_SOURCE="$BFCL_RUN/source"
export BFCL_WORKSPACE="$BFCL_RUN/workspace"
export DATA_DESIGNER_HOME="$BFCL_RUN/data-designer"

# Model-assisted source/probe/pack authoring. This model is exposed to the
# benchmark evidence and therefore must not be the evaluation candidate.
export BFCL_AUTHOR_PROVIDER="lepton_qwen"
export BFCL_AUTHOR_API_BASE="https://xtcr3jss-rkalani-qwen-3-5-122b-sdg-gcp.xenon.lepton.run/v1"
export BFCL_AUTHOR_MODEL="Qwen/Qwen3.5-122B-A10B"
export BFCL_AUTHOR_MODEL_CANONICAL="lepton:Qwen/Qwen3.5-122B-A10B@provider_managed"

# Independent BFCL evaluation candidate.
export BFCL_CANDIDATE_PROVIDER="lepton"
export BFCL_CANDIDATE_API_BASE="https://xtcr3jss-gemma-4-31b-it-vllm-ravirajj-az-32k.xenon.lepton.run/v1"
export BFCL_CANDIDATE_MODEL="google/gemma-4-31B-it"
export BFCL_CANDIDATE_CANONICAL="lepton:google/gemma-4-31B-it@provider_managed"
export BFCL_CANDIDATE_TOKENIZER="google/gemma-4-31B-it"

: "${LEPTON_API_TOKEN:?Export the Lepton endpoint token in the shell only}"

mkdir -p \
  "$BFCL_SOURCE" "$BFCL_RUN/configs" "$BFCL_RUN/keys" \
  "$DATA_DESIGNER_HOME" "$BFCL_RUN/metadata"
cp "$BFCL_INPUT_ROOT/tools.json" "$BFCL_SOURCE/tools.json"
cp "$BFCL_INPUT_ROOT/domain-brief.txt" "$BFCL_RUN/domain-brief.txt"

cat > "$DATA_DESIGNER_HOME/model_providers.yaml" <<'YAML'
providers:
  - name: lepton_qwen
    endpoint: https://xtcr3jss-rkalani-qwen-3-5-122b-sdg-gcp.xenon.lepton.run/v1
    provider_type: openai
    api_key: LEPTON_API_TOKEN
YAML

curl -fsS \
  -H "Authorization: Bearer $LEPTON_API_TOKEN" \
  "$BFCL_AUTHOR_API_BASE/models" \
  > "$BFCL_RUN/metadata/author-models.json"
curl -fsS \
  -H "Authorization: Bearer $LEPTON_API_TOKEN" \
  "$BFCL_CANDIDATE_API_BASE/models" \
  > "$BFCL_RUN/metadata/candidate-models.json"

python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["BFCL_INPUT_ROOT"])
assert sorted(path.name for path in root.iterdir()) == ["domain-brief.txt", "tools.json"]
tools = json.loads((root / "tools.json").read_text(encoding="utf-8"))
assert len(tools) == 10
assert sum(bool(tool.get("x-mutates")) for tool in tools) == 3
assert sum(bool(tool.get("x-requires-confirmation")) for tool in tools) == 2
assert {tool["function"]["name"] for tool in tools} >= {
    "create_booking", "cancel_booking", "rebuild_availability_index"
}

run = Path(os.environ["BFCL_RUN"])
author_ids = {
    item["id"] for item in json.loads(
        (run / "metadata/author-models.json").read_text(encoding="utf-8")
    )["data"]
}
candidate_ids = {
    item["id"] for item in json.loads(
        (run / "metadata/candidate-models.json").read_text(encoding="utf-8")
    )["data"]
}
assert os.environ["BFCL_AUTHOR_MODEL"] in author_ids, author_ids
assert os.environ["BFCL_CANDIDATE_MODEL"] in candidate_ids, candidate_ids
assert os.environ["BFCL_AUTHOR_MODEL"] != os.environ["BFCL_CANDIDATE_MODEL"]
PY
```

The listed routes and served ids come from the current Lepton-backed QA
deployment inventory. The `/models` checks are mandatory because deployments
are mutable. A `401` or `403` means the token is missing or not authorized; a
missing expected id means the route and model name no longer form a valid pair.
The provider file contains only the credential variable name.

First discover the step and scaffold the conventional source package. Use seed
`7`, temperature `0`, and the explicit reviewed provider-managed identity so
the authoring route is recorded without pretending its weights are pinned:

```bash
uv run nemotron steps show byob/bfcl

uv run python -m nemotron.steps.byob.scripts.scaffold_source_package \
  --tools "$BFCL_SOURCE/tools.json" \
  --output "$BFCL_SOURCE" \
  --dependency-lock \
  --draft-with-model \
  --domain-brief "$BFCL_RUN/domain-brief.txt" \
  --model-alias author \
  --model-provider "$BFCL_AUTHOR_PROVIDER" \
  --model "$BFCL_AUTHOR_MODEL" \
  --model-canonical-id "$BFCL_AUTHOR_MODEL_CANONICAL" \
  --seed 7 \
  --temperature 0 \
  --request-timeout 600
```

The generated `backend.py` and `fixtures.json` are proposals, not certified
domain truth. SQA must review and finish availability, occupancy, pricing, date,
confirmation, mutation, cancellation, contact-validation, structured-error,
reset/isolation, and blocking timeout behavior. Replace placeholder fixtures
with deterministic records. Do not weaken `tools.json` to make a generated
backend pass. Then run:

```bash
uv run python -m nemotron.steps.byob.scripts.check_source_package \
  --source "$BFCL_SOURCE"

! rg -n 'BFCL-TODO|NotImplementedError' \
  "$BFCL_SOURCE/backend.py" "$BFCL_SOURCE/fixtures.json"
```

Extract the reviewed error vocabulary, draft a probe plan, correct any
ungrounded model literals to fixture-backed values, and check A2 readiness:

```bash
uv run python -m nemotron.steps.byob.scripts.extract_error_vocabulary \
  --source "$BFCL_SOURCE" \
  --output "$BFCL_RUN/configs/error-vocabulary.json"

uv run python -m nemotron.steps.byob.scripts.draft_probe_plan \
  --source "$BFCL_SOURCE" \
  --domain-brief "$BFCL_RUN/domain-brief.txt" \
  --output "$BFCL_RUN/configs/probe-plan.json" \
  --clock 2026-06-01T09:00:00+00:00 \
  --error-vocabulary "$BFCL_RUN/configs/error-vocabulary.json" \
  --model-alias author \
  --model-provider "$BFCL_AUTHOR_PROVIDER" \
  --model "$BFCL_AUTHOR_MODEL" \
  --model-canonical-id "$BFCL_AUTHOR_MODEL_CANONICAL" \
  --seed 7 \
  --temperature 0 \
  --request-timeout 600

uv run python -m nemotron.steps.byob.scripts.check_probe_plan \
  --source "$BFCL_SOURCE" \
  --probe-plan "$BFCL_RUN/configs/probe-plan.json"
```

In particular, a date such as `2030-08-10` or a reason such as
`change_of_plans` is not trusted merely because a model emitted it. Unless its
parameter schema pins that literal, bind it to reviewed fixture data. The
reviewed plan must exercise successful calls for all ten tools, structured
errors, all three mutations, both confirmation gates, reset/isolation, and the
full availability-index timeout. `attainable_tier: A2` is only a static
readiness result; intake must execute the probes to earn A2.

Continue through the guides with the following stage sequence. Copy all example
policies and configs to `$BFCL_RUN/configs` before editing them. Human-owned
artifacts are `backend.py`, `fixtures.json`, `probe-plan.json`,
`reviewed-supplement.yaml`, policy/config choices, and approvals. Generated
evidence, drafts, packs, reports, Parquet files, and manifests must be corrected
upstream and regenerated rather than edited in place.

```text
bfcl_author --ci author
  -> execute intake probes and earn certification tier A2
bfcl_author --ci apply-policy
  -> bind reviewed exposure and clean-evidence policy
bfcl_author --ci draft
  -> produce bounded coverage, assertion, template, and validation proposals
bfcl_author --ci assemble
  -> combine reviewed proposals with reviewed-supplement.yaml
nemotron steps run byob/bfcl ... stage=prepare family=bfcl
  -> independently validate the candidate at Gold
bfcl_author --ci review
  -> bind the exact source, certification, draft, candidate, and Gold report
bfcl_author --ci release
  -> record approval, freeze exact bytes, revalidate, and publish
nemotron steps run byob/bfcl -c "$BFCL_RUN/configs/eval.cli.yaml"
  -> preflight and then evaluate from the published run_manifest.json
```

Use the exact command lines and required arguments from the four guides, and
capture each invocation in the evidence bundle. The case fails as a
documentation defect if an SQA engineer cannot determine a required artifact,
argument, review decision, or recovery action from those guides.

Resolve the evaluation config with this independent Lepton candidate. Its
endpoint does not publish an immutable weights digest or revision, so schema
1.2 must record the identity honestly as provider-managed and
`publication.requested` must remain `false`. The evaluation still produces the
required SQA artifacts; do not invent a digest from the model name. If the
deployment owner supplies a verified immutable revision or weights digest,
replace the two nulls and only then request publication of the score.
Do not enable Gemma as a profiling, paraphrasing, judging, or translation role
while building this BFCL publication; that would expose the benchmark to its
evaluation candidate and the contamination gate must refuse the run.

```yaml
schema_version: "1.2"
config_status: resolved

candidates:
  - alias: gemma_4_31b_it
    model: google/gemma-4-31B-it
    provider: lepton
    provider_api_version: v1
    api:
      base_url: https://xtcr3jss-gemma-4-31b-it-vllm-ravirajj-az-32k.xenon.lepton.run/v1
      api_key_env: LEPTON_API_TOKEN
    model_identity:
      source: lepton
      model: google/gemma-4-31B-it
      revision: null
      weights_digest: null
    inference:
      temperature: 0.0
      top_p: 1.0
      max_tokens: 8192
      seed: 42
      tool_choice: auto
      provider_extensions: {}

publication:
  requested: false
  require_same_task_ids: true
```

The matching tokenizer/checkpoint reference is `google/gemma-4-31B-it`. BFCL
evaluation sends chat tool
schemas directly and does not need a client-side tokenizer, but record this
reference beside the endpoint identity so the served deployment can be audited.

Success criteria:

- The step is discoverable as `byob/bfcl`, and SQA completes the flow from only
  the two checked-in inputs without pipeline-developer implementation guidance.
- SQA reviews and completes the executable source, deterministic fixtures, A2
  probe plan, semantic supplement, policy, and publication/evaluation configs;
  all model-generated source markers are removed through implementation rather
  than schema weakening.
- Intake executes the probes and records certification tier `A2`, including
  reset/isolation, mutation, confirmation, timeout cleanup, structured error,
  and success evidence.
- The assembled candidate Oracle Pack passes fresh independent validation with
  `tier: gold` and `gold_eligible: true`.
- Release freezes the reviewed pack, reruns Gold validation, and produces one
  immutable publication directory containing `benchmark.parquet`,
  `benchmark_raw.parquet`, and `run_manifest.json`.
- The published `run_manifest.json` records `gold_eligible: true`; an output
  marked `smoke_no_publication` does not satisfy this case.
- Evaluation reads the publication through `run_manifest.json` and completes
  with `eval_report.json`, `eval_task_results.parquet`, `eval_manifest.json`,
  and per-task results, without a fatal setup or infrastructure failure.
- Credentials and private keys do not appear in source, configs, commands,
  logs, manifests, reports, or Git status.
- Every instruction that cannot be completed from the linked guides is filed
  as a documentation defect with the failing step and missing information.

Benchmark domain quality is not an acceptance criterion. The intent is to
verify that the documented LLM-assisted BFCL pipeline can be operated from the
two initial domain inputs through publication and evaluation.

Evidence to collect: repository/base/head SHAs; input checksums; authoring and
candidate model identities; redacted commands/configs/logs; reviewed source,
fixtures, probe plan, supplement, and policy digests; A2 certification and
intake evidence; draft and assembly provenance; candidate and frozen-pack Gold
reports; review/freeze/release records; publication Parquet and manifest
checksums; evaluation report, manifest, per-task results, and caches; and any
documentation defect IDs.

## Persona MCQ SDG For SFT

### PMC-001 Discover Step, Plugin, And Dependencies

```bash
uv run nemotron steps list --category sdg
uv run nemotron steps show sdg/persona_mcq
uv run python - <<'PY'
from importlib.metadata import entry_points

matches = [ep for ep in entry_points(group="data_designer.plugins") if ep.name == "persona-mcq"]
assert len(matches) == 1, matches
print(matches[0].load())
PY
```

Success criteria:

- The step is discoverable as `sdg/persona_mcq`; stale `sdg/qasynth` naming is
  absent.
- The plugin entry point loads exactly once.
- The root `data-sdg` extra provides Data Designer, sentence-transformers, and
  Torch; there is no redundant Persona-only extra.
- `tiny` and `default` configs validate without resolving model credentials.

Evidence to collect: catalog, metadata, entry-point output, and versions.

### PMC-002 Persona Asset Staging And Secret Safety

Prerequisites: NGC CLI on `PATH`; valid `NGC_API_KEY` for the uncached arm;
`QWEN_API_BASE`, `OSS_API_BASE`, and `GEMMA_API_BASE` exported even for a
`personas`-only run. Config interpolation resolves every model endpoint before
any stage executes, so without them the case dies on a missing-endpoint
`InterpolationResolutionError` and never reaches the NGC check it is testing.
Budget time for the download: the two locales are ~4 GB and take ~15 minutes.

Use a fresh managed-assets directory. First run without a key and confirm a
clear failure. Then export the key and stage the tiny config locales. Finally
unset the key and repeat from cache.

```bash
export DATA_DESIGNER_MANAGED_ASSETS_PATH="$QA_ROOT/artifacts/persona-assets"
unset NGC_API_KEY

! uv run nemotron steps run sdg/persona_mcq -c tiny \
    pipeline.experiment_name=missing-key \
    'pipeline.stages=[personas]'

export NGC_API_KEY='<set-in-shell-only>'
uv run nemotron steps run sdg/persona_mcq -c tiny \
  pipeline.experiment_name=asset-stage \
  'pipeline.stages=[personas]'

unset NGC_API_KEY
uv run nemotron steps run sdg/persona_mcq -c tiny \
  pipeline.experiment_name=asset-stage \
  'pipeline.stages=[personas]'
```

Success criteria:

- Missing assets without a key fail with an explicit `NGC_API_KEY` recovery.
- Only unique configured locales (`en_IN`, `hi_Deva_IN`) download.
- Cached assets allow the repeat without a key.
- No NGC config containing the key is written under the experiment or managed
  assets; the key is absent from run config, summary, logs, and Git status.
- Malayalam correctly uses `en_IN` personas while generating Malayalam text.

Evidence to collect: redacted logs, asset file list/checksums, `run.json`, and
secret scan.

### PMC-003 Tiny End-To-End Generation

Prerequisites: all three model endpoints, `NVIDIA_API_KEY`, cached personas, and
HF access for multilingual E5 when needed.

```bash
: "${QWEN_API_BASE:?Set Qwen endpoint}"
: "${OSS_API_BASE:?Set OSS endpoint}"
: "${GEMMA_API_BASE:?Set Gemma endpoint}"
: "${NVIDIA_API_KEY:?Set model endpoint token}"

export NEMOTRON_RUN_DIR="$QA_ROOT/artifacts/persona-mcq"
uv run nemotron steps run sdg/persona_mcq -c tiny \
  pipeline.experiment_name=e2e-tiny
```

Success criteria:

- Stages run in order: personas, questions, lexical dedup, semantic dedup,
  answer seed, answers, build SFT, sample.
- English, Hindi, and Malayalam questions each contain exactly four distinct
  options and satisfy configured script-fraction gates.
- `summary.json` reports input/output yield and rejection reasons for every
  stage.
- Each answer-model/language path separates append-safe successes from
  retryable failures.
- Unanimous teacher agreement is enforced and the exported response teacher
  voted for the selected answer.
- Final records use `{messages, metadata}` and the localized final answer label.
  Reasoning-on rows have English `assistant.reasoning_content`.
- `run.json` records redacted configuration and dependency/repository versions.

Evidence to collect: run/summary JSON, stage paths and counts, representative
accepted/rejected rows per language, and redacted logs.

### PMC-004 Deduplication, Voting, And Language Gates

Run the focused offline test suite and inspect a tiny live sample.

Success criteria:

- Lexical exact/near duplicates and wrong-language questions are rejected with
  distinct counters.
- Semantic dedup is deterministic for a fixed seed and does not chain through
  an already dropped item.
- Answer choice shuffling and stable query IDs reproduce for the same seed.
- `unanimous` requires every parsed teacher answer to match.
- `majority` requires strictly more than half of all configured teachers; a tie
  or mere plurality is rejected.
- Target-language reasoning is rejected when English reasoning is configured.
- Hindi/Malayalam Latin gloss removal does not remove the target-script answer.

Evidence to collect: test output, relevant summary counters, and sampled rows.

### PMC-005 Resume, Stage Selection, And Config Identity

Prerequisites: completed `PMC-003` experiment.

```bash
uv run nemotron steps run sdg/persona_mcq -c tiny \
  pipeline.experiment_name=e2e-tiny \
  'pipeline.stages=[answers,build_sft,sample]'

# A material config change under the same identity must fail.
! uv run nemotron steps run sdg/persona_mcq -c tiny \
    pipeline.experiment_name=e2e-tiny \
    question_generation.num_records=9 \
    'pipeline.stages=[questions]'
```

Success criteria:

- Stage-list CLI syntax normalizes to a list and stage order remains canonical.
- Stage selection itself does not change experiment identity.
- Resume preserves prior stage summaries and reuses successes while retrying
  failure records.
- A material config mismatch is rejected with a new-name/overwrite recovery.
- `pipeline.overwrite=true`, when tested in a disposable copy, removes stale
  artifacts and resets summaries rather than merging incompatible runs.

Evidence to collect: before/after checksums, summaries, retry counts, and error
messages.

### PMC-006 Config-Driven Language And Geography Extension

Run two small variants from copied configs:

- override English locale to `en_US` and confirm prompt geography follows the
  sampled US persona rather than hard-coded India;
- add a Tamil language block with Tamil display name, answer label, script
  pattern, thresholds, and an available persona locale.

Success criteria:

- Geography is derived from persona metadata.
- Language behavior is driven entirely by the config block; no code dispatch
  on `english`, `hindi`, or `malayalam` is required.
- Target-language question/options/final answer and independently configured
  English reasoning pass their respective script gates.
- Unsupported locale or malformed regex/threshold fails with actionable text.

Evidence to collect: copied config, redacted prompts, sampled output, and
validation errors.

### PMC-007 Sampling And Reasoning-On/Off Contract

Use a controlled offline fixture with at least 10 accepted rows per language,
or run a bounded live configuration large enough to produce that intersection.
Set `sampling.per_language=10` and `reasoning_off_fraction=0.10`.

Success criteria:

- Exactly one row per language is reasoning-off and nine are reasoning-on for
  every teacher/view.
- Reasoning-off records omit `reasoning_content`; they do not set it to empty or
  leak it into `content`.
- All-language and `english_<target>` views are aligned and deterministic for
  the same seed.
- The summary explicitly records selected and reasoning-off counts by language.

Evidence to collect: row counts, schema validation, hashes, and sample summary.

### PMC-008 Downstream SFT Handoff

Prerequisites: completed Persona run with at least one emitted teacher view.

```bash
export PMC_VIEW="$NEMOTRON_RUN_DIR/persona_mcq/e2e-tiny/training/gemma/english_malayalam"

python -m json.tool "$PMC_VIEW/blend.json"
uv run nemotron steps run sft/automodel -c tiny --dry-run \
  dataset.path_or_dataset_id="$PMC_VIEW/train.jsonl"

uv run nemotron steps run data_prep/sft_packing -c tiny --dry-run \
  blend_path="$PMC_VIEW/blend.json"
```

Success criteria:

- Stable `training/<teacher>/train.jsonl` and `blend.json` paths exist.
- Each `english_<target>` view has equal English and target-language counts.
- Blend paths are absolute/resolvable and point to the emitted JSONL.
- AutoModel SFT accepts `train.jsonl`; SFT packing accepts the blend manifest.
- QA does not confuse Persona MCQ training data with `byob/mcq` held-out eval
  data.

Evidence to collect: manifests, counts, compiled downstream configs, and
artifact lineage.

### PMC-009 Remote Execution Profiles

Compile all shipped Persona profiles and run the tiny profile on at least one
remote backend:

| Backend | Production profile | Tiny profile |
| --- | --- | --- |
| Lepton | `lepton_sdg_persona_mcq` | `lepton_sdg_persona_mcq_tiny` |
| Slurm | `slurm_sdg_persona_mcq` | `slurm_sdg_persona_mcq_tiny` |
| DGX Cloud/Run:ai | `dgxcloud_sdg_persona_mcq` | `dgxcloud_sdg_persona_mcq_tiny` |

Success criteria:

- Production profiles request one GPU; tiny profiles remain CPU-portable.
- Required Data Designer, sentence-transformers, and compatible Torch packages
  install in the worker.
- Lepton startup installs the NGC CLI; other images already provide it or fail
  preflight clearly.
- `NGC_API_KEY`, model endpoint variables, `NVIDIA_API_KEY`, optional
  `HF_TOKEN`, and `NEMOTRON_RUN_DIR` reach the worker without secret values
  being serialized into checked-in config or run metadata.
- Detached output persists and can resume after the worker exits.

Evidence to collect: compiled job specs, one live job ID, dependency log,
mount path, artifacts, and redacted environment evidence.

### PMC-010 Agent-Driven Persona MCQ Workflow

Give the agent this intent:

> Create a tiny English/Hindi/Malayalam persona-grounded MCQ SFT dataset on
> Lepton, keep English reasoning, require unanimous agreement from the three
> teachers, and prepare the Gemma English/Malayalam view for SFT packing. Resume
> safely if the job is interrupted.

Success criteria:

- The agent chooses `sdg/persona_mcq -c tiny`, not `byob/mcq`.
- It asks for or verifies persona/model prerequisites without embedding secrets
  in configs.
- It uses a stable experiment name for resume, does not use overwrite casually,
  and selects the remote tiny profile.
- It validates agreement, language/reasoning scripts, run/summary metadata,
  emitted training JSONL, and blend manifest before proposing training.

Evidence to collect: transcript, generated commands/config, job ID, and
artifact review.

## Long-Context Chat SDG

### LCSDG-001 Validate Standalone Installation And CLI

Prerequisites: `SETUP-003`.

```bash
cd "$LC_ROOT"
uv run python pipeline.py --help
uv run python evaluate.py --help
uv run python - <<'PY'
from importlib.metadata import entry_points

matches = [ep for ep in entry_points(group="data_designer.plugins") if ep.name == "long-context-chat-sdg"]
assert len(matches) == 1, matches
print(matches[0].load())
PY
cd -
```

Success criteria:

- Pipeline stages are exactly `query_gen`, `query_prep`, `generate`, and `all`.
- Resume modes and explicit `--simulate-retrieval` are documented.
- Evaluate CLI exposes objective-only, judge, quality/overlap thresholds,
  reasoning stripping, limit, workers, and resume.
- The plugin loads in the isolated environment.

Evidence to collect: help and entry-point output.

### LCSDG-002 Prepare Corpus And Query-Generation Dry Run

Create at least 40 small chunks across multiple source documents:

```bash
export LC_QA="$QA_ROOT/artifacts/long-context"
mkdir -p "$LC_QA/data" "$LC_QA/configs"

uv run python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["LC_QA"])
with (root / "data/chunks.jsonl").open("w", encoding="utf-8") as f:
    for i in range(40):
        f.write(json.dumps({
            "text": f"Document {i // 4} passage {i}: a distinct factual statement for retrieval testing.",
            "chunk_id": f"chunk-{i:03d}",
            "source_id": f"doc-{i // 4:02d}",
        }) + "\n")
PY

cp "$LC_ROOT/config/pipeline.yaml" "$LC_QA/configs/pipeline.yaml"
```

Patch only the QA copy to set a fresh `exp_root`, the chunks path,
`query_gen.n_queries=8`, and `query_gen.embedding.backend=minilm`. Then run:

```bash
cd "$LC_ROOT"
uv run python pipeline.py \
  --config "$LC_QA/configs/pipeline.yaml" \
  --stage query_gen --limit 8 --dry-run
cd -
```

Success criteria:

- Input parsing preserves stable chunk/document IDs and skips malformed/empty
  rows with diagnostics.
- Dry run performs bounded sampling/embedding/clustering diagnostics without an
  LLM call or writing `queries.jsonl`.
- Sampling and sizing are deterministic for the configured seed.
- The clean synced environment imports NumPy/SciPy/scikit-learn without ABI or
  missing-module errors.

Evidence to collect: input checksum, copied config, diagnostics, output tree,
and network-call evidence.

Repeat the bounded query-generation arm against a QA LanceDB table containing
the same chunk IDs/text and precomputed vectors.

Additional success criteria for LanceDB:

- The configured URI/table opens without using the JSONL embed path.
- Stored vectors are used rather than recomputed.
- Field mapping, bounded sampling, topic grouping, and deterministic query-unit
  sizing match the JSONL contract.

### LCSDG-003 Query Preparation From Bring-Your-Own Queries

Write a small `queries.jsonl` containing exact duplicates, punctuation/case
duplicates, near duplicates, and distinct multi-hop queries. Disable personas
or point `persona.local_path` in the copied config to a QA fixture.

```bash
cd "$LC_ROOT"
uv run python pipeline.py \
  --config "$LC_QA/configs/pipeline.yaml" \
  --stage query_prep --limit 8
cd -
```

Success criteria:

- Exact normalized and embedding-near duplicates collapse at the configured
  threshold.
- KMeans, HDBSCAN, and agglomerative choices either execute when dependencies
  are installed or fail with an explicit missing optional dependency.
- Seeds are sampled across clusters, deterministic for a fixed seed, and retain
  query type, tools, cluster ID, and persona provenance when enabled.
- `output/seeds.jsonl` contains no credentials.

Evidence to collect: input/output rows, cluster distribution, dedup counts, and
config.

### LCSDG-004 Retrieval Mode Is Explicit And Fail-Closed

```bash
unset RETRIEVAL_ENDPOINT
cd "$LC_ROOT"

! uv run python pipeline.py \
    --config "$LC_QA/configs/pipeline.yaml" \
    --stage generate --limit 1

uv run python pipeline.py \
  --config "$LC_QA/configs/pipeline.yaml" \
  --stage generate --limit 1 --simulate-retrieval
cd -
```

Success criteria:

- HTTP mode without an endpoint fails before model generation.
- The pipeline never silently falls back from HTTP to simulation.
- Simulation occurs only with the explicit flag/config and every raw/exported
  row carries `retrieval_mode=simulated`.
- HTTP-backed rows carry `retrieval_mode=http`.

Evidence to collect: failure text, model call counts, and provenance fields.

### LCSDG-005 HTTP Retrieval Adapter Contract

Prerequisites: a QA HTTP retriever over the sample corpus.

**Conditional case.** This case is gated on retriever availability. Where no
HTTP retriever is provisioned, record it as `BLOCKED (retriever unavailable)`
naming the retriever owner, and proceed. It does not block signoff of the
remainder of the long-context group.

Run five trajectories while testing both the default schema and one remapped
schema using a copied config.

Success criteria:

- Request query/count fields, static body, result path, IDs, text, score, and
  document IDs follow `retrieval.field_map`.
- The client requests `top_k * oversample_factor`, deterministically samples
  back to `top_k`, and does not deduplicate valid results across successive
  calls.
- Missing result IDs receive stable content-hash IDs.
- Retry count and timeout are bounded; non-retryable schema errors surface.
- Retrieval logs preserve evidence IDs/text needed for citation checks but no
  authorization secret.

Evidence to collect: redacted server request/response log and raw trajectories.

### LCSDG-006 Live Five-Row Generation Smoke

Prerequisites: assistant and user-simulator endpoints, API keys if required,
prepared seeds, and a real retriever for the release-signoff arm.

**Split this case by retrieval mode.** The simulated arm needs only the two
model endpoints and must always be run. The HTTP arm is conditional on
retriever availability; without a retriever record it as
`BLOCKED (retriever unavailable)` and report the simulated arm on its own,
clearly labelled `retrieval_mode=simulated`.

```bash
: "${ASSISTANT_ENDPOINT:?Set assistant /v1 endpoint}"
: "${USER_MODEL_ENDPOINT:?Set user simulator /v1 endpoint}"
: "${RETRIEVAL_ENDPOINT:?Set retriever endpoint}"

cd "$LC_ROOT"
uv run python pipeline.py \
  --config "$LC_QA/configs/pipeline.yaml" \
  --stage generate --limit 5 --resume if_possible
cd -
```

Success criteria:

- Five rows are attempted and checkpoints support resume.
- Each valid trajectory has the configured turn range, bounded steps/tool calls,
  tool schema, retrieval log, persona, compression metadata, and scratchpad
  behavior.
- Search tool responses contain `{"results": [...]}` and cited chunk IDs exist
  in retrieved evidence.
- Context compaction retains recent raw tool results and reference-compacts
  older context without fabricating evidence IDs.
- `raw.jsonl` clearly distinguishes HTTP retrieval.

Evidence to collect: endpoint/model identities, raw rows, artifact checkpoints,
call/error counts, and duration.

### LCSDG-007 Objective Evaluation And SFT Export

```bash
cd "$LC_ROOT"
uv run python evaluate.py \
  --config "$LC_QA/configs/pipeline.yaml" \
  --limit 5
cd -
```

Success criteria:

- Tool-name/argument schema, required arguments, retrieval presence, final
  answer, answer citations, and reasoning citations are checked.
- Fabricated citation IDs fail the relevant integrity gate.
- `sft.jsonl` includes only kept rows and preserves messages, tools,
  `retrieval_mode`, and generation metadata.
- Assistant `reasoning_content` is retained by default.
- `summary.json` contains total/kept/keep rate, objective/citation failures,
  hops, context-length distribution, overlap, and retrieval-mode counts.

Evidence to collect: raw/SFT counts, summary, rejected examples, and schema
validation.

### LCSDG-008 Judge, Re-Thresholding, And Reasoning Policy

```bash
cd "$LC_ROOT"
uv run python evaluate.py \
  --config "$LC_QA/configs/pipeline.yaml" \
  --judge --limit 5 --min-quality 3 --min-overlap 0.02 \
  --resume if_possible

uv run python evaluate.py \
  --config "$LC_QA/configs/pipeline.yaml" \
  --judge --limit 5 --min-quality 4 --min-overlap 0.03 \
  --resume if_possible --strip-reasoning
cd -
```

Success criteria:

- `--judge` fails rather than silently selecting a hosted default when no judge
  model/endpoint is configured.
- The judge applies a binary train-harmful-defect gate and 1-5 quality score.
- Cached judge outputs are reused when only thresholds change.
- Whenever `summary.json` reports `judged: true`, `mean_quality` is non-null
  and `quality_distribution` is non-empty. A resume that reuses the GENERATE
  stage artifacts can skip the judge entirely and still report `judged: true`
  with `kept: 0` and exit 0 — which reads like a quality result and is not.
  Cross-check every judged run against the same run with `--resume never`.
- `--strip-reasoning` removes every assistant `reasoning_content`; without it,
  reasoning citations remain gated.
- Reports do not treat the character-ngram overlap proxy as factuality proof.

Evidence to collect: judge config, call counts before/after threshold change,
both summaries, and reasoning-field scan.

### LCSDG-009 Resume And Existing Experiment Behavior

Interrupt a five-row generation after at least one row-group checkpoint and
rerun with each resume mode against a disposable experiment.

Success criteria:

- `if_possible` resumes from the last valid checkpoint.
- `always` fails clearly if no compatible checkpoint exists.
- `never` starts a new generation rather than claiming to resume.
- Reusing an existing experiment prints the documented overwrite warning.
- QA uses a new `exp_name` for changed configuration and does not mistake
  overwrite behavior for config-identity protection.

Evidence to collect: checkpoint tree, before/after row IDs/hashes, call counts,
warning text, and final rows.

### LCSDG-010 Agent-Driven Long-Context Workflow

Give the agent this intent:

> From my chunked corpus, generate five multi-turn retrieval-grounded research
> chats for SFT. Use my real HTTP retriever, retain reasoning, resume safely,
> run objective and LLM quality gates, and show me citation-integrity and keep
> rate before I train.

Success criteria:

- The agent enters the standalone example and uses its isolated environment; it
  does not invent a `nemotron steps run` ID for this example.
- It requires the external HTTP retriever and does not use simulated retrieval
  unless explicitly requested.
- It stages query generation/preparation before generation, uses `--limit 5`,
  and chooses a fresh experiment name.
- It runs objective plus judge evaluation, validates evidence IDs and provenance,
  and reports limitations of keep rate/overlap as quality evidence.

Evidence to collect: transcript, copied config, commands, and artifact review.

## Super3 Long-Context SFT References

### LCSFT-001 Static Invariants For 128K And 256K Profiles

```bash
uv run python - <<'PY'
from pathlib import Path
import yaml

root = Path("src/nemotron/steps/sft/megatron_bridge/config")
expected = {
    "super3_128k.yaml": dict(seq=131072, nodes=8, ranks=64, tp=8, pp=1, cp=8, ep=64),
    "super3_256k.yaml": dict(seq=262144, nodes=16, ranks=128, tp=8, pp=2, cp=8, ep=8),
}
for name, want in expected.items():
    cfg = yaml.safe_load((root / name).read_text())
    model = cfg["model"]
    seqs = {
        cfg["recipe"].get("seq_length"),
        cfg["dataset"]["seq_length"],
        cfg["dataset"]["packed_sequence_specs"]["packed_sequence_size"],
        model["seq_length"],
    }
    assert seqs == {want["seq"]}, (name, seqs)
    assert cfg["dataset"]["packed_sequence_specs"]["pad_seq_to_mult"] == 128
    assert cfg["run"]["env"]["nodes"] == want["nodes"]
    assert cfg["run"]["env"]["nodes"] * cfg["run"]["env"]["gpus_per_node"] == want["ranks"]
    assert (model["tensor_model_parallel_size"], model["pipeline_model_parallel_size"],
            model["context_parallel_size"], model["expert_model_parallel_size"]) == (
                want["tp"], want["pp"], want["cp"], want["ep"])
    assert cfg["recipe"]["packed_sequence"] is True
    assert cfg["train"]["micro_batch_size"] == cfg["train"]["global_batch_size"] == 1
    assert model["mtp_num_layers"] == 0
    assert cfg["ddp"]["grad_reduce_in_fp32"] is False
    print(name, "ok")
PY
```

Success criteria: all asserted resource, recipe/dataset/model sequence, packing,
MTP, DDP, and batch invariants pass. A missing `recipe.seq_length` fails rather
than silently accepting the recipe factory's shorter default.

Evidence to collect: script output and config checksums.

### LCSFT-002 Generic-Step Dry-Run Guard

```bash
export SFT_PACKED_128K_DIR="$QA_ROOT/artifacts/fake-packed-128k"
export SFT_PACKED_256K_DIR="$QA_ROOT/artifacts/fake-packed-256k"
export SFT_OUTPUT_DIR="$QA_ROOT/artifacts/sft-output"
mkdir -p "$SFT_PACKED_128K_DIR"/splits/{train,valid}
mkdir -p "$SFT_PACKED_256K_DIR"/splits/{train,valid}

uv run nemotron steps run sft/megatron_bridge -c super3_128k --dry-run
uv run nemotron steps run sft/megatron_bridge -c super3_256k --dry-run
```

Success criteria:

- Both configs are discoverable and compile for planning.
- Resolved plans retain 64-rank and 128-rank topology respectively.
- User-facing metadata and README explicitly say the stock runner does not
  forward required context-parallel packing fields or preserve the required DDP
  precision choice.
- Neither command submits a job. Any agent or CLI path that presents these as
  safe to launch through the stock runner fails this case.

Evidence to collect: compiled configs and warnings.

### LCSFT-003 Packed-Data Contract Review

Prerequisites: candidate externally packed shards, if available.

Success criteria:

- Train and validation splits exist on shared storage and are non-empty.
- Packing used the exact tokenizer intended for training.
- Stored sequence size equals 131,072 or 262,144 as selected.
- Every stored subsequence satisfies
  `(subsequence_length - 1) % 128 == 0`.
- Loss masks, boundaries, and tool-call/chat templates are inspected on sampled
  records before any future compatible runner is allowed to launch.
- Lineage records source SFT JSONL, tokenizer revision, packer implementation,
  config, seed, and shard checksums.

**Conditional scope.** This case applies only when externally packed 128K/256K
shards exist to review. Confirm applicability with the QA owner before signoff
rather than assuming it either way.

If aligned packed data is unavailable, mark this case `BLOCKED`; do not weaken
the alignment criteria or use the stock packer without proving equivalence.

## Tokenizer Extension

### Tokenizer Group Standing Risk

Three of the four tokenizer steps reach a `device_map="auto"` model load, and
the corpus loader resolves its text field through a fallback chain. The group's
characteristic failure is therefore **a run that succeeds and is wrong**, not a
run that crashes. For every case in this group:

- treat exit 0 as necessary but never sufficient;
- assert on `summary.json` counters, not on log lines;
- confirm the output actually differs when an input that should change it
  changes, using a positive control rather than only a negative test;
- record the host type (CPU vs GPU), because several code paths are only taken
  when CUDA is visible.

### Tokenizer Test Data Setup

Use a small local corpus for plumbing and a separate local validation corpus.
The release-scale arm must later repeat on the intended target corpus/model.

```bash
export TOK_ROOT="$QA_ROOT/artifacts/tokenizer-extension"
mkdir -p "$TOK_ROOT"/{data,tokenizers,checkpoints,eval}

cat > "$TOK_ROOT/data/train_hi.jsonl" <<'EOF'
{"text":"भारत एक बहुभाषी देश है और यह वाक्य टोकनाइज़र प्रशिक्षण के लिए है।"}
{"text":"कृत्रिम बुद्धिमत्ता के लिए स्वच्छ और विविध पाठ उपयोगी होता है।"}
{"text":"नदी, पर्वत, विज्ञान, इतिहास और प्रौद्योगिकी अलग विषय हैं।"}
{"text":"लंबे शब्दों और सामान्य वाक्यांशों को कई बार दोहराया गया है।"}
EOF

cat > "$TOK_ROOT/data/eval_hi.jsonl" <<'EOF'
{"text":"यह स्वतंत्र मूल्यांकन पाठ है जो प्रशिक्षण पंक्तियों से अलग है।"}
{"text":"बेहतर टोकन विभाजन कम फर्टिलिटी दे सकता है।"}
EOF

cat > "$TOK_ROOT/data/eval_en.jsonl" <<'EOF'
{"text":"This independent English set checks base-language regression."}
{"text":"Every model must score the same bytes for a BPB comparison."}
EOF
```

### TOK-001 Discover Workflow, Profiles, And Languages

```bash
uv run nemotron steps list --category tokenizer_extension
for STEP in extend init_embeddings evaluate eval_init; do
  uv run nemotron steps show "tokenizer_extension/$STEP"
done

uv run python - <<'PY'
from nemotron.steps.tokenizer_extension.languages import LANGUAGES
print(sorted(LANGUAGES))
PY
```

Success criteria:

- All four steps are catalogued with the expected consumes/produces chain.
- Registered languages include Hindi, Marathi, Nepali, Sanskrit, Bengali,
  Punjabi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam, Urdu, and
  Vietnamese.
- `language` resolves normalizer, script, auxiliary encoder, and fastText code.
- Documentation and metadata agree that the default init needs one GPU for the
  auxiliary encoder path. Assert this THREE WAYS and require all three to match:
  `init_embeddings/step.toml` `min_gpus`, the step README's stated GPU count,
  and the shipped profile's `gpus_per_node`. A `min_gpus` that exceeds what the
  profile requests is a defect even though every command still runs — it
  misprices the job for anyone sizing a reservation from metadata. Only a
  large auxiliary model such as `gemma_weighted` may justify a higher figure,
  and then the metadata must say so per method rather than for the step.
- Generated env files expose extend, init, fertility, and BPB profiles for each
  supported backend, with required optional dependencies.

Evidence to collect: catalog, language registry, profile definitions, and
resource-consistency review.

### TOK-002 Extend Add Arm With Exact Budget

Use a BPE-compatible small HF model/tokenizer for plumbing, then repeat the
release arm with the intended Nemotron tokenizer. Example smoke command:

```bash
uv run nemotron steps run tokenizer_extension/extend -c default \
  model_id=sshleifer/tiny-gpt2 \
  trust_remote_code=false \
  language=hindi \
  method=add \
  extension_size=16 \
  corpus.hf_dataset=null \
  corpus.path="$TOK_ROOT/data/train_hi.jsonl" \
  "corpus.glob='*.jsonl'" \
  corpus.text_field=text \
  corpus.samples=100 \
  output_dir="$TOK_ROOT/tokenizers"
```

Success criteria:

- Output is under `tokenizers/add/` and loads with `AutoTokenizer`.
- `summary.json.tokens_spliced` equals 16 when at least 16 candidates exist.
- Final vocabulary growth equals the reported exact budget.
- Every new BPE token is reachable/emittable through its merge chain; no
  rank-dead token is present.
- Repeating with the same inputs/versions/seed produces the same token/merge
  ordering.

Evidence to collect: base/output vocab sizes, summary, tokenizer files,
reachability result, versions, and checksums.

### TOK-003 Replace And Expand Arms

Repeat `TOK-002` in separate jobs/output roots with `method=replace` and
`method=expand`.

Success criteria:

- Replace writes `id_remap.json`, densely reindexes survivors, prunes only the
  language-selected target script, and preserves byte alphabet/special tokens.
- Expand appends decoded atomic surfaces with no new merge rules.
- Each summary reports requested/available/actual budget; a candidate pool
  smaller than requested caps cleanly and never exceeds `extension_size`.
- Fertility comparisons use matched actual budgets; an unmatched capped arm is
  labelled non-comparable.
- Vietnamese replace pruning never treats plain ASCII Latin as target-specific.

Evidence to collect: per-arm summary, vocab/merge diff, remap validation,
special-token mapping, and encoded samples.

### TOK-004 Corpus And Dependency Failure Guards

Exercise copied configs for:

- neither HF nor local corpus set;
- both sources set;
- wrong HF config/split;
- missing text field;
- unsupported language/script;
- Indic `language=hindi` without `indic-nlp-library` in an isolated negative
  environment;
- memory-bounded retry with fewer samples, shorter documents, and higher
  `min_frequency`.

Success criteria:

- Invalid source/config/split/field errors enumerate actionable recovery and,
  where possible, available configs/splits.
- POSITIVE CONTROL for `text_field`: build the same corpus twice with two
  DIFFERENT populated field names and confirm the two tokenizers differ. The
  loader tries a fallback chain after the configured field, so a wrong or
  misspelled `text_field` can silently resolve to another field and produce a
  byte-identical tokenizer. A negative test alone cannot detect this.
- Setting BOTH `hf_dataset` and `path` is rejected up front with a message that
  names the conflict, not a downstream loader error.
- A run whose candidate pool resolves to 0 usable tokens FAILS. It must not
  exit 0 and save a tokenizer identical to the base while reporting success.
  Note that documents below the loader's minimum length are dropped silently,
  so a short-document corpus can reach 0 candidates with no diagnostic.
- Missing Indic normalizer hard-fails rather than silently using NFKC-only.
- No failed run leaves a tokenizer that looks complete.
- Memory mitigation changes are recorded as a new experiment, not merged into
  prior output.

Evidence to collect: commands, errors, partial-output inspection, and recovery
rerun.

### TOK-005 Language Profile And Override Precedence

Run small Hindi and Vietnamese `add`/`replace` builds. For a negative test set
`language=vietnamese` with `remove_script=devanagari` in a copied config.

Success criteria:

- Language-only configuration selects the documented normalizer/script/encoder
  and fastText defaults.
- Explicit `remove_script`, `script_normalizer`, `subword.bert_model`, and
  `focus.fasttext_model` override the language profile and are recorded in
  the durable `summary.json`, not only printed to stdout. A value that
  survives just in the job log does not satisfy this criterion.
- QA flags mismatched explicit overrides; the system does not claim they still
  represent the unmodified language profile.
- Legacy Hindi-neutral aliases remain backward compatible.

Evidence to collect: resolved configs, selected profile values, prune counts,
and warning/review result.

### TOK-006 Initialize Add/Expand Embeddings

Prerequisites: compatible tokenizer output from `TOK-002` and a base model whose
embedding row count matches that tokenizer's base vocabulary.

```bash
uv run nemotron steps run tokenizer_extension/init_embeddings -c default \
  base_model=sshleifer/tiny-gpt2 \
  trust_remote_code=false \
  language=hindi \
  arm=add \
  extended_tokenizer="$TOK_ROOT/tokenizers/add" \
  output_dir="$TOK_ROOT/checkpoints/add-subword" \
  method=subword \
  subword.input_averaging=uniform \
  subword.output_averaging=uniform
```

Success criteria:

- Output is a loadable HF checkpoint with tokenizer files.
- Input-embedding and LM-head row counts equal the extended tokenizer vocab,
  OR exceed it only by an explicitly logged padding to a tensor-parallel
  multiple. Padding is expected and correct; record the padded size, the
  multiple used, and confirm `config.json` `vocab_size` matches the padded row
  count. Rows beyond the tokenizer vocab must be unreachable by the tokenizer.
  Treat an UNANNOUNCED size difference as a defect.
- Existing append-arm rows remain unchanged **bit for bit**; only appended rows
  are initialized. Compare with exact equality, not a tolerance: a bf16
  round-trip inside the step alters every base row by ~1e-4 and still passes
  `allclose`. Record the saved `torch_dtype` alongside the comparison.
- Uniform subword initialization matches the mean-of-constituents algorithm.
- Input norm correction targets the configured median; output norm correction
  stays off by default.
- `expand` correctly pairs with `arm=add`.
- The step records the dtype it computed in and the dtype it saved. A step that
  computes in bf16 but writes fp32 pays full fp32 storage while carrying only
  bf16 precision, and silently perturbs every base row of an fp32 base model.
  Confirm the pairing is intentional and documented.

Repeat a release-scale smoke with the intended Nemotron base model and remote
GPU profile before signoff.

Evidence to collect: load test, shapes, old/new row comparisons, norm summary,
resolved config, and checkpoint checksums.

### TOK-007 Initialize Replace Embeddings And Method Matrix

Use the Replace tokenizer from `TOK-003` and test:

- baseline `hf_default` and `mean_all`;
- subword `uniform`, `char_weighted`, `max_char`, and `bert_weighted`;
- FOCUS with the correct target-language fastText file;
- negative cases: replace + `mean_target`, replace + `gemma_weighted`, missing
  `id_remap.json`, wrong arm, and missing fastText dependency/file.

Success criteria:

- Survivor embeddings copy exactly according to `id_remap.json` before new rows
  are initialized.
- Input/output averaging choices remain independent and are both forwarded.
- Unsupported Replace combinations fail loudly without substituting another
  method.
- FOCUS import is lazy: missing fastText affects FOCUS only, not baseline or
  subword.
- Base-model/tokenizer row mismatch fails before checkpoint save.
- FOCUS fastText staging is explicit about what it does. If `fasttext_model` is
  unset the step may auto-download the language's vectors; confirm it announces
  the URL, the size, and the destination BEFORE transferring, since the file is
  multi-GB. Documentation that calls the setting "required" while the code
  auto-fetches is a defect.
- The staged file lands under `FASTTEXT_CACHE_DIR`. Confirm every code path in
  the step agrees on that location and on its fallback, and that an empty
  `FASTTEXT_CACHE_DIR` is not accepted as a valid path. A default under `/tmp`
  is ephemeral on a container worker, so the multi-GB download repeats on every
  job; on a shared workstation it silently consumes local disk.
- After the case, confirm any cache written outside `QA_ROOT` is cleaned up.

Evidence to collect: method/arm matrix, remap row checks, shapes, failures, and
output integrity.

### TOK-008 Fertility Evaluation On Independent Corpus

```bash
for LABEL in base add replace expand; do
  case "$LABEL" in
    base) TOKENIZER=sshleifer/tiny-gpt2 ;;
    *) TOKENIZER="$TOK_ROOT/tokenizers/$LABEL" ;;
  esac
  uv run nemotron steps run tokenizer_extension/evaluate -c default \
    tokenizer="$TOKENIZER" \
    label="$LABEL" \
    corpus.hf_dataset=null \
    corpus.path="$TOK_ROOT/data/eval_hi.jsonl" \
    "corpus.glob='*.jsonl'" \
    corpus.text_field=text \
    corpus.num_docs=0 \
    output="$TOK_ROOT/eval/fertility_$LABEL.json"
done
```

Success criteria:

- Every JSON reports fertility, chars/token, unique tokens used, vocabulary
  coverage, document/token/word counts, and timing.
- All arms read exactly the same independent rows.
- Fertility is `sum(tokens)/sum(words)`, not an unweighted mean of per-row ratios.
- Results are compared only for matched budgets/corpus/preprocessing.
- Full-corpus streaming is memory-bounded; `skip_docs` can exclude a known
  training slice.

Evidence to collect: four reports, row checksums/counts, peak memory for a
larger stream, and comparison table.

### TOK-009 BPB Evaluation Across Vocabularies

Prerequisites: at least one initialized checkpoint from `TOK-006`/`TOK-007`.

```bash
uv run nemotron steps run tokenizer_extension/eval_init -c default \
  base_model=sshleifer/tiny-gpt2 \
  trust_remote_code=false \
  'models=['"$TOK_ROOT"'/checkpoints/add-subword]' \
  data_file="$TOK_ROOT/data/eval_hi.jsonl" \
  text_field=text \
  max_docs=2 \
  max_tokens=-1 \
  output_json="$TOK_ROOT/eval/bpb_hi.json"
```

Success criteria:

- Base and extended model score the same document bytes.
- Output reports loss, perplexity, BPB, and byte/token counts for the base
  reference and every candidate, so a delta vs base is derivable. If no delta
  field is emitted, record that the comparison is left to the reader and confirm
  the base row is unambiguously labelled as the reference.
- QA uses BPB—not per-token loss/perplexity—for cross-vocabulary comparison.
- A multi-model run with only `max_tokens` and no document cap is refused unless
  the explicit historical-comparison escape hatch is used.
- A separate English run records base-language regression.
- The case is executed on a GPU host and the host type is recorded. The
  `device_map="auto"` path is only taken when CUDA is visible, so a CPU-only
  run does not exercise the same code and must not be reported as a pass.

Evidence to collect: Hindi/English BPB JSON, byte equality, model/tokenizer
identities, and negative-test error.

### TOK-010 Conversion And Downstream CPT Handoff

Run the focused converter tests, then compile a Megatron conversion/CPT handoff
using the initialized HF checkpoint.

Success criteria:

- Internal tokenizer wrapper class names are replaced with a public Transformers
  tokenizer class when a `tokenizer.json` is available.
- Existing real class names and unrelated tokenizer config fields, including
  long `model_max_length`, are preserved.
- Missing/unreadable tokenizer config warns without corrupting conversion.
- The initialized checkpoint is accepted as the CPT `hf_model_path` and the
  exact tokenizer travels with it.
- Post-CPT evaluation routes through `eval/model_eval` with that same tokenizer.

Evidence to collect: before/after tokenizer config, converter logs, compiled CPT
config, and lineage to `EVAL-010`.

### TOK-011 Remote Profile Matrix And Release-Scale Smoke

Compile all tokenizer profiles for Lepton, Slurm, and DGX Cloud/Run:ai. Run at
least:

- one CPU `extend` build;
- one GPU `init_embeddings` build;
- one CPU fertility evaluation;
- one GPU BPB evaluation.

Success criteria:

- `extend` and fertility profiles request CPU-appropriate resources.
- Init and BPB profiles request sufficient GPU/memory for the selected model;
  one GPU is sufficient for shipped default auxiliary methods, while a large
  Gemma-weighted auxiliary model may require a multi-GPU override.
- Remote profiles install `indic-nlp-library`; FOCUS profiles also install
  `fasttext-wheel`; BPB profiles also install `accelerate`.
- Every tokenizer profile declares its own `resource_shape`, `gpus_per_node`,
  and a COMPLETE `pip_extras`. `pip_extras` is a list, so a child profile
  REPLACES the parent's rather than extending it: a profile that omits it
  silently ships without its dependencies, and one that omits
  `resource_shape` silently inherits the base GPU shape.
- HF cache, corpus, tokenizer, checkpoint, and result paths are mounted and
  durable across detached jobs.
- Release-scale `summary.json.tokens_spliced` equals the requested budget, or a
  capped candidate pool is reported and the comparison plan is adjusted.

Evidence to collect: compiled specs, job IDs, GPU/CPU/memory observations,
artifacts, summaries, and mount checks.

### TOK-012 Agent-Driven Tokenizer Extension Workflow

Give the agent this intent:

> Extend the Nano base tokenizer for Vietnamese by 30,000 tokens using Add,
> initialize the new rows with uniform subword averaging, compare fertility to
> the base tokenizer on held-out Vietnamese, and compare BPB on the same bytes.
> Prepare the resized checkpoint for CPT and later evaluation.

Success criteria:

- The agent discovers and orders all four tokenizer steps correctly.
- It sets `language=vietnamese`, `method=add`, and pairs it with `arm=add`.
- It uses distinct train/eval corpora, exact/matched extension budgets, the
  correct text fields, and a document rather than token budget for BPB.
- It validates reachability, vocab/embedding shapes, preserved base rows,
  fertility, BPB, checkpoint/tokenizer pairing, and durable outputs.
- It does not use MuRIL/Devanagari overrides for Vietnamese or treat perplexity
  as cross-vocabulary evidence.

Evidence to collect: transcript, configs/commands, artifact lineage, and metric
interpretation.

### TOK-013 Corpus Loader Contract

The extend corpus loader resolves the text field through a fallback chain and
applies an undocumented minimum document length. Both behaviors are silent, and
neither is covered by the guards in `TOK-004`.

Exercise, on a copied config:

- a JSONL with two populated fields, built once per field, plus once with a
  nonexistent field name;
- a corpus whose documents are all shorter than the loader minimum;
- `hf_dataset` and `path` set together.

Success criteria:

- Selecting a different populated `text_field` produces a DIFFERENT tokenizer.
  Identical output across different field names means the setting is ignored.
- A nonexistent `text_field` fails; it does not silently fall back to another
  field.
- An all-short-document corpus reports how many documents were dropped and why,
  and does not report success with 0 spliced tokens.
- Both sources set is rejected before any load is attempted.

Evidence to collect: per-field tokenizer checksums, summary counters, drop
diagnostics, and error text.

### LCSDG-011 Endpoint Context Budget

Prerequisites: the assistant and user-simulator endpoints from `LCSDG-006`.

Read the served context window from each endpoint's `/v1/models` and compare it
with the `max_tokens` configured for every model alias that uses it.

Success criteria:

- Each alias's `max_tokens` plus its worst-case prompt fits the served window.
- The context sizes stated in config comments match the endpoints actually in
  use. A comment that disagrees with the deployment is filed as a defect, since
  it is the only sizing guidance a user has.
- A trajectory that exceeds the window fails visibly rather than being silently
  truncated into a low-quality row that later passes the objective gate.

Evidence to collect: `/v1/models` output per endpoint, configured
`inference_parameters`, and the observed context-length distribution.

### PMC-011 Asset Staging Cost And Cache Reuse

Success criteria:

- The staging download size and duration are recorded before the tiny run is
  scheduled; the shipped locales are multi-GB and take double-digit minutes.
- Only the unique configured locales download. A target language that reuses
  another locale's personas does not trigger its own download.
- A second run with the key unset completes from cache.
- No NGC configuration file containing the key is written under the experiment
  or the managed-assets directory.

Evidence to collect: transfer size/duration, asset listing, cached-rerun log,
and secret scan.

## Cross-Feature Security And Reproducibility

### SEC-001 Credential Scan

After all cases, scan only QA-generated text/config artifacts using secret
names and a securely supplied fingerprint of each actual secret. Do not print
the secrets while constructing the scan.

Success criteria:

- Raw model, NGC, HF, endpoint, embedding, retriever, and W&B tokens do not
  appear in logs, configs, manifests, summaries, commands, or checked-in files.
- Endpoint URLs in manifests have userinfo and query strings removed.
- Lepton secrets are references in `secret_vars`/platform secret fields, not
  plaintext job `env_vars`.
- Secret *variable names* and redacted placeholders remain for diagnosis.

Evidence to collect: scan command/method and zero-match result, with secret
values omitted.

### REP-001 Reproducibility Bundle

For each successful live case, archive or register:

- branch/base/head/merge-base SHAs and dirty status;
- copied config and resolved redacted config;
- package versions and container image digest;
- endpoint model handle/type and tokenizer revision/path;
- dataset/corpus/retriever identifiers and checksums where permitted;
- seeds, limits, task/subset names, job IDs, commands, and exit status;
- summaries, manifests, metrics, logs, and output checksums.

Success criteria: another QA engineer can identify the exact code, config,
inputs, runtime image, model/tokenizer pair, and output without access to a raw
credential.

## Reporting

Use one row per case:

| Case | Status | Failed criterion | Backend | SHA | Job ID | Evidence path | Defect/blocker |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `EVAL-004` | PASS/FAIL/BLOCKED | `<criterion text, or -->` | Lepton | `<sha>` | `<id>` | `<path>` | `<id/reason>` |

Record the specific criterion that failed. Many cases pass their command and
fail one sub-criterion; without this column a report cannot distinguish "the
command broke" from "the contract was violated".

Status values are defined in **Status Definitions** above.

File defects separately when metadata, README, checked-in config, and runtime
disagree even if QA can work around the issue.

## Exit Criteria

GA_v2 feature signoff requires:

- `SETUP-001` through `SETUP-005`, `SEC-001`, and `REP-001` pass.
- Every in-scope step is discoverable and every checked-in config parses.
- All new offline tests pass in clean, correctly isolated environments.
- One real-corpus release run completes on an ungated public corpus, or is
  recorded `BLOCKED (no network)`. Fixture cases prove the machinery; they
  cannot show how the module behaves on real text.
- All six curate steps are exposed, all six CPU smoke paths pass, a policy is
  measured and deliberately approved against the same corpus fingerprint, and
  a drifted corpus is refused. Unicode/language-pack guards pass, audit
  reconciles the filter ledger, and every subset tier is cut only from the
  decontaminated corpus. The GPU similarity arm may be `BLOCKED` only when the
  required GPU runtime is unavailable; the CPU identity/order arm must pass.
- At least one live direct evaluation smoke passes on a remote backend, with
  durable summary/manifest/log artifacts and no credential leak.
- Both a base/completions and instruct/chat evaluation path are validated; each
  uses the matching client-side tokenizer.
- `BFCL-001` completes from only the checked-in hotel domain brief and tool
  catalog: intake earns A2, the assembled and frozen packs pass Gold, a real
  publication writes both Parquet files plus a Gold-eligible run manifest, and
  candidate evaluation completes from that manifest. Any missing operator
  instruction is tracked as a documentation defect.
- Persona MCQ completes one tiny end-to-end live run, validates all three
  shipped languages, resumes safely, and produces a usable downstream blend.
- Long-context SDG completes a five-row generation and objective export. The
  HTTP-retrieval arm is required only when a retriever is available; otherwise
  the simulated arm satisfies this criterion and every row must remain
  explicitly labelled `retrieval_mode=simulated`. Simulated results are never
  reported as retrieval-grounded quality evidence.
- Tokenizer extension completes Add plus at least one Replace or Expand arm,
  initializes a checkpoint, and produces fertility and same-byte BPB reports.
- All backend profiles compile. A live remote run is required on Lepton only;
  Slurm and DGX Cloud/Run:ai are compile-only and out of scope for live
  execution. Other unavailable live infrastructure is documented,
  with at least one live remote workflow per feature group where the group
  supports remote execution.
- The 128K/256K SFT configs pass static/dry-run review and remain clearly blocked
  from stock-runner launch until the runner supports their packing/DDP contract.
- No unresolved P0/P1 defect remains. Any accepted lower-severity issue has an
  owner, rationale, and documented user impact/workaround.
- Agent-driven cases choose the correct workflow and reproduce the same config,
  artifact, safety, and interpretation contracts as direct CLI cases.
