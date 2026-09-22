<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(bfcl-assisted-authoring-demo-tutorial)=
# Run Assisted Authoring End to End

Use this tutorial to observe the complete assisted-authoring lifecycle in one run. The
demonstration starts with a reviewed local Python source, certifies it through live probes,
drafts an Oracle Pack, crosses explicit human-review boundaries, validates and freezes the
pack, publishes a benchmark, and evaluates a candidate against it.

The default run needs no model endpoint or API key. It uses scripted authoring responses
and a loopback evaluation candidate, while the source probes, certification, grounding,
Gold validation, publication, and scoring paths remain real.

## Outcomes

By the end of the tutorial, you will have:

- an A2-certified source evidence bundle;
- a reviewed and frozen Gold Oracle Pack;
- a committed benchmark with `run_manifest.json`;
- a complete trace-evaluation report; and
- a second evaluation showing how a deliberately incorrect candidate is attributed.

## Before You Start

- Use Python 3.11–3.13 and run commands from the repository root.
- Install `uv`.
- Choose a temporary work directory that does not already exist. The demonstration
  refuses existing state so stale evidence or approvals cannot enter the walkthrough.

Install the BFCL dependencies and confirm the demonstration entry point:

```console
$ uv sync --extra byob
$ uv run python scripts/bfcl_assisted_authoring_demo.py --help
```

## Step 1: Run the Credential-Free Journey

```console
$ export BFCL_DEMO_ROOT="${TMPDIR:-/tmp}/bfcl-assisted-authoring-walkthrough"
$ test ! -e "$BFCL_DEMO_ROOT"
$ uv run python scripts/bfcl_assisted_authoring_demo.py \
    --workdir "$BFCL_DEMO_ROOT"
```

The script prints nine numbered stages:

| Stage | What happens |
| --- | --- |
| 1. Intake | Probe the reviewed source and derive its certification tier. |
| 2. Authorize | Record whether the reviewed source may be exposed to an authoring model. |
| 3. Approve evidence | Bind human approval to the normalized evidence digests. |
| 4. Draft | Propose coverage, validation cases, task templates, and assertions. |
| 5. Assemble | Combine the drafts with human-owned semantic decisions. |
| 6. Validate | Run unmocked Oracle Pack validation and require Gold eligibility. |
| 7. Review and freeze | Bind the review packet, approval, and immutable release. |
| 8. Publish | Validate again, generate benchmark rows, and commit the manifest. |
| 9. Evaluate | Score a loopback candidate through the real trace evaluator. |

The output labels the four simulated human decisions. Those labels are important: a
model may propose pack material, but it cannot authorize its own source access, approve
its own evidence, supply human-owned semantics, or approve its release.

## Step 2: Inspect the Published Artifacts

The committed benchmark is under:

```text
<workdir>/workspace/generated/bfcl-demo/
├── benchmark.parquet
├── benchmark_raw.parquet
├── run_manifest.json
└── stage_cache/
```

The first evaluation is written to `<workdir>/eval-1/`. Confirm the commit markers:

```console
$ test -f "$BFCL_DEMO_ROOT/workspace/generated/bfcl-demo/run_manifest.json"
$ test -f "$BFCL_DEMO_ROOT/eval-1/eval_manifest.json"
```

The loopback candidate returns the benchmark's recorded assistant turns. A clean run
therefore scores `1.0`; this proves the generated benchmark is passable and the evaluator
is wired, not that an independent model is perfect.

## Step 3: Make the Evaluator Catch a Failure

Select a published task ID:

```console
$ export BFCL_WRONG_TASK="$(uv run python -c \
  'import pyarrow.parquet as pq, sys; print(pq.read_table(sys.argv[1], columns=["task_id"])["task_id"][0].as_py())' \
  "$BFCL_DEMO_ROOT/workspace/generated/bfcl-demo/benchmark.parquet")"
```

Re-evaluate the same benchmark while forcing that task to return text where a function
call is expected:

```console
$ uv run python scripts/bfcl_assisted_authoring_demo.py \
    --workdir "$BFCL_DEMO_ROOT" \
    --stage eval \
    --wrong-answer-task "$BFCL_WRONG_TASK"
```

The new evaluation is written to `<workdir>/eval-2/`. Its task-success rate drops and its
failure records identify the failed gate and attribution. The earlier evaluation remains
unchanged because committed evaluation directories are immutable.

## Step 4: Relate the Demo to a Real Authoring Run

The demonstration substitutes scripted authoring responses and printed review decisions.
A real run must provide an authorized model route and actual reviewers. The same prompts
can be sent to a live authoring model with `--author-model live` plus the model-provider,
model, and canonical-identity arguments, but do that only after configuring the provider
and reviewing the source-exposure policy.

For the production command sequence and every recovery path, use
{doc}`../how-to/assisted-authoring`. The normative command and approval contracts remain
in the [`bfcl-authoring-user-guide.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/byob/references/bfcl-authoring-user-guide.md)
and [`bfcl-assisted-authoring-runbook.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/byob/references/bfcl-assisted-authoring-runbook.md).

## Next Steps

- Use {doc}`../explanation/authoring-flows` to compare manual, conventional-source, and MCP authoring.
- Follow {doc}`../how-to/assisted-authoring` with your reviewed source and real approval identities.
- Follow {doc}`../how-to/run-evaluation` to replace the loopback candidate with an independent model endpoint.
