<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(bfcl-manual-pack-lifecycle-tutorial)=
# Publish and Evaluate a Manual Oracle Pack

Use this tutorial after you have a reviewed Oracle Pack and want to carry it through the
complete publication and evaluation lifecycle. You will validate the pack without model
traffic, publish a benchmark, bind an independent candidate and the exact source oracle,
run evaluation preflight, and inspect the committed results.

If you do not have a pack yet, first complete {doc}`../how-to/author-a-pack`. The required
files and cross-file relationships are listed in {doc}`../reference/oracle-pack-inputs`.

## Before You Start

You need:

- Python 3.11–3.13, `uv`, and a Nemotron clone;
- a reviewed pack containing exactly one oracle implementation—`backend.py` or
  `endpoint_config.yaml`;
- persistent output storage outside the pack directory;
- an OpenAI-compatible candidate endpoint for the evaluation portion; and
- an immutable candidate revision or weights digest for a publication-eligible score.

Install the runtime from the repository root:

```console
$ uv sync --extra byob
$ export NEMOTRON_ROOT="$PWD"
```

Set absolute locations for your pack and run. Relative paths in generation configs resolve
from `src/nemotron/steps/byob`, not from your shell or the copied config.

```console
$ export BFCL_PACK_ROOT=/absolute/path/to/oracle-pack
$ export BFCL_PACK_MANIFEST="$BFCL_PACK_ROOT/manifest.yaml"
$ export BFCL_RUN_ROOT=/persistent/path/to/bfcl-runs/my-pack-v1
$ export BFCL_GEN_CONFIG="$BFCL_RUN_ROOT/generation.yaml"
$ mkdir -p "$BFCL_RUN_ROOT"
```

## Step 1: Resolve a Generation Configuration

Copy the closest checked-in profile rather than starting from an empty file:

```console
$ cp src/nemotron/steps/byob/bfcl/config/default.yaml "$BFCL_GEN_CONFIG"
```

Resolve every placeholder and set at least:

- `config_status: resolved`, `family: bfcl`, and `stage: all`;
- a unique `expt_name` and persistent `output_dir`;
- `oracle_pack.manifest_path` to the absolute manifest path;
- `oracle_runtime.worker: process` and `allowed_roots` containing the pack root;
- a frozen clock and explicit timeouts; and
- task budgets derived from this pack's reachable inventory.

Do not copy another domain's task counts, balance targets, or cluster count. Use
{doc}`../reference/generate-config` for every field and
{doc}`../how-to/publish-a-release` for publication sizing and exports.

## Step 2: Validate Before Generation

Run the standalone validator for the fast authoring loop:

```console
$ uv run python -m nemotron.steps.byob.scripts.validate_oracle_pack \
    --config "$BFCL_GEN_CONFIG"
```

Then run the pipeline preflight:

```console
$ uv run nemotron steps run byob/bfcl \
    -c "$BFCL_GEN_CONFIG" \
    stage=prepare \
    family=bfcl
```

Inspect `<output_dir>/<expt_name>/stage_cache/oracle_validation_report.json`. Continue
only when it records `gold_eligible: true`. Do not edit the report or the pack after this
check; generation validates the source again and rejects drift.

## Step 3: Generate and Commit the Benchmark

```console
$ uv run nemotron steps run byob/bfcl \
    -c "$BFCL_GEN_CONFIG" \
    stage=generate \
    family=bfcl
```

Set the resulting publication directory and verify its commit marker:

```console
$ export BFCL_PUBLICATION_DIR=/absolute/path/to/output_dir/expt_name
$ export BFCL_RUN_MANIFEST="$BFCL_PUBLICATION_DIR/run_manifest.json"
$ test -f "$BFCL_RUN_MANIFEST"
$ test -f "$BFCL_PUBLICATION_DIR/benchmark.parquet"
$ test -f "$BFCL_PUBLICATION_DIR/benchmark_raw.parquet"
```

`run_manifest.json` is written last. A Parquet file without its matching manifest is not a
published BFCL benchmark.

## Step 4: Resolve an Independent Evaluation

Create a separate directory; never write evaluation artifacts into the publication:

```console
$ export BFCL_EVAL_ROOT="$BFCL_RUN_ROOT/eval/candidate-v1"
$ mkdir -p "$BFCL_EVAL_ROOT"
$ cp src/nemotron/steps/byob/bfcl/config/eval.default.yaml "$BFCL_EVAL_ROOT/eval.yaml"
$ cp src/nemotron/steps/byob/bfcl/config/eval.cli.yaml "$BFCL_EVAL_ROOT/eval.cli.yaml"
```

In `eval.yaml`, resolve:

- `source_run_manifest` to the committed generation manifest;
- `source_oracle` to the exact pack manifest and backend or endpoint resource used by generation;
- `eval.mode` to `[trace, executable]` for a complete evaluation;
- `scoring.contract` to the absolute path of the checked-in BFCL scoring contract;
- the candidate endpoint, credential environment-variable name, served model ID, and immutable identity; and
- `outputs.output_dir` to a new path under `$BFCL_EVAL_ROOT`.

The candidate must be independent of every model exposed to benchmark rows during
profiling, paraphrasing, judging, or translation. Use {doc}`../reference/eval-config` for
the schema and {doc}`../how-to/run-evaluation` for candidate identity and backend choices.

## Step 5: Preflight Without Candidate Inference

Keep `dry_run: true` in `eval.cli.yaml` and point `eval_config_path` at the absolute
path to `$BFCL_EVAL_ROOT/eval.yaml`. YAML does not expand the shell variable, so write
the resolved path into the file before running:

```console
$ uv run nemotron steps run byob/bfcl -c "$BFCL_EVAL_ROOT/eval.cli.yaml"
```

The result should report `status: preflight_passed` and
`candidate_network_used: false`. Resolve source-verification, oracle-probe, identity, and
contamination failures before spending candidate tokens.

## Step 6: Run and Inspect the Evaluation

Change only `dry_run` to `false`, then run the same command:

```console
$ uv run nemotron steps run byob/bfcl -c "$BFCL_EVAL_ROOT/eval.cli.yaml"
```

A complete run writes `eval_report.json`, task-level Parquet results, source-verification
and contamination reports, caches, and `eval_manifest.json`. The evaluation manifest is
the final commit marker, just as `run_manifest.json` is for generation.

Interpret trace and executable metrics separately: trace success means the candidate
proposed the expected calls; executable success additionally means those calls ran against
the oracle and satisfied the pack assertions.

## Completion Checklist

- Fresh validation awarded Gold eligibility.
- `run_manifest.json` committed the generated benchmark.
- Evaluation used the exact source oracle and an independent candidate.
- Preflight passed before candidate traffic was enabled.
- Source verification and contamination gates passed.
- `eval_manifest.json` committed the evaluation artifact set.

For the exhaustive lifecycle, including paraphrasing, recovery, and reference deployment
values, consult the source-tree
[`bfcl-manual-oracle-pack-flow.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/byob/references/bfcl-manual-oracle-pack-flow.md).

## Next Steps

- Use {doc}`../how-to/publish-a-release` to audit and archive the publication.
- Use {doc}`../how-to/run-evaluation` to compare candidates or use the launcher backend.
- Use {doc}`../reference/output-files` to inspect every generation and evaluation artifact.
