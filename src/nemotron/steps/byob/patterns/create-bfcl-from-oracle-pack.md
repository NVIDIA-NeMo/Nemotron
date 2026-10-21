---
id: create-bfcl-from-oracle-pack
title: Create a function-calling benchmark from an oracle pack
tags: [byob, benchmark, function-calling, bfcl]
triggers:
  - Generate a function-calling benchmark from executable tools and fixtures.
  - Validate an oracle pack and produce replay-verified tool-call conversations.
steps: [byob]
confidence: high
---

Use the BFCL family and follow `references/bfcl-oracle-pack.md`. Start from
`bfcl/config/default.yaml`, point `oracle_pack.manifest_path` at the pack, and keep
all domain data, tool names, fixtures, and assertions inside that pack.

Run `stage=prepare` first to inspect `oracle_validation_report.json`. Generation
requires every validation check to pass with `oracle_runtime.worker: process`.
Then run `stage=generate`, or use `stage=all`, to expand templates, render the
conversation, validate the expected trace, replay it twice, and write the parquet
artifacts plus `run_manifest.json`.

Set `exports.bfcl_json` and/or `exports.nemo_evaluator_bundle` to emit optional
compatibility trees from the published parquet. Stage 12 reads them back for
equivalence and writes `exports/export_validation_report.json` before committing
`run_manifest.json`; the NeMo bundle remains input for a native-tool adapter.

To evaluate candidate models on the result, write a separate `eval_config.yaml`
from `bfcl/config/eval.default.yaml`. It points at the run's `run_manifest.json`
rather than the parquet, pins the scoring contract in
`references/bfcl-eval-scoring-contract.md` by content hash, and requires an
immutable commit or weights digest per candidate. Executable mode also pins the
matching pack manifest and concrete backend.py or endpoint config in
`source_oracle`; the lineage label in `run_manifest.json` cannot locate a resource
by itself. Keeping the eval config separate is what stops a new candidate from
changing the benchmark's own lineage hash; `stage=generate` refuses
`eval_config_path` and inline `eval` blocks because evaluation has its own
resolved configuration and runner.

A valid config only records what was named, so `verify_eval_source()` reads the
source back before any candidate is contacted: the manifest still hashes to what
was resolved, both tables match every hash the publication declares, the
published table is an unmodified selection of the raw table with no held-out row,
the rows decode into a unique task set, and an executable run's oracle pack still
fingerprints to what generation certified. It also records every model that read
a published row — the profile, paraphrase, and surface-judge roles, plus the
translator of a translated benchmark — with the rows each one read. It writes
`source_verification_report.json` into the eval output directory, and
`assert_source_unchanged()` re-pins everything immediately before execution so a
run cannot span two sources.

`evaluate_contamination()` then gates the candidates against that inventory and
returns the task set each one is authorized to answer. A candidate that turns out
to be the model that paraphrased, judged, or translated the rows is refused under
the locked policy; a comparison that cannot tell the two apart is recorded and
blocks publication rather than being guessed either way. Under
`common_intersection` every candidate answers the same rows, so two numbers are
comparable by construction. The decision lands in `contamination_report.json`,
and `assert_plan_unchanged()` re-derives it immediately before the first request.

## Scaffold a pack

Create a runnable starter from the repository root:

```bash
python -m nemotron.steps.byob.scripts.scaffold_oracle_pack \
  --domain "inventory service" \
  --target /tmp/inventory_oracle_pack \
  --transport python \
  --include-held-out
```

The command atomically writes the complete pack contract, optional
`held_out.yaml`, pack README, and `validate.yaml`; it refuses to overwrite an
existing target. Its generic `get_record` examples cover success, structured
error, and irrelevant requests. Replace those sample concepts with reviewed
domain evidence rather than editing BFCL runtime code.

## Python-backend quick start

```bash
PACK=/tmp/inventory_oracle_pack
python -m nemotron.steps.byob.scripts.validate_oracle_pack \
  --config "$PACK/validate.yaml" \
  --output-dir /tmp/inventory-pack-validation
python -m nemotron.steps.byob.scripts.run \
  --config "$PACK/validate.yaml" \
  --stage prepare
python -m nemotron.steps.byob.scripts.run \
  --config "$PACK/validate.yaml" \
  --stage generate
```

`backend.py` must restore isolated fixture state in `reset`, validate every
argument in `call_tool`, return structured business errors, expose copied state,
and derive times or generated IDs from the supplied context. Declare mutations
and confirmations in `tools.json` and prove no pre-confirmation state change in
`validation_cases.yaml`.

## Endpoint-backed quick start

```bash
python -m nemotron.steps.byob.scripts.scaffold_oracle_pack \
  --domain "inventory service" \
  --target /tmp/inventory_endpoint_pack \
  --transport endpoint
PACK=/tmp/inventory_endpoint_pack
export BFCL_ORACLE_TOKEN=REPLACE_ME
python -m nemotron.steps.byob.scripts.validate_oracle_pack \
  --config "$PACK/validate.yaml" \
  --output-dir /tmp/inventory-endpoint-validation
```

Replace the scaffold's `.invalid` URL and zero digest with values observed from
`GET /v1/metadata`. The server implements BFCL Oracle HTTP v1 metadata, tools,
session creation, tool call, state, and session deletion routes. Keep credentials
out of the pack and pin TLS, identity, permissions, and optional attestation.

## Publication and evaluation checkpoints

After prepare succeeds, use `--stage all` for a full run. Inspect
`oracle_validation_report.json`, stage reports, both Parquet files, and the final
`run_manifest.json`. Resolve `bfcl/config/eval.default.yaml` against that
manifest. Trace mode opens no Oracle; executable mode points `source_oracle` at
the exact Python backend or endpoint configuration used by generation:

```bash
python -m nemotron.steps.byob.scripts.run \
  --config /path/to/eval.cli.yaml \
  --stage eval
```
