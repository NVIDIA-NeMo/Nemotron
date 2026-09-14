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

## Publication checkpoints

After prepare succeeds, use `--stage all` for a full run. Inspect
`oracle_validation_report.json`, stage reports, both Parquet files, optional
compatibility exports, and the final `run_manifest.json` commit marker.
