# BYOB BFCL

The `bfcl` benchmark family builds function-calling benchmark artifacts from an
executable Oracle Pack. Pack templates define conversations; a local Python
backend or pinned HTTPS endpoint and deterministic assertions establish the
expected behavior. Every task is replayed against the oracle before publication.

## Quick start

Install the BYOB dependencies:

```bash
uv sync --extra byob
```

Run the bundled tiny pack from the repository root:

```bash
nemotron steps run byob/bfcl -c src/nemotron/steps/byob/bfcl/config/tiny.yaml stage=all family=bfcl
```

The tiny config writes a deliberately non-publishable smoke run under
`/tmp/bfcl/tiny_out`. Validate a pack without generating rows with
`stage=prepare`, or run:

```bash
python -m nemotron.steps.byob.scripts.validate_oracle_pack --config <CONFIG>
```

## Generation pipeline

`prepare` normalizes and validates the pack. `generate` requires a gold-eligible
validation report and runs the generation stages. `all` chains those operations.

The generation stages are `reference_profile`, `expand`, `state_machine`,
`render`, `expected_trace`, `schema_validation`, `executable_replay`, optional
`surface_quality`, optional `dedup_balancing`, and `final_output`. Each stage
writes a task-indexed artifact and a verified checkpoint under `stage_cache/`.
`skip_until=<stage>` resumes only after recursively verifying the predecessor
chain, config identity, task order, and pack or endpoint identity.

The pipeline publishes:

- `benchmark_raw.parquet`, containing every replay-valid row;
- `benchmark.parquet`, a selection of raw rows without truth-field rewrites;
- optional BFCL JSONL and NeMo Evaluator compatibility bundles under `exports/`;
- `exports/export_validation_report.json`, proving enabled exports match the
  canonical projection; and
- `run_manifest.json`, the publication commit marker written last.

If `run_manifest.json` is absent, the neighboring artifacts are not a completed
publication.

## Bundled packs and configs

`data/tiny_oracle_pack` is the smallest runnable example.
`data/banking_vn_oracle_pack` exercises all supported conversation policies.
The generation configs under `config/` include tiny, smoke, publication, and
paraphrase examples. Their budgets are examples for those packs, not framework
defaults.

Model-assisted generation roles are optional. Enabled roles use Data Designer;
their canonical identities and routes are recorded in the run lineage.

## Author and operate packs

- [Getting started](../../../../../docs/build-benchmarks/function-calling/getting-started.md)
- [Hand-author an Oracle Pack](../../../../../docs/build-benchmarks/function-calling/how-to/author-a-pack.md)
- [Use assisted authoring](../../../../../docs/build-benchmarks/function-calling/how-to/assisted-authoring.md)
- [Start from domain assets](../../../../../docs/build-benchmarks/function-calling/how-to/start-from-domain-data.md)
- [Onboard an MCP server](../../../../../docs/build-benchmarks/function-calling/how-to/mcp-server.md)
- [Publish a reviewed release](../../../../../docs/build-benchmarks/function-calling/how-to/publish-a-release.md)
- [Oracle Pack explanation](../../../../../docs/build-benchmarks/function-calling/explanation/oracle-pack.md)
- [Authoring flows](../../../../../docs/build-benchmarks/function-calling/explanation/authoring-flows.md)
- [Pipeline overview](../../../../../docs/build-benchmarks/function-calling/explanation/pipeline-overview.md)
- [Worked pipeline example](../../../../../docs/build-benchmarks/function-calling/explanation/pipeline-worked-example.md)
- [Generation configuration](../../../../../docs/build-benchmarks/function-calling/reference/generate-config.md)
- [Output files](../../../../../docs/build-benchmarks/function-calling/reference/output-files.md)
- [Troubleshooting](../../../../../docs/build-benchmarks/function-calling/reference/troubleshooting.md)

The normative engineering contracts are:

- [`../references/bfcl-oracle-pack.md`](../references/bfcl-oracle-pack.md)
- [`../references/bfcl-manual-oracle-pack-flow.md`](../references/bfcl-manual-oracle-pack-flow.md)
- [`../references/bfcl-bias-audit-contract.md`](../references/bfcl-bias-audit-contract.md)
- [`../references/bfcl-banking-vn-pack-operations.md`](../references/bfcl-banking-vn-pack-operations.md)

Create a complete pack skeleton with:

```bash
python -m nemotron.steps.byob.scripts.scaffold_oracle_pack \
  --domain <NAME> --target <EMPTY_PATH> --transport python
```

Use `--transport endpoint` for a pinned HTTPS Oracle HTTP v1 skeleton. The
scaffolder never overwrites an existing target.
