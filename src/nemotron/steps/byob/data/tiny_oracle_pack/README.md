# Tiny library Oracle pack

This is the smallest complete local-Python BFCL example. It is offline and
deterministic; Gold claims require process-worker isolation.

- Pack ID: `tiny_library`
- Tools: `get_book_status` and mutating, confirmation-protected `checkout_book`
- Deliberately absent ID: `BK-ABSENT-1`

## File map

- `manifest.yaml`: identity, language, frozen clock, absent IDs, messages, and
  authoritative paths.
- `tools.json`: strict model-facing schemas plus mutation and confirmation flags.
- `fixtures.json`: reset state and slot inventory for books, patrons, and loans.
- `backend.py`: deterministic reset/state, dispatch, errors, and confirmed mutation.
- `task_templates.yaml`: lookup, confirmation, parallel, and irrelevant examples.
- `assertions.py`: result/state/no-tool checks and capability declarations.
- `validation_cases.yaml`: success, not-found, confirmation, mutation, and type probes.
- `README.md`: this file map and runnable recipe.

## Validate and run

From the repository root:

```bash
CONFIG="$(pwd)/src/nemotron/steps/byob/bfcl/config/tiny.yaml"
python -m nemotron.steps.byob.scripts.validate_oracle_pack \
  --config "$CONFIG" \
  --output-dir /tmp/bfcl-tiny-validation
python -m nemotron.steps.byob.scripts.run \
  --config "$CONFIG" \
  --stage prepare
python -m nemotron.steps.byob.scripts.run \
  --config "$CONFIG" \
  --stage generate
```

The standalone validator exits zero and prints the structured report that
`prepare` writes as `oracle_validation_report.json`. Generation expands all four
templates, replays traces in process workers, and writes raw/published Parquet
plus `run_manifest.json`. The bundled config uses `smoke_no_publication`, so it
verifies plumbing rather than production lineage.

```bash
python -m nemotron.steps.byob.scripts.run \
  --config "$(pwd)/src/nemotron/steps/byob/bfcl/config/tiny.yaml" \
  --stage all
```
