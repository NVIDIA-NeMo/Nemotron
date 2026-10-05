# Curated Corpus Integrity Audit

Use `curate/audit` to find out whether a curated corpus is intact before you
train on it. It reads the files on disk — no Ray cluster, no GPU — and reports
per-shard readability, row counts, and a content digest.

Use this README for workflow and pitfalls; use `step.toml` for the exact
artifact, parameter, strategy, and error manifest before editing configs.

## What It Detects

A curation run can lose records and still exit 0. Workers that catch every
exception, log it, and carry on leave a corpus that looks finished: the job
succeeded, the output directory has files in it, and every file parses. The
missing rows are invisible until something downstream depends on them.

The audit catches that by measuring the delivered files and comparing them
against what the producing step said it wrote.

```text
filtered_jsonl  ->  curate/audit  ->  curation_report (+ non-zero exit on findings)
```

## Three Things It Will Not Claim

**Readable is not complete.** A well-formed JSONL file can be missing rows.
Without a manifest the report says so explicitly and reports counts as
informational rather than asserting the corpus is whole.

**A row delta is not an error.** Filtering removes records on purpose. When
`reference_glob` is set the report carries the delta as an observation. Only a
count that contradicts the producer's own manifest becomes a finding.

**It detects. It attributes only with a ledger.** Running after the pipeline,
the audit sees inputs and outputs. Nothing in them distinguishes a record
removed by a filter from one lost to a swallowed exception. Set `ledger_glob`
and the producing stages' own accounting supplies the difference; leave it unset
and the report states plainly that it cannot attribute what it found.

## Attribution

Without `ledger_glob`, the audit can say 5,187,587 records are missing. Only a
producer-emitted ledger can say whether a language filter removed them or sixty
shards were lost. Every stage's ledger must reconcile:

```
n_input == n_success + n_filtered + n_failed + n_quarantined
```

The report then carries `filtered_by_reason` and, more importantly,
`unexplained` — records that left the pipeline for a reason nobody recorded.
Anything other than zero there is a finding, and no amount of reading the output
afterwards will recover the reason.

**Counting records cannot detect a lost shard.** The obvious gate, "fail if
`n_failed + n_quarantined > 0`", cannot see the failure it is written for: those
counts come from reading the shard, and a shard truncated by a killed job
reports zero rows. So a stage can lose a whole file and compute a loss of
exactly zero. The audit counts **units** instead, and every record figure in a
loss report is labelled a floor.

## Getting A Manifest

Completeness claims need `declared_manifest`. `curate/nemo_curator` writes one
when you set `emit_manifest`:

```yaml
# in the curate/nemo_curator config
emit_manifest: ./output/filtered_jsonl/run_manifest.json
metadata_fields: [id, source]
id_field: id
source_field: source
```

A manifest with no `completed_at` means the producing run never reached its
write barrier. The audit reports that as `manifest_incomplete` rather than
comparing counts against a partial run.

## Modes

| Mode | Adds |
|---|---|
| `integrity` | Per-shard readability and row counts. The default |
| `digest` | A content digest of the corpus |
| `containment` | Every target record is present in `reference_glob` |
| `all` | All of the above |

**On the digest.** It is independent of the order the filesystem enumerates
files in, and deliberately *not* independent of shard names — a digest blind to
names could not tell you which shard changed. It is also not independent of how
rows are distributed across shards; that stronger property is what
`containment` provides, at the cost of hashing every row.

**On `comparison_fields`.** Containment mode requires an explicit choice and
fails on an empty list. There is no "all common fields" default because the
pipeline adds columns of its own — `language` and `domain` among them — so
comparing everything the two corpora share would report differences that are
the pipeline working correctly. Use `[id]` when the corpus carries a stable
identifier.

## Run It

Smoke first, over the packaged fixture:

```bash
uv run nemotron steps run curate/audit -c tiny
```

Then against a real corpus:

```bash
uv run nemotron steps run curate/audit \
  target_glob='./output/filtered_jsonl/**/*.jsonl' \
  declared_manifest=./output/filtered_jsonl/run_manifest.json \
  output_dir=./output/audit
```

The step exits non-zero when it finds anything, so it can gate a pipeline.

## Repository Layout

- Manifest: [step.toml](step.toml)
- Runner: [step.py](step.py) delegating to `../scripts/run_audit.py`
- Measurement: `../runtime/integrity.py`
- Manifest schema shared with the producer: `../runtime/manifest.py`
- Configs: `config/default.yaml`, `config/tiny.yaml`
- Fixture: `data/tiny/`

## Guardrails

- Audit the corpus you are about to train on, not a copy — the digest is over
  the files you point it at.
- Run it before deleting the upstream corpus, while `reference_glob` can still
  resolve.
- A finding names the shard and the byte offset where parsing stopped. Re-run
  the producing stage for that partition rather than patching the file.
