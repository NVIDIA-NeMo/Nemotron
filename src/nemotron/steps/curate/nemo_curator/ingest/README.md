# Corpus Ingestion

Use `curate/ingest` to turn a raw corpus into one the rest of the category can
read — without converting files or minting identifiers by hand.

Use this README for workflow and pitfalls; use `step.toml` for the exact
artifact, parameter, strategy, and error manifest before editing configs.

## What It Closes

Two things stood between "here is my data" and running the flow, and both were
being pushed onto the user:

**Format.** Corpora arrive as parquet at least as often as JSONL. The reading
here is thin on purpose: Curator's `ParquetReaderStage.read_data` would serve,
and calling it does not require a cluster. Two narrow differences are why it is
not called — it reads a whole file (`pd.read_parquet` per path, then `concat`)
where this streams row-group by row-group, and `pd.read_json(lines=True)` raises
on the first malformed line where this counts it and carries on. A corpus with
forty bad lines in ten million should be describable, not fatal.

**Identity.** `curate/subset` and `curate/decontamination` are statements about
*sets of document ids*, and most web corpora carry none. This is the part Curator
really does not cover. Its readers can generate ids, but `_generate_ids_func`
assigns `np.arange(min_id, min_id + num_rows)` from a Ray actor: positional, so
resharding renames every document and any claim made about the old ids silently
becomes false — and cluster-bound, so ids cannot be minted before one exists.

So an id is minted as a full SHA-256 over a canonical JSON array of configured
identity-field values, which must include the document text. JSON framing keeps
field boundaries unambiguous even when a value contains newlines. Reshard,
reorder, re-split: the id does not move, and the digest is not truncated into a
collision-prone 64-bit namespace.

## The Consequence This Step Will Not Hide

Two documents with the same configured identity fields mint the **same id**.
Because text must be one of those fields, byte-identical text is the common
case. This is not hypothetical — measured on Sangraha's Hindi `verified`
split:

```
20,000 documents
    8 groups of byte-identical text
  328 redundant documents (1.64%)
  293 copies in the largest group
    0 duplicate doc_ids            ← the corpus's own ids are all unique
```

That last line is the interesting one: the corpus assigns *distinct* ids to
identical documents, so an id-based deduplication finds nothing at all. Only
content comparison sees it.

What to do about it has three defensible answers, so `on_duplicate` has no safe
default and the run stops:

| `on_duplicate` | |
|---|---|
| `refuse` *(default)* | stop, report how many documents and how many groups |
| `drop` | keep the first of each group |
| `suffix` | keep every copy under a distinguishable id |

`drop` and `suffix` both change what the corpus *is*. Neither happens silently.
When the corpus provides ids, the same choices apply to duplicate ids, but the
error does not mislabel two different texts under one id as byte-identical.

## Two Real Corpora

```yaml
# C4-vi: parquet, no id, one source
input: ./raw/*.parquet
id_from: null                 # mint
id_fields: [url, text]
id_prefix: "c4vi-"
source: c4_vi
keep_fields: [url, timestamp]
```

```yaml
# Sangraha verified/hin: parquet, has doc_id, provenance in `type`
input: ./raw/*.parquet
id_from: doc_id               # use the corpus's own
source_from: type             # web / pdf / speech
keep_fields: []
```

The second matters more than it looks. `type` becomes `source`, and per-source
figures are how the OCR and ASR portions of a corpus become visible — measured
there, a `sentence_end_ratio >= 0.8` gate keeps 86.8% of `web` but only 15.9% of
`pdf` and 7.7% of `speech`, while the corpus figure is a reassuring 78.4%.

## What It Does Not Do

**It does not rewrite text.** Normalisation that changes content is a filtering
decision somebody approves, not part of reading a file. This step selects
fields, mints an identifier, and writes JSONL.

Columns not named in `keep_fields` are dropped, so a later step cannot come to
depend on a column nobody asked for. `ingest_report.json` lists
`columns_available`, so nothing is lost silently.

## Run It

```bash
uv run nemotron steps run curate/ingest -c tiny

uv run nemotron steps run curate/ingest \
  input='./raw/*.parquet' output_dir=./output/ingested \
  id_prefix=c4vi- source=c4_vi
```

Inside the six-step flow it runs first, and downstream steps read its output
automatically — set `steps.ingest.enabled: true` and point `corpus.input` at the
raw files.

## Output

```
output/ingested/
├── part_0.jsonl          50,000 documents per shard
├── part_1.jsonl
└── ingest_report.json
```

`ingest_report.json` records how each id was derived — `recipe`, `fields`,
`prefix` — because an id whose derivation nobody can reproduce cannot be
regenerated when the corpus is re-ingested.

## Repository Layout

- Manifest: [step.toml](step.toml)
- Runner: [step.py](step.py) delegating to `../scripts/run_ingest.py`
- Format handling and id minting: `../runtime/ingest.py`

## Guardrails

- Read `counts` in `ingest_report.json` before trusting the output size.
  Missing text/id, non-object JSON values, and unparsable lines are counted,
  not hidden.
- A refused or failed run leaves neither `part_*.jsonl` nor a success report;
  shards are staged and `ingest_report.json` is committed last.
- Set `id_prefix` when the corpus will later be mixed with another. Two corpora
  minting from the same fields can otherwise collide by construction.
- Re-ingesting under a different `id_fields` produces different ids for the same
  documents. Any policy, subset, or decontamination report keyed on the old ones
  describes a corpus that no longer exists.
