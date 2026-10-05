# Holdout Near-Duplicate Decontamination

Use `curate/decontamination` to remove training documents that near-duplicate a
held-out split, so held-out scores measure generalization rather than recall.

Use this README for workflow and pitfalls; use `step.toml` for the exact
artifact, parameter, strategy, and error manifest before editing configs.

## What It Detects, And What It Cannot

**Threat model (A): whole-document near-duplicates.** A training document
largely the same as a holdout document.

**Not (B): substring contamination.** A 30-token benchmark question sitting
inside a 4,000-token training page. Whole-document Jaccard cannot see it — the
question moves the similarity by far less than any usable threshold. This is not
a tuning problem; it is what the method measures. Catching (B) needs a
containment-oriented algorithm and is a separate design.

The test suite pins this as a measurement rather than a caveat: there is a test
that *asserts the embedded question is missed*. If it ever starts passing, the
scope claim needs revisiting — it does not mean the step improved.

So every phrase here is **"near-duplicate overlap detected and removed"** and
never "holdout verified clean". The second is a claim this method cannot
support, and it is the overclaim most likely to be made.

## Three Passes, Cheapest First

| Pass | Needs GPU | Finds |
|---|---|---|
| Source identity | no | The same source document on both sides, at any similarity |
| MinHash / LSH candidates | yes | Pairs that *might* be near-duplicates |
| Exact Jaccard verification | no | Which candidates actually are |

**The first pass is not redundant.** A page re-crawled and rewritten enough that
its shingles no longer overlap is still the same source document and still must
not span the split. Identity precedence is canonical URL, then source-namespaced
id, then normalised-text hash. URL canonicalisation strips scheme, `www.`,
fragment and **tracking parameters** — so `http://a.example/x?utm_source=ads` and
`https://www.A.example/x/` are one group.

Non-tracking parameters are **kept**. Dropping the whole query string is the
tempting shortcut and it merges distinct pages on any query-driven site: measured
on Vietnamese C4, forum threads differing only by `?t=<id>` collapsed into one
group and pulled 29 unrelated documents out of a training split. `ref` is
deliberately not treated as tracking — on Git-hosting sites it selects a branch
or commit.

The URL field is looked up under several names — a corpus that calls it
`warc-target-uri` would otherwise fall through to content hashing, which is
exactly the case same-page leaks hide in.

**The third pass is not optional.** LSH buckets contain false positives by
construction; that is the trade it makes for speed. Removing a document because
it shared a bucket discards training data on the strength of a hash collision.

## Shingling Must Match The Candidate Generator

Curator's `FuzzyDeduplicationWorkflow` shingles on **characters**, 24-grams by
default. So does the verification here. Verifying at word 5-grams what was
proposed at char 24-grams computes a similarity over a different set than the
one MinHash approximated, and the threshold stops referring to the quantity the
candidates were selected by. `shingle_kind` and `minhash.char_ngrams` are both
recorded in the report.

Normalization is also recorded, for the same reason: two runs that normalized
differently did not measure the same thing, and nothing in the output would
otherwise show it.

Punctuation stripping is **category-based, not a regex**. Python's `\w` covers
letters and digits but not category `M`, so `[^\w\s]` strips Devanagari matras
and turns `भाषा` into `भ ष` — deleting the vowels and changing every shingle in
the document. The shared definition in `runtime/signals.py` accepts `L`, `N`,
all of `M`, and the two joiners obligatory inside Indic conjuncts.

## Only The Training Split Shrinks

Never the holdout. Removing a leaked document from a benchmark makes the
benchmark agree with the training data by changing the benchmark, which
invalidates every result measured against it — including results already
published from earlier runs. The step asserts this even though it never writes
the holdout, so a future edit that started to would fail loudly.

## Recall Is Measured, Not Asserted

LSH recall depends on the band and row structure *and* on the corpus, so any
number stated in documentation would be a guess.
`runtime/decon.candidate_recall` computes it by brute force on a sample and
names the pairs that were missed. It is quadratic on purpose — never call it on
a full corpus.

Candidate pairs that cannot be verified — text too short to shingle, or a
document not found — are reported as **unverifiable** and are not removed.
Counting them as similarity zero would report a clean result for a comparison
that never happened.

The split fingerprints in `decontamination_report.json` cover id membership.
They prove which named split was checked, but they are intentionally not the
profile/policy corpus fingerprint, which also binds document content. Do not
compare the two fields as if they used one digest contract.

## Run It

```bash
# Exact source-identity pass only. No GPU.
uv run nemotron steps run curate/decontamination -c tiny
```

Full run:

```bash
# Adds NeMo Curator's CUDA MinHash/LSH dependencies (RAPIDS).
uv sync --extra curate-gpu

uv run nemotron steps run curate/decontamination \
  train_glob='./output/filtered_jsonl/**/*.jsonl' \
  holdout_glob='./data/holdout/**/*.jsonl' \
  output_dir=./output/decontaminated \
  id_field=id threshold=0.8
```

This is the only step in `curate/` that declares `gpus_per_node = 1`. Set
`skip_similarity: true` to run the identity pass alone on CPU; the report then
says near-duplicate overlap was **not measured**, rather than reporting none.
For isolated Curator runtimes, select the `curate-gpu` profile; the ordinary
`curate` extra intentionally remains CPU-only for the other five steps.

## Repository Layout

- Manifest: [step.toml](step.toml)
- Runner: [step.py](step.py) delegating to `../scripts/run_decontamination.py`
- Similarity and verification: `../runtime/decon.py`
- Source identity: `../runtime/grouping.py`

## Guardrails

- Decontaminate against **every** holdout you will report on, in one run. A
  split cleaned against one benchmark says nothing about another.
- Quote `train_documents_removed`, not "the holdout is clean".
- Read `unverifiable_pairs` before trusting a low removal count. Zero removals
  and zero verifiable candidates are different results.
- Keep the threshold in config and record it beside the scores it protects. A
  number reported without the threshold it was decontaminated at cannot be
  compared with another lab's.
