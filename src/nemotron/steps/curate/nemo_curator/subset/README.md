# Nested Token-Budget Subsets

Use `curate/subset` when you want to compare two curation policies and know that
what you measured was the *policy*.

Use this README for workflow and pitfalls; use `step.toml` for the exact
artifact, parameter, strategy, and error manifest before editing configs.

**Run it after `curate/decontamination`, not before.** A tier is written out and
read as a corpus in its own right, so a tier cut before decontamination carries
exactly the documents decontamination was asked to remove — and nothing says so
afterwards: the subset report counts documents, not their provenance, and the
decontamination report describes a corpus the tiers were not taken from. The flow
orders the two steps and points `input_glob` at `train_decontaminated.jsonl`
automatically. Running this step by hand is the case to watch: point `input_glob`
at the decontaminated corpus yourself.

## The Problem It Addresses

Policy A retains 80% of a corpus. Policy B retains 95%. Train on whatever each
one leaves and the result measures filter quality and dataset size at the same
time, with no way to separate them afterwards. Fixing the token budget is what
makes the comparison mean anything.

## The Guarantee, And The One It Rules Out

Two properties are wanted from budgeted subsets and **cannot both hold**.
Documents of 4, 3 and 2 tokens:

```
budget 4:  fullest selection = {4}        → 4 tokens
budget 5:  fullest selection = {3,2}      → 5 tokens
{4} ⊄ {3,2}                               → nesting violated
```

So this step chooses:

> **Nesting is guaranteed. Filling the budget is not.**

Tokens delivered are at most the budget. The difference is reported per tier as
`token_shortfall` and per stratum as `per_stratum_deviation`, and is **never**
made up by taking more from another stratum — moving tokens between strata to
hit a number would change the composition, which is the thing a stratified
subset exists to hold fixed.

Note also that "shortfall is bounded by the largest document" is false once
there is more than one stratum: each stratum stops independently, so shortfall
accumulates.

## How Nesting Is Achieved

Three mechanisms, all of them structural rather than checked-after-the-fact:

1. **One plan, one run.** Every tier comes from a single plan over a single
   scan. Tiers produced by separate runs cannot be shown to nest, because the
   strata themselves depend on the corpus each run saw.
2. **Prefix selection.** Within a stratum there is one total ordering —
   `stable_uint64(seed, doc_id)`, ties broken by `doc_id` — and each tier takes
   a *prefix* of it. A prefix is contained in every longer prefix, so the
   guarantee does not depend on which budgets were requested.
3. **House-monotone quotas.** Per-stratum budgets use Jefferson's divisor
   method, under which raising the budget can only *add* units to a stratum.
   The obvious alternative, largest remainder, admits the Alabama paradox: a
   stratum can lose a unit when the budget grows, which is exactly a document
   leaving a larger tier. `runtime/subset.py` keeps `largest_remainder` solely
   so the test suite can demonstrate that failure against the method in use.

The ordering is deliberately *not* a sort on a score column.
`sort_values(score)` on a float column with many ties is not a total order, so
membership at the cut boundary becomes run-dependent — nesting breaks and
nothing fails.

## Stratification

The stratum key is `source × length band`, plus `score decile` when
`quality_score_field` is set.

Length is in the key because a subset with the right source mix and the wrong
length mix is not neutral — short and long documents behave differently under
nearly every filter. Score deciles come from `curate/nemo_curator` run with
`mode: annotate` or `mode: both`.

A `quality_score_field` that is set but absent from the data is an **error**,
not a fallback to coarser strata. Silently stratifying on less than was asked
for changes what the subset represents.

Watch for the opposite failure: finer stratification with a small budget gives
each stratum less than its shortest document, and the tier comes back nearly
empty. The run warns and names the starved strata rather than leaving you to
work out why a tier is small.

## Tokenizer

`tokenizer.revision` is **required**. `TokenCountFilter` has no `revision`
parameter but forwards `transformers_init_kwargs` verbatim to
`AutoTokenizer.from_pretrained`, which is where the pin lands. The resolved name
and revision are recorded in `plan.json` and `subset_report.json`.

Two subsets counted under different revisions are not comparable and must not be
presented as an ablation pair.

Set `tokenizer: null` to count whitespace words instead — useful for smoke runs
and for corpora you do not want to tokenize twice. Every artifact then records
`unit: words`, so the two can never be confused for one another.

Counts are cached keyed by `(tokenizer, revision, id, content SHA-256)`. The
content digest prevents edited or re-filtered text from reusing a stale count
merely because its id stayed the same. A cache written under a different
tokenizer, revision, or schema is ignored with a warning, never reused.

## Relationship To CLIMB

Curator ships `tutorials/text/nemotron-climb-data-curation/`, and this repo has
`src/nemotron/data_prep/blend.py`. They solve a different problem:

| | CLIMB | `curate/subset` |
|---|---|---|
| Unit | tokenized `.bin` files | documents |
| Chooses | proportions **between** groups | which documents **within** a group |
| Output | blend weight files | a materialized corpus |
| Quality step | drops whole low-scoring clusters | per-document gates applied upstream |
| Nesting | none | guaranteed by construction |

They compose: `subset` produces the corpora that CLIMB then mixes.

## Run It

```bash
uv run nemotron steps run curate/subset -c tiny
```

Against a real corpus:

```bash
uv run nemotron steps run curate/subset \
  input_glob='./output/filtered_jsonl/**/*.jsonl' \
  output_dir=./output/subset \
  id_field=id source_field=source
```

Output:

| Path | Contents |
|---|---|
| `plan.json` | Per-stratum quota for every tier, written **before** anything is materialized |
| `budget_<N>_<unit>/subset.jsonl` | The tier itself; records pass through unchanged |
| `subset_report.json` | Per tier: `achieved_tokens`, `token_shortfall`, `per_stratum_deviation`, `documents_refilled`, `strata_exhausted` |

## Repository Layout

- Manifest: [step.toml](step.toml)
- Runner: [step.py](step.py) delegating to `../scripts/run_subset.py`
- Algorithm: `../runtime/subset.py`
- Ordering: `../runtime/determinism.py`

## Guardrails

- **Ask for every tier you will ever want, in the first run.** Adding a budget
  later means re-running with the original budgets, the same seed and the same
  tokenizer revision. A new budget run on its own may produce a corpus that does
  not contain the tiers you already trained on.
- Subset the corpus you will train on, after filtering — not before. Subsetting
  first and filtering after leaves each tier a different size again, which is
  the confound the step exists to remove.
- Quote `achieved_tokens`, not the budget. They are not the same number and the
  report keeps them separate on purpose.
- Every usable document must have a positive token count, and every configured
  quality score must be finite. Zero/negative costs and NaN/infinite scores are
  refused because they make budget and decile semantics undefined.
- Budgets and length-band edges are positive integers; bands must be strictly
  increasing. Values are validated rather than truncated with `int()`.
- If materialization cannot write every id selected by the plan, the run removes
  all tiers and the success report instead of publishing a plan/output mismatch.
