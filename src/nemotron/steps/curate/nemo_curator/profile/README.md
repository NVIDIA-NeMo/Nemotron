# Corpus Filtering Impact Profile

Use `curate/profile` to find out what a filtering threshold would actually do to
*your* corpus before you apply it.

Use this README for workflow and pitfalls; use `step.toml` for the exact
artifact, parameter, strategy, and error manifest before editing configs.

## The Problem It Addresses

Today a user sets `min_words: 50` because that number was in the sample config.
Nobody knows whether it removes 8% of their corpus or 80%. Curator's heuristic
defaults were calibrated on English-language work, and its own non-English
preset handles the mismatch by deleting three filters rather than retuning them.

This step measures instead of guessing:

```
word_count                 min 50           retains 91.9%   corpus p5 = 41 words
non_alpha_numeric          threshold 0.25   retains  5.8%   corpus p50 = 0.46  <-
repeating_duplicate_ngrams threshold 0.2    retains 97.2%
```

The marked line is the one nobody gets today. Right now you see a suspiciously
small output and cannot tell which gate caused it — the `empty_or_tiny_output`
failure the category documents but cannot diagnose.

## What It Does Not Establish

**A retention curve is descriptive.** It answers *how much does this threshold
remove*. It does not answer *is what it removes bad*. A corpus can have a small
but valuable tail, or a large body of spam, and a distribution cannot tell them
apart.

So the report's vocabulary is deliberate — *candidate* threshold, not correct;
*retention-stable* band, not feasible. And the output is not executable:
`candidate_policies.yaml` carries `approved: false`. Promotion to an
`approved_policy.yaml` is a separate act that records who approved it and on
what evidence.

`min_keep_rate` and `max_keep_rate` are **analysis constraints you chose**, not
properties discovered in the data. They bound which part of a curve gets
reported.

## Two Views Of Every Figure

Sampling the same number of documents from a large source and a small one is
unbiased within each source and skewed at corpus level. Both views are reported
and both are labelled:

| View | Weighting | Answers |
|---|---|---|
| `macro` | each source equally | is this threshold reasonable for each source? |
| `micro` | each sampled document by how many it stands for | how much of the corpus does it remove? |

Any figure that did not say which view it came from could not be acted on.

## Language Packs

`language` is a BCP-47 tag and has **no default**. A wrong default produces
plausible numbers for the wrong language, which is worse than refusing to start.

A pack is data — word lists, a character set, boilerplate patterns, a fold map —
plus a declaration of what can meaningfully be measured for that language. See
[../data/langpacks/SPEC.md](../data/langpacks/SPEC.md) for the file format, and
[../data/langpacks/ADDING_A_LANGUAGE.md](../data/langpacks/ADDING_A_LANGUAGE.md)
for the procedure around it.

Nemotron ships three **opt-in example packs** under `../data/langpacks/` —
`en/`, `vi/` and `hi/`. They supply score inputs, not filtering thresholds, and
neither the language nor its directory is selected by default. Run `-c en` to
choose the English one explicitly. For another language, set `langpack_dir` to
a reviewed pack root owned by your corpus workflow.

The `x-test-*` packs under `tests/steps/curate/fixtures/langpacks/` validate the
implementation but are not installed, user defaults, or claims of supported
languages.

The capability declaration does real work. Vietnamese tone marks strip to
degraded but readable text, so a diacritic ratio measures something. Devanagari
matras are obligatory vowels; stripping them yields nonsense, so the `hi` pack
declares neither `diacritic_ratio` nor `stopword_ratio_folded` and both are
**absent from the Hindi report**, with a recorded warning, rather than computed
on a false premise.

Naming a signal the pack cannot support is an error. Leaving `signals` empty
skips it with a warning — the difference is that a named signal is a question
the caller asked.

## Signals

A closed allowlist in `../runtime/registry.py`. Config names a signal, never an
import path.

The allowlist has 24 entries in two groups. Which group a signal is in decides
where to read about what it measures.

### Language dependence

Which of the three groups a signal is in decides what it costs to use on a new
corpus, and whether its number means the same thing there as it did here.

**Language-agnostic (6).** The rule is defined over Unicode or over characters
that mean the same thing in every script, so the measurement transfers and there
is nothing to supply.

Transferring is not the same as being comparable. `white_space` is well defined
everywhere and its ordinary range still differs threefold between a language that
separates words with spaces and one that does not — 0.186 median across en, vi
and hi against 0.059 across th and ja. Agnostic means the number is meaningful on
any corpus, not that a threshold chosen on one corpus belongs on another. That is
what the profile is for.

| signal | bound | units |
|---|---|---|
| `bullet_ratio` | `max` | ratio |
| `ellipsis` | `max` | ratio |
| `parentheses_ratio` | `max` | ratio |
| `unicode_alpha_numeric` | `max` | ratio |
| `urls_ratio` | `max` | ratio |
| `white_space` | `max` | ratio |

**Language-dependent, and it says so (18).** These declare a
requirement and are skipped with a note when the pack does not meet it, so a
corpus never gets a number the pack could not support.

| signal | bound | units | needs |
|---|---|---|---|
| `boilerplate_hits` | `max` | patterns | `boilerplate_hits` |
| `diacritic_ratio` | `min` | ratio | `diacritic_ratio` |
| `foreign_script_ratio` | `max` | ratio | `script_ratio` |
| `latin_ratio` | `max` | ratio | `script_ratio` |
| `script_ratio` | `min` | ratio | `script_ratio` |
| `sentence_end_ratio` | `min` | ratio | `sentence_end_ratio` |
| `stopword_ratio` | `min` | ratio | `stopword_ratio` |
| `stopword_ratio_folded` | `min` | ratio | `stopword_ratio_folded` |
| `max_word_length` | `max` | characters | `word_segmentation` |
| `mean_word_length` | `interval` | characters | `word_segmentation` |
| `non_alpha_numeric` | `max` | ratio | `ascii_alphabet` |
| `numbers_ratio` | `max` | ratio | `ascii_digits` |
| `punctuation` | `max` | ratio | `ascii_punctuation` |
| `repeating_duplicate_ngrams` | `max` | ratio | `word_segmentation` |
| `symbol_to_word` | `max` | ratio | `word_segmentation` |
| `token_count` | `interval` | tokens | `tokenizer` |
| `word_count` | `interval` | words | `word_segmentation` |
| `words_with_alphabets` | `min` | ratio | `word_segmentation` |

**Language-dependent without declaring it (0).** Empty, and that is the
point of the column above. Every signal whose measurement assumes a writing
system now says which assumption it makes, so a pack that does not hold it
skips the signal with a named warning and a config that asks for it by name
is refused.

`non_alpha_numeric` was the last entry here. Curator counts only
`[a-zA-Z0-9\n?!,.]` as content, so on any other alphabet correct text scores
as junk: one ordinary Vietnamese sentence scores 0.214 through
`unicode_alpha_numeric` and 0.429 through this one, and 0.429 is rejected at
the shipped 0.25 default. Having a correct replacement was not enough --
profile still offered the broken signal for every language and a policy could
still promote it. It now requires `ascii_alphabet`, which only English
declares.

The last group is where a threshold silently stops meaning what it meant.
Measured on 20,000 C4 documents per language:

| | en | vi | hi | th | ja |
|---|---:|---:|---:|---:|---:|
| `word_count` p50 | 191 | 405 | 316 | **86** | **72** |
| `mean_word_length` p50 | 4.96 | 3.63 | 4.36 | **11.96** | **19.25** |
| `max_word_length` p50 | 14 | 12 | 16 | **62** | **157** |

Thai and Japanese do not separate words with spaces, so a whole clause becomes one
"word". A `mean_word_length` p95 of 121 characters is not a long word; it is the
profile telling you word-based gates do not apply to this corpus. The shipped
`min_words: 50` removes 39% of the Japanese corpus and 33% of the Thai one, and
because `word_count` is not part of a policy it does not appear in the policy
simulation — it is charged before that section is reached.

`non_alpha_numeric` and `punctuation` embed a different assumption: an ASCII
alphabet and Latin sentence terminators. Curator's own non-English cascade drops
the first for that reason. `unicode_alpha_numeric` and `sentence_end_ratio` are the
script-aware replacements — at Curator's own 0.25 default, `non_alpha_numeric`
retains 99.66% of English and 0.11% of Hindi, where `unicode_alpha_numeric` retains
96.53%. `punctuation` scores 1.000 on Hindi and Japanese text that ends every
sentence correctly, because it looks for `.`, `!` and `?` and Hindi ends with `।`.
### Wrapping a Curator filter (15)

These score exactly what Curator's filter scores; this step adds a swept grid, a
verified direction and a retention curve. What each one measures is documented in
[NeMo Curator's heuristic quality assessment](https://docs.nvidia.com/nemo/curator/curate-text/process-data/quality-assessment/heuristic).

| signal | bound | units | needs |
|---|---|---|---|
| `bullet_ratio` | `max` | ratio | — |
| `ellipsis` | `max` | ratio | — |
| `max_word_length` | `max` | characters | `word_segmentation` |
| `mean_word_length` | `interval` | characters | `word_segmentation` |
| `non_alpha_numeric` | `max` | ratio | — |
| `numbers_ratio` | `max` | ratio | `ascii_digits` |
| `parentheses_ratio` | `max` | ratio | — |
| `punctuation` | `max` | ratio | `ascii_punctuation` |
| `repeating_duplicate_ngrams` | `max` | ratio | `word_segmentation` |
| `symbol_to_word` | `max` | ratio | `word_segmentation` |
| `token_count` | `interval` | tokens | `tokenizer` |
| `urls_ratio` | `max` | ratio | — |
| `white_space` | `max` | ratio | — |
| `word_count` | `interval` | words | `word_segmentation` |
| `words_with_alphabets` | `min` | ratio | `word_segmentation` |

### Implemented here (9)

Curator has no equivalent, or has one that cannot be used outside English. The
`replaces` column names the filter each one stands in for, where there is one.

| signal | bound | units | needs | replaces | what it measures |
|---|---|---|---|---|---|
| `boilerplate_hits` | `max` | patterns | `boilerplate_hits` | `BoilerPlateStringFilter` | Counts matches of the pack's boilerplate patterns. Curator's version hardcodes English cookie and privacy phrases, so on any other language it matches nothing and reports nothing. |
| `diacritic_ratio` | `min` | ratio | `diacritic_ratio` | — | Share of letters carrying a mark this language treats as removable. Only declared where marks are removable at all: Vietnamese tone marks strip to degraded but readable text, Devanagari matras are obligatory vowels. |
| `foreign_script_ratio` | `max` | ratio | `script_ratio` | — | Share of letters from neither this language's script nor base Latin. |
| `latin_ratio` | `max` | ratio | `script_ratio` | — | Share of letters that are plain unmarked Latin. High values on a non-Latin corpus mean untranslated boilerplate, code, or the wrong language entirely. |
| `script_ratio` | `min` | ratio | `script_ratio` | `HistogramFilter` | Share of letters in the pack's own script. Continuous, unlike Curator's, which thresholds internally and returns 0 or 1 — a binary value cannot be swept. |
| `sentence_end_ratio` | `min` | ratio | `sentence_end_ratio` | `PunctuationFilter` | Share of sentence-like spans ending in a terminator this language actually uses. Curator's looks for `.`, `!` and `?`; Hindi ends sentences with `।`. |
| `stopword_ratio` | `min` | ratio | `stopword_ratio` | `CommonEnglishWordsFilter` | Function-word density from the pack's list. Prose has a characteristic density; keyword lists and navigation furniture do not. |
| `stopword_ratio_folded` | `min` | ratio | `stopword_ratio_folded` | `CommonEnglishWordsFilter` | The same density measured after removing marks from both sides, so diacritic-stripped prose is still recognised as prose. |
| `unicode_alpha_numeric` | `max` | ratio | — | `NonAlphaNumericFilter` | Counts Unicode categories L, N and all of M instead of `[a-zA-Z0-9\n?!,.]`. At Curator's own 0.25 default the ASCII version retains 99.66% of English and 0.11% of Hindi. |

`needs` names a language-pack capability, or a tokenizer. A signal whose
requirement the pack does not declare is skipped with a note rather than scored on
data it cannot read. Fifteen need nothing and run on any corpus.

Eleven Curator filters are deliberately absent, and naming one in a policy is an
error rather than a silent skip: English-hardcoded word lists, binary scores that
cannot be swept, and four repetition filters whose parameter names and
`keep_document` comparisons disagree. Each reason is recorded in
`registry.EXCLUDED`, so the Curator page above will describe filters this step
will not run.

Curator's filters do not share one shape — some are upper bounds, some lower,
and `word_count` and `mean_word_length` gate from both sides at once. A
two-sided gate produces a **retention surface**, not a curve: the retention of a
lower bound depends on where the upper bound sits, so reporting a single line
would fix one bound at an unstated value and attribute all of the effect to the
other.

Several Curator filters are deliberately excluded, with the reason recorded in
`registry.EXCLUDED` — English-hardcoded word lists, binary scores that cannot be
swept, and four repetition filters whose parameter names and `keep_document`
comparisons disagree.

**The Unicode signal restores a gate, it does not fix a broken default.**
Curator's `heuristic_filter_non_english_pipeline.yaml` already omits
`NonAlphaNumericFilter`, `CommonEnglishWordsFilter` and
`WordsWithoutAlphabetsFilter` — the three its English cascade uses and that assume
ASCII. That omission is correct, and it leaves a non-English corpus with no
character-composition gate and no vocabulary gate. `unicode_alpha_numeric` and
`stopword_ratio` are the script-aware replacements for the first two.

**And the Unicode handling is not a Vietnamese fix.** `UnicodeAwareNonAlphaNumericFilter`
accepts Unicode categories L, N and *all* of M. Writing it as `\p{Mn}` — the
natural thing to reach for with only Vietnamese to test against — passes every
Vietnamese case and rejects correct Devanagari outright, because matras split
across `Mn` and `Mc` with `Mc` in the majority. It also accepts ZWJ and ZWNJ,
which are category `Cf` and obligatory inside Indic conjuncts; treating them as
junk moved a correct Hindi paragraph from 0.218 to 0.265, across the 0.25
default.

**Direction is verified, not trusted.** The sweep compares scores to thresholds
directly rather than constructing a filter at every grid point. That shortcut is
only sound if the registry's stated direction matches the filter's own
`keep_document`, so it is checked against the real implementation on real scores
before any figure is produced. A mismatch stops the run.

## Run It

Run the packaged English smoke profile with no model download:

```bash
uv run nemotron steps run curate/profile -c en
```

For your own corpus, profile the *unfiltered* input and supply the pack
explicitly:

```bash
uv run nemotron steps run curate/profile \
  input_glob='./output/raw_jsonl/**/*.jsonl' \
  output_dir=./output/profile \
  source_field=source id_field=id \
  language=vi langpack_dir=./langpacks
```

Four files land in `output_dir`:

| File | Contents |
|---|---|
| `profile_summary.md` | **Read this one.** The same measurements rendered for a person |
| `profile_report.json` | Quantiles, retention curves and surfaces, co-occurrence, policy simulation, what each Curator default would keep |
| `candidate_policies.yaml` | Proposed threshold sets, `approved: false` |
| `sample_manifest.json` | Seed, per-source `(population, sampled, weight)`, and the sampled key hashes |

## Reading The Summary

`profile_summary.md` has four parts, in the order you need them.

**Language composition.** What languages the corpus is actually in, scored with
the same FastText model the filter uses. No signal below it can answer this:
languages that share a script are indistinguishable to a script ratio. The gap is
not academic — naming the wrong language in `steps.filter.language_codes` removes
almost everything and still reports success, because the row counts reconcile
either way. Confidence is reported as its own distribution, because which
languages to keep and how sure the model must be are two separate decisions.
Absent a model the section is omitted rather than guessed.

**One section per signal.** Quantiles, a sparkline, then a `gate at` table:

```text
| to keep | set `max:` | actually keeps | drops |
| ~99% | 0.9365 | 99.03% | 193 docs |
| ~95% | 0.8730 | 95.97% | 807 docs |
| ~90% | 0.8254 | 91.46% | 1,708 docs |
| ~80% | 0.7778 | 80.32% | 3,936 docs |
```

Every row is **this gate applied alone**. The 80% row exists because a
concentrated signal makes 99/95/90 resolve to the same grid point and the table
then repeats one number three times; it is the floor because
`band_search.min_keep_rate` defaults there.

Which side of a signal is desirable is deliberately not stated. It does not
generalise: a ratio that marks noise in one language is ordinary in another.
`max` means an upper bound and documents above it are dropped, `min` a lower
bound, `interval` a two-sided pair — and which key a signal takes comes from the
registry, not from you.

**Policy simulation.** What all the gates cost *together*. A policy is a
conjunction, so its cost is the union of the rejects, and the union is smaller
than the sum by however many documents more than one gate rejects — on a
20,000-document Vietnamese sample the per-gate figures summed to 2,718 while the
union was 1,808. Each gate also reports what it is alone responsible for:
`foreign_script_ratio` rejected 194 documents there and not one of them
uniquely, so deleting that gate would have changed nothing.

Order-dependent and order-independent figures are separated, because only one of
them is a property of the policy. `fails alone`, `shared`, `unique` and
`keeps without` do not move when stages are reordered; `first to reject` does.
That last one is the shape `curation_ledger.json` records, which is why its
per-reason counts understate what a gate rejects.

**The approve block.** Paste-ready YAML with each threshold's retention on the
same line, so the two cannot be mis-paired. Take thresholds from here rather
than from `candidate_policies.yaml`: a band's `threshold_low` and
`threshold_high` are two *alternative* single bounds, not a range, and choosing
the wrong end is accepted without a warning. The block covers one-sided signals
only — `word_count`, `mean_word_length` and `token_count` gate from both sides
and are written by hand with both bounds.

The sample is reproducible from `(seed, max_total_docs)` — hash-bottom-k, never
Python's salted `hash()`.

## Adding A Signal

The allowlist above is closed: a config names a registered signal and never an
import path. To extend it -- a custom domain filter, or any measurement a
`DocumentFilter` can express -- see
[../ADDING_A_SIGNAL.md](../ADDING_A_SIGNAL.md).

## After The Thresholds Are Approved

This step measures distributions; it does not and cannot say a threshold is
correct. Once a person has approved one, check it against labelled documents:

```bash
uv run --extra curate python -m nemotron.steps.curate.nemo_curator.scripts.run_evaluate \
  --policy <output_root>/policy/approved_policy.yaml \
  --labelled <labelled.jsonl> --langpack-dir <pack root>
```

It reports false-rejection and noise-removal rates per failure mode and per
signal. See [../../README.md](../../README.md) § Checking A Policy Against
Labelled Documents.

## Repository Layout

- Manifest: [step.toml](step.toml)
- Runner: [step.py](step.py) delegating to `../scripts/run_profile.py`
- Signals: `../runtime/registry.py`
- Measurement: `../runtime/profiling.py`
- Sampling: `../runtime/determinism.py`
- Policy schema: `../runtime/policy.py`

## Guardrails

- Profile the input you want to filter, not the already-filtered output — the
  gates you are trying to understand have already run on the latter.
- `source_field` matters. Without it each shard becomes its own "source" and the
  per-source figures describe shards; the report says so, but the numbers are
  easy to misread.
- Do not quote a co-occurrence figure without its operating point. Overlap is
  only defined at a specific threshold per signal, and every entry in the report
  carries the thresholds it was computed at.
