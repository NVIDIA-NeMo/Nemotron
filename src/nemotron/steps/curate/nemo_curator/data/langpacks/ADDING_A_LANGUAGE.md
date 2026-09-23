# Adding A Language

The ordered procedure for bringing a new language into `curate`, end to end:
from an empty directory to a filtered corpus you can defend.

`SPEC.md` is the format reference — what a file must contain, and why a
capability is declined. Read it while writing files. This document is the
procedure: it says what to do, in what order, and what to check before moving
on. Steps 1–3 delegate to `SPEC.md` rather than restating it.

Onboarding a language is a human process with measurements in the middle. Two
of the steps need a person and are marked; the rest are commands. Step 8 is both:
labelling documents is human work, and once labelled, the rates are computed by
`curate/evaluate` rather than by hand.


| #   | Step                                   | Needs a person?            |
| --- | -------------------------------------- | -------------------------- |
| 1   | Create the pack directory              | no                         |
| 2   | Decide required vs optional files      | no                         |
| 3   | Write `charset.txt` and the word lists | yes — sourcing             |
| 4   | Set up FastText language ID            | no                         |
| 5   | Validate the pack loads                | no                         |
| 6   | Profile a real corpus                  | no                         |
| 7   | Approve thresholds                     | **yes — this is the gate** |
| 8   | Evaluate the result                    | **yes — read documents**   |
| 9   | Activate it in a config                | no                         |


---



## 1. Create the pack directory

The directory name **is** the tag, and the tag inside `pack.toml` must match it.
Loading refuses a pack whose `pack.toml` names a different language, because a
pack filed under the wrong name puts one language's word lists behind another
language's profile.

```
./langpacks/<bcp47-tag>/
```

Use a BCP-47 tag (`vi`, `hi`, `pt-BR`), not an ISO 639-1 code. Private-use tags
(`x-…`) are valid — the test fixtures use them.

The tag is a single directory name, not a path: `../other` and absolute paths
are refused.

See `SPEC.md` § Layout.

## 2. Decide required vs optional files


| File              | Required?                                            | Needed for         |
| ----------------- | ---------------------------------------------------- | ------------------ |
| `pack.toml`       | **required**                                         | always             |
| `charset.txt`     | when you declare `script_ratio` or `diacritic_ratio` | script signals     |
| `stopwords.txt`   | when you declare `stopword_ratio`                    | stopword signals   |
| `boilerplate.txt` | optional                                             | `boilerplate_hits` |
| `LICENSES/`       | when a list is derived from a licensed source        | provenance         |
| `SOURCES.md`      | recommended                                          | provenance         |


A capability declared without the file behind it is refused at load time — the
alternative is a report full of zeroes, which reads as a finding rather than a
missing file. The shipped `en/` pack ships no `boilerplate.txt` and declares no
capability that needs one.

## 3. Write the character set and word lists

The trap that costs the most time is in `charset.txt`: it is parsed as the union
of every character on every line, and **ranges are not expanded**. A line
reading `a-z` gives you `a`, `-` and `z`.

The second trap is script-specific and is the reason this step needs a person:
for Devanagari, a fix that handles only `Mn` (non-spacing marks) passes every
Vietnamese test and still destroys Hindi, because Hindi matras span `Mc`
(spacing combining) as well and `Mc` is the majority. `SPEC.md` § Character
classification covers this in full. Read it before sourcing a character set.

Record provenance in `SOURCES.md` and put licence texts in `LICENSES/`.

## 4. Set up FastText language ID

This is a separate decision from the language pack, and the two are configured
independently.

**The model.** `language_codes` needs a FastText language-ID model. The one this
repo already uses elsewhere is `lid.176.bin`:

```
https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

That URL is the same one `src/nemotron/recipes/data/curation/nemotron-cc/step_1-download_extract.py`
downloads. Record the checksum of the file you actually downloaded alongside it;
this document deliberately does not pin one, because the artefact is fetched
from an upstream that does not publish a stable digest per release.

Point the config at wherever you put it:

```yaml
models:
  fasttext_langid: ./models/lid.176.bin
```

**Confirm your language is covered.** `lid.176` predicts 176 languages. If yours
is not among them, `language_codes` cannot gate on it and you must leave that
knob empty and rely on `script_ratio` from your pack instead.

**Two language settings, not one.** They are matched independently and it is
worth being explicit about which is which:


| Setting                       | Namespace                        | Decides                      |
| ----------------------------- | -------------------------------- | ---------------------------- |
| `corpus.language`             | BCP-47, matches a pack directory | which **pack** is loaded     |
| `steps.filter.language_codes` | `lid.176` labels, upper-cased    | which documents are **kept** |


The comparison is case-folded, and a label carrying a script suffix matches on
the part before the underscore, so `ZH` selects `zh_Hans`. Nothing currently
checks that these two agree — setting `language: vi` with
`language_codes: [JA]` produces a corpus scored by Vietnamese rules and filtered
to Japanese documents, and every row count still reconciles. Set them to the
same language unless you specifically mean not to.

## 5. Validate that the pack loads

Before profiling anything, confirm the pack is well formed:

```bash
uv run --extra curate python -c "
from nemotron.steps.curate.nemo_curator.runtime import langpack
pack = langpack.load('<your-tag>', './langpacks')
print(pack.describe())
"
```

`describe()` reports the tag, version, content hash, and the counts behind every
declared capability. Check that the counts are what you expect — a `charset` of
3 where you meant 26 is the range trap from step 3.

Loading refuses, with the reason named, when: the tag and `pack.toml` disagree,
a declared capability has no file behind it, a capability name is misspelled, or
the schema version is wrong.

## 6. Profile a real corpus

Measure before gating. `curate/profile` reports the distribution of every signal
your pack supports, plus what each candidate threshold would retain.

```bash
uv run --extra curate python -m nemotron.steps.curate.nemo_curator.scripts.run_profile \
  --config <your-profile-config>.yaml
```

Read `profile_summary.md`. It is a distribution summary and deliberately does
not nominate a best threshold — the right operating point depends on your token
budget, and 15B and 50B tokens out of the same corpus do not land in the same
place.

See `../../profile/README.md` for what each artifact contains.

## 7. Approve thresholds

**This is the gate, and it needs a person.** `curate/profile` writes
`candidate_policies.yaml` with `approved: false`. Nothing executes it until a
human fills in the approval block with an approver and the evidence they looked
at, and promotes it.

An empty approval block is not an approval. The manifest records which of
`approved`, `override_unvalidated` or `unapproved` a run actually had, and a
corpus produced under an override says so.

See `../../README.md` § Applying A Profiled Policy.

## 8. Evaluate the result

Retention says how much a gate removed. It cannot say whether removing it was
right, and the two questions fail in opposite directions: a threshold keeping 93%
may be discarding exactly the 7% that was good, and one keeping 99% may be
leaving the noise in. Two rates, and neither means anything alone:

- **false rejection** — of documents you labelled `keep`, how many were dropped
- **noise removal** — of documents you labelled `drop`, how many were dropped

A gate that removes nothing scores perfectly on the first. A gate that removes
everything scores perfectly on the second.

**Label a held-out set.** JSONL, one document per line. `phenomenon` is required
and closed, because a set that happens to contain no OCR noise reports a rate
that says nothing about OCR noise:

```json
{"id": "vi-001", "language": "vi", "label": "keep", "phenomenon": "clean", "text": "…"}
```

`label` is `keep` or `drop`. `phenomenon` is one of `clean`, `nfd`,
`local_digits`, `ocr_noise`, `code_mixing`, `langid_error` — the five failure
modes that have produced a wrong result before, plus the baseline. Aim for both
labels in every mode you care about; 50 documents finds a systematic error, 160
is enough to quote a rate.

The sets under `../evaluation/` ship as worked examples of the format, one per
example language. They are constructed to exercise the failure modes above, so
they show the pipeline still handles those modes -- they are not a benchmark and
they do not describe your corpus. Write your own and point `--labelled` at it.

**Compute the rates.**

```bash
uv run --extra curate python -m nemotron.steps.curate.nemo_curator.scripts.run_evaluate \
  --policy ./output/<lang>/policy/approved_policy.yaml \
  --labelled ./eval/<lang>.jsonl \
  --langpack-dir ./langpacks \
  --report ./output/<lang>/evaluation.json
```

Read the per-phenomenon table, not the headline. An aggregate over a
mostly-clean set reports a good number while rejecting every OCR-noised document
in it. The per-signal table names which threshold to move: "the policy rejects
12% of good Hindi" sends someone hunting, "`script_ratio` rejects 12% of good
Hindi" does not.

**What it will not tell you.** Only the locally implemented signals are scored —
the pack-backed ones plus `unicode_alpha_numeric`. Signals wrapping a NeMo
Curator filter are listed as unscored rather than skipped quietly. And a rate is
worth exactly what its labelled set is worth: a constructed set proves the
pipeline still handles the modes it contains, and establishes no rate for any
real corpus.

**Re-run after any edit to `charset.txt`** — a character added or removed moves
every script-based score.

Any threshold range you publish is provisional until a blind human review backs
it. Label it so.

## 9. Activate it

Two shapes, depending on which entry point you use.

**Flow config** — the pack is selected once and derived into every step:

```yaml
corpus:
  language: <your-tag>          # selects the pack
  langpack_dir: ./langpacks     # explicit; there is no default
```

**Standalone** `curate/nemo_curator` — the policy carries the pack identity:

```yaml
heuristic_filters:
  approved_policy: ./policy/approved_policy.yaml
  langpack_dir: ./langpacks
```

The approved policy records the pack's `content_hash`, and the filter run
refuses if the pack on disk no longer matches. That check certifies *the same
bytes as when this was approved* — it does not certify the right language, which
is why step 1's tag agreement matters separately.

There is no default language and no default pack directory. Both are required,
because a wrong default produces plausible numbers for the wrong language.

---



## Before you ship the pack

Only the things nothing else checks. A capability declared without its file, a
tag that disagrees with its directory, a malformed manifest — the loader already
refuses all of those, so they are not on this list. What is left is what fails
quietly.

- [ ] `charset.txt` written character by character; the `charset` count from
  ```
  `describe()` is the number you expected, not three.
  ```
- [ ] Devanagari-style `Mc` marks handled, and checked in both NFC and NFD.
- [ ] Every declined capability recorded in `[notes]` with the measurement
  ```
  behind it, so the omission reads as a decision.
  ```
- [ ] Provenance recorded in `SOURCES.md`, licence texts in `LICENSES/`.
- [ ] FastText model checksum recorded, and your language confirmed to be among
  ```
  the 176 the model predicts.
  ```
- [ ] `corpus.language` and `steps.filter.language_codes` name the same
  ```
  language — nothing checks this, and disagreeing still reconciles.
  ```
- [ ] Thresholds approved by a named person against named evidence.
- [ ] Kept *and* dropped documents read by hand; false-rejection and
  ```
  noise-retained rates both recorded.
  ```
- [ ] Any threshold range you publish labelled provisional.