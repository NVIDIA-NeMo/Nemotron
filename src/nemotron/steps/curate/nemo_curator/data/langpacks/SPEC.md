# Language Pack Specification

A language pack is **data**. Nothing in `runtime/` knows any language exists —
character sets, word lists, patterns and fold maps all arrive from here. Adding
a language means adding a directory, not editing code.

Nemotron ships three opt-in example packs — `en/`, `vi/` and `hi/` — chosen
because they break different assumptions: English is unmarked Latin, Vietnamese
is Latin whose NFD produces only `Mn` marks, and Hindi is an abugida whose
matras span `Mn` and `Mc` with `Mc` in the majority. The `en` pack is built from
the Snowball stopword list and Unicode CLDR 48 exemplar characters, with source
content hashes, transformations and license texts stored beside it; `vi` and
`hi` were assembled for this repository and record origin and licence per list.
None of them is an implicit language, pack root, threshold set, or quality
claim.

For another language, a pack is a reviewed, versioned input owned by the corpus
workflow using it; point `langpack_dir` at that external directory. The
language-specific packs in `tests/steps/curate/fixtures/langpacks/` are
private-use validation fixtures, not defaults or claims of supported languages.

## Layout

```
<langpack-dir>/<bcp47-tag>/
├── pack.toml        the manifest
├── stopwords.txt    one function word per line
├── charset.txt      one character per line (see below)
└── boilerplate.txt  one regular expression per line
```

Blank lines and lines starting with `#` are ignored in every `.txt` file, so
each can carry its own provenance header.

`charset.txt` is parsed as the **union of every character on every non-comment
line** — `frozenset("".join(lines))`. One per line is the readable convention,
not a rule, so two consequences are worth stating because neither raises:

- **Ranges are not expanded.** A line reading `a-z` contributes exactly `a`,
  `-` and `z`, so the pack silently gains a hyphen and 24 of the 26 letters are
  missing. Write the characters out.
- **`#` and the space character cannot be members.** A line starting with `#` is
  a comment, and a line that is only a space is stripped to nothing.

The directory name is a **BCP-47 tag** (`vi`, `hi`, `pt-BR`), not an ISO 639-1
code. Private-use tags (`x-…`) are valid and are what test fixtures use.

## `pack.toml`

```toml
[pack]
pack_id      = "vi-generic"   # identifies the pack; may differ from the tag
language_tag = "vi"           # BCP-47
version      = "2.0"
schema       = 1

[sources]
stopwords   = { file = "stopwords.txt",   n = 287, origin = "…", license = "…" }
charset     = { file = "charset.txt",     script = "Latn", origin = "…", license = "…" }
boilerplate = { file = "boilerplate.txt", n = 31,  origin = "…", license = "…" }

[orthography]
sentence_terminators = [".", "!", "?"]

[fold_map]
"đ" = "d"

[capabilities]
supports = ["script_ratio", "stopword_ratio", "boilerplate_hits", "sentence_end_ratio"]
```

## Capabilities

`supports` is the load-bearing field. It declares what this language can
meaningfully be measured for, and a signal whose capability is undeclared is
**absent from the report** rather than computed on a false premise.

| Capability | Needs | Enables |
|---|---|---|
| `script_ratio` | `charset` | `script_ratio`, `latin_ratio`, `foreign_script_ratio` |
| `diacritic_ratio` | `charset`, `fold_map` | `diacritic_ratio` |
| `stopword_ratio` | `stopwords` | `stopword_ratio` |
| `stopword_ratio_folded` | `stopwords`, `fold_map` | `stopword_ratio_folded` |
| `boilerplate_hits` | `boilerplate` | `boilerplate_hits` |
| `sentence_end_ratio` | `sentence_terminators` | `sentence_end_ratio` |

Three more are assertions about the **writing system** rather than claims to
carry a file, so they have no `Needs` column and the loader cannot check them.
They must be declared by someone who knows the language, and they exist because
documenting a hazard was not enough — the measurement stayed available and a
policy could still name it:

| Capability | Assert it when | Withholding it disables |
|---|---|---|
| `word_segmentation` | words are delimited by whitespace | `word_count`, `mean_word_length`, `max_word_length`, `symbol_to_word`, `words_with_alphabets`, `repeating_duplicate_ngrams` |
| `ascii_digits` | numbers are written with `[0-9]` | `numbers_ratio` |
| `ascii_punctuation` | sentences use ASCII `.`, `!`, `?` | `punctuation` |
| `ascii_alphabet` | the alphabet itself is ASCII `a-zA-Z` | `non_alpha_numeric` |

A pack that does not declare one has those signals skipped with a named warning
during profiling, and a config or policy that asks for one by name is refused.
The shipped `hi` pack is the worked example: Hindi **does** delimit words with
whitespace, so it declares `word_segmentation`; its digits are Devanagari and its
sentences end with the danda, so it declares neither ASCII capability. Declaring
them because they look harmless puts a whitespace word count on a script that
has no word boundaries — 39% of a Japanese corpus removed for having no spaces —
or Curator's ASCII content class on an alphabet that is not ASCII, which scores
one ordinary Vietnamese sentence at 0.429 against a shipped default of 0.25.
Only English declares `ascii_alphabet`; use `unicode_alpha_numeric` instead,
which accepts Unicode categories L, N and M.

Declaring a capability without the data behind it is rejected at load. It would
otherwise fill a report with zeroes, which reads as a finding about the corpus
rather than a hole in the pack.

Every source entry must record both `origin` and `license`. The loader rejects
an asset whose provenance is absent, even when the file itself is present.

**Do not declare a capability that does not apply to your language.** Vietnamese
tone marks strip to degraded but readable text, so a diacritic ratio measures
something real. Devanagari matras are obligatory vowels; stripping them yields
nonsense, so the `hi` pack declares neither `diacritic_ratio` nor
`stopword_ratio_folded`. That absence is the correct answer, not a gap to fill.

### Worked example: declining `stopword_ratio`

Every signal here tokenises on whitespace, so a language that does not delimit
words that way cannot support the ones that count tokens. Measured on 20,000 C4
documents per language:

| Language | Score exactly zero | Of those, correct native script |
|---|---|---|
| Japanese | 93.7% | 87.9% |
| Thai | 53.1% | 90.1% |

A "token" is a whole run of text and never matches a stopword. This is not an
undercount to be corrected with a lower threshold: the signal cannot distinguish
"not Japanese" from "Japanese, written normally", which is the one distinction it
exists to make. Declaring it would produce a clean-looking distribution over
nothing.

Record the measurement in the pack's `[notes]` table, so the omission reads as a
decision rather than an oversight:

```toml
[notes]
stopword_ratio_not_declared = """Measured on 20,000 C4-ja documents: 93.7% score
EXACTLY zero, and 87.9% of those are correct Japanese (script_ratio > 0.5)."""
```

`[notes]` is a top-level table and is the only place a note is read from. A key
placed inside `[capabilities]` beside `supports` is **silently ignored** — the
loader reads `capabilities.supports` and nothing else from that table, so the
measurement would never reach `describe()` or any report.

A morphological tokeniser — MeCab for Japanese, pythainlp for Thai — would change
this, but it is a runtime change rather than a pack change. When one exists the
capability can be declared.

## `fold_map`

Only for marks the language treats as removable. The runtime already drops
Unicode combining marks; the map is for characters that do **not** decompose —
Vietnamese `đ` is the reason it exists.

Folding merges distinct words together (`mà`, `má`, `mã`, `mạ` all become `ma`).
That collision count is reported alongside any folded figure. It is a property
of the language, not a defect in your word list.

## `charset`

Every character that counts as this language's own script, including marked
forms. If you list only base letters, correct text scores as foreign.

Text resources and fold-map entries are normalized to NFC when loaded, matching
the normalization used before signal scoring.

For an abugida, include the matras — and note that they split across Unicode
categories `Mn` (nonspacing) and `Mc` (spacing combining). Omitting `Mc` is the
single most common way to break an Indic or South-East Asian pack: a filter
written against `Mn` alone passes every Latin-script test and rejects correct
Devanagari outright.

## `sentence_terminators`

Defaults to `.`, `!`, `?`. Override it for any script that ends sentences
otherwise — Hindi and Sanskrit use the danda `।` (U+0964), which is Unicode
category `Po` and matches none of the defaults.

## Before you ship one

1. `load()` it and check `describe()` reports the counts you expect.
2. Run `curate/profile` against a corpus in that language with `signals: []` and
   confirm the skipped-capability warnings say what you intended.
3. Score correct text through `UnicodeAwareNonAlphaNumericFilter` and confirm it
   is kept at the 0.25 default. If it is not, the charset is incomplete.
4. Score text in **both** NFC and NFD. Some scripts differ materially between
   them; some do not. A pack tested in only one form is tested in neither.

## What packs are not for

Thresholds. A pack says what *can* be measured for a language, never what a good
value is — that depends on the corpus, and finding it is what `curate/profile`
is for. Numbers chosen for one corpus do not transfer to another in the same
language.
