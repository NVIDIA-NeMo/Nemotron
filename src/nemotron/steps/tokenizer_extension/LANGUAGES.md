# Adding a language

The tokenizer-extension pipeline started life Hindi-only. It is now driven by a
single `language:` key that every step understands. Adding a language should be
one edit in `languages.py` (plus one in `script_ranges.py` if the language has
no entry yet) — not a code change in any step.

## Using an existing language

Set `language:` at the top level of any extend / init_embeddings config:

```yaml
language: vietnamese
method: replace
extension_size: 30000
```

That one key supplies:

| What | Resolved from | Used by |
|---|---|---|
| Corpus text normalizer | `LanguageProfile.normalizer` | extend (BPE training) |
| Tokens `method=replace` prunes | `LanguageProfile.script` | extend |
| Base model's existing target rows | `LanguageProfile.script` | init_embeddings (norm correction, FOCUS pool) |
| Auxiliary encoder for `bert` init | `LanguageProfile.encoder` | init_embeddings |
| fastText vectors for `focus` init | `LanguageProfile.fasttext` | init_embeddings |

Every one of these stays individually overridable — `script_normalizer:`,
`remove_script:`, `subword.bert_model:`, `focus.fasttext_model:`,
`focus.fasttext_url:`. A one-off experiment never needs a registry entry.

Currently registered: hindi, marathi, nepali, sanskrit, bengali, punjabi,
gujarati, odia, tamil, telugu, kannada, malayalam, urdu, vietnamese.

## Adding a new language

**1. Unicode ranges** — `script_ranges.py`, only if the script is not already
there. A token belongs to a language if any character of its *decoded* surface
falls in a range.

```python
"thai": [(0x0E00, 0x0E7F)],
```

For a language with its own block, that block is the answer. For a
**Latin-script language**, there is no block — enumerate the codepoints only
that language uses, the way `vietnamese` does (precomposed vowel+tone, horned
o/u, combining horn). Never include plain ASCII Latin: it is shared with every
other Latin-script language, and pruning it under `method=replace` would
destroy the byte alphabet that new merges rebuild from.

**2. Profile** — `languages.py`:

```python
"thai": LanguageProfile("none", "thai", "th", _XLMR),
#                        │       │       │     └─ encoder covering the language
#                        │       │       └─ fastText cc.<code>.300
#                        │       └─ key in SCRIPT_UNICODE_RANGES
#                        └─ extra normalizer beyond NFKC ("none" for most)
```

Pick the encoder with care. MuRIL covers 17 Indian languages **and nothing
else** — it is right for the Indic entries and wrong everywhere else, which is
why Vietnamese uses XLM-R. An encoder that does not cover the target language
produces a `bert` init no better than noise.

Only add a `normalizer` if the script genuinely needs one beyond NFKC. NFKC is
always applied; `devanagari` additionally runs indic-nlp's `DevanagariNormalizer`.
New normalizers go in `languages.get_normalizer()`.

**3. That is all.** No step needs editing.

## Backward compatibility

Configs with no `language:` key behave exactly as before: Devanagari
normalizer, Devanagari prune set, MuRIL, `cc.hi.300`. Every committed Hindi
config keeps working untouched.

Config keys were renamed language-neutrally, with the old names still accepted:

| Old | New |
|---|---|
| `subword.input_hindi_norm` | `subword.input_target_norm` |
| `subword.output_hindi_norm` | `subword.output_target_norm` |
| `baseline.mode: mean_hindi` | `baseline.mode: mean_target` |
| `focus.candidate_pool: hindi` | `focus.candidate_pool: target` |

The `--input-hindi-norm` / `--output-hindi-norm` CLI flags keep their names for
backward compatibility. The internal script test is `is_target_script()`, which
delegates to the active target script.

## Caveats

- **`focus` on syllable-spaced languages.** fastText tokenizes on whitespace, so
  for Vietnamese the `cc.vi.300` vectors are largely *syllable* vectors. New
  tokens spanning several syllables have no direct entry and fall back to
  subword composition. FOCUS was already the weakest init in the Hindi study;
  expect it to be weaker still here.
- **`method=replace` on Latin-script languages** frees far fewer ids than for a
  language with its own block — Vietnamese yields 761 rows against Hindi's
  ~1,569 — because only tokens carrying a diacritic qualify.
