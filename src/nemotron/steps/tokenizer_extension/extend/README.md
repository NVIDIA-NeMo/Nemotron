# tokenizer_extension/extend

Train one BPE on a corpus and splice it into the base tokenizer as **Add**,
**Replace**, or **Expand** — one arm per job. CPU-only.

## Files
- `step.py` — executor (thin; loads YAML, calls `extension.run_extension`)
- `extension.py` — corpus reader (HF / local parquet / jsonl) + the arm builder selected by `method`
- `continued_bpe.py` — shared core: streaming, merge-diff, **constructive splice**, rank-dead check
- `replace_bpe.py` — prune + dense re-index + NFKC-preserving wrap (Replace path)

## Corpus input (config `corpus:`)
Provide **either** an HF dataset **or** a local path, plus the text column:
```yaml
corpus:
  hf_dataset: ai4bharat/sangraha   # OR set path: below for local data
  hf_name: verified                # dataset config/subset
  hf_split: hin
  path: null                       # local parquet dir/glob or jsonl
  glob: "*.parquet"
  text_field: text                 # the text column
  samples: 1000000
  max_doc_chars: 20000
  min_frequency: 0                 # 0 = keep all merges
```

## Add vs Replace vs Expand
`method: add | replace | expand` — one arm per job.
- **add** keeps the base residual script tokens and appends new ones.
- **replace** prunes them (script chosen by `language:`, or `remove_script:`) and
  refills with corpus-optimal tokens spliced into the merge table.
- **expand** adds the decoded surfaces atomically via `add_tokens()` with **no
  merge rules** — the naive baseline. Cheaper to build. Usually worse fertility
  than `add`, but not always: where the base vocabulary barely covers the target
  script there is little for merge training to add, and `expand` has measured
  *better* (Malayalam). Compare on your own language rather than assuming.

Set `language:` (hindi, vietnamese, malayalam, ...) to pick the normalizer and the
Replace prune script; see `../LANGUAGES.md`. See also the category `guide.md`.

## Why the splice is safe
Naively appending trained merges below the base merges can leave tokens in the
vocab that greedy BPE can never emit (they get shadowed by a higher-priority base
rule) — which silently inflates fertility while every graph-reachability check
still passes. `continued_bpe._apply_bpe_extension_backend` instead builds each new
token from the pieces the *current* merge table yields, so the appended rule is the
one that fires. `find_rank_dead_tokens` asserts zero unemittable tokens; the build
**hard-fails** otherwise.

## Dependencies
`language:` hindi/marathi/nepali/sanskrit needs `indic-nlp-library`
(`uv pip install -e '.[tokenizer-extension]'`, or add it to the profile's
`pip_extras`). Missing, the run fails rather than silently using NFKC-only
normalization, which trains a different tokenizer. See `../guide.md`.

## Run
```bash
L=vietnamese          # your target language; see ../LANGUAGES.md
uv run nemotron steps run tokenizer_extension/extend \
  -b lepton_tokenizer_extend -c default language="$L" method=add
```
`-b` names a profile in your repository-root env file; see `../guide.md`.
Output: `output_dir/<method>/` — `add/`, `replace/` or `expand/` — containing the
tokenizer and `summary.json`. Check `tokens_spliced` in `summary.json` matches
`extension_size`; fertility is only comparable between arms at a matched budget.
