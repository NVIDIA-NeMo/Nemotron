# tokenizer_extension

Extend a base tokenizer with target-language subwords, initialize the model
embeddings for the new vocab, and (optionally) measure fertility — producing a
resized HF checkpoint ready for continued pretraining.

> **Choosing a recipe?** See the customer-facing
> [Tokenizer Extension Guidebook](guidebook/README.md) for measured vocabulary
> knees, initialization guidance, BPB comparisons, and serving results.

## Steps (typed-artifact pipeline)

| Step | Compute | Consumes → Produces |
|------|---------|---------------------|
| `extend` | CPU | `checkpoint_hf` (tokenizer) + corpus → `tokenizer` (one arm per job: add/, replace/ or expand/) |
| `init_embeddings` | GPU | `tokenizer` + base `checkpoint_hf` → resized `checkpoint_hf` |
| `evaluate` | CPU | `tokenizer` → `eval_results` (fertility) |
| `eval_init` | GPU | resized `checkpoint_hf` → `eval_results` (BPB / perplexity) |

```
extend ──▶ tokenizer ──▶ init_embeddings ──▶ checkpoint_hf ──▶ pretrain/megatron_bridge (CPT)
   └─────▶ evaluate (fertility)
```

## Add vs Replace
- **Add** keeps the base's residual script tokens and appends new ones (`vocab = V + N`). Safer across a script family (e.g. Hindi + Marathi).
- **Replace** prunes the target script's residual tokens, then splices fresh corpus-optimal ones into the pruned base (`vocab = V − R + N`). Fewer rows; best for a single target language.

The splice is *constructive* (rank-dead-safe): every added token is emittable by the merge table, so the tokenizer never silently byte-fragments (see `extend/README.md`).

## Design notes
- Each step is **self-contained**: `step.py` is the executor and its supporting `.py` files live beside it. The only shared modules are `languages.py` and `script_ranges.py` at the block root, which every sub-step imports. Backend (Slurm / Lepton / DGX) is chosen at run time by the selected `env.toml` profile, not the step.
- **`init_embeddings` is the growth point** for embedding-init research: `embeddings.py` dispatches `method:` (`baseline` | `subword` | `focus`, see `METHODS`) to a separate engine script — `baseline_init.py`, `subword_init.py`, `focus_init.py`, and `replace_init.py` for the Replace arm. Add a technique by adding an engine and a `METHODS` entry.
- Post-CPT model / downstream evaluation is **not** here — use the `eval` catalog.

## Dependencies

These packages are **not** in the base install and are declared as extras
(`pyproject.toml` → `[project.optional-dependencies]`):

| Package | Needed by | If missing |
|---|---|---|
| `indic-nlp-library` | `extend` with `script_normalizer: devanagari` (i.e. `language:` hindi/marathi/nepali/sanskrit) | **hard error** — it used to fall back to NFKC-only, which silently trained a *different* tokenizer |
| `fasttext-wheel` | `init_embeddings` with `method: focus` | **hard error** at the point of use (import is lazy, so other methods are unaffected) |
| `mamba-ssm`, `causal-conv1d` (extra `tokenizer-extension-gpu`) | the **default** `base_model`, which is a Mamba-hybrid | **hard error** while loading the model |

Local install:
```bash
uv pip install -e '.[tokenizer-extension]'
```
For a remote runner the container is built from the profile, so add them there —
in your `env.toml`, on the profile used for these steps:
```toml
pip_extras = ["typer", "rich", "pydantic-settings", "indic-nlp-library", "fasttext-wheel"]
```
Neither is installed by the default profile; a run will fail with the exact
install command rather than produce a quietly different tokenizer.

## Quickstart — extend a tokenizer for your language

Four commands. Each step writes what the next one reads, so run them in order.

**First, set up a run profile.** `-b <profile>` names a profile in a
repository-root env file, which is site configuration and is not checked in.
Create or verify it before any remote run: the default lookup is `env.toml`,
and a generated example such as `env.lepton.toml` must be selected explicitly.

```bash
export NEMOTRON_ENV_FILE=$PWD/env.lepton.toml   # only if not using env.toml
```

The profile names below (`lepton_tokenizer_*`) are placeholders — substitute the
profiles your env file actually defines. Each of these steps needs
`indic-nlp-library` (and `fasttext-wheel` for `method: focus`) in that profile's
`pip_extras`, as shown above. Omit `-b` to run locally instead.

Placeholders below are quoted so they are safe to paste; replace the quoted text
before running.

```bash
L=vietnamese                                  # your target; see LANGUAGES.md
OUT=./output/tokenizer_extension              # extend's output_dir
DATASET="your-org/your-dataset"               # HF dataset id for BPE training
SPLIT="train"                                 # its split

# 1. Build the extended tokenizer (CPU). Pick ONE arm per job.
uv run nemotron steps run tokenizer_extension/extend \
  -b lepton_tokenizer_extend -c default \
  language=$L method=add extension_size=30000 \
  corpus.hf_dataset="$DATASET" corpus.hf_split="$SPLIT"
#    -> $OUT/add/   (tokenizer + summary.json; check tokens_spliced == extension_size)

# 2. Initialise the new embedding rows -> resized HF checkpoint (GPU).
uv run nemotron steps run tokenizer_extension/init_embeddings \
  -b lepton_tokenizer_init_embeddings -c default \
  language=$L arm=add extended_tokenizer=$OUT/add \
  output_dir=./output/resized_checkpoint
#    -> ./output/resized_checkpoint/   <- this is CPT's hf_model_path

# 3. (optional) How many tokens per word? Lower is better.
uv run nemotron steps run tokenizer_extension/evaluate \
  -b lepton_tokenizer_evaluate -c default tokenizer=$OUT/add

# 4. (optional) BPB vs the base model — the cross-vocabulary quality check.
uv run nemotron steps run tokenizer_extension/eval_init \
  -b lepton_tokenizer_eval_init -c default \
  models=[./output/resized_checkpoint] max_docs=2000
```

Add `-d` to any command to print the compiled config without running it.

## Choosing the arm

| `method` | What it does | Use when |
|---|---|---|
| `add` | trains a BPE on your corpus and splices the new tokens **into the merge table** | default choice; robust across a script family |
| `replace` | first prunes the base's existing target-script tokens, then splices | one target language, want the smallest vocab |
| `expand` | registers decoded surfaces via `add_tokens()`, **no merge rules** | baseline for comparison only — atomic tokens do not compose, so extra vocabulary buys little |

The init step's `arm:` must match: `add` and `expand` are both append-style so
both use `arm=add`; only `replace` uses `arm=replace`. Not every init method is
available on `replace` — see `init_embeddings/README.md`.

## Adapting to your language

Only three things are language-specific:

1. **`language:`** — picks the corpus normalizer, the Replace prune script, the
   auxiliary encoder and the fastText vectors, all from one key. Supported values
   and how to add a new one: `LANGUAGES.md`.
2. **`corpus:`** — an HF dataset (`hf_dataset` / `hf_name` / `hf_split`) or a
   local path (`path` + `glob`), plus `text_field`.
3. **`extension_size:`** — how many tokens to add. 30k is a reasonable default.

Do **not** set `remove_script:` or `subword.bert_model:` unless you are
deliberately overriding the profile — an explicit value silently wins over
`language:`.

## Outputs

| Step | Writes | Next consumer |
|---|---|---|
| `extend` | `output_dir/{add,replace}/` + `summary.json` | `init_embeddings.extended_tokenizer` |
| `init_embeddings` | `output_dir/` (weights + tokenizer) | `pretrain/megatron_bridge.hf_model_path` |
| `evaluate` | fertility JSON | — |
| `eval_init` | BPB/perplexity JSON | — |

`summary.json` records `tokens_spliced`; check it equals `extension_size`.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `Unknown script 'None'` | `remove_script:` set to null in an older config — remove the key |
| `tokens_spliced` < `extension_size` | corpus too small for the requested budget; widen it or lower `extension_size` |
| BPB run refuses to start | scoring several tokenizers under `max_tokens`; set `max_docs` instead |
| FOCUS init fails on import | install the `tokenizer-extension` extra |
