# tokenizer_extension/init_embeddings

Attach an extended tokenizer to the base model, initialize the new embedding (and
LM-head) rows, and save a **resized HF checkpoint** for CPT.

The base model is embedding surgery only (no forward pass) and is loaded
**host-resident**, not on GPU. A GPU is used only by the auxiliary encoders:
`bert_weighted` (MuRIL-class) and `gemma_weighted`.

The step and the shipped profiles both request **one** GPU, which is enough for
every method except `gemma_weighted` on a large Gemma — that path shards with
`device_map='auto'`, so give it a multi-GPU profile if you use it.

## Files
- `step.py` — executor (loads YAML, calls `embeddings.run_init`)
- `embeddings.py` — the `method:` dispatcher + Add/Replace argv construction

## Adding a new init technique (the whole point)
`embeddings.py` does not hold the strategies itself; it selects an **engine
script** from `METHODS = ("baseline", "subword", "focus")` and builds its argv.
The Replace arm routes through `replace_init.py`, which additionally copies
survivor rows via `id_remap.json`.

To add a technique:
1. write `<name>_init.py` next to the others, taking `--base-model`,
   `--extended-tokenizer`, `--output-dir`, `--dtype` and `--language`;
2. add `"<name>"` to `METHODS`;
3. add a `_<name>_argv(cfg)` builder and a dispatch branch in `embeddings.py`
   (and in `_replace_argv` if it should support the Replace arm).

Select it with `method: <name>`.

Shipped: `baseline` (`hf_default` | `mean_all` | `mean_target`), `subword`
(`input_averaging`: `uniform` = mean-of-constituents, `char_weighted`,
`max_char`, `bert_weighted`, `gemma_weighted`), and `focus` (fastText +
sparsemax). Set `language:` so the auxiliary encoder and fastText vectors are
chosen for your target — passing `subword.bert_model` explicitly overrides it.

## What each method does

New rows have to come from somewhere. The three families differ in what
information they use to place a new token in embedding space.

### `baseline` — no per-token information

| `baseline.mode` | New row is initialized from |
|---|---|
| `hf_default` | HuggingFace's own resize: a multivariate normal fitted to the mean and covariance of the existing embeddings |
| `mean_all` | the mean of **every** existing embedding |
| `mean_target` | the mean of the base model's existing **target-language** rows only |

Every new token gets essentially the same vector, so the model must learn the
distinctions during CPT. Cheap, and a fair floor to compare against.

### `subword` — compose from the pieces the base already knows

Each new token is decomposed into subwords of the **original** vocabulary and
initialized from a weighted average of their rows. Input and output sides are
weighted independently.

| `input_averaging` / `output_averaging` | Weighting |
|---|---|
| `uniform` | plain mean of the subword rows (mean-of-constituents) |
| `char_weighted` | weighted by each subword's character length |
| `max_char` | the longest subword's row alone (ties: first occurrence) |
| `bert_weighted` | `softmax(cos(subword, full token) / temperature)`, using mean-pooled encoder hidden states |
| `gemma_weighted` | the same weighting from a decoder's input embedding table instead of a forward pass |

`uniform` is the default and a strong baseline: a new token means roughly what
its parts meant. The encoder-weighted variants ask an auxiliary model which
parts matter most, and need that model to actually cover your language — set
`language:` and let the profile choose it.

### `focus` — borrow from similar tokens in an auxiliary space

FOCUS (Dobler & de Melo, EMNLP 2023). Each new token is a Sparsemax-weighted
combination of the base tokens closest to it in a fastText space. fastText uses
character n-grams, so it can embed strings that never appeared in its own
vocabulary. Cosine similarities are divided by a temperature before Sparsemax,
which zeroes most weights so only genuinely similar tokens contribute — without
the temperature the mass spreads over thousands of tokens and degenerates into a
dense average.

fastText tokenizes on whitespace, so it suits languages with space-separated
words better than syllable-spaced ones.

### Norm correction

`input_norm_correction` rescales new **input** rows so their L2 norm matches the
median norm of the existing rows. Output (LM-head) rows are deliberately **not**
rescaled: inflating their norm makes the model over-confidently predict the new
tokens and the training loss explodes.

## Add vs Replace surgery

The initialization *math* is arm-independent: a new row is built the same way
whether the base kept or pruned its target-script tokens. What the arm changes
is the surgery around it.

- **add** → resize, keep base rows, init appended rows `[old_n, new_n)`.
- **replace** → resize, permute survivor rows via `id_remap.json`, then init new
  rows `[pruned_size, final_n)`. (`extended_tokenizer` must be extend's
  `replace/` output, which carries that remap.)

### Method support by arm

The two arms run through different engines (`replace_init.py` for Replace), and
that engine does not implement every method. These are implementation gaps, not
statements about which init is appropriate — the step **fails loudly** rather
than substituting a near-equivalent, because a silent substitution would report
one init while running another.

| Method | `add` | `replace` |
|---|:--:|:--:|
| `baseline.hf_default` | ✅ | ✅ |
| `baseline.mean_all` | ✅ | ✅ |
| `baseline.mean_target` | ✅ | ❌ no target-script-mean engine |
| `subword.uniform` / `char_weighted` / `max_char` | ✅ | ✅ |
| `subword.bert_weighted` | ✅ | ✅ |
| `subword.gemma_weighted` | ✅ | ❌ no gemma path |
| `focus` | ✅ | ✅ |

For a Replace run needing target-script-mean behaviour, the closest supported
option is `subword` with `input_averaging: uniform` (mean of constituents).

Both sides (`input_averaging` / `output_averaging`) are always forwarded on both
arms, so input and output stay independently weighted.

**Pairing with `extend`.** `arm` must match how the tokenizer was built:

| `extend` `method` | Use `arm` | Why |
|---|---|---|
| `add` | `add` | rows are appended; base ids unchanged |
| `expand` | `add` | also append-only (`add_tokens()`), so the same surgery applies |
| `replace` | `replace` | pruning renumbers ids; survivors are permuted via `id_remap.json` |

`arm=replace` requires extend's replace output — it is the only one that writes
`id_remap.json`. Pointing it at an append-style tokenizer has no remap to apply.

## Dependencies
`method: focus` needs `fasttext-wheel`
(`uv pip install -e '.[tokenizer-extension]'`, or add it to the profile's
`pip_extras`). The import is lazy, so the other methods run without it.
See `../guide.md`.

## Run
```bash
L=vietnamese          # must match the language used by extend
uv run nemotron steps run tokenizer_extension/init_embeddings \
  -b lepton_tokenizer_init_embeddings -c default \
  language="$L" arm=add extended_tokenizer=./output/tokenizer_extension/add
```
`-b` names a profile in your repository-root env file; see `../guide.md`.
Output: `output_dir/` = resized HF checkpoint (weights + tokenizer) → set as
`pretrain/megatron_bridge` `hf_model_path`.
