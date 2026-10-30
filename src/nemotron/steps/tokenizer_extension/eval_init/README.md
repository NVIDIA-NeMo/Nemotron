# tokenizer_extension/eval_init

Score a resized checkpoint with **BPB (bits per byte)** and perplexity, against
the base model. Needs a GPU.

## Why BPB and not perplexity
Perplexity is per-token, so it drops mechanically when a tokenizer covers more
bytes per token — a bigger vocabulary looks better without being better. BPB
divides by UTF-8 bytes instead, so it is comparable **across vocabularies**.

## The one rule
Every model in a comparison must score the **same bytes**.

- ✅ `max_docs: N` (or `-1` for the whole corpus) — document counts are
  tokenizer-independent.
- ❌ `max_tokens: N` — each tokenizer reaches a token budget after a *different*
  amount of text, so the BPB values are not comparable.

Scoring more than one model under `max_tokens` with no `max_docs` is refused.
Pass `--allow-token-cap-comparison` only to reproduce a historical run.

## Files
- `step.py` — executor
- `bpb.py` — sliding-window scoring, byte accounting, comparison table

## Config
```yaml
base_model: <hf model id or path>     # scored alongside, as the reference
models:                                # one or more resized checkpoints
  - ./output/resized_checkpoint
data_file: ./data/eval/target_val.jsonl   # .jsonl with text_field, or .txt
text_field: text
max_docs: 2000                        # tokenizer-independent budget
max_tokens: -1
max_length: 2048
stride: 512
```

## Run
```bash
uv run nemotron steps run tokenizer_extension/eval_init \
  -b lepton_tokenizer_eval_init -c default \
  models=[./output/resized_checkpoint] max_docs=2000
```
Output: per-model BPB / perplexity / token + byte counts, and the delta vs base.
Lower BPB is better. Compare within a language only.
