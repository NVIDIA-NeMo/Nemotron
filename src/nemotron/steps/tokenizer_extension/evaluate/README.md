# tokenizer_extension/evaluate

Corpus-level token **fertility** (`sum(tokens)/sum(words)`) for a tokenizer on an
eval corpus. CPU-only, streamed (memory-safe on 10M+ rows).

> Scope: **tokenizer-level only**. Model / downstream evaluation after CPT is done
> with the existing `eval` catalog — not here.

## Files
- `step.py` — executor (loads YAML, calls `fertility.run_fertility`)
- `fertility.py` — streaming eval-corpus reader + fertility computation

## Config
```yaml
tokenizer: ./output/tokenizer_extension/replace   # HF id or local dir
corpus:
  hf_dataset: ai4bharat/samanantar   # OR set path: for local parquet/jsonl
  hf_config: hi
  hf_split: train
  text_field: tgt                    # samanantar Hindi column
  num_docs: 0                        # 0 = full
  skip_docs: 0                       # skip a train-overlapping slice if needed
output: ./output/fertility/replace_samanantar_hi.json
```
Run the same corpus block across tokenizers to keep numbers comparable.

## Run
```bash
uv run nemotron steps run tokenizer_extension/evaluate \
  -b lepton_tokenizer_evaluate -c default tokenizer=./output/tokenizer_extension/add
```
Output: a JSON with `fertility`, `chars_per_token`, `unique_tokens_used`,
`vocab_coverage`, and timing.
