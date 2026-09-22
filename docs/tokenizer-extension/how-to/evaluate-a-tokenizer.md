---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Compare tokenizer fertility with tokenizer_extension/evaluate and bits per byte with tokenizer_extension/eval_init on identical text."
topics: ["Tokenizer Extension", "Evaluation"]
tags: ["How-To", "Tokenizer Extension"]
content:
  type: "How-To"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Evaluate a Tokenizer

Use `tokenizer_extension/evaluate` to compare fertility across tokenizers, and `tokenizer_extension/eval_init` to compare bits per byte (BPB) across resized checkpoints.
For the reasoning behind the metrics, see {doc}`../explanation/evaluation-metrics`.

## Compare Fertility

The `evaluate` step is CPU-only and streams its corpus.

1. Choose an evaluation corpus that is distinct from the corpus that trained the extension.
   The default is `ai4bharat/samanantar` with `hf_config: hi` and `text_field: tgt`.
   When the two share a source, set `corpus.skip_docs` to skip the slice that `extend` trained on.

1. Run the step once per tokenizer with the same `corpus` block.
   Include the base tokenizer as a reference.

   ```console
   $ uv run nemotron steps run tokenizer_extension/evaluate -c default \
       tokenizer=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 label=base \
       output=./output/eval/fertility_base.json
   $ uv run nemotron steps run tokenizer_extension/evaluate -c default \
       tokenizer=./output/tokenizer_extension/add label=add-30k \
       output=./output/eval/fertility_add.json
   $ uv run nemotron steps run tokenizer_extension/evaluate -c default \
       tokenizer=./output/tokenizer_extension/replace label=replace-30k \
       output=./output/eval/fertility_replace.json
   ```

   `tokenizer` accepts a Hugging Face identifier or a local directory.
   `label` is a free-text tag recorded in the output JSON.

1. Compare `fertility` across the reports. Lower is better.
   Before comparing two arms, confirm that their `summary.json` files report the same `tokens_spliced`.

To use a local corpus, set `corpus.hf_dataset=null corpus.path=<dir-or-glob>` and the matching `corpus.text_field`.
`corpus.num_docs=0` scores the full corpus.

## Compare Bits per Byte

The `eval_init` step needs a GPU and reads a local corpus by default: a JSON Lines file with `text_field`, or a text file with one document per line.

1. Prepare held-out target-language text at `data_file`, and a second file in English or the base language if you want a regression check.

1. Score the resized checkpoints against the base model.

   ```console
   $ uv run nemotron steps run tokenizer_extension/eval_init -c default \
       models=[./output/resized_checkpoint,./output/resized_checkpoint_replace] \
       data_file=./data/eval/target_val.jsonl max_docs=2000 \
       output_json=./output/eval/bpb_target.json
   ```

   `base_model` defaults to `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` and is scored first as the reference; the report includes the delta for each model.

1. Repeat with the English or base-language file to check for regression.

1. Read `bpb`, not `loss` or `perplexity`, when the models carry different tokenizers.

### Keep the Comparison Valid

Always bound the run with `max_docs` (`-1` scores the whole corpus).
If more than one model would be scored under `max_tokens` with no `max_docs`, the step refuses to run because each tokenizer would stop after a different amount of text.
Setting `allow_token_cap_comparison: true` overrides the refusal and exists only to reproduce a historical result.

## Next Steps

- Field reference: {doc}`../reference/evaluate-config` and {doc}`../reference/eval-init-config`
- Output file layout: {doc}`../reference/outputs`
