---
name: nemotron-3-5-lightning-text2sql-lora
description: >-
  Run the Nemotron-3.5 Lightning Text2SQL LoRA fine-tuning tutorial (NeMo Megatron-Bridge) end-to-end for
  the user on a single node: data prep, checkpoint conversion, LoRA fine-tuning of the 30B-A3B hybrid
  Mamba-Transformer MoE, and merging the adapter back to a Hugging Face checkpoint. Use when the user
  wants to run this cookbook, fine-tune Nemotron-3.5 Lightning with LoRA, or adapt the notebook to their
  own machine.
---

# Nemotron-3.5 Lightning Text2SQL LoRA — runbook for a coding agent

This skill helps you run the cookbook in this directory (`mbridge_lora_cookbook.ipynb`) on the
user's behalf. Your job is to gather a few environment details, pick a GPU configuration that fits
their hardware, run the four steps in order, and confirm each one produced what it should.

## What the tutorial does

Four steps, in order. **Only step 3 needs a GPU** — this is the single most useful thing to know
when planning the run.

1. **Data prep** (CPU, ~2 min) — builds a BIRD Text2SQL `training.jsonl` from the no-reasoning and
   reasoning splits, formatted with Nemotron-3.5's chat template.
2. **Convert** (CPU, ~4 min) — imports the Hugging Face checkpoint into Megatron-Bridge format.
3. **LoRA fine-tune** (GPU) — packed-sequence LoRA training; saves an adapter.
4. **Merge & export** (CPU) — merges the adapter into the base weights, writing a standard Hugging
   Face checkpoint.

## Information to gather from the user

Ask for these up front, in one batch:

- **Path to the Nemotron-3.5 Lightning checkpoint** (already downloaded), or confirmation that you should
  download it and where to put it. It is ~62 GB.
- **How many GPUs** they want to use, and what kind. Drives the whole training config.
- **Where to write outputs** — needs ~130 GB free.
- **A Hugging Face token** (`$HF_TOKEN`) so BIRD can be downloaded during data prep. Reference it by
  environment variable; never print it.
- **The container image** to use, if it differs from the one in the notebook.

## Choosing the GPU configuration

The model's 128 experts are split across GPUs with expert parallelism, so `n_devices` is the only
knob that really matters — set `EP = n_devices`. Measured peak memory per GPU on 80GB H100s at
`seq_length=2048`:

| GPUs | Peak/GPU | One epoch | Recommendation |
| --- | --- | --- | --- |
| 1 | 78.8 GB | ~62 min | Works only with `REDUCE_MTP_HEADS=1`. ~0.4 GB margin — fine if that is all they have. |
| 2 | 51.0 GB | ~34 min | **Default to this when available.** Stock recipe, ~28 GB margin. |
| 4 | 34.8 GB | ~18 min | Good if available. |
| 8 | 26.8 GB | ~8 min | Fastest. |

All measured over a full epoch (189 iterations, GBS 32, `seq_length=2048`) on the complete
12,544-example dataset. Final loss lands within ~2% across all four, so choose on hardware
availability and how long the user is willing to wait — not on expected quality.

If the user has GPUs smaller than 80 GB, scale by the same logic: peak memory is roughly
(model weights ÷ EP) + ~12 GB of overhead. Spare memory is best spent raising `seq_length`, which
increases how much of the dataset survives the length filter — not just headroom.

## How to run it

1. Launch the container with the notebook directory and the checkpoint path mounted, and `$HF_TOKEN`
   exposed. Use the `docker run` invocation in the notebook's first cell as the template.
2. Fill in the notebook's **Configuration** cell (paths, `n_devices`) — it is the only cell that
   should need editing.
3. Run the steps in order. After each, run its sanity-check cell before moving on.
4. Training is the long step. Run it in the background and poll; do not hold a blocking session
   open, and do not stream the full log.

You can also run the steps directly rather than through the notebook — each is a plain script driven
by environment variables (`MODEL_ID`, `MAX_SEQ_LEN`, `DATAPREP_OUTPUT_DIR` for data prep;
`HF_MODEL`, `MEGATRON_MODEL_PATH` for convert; and `N_DEVICES`, `EP`, `DATASET_DIR`,
`TRAINING_OUTPUT_DIR`, `EXPERIMENT_NAME` for training).

## Verifying success per step

- **Data prep:** `$DATAPREP_OUTPUT_DIR/training.jsonl` exists with ~12,500 rows at `seq_length=2048`.
  Spot-check one record: `input` should end with `<think>\n` (reasoning) or `<think></think>`
  (non-reasoning), and `output` should continue directly from there.
- **Convert:** `$MEGATRON_MODEL_PATH/latest_checkpointed_iteration.txt` plus an `iter_*` directory
  exist (~62 GB).
- **Train:** an `iter_*` adapter checkpoint under `$TRAINING_OUTPUT_DIR/$EXPERIMENT_NAME`, and the
  log shows `lm loss` trending down.
- **Merge:** the output directory contains `model-*.safetensors` shards, `config.json`, and the
  tokenizer files, and the log ends with `Success: All tensors from the original checkpoint were
  written.`

Report per-step status, wall-clock time, and the final training loss.

## Things already handled — do not change them

- **The recipe supplies everything model-specific.** `train.py` calls
  the shipped PEFT recipe and overrides only local paths, parallelism, dataset, and
  schedule. Don't hand-write LoRA target modules — the recipe's already cover the Mamba
  projections, attention, and both routed and shared experts.
- **The MoE dispatcher is set to `alltoall`** rather than the recipe's default `flex`/DeepEP, for
  portability. Only change this if DeepEP is known good on the user's system.
- **Synchronous checkpoint saving** (`async_save=False`) is deliberate.
- **Steps are idempotent:** data prep skips if `training.jsonl` exists; convert skips if the
  checkpoint exists.

## Expected friction (so you don't misread it)

- **The first training iteration takes 1–2 minutes** with no output while CUDA graphs are captured
  and the MoE warms up. Subsequent iterations are seconds. Do not cancel the job.
- **Startup log noise is not failure.** `Failed to import Triton kernels`, `MimoModelConfig is
  experimental`, `Unable to import torchao`, and `torch_dtype is deprecated` all appear on healthy
  runs. Judge by the sanity checks.
- **Do not enable `RECOMPUTE_ACTIVATIONS`.** It lowers memory but fails at iteration 2 with an
  assertion in Megatron's gradient buffer. If the user is out of memory, add a GPU or lower
  `seq_length` instead.
- **`REDUCE_MTP_HEADS` reduces to one head, it cannot disable MTP.** The hybrid model asserts
  `mtp_num_layers > 0`.
- **If you point the recipe at local data**, you must also clear its Hugging Face dataset fields —
  a dataset config accepts exactly one source. `train.py` already does this; preserve it if you
  refactor.
- **Serve with vLLM, not Transformers.** vLLM supports this architecture natively and works.
  `transformers.generate()` currently fails inside the model's bundled remote code — on the *base*
  checkpoint too, so don't diagnose it as a fine-tuning problem.
- **Any vLLM script needs an `if __name__ == "__main__":` guard.** vLLM spawns workers; without it
  the failure surfaces as `Engine core initialization failed` wrapping a `multiprocessing`
  bootstrap error that never mentions vLLM.

## What success looks like at the end

Serving the merged checkpoint and prompting it the way data prep formatted training examples should
yield bare SQL, e.g. `SELECT T2.dept_name FROM employees AS T1 INNER JOIN ...`. The base model
instead answers conversationally with fenced SQL and a prose explanation. If the fine-tuned model
still explains itself, something upstream went wrong — suspect the chat-template format first.
