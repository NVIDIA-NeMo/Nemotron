---
name: nemotron-3-ultra-text2sql-lora
description: >-
  Run the Nemotron-3 Ultra Text2SQL LoRA fine-tuning tutorial (NeMo Megatron-Bridge) end-to-end
  for the user on their SLURM cluster: data prep, distributed checkpoint conversion, and packed
  LoRA fine-tuning of the 550B hybrid Mamba-Transformer MoE, ending at a saved adapter. Use when
  the user wants to run this cookbook, fine-tune Nemotron-3 Ultra with LoRA, or adapt the notebook
  to their own cluster.
---

# Nemotron-3 Ultra Text2SQL LoRA — runbook for a coding agent

This skill helps you run the cookbook in this directory (`mbridge_lora_cookbook.ipynb`) on the
user's behalf. The notebook is generic and ships with placeholders; your job is to gather the
user's environment details, fill them in, launch the SLURM jobs, watch them, and report results.

## What the tutorial does

Three steps, in order, each a SLURM job:

1. **Data prep** — builds a BIRD Text2SQL `training.jsonl` from both the no-reasoning and reasoning
   splits, formatted with Ultra's tokenizer/chat template. Short CPU job.
2. **Convert** — distributed import of the Hugging Face base checkpoint into Megatron-Bridge format.
   A multi-node GPU job (CPU import is not feasible for a 550B model).
3. **LoRA fine-tune** — packed-sequence LoRA training on the prepared data; saves a LoRA adapter.
   A multi-node GPU job.

## What you must understand before running

- **Ultra is a 550B-total / A55B-active hybrid Mamba-Transformer MoE.** It does **not** fit on one
  node, so every heavy step is a **multi-node SLURM job** submitted with `sbatch` and run in a
  container via Pyxis/enroot. Run everything from a cluster **login node** where `sbatch`/`squeue`/
  `sacct` are available.
- **Scale.** At the shipped parallel settings, both convert and train need **48 GPUs**. Node count
  is derived automatically as `48 / GPUS_PER_NODE` (e.g. 12 nodes at 4 GPUs/node). The user's QOS
  must permit a job of that size — an interactive or small-node-capped QOS will not work.
- **Single config.** Everything is driven by one file, `config.env`, which the notebook's setup
  cell generates from the values you fill in. Every step and every `slurm/*.sbatch` script sources
  it. You can run the notebook cell, or write `config.env` directly with the same keys.
- **One output root.** `WORKSPACE` is the single output root; everything generated lands under
  `$WORKSPACE/{base, dataprep, trained, cache/hf, logs}`. The base checkpoint (`HF_MODEL_PATH`) is
  the only separate, read-only path.
- **The rhythm per step:** a **launch** cell submits the job, a re-runnable **check** cell shows
  status (`sacct`/`squeue`), and a **sanity** cell confirms the expected output exists before you
  move on. Follow this loop; don't skip the sanity check.

## Information to gather from the user

Before launching anything, ask the user for the following and confirm the prerequisites. Don't
guess these — a wrong value wastes a large multi-node allocation. Prefer asking all of them up front
in one batch.

**How to reach the cluster**
- How do you connect to the login node where SLURM jobs are submitted (e.g. the ssh host)?
- Is there a separate data-transfer host you prefer for large file moves?

**SLURM settings**
- SLURM **account** to charge.
- **GPU partition** and a **QOS** that allows a multi-node job of `48 / GPUS_PER_NODE` nodes (not an
  interactive or small-node-capped QOS). Confirm the wall-clock limit is enough (convert is short;
  training is well under a couple of hours by default).
- **CPU partition** and **QOS** for the short data-prep job.
- **GPUs per node** on the target nodes (the tutorial targets GB200 at 4 GPUs/node; the node count
  derives from this).

**Paths (all on a shared filesystem the compute nodes can mount)**
- **`WORKSPACE`** — the output root to create/use.
- **`HF_MODEL_PATH`** — where the **already-downloaded** Ultra base checkpoint lives (read-only
  input). The tutorial does **not** download the base model; confirm it is present.
- The **shared-filesystem root** to bind-mount into the container (must contain both `WORKSPACE`
  and `HF_MODEL_PATH`).

**Container & credentials**
- The **container image** to use (path to a prepared image or a registry reference). The notebook
  ships a placeholder; this must be filled with a real Ultra-capable image.
- A **Hugging Face token** so BIRD can be downloaded during data prep. The tutorial expects it at
  `${WORKSPACE}/cache/hf/token`; ask the user to place it there (or provide it so you can), and
  reference it by path — never print or echo a token.

If the user has an environment-reference document for their cluster, ask for it first and pull these
values from there instead of asking one by one.

## How to run it

1. From the login node, `cd` into this cookbook directory (it must be on the shared filesystem).
2. Fill the config: either edit the notebook's **Environment & SLURM Setup** cell and run it, or
   write `config.env` directly with the values gathered above. The setup cell has a guard that
   refuses to proceed while any placeholder (`<...>`) remains — make sure none are left.
3. Run the three steps in order. For each: submit via the launch cell/`sbatch`, **poll** the check
   cell until the job reaches `COMPLETED`, then run the sanity cell.
4. **Poll, don't block.** These are long-running multi-node jobs. Submit, then check back
   periodically with `sacct`/`squeue` — do not hold an interactive session open waiting, and do not
   stream logs live.

## Verifying success per step

- **Data prep:** `$WORKSPACE/dataprep/training.jsonl` exists and has many rows; a sampled record
  shows the Nemotron-3 chat template.
- **Convert:** `$WORKSPACE/base/latest_checkpointed_iteration.txt` plus an `iter_*` checkpoint dir
  exist.
- **Train:** under `$WORKSPACE/trained/<experiment-name>/` there is a
  `latest_checkpointed_iteration.txt` and an `iter_*` adapter checkpoint; the training log shows the
  loss trending down and ends with a `LORA_TRAIN_DONE` marker.

Report per-step status and elapsed time (from `sacct`) and the final training loss.

## Things already handled — do not change them

- **Synchronous checkpoint saving** is set on purpose (`async_save=False`). Under some container
  runtimes the async-save path can hang; leave it as configured.
- **Parallelism / resharding.** Convert uses one tensor-parallel layout and train uses another; only
  the tensor-parallel degree differs, so the converted checkpoint reshards cleanly on load. Don't
  retune these unless you change `GPUS_PER_NODE`, in which case keep the world size at 48 GPUs.
- **Packed sequences** and the **LoRA target modules** (including the Mamba projections) come from
  the Ultra recipe — no need to configure them.
- **Steps are idempotent:** data prep skips if `training.jsonl` exists; convert skips if the
  checkpoint already exists. Safe to re-run.

## Expected friction (so you don't misread it)

- **The first training iteration is slow** — graph capture and MoE warmup can take on the order of
  ~15 minutes with no log output and the GPUs at 100%. This is normal; do not cancel the job. Later
  iterations are fast.
- A multi-node GPU job that, in the rare case, sits at "loading distributed checkpoint" with zero
  progress for far longer than the warmup window can be cancelled and resubmitted; a fresh
  allocation usually clears it.
- Benign noise in the convert log (framework stack-trace fragments, bare NCCL version lines) is not
  a crash — judge success by the job state and the sanity check, not by log chatter.
