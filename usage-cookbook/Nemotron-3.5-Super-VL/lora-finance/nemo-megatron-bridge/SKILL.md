---
name: nemotron-3-5-super-vl-finance-lora
description: >-
  Run the Nemotron-3.5 Super VL finance LoRA cookbook (NeMo Megatron-Bridge) end-to-end for the
  user, on one 8-GPU node or a Slurm cluster: data prep, checkpoint conversion, text+vision LoRA
  fine-tuning of the 120B-A12B model, merge to a Hugging Face checkpoint, and a vLLM try-out. Use
  when the user wants to run this cookbook, fine-tune Nemotron-3.5 Super VL with LoRA, or adapt it
  to their hardware.
---

# Nemotron-3.5 Super VL finance LoRA — runbook for a coding agent

This skill helps you run the cookbook in this directory on the user's behalf: gather their
environment details, fill in `config.env`, run the steps, check each one, and report results.

## What the cookbook does

Six steps, in order. Each is one script in `steps/` that runs inside the NeMo container and reads
`config.env`; everything is written under `WORKSPACE`.

| Step | Script | Needs | Takes (H200) |
| --- | --- | --- | --- |
| Download the checkpoint | `steps/download.sh` | CPU, network | ~10 min |
| Build the training data | `steps/dataprep.sh` | CPU, network | ~10–15 min |
| Convert to Megatron-Bridge | `steps/convert.sh` | 4 GPUs | ~30 min |
| LoRA fine-tuning | `steps/train.sh` | 8 GPUs | ~1.75 h + 5–12 min startup |
| Merge and export to HF | `steps/merge.sh` | 4 GPUs | ~15–25 min |
| Try with vLLM | `steps/try.sh`, then `steps/try.sh base` | 8 GPUs | 5–50 min each, mostly loading |

There are two ways to run them; both use the same scripts:

- **One 8-GPU node:** inside the container (the notebook's first cell has the `docker run` line),
  run `bash steps/<step>.sh`, or run the notebook cells.
- **Slurm with Pyxis:** from the login node, in this directory,
  `sbatch <resources> slurm/step.sbatch <step>`. The README lists the resources for each step.
  Submit the next step only after the previous job completes.

## Information to gather from the user

Ask in one batch, before running anything. Don't guess: a wrong value wastes hours of GPU time.

- **Where to run:** one 8-GPU node, or a Slurm cluster with Pyxis? For Slurm: how to reach the login
  node, and the `--account`, `--partition`, and `--time` to use.
- **GPUs:** type and count per node. This picks the training layout below.
- **`WORKSPACE`:** an absolute path with about 800 GB free. On Slurm it, and this directory, must be
  on a filesystem every node can see.
- **Container:** the NeMo container to use (`CONTAINER_IMAGE`), if not the one in `config.env`.
- **Hugging Face token** with access to the model, exported as `HF_TOKEN`. Never print it.

Then edit `config.env`. For the standard run it is the only file you should change.

## Pick the training layout

Choose a measured layout from the README's hardware table and set `GPUS_PER_NODE`, `TRAIN_TP`,
`TRAIN_PP`, `TRAIN_EP` in `config.env`; on Slurm, request the same GPUs per node
(`--gpus-per-node`) and `--nodes=2` for the two-node layout. Two GPUs run out of memory. The
try-out was verified on 8 GPUs only.

Smoke-test training first: set `MAX_STEPS=3` and `EXPERIMENT_NAME=smoke`, run `steps/train.sh`,
then restore both. In a full run, loss starts around 1–1.5, falls below 0.5 around step 35, and
ends around 0.05–0.1.

## Verifying success per step

- **Download:** `$WORKSPACE/model` holds 64 `*.safetensors` files: 63 `model-*-of-00063` shards
  plus `model-summary-idxs.safetensors`.
- **Data prep:** writes 22,222 rows to `train.jsonl` and 2,943 to `test.jsonl`; about 8,000 training
  rows carry an image.
- **Convert:** `$WORKSPACE/base/latest_checkpointed_iteration.txt` exists.
- **Train:** `$WORKSPACE/trained/$EXPERIMENT_NAME/latest_checkpointed_iteration.txt` and an `iter_*`
  directory exist.
- **Merge:** `$WORKSPACE/trained/$EXPERIMENT_NAME/merged_hf/` with the same 64 `*.safetensors` files.
- **Try:** prints `reference:` / `model:` pairs. The fine-tuned model answers in the reference's
  short format; the base model tends to explain.

On Slurm, also check the job state with `sacct -j <id>`, and the log in `slurm-<id>.log`.

## Things already handled — do not change them

- `train.py` takes everything model-specific from the shipped recipe
  (`nemotron_35_super_vl_peft_config`) and only sets paths, layout, and schedule.
- The three small workarounds at the top of `train.py` (collator registration, tensor-to-PIL image
  conversion, recipe Hub id) each fix a real failure.
- `steps/download.sh` loads the checkpoint's own model code once, so the multi-GPU steps don't race
  to write it. Run it before the GPU steps, even if the weights are already there.
- Every step after download reads the local copy in `$MODEL_DIR`, never the Hub ID.
- Image paths in `train.jsonl` are absolute: keep `WORKSPACE` at the same path everywhere.
- Convert is safe to re-run. Train refuses to overwrite an adapter: pick a new `EXPERIMENT_NAME`.

## Expected friction

- Harmless log noise: `Failed to import Triton kernels`, `MimoModelConfig is experimental`,
  `torchao`, `torch_dtype is deprecated`, `MuPDF error: cannot find page` (data prep),
  `Overwriting old incomplete / corrupted checkpoint` (convert), `Unrecognized mapping type for
  ...mtp...` (convert, merge), `(TP, PP) mismatch after resume` (train), `Ignoring Transformer
  Engine checkpoint key` (merge), `Unexpected gate/up projection names` (vLLM). Judge success by
  the checks above, not by grepping for `Error`. Logs contain carriage returns; pipe through
  `tr '\r' '\n'` before grepping.
- An out-of-memory error on one GPU can leave the others hanging. If the log shows
  `OutOfMemoryError`, stop the run and pick a lower-memory layout from the README: raising TP from
  2 to 4 cuts memory without adding GPUs.
- If one-node training fails before iteration 1 with an NCCL timeout and no OOM, retry once
  unchanged.
- On Slurm, if Pyxis is slow or runs out of memory importing the container on every job, import it
  once with `enroot import -o <file>.sqsh docker://<image>` and set `CONTAINER_IMAGE` to that file.
- For two nodes, both must be on the same InfiniBand island, or NCCL fails with
  `ibv_modify_qp ... Connection timed out`. If the scheduler doesn't enforce that, pin the nodes
  with `--nodelist`.
