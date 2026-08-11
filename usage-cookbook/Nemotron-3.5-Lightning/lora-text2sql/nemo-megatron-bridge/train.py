#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LoRA fine-tune Nemotron-3.5 Lightning on the prepared Text2SQL data.

Everything model-specific -- layer config, LoRA target modules (including the Mamba
`in_proj`/`out_proj`), MoE settings, optimizer defaults -- comes from the shipped
Megatron-Bridge recipe. This script only:

  1. points the recipe at your local converted checkpoint, tokenizer, and dataset,
  2. sets the parallelism to match the GPUs you actually have,
  3. sizes the training schedule from the dataset.

Launch with torchrun; see the notebook.
"""

import json
import math
import os
import shutil

import torch

from megatron.bridge.recipes.nemotronh import nemotron_3_5_nano_peft_config as peft_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step


def _env(name, default=None, cast=str):
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"{name} must be set (see the notebook's configuration cell).")
    return cast(value)


# ----------------------------------------------------------------------------------
# Configuration (all from the environment, so the notebook is the single source of truth)
# ----------------------------------------------------------------------------------
HF_MODEL = _env("HF_MODEL")
MEGATRON_MODEL_PATH = _env("MEGATRON_MODEL_PATH")
DATASET_DIR = _env("DATASET_DIR")
TRAINING_OUTPUT_DIR = _env("TRAINING_OUTPUT_DIR")
EXPERIMENT_NAME = _env("EXPERIMENT_NAME", "lightning35-text2sql-lora")

N_DEVICES = _env("N_DEVICES", cast=int)

# Parallelism. Nemotron-3.5 Lightning is a Mixture-of-Experts model, so the cheapest way to
# split it across GPUs is expert parallelism (EP): each GPU holds a subset of the 128
# experts. Default to sharding experts across every GPU you have.
TP = _env("TP", "1", int)  # tensor parallel
PP = _env("PP", "1", int)  # pipeline parallel
EP = _env("EP", str(N_DEVICES), int)  # expert parallel

MAX_SEQ_LEN = _env("MAX_SEQ_LEN", "2048", int)
GLOBAL_BS = _env("GLOBAL_BS", "32", int)
MICRO_BS = _env("MICRO_BS", "1", int)
EPOCHS = _env("EPOCHS", "1", int)
LR = _env("LR", "1e-4", float)
MIN_LR = _env("MIN_LR", "1e-5", float)
LORA_RANK = _env("LORA_RANK", "32", int)
LORA_ALPHA = _env("LORA_ALPHA", str(LORA_RANK), int)
MAX_STEPS = os.environ.get("MAX_STEPS")  # optional cap, useful for smoke runs

# The recipe defaults to the "flex" dispatcher backed by DeepEP kernels, which are fast but
# depend on the DeepEP build matching your system. "alltoall" is the portable choice.
MOE_TOKEN_DISPATCHER_TYPE = _env("MOE_TOKEN_DISPATCHER_TYPE", "alltoall")

# Memory knobs, in the order you should reach for them when you run out of GPU memory.
# Recomputing activations trades compute for memory: activations are discarded on the forward
# pass and recomputed during backward. Costs roughly 20-30% throughput.
RECOMPUTE_ACTIVATIONS = _env("RECOMPUTE_ACTIVATIONS", "0") == "1"
# The recipe trains 2 multi-token-prediction heads (they power speculative decoding). Falling
# back to the checkpoint's own single head saves several GB and is what makes single-GPU
# training fit. Note: setting the head count to 0 does NOT work -- it fails the grad-buffer
# assertion below -- so this reduces to 1 rather than disabling MTP outright.
REDUCE_MTP_HEADS = _env("REDUCE_MTP_HEADS", "0") == "1"


def _count_packed_sequences(dataset_dir: str, seq_length: int) -> tuple[int, int]:
    """Estimate how many packed sequences the dataset yields.

    Training uses packed sequences: instead of padding every example out to the longest in
    its batch, examples are concatenated into dense `seq_length`-token blocks. Text2SQL
    examples vary a lot in length (a bare SQL answer vs. a long reasoning trace), so packing
    covers the same data in far fewer, fully-dense steps. dataprep.py wrote a per-row token
    `length`, so summing those tells us how many blocks to expect.
    """
    total_tokens = 0
    n_examples = 0
    with open(os.path.join(dataset_dir, "training.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_tokens += int(json.loads(line).get("length", 0))
            n_examples += 1
    return n_examples, max(1, math.ceil(total_tokens / seq_length))


def build_config():
    # The recipe carries every model-specific choice; we only override what's local to us.
    cfg = peft_config("lora")

    # --- Point at our checkpoint / tokenizer -------------------------------------------
    # The recipe hardcodes the public Hub ID, which we override with whatever HF_MODEL is
    # (a local path or a Hub ID).
    cfg.model.hf_model_id = HF_MODEL
    cfg.tokenizer.tokenizer_model = HF_MODEL
    cfg.checkpoint.pretrained_checkpoint = MEGATRON_MODEL_PATH

    # --- Parallelism -------------------------------------------------------------------
    cfg.model.tensor_model_parallel_size = TP
    cfg.model.pipeline_model_parallel_size = PP
    cfg.model.expert_model_parallel_size = EP
    cfg.model.seq_length = MAX_SEQ_LEN

    # --- Data --------------------------------------------------------------------------
    # The recipe ships pointed at a Hugging Face dataset. A dataset config accepts exactly
    # ONE source, so pointing it at our local training.jsonl means clearing the HF source
    # (and its split/materialization settings) as well -- otherwise validation fails with
    # "Exactly one text-only SFT source must be set".
    cfg.dataset.hf_dataset = None
    cfg.dataset.hf_validation_dataset = None
    cfg.dataset.hf_test_dataset = None
    cfg.dataset.hf_output_root = None
    cfg.dataset.hf_validation_proportion = None
    cfg.dataset.hf_rewrite = False

    cfg.dataset.dataset_root = DATASET_DIR
    cfg.dataset.seq_length = MAX_SEQ_LEN
    cfg.dataset.do_validation = False  # dataprep only produces training.jsonl
    cfg.dataset.do_test = False
    if cfg.dataset.offline_packing_specs is not None:
        cfg.dataset.offline_packing_specs.packed_sequence_size = MAX_SEQ_LEN

    # --- Schedule ----------------------------------------------------------------------
    n_examples, packed_seqs = _count_packed_sequences(DATASET_DIR, MAX_SEQ_LEN)
    train_iters = max(1, math.ceil(packed_seqs / GLOBAL_BS) * EPOCHS)
    if MAX_STEPS is not None:
        train_iters = min(train_iters, int(MAX_STEPS))

    cfg.train.train_iters = train_iters
    cfg.train.global_batch_size = GLOBAL_BS
    cfg.train.micro_batch_size = MICRO_BS
    cfg.scheduler.lr_warmup_iters = max(1, train_iters // 10)
    cfg.scheduler.lr_decay_iters = train_iters
    cfg.optimizer.lr = LR
    cfg.optimizer.min_lr = MIN_LR

    # --- LoRA --------------------------------------------------------------------------
    # Target modules come from the recipe (they include the Mamba projections); only the
    # capacity knobs are exposed here.
    cfg.peft.dim = LORA_RANK
    cfg.peft.alpha = LORA_ALPHA

    # --- MoE dispatcher ----------------------------------------------------------------
    cfg.model.moe_token_dispatcher_type = MOE_TOKEN_DISPATCHER_TYPE
    if MOE_TOKEN_DISPATCHER_TYPE != "flex":
        cfg.model.moe_flex_dispatcher_backend = None

    # --- Memory knobs ------------------------------------------------------------------
    if RECOMPUTE_ACTIVATIONS:
        cfg.model.recompute_granularity = "full"
        cfg.model.recompute_method = "uniform"
        cfg.model.recompute_num_layers = 1
    if REDUCE_MTP_HEADS:
        # None -> fall back to the checkpoint's own head count (1). The hybrid model asserts
        # mtp_num_layers > 0, so this cannot go to zero.
        cfg.model.mtp_num_layers = None

    # --- Output ------------------------------------------------------------------------
    run_dir = os.path.join(TRAINING_OUTPUT_DIR, EXPERIMENT_NAME)
    cfg.checkpoint.save = run_dir
    cfg.checkpoint.save_interval = train_iters  # one checkpoint at the end
    cfg.checkpoint.async_save = False  # synchronous save; async can hang under some runtimes
    cfg.logger.log_interval = 1

    if int(os.environ.get("RANK", "0")) == 0:
        os.makedirs(run_dir, exist_ok=True)
        shutil.copy(os.path.abspath(__file__), run_dir)  # record what produced this run
        latest = os.path.join(TRAINING_OUTPUT_DIR, "latest")
        if os.path.islink(latest):
            os.remove(latest)
        os.symlink(run_dir, latest)
        print(
            f"[train] examples={n_examples} packed_seqs~{packed_seqs} train_iters={train_iters} "
            f"GBS={GLOBAL_BS} MBS={MICRO_BS} seq={MAX_SEQ_LEN} (TP{TP} PP{PP} EP{EP}) "
            f"lora_rank={LORA_RANK} -> {run_dir}",
            flush=True,
        )

    return cfg


def main() -> None:
    if "RANK" not in os.environ:
        raise RuntimeError("Launch this with torchrun, not `python train.py` (see the notebook).")

    cfg = build_config()
    if os.environ.get("DRY_RUN", "0") == "1":
        print("[train] DRY_RUN=1 -> config built OK, skipping finetune().", flush=True)
        return

    finetune(config=cfg, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# Guard so 'spawn' workers (e.g. during checkpoint save) re-import this module WITHOUT
# re-running training.
if __name__ == "__main__":
    main()
