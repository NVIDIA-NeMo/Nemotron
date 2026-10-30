#!/usr/bin/env python3
# Nemotron-3 Ultra LoRA fine-tune on a local text2sql dataset, using PACKED sequences.
#
# run_recipe.py's --dataset llm-finetune-preloaded builds an UNPACKED FinetuningDatasetConfig
# and ignores --packed_sequence, so for packed + local data we build the config directly here:
# the Ultra PEFT recipe supplies the model/parallelism/LoRA config, and we swap in a packed
# FinetuningDatasetConfig pointed at our local training.jsonl (the FinetuningDatasetBuilder packs
# it at prep time -- no HF download). Launch via torchrun/srun.
import json
import math
import os

import torch

from megatron.bridge.recipes.nemotronh.nemotron_3_ultra import (
    nemotron_3_ultra_peft_openmathinstruct2_packed_config as ultra_peft_config,
)
from megatron.bridge.training.config import FinetuningDatasetConfig
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step


def _env(name, default=None, cast=str):
    v = os.environ.get(name, default)
    if v is None:
        raise RuntimeError(f"{name} must be set")
    return cast(v)


def build_config():
    dataset_dir = _env("DATASET_DIR")
    megatron_model_path = _env("MEGATRON_MODEL_PATH")
    hf_model_path = _env("HF_MODEL_PATH")
    save_dir = _env("SAVE_DIR")
    seq = _env("MAX_SEQ_LEN", "4096", int)
    gbs = _env("GLOBAL_BS", "32", int)
    mbs = _env("MICRO_BS", "1", int)
    epochs = _env("EPOCHS", "1", int)
    tp = _env("TRAIN_TP", "2", int)
    pp = _env("TRAIN_PP", "6", int)
    ep = _env("TRAIN_EP", "8", int)
    etp = _env("TRAIN_ETP", "1", int)
    cp = _env("TRAIN_CP", "1", int)
    lr = _env("LR", "1e-4", float)
    min_lr = _env("MIN_LR", "1e-5", float)

    # Estimate packed-sequence count from summed token lengths (dataprep wrote a "length" field)
    # so we can size train_iters for ~EPOCHS passes. Packing fills SEQ-length blocks.
    total_tokens = 0
    n = 0
    with open(os.path.join(dataset_dir, "training.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_tokens += int(json.loads(line).get("length", 0))
            n += 1
    packed_seqs = max(1, math.ceil(total_tokens / seq))
    train_iters = max(1, math.ceil(packed_seqs / gbs) * epochs)
    warmup = max(1, train_iters // 10)

    cfg = ultra_peft_config(peft="lora", hf_path=hf_model_path, seq_length=seq)

    # Swap the recipe's (OpenMath HF) dataset for our LOCAL packed dataset.
    cfg.dataset = FinetuningDatasetConfig(
        dataset_root=dataset_dir,
        seq_length=seq,
        seed=1234,
        num_workers=8,
        pin_memory=True,
        do_validation=False,
        do_test=False,
        packed_sequence_specs=PackedSequenceSpecs(packed_sequence_size=seq),
    )

    cfg.model.tensor_model_parallel_size = tp
    cfg.model.pipeline_model_parallel_size = pp
    cfg.model.expert_model_parallel_size = ep
    cfg.model.expert_tensor_parallel_size = etp
    cfg.model.context_parallel_size = cp
    cfg.model.seq_length = seq

    cfg.checkpoint.pretrained_checkpoint = megatron_model_path
    cfg.checkpoint.save = save_dir
    cfg.checkpoint.save_interval = train_iters
    cfg.checkpoint.async_save = False  # required: async-save CUDA-IPC (pidfd_getfd) is blocked by enroot

    cfg.train.train_iters = train_iters
    cfg.train.global_batch_size = gbs
    cfg.train.micro_batch_size = mbs
    cfg.validation.eval_interval = train_iters
    cfg.scheduler.lr_warmup_iters = warmup
    cfg.scheduler.lr_decay_iters = train_iters
    cfg.optimizer.lr = lr
    cfg.optimizer.min_lr = min_lr
    cfg.logger.log_interval = 1

    print(
        f"[packed] examples={n} total_tokens={total_tokens} packed_seqs~{packed_seqs} "
        f"train_iters={train_iters} GBS={gbs} (TP{tp} PP{pp} EP{ep}) seq={seq}",
        flush=True,
    )
    return cfg


def main():
    cfg = build_config()
    if os.environ.get("DRY_RUN", "0") == "1":
        print("[packed] DRY_RUN=1 -> config built OK, skipping finetune().", flush=True)
        return
    finetune(config=cfg, forward_step_func=forward_step)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# Guard so multiprocessing 'spawn' workers (e.g. during checkpoint save) re-import this module
# WITHOUT re-running training -- otherwise the child re-calls finetune() and hits EADDRINUSE.
if __name__ == "__main__":
    main()
