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
"""LoRA fine-tune Nemotron-3.5 Super VL on the finance chat JSONL. Launched by `steps/train.sh`.

The shipped Megatron-Bridge recipe supplies everything model-specific: the LoRA target modules
(language-model attention, Mamba, and expert projections), the frozen vision tower and projector,
and the MoE and optimizer settings. This script points it at the checkpoint, the data, and your
GPU layout, and sizes the schedule from the data.
"""

import os

import torch
from megatron.bridge.data.builders import ChatSFTPreprocessingConfig, DirectHFSFTDatasetConfig, HFDatasetSourceConfig
from megatron.bridge.data.collators import registry
from megatron.bridge.recipes.nemotronh_multimodal import nemotron_35_super_vl_peft_config
from megatron.bridge.recipes.nemotronh_multimodal.h100 import nemotron_35_super_vl as recipe_module
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.nemotron_omni_step import forward_step
from PIL import Image
from transformers.dynamic_module_utils import get_class_from_dynamic_module

MODEL_DIR = os.environ["MODEL_DIR"]  # the downloaded Hugging Face checkpoint
WORKSPACE = os.environ["WORKSPACE"]
TRAIN_JSONL = os.path.join(WORKSPACE, "dataprep", "train.jsonl")
RUN_DIR = os.path.join(WORKSPACE, "trained", os.environ["EXPERIMENT_NAME"])

# Megatron-Bridge picks the batch collator by the processor's class name; register this
# checkpoint's processor with the Nemotron Omni collator it uses.
registry._MODEL_COLLATE_SPECS.setdefault(
    "NemotronH_Omni_Reasoning_V3Processor",
    registry._MODEL_COLLATE_SPECS["NemotronH_Nano_Omni_Reasoning_V3Processor"],
)


def accept_tensor_images():
    """The checkpoint's image processor expects PIL images; transformers hands it uint8 CHW tensors."""
    cls = get_class_from_dynamic_module(
        "image_processing_nemotron_h_omni.NemotronH_Omni_Reasoning_V3ImageProcessor", MODEL_DIR)
    original = cls._preprocess
    if getattr(original, "accepts_tensors", False):
        return

    def to_pil(image):
        if isinstance(image, (list, tuple)):
            return [to_pil(item) for item in image]
        if isinstance(image, torch.Tensor):
            if image.dtype != torch.uint8 or image.ndim != 3:
                raise TypeError(f"expected a uint8 CHW image tensor, got {image.dtype} {tuple(image.shape)}")
            return Image.fromarray(image.permute(1, 2, 0).cpu().numpy())
        return image

    def preprocess(self, images, *args, **kwargs):
        return original(self, to_pil(images), *args, **kwargs)

    preprocess.accepts_tensors = True
    cls._preprocess = preprocess


accept_tensor_images()


def dataset_config(seq_length: int) -> DirectHFSFTDatasetConfig:
    """Chat JSONL rendered by the model's own processor and chat template, loss on assistant turns."""
    return DirectHFSFTDatasetConfig(
        seq_length=seq_length,
        preprocessing=ChatSFTPreprocessingConfig(loss_mode="assistant"),
        hf_processor_path=MODEL_DIR,
        trust_remote_code=True,
        source=HFDatasetSourceConfig(path_or_dataset="json", split="train",
                                     load_kwargs={"data_files": {"train": TRAIN_JSONL}}),
        do_validation=False,
        do_test=False,
        dataloader_type="cyclic",
    )


def build_config():
    # The recipe reads the model config from its Hub id while it builds; point it at the download.
    recipe_module.NEMOTRON_35_SUPER_VL_HF_MODEL_ID = MODEL_DIR
    recipe_module.NEMOTRON_35_SUPER_VL_HF_REVISION = None
    cfg = nemotron_35_super_vl_peft_config("lora")

    cfg.model.hf_model_id = MODEL_DIR
    cfg.checkpoint.pretrained_checkpoint = os.path.join(WORKSPACE, "base")
    cfg.dataset = dataset_config(cfg.model.seq_length)

    cfg.model.tensor_model_parallel_size = int(os.environ["TRAIN_TP"])
    cfg.model.pipeline_model_parallel_size = int(os.environ["TRAIN_PP"])
    cfg.model.expert_model_parallel_size = int(os.environ["TRAIN_EP"])
    if cfg.model.pipeline_model_parallel_size == 1:  # the recipe's stage split is for two stages
        cfg.model.num_layers_in_first_pipeline_stage = None
        cfg.model.num_layers_in_last_pipeline_stage = None

    with open(TRAIN_JSONL) as f:
        rows = sum(1 for _ in f)
    iters = -(-rows // cfg.train.global_batch_size) * int(os.environ["EPOCHS"])
    iters = min(iters, int(os.environ.get("MAX_STEPS") or iters))
    cfg.train.train_iters = iters
    cfg.scheduler.lr_warmup_iters = max(1, iters // 10)
    cfg.scheduler.lr_decay_iters = iters
    cfg.validation.eval_interval = 0

    cfg.checkpoint.save = RUN_DIR
    cfg.checkpoint.save_interval = iters
    cfg.checkpoint.async_save = False
    cfg.logger.log_interval = 1

    if os.environ.get("RANK", "0") == "0":
        print(f"{rows} training rows -> {iters} steps of {cfg.train.global_batch_size} examples, "
              f"sequence length {cfg.model.seq_length}, learning rate {cfg.optimizer.lr}, "
              f"LoRA rank {cfg.peft.dim}, layout TP{cfg.model.tensor_model_parallel_size} "
              f"PP{cfg.model.pipeline_model_parallel_size} EP{cfg.model.expert_model_parallel_size}", flush=True)
    return cfg


if __name__ == "__main__":
    if os.path.exists(os.path.join(RUN_DIR, "latest_checkpointed_iteration.txt")):
        raise SystemExit(f"{RUN_DIR} already holds a trained adapter; set a new EXPERIMENT_NAME.")
    finetune(config=build_config(), forward_step_func=forward_step)
