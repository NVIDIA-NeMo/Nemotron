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
"""Import the Hugging Face checkpoint into Megatron-Bridge format.

Megatron-Bridge trains from its own checkpoint format, so this one-time step converts the
HF weights before training. The checkpoint is written as `torch_dist`, which reshards on
load -- so you convert once here (on CPU, no model parallelism) and can then train at any
TP/PP/EP layout without re-converting.

Run once:

    HF_MODEL=<hf id or local path> MEGATRON_MODEL_PATH=<out dir> python convert.py

Idempotent: exits early if the target checkpoint already exists.
"""

import os
import sys

import torch

from megatron.bridge import AutoBridge


def main() -> None:
    for var in ("HF_MODEL", "MEGATRON_MODEL_PATH"):
        if var not in os.environ:
            raise RuntimeError(f"{var} must be set (see the notebook's configuration cell).")

    hf_model = os.environ["HF_MODEL"]
    megatron_path = os.environ["MEGATRON_MODEL_PATH"]

    # torch_dist checkpoints record their iteration here; its presence means we already converted.
    if os.path.exists(os.path.join(megatron_path, "latest_checkpointed_iteration.txt")):
        print(f"Megatron checkpoint already present at '{megatron_path}'. Nothing to do.")
        sys.exit(0)

    os.makedirs(megatron_path, exist_ok=True)

    print(f"Importing '{hf_model}' -> '{megatron_path}' (bf16, CPU init)", flush=True)
    AutoBridge.import_ckpt(
        hf_model,
        megatron_path,
        torch_dtype=torch.bfloat16,
        # Harmless for the GA checkpoint, which relies on native Transformers support. Kept
        # because pre-release builds shipped modeling code in the repo (config.json `auto_map`),
        # and because it is still required for other Nemotron checkpoints.
        trust_remote_code=True,
    )
    print("CONVERT_DONE", flush=True)


if __name__ == "__main__":
    main()
