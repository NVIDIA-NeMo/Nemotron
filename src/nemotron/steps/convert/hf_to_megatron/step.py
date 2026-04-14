#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/convert/hf_to_megatron"
#
# [tool.runspec.run]
# launch = "python"
# ///
"""HF -> Megatron pattern. Real script: Megatron-Bridge/examples/conversion/convert_checkpoints.py"""
import torch
from megatron.bridge import AutoBridge


def main() -> None:
    hf_model = "nvidia/Nemotron-3-Nano-30B-A3B"
    megatron_path = "/path/to/megatron_ckpt"
    AutoBridge.import_ckpt(hf_model_id=hf_model, megatron_path=megatron_path, torch_dtype=torch.bfloat16)


if __name__ == "__main__":
    main()
