#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/convert/megatron_to_hf"
#
# [tool.runspec.run]
# launch = "python"
# ///
"""Megatron -> HF pattern. Real script: Megatron-Bridge/examples/conversion/convert_checkpoints.py"""
from megatron.bridge import AutoBridge


def main() -> None:
    megatron_path = "/path/to/megatron_ckpt"
    hf_model = "nvidia/Nemotron-3-Nano-30B-A3B"
    hf_path = "/path/to/hf_safetensors"
    bridge = AutoBridge.from_auto_config(megatron_path, hf_model, trust_remote_code=True)
    bridge.export_ckpt(megatron_path=megatron_path, hf_path=hf_path)


if __name__ == "__main__":
    main()
