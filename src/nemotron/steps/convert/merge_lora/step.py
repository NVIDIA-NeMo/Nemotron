#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/convert/merge_lora"
#
# [tool.runspec.run]
# launch = "python"
# ///
"""Merge LoRA into a base model. Real script: Megatron-Bridge/examples/peft/merge_lora.py"""
from subprocess import run


def main() -> None:
    run(
        ["python", "Megatron-Bridge/examples/peft/merge_lora.py", "--lora-checkpoint", "/path/to/lora_ckpt",
         "--hf-model-path", "nvidia/Nemotron-3-Nano-30B-A3B", "--output", "/path/to/merged_checkpoint",
         "--pretrained", "/path/to/base_ckpt", "--cpu"],
        check=True,
    )


if __name__ == "__main__":
    main()
