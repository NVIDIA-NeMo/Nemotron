---
name: nemotron-optimizer-quantization
description: Configure Nemotron optimize/modelopt/quantize for post-training quantization with NVIDIA ModelOpt and Megatron-Bridge. Use for FP8 on Hopper, NVFP4 on Blackwell, calibration sizing, quantization recipes, export checks, and Megatron checkpoint output.
---

# ModelOpt Quantization

Use `optimize/modelopt/quantize` for post-training quantization of Hugging Face checkpoints into Megatron distributed output.

Before changing configs or code, read `step.toml` to understand the step flow, consumed and produced artifacts, important parameters, strategies, failure modes, and upstream references.

## Inputs And Outputs

- Consume `checkpoint_hf`.
- Produce quantized `checkpoint_megatron`.
- Validate export and checkpoint writing with a short calibration run before quality evaluation.

## Configure

- Set `args.hf_model_id` to the HF checkpoint to quantize; merge LoRA inputs
  before this step.
- Use `config/fp8.yaml` for Hopper or H100 targets when FP8 is the intended serving format.
- Use `config/nvfp4.yaml` for Blackwell or B200 targets when NVFP4 is supported by the model and serving stack.
- Set `args.export_quant_cfg` to a value accepted by the installed upstream script: `int8_sq`, `fp8`, `fp8_blockwise`, `int4_awq`, `w4a8_awq`, or `nvfp4`.
- Set `args.calib_size` high enough for representative activation ranges.
- Keep `args.megatron_save_path` outside the source checkpoint path.
- Use `extra_args` for new upstream ModelOpt or Megatron-Bridge flags.

## Config Nuances

- Use the base quantization config for quick launch validation, `fp8` for Hopper FP8, and `nvfp4` only on Blackwell-ready resources.
- Prefer a smaller BF16 source checkpoint such as `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` for launch validation; switch `args.hf_model_id` to larger production checkpoints only after the path is proven.
- Treat very small `args.calib_size` values as launch validation only; increase calibration with representative prompts before evaluating quality.
- Keep `args.megatron_save_path` separate from the input checkpoint path so failed exports do not corrupt the source artifact.
- Avoid model-specific recipe names unless the container's `quantize.py --help` lists them.
- Keep `torchrun.nproc_per_node * torchrun.nnodes` divisible by `args.tp * args.pp`; adjust `args.pp` down before changing the launcher shape.
- For Mamba/Megatron-Bridge models, ensure Mamba group counts are divisible by tensor parallel size.
- Keep quantization and pruning W&B logging through the wrapper config; upstream quantize does not expose the native W&B flags that distill does.

## Local Files

- Contract: `src/nemotron/steps/optimize/modelopt/quantize/step.toml`
- Runner: `src/nemotron/steps/optimize/modelopt/quantize/step.py`
- Configs: `config/default.yaml`, `config/tiny.yaml`, `config/fp8.yaml`, `config/nvfp4.yaml`

## Guardrails

- Do not judge quality from tiny calibration runs.
- Export the produced Megatron checkpoint when downstream tools need Hugging Face layout.
- Run task eval and serving smoke tests after quantization and export.
