---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Command-line reference for the optimize/modelopt/quantize step."
topics: ["Training", "Reference", "CLI", "Optimization", "Quantization"]
tags: ["Reference", "CLI", "Steps", "Optimization", "Quantization", "ModelOpt", "Megatron-Bridge", "FP8", "NVFP4"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Developer"]
---

# optimize/modelopt/quantize

This step runs post-training quantization (PTQ) on a Hugging Face (HF) format checkpoint by using NVIDIA Model Optimizer through NVIDIA Megatron-Bridge.
The step supports every quantization recipe that the installed Megatron-Bridge quantization script accepts.
The step produces a quantized Megatron distributed checkpoint that you can export back to Hugging Face format with the upstream `export.py` script.

## Syntax

```bash
nemotron steps run optimize/modelopt/quantize \
    [-c <config-name-or-path>] \
    [-r <run-profile> | -b <batch-profile>] \
    [-d] \
    [--force-squash] \
    [<dotlist-overrides>...] \
    [<passthrough-args>...]
```

Refer to the [Nemotron Steps CLI Reference](../cli-reference.md) for the shared flag set.

## Configuration Files

The step ships four configuration files under `src/nemotron/steps/optimize/modelopt/quantize/config/`.

| File | Purpose |
| --- | --- |
| `default.yaml` | Generic post-training quantization configuration with `fp8` selected and `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` as the input checkpoint. |
| `fp8.yaml` | FP8 quantization configuration tuned for Hopper-class hardware. |
| `nvfp4.yaml` | NVFP4 quantization configuration tuned for Blackwell-class hardware. |
| `tiny.yaml` | Short validation run that exercises the quantization pipeline. |

Pass the configuration name with `-c`:

```console
$ nemotron steps run optimize/modelopt/quantize -c tiny
$ nemotron steps run optimize/modelopt/quantize -c default
$ nemotron steps run optimize/modelopt/quantize -c fp8
$ nemotron steps run optimize/modelopt/quantize -c nvfp4
```

## Inputs and Outputs

| Direction | Artifact Type | Required | Description |
| --- | --- | --- | --- |
| Consumes | `checkpoint_hf` | Yes | A Hugging Face base or aligned checkpoint to quantize. |
| Produces | `checkpoint_megatron` | — | A quantized Megatron distributed checkpoint. Export the artifact to Hugging Face format with `/opt/Megatron-Bridge/examples/quantization/export.py` when you need a deployable Hugging Face checkpoint. |

## Step Parameters

The manifest declares three quantization parameters.
Pass them as dotlist overrides.

```{option} args.export_quant_cfg=<recipe>

The quantization recipe to apply.
The supported recipes match the choices that the installed upstream `quantize.py` accepts.

Choices: `int8_sq`, `fp8`, `fp8_blockwise`, `int4_awq`, `w4a8_awq`, `nvfp4`.

Default: `fp8`.

Example: `args.export_quant_cfg=nvfp4`
```

```{option} args.calib_size=<n>

The number of calibration samples used to determine quantization scales.

Default: `512`.

Example: `args.calib_size=1024`
```

```{option} extra_args=<list>

Literal upstream arguments that the step forwards to the underlying quantization script.
Use this parameter to pass newly added Model Optimizer flags that do not yet have a dedicated `args.*` entry.

Default: `[]`.

Example: `extra_args=["--my_new_flag", "value"]`
```

Frequently used dotlist overrides drawn from the default configuration include the following.

```{option} args.hf_model_id=<id-or-path>

The Hugging Face identifier or local path for the checkpoint to quantize.

Example: `args.hf_model_id=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`
```

```{option} args.megatron_save_path=<path>

The destination directory for the quantized Megatron distributed checkpoint.

Example: `args.megatron_save_path=/lustre/runs/quantized/nano3-fp8`
```

## Strategies

The manifest records three operator strategies for `optimize/modelopt/quantize`.

- When the target hardware is Hopper or H100, start from `config/fp8.yaml` and set `args.export_quant_cfg=fp8`.
- When the target hardware is Blackwell or B200, start from `config/nvfp4.yaml` and set `args.export_quant_cfg=nvfp4`.
- When you need a Hugging Face checkpoint, export the produced Megatron checkpoint by using `/opt/Megatron-Bridge/examples/quantization/export.py` after the quantization run completes.

## Command Examples

Run the tiny validation configuration locally:

```console
$ nemotron steps run optimize/modelopt/quantize -c tiny
```

Compile the FP8 configuration without submitting the job:

```console
$ nemotron steps run optimize/modelopt/quantize -c fp8 --dry-run
```

Submit an attached FP8 run on a Lepton profile for the Nano3 base model:

```console
$ nemotron steps run optimize/modelopt/quantize -c fp8 -r lepton_optimize_modelopt_quantize \
    args.hf_model_id=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    args.calib_size=1024
```

Submit a detached NVFP4 run on a Slurm profile for the Super3 base model:

```console
$ nemotron steps run optimize/modelopt/quantize -c nvfp4 -b slurm_optimize_modelopt_quantize \
    args.hf_model_id=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
    args.megatron_save_path=/lustre/quantized/super3-nvfp4
```

## Related Skill

Run the `nemotron-optimizer-quantization` skill with your agent.

## Related Documentation

- [Nemotron Steps CLI Reference](../cli-reference.md) covers the shared option set, dotlist overrides, and passthrough arguments.
- [Run Post-Training Optimization](../../how-to/run-optimization.md) explains the ordering of prune and distill, hardware targets, and quality recovery.

### Upstream

- [NVIDIA Model Optimizer Repository](https://github.com/NVIDIA/Model-Optimizer)
- [NVIDIA Model Optimizer Documentation](https://nvidia.github.io/Model-Optimizer/)
- [Megatron-Bridge Documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/)
- [Megatron-Bridge Quantization Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/modelopt/quantization.html)
