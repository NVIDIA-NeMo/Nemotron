# ModelOpt Quantization

Use `optimize/modelopt/quantize` for post-training quantization of Hugging Face checkpoints into Megatron distributed output.

Use this README for workflow and pitfalls; use `step.toml` for the exact artifact, parameter, strategy, and error manifest before editing configs or code.

## Inputs And Outputs

- Consume `checkpoint_hf`.
- Produce quantized `checkpoint_megatron`.
- Validate export and checkpoint writing with a short calibration run before quality evaluation.

## Config Nuances

- Use the base quantization config for quick launch validation, `fp8` for Hopper FP8, and `nvfp4` only on Blackwell-ready resources.
- Prefer a smaller BF16 source checkpoint such as `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` for launch validation; switch `args.hf_model_id` to larger production checkpoints only after the path is proven.
- Treat very small `args.calib_size` values as launch validation only; increase calibration with representative prompts before evaluating quality.
- Keep `args.megatron_save_path` separate from the input checkpoint path so failed exports do not corrupt the source artifact.
- Avoid model-specific recipe names unless the container's `quantize.py --help` lists them.
- Keep `torchrun.nproc_per_node * torchrun.nnodes` divisible by `args.tp * args.pp`; adjust `args.pp` down before changing the launcher shape.
- For Mamba/Megatron-Bridge models, ensure Mamba group counts are divisible by tensor parallel size.
- Keep quantization and pruning W&B logging through the wrapper config; upstream quantize does not expose the native W&B flags that distill does.

## CLI And Overlay Knobs

Start from `config/tiny.yaml` for launch validation, `config/fp8.yaml` for
Hopper FP8, and `config/nvfp4.yaml` for Blackwell NVFP4. In a project overlay,
developers usually change:

- `args.hf_model_id`: clean HF checkpoint to quantize.
- `args.export_quant_cfg`: recipe such as `fp8`, `nvfp4`, or `int4_awq`.
- `args.calib_size`: representative calibration size.
- `args.megatron_save_path`: fresh output path.
- `args.tp`, `args.pp`, and launcher world size.
- `extra_args`: upstream script flags that the wrapper does not expose directly.

Example shape:

```bash
uv run nemotron steps run optimize/modelopt/quantize \
  -c <project>/config/quantize.yaml \
  args.hf_model_id=<hf-checkpoint> \
  args.megatron_save_path=<quantized-output>
```

## Run It

Smoke first to validate wiring, imports, data access, and output paths:

```bash
uv run nemotron steps run optimize/modelopt/quantize -c tiny --dry-run
```

Then run the real job from a project overlay:

```bash
uv run nemotron steps run optimize/modelopt/quantize \
  -c <project>/config/optimize_modelopt_quantize.yaml
```

## Repository Layout

- Manifest: `src/nemotron/steps/optimize/modelopt/quantize/step.toml`
- Runner: `src/nemotron/steps/optimize/modelopt/quantize/step.py`
- Configs: `config/default.yaml`, `config/tiny.yaml`, `config/fp8.yaml`, `config/nvfp4.yaml`

## Guardrails

- Do not judge quality from tiny calibration runs.
- Export the produced Megatron checkpoint when downstream tools need Hugging Face layout.
- Run task eval and serving smoke tests after quantization and export.
