# ModelOpt Pruning

Use `optimize/modelopt/prune` when the checkpoint architecture itself should be reduced.

Use this README for workflow and pitfalls; use `step.toml` for the exact artifact, parameter, strategy, and error manifest before editing configs or code.

## Inputs And Outputs

- Consume `checkpoint_hf`.
- Produce pruned `checkpoint_hf`.
- Smoke with `nemotron steps run optimize/modelopt/prune -c tiny`.

## Config Nuances

- Use the ModelOpt checkout and installed package together; version drift shows up as missing utilities such as `warn_rank_0` or mismatched `megatron_mmlu` keyword arguments.
- Keep the `moe_grouped_gemm` compatibility patch when the NeMo image's Megatron-Bridge loader does not accept that keyword.
- For target-parameter search, make `args.prune_target_params` realistic for the source model and leave `args.prune_export_config` unset.
- For fixed architecture export, set `args.prune_target_params=null` and pass the exact `args.prune_export_config`.
- Keep `args.output_hf_path` outside the input checkpoint directory; pruning writes intermediate search state beside the output.
- Prune can find a valid architecture before score evaluation fails; treat score-function errors as ModelOpt/package compatibility issues, not proof that architecture search failed.

## CLI And Overlay Knobs

Start from `config/tiny.yaml` for launch validation and `config/default.yaml`
for production-shaped pruning. In a project overlay, developers usually change:

- `args.hf_model_name_or_path`: clean HF source checkpoint.
- `args.output_hf_path`: fresh output directory.
- `args.prune_target_params`: target parameter budget for search.
- `args.prune_export_config`: fixed architecture export config.
- `args.hparams_to_skip`: dimensions that must remain unchanged.
- `args.pp_size` and `extra_args`: parallelism and upstream script flags.

Example shape:

```bash
uv run nemotron steps run optimize/modelopt/prune \
  -c <project>/config/prune.yaml \
  args.hf_model_name_or_path=<hf-checkpoint> \
  args.output_hf_path=<pruned-output>
```

## Run It

Smoke first to validate wiring, imports, data access, and output paths:

```bash
uv run nemotron steps run optimize/modelopt/prune -c tiny --dry-run
```

Then run the real job from a project overlay:

```bash
uv run nemotron steps run optimize/modelopt/prune \
  -c <project>/config/optimize_modelopt_prune.yaml
```

## Repository Layout

- Manifest: `src/nemotron/steps/optimize/modelopt/prune/step.toml`
- Runner: `src/nemotron/steps/optimize/modelopt/prune/step.py`
- Configs: `src/nemotron/steps/optimize/modelopt/prune/config/default.yaml`, `src/nemotron/steps/optimize/modelopt/prune/config/tiny.yaml`

## Guardrails

- Check pipeline-parallel and attention-head divisibility after pruning layer counts or hidden dimensions.
- Distill the pruned model when quality recovery matters.
- Keep the ModelOpt checkout and installed package in sync before debugging
  wrapper-level changes.
- Validate export and downstream loading before claiming the pruned artifact is usable.
