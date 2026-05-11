---
name: nemotron-optimizer-pruning
description: Configure Nemotron optimize/modelopt/prune for structured pruning with NVIDIA ModelOpt and Megatron-Bridge. Use for Minitron target-parameter search, fixed architecture pruning, hparam skip lists, divisibility checks, and HuggingFace pruned checkpoint output.
---

# ModelOpt Pruning

Use `optimize/modelopt/prune` when the checkpoint architecture itself should be reduced.

Before changing configs or code, read `step.toml` to understand the step flow, consumed and produced artifacts, important parameters, strategies, failure modes, and upstream references.

## Inputs And Outputs

- Consume `checkpoint_hf`.
- Produce pruned `checkpoint_hf`.
- Smoke with `nemotron steps run optimize/modelopt/prune -c tiny`.

## Configure

- Set `args.prune_target_params` when ModelOpt should search for a target budget.
- Set `args.prune_export_config` when the deployment architecture is fixed.
- Set `args.prune_target_params=null` when using a fixed export config.
- Use `args.hparams_to_skip` for dimensions that must remain unchanged.
- Use `extra_args` for new upstream pruning flags.

## Config Nuances

- Use the ModelOpt checkout and installed package together; version drift shows up as missing utilities such as `warn_rank_0` or mismatched `megatron_mmlu` keyword arguments.
- Keep the `moe_grouped_gemm` compatibility patch when the NeMo image's Megatron-Bridge loader does not accept that keyword.
- For target-parameter search, make `args.prune_target_params` realistic for the source model and leave `args.prune_export_config` unset.
- For fixed architecture export, set `args.prune_target_params=null` and pass the exact `args.prune_export_config`.
- Keep `args.output_hf_path` outside the input checkpoint directory; pruning writes intermediate search state beside the output.
- Prune can find a valid architecture before score evaluation fails; treat score-function errors as ModelOpt/package compatibility issues, not proof that architecture search failed.

## Local Files

- Contract: `src/nemotron/steps/optimize/modelopt/prune/step.toml`
- Runner: `src/nemotron/steps/optimize/modelopt/prune/step.py`
- Configs: `src/nemotron/steps/optimize/modelopt/prune/config/default.yaml`, `src/nemotron/steps/optimize/modelopt/prune/config/tiny.yaml`

## Guardrails

- Check pipeline-parallel and attention-head divisibility after pruning layer counts or hidden dimensions.
- Distill the pruned model when quality recovery matters.
- Validate export and downstream loading before claiming the pruned artifact is usable.
