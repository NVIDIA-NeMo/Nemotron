# ModelOpt Distillation

Use `optimize/modelopt/distill` when a student checkpoint should learn from a teacher checkpoint.

Use this README for workflow and pitfalls; use `step.toml` for the exact artifact, parameter, strategy, and error manifest before editing configs or code.

## Inputs And Outputs

- Consume teacher and student `checkpoint_hf` paths.
- Optionally consume `binidx` data from `data_prep/pretrain_prep`.
- Produce `checkpoint_megatron`.
- Validate launch and checkpoint writing before using real distillation data for quality evaluation.

## Config Nuances

- Keep `args.teacher_hf_path`, `args.student_hf_path`, and tokenizer expectations explicit; do not infer them from previous optimize outputs.
- Use `args.use_mock_data=true` only for launch validation, then switch to real `args.data_paths` for any quality signal.
- Keep `args.output_dir` distinct from teacher and student checkpoint roots.
- Distill has native upstream W&B flags; keep `wandb_project`, `wandb_entity`, and experiment name wired from the run environment instead of wrapping the process.
- After pruning, point `args.student_hf_path` at the pruned HF output and keep the original BF16 checkpoint as `args.teacher_hf_path` for quality recovery.
- Use the same tokenizer/chat-template assumptions for teacher, student, and distillation data, especially after structural pruning.

## CLI And Overlay Knobs

Start from `config/tiny.yaml` with mock data for launch validation. In a project
overlay, developers usually change:

- `args.teacher_hf_path`: full-quality teacher checkpoint.
- `args.student_hf_path`: smaller or optimized student checkpoint.
- `args.data_paths`: real bin/idx distillation data.
- `args.output_dir`: fresh output directory.
- `args.use_mock_data`: `true` only for smoke runs.
- `args.hf_export_path`, W&B fields, and `extra_args`.

Example shape:

```bash
uv run nemotron steps run optimize/modelopt/distill \
  -c <project>/config/distill.yaml \
  args.teacher_hf_path=<teacher-hf> \
  args.student_hf_path=<student-hf> \
  args.data_paths=<prep-output>/blend.json
```

## Run It

Smoke first to validate wiring, imports, data access, and output paths:

```bash
uv run nemotron steps run optimize/modelopt/distill -c tiny --dry-run
```

Then run the real job from a project overlay:

```bash
uv run nemotron steps run optimize/modelopt/distill \
  -c <project>/config/optimize_modelopt_distill.yaml
```

## Repository Layout

- Manifest: `src/nemotron/steps/optimize/modelopt/distill/step.toml`
- Runner: `src/nemotron/steps/optimize/modelopt/distill/step.py`
- Configs: `src/nemotron/steps/optimize/modelopt/distill/config/default.yaml`, `src/nemotron/steps/optimize/modelopt/distill/config/tiny.yaml`

## Guardrails

- Use the original BF16 model as teacher when recovering from pruning or quantization.
- Do not treat mock-data runs as quality validation.
- Keep teacher, student, tokenizer, and distillation-data assumptions explicit.
- Choose distillation data that matches the deployment domain.
