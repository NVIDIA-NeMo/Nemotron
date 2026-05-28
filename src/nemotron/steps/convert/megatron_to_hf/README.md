# Megatron To HF Conversion

Use `convert/megatron_to_hf` when a downstream HF-native step needs
`checkpoint_hf` but the upstream artifact is `checkpoint_megatron`.

Use this README for conversion workflow and guardrails; use `step.toml` for exact parameters, strategies, and failure modes.

## Inputs And Outputs

- Consume a specific Megatron checkpoint iteration, normally an `iter_*`
  directory.
- Produce a standalone HF safetensors checkpoint.
- Preserve tokenizer and config expectations from the original HF model id.

## CLI And Overlay Knobs

Start from `config/default.yaml`, then override every source and destination
path. Developers usually change:

- `megatron_path`: concrete `iter_*` checkpoint, not the parent run directory.
- `hf_model_id`: original HF model/config/tokenizer source.
- `hf_path`: fresh HF export directory.
- `trust_remote_code`, `show_progress`, and `strict`.

## Run It

Preview the compiled export first; the shipped `default.yaml` shows the shape
but should not point at a parent training run:

```bash
uv run nemotron steps run convert/megatron_to_hf \
  -c default \
  megatron_path=<run>/iter_<n> \
  hf_model_id=<original-hf-model> \
  hf_path=<hf-export-output> \
  --dry-run
```

Then run the real export without `--dry-run`:

```bash
uv run nemotron steps run convert/megatron_to_hf \
  -c default \
  megatron_path=<run>/iter_<n> \
  hf_model_id=<original-hf-model> \
  hf_path=<hf-export-output>
```

## Repository Layout

- Manifest: `src/nemotron/steps/convert/megatron_to_hf/step.toml`
- Runner: `src/nemotron/steps/convert/megatron_to_hf/step.py`
- Config: `src/nemotron/steps/convert/megatron_to_hf/config/default.yaml`

## Guardrails

- Do not export while async checkpoint save is still in progress.
- Do not guess among multiple checkpoint iterations; pick the validated one.
- Validate that the exported HF checkpoint loads before using it for eval or
  deployment.
