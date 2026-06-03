# HF To Megatron Conversion

Use `convert/hf_to_megatron` when a downstream Megatron-Bridge step needs
`checkpoint_megatron` but the upstream artifact is `checkpoint_hf`.

Use this README for conversion workflow and guardrails; use `step.toml` for exact parameters, strategies, and failure modes.

## Inputs And Outputs

- Consume a clean HF checkpoint directory or model id.
- Produce a Megatron distributed checkpoint in a fresh output directory.
- Keep tokenizer and model config files resolvable during import.

## CLI And Overlay Knobs

Start from `config/default.yaml`, then override every source and destination
path. Developers usually change:

- `hf_model_id`: HF model ID or local clean HF checkpoint path.
- `megatron_path`: fresh Megatron output directory.
- `torch_dtype` / `dtype`: match the source checkpoint and target stack.
- `device_map`: only when the installed stack requires it.
- `trust_remote_code`: keep `true` only for trusted supported model repos.

## Run It

Preview the compiled conversion first; the shipped `default.yaml` documents
the expected fields rather than a production path:

```bash
uv run nemotron steps run convert/hf_to_megatron \
  -c default \
  hf_model_id=<hf-model-or-path> \
  megatron_path=<megatron-output> \
  --dry-run
```

Then run the real conversion without `--dry-run`:

```bash
uv run nemotron steps run convert/hf_to_megatron \
  -c default \
  hf_model_id=<hf-model-or-path> \
  megatron_path=<megatron-output>
```

## Repository Layout

- Manifest: `src/nemotron/steps/convert/hf_to_megatron/step.toml`
- Runner: `src/nemotron/steps/convert/hf_to_megatron/step.py`
- Config: `src/nemotron/steps/convert/hf_to_megatron/config/default.yaml`

## Guardrails

- Do not import trainer-state directories, optimizer folders, or adapter-only
  outputs.
- Do not write the Megatron output under the HF source directory.
- Keep `trust_remote_code=true` only for model repos you trust and whose
  architecture is supported by the installed Megatron-Bridge AutoBridge.
