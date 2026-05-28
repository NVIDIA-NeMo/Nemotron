# Merge LoRA

Use `convert/merge_lora` when a downstream consumer needs a standalone
`checkpoint_hf` instead of a separate adapter artifact.

Use this README for conversion workflow and guardrails; use `step.toml` for exact parameters, strategies, and failure modes.

## Inputs And Outputs

- Consume `checkpoint_lora` plus the original base checkpoint.
- With `backend=hf_peft`, consume the original HF base and write HF output
  directly.
- With `backend=megatron_bridge`, consume the original dense Megatron base,
  write a merged Megatron checkpoint, then export it to HF when `export_hf=true`.

## CLI And Overlay Knobs

Start from `config/default.yaml`, then override adapter, base, and output paths.
Developers usually change:

- `backend`: `auto`, `hf_peft`, or `megatron_bridge`.
- `lora_checkpoint`: adapter output from PEFT.
- `base_hf_path` or `base_megatron_path`: exact base used for adapter training.
- `hf_model_id` / `hf_model_path`: HF config/tokenizer source for export.
- `output_hf_path` and, for Megatron-Bridge merges, `output_megatron_path`.
- `cpu`, `tp`, `pp`, and `ep` for the merge/export topology.

## Run It

Preview the compiled merge config first; the shipped `default.yaml` is a shape
reference, not a runnable merge against your adapter:

```bash
uv run nemotron steps run convert/merge_lora \
  -c default \
  lora_checkpoint=<adapter> \
  base_hf_path=<original-base> \
  output_hf_path=<merged-output> \
  --dry-run
```

Then run the real merge without `--dry-run`:

```bash
uv run nemotron steps run convert/merge_lora \
  -c default \
  lora_checkpoint=<adapter> \
  base_hf_path=<original-base> \
  output_hf_path=<merged-output>
```

For Megatron-Bridge adapters, set `backend=megatron_bridge`,
`base_megatron_path=<dense-base>`, and `output_megatron_path=<merged-megatron>`;
add `output_hf_path` to export the merged checkpoint to HF in the same run.

## Repository Layout

- Manifest: `src/nemotron/steps/convert/merge_lora/step.toml`
- Runner: `src/nemotron/steps/convert/merge_lora/step.py`
- Config: `src/nemotron/steps/convert/merge_lora/config/default.yaml`

## Guardrails

- Never merge into a different base, even if the model name looks compatible.
- Evaluate after merge; adapter-loaded and merged-model scores can differ.
- Keep tokenizer, chat template, LoRA rank, alpha, and target module provenance
  with the merged artifact.
