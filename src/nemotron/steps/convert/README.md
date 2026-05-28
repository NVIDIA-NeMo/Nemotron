# Convert (checkpoint format bridges)

Pick a convert step whenever the producer and consumer in your plan disagree
on `checkpoint_*` type. The artifact graph in
[../types.toml](../types.toml) tells you which converter to insert via the
`convert_to` map.

## Developer Journey

Conversion is a checkpoint boundary, not a default stage. Insert it only when a
producer and consumer disagree on checkpoint layout.

1. Identify the checkpoint artifact produced by the previous step.
2. Identify the exact checkpoint artifact consumed by the next step.
3. Pick the converter only if those types differ.
4. Convert from a clean, validated checkpoint iteration or base model.
5. Validate the converted checkpoint loads before continuing to eval, deploy, or
   further optimization.

| Source type | Target type | Step |
|---|---|---|
| `checkpoint_megatron` | `checkpoint_hf` | [megatron_to_hf](megatron_to_hf/README.md) |
| `checkpoint_hf` | `checkpoint_megatron` | [hf_to_megatron](hf_to_megatron/README.md) |
| `checkpoint_lora` (+ original base) | `checkpoint_hf` (merged) | [merge_lora](merge_lora/README.md) |

## Data And Artifact Flow

```text
checkpoint_megatron -> convert/megatron_to_hf -> checkpoint_hf
checkpoint_hf       -> convert/hf_to_megatron -> checkpoint_megatron
checkpoint_lora + original base -> convert/merge_lora -> checkpoint_hf
```

Keep training-state files, optimizer state, and parent run directories out of
conversion inputs unless the converter explicitly expects them. Most conversion
steps want a model checkpoint directory or a specific `iter_*` checkpoint.

## When to insert

- Megatron-Bridge SFT/PEFT/pretrain produces `checkpoint_megatron`. Eval/RL
  consumers that expect HF format need `megatron_to_hf` first.
- AutoModel SFT/PEFT produces `checkpoint_hf`. Megatron-Bridge consumers need
  `hf_to_megatron` first.
- Any LoRA producer (`peft/*`) emits `checkpoint_lora`. HF/PEFT adapters can
  merge directly with `merge_lora backend=hf_peft`; Megatron-Bridge adapters
  use `backend=megatron_bridge`. `backend=auto` chooses from the base path
  fields and can export a Megatron-Bridge merge to HF in the same step.

## Patterns to cite

- [../patterns/convert-checkpoint-safety.md](../patterns/convert-checkpoint-safety.md) —
  convert from a clean checkpoint, not from training-state files.
- [../patterns/peft-adapter-merge-discipline.md](../patterns/peft-adapter-merge-discipline.md) —
  validate the adapter alone before merging.

## Guardrails

- Don't add a converter "just in case." Pick one input artifact type per
  consumer and configure to match.
- Read the selected converter's `step.toml`; it now carries required paths,
  merge provenance, and conversion failure modes.
- When converting Megatron → HF, point at the specific `iter_*` directory,
  not the parent run dir.
- When merging LoRA, you need the *original* base checkpoint the adapter was
  trained against. For Megatron-Bridge adapters, preserve the dense Megatron
  base and an HF model/config source for export.
