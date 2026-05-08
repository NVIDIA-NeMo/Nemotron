---
name: nemotron-convert
description: Choose between Nemotron convert steps that bridge checkpoint formats — Megatron distributed ↔ HuggingFace safetensors, and LoRA-adapter merge into base. Use when artifact wiring needs a converter to chain producer (sft/pretrain/peft) and consumer (eval/rl/optimize) types.
---

# Convert (checkpoint format bridges)

Pick a convert step whenever the producer and consumer in your plan disagree
on `checkpoint_*` type. The artifact graph in
[../types.toml](../types.toml) tells you which converter to insert via the
`convert_to` map.

| Source type | Target type | Step |
|---|---|---|
| `checkpoint_megatron` | `checkpoint_hf` | [megatron_to_hf](megatron_to_hf/step.toml) |
| `checkpoint_hf` | `checkpoint_megatron` | [hf_to_megatron](hf_to_megatron/step.toml) |
| `checkpoint_lora` (+ base `checkpoint_hf`) | `checkpoint_hf` (merged) | [merge_lora](merge_lora/step.toml) |

## When to insert

- Megatron-Bridge SFT/PEFT/pretrain produces `checkpoint_megatron`. Eval/RL
  consumers that expect HF format need `megatron_to_hf` first.
- AutoModel SFT/PEFT produces `checkpoint_hf`. Megatron-Bridge consumers need
  `hf_to_megatron` first.
- Any LoRA producer (`peft/*`) emits `checkpoint_lora`. Eval and RL almost
  always want a merged HF model, not the adapter alone — chain `merge_lora`.

## Patterns to cite

- [../patterns/convert-checkpoint-safety.md](../patterns/convert-checkpoint-safety.md) —
  convert from a clean checkpoint, not from training-state files.
- [../patterns/peft-adapter-merge-discipline.md](../patterns/peft-adapter-merge-discipline.md) —
  validate the adapter alone before merging.

## Guardrails

- Don't add a converter "just in case." Pick one input artifact type per
  consumer and configure to match.
- When converting Megatron → HF, point at the specific `iter_*` directory,
  not the parent run dir.
- When merging LoRA, you need the *original* base checkpoint the adapter was
  trained against. Don't merge into a different base.
