---
name: nemotron-translate-nemo-skills
description: Configure Nemotron translate/nemo_skills to translate filtered_jsonl into a target language using NeMo Skills (Speaker) and attach FAITH quality signals. Use for multilingual SFT data prep, NIM/vLLM-served translation models, faith-threshold filtering, and segmented-message preservation.
---

# Translation + FAITH Scoring (NeMo Skills)

Use `translate/nemo_skills` to localize `filtered_jsonl` and gate by FAITH
score before downstream training.

Read `step.toml` for full strategies, errors, and parameter choices.

## Inputs and outputs

- Consume: `filtered_jsonl`.
- Produce: `translated_jsonl` (target-language messages + FAITH metadata).

## Configure

- **Backend**:
  - `nim` for production-ready managed endpoints (default).
  - `vllm` for self-hosted or local checkpoint evaluation.
  - Either way, the model is reached over OpenAI-compatible chat endpoints —
    `missing_openai_endpoint` recovery is "provision a reachable endpoint and
    pass server address + model name into the Speaker config."
- **Faith threshold**: `0.7` keeps high-confidence translations. Set to `0.0`
  to score without filtering (useful for analysis runs).
- **Long or structured messages**: enable fine segmentation so Speaker keeps
  separators and reconstructs structure faithfully.
- **FAITH must run on raw Speaker output**, not on a downstream JSONL that's
  already dropped `translations.segmented_translation` metadata. If you see
  `faith_input_missing_segmented_translation`, you regressed the input.
- Reference [src/nemotron/steps/patterns/multilingual-tokenizer-check.md](../../patterns/multilingual-tokenizer-check.md)
  before chaining into a multilingual SFT step.

## Local files

- Contract: [step.toml](step.toml)
- Runner: [step.py](step.py)
- Configs: `config/default.yaml`, `config/tiny.yaml`

## Guardrails

- Don't lower `faith_threshold` to "rescue" rejected records — the threshold
  is a quality gate, not a quota knob.
- Inspect a sample of translated records before training; FAITH is a useful
  signal, not a proof of meaning preservation.
- Don't run translation and SFT off different model lineages without a
  tokenizer-coverage check on the target language.
