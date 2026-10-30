---
name: nemotron-transcribe
description: Reference and routing skill for NVIDIA Nemotron 3.5 Transcribe. Use for model facts, checkpoint selection, inference constraints, and locating the canonical NeMo Speech main-stage training, fine-tuning, DFlash2, timestamp-serving, and quantization workflows; do not use as a substitute for those operational skills.
---

# nemotron-transcribe

Invocation: `/nemotron-transcribe`.

Use this skill as the reference desk and workflow router for **NVIDIA Nemotron
3.5 Transcribe**. Keep NeMo Speech as the source of truth for executable speech
training and export procedures.

## Start here

1. Read `INDEX.md` to locate the smallest relevant source.
2. Use `context/quick-reference.md` for fast routing.
3. Use `model-card.md` for identity, variants, capabilities, and inference
   boundaries.
4. Use `recipes/overview.md` to select an available upstream workflow and see
   which stages are still missing.
5. For deployment examples, read
   `usage-cookbook/Nemotron-3.5-Transcribe/README.md` from the repository root.

## Source priority

Resolve conflicts in this order:

1. the selected checkpoint's Hugging Face model card;
2. the canonical NeMo Speech skill and implementation for the requested stage;
3. the [technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Transcribe-Technical-Report.pdf) for
   model-development, architecture, and reported-evaluation claims;
4. this skill's compact summaries;
5. the Nemotron cookbook.

Cite the stable report URL above only after it is publicly accessible.

Use the BF16 model card while planning fine-tuning and the destination
precision's model card while planning deployment. The model cards define
serving flags per precision; do not merge BF16, FP8, and NVFP4 launch settings
into a single assumed profile.

## Route operational requests

| User intent | Canonical source |
|---|---|
| Construct a SALM from pretrained language and speech components | [NeMo Speech `train-speech-llm`](https://github.com/NVIDIA-NeMo/Speech/blob/main/.claude/skills/train-speech-llm/SKILL.md) with [`salm_nemotron.yaml`](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/speechlm2/conf/salm_nemotron.yaml) |
| Reproduce the exact released checkpoint | The main-stage template is parameterized; production component checkpoints, data blends, cluster values, and full validation evidence are not included |
| Fine-tune for a domain, language, or speech task | [NeMo Speech `finetune-speech-llm`](https://github.com/NVIDIA-NeMo/Speech/blob/main/.claude/skills/finetune-speech-llm/SKILL.md) |
| Train or adapt a DFlash/DFlash2 draft | [NeMo Speech `train-dflash`](https://github.com/NVIDIA-NeMo/Speech/blob/main/.claude/skills/train-dflash/SKILL.md) |
| Quantize a SALM checkpoint to FP8 or NVFP4 | [NeMo Speech `quantize-speech-llm`](https://github.com/NVIDIA-NeMo/Speech/blob/main/.claude/skills/quantize-speech-llm/SKILL.md) |
| Serve BF16, FP8, or NVFP4 with vLLM | Nemotron Transcribe vLLM cookbook plus the selected model card |
| Request word timestamps and diarization metadata | Nemotron Transcribe vLLM cookbook, the selected model card, and the maintained [NeMo Speech middleware](https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/speechlm2/vllm/salm/ctc_serving.py); [PR #16344](https://github.com/NVIDIA-NeMo/Speech/pull/16344) is change provenance |
| Reproduce RL post-training | No public Transcribe-specific recipe is linked yet; state the gap |
| Train the timestamp heads | No public training recipe is linked yet; do not substitute timestamp serving support |
| Reproduce published evaluation | No public automated harness, datasets, normalization, and scoring bundle is linked yet; state the gap |

When routing, give the relevant stage order and prerequisites first, then point
to the upstream skill. Do not reproduce its detailed procedure locally.

## Precision selection

- **BF16** is the reference checkpoint and the starting point for fine-tuning.
- **FP8** is a ModelOpt export with quantized decoder, audio encoder, and MTP
  head. Let the serving stack read its quantization configuration.
- **NVFP4** is a mixed NVFP4/FP8 export and requires Blackwell. It uses an
  explicit FP8 KV cache in the validated profile.

Name the exact model ID and precision in every serving or evaluation answer.
Do not imply that a quantized release is the preferred training source.
When a user fine-tunes BF16 and then requests FP8 or NVFP4, route them to
quantize their merged customized checkpoint. The published quantized releases
are quantizations of NVIDIA's released BF16 model, not of the user's fine-tune.

## Inference rules worth surfacing

- The model is offline and supports recordings up to the one-hour training
  limit; internal 30-second chunks are not streaming.
- For transcription, translation, and language identification, start with
  reasoning disabled and temperature 0.
- Preserve special tokens for speaker and event tags.
- Timestamp output is optional metadata. Check its error field before using
  word alignments, and keep the request to one audio, one non-streaming
  completion.
- MTP and DFlash2 are alternatives. Current DFlash2 serving does not combine
  with CTC timestamps.
- Smoke-test MTP and CTC timestamps on the merged BF16 fine-tune before
  quantization, then test them again after assembly and quantization.
- Prefer the selected model card's validated speculative-decoding profile over
  an illustrative default in a generic stage guide.
- Treat NeMo Speech and the checkpoint as one compatibility bundle. Current
  checkpoints use `Nemotron3_5TranscribeForConditionalGeneration`; they require
  a NeMo Speech revision that registers that architecture and its
  `Nemotron3_5TranscribeMTPModel` draft. Old/old pairs may still work, but do
  not mix pre-rename and post-rename components.
- Pin the GPU-tested NeMo Speech and checkpoint revisions; `main` and an
  unqualified model ID are moving dependencies.

## Answering style

- Start with the direct answer.
- Distinguish model-card claims, technical-report results, and public recipe
  coverage as separate source layers.
- Link the exact upstream skill, script, or guide used.
- State public gaps plainly. Never invent a checkpoint, dataset, stage config,
  or release pin.
- For exhaustive benchmark numbers, prompt lists, language coverage, safety,
  and licensing, link the model card instead of copying it.

## Boundaries

Do:

- provide model-specific routing, stage ordering, variant selection, and
  known compatibility constraints;
- distinguish main-stage construction from released-checkpoint adaptation and
  from exact release reproduction;
- use NeMo Speech for speech-model implementation details;
- use the Nemotron cookbook for concise deployment examples.

Do not:

- paste or fork the NeMo Speech operational skills into this directory;
- fill the main-stage template's production placeholders with guessed values
  or present it as the exact released-checkpoint recipe;
- claim that timestamp serving is a timestamp-head training recipe;
- claim public RL reproduction until a maintained Transcribe recipe exists;
- apply generic LLM fine-tuning guidance when the speech checkpoint's encoder,
  projection, auxiliary heads, and prompt format must be preserved;
- expose credentials or internal paths, or publish a checkpoint revision before
  confirming that the corresponding artifact is public.
