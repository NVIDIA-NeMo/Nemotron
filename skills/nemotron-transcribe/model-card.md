# Nemotron 3.5 Transcribe release reference

## Identity

| Property | Value |
|---|---|
| Model family | NVIDIA Nemotron 3.5 Transcribe |
| Architecture | Speech-augmented hybrid Mamba-2/Transformer MoE with parallel ASR and speaker encoders |
| Parameters | Approximately 34B total / 4.3B active, including the audio encoder |
| Context | 65,536 tokens; recordings up to one hour in training |
| Input / output | Audio plus text instructions / text, with optional timestamp metadata |
| Languages | 40; use the model card's language matrix for task-level coverage |
| License | [OpenMDW-1.1](https://openmdw.ai/license/1-1/) |
| Release date | October 28, 2026 |
| Technical report | [NVIDIA Nemotron 3.5 Transcribe Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Transcribe-Technical-Report.pdf) |

## Checkpoint family

- [BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B):
  reference weights and recommended fine-tuning source.
- [FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8):
  ModelOpt FP8 decoder, audio encoder, and MTP head.
- [NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4):
  Blackwell-only mixed NVFP4/FP8 serving checkpoint.
- [DFlash2](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2):
  optional speculative draft referenced by the three primary model cards and
  used with a target checkpoint, not as a standalone transcription model.

## Capability boundary

The model supports controlled transcription, contextual vocabulary, speaker
attribution, rich tags, timestamps, translation, language identification, and
audio questions. Audio understanding, reasoning, and spoken-instruction
following were evaluated only in English and may not work reliably in other
languages. It contains limited text instruction-following and PII-related
training, but it is a specialized speech model rather than a general-purpose
assistant.

It consumes 16 kHz mono audio. The encoder processes internal chunks, but the
public model is an offline service rather than a streaming recognizer.

## Serving baseline

The current launch model cards require vLLM 0.32.0 with the NeMo Speech
`nemo_speechlm` plugin. Their configs use the dedicated
`Nemotron3_5TranscribeForConditionalGeneration` architecture; native MTP is
registered as `Nemotron3_5TranscribeMTPModel`. These names were introduced as
a matched loader/checkpoint change, so pre-rename and post-rename components
must not be mixed.

All variants use BF16 compute/residual dtype, native MTP through ReplaySSM as
the default speculative method, the Nemotron reasoning parser, and optional
CTC timestamp middleware. Variant-specific memory, concurrency, quantization,
and KV-cache settings remain in the selected model card.

Do not copy performance from an optimized NIM measurement to the open-source
vLLM path. They use different optimizations.

## Prompt defaults

| Task | Reasoning | Sampling |
|---|---|---|
| Transcription, translation, language identification | Off | temperature 0, top-p 1 |
| Audio understanding and spoken reasoning | On | temperature 0.6, top-p 0.95; follow model-card budgets |

For transcription and translation, keep the system prompt empty and put the
desired style, language, formatting, and context terms in the user instruction.
Link the model card for the evaluated prompt inventory rather than mirroring it
here.

## Timestamp output

With CTC capture enabled, a response can include aligned words, start/end
times, transcript speaker tags, diarization segments, the tag-to-diarizer
mapping, and an error field. Timestamp requests require one audio input,
`stream=false`, and `n=1`. Treat a non-null error as an alignment failure.

Current DFlash2 serving disables the CTC timestamp path.
