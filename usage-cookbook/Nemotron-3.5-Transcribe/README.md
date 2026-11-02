# Nemotron 3.5 Transcribe cookbook

Deployment and usage examples for **NVIDIA Nemotron 3.5 Transcribe**, an
offline speech-and-audio language model for multilingual transcription,
translation, speaker attribution, timestamps, and English-only evaluated audio
understanding.

## What's inside

### Deployment and usage

- **[vllm_cookbook.ipynb](vllm_cookbook.ipynb)** — prepare the NeMo Speech
  vLLM environment, choose BF16/FP8/NVFP4, launch an OpenAI-compatible server,
  and run transcription, contextual ASR, translation, English audio QA, and
  timestamp requests.

### Customization

The maintained training implementations and operational skills live in
[NeMo Speech](https://github.com/NVIDIA-NeMo/Speech). Start from the
[Nemotron recipe map](../../docs/nemotron/transcribe35/README.md#open-recipe-coverage),
which links the parameterized main-stage construction, fine-tuning, DFlash2,
and quantization workflows and records what is not included in this
contribution.

## Choose a checkpoint

| Variant | Model | Model-card launch profile | Important setting |
|---|---|---|---|
| BF16 | [NVIDIA-Nemotron-3.5-Transcribe-30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B) | 1× H100 80 GB | Reference weights; `--kv-cache-dtype auto` |
| FP8 | [NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8) | 1× RTX PRO 6000 Blackwell 96 GB | Keep ModelOpt auto-detection and `--kv-cache-dtype auto` |
| NVFP4 | [NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4) | 1× RTX PRO 6000 Blackwell 96 GB | Blackwell only; set `--kv-cache-dtype fp8` |

The separately released
[DFlash2 draft](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2)
is paired with a compatible target checkpoint; it is not a standalone model.
The notebook keeps native MTP as its default and points to the
selected target model card for the DFlash2-specific serving profile.

These starting profiles are transcribed from the model cards; this
contribution has not independently GPU-validated the exact pins. They are not
the complete hardware support matrix. Refer to the selected model card before
changing GPU type, concurrency, context length, precision, or speculative
decoding.

## Serving dependencies

The launch integration uses:

- vLLM 0.32.0;
- the NeMo Speech `nemo_speechlm` vLLM plugin;
- Python 3.13 and the locked NeMo Speech `asr` and `vllm` extras;
- PyAV 19.0.2 for compressed formats such as M4A;
- CUDA 13 toolkit access for first-use compilation of quantized kernels.

This cookbook pins NeMo Speech commit
`cbc617bb94a31451fe0a3503741e2a8e7568ff22` and a matching post-rename revision
for each of the three models. The checkpoint configs name
`Nemotron3_5TranscribeForConditionalGeneration`, which this Speech commit
registers together with its MTP draft architecture. Do not combine these
checkpoints with a pre-rename Speech loader, or an older checkpoint with this
loader. The pinned set is deterministic but has not been independently
GPU-validated; validate it before production use.

## Behavioral constraints

- The cookbook binds the unauthenticated example server to `127.0.0.1`
  explicitly. Add network exposure only behind appropriate authentication and
  access controls.
- The model is offline. Encoder chunking does not provide streaming ASR.
- The 40-language claim does not extend to every task. Audio understanding,
  reasoning, and spoken-instruction following were evaluated only in English.
- Use reasoning off and temperature 0 for transcription, translation, and
  language identification. Reasoning-enabled audio understanding uses a
  different sampling profile.
- Timestamp requests support one audio input, `stream=false`, and `n=1`; check
  the returned `ctc_timestamps.error` field.
- A dense one-hour conversation can need about 20,000 output tokens. Raise or
  remove `max_tokens` for long recordings to avoid truncating the transcript.
- Preserve special tokens for speaker and rich-transcription tags.
- Native MTP and DFlash2 are alternative speculative modes. The current
  DFlash2 serving path does not provide CTC timestamps.
- Do not treat the optimized NIM performance numbers in the model card as
  measurements of this open-source vLLM workflow.

## Model resources

- [Technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Transcribe-Technical-Report.pdf)
- [Model-family training and adaptation map](../../docs/nemotron/transcribe35/README.md)
- [Nemotron Transcribe reference skill](../../skills/nemotron-transcribe/SKILL.md)
- [NeMo Speech main-stage training skill](https://github.com/NVIDIA-NeMo/Speech/blob/main/.claude/skills/train-speech-llm/SKILL.md)
- [NeMo Speech SALM Nemotron configuration](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/speechlm2/conf/salm_nemotron.yaml)
- [NeMo Speech fine-tuning toolkit](https://github.com/NVIDIA-NeMo/Speech/tree/main/examples/speechlm2/finetuning)
- [NeMo Speech DFlash2 training guide](https://github.com/NVIDIA-NeMo/Speech/blob/main/docs/source/speechlm2/dflash_training.rst)
- [NeMo Speech quantization workflow](https://github.com/NVIDIA-NeMo/Speech/tree/main/scripts/speechlm2/quantization)
