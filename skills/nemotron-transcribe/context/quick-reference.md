# Nemotron 3.5 Transcribe quick reference

Use this file for model selection and routing. Read the selected Hugging Face
card for checkpoint-specific settings and the linked NeMo Speech skill before
executing a training or export stage.

## Identity

- **Model:** NVIDIA Nemotron 3.5 Transcribe 30B-A4B.
- **Size:** approximately 34B total / 4.3B active, including the audio encoder.
- **Backbone:** Nemotron 3.5 Lightning hybrid Mamba-2/Transformer MoE.
- **Input/output:** 16 kHz mono audio plus text instruction / generated text,
  with optional CTC alignment metadata.
- **Limits:** offline recordings up to the one-hour training limit and a
  65,536-token training context.
- **Languages:** transcription covers 40 languages. Audio understanding,
  reasoning, and spoken-instruction following were evaluated only in English.

## Checkpoint choice

| Goal | Start from |
|---|---|
| Fine-tune or adapt | BF16 `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` |
| Serve NVIDIA's FP8 release | `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8` |
| Serve NVIDIA's Blackwell-only NVFP4 release | `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4` |
| Add the released DFlash2 speculative path | Pair `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2` with a target checkpoint; do not use the draft standalone |
| Deploy a customized quantized model | Quantize the merged customized BF16 checkpoint; do not substitute a published quantized release |

## Public workflow routing

| Need | Route |
|---|---|
| Construct a SALM from pretrained components | NeMo Speech `train-speech-llm` plus `salm_nemotron.yaml` |
| Reproduce the exact released checkpoint | Not included; the public template retains user-supplied component, data, and cluster values |
| Fine-tune a released checkpoint | NeMo Speech `finetune-speech-llm` |
| Reproduce RL post-training | Not included in this contribution |
| Train timestamp heads | Not included; timestamp serving is not training |
| Train or adapt a DFlash2 draft | NeMo Speech `train-dflash` |
| Quantize customized BF16 to FP8/NVFP4 | NeMo Speech `quantize-speech-llm` |
| Serve and issue requests | Nemotron Transcribe cookbook plus the selected model card |
| Reproduce benchmark evaluation | Missing public automated harness, datasets, normalization, and scoring |

## Serving invariants

- Use a matched NeMo Speech and checkpoint pair that has been GPU-tested. The
  cookbook's included pinned pair is deterministic but still needs GPU
  validation. Current checkpoints name
  `Nemotron3_5TranscribeForConditionalGeneration`; do not mix them with the
  pre-rename loader, or mix an old checkpoint with the renamed loader.
- Use vLLM 0.32.0. Native MTP on the hybrid backbone uses ReplaySSM with the
  FlashInfer Mamba backend, `--mamba-cache-mode none`, and prefix caching off.
- Transcription, translation, and language identification normally use
  reasoning off and temperature 0.
- Audio reasoning uses the model card's separate sampling and token budgets.
- Preserve special tokens for speaker and rich-transcription tags.
- Native MTP and DFlash2 are alternative speculative paths.
- CTC timestamp requests use one audio input, `stream=false`, and `n=1`; reject
  a non-null alignment error.
- Current DFlash2 serving does not combine with CTC timestamps.
- NVFP4 requires Blackwell and uses an explicit FP8 KV cache in the validated
  profile.
- Bind the unauthenticated local example to loopback. Put intentional network
  exposure behind authentication, TLS, and access controls.

## Source boundary

Use the [technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Transcribe-Technical-Report.pdf)
for model-development, architecture, and reported-evaluation claims. Detailed
main-stage training, fine-tuning,
DFlash2, and quantization procedures remain in NeMo Speech rather than being
mirrored in this skill.
