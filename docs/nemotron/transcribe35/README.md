# Nemotron 3.5 Transcribe

NVIDIA Nemotron 3.5 Transcribe is an offline speech-and-audio language model
for instruction-controlled transcription and audio understanding. It combines
a parallel ASR and speaker encoder with the Nemotron 3.5 Lightning hybrid
Mamba-Transformer mixture-of-experts backbone.

This page is the model-family recipe map for Nemotron. The executable speech
training and export implementations live in
[NeMo Speech](https://github.com/NVIDIA-NeMo/Speech); this repository links to
those maintained implementations instead of copying them.

## Model releases

| Checkpoint | Precision | Model-card serving profile | Notes |
|---|---|---|---|
| [NVIDIA-Nemotron-3.5-Transcribe-30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B) | BF16 | 1× H100 80 GB | Reference checkpoint for customization |
| [NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8) | FP8 | 1× RTX PRO 6000 Blackwell 96 GB in the model card | ModelOpt export; FP8 decoder, audio encoder, and MTP head |
| [NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4) | NVFP4 | Blackwell GPU; model-card profile uses 1× RTX PRO 6000 Blackwell 96 GB | Mixed NVFP4/FP8 ModelOpt export |
| [NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2) | Draft model | Paired with a compatible target; follow the target model card | Optional speculative draft; not a standalone transcription model |

These profiles describe serving, not fine-tuning memory. Use the
[NeMo Speech fine-tuning guide](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/speechlm2/finetuning/docs/fine-tuning-guide.md)
to size training for the selected method and sequence length.

The three target checkpoints are released under
[OpenMDW-1.1](https://openmdw.ai/license/1-1/). Consult each model card for its
complete hardware matrix, language matrix, benchmark methodology, and safety
information.

## What the model supports

- transcription in verbatim or readable written form;
- vocabulary- and context-biased transcription;
- speaker-attributed and rich transcription;
- optional word timestamps and diarization metadata;
- speech translation and spoken-language identification;
- audio question answering and spoken-instruction following, evaluated only in
  English and not established as reliable in the other transcription
  languages.

The model supports 40 languages, accepts 16 kHz mono audio, and was trained on
recordings up to one hour and sequences up to 65,536 tokens. Its internal
30-second encoder chunks are an implementation detail: the model is offline,
not a real-time streaming recognizer.

## Open recipe coverage

The table distinguishes the model's development stages from what has a public,
maintained recipe today.

| Stage | Public status | Canonical implementation |
|---|---|---|
| Main-stage SALM construction from pretrained components | Available as a parameterized template | [Main-stage training skill](https://github.com/NVIDIA-NeMo/Speech/blob/main/.claude/skills/train-speech-llm/SKILL.md), annotated [`salm_nemotron.yaml`](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/speechlm2/conf/salm_nemotron.yaml), [training entry point](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/speechlm2/salm_train.py), and [HF exporter](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/speechlm2/to_hf.py) |
| Exact released-checkpoint reproduction | Not included | The public template intentionally leaves component checkpoints, production data blends, cluster layout, and other values to the user. It is not the exact configuration used to produce the released checkpoint. |
| Downstream checkpoint fine-tuning and task adaptation | Available | [Fine-tuning skill](https://github.com/NVIDIA-NeMo/Speech/blob/main/.claude/skills/finetune-speech-llm/SKILL.md), [toolkit](https://github.com/NVIDIA-NeMo/Speech/tree/main/examples/speechlm2/finetuning), and [tutorials](https://github.com/NVIDIA-NeMo/Speech/tree/main/tutorials/speechlm2/finetuning) |
| Reinforcement-learning post-training | Not included | This contribution does not claim a public Transcribe-specific RL recipe. Do not infer the production RL setup from a generic language-model recipe. |
| CTC timestamp-head training | Not included | The maintained [timestamp middleware](https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/speechlm2/vllm/salm/ctc_serving.py) provides serving support, not the timestamp-head training recipe; [PR #16344](https://github.com/NVIDIA-NeMo/Speech/pull/16344) is change provenance. |
| DFlash/DFlash2 draft training | Available | [DFlash skill](https://github.com/NVIDIA-NeMo/Speech/blob/main/.claude/skills/train-dflash/SKILL.md), [training guide](https://github.com/NVIDIA-NeMo/Speech/blob/main/docs/source/speechlm2/dflash_training.rst), [entry point](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/speechlm2/dflash_train.py), and [configuration](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/speechlm2/conf/dflash.yaml) |
| FP8 and NVFP4 quantization | Available | [Quantization skill](https://github.com/NVIDIA-NeMo/Speech/blob/main/.claude/skills/quantize-speech-llm/SKILL.md), [scripts](https://github.com/NVIDIA-NeMo/Speech/tree/main/scripts/speechlm2/quantization), and [guide](https://github.com/NVIDIA-NeMo/Speech/blob/main/docs/source/speechlm2/quantization.rst) |
| Inference | Available | The [vLLM usage cookbook](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3.5-Transcribe) and the exact prompts and limits in each model card |
| Reproducible evaluation | Not included | The model cards report benchmark methods and results, but this contribution does not contain the datasets, normalization, scoring, or an automated evaluation harness. |

Public merge provenance: main-stage training [Speech PR #16459](https://github.com/NVIDIA-NeMo/Speech/pull/16459),
fine-tuning [Speech PR #16349](https://github.com/NVIDIA-NeMo/Speech/pull/16349),
DFlash/DFlash2 [Speech PR #16271](https://github.com/NVIDIA-NeMo/Speech/pull/16271),
and quantization [Speech PR #16361](https://github.com/NVIDIA-NeMo/Speech/pull/16361).
Use the maintained `main` paths above for current documentation; use the PRs to
audit the originally reviewed change.

The main-stage and fine-tuning routes serve different starting points. Use
`train-speech-llm` when constructing a SALM from pretrained language and
speech components. Use `finetune-speech-llm` when adapting the released BF16
Transcribe checkpoint; this is the recommended route for domain or task
customization.

## Recommended adaptation flow

1. Start from the BF16 checkpoint and establish a prompt-only baseline.
2. Follow the NeMo Speech fine-tuning skill to prepare manifests, derive the
   configuration from the checkpoint, run coverage checks, train, export, and
   evaluate on both target and general-domain data.
3. Smoke-test the merged BF16 fine-tune with native MTP and CTC timestamps so
   missing auxiliary components are caught before another transformation.
4. If speculative-serving latency matters, train or adapt a DFlash2 draft with
   the frozen fine-tuned target. MTP and DFlash2 are alternative speculative
   decoding modes.
5. Quantize the merged customized BF16 checkpoint to FP8 or NVFP4 with the
   NeMo Speech workflow. Do not substitute NVIDIA's published quantized
   checkpoint, which does not contain the user's fine-tune. Re-evaluate quality
   and timestamp behavior after quantization.
6. Serve with the NeMo Speech vLLM plugin and the configuration for the chosen
   precision.

Main-stage training, fine-tuning, DFlash2, and quantization are separate
workflows with different inputs and validation requirements. Follow the linked
NeMo Speech skill for the stage being run rather than translating commands
between stages.

## Architecture at a glance

| Property | Value |
|---|---|
| Total / active parameters | Approximately 34B total / 4.3B active, including the audio encoder |
| Language backbone | Nemotron 3.5 Lightning hybrid Mamba-2/Transformer MoE |
| Audio path | 0.6B-parameter Transformer ASR encoder plus Nemotron 3 Diarization speaker encoder |
| Speculative decoding | Native MTP; optional separately trained DFlash2 draft |
| Timestamp path | CTC alignment heads exposed through NeMo Speech vLLM middleware |
| Input / output | Audio and text instructions / text, with optional timestamp metadata |

## Important operating constraints

- Use vLLM 0.32.0 with a compatible, GPU-validated NeMo Speech and model
  revision pair. Current checkpoints identify the dedicated
  `Nemotron3_5TranscribeForConditionalGeneration` architecture and require the
  corresponding NeMo Speech loader. Do not mix a checkpoint from before that
  rename with a loader from after it, or the reverse. The cookbook pins
  compatible revisions for reproducibility; run a GPU smoke test before
  production use.
- Transcription, translation, and language identification normally use
  reasoning disabled and temperature 0. Audio reasoning uses different sampling
  settings; follow the model card.
- Timestamp requests accept one recording, non-streaming output, and one
  completion. Check the returned alignment error before consuming timestamps.
- DFlash2 serving and CTC timestamps are not currently combined.
- NVFP4 requires Blackwell. FP8 and NVFP4 use different KV-cache settings; do
  not substitute one launch profile for another.
- Keep special tokens when requesting speaker or rich-transcription tags.

## Related resources

- [Technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Transcribe-Technical-Report.pdf)
- [Usage cookbook](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3.5-Transcribe)
- [Nemotron Transcribe skill](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/skills/nemotron-transcribe/SKILL.md)
- [NeMo Speech main-stage training skill](https://github.com/NVIDIA-NeMo/Speech/blob/main/.claude/skills/train-speech-llm/SKILL.md)
- [NeMo Speech SALM Nemotron configuration](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/speechlm2/conf/salm_nemotron.yaml)
- [NeMo Speech fine-tuning guide](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/speechlm2/finetuning/docs/fine-tuning-guide.md)
- [NeMo Speech repository](https://github.com/NVIDIA-NeMo/Speech)
