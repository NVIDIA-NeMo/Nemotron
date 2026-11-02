# Public workflow map

This file routes to maintained NeMo Speech assets. It is not a second copy of
their procedures.

## Stage availability

| Workflow | Status | Maintained source |
|---|---|---|
| Main-stage SALM construction from pretrained components | Available as a parameterized template | [`train-speech-llm`](https://github.com/NVIDIA-NeMo/Speech/blob/main/.claude/skills/train-speech-llm/SKILL.md), [`salm_nemotron.yaml`](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/speechlm2/conf/salm_nemotron.yaml), [`salm_train.py`](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/speechlm2/salm_train.py), and [`to_hf.py`](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/speechlm2/to_hf.py) |
| Exact released-checkpoint reproduction | Not included | The main-stage template leaves production component checkpoints, data blends, cluster values, and other settings to the user and is not claimed as the exact release recipe. |
| Downstream checkpoint fine-tuning and task adaptation | Available | [`finetune-speech-llm`](https://github.com/NVIDIA-NeMo/Speech/blob/main/.claude/skills/finetune-speech-llm/SKILL.md), [toolkit](https://github.com/NVIDIA-NeMo/Speech/tree/main/examples/speechlm2/finetuning), [guide](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/speechlm2/finetuning/docs/fine-tuning-guide.md), [tutorials](https://github.com/NVIDIA-NeMo/Speech/tree/main/tutorials/speechlm2/finetuning) |
| RL post-training | Not included | This contribution does not claim a public Transcribe-specific recipe or skill. |
| Timestamp-head training | Not included | The maintained [timestamp middleware](https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/speechlm2/vllm/salm/ctc_serving.py) provides serving support only; [PR #16344](https://github.com/NVIDIA-NeMo/Speech/pull/16344) is change provenance. |
| DFlash/DFlash2 training | Available | [`train-dflash`](https://github.com/NVIDIA-NeMo/Speech/blob/main/.claude/skills/train-dflash/SKILL.md), [guide](https://github.com/NVIDIA-NeMo/Speech/blob/main/docs/source/speechlm2/dflash_training.rst), [entry point](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/speechlm2/dflash_train.py), [config](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/speechlm2/conf/dflash.yaml) |
| FP8/NVFP4 quantization | Available | [`quantize-speech-llm`](https://github.com/NVIDIA-NeMo/Speech/blob/main/.claude/skills/quantize-speech-llm/SKILL.md), [scripts](https://github.com/NVIDIA-NeMo/Speech/tree/main/scripts/speechlm2/quantization), [docs](https://github.com/NVIDIA-NeMo/Speech/blob/main/docs/source/speechlm2/quantization.rst) |

Reviewed change provenance: main-stage training
[#16459](https://github.com/NVIDIA-NeMo/Speech/pull/16459), fine-tuning
[#16349](https://github.com/NVIDIA-NeMo/Speech/pull/16349), DFlash/DFlash2
[#16271](https://github.com/NVIDIA-NeMo/Speech/pull/16271), and quantization
[#16361](https://github.com/NVIDIA-NeMo/Speech/pull/16361).

## Artifact flow

```text
pretrained language model + pretrained speech encoders
        |
        +--> parameterized main-stage SALM construction --> exported BF16 target

BF16 released checkpoint
        |
        +--> prompt baseline
        |
        +--> downstream fine-tuning / adaptation --> merged HF target --> quality evaluation
                                      |
                                      +--> DFlash2 draft training --> target + draft serving
                                      |
                                      +--> FP8 or NVFP4 quantization --> quantized serving + re-evaluation
```

The main-stage route is a reusable construction template, not exact
released-checkpoint reproduction. Release-specific production inputs, RL,
timestamp-head training, and reproducible evaluation assets are not included
in this contribution.

## Cross-stage invariants

- Read architecture and prompt-format settings from the checkpoint rather than
  starting from a generic SALM configuration.
- Replace every required placeholder and run the upstream preflight before a
  main-stage launch; do not infer production values from the released model.
- Preserve auxiliary MTP and CTC components through export when the downstream
  stage needs them.
- Fine-tune and quantize only after establishing a repeatable BF16 baseline.
- Quantize the merged customized BF16 export. Do not replace it with NVIDIA's
  published FP8 or NVFP4 checkpoint, which does not contain the user's changes.
- Smoke-test the merged BF16 export with MTP and CTC timestamps before
  quantization so missing auxiliary components are caught at the export seam.
- Evaluate target quality, a general-domain guard, and the serving features
  used in production after every transformation.
- Use the destination model card's model-specific speculative-decoding profile
  when it differs from an illustrative default in a generic workflow guide.
- Record the exact NeMo Speech and checkpoint revisions as a compatibility
  pair. The dedicated architecture and loader must both be from after the
  rename; mixing generations fails. The cookbook pins compatible revisions
  for reproducibility; validate them on GPU before production use.
