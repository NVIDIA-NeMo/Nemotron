# Nemotron 3.5 Transcribe knowledge map

This is intentionally a compact routing skill. Detailed training procedures
remain in NeMo Speech.

## Files

| Question | Read |
|---|---|
| What are the fastest model-selection and workflow-routing facts? | `context/quick-reference.md` |
| What is the model, which checkpoints exist, and what are the runtime constraints? | `model-card.md` |
| Which public workflow handles main-stage training, fine-tuning, DFlash2, timestamps, or quantization? | `recipes/overview.md` |
| How should the skill choose and cite sources? | `SKILL.md` |
| How do I launch and query a vLLM server? | `usage-cookbook/Nemotron-3.5-Transcribe/README.md` and its notebook |
| Where is the user-facing model-family guide? | `docs/nemotron/transcribe35/README.md` |

## Source layers

1. Hugging Face model cards for checkpoint-specific facts and launch settings.
2. NeMo Speech skills and code for executable workflows.
3. The [technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Transcribe-Technical-Report.pdf) for
   model-development, architecture, and reported-evaluation claims.
4. Nemotron docs and cookbook for navigation and concise examples.
