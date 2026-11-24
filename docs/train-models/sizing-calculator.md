<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Nemotron Post-training Calculator

Use this calculator to create a directional GPU sizing, training-time, and compute-cost estimate for customizing a released Nemotron checkpoint with supervised fine-tuning (SFT), low-rank adaptation (LoRA), or group relative policy optimization (GRPO).

Select a model, GPU, and customization workload. The calculator reports a verified memory-fit GPU count and training setting, estimates throughput and runtime from the closest measured benchmark calibration, and calculates compute cost after you enter your own GPU-hour rate.

```{admonition} Planning estimate
:class: caution

This calculator provides a directional planning estimate. It is not a performance guarantee, procurement quote, or service-level commitment. Validate the proposed configuration on your infrastructure before committing capacity or budget.
```

<p>
  <a href="../_static/calculators/nemotron-post-training-calculator.html" target="_blank" rel="noopener noreferrer">Open the calculator in a full browser window</a>.
</p>

<iframe
  src="../_static/calculators/nemotron-post-training-calculator.html"
  title="Nemotron Post-training Calculator"
  style="width: 100%; height: 1200px; border: 1px solid #d9d9d9; border-radius: 8px; background: #fff;"
  loading="lazy">
</iframe>

## Methodology and assumptions

The calculator separates memory fit from performance estimation:

1. **GPU sizing.** It selects the exact or next larger verified memory-fit result for the chosen model, GPU, training method, and maximum sequence length. The memory-fit sizing was performed with Megatron Bridge, and the reported TP, PP, CP, and EP training settings follow Megatron parallelism semantics. The GPU count and parallelism settings are starting points, not proof of the minimum or performance-optimal configuration, and they may not apply directly to other training frameworks or libraries.
2. **Throughput calibration.** For SFT and LoRA, it selects the closest BF16 performance reference by model, hardware, training method, and sequence length. For GRPO, it selects the closest measured run by model family, hardware, placement, training-GPU count, and generated sequence length.
3. **Runtime.** SFT and LoRA runtime is processed tokens divided by aggregate estimated throughput. GRPO estimates generation, policy-training, and log-probability phase times separately. Colocated phases are added; split placement uses the slower side of the asynchronous pipeline as the critical path.
4. **Compute cost.** Compute cost is allocated GPU-hours multiplied by the GPU-hour rate entered by the user. No cloud provider or default price is selected.

The calculator applies a configurable planning factor to measured efficiency and a runtime allowance for operational variability. Sequence distribution, batch shape, parallelism, software versions, networking, checkpointing, data loading, reward functions, and tool or environment latency can materially change observed performance.

GRPO generation efficiency is especially workload-dependent. Long-running agent tasks can spend substantial time waiting for tools, environments, or external services; the measured generation references may therefore differ from the target workload.

Calibration links in the calculator point to public [Megatron Bridge performance](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html), [NeMo AutoModel performance](https://docs.nvidia.com/nemo/automodel/latest/performance/performance-summary), and [NeMo RL performance](https://docs.nvidia.com/nemo/rl/latest/about/performance-summary.html) documentation. Some exact calibration runs or detailed run records may not be publicly available.

The calculator assumes BF16 training. Its training-time estimate covers the modeled GPU execution; it does not include queueing, data preparation, evaluation, deployment, or other work outside the modeled training run.

### Nemotron-3.5-Super-VL sizing notes

Nemotron-3.5-Super-VL uses Nemotron-3-Super as the reference for GPU sizing and training-time estimates because both models share the same language-backbone architecture.

- **Text-only training:** Assumes the vision encoder and projector are excluded, resulting in memory requirements and training performance similar to Nemotron-3-Super.
- **Vision-language training:** Assumes the vision encoder is frozen and the projector and language model are trainable, consistent with the published Megatron Bridge SFT recipes. The additional parameter-state memory is relatively small compared with the language backbone, so the calculator uses Nemotron-3-Super as the sizing and performance reference.
- **Visual-token estimation:** Both text and visual tokens contribute to maximum sequence length and total dataset tokens. Visual-token counts depend on image or video resolution and preprocessing. Estimate them with the [Nemotron-3.5-Super-VL Hugging Face image processor](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16/blob/main/image_processing_nemotron_h_omni.py).

### Nemotron-3.5-Decision-8B sizing notes

Nemotron-3.5-Decision-8B is a dense 8B bidirectional text model. The calculator supports LoRA and full SFT for it. GRPO is not available because the model has no RL calibration.

- **GPU sizing:** Memory-fit rows for this model are formula-based estimates, not Megatron Bridge measurements. They follow the Decision training recipe, which supports data parallelism only (TP, PP and CP are 1). Full SFT keeps float32 parameters, gradients and AdamW states (16 bytes per trainable parameter) with BF16 compute; LoRA keeps BF16 weights. The estimates also assume activation recompute and 80% usable GPU memory. The calculator labels these rows as estimated. H100 at 128K tokens is N/A, because without context parallelism a single 128K sequence leaves too little room on 80 GB. The only run on this model was a full SFT job on 8 GPUs with short examples (8,192-token cap).
- **Throughput:** No bundled benchmark was measured on this model. MFU is transferred from the Nemotron-3-Super AutoModel H100 BF16 reference and applied to this model's own FLOPs per token, so tokens/s/GPU and runtime are directional estimates. Attention uses bidirectional FLOPs, which are twice the causal count per attended token.
- **Parameters:** The model has 7.42B trainable decoder parameters. The embeddings and the diffusion head stay frozen during training.
- **DGX Station:** Sized with the same estimated memory-fit rows (one station) and the same transferred MFU, using the Nemotron-3-Super GB300 reference. No DGX Station run was measured on this model, so the default 60% planning factor applies instead of the 90% used for measured Station rows. Treat the DGX Station throughput as an optimistic upper bound: it carries no GB300-specific efficiency discount, and measured Station rows for other models show lower MFU.
