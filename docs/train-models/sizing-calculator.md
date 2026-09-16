<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Nemotron Post-training Calculator

Use this calculator to create a directional GPU sizing, training-time, and compute-cost estimate for Nemotron supervised fine-tuning (SFT), low-rank adaptation (LoRA), or group relative policy optimization (GRPO).

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
