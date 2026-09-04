---
license: Apache-2.0
copyright: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Frequently asked questions about how Nemotron models are trained and how to fine-tune or adapt them, answered by the Nemotron research and engineering teams."
topics: ["Training", "Explanation", "FAQ"]
tags: ["Fine-Tuning", "MoE", "Synthetic Data", "Data Curation", "Post-Training", "Nemotron 3"]
content:
  type: "Explanation"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Developer", "Researcher"]
---

(train-models-faq)=
# Nemotron Training FAQ

Answers to the questions we hear most often from teams training with Nemotron: how the models were built, and how to fine-tune or adapt them on your own data.
The answers come from the Nemotron research and engineering teams.
This page grows as more questions are answered; if yours is not here, [open an issue](https://github.com/NVIDIA-NeMo/Nemotron/issues).

For the methodology in full, read the [Nemotron 3 Nano](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf) and [Nemotron 3 Super](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf) technical reports.
For runnable pipelines, see the [Nemotron 3 Nano](../../nemotron/nano3/README.md) and [Nemotron 3 Super](../../nemotron/super3/README.md) training recipes and the [model training steps](../index.md).

## Fine-tuning

### Should I fine-tune from the Base model or the post-trained model?

In most cases, start from the post-trained model.

Start from Base if you plan to re-run the official post-training stages yourself, or in the uncommon case that you do not need the behaviors those stages add.
A domain fine-tune on its own should not be expected to reproduce the instruction following, math reasoning, code knowledge, and other characteristics that post-training adds; recovering them means re-running those stages.
The choice also depends on how much data and compute you have, and on whether your domain is new to the model or one it already knows.

### What learning rate and warmup should I use when fine-tuning Nemotron 3 Super?

The Super pre-training peak is 4.5e-4 after a 200B-token warmup; do not treat it as an SFT default, and expect it to be high for a fine-tune.
The published full-SFT recipe for Super instead uses 1e-5, one forty-fifth of that peak, constant after a 30,000-sample linear warmup.
Judge any departure from that recipe against your dataset, sequence length, and the whole learning-rate schedule, not only the peak.

Convert warmup steps into processed tokens using your sequence length and global batch size, then compare that with your dataset's token count.
A warmup spanning billions of tokens is too long when the dataset is not at least tens of billions of tokens; a large share of the run is then spent in warmup, and overfitting is likely.

### What can cause loss spikes or repetition collapse during fine-tuning?

Possible causes of early loss spikes include a learning rate that is too high, a poor data distribution, and a bad initialization of the model.

For repetition collapse, check data quality and composition, the training sequence length, the learning-rate schedule, and the initialization.
We are not aware of anything that makes the post-trained Super checkpoint inherently more prone to repetition collapse than Base; look at the data and the recipe first.

### How should I compose my dataset when fine-tuning an MoE model on a narrow domain, to reduce the risk of expert collapse?

Do not train on the narrow domain alone.
Mix the domain-specific data with other data types, keeping the mixture close to the official post-training blend: the Nemotron 3 Nano SFT blend is published in the technical report (§3.1.4, Figure 5), and the post-training datasets are released in the [nemotron-post-training-v3](https://huggingface.co/collections/nvidia/nemotron-post-training-v3) collection.
The practical recipe is to take a small slice from every other domain and reassign it to your new domain, so the overall mixture stays close to the original.
This is the recommended mixture strategy, not a guarantee against collapse.

## MoE training

### Beyond loss, load-balancing loss, loss scale, and grad norm, what should I monitor when training an MoE?

In addition to `lm_loss`, `seq_load_balancing_loss`, `loss_scale`, and `grad_norm`, we track `params_norm` and `num_zeros`.

## Synthetic data

### Do you use a single teacher model or several?

A mix.
For Nemotron synthetic-data generation we always use a mix of teacher models rather than one standardized teacher.

### Is the reasoning format fixed? Do all samples keep their reasoning traces?

The reasoning format is fixed.
All synthetic data is generated with reasoning traces.
For controllable reasoning, we discard the trace on a subset of samples and train on the final response only.

### Is synthetic data validated after generation?

Yes, through small-scale ablations.
We train various models on the data to check that they learn something from it.

### You validate on small models. Does that transfer to large ones?

We generally ablate on models under 10B parameters.
The results seem to generalize better for models larger than about 3B parameters; treat that as an observation, not a guaranteed threshold.

### Does synthetic data go into pre-training or post-training?

Both.
Nemotron uses synthetic data in post-training, and its pre-training corpus also includes synthetic and rephrased sources, which the technical reports document.

The web component includes synthetic counterparts of the higher-quality crawl tiers (`syn-crawl-medium-high` and `syn-crawl-high`, generated by rephrasing filtered web documents); raw Python source was rephrased with an LLM; the corpus includes specialized synthetic datasets for areas such as STEM reasoning and scientific coding; and the blend has explicit `general-sft`, `stem-sft`, and `code-sft` categories, with the Super report noting that reasoning-focused datasets are included in pre-training for their effectiveness.
In Nano's final long-context extension phase, document-QA data is 20% of the blend and synthetic retrieval-focused data 1%, with the remaining 79% downscaled Phase 2 data.
See §2.2, §2.3, and §2.5 of the Nano technical report and §2.3.7 of the Super technical report.

## Data curation

### How do you handle samples whose final answer is correct but whose reasoning path is poor?

We generate with multiple models at the same time, which gives us semantically similar samples, and then filter those samples using quality criteria.

### MinHash LSH deduplication is slow at our scale. What is faster?

Use the GPU-based MinHash implementation in [NeMo Curator](https://github.com/NVIDIA/NeMo-Curator).

## Training strategy

### Data, training methodology, architecture, numerical stability: what matters most?

In order: data and training infrastructure first.
High-quality data and stable infrastructure beat everything else.
Then training methodology, such as how SFT and RL are scheduled.
Architecture last.

### Which stage contributes most to model intelligence: mid-training, SFT, or RL?

It depends on what you mean by intelligence.

Mid-training provides the general domain knowledge that every later stage builds on; RL should not be expected to compensate by itself for a weak base model.
SFT changes how the model responds most visibly: it moves evaluation scores and makes the model feel markedly better to humans than the base model, which is often what people mean by adding intelligence.
RL can produce new model characteristics on its own and unlock results the earlier stages did not show.

Internally, we prefer to scale RL and keep the SFT phase minimal.
For a new domain, however, an SFT warmup is generally required.

### Across CPT, SFT, and RL, what matters most: dataset, architecture, or methodology?

The dataset, and especially so for MoE models.
A perfect methodology and architecture cannot overcome poor data; a strong dataset can overcome limitations in both.

### Should I invest in scaling pre-training or in refining post-training?

Post-training, generally.

### Can a smaller model with a better recipe beat a larger one?

There is no single measurement, but in several specialized-domain evaluations we have seen a customized Nemotron 3 Nano beat the proprietary models used as comparators.
With a good dataset and training recipe, a smaller model can close a large part of the gap.
The hardest domains to gain ground in are those that are already heavily optimized, such as math and coding, Python especially, where customization has the least headroom.

### How do you know when a training stage is done and it is time to move to the next?

Watch the loss curves, and establish cutoff points experimentally from benchmark evaluations.
In practice this is a rule of the form: when the model reaches X% on benchmark A and roughly Y% on benchmark B, stop this stage, because continuing past that point costs you on benchmark C in the next stage.
