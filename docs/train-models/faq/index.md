---
license: Apache-2.0
copyright: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Frequently asked questions about how Nemotron models are trained and how to fine-tune or adapt them, answered by the Nemotron research and engineering teams and from the Nemotron technical reports."
topics: ["Training", "Explanation", "FAQ"]
tags: ["Fine-Tuning", "MoE", "Synthetic Data", "Data Curation", "Pre-Training", "Post-Training", "Reinforcement Learning", "Distillation", "Quantization", "Mid-Training", "Evaluation", "Nemotron 3", "Nemotron 3.5"]
content:
  type: "Explanation"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Developer", "Researcher"]
---

(train-models-faq)=
# Nemotron Training FAQ

Answers to the questions we hear most often from teams training with Nemotron: how the models were built, and how to fine-tune or adapt them on your own data.
The answers come from the Nemotron research and engineering teams and from the published technical reports.
This page grows as more questions are answered; if yours is not here, [open an issue](https://github.com/NVIDIA-NeMo/Nemotron/issues).

For the methodology in full, read the [Nemotron 3 Nano](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf), [Nemotron 3 Super](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf), and [Nemotron 3 Ultra](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf) technical reports, the [Nemotron 3.5 Super and Lightning](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Super-and-Lightning-Technical-Report.pdf) technical report for the latest models, and the [Nemotron-H paper](https://arxiv.org/abs/2504.03624) for the hybrid Mamba-Transformer architecture they build on.
For runnable pipelines, see the [Nemotron 3 Nano](../../nemotron/nano3/README.md), [Nemotron 3 Super](../../nemotron/super3/README.md), [Nemotron 3.5 Super](../../nemotron/super35/README.md), and [Nemotron 3.5 Lightning](../../nemotron/lightning35/README.md) training recipes and the [model training steps](../index.md).
Report-derived answers name the report they come from.
Answers from the Nemotron 3.5 report describe Nemotron 3.5 Super unless they say otherwise.
Other report-derived answers describe the named Nemotron 3 or Nemotron-H models.

## Fine-tuning

### Should I fine-tune from the Base model or the post-trained model?

In most cases, start from the post-trained model.

Start from Base if you plan to re-run the official post-training stages yourself, or in the uncommon case that you do not need the behaviors those stages add.
A domain fine-tune on its own should not be expected to reproduce the instruction following, math reasoning, code knowledge, and other characteristics that post-training adds; recovering them means re-running those stages.
The choice also depends on how much data and compute you have, and on whether your domain is new to the model or one it already knows.

### What learning rate and warmup should I use when fine-tuning Nemotron 3 or 3.5 Super?

The Super pre-training peak is 4.5e-4 after a 200B-token warmup; do not treat it as an SFT default, and expect it to be high for a fine-tune.
The published full-SFT recipe for Super instead uses 1e-5, one forty-fifth of that peak, constant after a 30,000-sample linear warmup.
Judge any departure from that recipe against your dataset, sequence length, and the whole learning-rate schedule, not only the peak.
Nemotron 3.5 Super's SFT landed on the same 1e-5, reached after a warmup of about 9,100 original records and then held constant, with a global batch of 32 packed sequences of up to 524,288 tokens.
The [3.5 Super SFT recipe page](../../nemotron/super35/sft.md) lists those report values next to the recipe defaults, which differ.

Convert warmup steps into processed tokens using your sequence length and global batch size, then compare that with your dataset's token count.
A warmup spanning billions of tokens is too long when the dataset is not at least tens of billions of tokens; a large share of the run is then spent in warmup, and overfitting is likely.

### What can cause loss spikes or repetition collapse during fine-tuning?

Possible causes of early loss spikes include a learning rate that is too high, a poor data distribution, and a bad initialization of the model.

For repetition collapse, check data quality and composition, the training sequence length, the learning-rate schedule, and the initialization.
We are not aware of anything that makes the post-trained Super checkpoint inherently more prone to repetition collapse than Base; look at the data and the recipe first.

### How should I compose my dataset when fine-tuning an MoE model on a narrow domain, to reduce the risk of expert collapse?

Do not train on the narrow domain alone.
Mix the domain-specific data with other data types, keeping the mixture close to the official post-training blend: the Nemotron 3 Nano SFT blend is published in the [technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf) (§3.1.4, Figure 5), and the post-training datasets are released in the [Nemotron Post-Training Datasets](https://huggingface.co/collections/nvidia/nemotron-post-training-datasets) collection.
For a runnable example of an SFT blend built entirely from that open collection, see the [Nemotron 3.5 Lightning SFT recipe](../../nemotron/lightning35/sft.md).
The practical recipe is to take a small slice from every other domain and reassign it to your new domain, so the overall mixture stays close to the original.
This is the recommended mixture strategy, not a guarantee against collapse.

## Pre-training

### How do you set the pre-training data mixture ratios?

Sources of similar estimated quality get similar weight, and higher-quality sources get proportionally more.
That heuristic comes from the Nemotron-H paper and is restated in the Nano and Super reports.
As a separate data-volume constraint, the Nemotron-H paper says more unique tokens enabled long training horizons without exceeding 4 to 8 epochs over the full dataset, and that higher epoch counts led to diminishing returns.

Nano and Super use two-phase curricula in which Phase 1 emphasizes diversity and coverage and Phase 2 shifts to primarily high-quality sources such as Wikipedia; Ultra also uses two phases, described as a shift from a diversity-biased mixture to a quality-biased one.
Nano switches at 94% of the training tokens, Super at 80%, and Ultra at about 75%.
Nemotron-H instead used four phases.
The per-category ratios are published as figures: Nano [Figure 3](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf), Super [Figure 10](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf), and Ultra [Figure 4](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf); in Ultra, quality-filtered and synthetic web crawl makes up about 49% of Phase 1 tokens and 38% of Phase 2 tokens.
The Nemotron-H paper measured the phased approach beating random data ordering by 3.4% at 8B parameters and 1T tokens.
Nemotron 3.5 Super and Lightning were not pre-trained from scratch: each was initialized from its Nemotron 3 counterpart and distilled for 1T tokens, Lightning on the Nemotron 3 Ultra Phase 2 blend with Super 3 as teacher, and Super with the same recipe and Ultra 3 as teacher (3.5 report, §2.2).
Super's Indic continued pre-training and software-engineering mid-training blends are covered under [mid-training](#mid-training-and-continued-pretraining).

### How were the models trained for long context?

The Nano, Super, and Ultra reports each describe a long-context continued-pretraining phase at the end of pre-training, not a separate fine-tuning recipe.

Nano: constant learning rate of 1e-5, global batch size 48, 121B tokens, with 8-way context, tensor and expert parallelism and 4-way pipeline parallelism.
Training on 512k-token sequences alone slightly hurt short-context scores, so the phase mixes 512k and 4k sequences.
The blend is 20% document QA, 1% synthetic retrieval data, and 79% downscaled Phase 2 data.
Long-context SFT data for Nano is synthesized at a mean length of 128k tokens and a maximum of 256k.

Super: constant learning rate of 4.5e-6, global batch size 16, with 64-way context, 2-way tensor and 64-way expert parallelism.
It first trained 34B tokens at 1M context, then 17B tokens alternating 1M and 4k sequences to mitigate a minor impact on math benchmarks.

Ultra: constant learning rate of 2.5e-6, 33B tokens, with 1M-token sequences for 92% of the iterations and 4K for the remaining 8%; math and code SFT-style data appears only in the 4K iterations.

Nemotron 3.5 follows the same pattern, extending both models to 1M-token context on Nemotron 3 Ultra's long-context blend (3.5 report, §2.4).
Lightning: constant learning rate of 1e-5, 21B tokens, with 8-way context, tensor and expert parallelism and 2-way pipeline parallelism, on GB200.
Super: constant learning rate of 4.5e-6, 35B tokens, with 16-way context, 8-way tensor and 64-way expert parallelism, on GB300.
Super needed a further 9B tokens of long-context training after mid-training to recover its RULER score; see the [mid-training section](#did-continued-pre-training-cause-forgetting-and-how-was-it-fixed).

### Do the attention layers in the hybrid architecture need RoPE or another positional encoding?

Not required.
Nemotron-H, Nano, and Super all state that they use no positional embeddings; the Super report calls this consistent with prior Nemotron models and notes that the configuration supports context lengths up to 1M tokens.
Within the Nemotron-H paper, only the Nemotron-T-8B Transformer baseline used RoPE.
The attention layers use Grouped-Query Attention.
Treat this as a design choice rather than a rule: some hybrid Mamba-attention models from other groups keep RoPE in their attention layers.

### How does checkpoint merging work during pre-training, and what does it buy?

In the Super report, checkpoints are saved on an iteration schedule and merge windows are defined in tokens.
Super saved a checkpoint every 1,000 iterations, about 25B tokens at its global batch of 3,072 sequences of 8,192 tokens, and evaluated sliding merge windows of 125B, 250B, and 500B tokens.
A merge is a weighted average of the checkpoints inside the window, with coefficients from a minus-sqrt decay emulation, so a 125B window spans about five checkpoint intervals and a 500B window about twenty.

Merging is cheap relative to training, so all three windows were evaluated at every checkpoint and the best one kept.
During Super's stable learning-rate phase, the best merge consistently beat the corresponding trained checkpoint by 2–4 points on the unweighted average of its 12-benchmark suite, and the report estimates that using merges instead of dedicated decay runs for intermediate evaluation could save roughly 16% of the pre-training FLOP budget.
The base checkpoint selected for alignment was itself a 500B-token merge.

### Did you try merge schedules other than minus-sqrt, or a finer checkpoint granularity?

Not for Super.
Its report says the experiments explored only a single merge schedule (minus-sqrt) and a fixed checkpoint granularity, and that alternative coefficient schemes, finer-grained checkpoint windows, or strategies tailored to longer decay horizons remain plausible.
It also notes that during Super's final 5T-token decay phase the decay-trained checkpoints matched or surpassed the merge readouts, so merging appears most effective for shorter annealing horizons.

Ultra kept the 25B-token checkpoint interval but widened the search: final selection compared merges with windows from 125B to 1T tokens and with sequential, random, and reversed checkpoint orderings.

## Mid-training and continued pretraining

### Does Nemotron use a separate mid-training stage between pre-training and post-training?

Nemotron 3 had no explicitly named one. Nemotron 3.5 Super does: an agentic software-engineering stage that runs after the long-context phase, 65.4B tokens at 256K sequence length, with the learning rate reset, warmed up linearly to 1e-5 (about twice the final pre-training rate) and held there ([3.5 report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Super-and-Lightning-Technical-Report.pdf), §2.5).
The blend in Table 4 is mostly natural data left in its original format, with no chat templates: GitHub issues paired with repository context and the merged patch (41%), whole repositories packed so that files that import each other sit together (20%), agent trajectories from many scaffolds with tool responses masked from the loss (10%), multi-file commit diffs and per-file edit histories (3%) and raw build and CI logs (1%), plus 20% phase-2 pre-training replay and 5% long-context data.
Everything is decontaminated against SWE-bench and Terminal-Bench and scrubbed of secrets.
Lightning has no mid-training stage, and the report does not say why Nemotron 3 went without one.
The [Nemotron 3.5 Super pretraining recipe](../../nemotron/super35/pretrain.md) summarizes the stage.

### Why did Nemotron 3.5 Super add mid-training and Indic continued pre-training?

For different reasons. The 3.5 report explains the thinking behind the software-engineering mid-training and reports the gains from Indic continued pre-training.
Pre-training runs at short context and sees almost no agent interactions, and natural software-engineering data (whole repositories, pull requests with their issues, commit histories, build logs) is often too long for it.
SFT, at the other end, is mostly synthetic, and very long SFT stages can reduce how much the model later gains from RL.
Mid-training fills the gap: it trains the base model at long context on a broad mix of natural data with only a small share of agent traces, and leaves behavior shaping to post-training.
Indic capability came from a separate continued-pre-training run with a nominal 100B-token budget and a 10% Indic allocation, which lifted all five Indic benchmarks by 4.98 to 9.82 points (Table 2).
Neither result is a general rule for when to prefer continued pre-training or mid-training over SFT.
The experimental-SFT comparison below measures what mid-training added; there is no matching comparison of Indic continued pre-training against SFT.

### Did continued pre-training cause forgetting, and how was it fixed?

On long context there was a clear regression, and it was recovered.
Mid-training Nemotron 3.5 Super at 256K context dropped its RULER-1M score from 79.9 to 33.5; a short recovery stage of about 9B tokens, with the same recipe as the 1M-context phase, brought it back to 81.0 (3.5 report, §2.4).
The report records the drop and the fix without calling it catastrophic forgetting.
Both stages carried replay: the mid-training blend kept 20% phase-2 pre-training data and 5% long-context data, and the Indic stage gave 90% of its nominal 100B-token budget to the preceding mixture in its original source proportions.
After the Indic stage, a broad set of English benchmarks showed no regressions in the categories tested; the report does not say how much of that is down to the replay.

### How do you validate a mid-training or continued pre-training stage before the full post-training pipeline exists?

Nemotron 3.5 used two methods, both of which compare trained checkpoints.
To validate mid-training, the same short experimental SFT, an agentic blend of about 33B tokens over 2K steps, was run on the checkpoint before and after the stage and the two were compared.
The team has found this a reliable signal of how a checkpoint will do after full post-training; in that comparison, OpenHands SWE-bench results stayed within run-to-run variation of about one point (3.5 report, §2.5, Table 5).
To choose a mixture before committing the big run, they ablated on the smaller model: the Indic share was swept at 10%, 20% and 30% on Lightning with 200B tokens each (four-benchmark mean 63.1, 64.6 and 65.6; Table 3), which informed Super's 10% allocation.
Ablations like that guide the design but do not pin down an optimal share for the larger model, and neither method validates a blend before any checkpoint exists.

### What does the software-engineering mid-training data look like?

Mostly natural software-engineering data in its original format, 65.4B tokens at 256K context (3.5 report, Table 4; the full blend is in the [first entry of this section](#does-nemotron-use-a-separate-mid-training-stage-between-pre-training-and-post-training)).
Issue–pull-request pairs with repository context (41%) and whole repositories (20%) dominate; agent trajectories (10%), commit histories (3%) and build and CI logs (1%) are smaller shares.
Repositories are packed so that files that import each other sit together, and the agent traces are the only SFT-style data, kept small, with tool responses masked from the loss.
The 3.5 report does not describe Nemotron 3's pre-training code data at this level of detail; the Ultra report covers its code refresh in §2.3.1.

## MoE training

### Beyond loss, load-balancing loss, loss scale, and grad norm, what should I monitor when training an MoE?

In addition to `lm_loss`, `seq_load_balancing_loss`, `loss_scale`, and `grad_norm`, we track `params_norm` and `num_zeros`.

The technical reports describe further signals watched during Nemotron pre-training.
Training and validation cross-entropy are tracked together with the per-head MTP losses; in Ultra's first divergence the MTP-2 loss diverged before the main loss.
The weight-gradient L2 norm rose together with the loss in that divergence.
Expert imbalance is measured with MaxVio, the peak load on any expert divided by the mean load, with dead experts that receive zero or near-zero tokens as the extreme case.
Residual-stream activation norms across depth were examined during Ultra's instability analysis: they differed by three orders of magnitude for Super and four for Ultra. For Ultra, early-layer norms began rising around 7.5T tokens and spiked around 11T; the report says this indicated poor signal propagation but does not identify it as the cause of the divergence.
Under low precision, the count of zero-valued weight-gradient elements is tracked, and the loss gap against BF16 is measured with ablations branched from intermediate checkpoints (5T, 10T, and 16T tokens for Ultra).

## Synthetic data

### Do you use a single teacher model or several?

A mix, across datasets; a single generation pipeline may still rely on one model.

For Nemotron 3.5 the teachers are listed by domain in the report: Kimi K3 for mathematics and STEM trajectories, DeepSeek V4 Pro for mathematical proofs, GLM-5.2 for competitive programming, Qwen3.5 for additional vision data, Kimi K2.6 and K3 for agentic vision data, and Qwen, Kimi and Gemma responses for chat.
The twelve-language Indic pipeline is the single-model case: Gemma-4-31B-it does the translation, the assessment and the rewriting (see the [Data Designer entry](#how-do-you-put-automated-quality-grading-inside-a-nemo-data-designer-pipeline)).

### Is the reasoning format fixed? Do all samples keep their reasoning traces?

The reasoning format is fixed.
All synthetic data is generated with reasoning traces.
For controllable reasoning, we discard the trace on a subset of samples and train on the final response only.

Nemotron 3.5 Super adds reasoning-data filtering, three reasoning modes and inference-time budget control on top of that (3.5 report, §3.1 and §3.9).
Its reasoning SFT data holds chain-of-thought and tool-integrated reasoning trajectories, with any response shorter than 5,000 tokens dropped, and the model is trained for three modes: reasoning off, regular and medium effort.
Medium-effort SFT data comes from Kimi K3's low-effort mode, and about 4% of RL prompts use that mode.
Budget control at inference follows the Ultra recipe: a small fraction of SFT traces are truncated to random reasoning budgets with the final response left intact, and the `</think>` token is masked from the loss.

### Is synthetic data validated after generation?

Yes, through small-scale ablations.
We train various models on the data to check that they learn something from it.
Nemotron 3.5 also validated per sample for its twelve priority Indic languages: each translation is checked against its source and scored from 0 to 100, anything at 75 or above is kept, and lower-scoring translations are rewritten, checked again and kept only if they pass (3.5 report, §2.3; the pipeline is laid out in the [Data Designer entry](#how-do-you-put-automated-quality-grading-inside-a-nemo-data-designer-pipeline)).

### You validate on small models. Does that transfer to large ones?

We generally ablate on models under 10B parameters.
The results seem to generalize better for models larger than about 3B parameters; treat that as an observation, not a guaranteed threshold.
One public example: Nemotron 3.5 Super's Indic allocation came out of mixture ablations on Lightning, which has 30B total and 3B active parameters (3.5 report, §2.3).
That is a design input, not a validation of the under-10B observation above or proof of an optimal share for Super; see [how the stage was validated](#how-do-you-validate-a-mid-training-or-continued-pre-training-stage-before-the-full-post-training-pipeline-exists).

### Does synthetic data go into pre-training or post-training?

Both.
Nemotron uses synthetic data in post-training, and its pre-training corpus also includes synthetic and rephrased sources, which the technical reports document.

The web component includes synthetic counterparts of the higher-quality crawl tiers (`syn-crawl-medium-high` and `syn-crawl-high`, generated by rephrasing filtered web documents); raw Python source was rephrased with an LLM; the corpus includes specialized synthetic datasets for areas such as STEM reasoning and scientific coding; and the blend has explicit `general-sft`, `stem-sft`, and `code-sft` categories, with the Super report noting that reasoning-focused datasets are included in pre-training for their effectiveness.
In Nano's final long-context extension phase, document-QA data is 20% of the blend and synthetic retrieval-focused data 1%, with the remaining 79% downscaled Phase 2 data.
Nemotron 3.5 Super carried this into continued pre-training: its Indic stage mixed synthetic translations of English documents with natural Indic text, while its software-engineering mid-training blend stayed mostly natural data in its original format (3.5 report, §2.3 and §2.5).
See §2.2, §2.3, and §2.5 of the [Nano technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf) and §2.3.7 of the [Super technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf).

### How do you put automated quality grading inside a NeMo Data Designer pipeline?

Nemotron 3.5's translation pipeline for its twelve priority Indic languages is a worked example (3.5 report, §2.3, Figure 2).
It is built in [NeMo Data Designer](https://docs.nvidia.com/nemo/datadesigner/latest/concepts/columns) as four stages, all run by one model (Gemma-4-31B-it) under different prompts: translate a whole document; check the translation against its source, score it from 0 to 100 and list the errors; rewrite anything below 75 from the source, the translation and the error list; score the rewrite and keep or discard it.
Accepted records keep their source ID, language, script, text and quality metadata, and sampled human review across languages and domains feeds back into the prompts and checks.
One caveat the report itself raises: a model judging its own translations may favor them.
A public [Data Designer recipe](https://docs.nvidia.com/nemo/datadesigner/latest/recipes/chart-qa/chart-question-answering) for chart question answering shows the same generate-then-filter approach for vision data, and the linked column types are where a scoring step like stage 2 would go.

## Data curation

### How do you handle samples whose final answer is correct but whose reasoning path is poor?

We generate with multiple models at the same time, which gives us semantically similar samples, and then filter those samples using quality criteria.

Nemotron 3.5 Super's vision SFT data shows what those criteria look like in practice (3.5 report, §3.1).
Its checks drop incomplete reasoning traces, hallucinated labels or values, wrong visual or numerical interpretations, excessive repetition and unproductive self-correction.
SpatialClaw video demonstrations are kept only if they finish normally, match the reference answer and execute the required tools; a second pass rejects malformed actions, execution failures, exceeded display budgets and trajectories with no successful grounded action.

### MinHash LSH deduplication is slow at our scale. What is faster?

Use the GPU-based MinHash implementation in [NeMo Curator](https://github.com/NVIDIA/NeMo-Curator).

### Which curation practices do the reports highlight, from collection through quality verification?

Web data, per the Nemotron-H paper: HTML-to-text extraction, language filtering, global fuzzy de-duplication and exact-substring de-duplication, then an ensemble of three model-based classifiers that buckets each document into five quality tiers.
The paper found that the ensemble yields more diverse high-quality tokens than a single classifier.
Heuristic and perplexity filters are applied to the low, medium-low, and medium tiers, and low-quality tokens are rephrased.
Nano found that uninformative documents such as ads and daily conversations were still scoring high after being translated into English, and added an LLM-based quality pass that removed about 10.6% of the translated subset's tokens and slightly improved benchmark accuracy in an internal ablation.

SFT data, per the Nano report: structural checks discard malformed samples, such as tool calls without tool definitions; reasoning traces with pathological repetition, such as repeated n-grams within a sliding window or across the whole trajectory, are filtered aggressively; and audits of synthetic datasets catch teacher-specific artifacts in traces and responses.

Synthetic tool-calling data, per the Super report: a turn-level and a trajectory-level LLM judge, with the turn-level judge paired with rule-based verification.

Agentic trajectories, per the Ultra report: a heuristic analyzer rejects rollouts with missing submissions, disallowed git operations, edit-test thrashing, lost-in-exploration behavior, malformed tool calls, debug artifacts, or edits that were never tested.

Packing, per the Ultra report: length-aware best-fit packing that never truncates or splits a conversation and forbids duplicate prompts within a pack.

Nemotron 3.5 Super drops reasoning SFT responses shorter than 5,000 tokens (3.5 report, §3.1).
Its SpatialClaw video demonstrations go through the trajectory checks described [above](#how-do-you-handle-samples-whose-final-answer-is-correct-but-whose-reasoning-path-is-poor).
RL data is refreshed and reward-profiled before training (§3.2).
Its [Data Designer pipeline](#how-do-you-put-automated-quality-grading-inside-a-nemo-data-designer-pipeline), [mid-training blend](#what-does-the-software-engineering-mid-training-data-look-like) and [benchmark decontamination](#is-post-training-data-decontaminated-against-benchmarks) have their own entries.

### How do the reports construct and filter multilingual training data?

What the reports document, with the measured effects where they report them:

- Multilingual pre-training data at 3.7–5.0% of the blend depending on model and phase: 9 languages in Nemotron-H, 19 in Nano, 11 in Ultra.
- Translated SFT data with wrong-language outputs and other common failures filtered, plus a lightweight post-editing step that restores the prompt-and-answer format alignment lost in translation (Super).
- For translated math, code and science SFT data, an end-to-end pipeline that translates the full JSON object instead of line by line; ablations on Japanese showed significant MMLU-ProX gains (Ultra).
- Separately, back-translating the translated safety data and dropping examples whose similarity to the English original falls below 0.8, which removed roughly 10–15% of examples per language (Ultra).
- A sentence-level parallel corpus to improve machine translation, expanded with additional Chinese–English pairs (Super).
- An Indic corpus spanning 22 languages (Nemotron 3.5): twelve priority languages go through Gemma-4-31B-it for whole-document translation, assessment and rewriting; ten more are translated with Bodhan-MT and assessed by Gemma-4-31B-it, with language-specific acceptance thresholds. The twelve-language procedure is in the [Data Designer entry](#how-do-you-put-automated-quality-grading-inside-a-nemo-data-designer-pipeline).
- Long-document translation SFT data (Nemotron 3.5 Super): PG19 books and FinePDFs-Edu documents translated in chunks with DeepSeek V4 Flash, with long-translation quality measured by SEGALE. Adding synthetic transduction tasks that exercise position tracking, retrieval, mutation and output improved that score.

### Is post-training data decontaminated against benchmarks?

Yes.
For Nemotron 3.5, all SFT and RL data is checked against 30 benchmarks (3.5 report, §3.12).
String matching is not enough, because contamination arrives as paraphrases, translations and augmentations, and because the same benchmarks guide checkpoint selection, so a contaminated set can also steer you to a worse final model.
The pipeline has four stages: normalize each sample to its problem statement; shortlist candidates against every benchmark question by verbatim n-gram overlap (word 8- and 13-grams) and embedding similarity; have an LLM judge label each shortlisted pair as contamination (direct copy, paraphrase, answer leakage or method leakage) or benign and rate its severity; drop samples judged severely contaminated against any question.
On synthetic contamination built from 13 augmentations, the judge caught 87.5% of Terminal-Bench 2.1 variants and 79.1% of GPQA Diamond variants and flagged a single distractor.
Run over the Nemotron 3.5 post-training data, it removed 20,671 samples (about 0.09%; 14,793 SFT and 5,878 RL) matching 1,808 benchmark questions across 22 of the 30 benchmarks.
The pipeline is slated for release in NeMo Curator.
Pre-training data is not covered at this level in the report.

## Post-training

### Is SFT done in a single pass or in stages?

It depends on the model. Nemotron 3 Super, Ultra and Nemotron-H use stages; Nemotron 3.5 Lightning has a single text-only SFT stage.
Super and Ultra run two SFT stages.
Stage 1 uses token-level loss normalization and shorter packed sequences to build reasoning.
Stage 2 switches to per-conversation normalization so long outputs do not dominate the loss, extends packing (512k tokens for Super, 515,000 for Ultra), and adds long-context data.
The Super report says single-stage SFT led to a marked degradation on long-input, short-output scenarios, which the second stage restores while retaining reasoning.

Nemotron-H also used two stages: math, code, and science reasoning first, at a 5:1 ratio of reasoning to non-reasoning samples, then a roughly 10× smaller stage for instruction following, dialogue, and safety that kept sampling Stage 1 data.

Nemotron 3.5 Super also runs two stages, for a different reason: learning vision and text at the same time is hard (3.5 report, §3.1).
Stage 1 is a general mixture of vision, reasoning, chat, long-context, safety and agentic data; a small stage 2 shifts the blend toward whatever stage 1 left under-learned.
Lightning gets a single text-only SFT stage.

### Does Nemotron have built-in defenses against adversarial or malicious agentic use?

Yes; the Super and Ultra reports document learned defenses. Deployment-time guardrail products are not described in the reports.

Super trains RLVR environments for jailbreak robustness, with PAIR-style iterative attacks run against an early SFT-only checkpoint, and for over-refusal reduction.
Its safety data extends to indirect prompt-injection attacks, and it adds an explicit response-policy framework: each prompt is classified by safety category and by lightweight classifiers for self-harm risk, demographic targeting, and embedded adversarial instructions, then routed to a response mode such as supportive resources, a brief refusal, or answering the benign portion of the request while ignoring the malicious content.

Ultra adds an agentic safety teacher trained on injections hidden in tool responses across four attack categories: unauthorized actions, data modification, denial of service, and data exfiltration.
The data comes from automated red-teaming with Super as the attacker and Nano as the defender, and a deterministic verifier marks an injection as resisted only if the agent does not invoke the attacker's target tool with the target arguments.

Nemotron 3.5 Super largely keeps Ultra's safety-data blend and adds data on dual-risk CBRNE content and demographic biases (3.5 report, §3.1).

### What does the final MTP-only training stage correct?

Two things: it exposes the speculative-decoding head to more inference-like conditions, narrowing the train–inference mismatch, and it distills from the frozen backbone, which improves head–backbone alignment and helps most under non-greedy sampling (3.5 report, §3.7).
Table 8 shows the gap at draft length 7: Super's MTP head accepts 4.589 tokens on average on the SFT checkpoint but 4.287 on the final post-trained model, under greedy decoding.
MTP boosting, the last stage, freezes the backbone and trains only the head on on-policy rollouts from the final backbone.
It keeps the hidden-state mixing from MTP pre-training, so the head sees its own earlier-step states as it would at inference, and swaps cross-entropy for a knowledge-distillation loss against the frozen backbone.
After 25K boosting iterations the acceptance length reaches 4.718 under greedy decoding and 4.612 under sampled drafting, with the biggest gains at the deeper draft positions where the mismatch is worst (Figure 6).
A head boosted on the SFT checkpoint and then carried over keeps much of its gain, so Super's final run starts from one; Lightning skips the warm start and runs 48K boosting iterations against its final backbone.

## Reinforcement learning

### How do you curate and filter RL prompts?

Across the Nano, Super, and Ultra reports, difficulty filtering relative to the policy recurs; Nano additionally bounds verification cost.
Nano profiles every RL task with the SFT checkpoint and drops prompts it already solves with a 100% pass rate; Super drops prompts the SFT model consistently answers correctly; Ultra refreshes RL data and runs reward profiling before training, and for its Competitive Coding Teacher it filters out prompts that the General Reasoning Teacher solves correctly in all 8 of 8 rollouts, leaving 3.5K samples.
Nano caps competitive-coding verification at 50 unit tests to cut verification time, leaving 22K tasks, and applies complexity controls and rejection sampling to structured-output data.
When progress plateaus, Nano re-profiles tasks with the best RL checkpoint.
For Ultra's science-reasoning data, 3,000 held-out problems for RL evaluation were selected to have pass rates between 0.25 and 0.80 and median correct-solution lengths below 64k tokens.

Nemotron 3.5 Super does the same refresh and reward profiling before training, and its specialist teachers add their own checks (3.5 report, §3.2, §3.3.1, §3.4.2 and §3.4.3).
Its agentic teacher's tasks were difficulty-profiled against the step-120 text-RLVR checkpoint, verified finance questions were routed to SFT or RL by measured difficulty, and the vision tool-use teacher kept only questions that are nearly impossible to answer without the tools.
Vision reasoning RLVR data was checked for reference-answer and reward-function compatibility, and exact question–answer duplicates were removed from the second-stage mixture.

### Is there a curriculum over RL prompts?

Yes. Nemotron 3 Nano introduced a Gaussian curriculum, described in its report, and the Super and Ultra reports adopt the same approach.
All prompts are profiled with the SFT checkpoint and those with a 100% pass rate are dropped.
Within each domain the target pass-rate distribution is a Gaussian whose mean decreases linearly over training, so batches move from easier to harder samples while the domain ratios in each batch stay fixed and samples are shuffled.
When progress plateaus, tasks are re-profiled with the best RL checkpoint and a new curriculum is built.
In the Nano report's comparison, curriculum sampling ensured stable learning across multiple domains, whereas random sampling biased the model toward easier tasks and prevented effective learning of harder ones.
The 3.5 report does not describe a pass-rate curriculum; its reasoning teacher trains on a fixed per-batch mix of 90% science, 5% mathematics and 5% competitive coding.

### Which reward signals do you use beyond accuracy, format, and language?

The reports document these additional signals:

- Length rewards: Super adjusts low-effort rollout rewards as a function of correctness and generated-token count, Ultra applies length-based adjustments to medium-effort RL rewards, and Nano's RLHF adds a zero-mean, group-relative length bonus favoring shorter responses plus an optional quality-gated conciseness bonus for the shortest responses that reach top-tier quality.
- An abstention reward, dynamically calibrated during training to balance accuracy against hallucination (Ultra's instruction-following and factuality teacher).
- Safety rewards for over-refusal reduction and jailbreak robustness (Super).
- A deterministic prompt-injection verifier that counts an injection as resisted only if the attacker's target tool is not invoked with the target arguments (Ultra).
- Degree of instruction satisfaction from a rule-based verifier, and general helpfulness from a reward model (Nemotron-H).
- A GenRM judgment conditioned on user-defined principles; during RLHF only its overall score is used as the reward signal (Ultra). Nemotron 3.5 Super's alignment teacher instead averages the GenRM's per-rubric scores and compares each policy response with a fixed response from Gemma 4 31B-it.
- Structure and length shaping in Nemotron 3.5 Super's video reasoning and tracking teacher: a response without exactly one properly matched `<think>...</think>` block gets zero task reward and its advantage overridden to −5. Correct responses keep full task reward up to 1,024 reasoning tokens, after which the multiplier falls linearly to 0.95 at 4,096 tokens and stays there.
- A continuous tracking reward for that teacher's SA-V tasks, scoring localization quality, object visibility, and track validity and completeness (Nemotron 3.5 Super).
- An auxiliary penalty on duplicate and malformed tool calls in Nemotron 3.5 Super's vision tool-use teacher.

### What reward hacking did you see, and how did you mitigate it?

The Nano and Ultra reports document three cases.

1. SWE agents could cheat by reading the gold patch out of the task container. Ultra closes two leak channels: the in-container repository is rewritten to look like a fresh clone at the base commit with future commits physically deleted, and a runtime command filter blocks remote git operations and downloads from GitHub's web, raw-content, and Pages domains.
2. Ultra reports an increasing tendency for larger policies to exploit reward-model weaknesses during RLHF, particularly when the reward model is smaller or less capable. Reasoning GenRMs help mitigate this but substantial failure cases remain, so Ultra scales up the capacity and training data of its GenRM.
3. A case the Nano report treats as distinct from reward hacking: RLHF response length grows mostly because the reasoning trace lengthens while only the final answer is judged. Nano handles this with Group Relative Length Control.

### Is RL strictly on-policy, or asynchronous and off-policy?

The reports show Nemotron moving from synchronous to asynchronous RL.
Nano used synchronous GRPO with on-policy updates at a batch size of 2048, masked importance sampling, and a maximum generation length of 49K tokens.
Super uses one-step off-policy asynchronous GRPO with training and inference on separate GPUs, in-flight weight updates, and no KV-cache recomputation, so a single trajectory can contain tokens produced by different model versions; inference workers are kept at most one step behind the latest model, and the training/inference log-probability mismatch is masked.
Its maximum generation length starts at 49K tokens and later rises to 64K.
Ultra uses one-step off-policy asynchronous training for both RLVR and MOPD.
For MOPD specifically, the maximum generation length is 192K, the behavior policy is decoupled from the proximal policy, and IcePop masking is applied at the token level.

Nemotron 3.5 Super keeps Ultra's asynchronous algorithm and adds router replay and an FP32 LM head to shrink the train–inference gap (3.5 report, §3.2).
Its main RL stage runs a global batch of 8,192 with 16 rollouts per sample, and the maximum generation length starts at 80K tokens and rises to 128K.
Trajectories are at most one update stale for RLVR and the agentic teacher, at most two for the reasoning teacher, and MOPD is likewise one step off-policy (§3.3 and §3.6).

### How do you handle the mismatch between rollout and training log-probabilities?

With masking, and in Nemotron 3.5 by shrinking the mismatch at the source.
The Nano report says it uses masked importance sampling to mitigate training-inference misalignment.
The Super report explicitly masks the importance-sampling ratio computed from the training and inference log-probabilities to reduce mismatch and policy-lag effects.
Neither report states whether the mask is applied per sample, per sequence, or per token.
The Ultra report's asynchronous MOPD explicitly applies IcePop token-level masking plus PPO-style clipping around a proximal policy that is decoupled from the stale behavior policy.

Nemotron 3.5 goes after the gap itself for MoE models: the expert-routing decisions recorded during rollout are replayed when the trainer scores the policy, and the LM head runs in FP32; together these cut the KL error between training and inference by more than an order of magnitude (3.5 report, §3.2 and §3.4.1).
MOPD then applies importance weights per token and masks any token whose weight falls outside [0.2, 5] (§3.6).

### How do you specialize an agent model to a domain without losing general capability?

The reports document several tactics.

- PivotRL (Super): SFT on agentic trajectories is cheap but often degrades performance outside the target domain, while end-to-end RL avoids that outcome to a large part but is costly. PivotRL reuses expert trajectories at informative pivot turns during RL, which the report says greatly improves RL efficiency without the out-of-domain degradation of SFT.
- Mixing RLHF data into domain-focused RLVR (Ultra's instruction-following teacher) to avoid behavioral collapse and overfitting to the training environments.
- A light MOPD warmup SFT (Ultra): Ultra intentionally limits this stage's scale so it induces minimal regression on unrelated domains, and reports that MOPD recovers any residual degradation.
- Self-anchor regularization during MOPD (Nemotron 3.5 Super): in stage 2, the agentic and reasoning environments switch from specialist teachers to a frozen stage-1 checkpoint, which pulls the student back toward its stage-1 behavior on those environments while the remaining specialists are distilled.

### How do you keep agentic training data from being tied to one harness or system prompt?

The Super and Ultra reports vary harnesses and tool formats over shared tasks.

- SFT (Super): the same agentic CLI task sets are run through Codex, OpenCode, Qwen Code CLI, and Stirrup, recorded, and normalized to the OpenAI message format with varied tool definitions.
- SFT (Ultra): SWE trajectories are captured with the OpenHands, SWE-agent, Mini-SWE-agent, and OpenCode harnesses; the report says this multi-model, multi-framework approach promotes generalization across problem-solving modes and agentic environments.
- SWE-RL (Super): OpenCode and Codex agent classes implemented inside OpenHands match the tool formats of Claude Code and Codex CLI, reusing one harness while varying tools and prompts, which improved generalization across all target harnesses at inference time.
- RLVR (Ultra): a diverse collection of harness implementations, to avoid overfitting to any one design.
- Mid-training (Nemotron 3.5 Super): trajectories from many agent scaffolds go into the blend. With identical experimental SFT applied before and after the stage, the mid-trained checkpoint improved under all five additional harnesses tested, by 8.6 points on SWE-bench Verified and 6.8 on Terminal-Bench 2.1 on average. That measures the whole mid-training stage, not the agent traces alone.
- Evaluation (Nemotron 3.5 Super): the final model is scored across six harnesses with unmodified system prompts and standard configurations. Harness rankings differ by benchmark, so the report stresses stating the harness and configuration next to every number.

## Distillation

### Which on-policy distillation objective do you use: sampled token, top-k, or full vocabulary?

For MOPD, sampled-token supervision.
The MOPD objective in the Ultra report is the sampled negative reverse KL on student-generated tokens.
In Ultra's preliminary MOPD experiments, distribution-level matching over the top-k tokens or the full vocabulary did not improve results and consistently underperformed the sampled-token objective on some agentic benchmarks, such as Terminal Bench.
The report hypothesizes that full-distribution matching may impose an overly strong local constraint on student-sampled prefixes that have limited teacher support and may amplify noise from off-support states; it calls characterizing when broader distributional supervision helps an open problem.
The selected sampled-token objective has no k hyperparameter.
Nemotron 3.5 Super's MOPD is sampled-token as well; the objective and its stage-2 target are in [the verifier entry below](#does-on-policy-distillation-use-a-reward-model-or-verifier).
Its MTP boosting is the exception: that distills the full next-token distribution with an LK loss.

### Where does on-policy distillation sit in the pipeline?

The Ultra report places MOPD after RLVR and before MTP boosting, and runs it in two iterations rather than one.
The report also notes that the small regression on unrelated domains introduced by the light warmup SFT is subsequently recovered through MOPD training; it did not test MOPD as a general recovery step for forgetting.
Nemotron 3.5 Super puts MOPD after joint RLVR and RLHF and before the final MTP boosting, starting the student from the step-120 RL checkpoint (3.5 report, §3.6).
Lightning skips MOPD but still gets the final MTP boosting.

### Can on-policy distillation be iterated?

Yes; the Ultra report documents two iterations.
After the first round, Ultra initialized new teachers (chat, conversational tool use, SWE, office tasks, and coding) from the MOPD1 student and distilled them, together with reused first-round teachers, into MOPD2.
MOPD2 improved on MOPD1 on most benchmarks (Terminal Bench 2.0 50.8→54.0, SWE-Bench Verified 70.1→71.7, BrowseComp 41.0→44.4, HLE without tools 25.9→26.7) with a small drop on LiveCodeBench v6 (90.0→89.0).
Two iterations were run; a breakdown point was not reached or reported.

Nemotron 3.5 Super runs two MOPD stages instead of Ultra's rebuild-the-teachers cycle (3.5 report, §3.6).
Some benchmarks peak early and then slide, so stage 1 leans on the fast-improving domains and takes the checkpoint at step 30.
Stage 2 goes after the slower chat and vision domains for 40 steps, one pass over its prompts, with the stage-1 model as both student and self-anchor; the agentic and reasoning teachers are retired and their environments move into the self-anchor stream.

### Does on-policy distillation use a reward model or verifier?

No.
In Nemotron 3.5 Super's MOPD every generated token gets a dense advantage computed from teacher probabilities in one forward pass, with no verifier and no sampling from the teacher (3.5 report, §3.6).
In stage 1 the advantage is the stop-gradient difference between the domain teacher's log-probability of the sampled token and that of a student snapshot frozen before the update: zero where they agree, so the signal lands where the student departs from the specialist.
In stage 2 the teacher side of that difference becomes a frozen mixture of 20% teacher and 80% student-snapshot probabilities, which bounds the advantage below at about -0.22 and steadies training once the student has absorbed one round of teachers.
Only student-generated tokens enter the loss; one-step-stale tokens are importance-weighted and masked outside [0.2, 5].
Prompts from environments no specialist owns go to a self-anchor, a frozen copy of the student's starting checkpoint, which gives no signal while the student still matches it and pulls back departures from it.

## Precision and quantization

### How does AutoQuantize choose per-layer formats, and how is its cost model calibrated?

The cost model is analytic rather than calibrated.
In the Super report, AutoQuantize defines the deployment cost of each operation and format pair as its FLOP count and minimizes the summed sensitivity subject to a total FLOP budget, solved as a knapsack-style optimization.
Sensitivity is a second-order Taylor term inspired by Optimal Brain Surgeon, generalizing LLM-MQ to joint weight-and-activation quantization; the diagonal Hessian is estimated with the diagonal Fisher information, measured on linear-layer outputs against BF16.
Deployment constraints are enforced in the search, such as fused Q/K/V projections sharing one format.
The public [Model Optimizer `auto_quantize` API](https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html) exposes the budget as effective bits rather than FLOPs and describes the sensitivity as gradient-based.

Ultra used Super's sensitivity analysis only as a starting heuristic and chose its own bits-per-element budget empirically, by quantizing a fixed intermediate checkpoint at a range of settings; only long-context AA-LCR was sensitive, plateauing at 5.03 bits per element.

For Nemotron 3.5 the AutoQuant budget is a target effective bitwidth that includes the modeled storage overhead of each format (3.5 report, §4.3).
Sensitivity is quantization-induced output error weighted by the squared task-loss gradient.
Operators that must share a format are grouped, including Q/K/V projections and the routed experts within an MoE layer; the vision encoder, the cache quantizers and the fixed MTP recipe stay outside the search.
Nearly half of the measured NVFP4 sensitivity sits in the LM head, attention and Mamba projections come next, and the MoE experts are comparatively robust.
AutoQuant produced two of the three QAD candidates, but the released Super checkpoint uses a predefined PTQ recipe followed by quantization-aware distillation.

### How do post-training quantization choices differ from the BF16 carve-outs used in low-precision pre-training?

The Super and Ultra reports show two different logics.

Super pre-training ([Table 3](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf) of its report) uses fixed carve-outs by role: the final 15% of the network, latent projections, MTP layers, QKV and attention projections, and embedding layers stay in BF16, the Mamba output projection uses MXFP8 because of underflows seen under NVFP4 at smaller scales, and every other linear layer is NVFP4.

Super post-training quantization ([Table 7](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)) lets AutoQuantize search NVFP4, FP8, and BF16 per eligible operator.
Routed experts land in NVFP4, shared experts in a mix of NVFP4, FP8, and BF16, and attention output, latent, and Mamba projections in FP8 or BF16, while embeddings, output layers, and attention QKV stay BF16 and the KV cache is FP8.
The depth-based "last 15%" rule disappears; precision follows measured sensitivity.

Ultra's post-training recipe ([Table 12](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf)) is a heuristic informed by that analysis: routed experts in NVFP4, shared experts and Mamba mixer linears in per-tensor FP8, attention linears and the latent MoE in BF16, and the SSM cache in FP16 with stochastic rounding.

Nemotron 3.5 Super's released checkpoint follows the aggressive PTQ recipe in [Table 12](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Super-and-Lightning-Technical-Report.pdf) of its report, then 200 iterations of SFT-based quantization-aware distillation.
Routed and shared experts are NVFP4; attention QKV and output projections, the Mamba linear layers and the LM head are FP8; latent projections and embeddings stay BF16, and the released MTP keeps BF16 attention weights.
The KV cache is FP8 and the SSM cache FP16 with stochastic rounding.
The quoted 4.82 bits per element counts stored quantization scales and the unquantized modules but not runtime caches or activations.

The ablations behind that recipe: FP4 Mamba output-projection weights lengthened generations, FP4 LM-head weights disturbed stop-versus-continue decisions, and Q/K/V projections did not need BF16, though per-tensor FP8 limited attention-logit granularity.

Lightning ships as W4A16: NVFP4 routed-expert, shared-expert and LM-head weights with 16-bit activations, FP8 weights and activations in the Mamba input and output projections, and BF16 attention projections.

## Evaluation

### Which NVIDIA tool measures tool-calling success against baselines?

[NeMo Gym](https://github.com/NVIDIA-NeMo/Gym).
All post-trained Nemotron 3.5 Super and baseline results in the 3.5 report were collected through it (§3.8.1); the settings are in its recipes folder, with a [reproducibility tutorial](https://github.com/NVIDIA-NeMo/Gym/blob/main/nemotron_recipes/super-3.5/reproducibility.md).
For conversational tool use it runs the Retail, Airline and Telecom domains of τ²-Bench and the Banking domain of τ³-Bench, each averaged over eight trials, and its BrowseComp harness gives the model search and browse tools in a per-sample workspace.
Every model gets the same agentic resources, tools, prompts, repeats and metric implementations, with sampling parameters taken from its model card.
Some benchmarks had not yet been onboarded and were run with their official implementations or internal scaffolding, so not every number in the report is reproducible in NeMo Gym today.

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
There is also a public data point inside the family.
In Table 10 of the 3.5 report, Nemotron 3.5 Super (120B-A12B) beats Nemotron 3 Ultra (550B-A55B) on SWE-bench Pro, Terminal-Bench 2.1, BrowseComp and GPQA Diamond.
It trails Ultra on SWE-bench Verified, all three reported τ²-Bench domains, IFBench and MMLU-ProX, among others.
The SWE evaluations ran under OpenCode, and Terminal-Bench under Terminus 2 with a three-hour task timeout.
A smaller model can win on particular evaluations, then, but that table does not isolate the recipe as the reason.
With a good dataset and training recipe, a smaller model can close a large part of the gap.
The hardest domains to gain ground in are those that are already heavily optimized, such as math and coding, Python especially, where customization has the least headroom.
The only way to know for your domain is to measure it: evaluate the customized model against the larger one on your own task set, not on public leaderboards alone; see the [model evaluation](../../model-eval/index.md) docs and the per-model evaluation pages in the training recipes.

### How do you know when a training stage is done and it is time to move to the next?

Watch the loss curves, and establish cutoff points experimentally from benchmark evaluations.
In practice this is a rule of the form: when the model reaches X% on benchmark A and roughly Y% on benchmark B, stop this stage, because continuing past that point costs you on benchmark C in the next stage.

The 3.5 report has two examples from Super, neither of them a general threshold (§3.6 and §4.4).
During MOPD, some benchmarks peaked within a few steps and then declined, so the domain weightings were changed between stages and the stage-1 checkpoint was taken at step 30.
In the final quantization-aware distillation, more iterations helped some recipes and not others: going from 200 to 300 iterations helped the 4.8- and 4.9-bits-per-element candidates, the 4.82 candidate was best at 200, and the main gains had largely saturated by 300.
