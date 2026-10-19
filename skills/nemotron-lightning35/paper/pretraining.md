---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-lightning35"
section: "pretraining"
paper_sections: ["1", "2", "2.1", "2.2", "2.4", "2.6", "6"]
title: "Nemotron 3.5 Lightning Pretraining: MTP Warm-up, 1T-Token Distillation from Super 3 Base, Long Context"
summary: |
  Lightning 3.5 Base is not pretrained from scratch. It starts from Nemotron 3 Nano Base and gets an MTP
  head adapted from Super 3 Base, which is warmed up for 100B tokens with the backbone frozen. It is then
  distilled for 1T tokens from Super 3 Base:
  - teacher signal: top-128 logits on Ultra 3 Phase 2 data, plus a sink token for the residual mass
  - loss: forward KL 0.9 + cross-entropy 0.1
  - LR: ramped from 1e-5 to 1e-4 over 200B tokens, then decayed linearly to 1e-5 over 800B
  - MTP loss scale 0.1; MoE sequence load-balancing coefficient 1e-4; everything else as Nano 3 Base
  After distillation it gets long-context training, for which the report gives no Lightning-specific
  settings. There is no Indic CPT and no agentic mid-training. The Base model card describes the base
  differently (>20T tokens, NVFP4 recipe, MTP continued pretraining, no distillation); the repo docs
  follow the report and note the card. The repo's pretrain recipe trains from random init on open data.
key_facts:
  - "Init: Lightning 3.5 Base from Nano 3 Base, then distilled for 1T tokens (§2.2)."
  - "MTP warm-up: 100B tokens, backbone frozen, only MTP head optimized, standard 2-step recursive MTP training; no hidden-state mixing (Super only) (§2.1)."
  - "Distillation teacher: Super 3 Base; top-128 logits; data: 1T tokens of the Ultra 3 Phase 2 data (NVIDIA, 2026c) (§2.2)."
  - "Loss: standard forward KL with an extra sink token for the probability mass outside the top-128, combined with cross-entropy; weights KL 0.9, CE 0.1 (CE gives signal for rare tokens) (§2.2)."
  - "LR: start at Nano 3 Base's minimum LR 1e-5, ramp 'slowly' to 1e-4 (10% of Nano 3 Base's peak LR) over 200B tokens, then decay linearly back to 1e-5 over the remaining 800B (§2.2)."
  - "All other hyperparameters match Nano 3 Base, except the added MTP layers; MTP loss scaling factor 0.1 (§2.2)."
  - "MoE sequence load-balancing loss coefficient: 1e-4 for Lightning (Super used 1e-5) (§2.2)."
  - "Long context: 'Following distillation, Lightning undergoes long-context training' (§1); no Lightning token count, LR, context length, or parallelism is given. §2.4 numbers are Super's."
  - "Released Lightning did not include Indic CPT (§2.3); no agentic mid-training or LC recovery (§1, §2 intro)."
  - "Base model card conflict: 'pre-trained with over 20T tokens', 'pre-trained using an NVFP4 recipe', MTP layers trained in a 'Continued Pre-Training' phase, and 'no supervised fine-tuning, reinforcement learning, or distillation has been applied'. Not reconciled with the report; give both and attribute each."
  - "Repo recipe (open-data reference implementation, not the report): Megatron-Bridge nemotron_3_5_lightning_pretrain_config, GBS 512, seq 8192, CP=2, EP=8, TP=1, PP=1, MTP depth 2 with loss scaling 0.3."
related_steps: []
currency: "frozen"
---

# Scope
Answers:
- What did Lightning 3.5 Base start from, and in what order were its stages run?
- How was the MTP head warmed up?
- What exactly was the distillation recipe (teacher, logits, loss, weights, LR schedule, load balancing)?
- What does the report say (and not say) about Lightning's long-context training?
- How does the repo's `lightning35 pretrain` recipe relate to the report?

MTP-head construction (projection fusion and pruning) is in architecture.md. Indic ablations are in data.md. Base evaluation (Table 6) is in evaluation-base.md.

# Stage Order (report §1, §2)

| # | Stage | Tokens | LR | Source |
|---|---|---|---|---|
| 0 | Initialize backbone from Nano 3 Base | — | — | §1, §2.2 |
| 1 | MTP head from Super 3 Base (fuse + prune), warm-up with backbone frozen | 100B | not reported | §1, §2.1 |
| 2 | Distillation from Super 3 Base | 1T | 1e-5 -> 1e-4 (200B) -> 1e-5 (800B, linear) | §2.2 |
| 3 | Long-context training | not reported | not reported | §1 ("Lightning undergoes long-context training"), §6 |

Not done for released Lightning: Indic CPT (§2.3: "the released Lightning 3.5 model ... did not include this Indic CPT stage"), agentic SWE mid-training, long-context recovery. §1 and the §2 intro list these under Super only.

# §2.1 MTP Warm-up
- **100B tokens**, backbone parameters frozen, only the MTP head optimized (shared with Super: "Both models undergo a dedicated 100 billion token MTP warm-up with the backbone frozen", §1).
- "For Nemotron 3.5 Lightning, we applied the standard 2-step recursive MTP training."
- The report then explains that this naive training causes a train-inference conditioning mismatch. At training step k the head sees ground-truth hidden states from step k-1. At inference it sees a mix of backbone states and its own synthetic states. The mismatch grows with draft depth and lowers acceptance at later positions. The **hidden-state-mixing** fix is described "in Nemotron 3.5 Super" only.
- Not reported for the warm-up: LR, batch size, sequence length, data, acceptance-length results.

# §2.2 Distillation (1T tokens)
Lightning's recipe is the one the report spells out in full. Super's is then described as "the same distillation recipe as Lightning 3.5 Base but with" changes.

| Setting | Lightning 3.5 Base | Report wording / note |
|---|---|---|
| Initialization | Nano 3 Base | "We initialized ... Lightning 3.5 Base from Nano 3 Base" |
| Token budget | 1T | "distilled each model for 1 trillion tokens" |
| Teacher | Super 3 Base | "the next-larger Nemotron form factor as teacher" (§1) |
| Teacher signal | top-128 logits | "top-128 logits from Super 3 Base" |
| Data | Ultra 3 Phase 2 data | "on 1 trillion tokens of the Ultra 3 Phase 2 data (NVIDIA, 2026c)" |
| Distillation loss | standard forward KL divergence | (Super instead: Skew Reverse KL, alpha 0.1) |
| Residual mass | extra sink token | "to capture the residual probability mass outside top-128" |
| Loss mixing | KL 0.9 + cross-entropy 0.1 | CE added "to provide training signal for rare tokens" |
| LR start | 1e-5 | "the minimum Learning Rate (LR) of Nano 3 Base" |
| LR peak | 1e-4 | "10% of the Nano 3 Base's peak LR" |
| Warm-up / ramp | 1e-5 -> 1e-4 over 200B tokens | "ramped it up slowly"; ramp shape (linear or otherwise) not stated |
| Decay | linear, 1e-4 -> 1e-5 over the remaining 800B tokens | |
| MTP | MTP layers added; MTP loss scaling factor 0.1 | |
| MoE sequence load-balancing loss coefficient | 1e-4 | (Super: 1e-5) |
| Everything else | matches Nano 3 Base | "All other hyperparameters and settings match Nano 3 Base, except for the added MTP layers" |

Rationale stated: start at the minimum LR and ramp slowly "to limit early drift from the initialization".

Implied Nano 3 Base values (derived from the sentence above, not separately stated): minimum LR 1e-5, peak LR 1e-3. For Nano 3 Base's other settings (sequence length, batch size, optimizer), see `skills/nemotron-nano3/paper/pretraining.md`. The 3.5 report does not restate them.

Not reported: global batch size, sequence length, optimizer, hardware, or parallelism for Lightning distillation; distillation loss curves; an ablation of forward KL vs alternatives for Lightning.

# Long-Context Training
- §1: "Following distillation, Lightning undergoes long-context training, while Super proceeds through additional continued-pretraining and agentic mid-training stages."
- Abstract / §6: "Both models combine large-scale distillation, long-context training, and native speculative decoding."
- §2.4 ("Similar to Nemotron 3 series, we added a long-context phase (LC-Phase) at the end of pretraining") gives CPT at 1M context, constant LR 4.5e-6, CP16 / TP8 / EP64 on GB300, the Nemotron 3 Ultra data blend, and 35B tokens. It **does not name Lightning**. The §2 intro attributes the 1M extension to Super ("For Super, we further ... extend the context length to 1M tokens"). The 4.5e-6 LR equals Super's distillation minimum, and 35B tokens matches §1's Super description. **Do not apply §2.4 numbers to Lightning.**
- Lightning's long-context token count, LR, target context length, data, and base-model RULER results: **not reported**. (The post-trained model is evaluated on RULER-v1 at 1M in Table 21: 91.52 BF16 / 90.83 NVFP4.)
- Model cards: "Maximum context length up to 1M tokens". The Base model card reports base RULER 256K 76.88 and RULER 1M 69.62 (NeMo Skills harness); Table 6 has no long-context scores. docs evaluate.md says the base checkpoint's RULER tasks run via nemo-evaluator-launcher with configs in the Gym repo.

# Base Results (pointer)
§2.6 Table 6 reports Lightning 3.5 30B-A3B Base on 24 benchmarks (e.g. MMLU 78.63, GSM8K 91.74, MATH-500 79.53, HumanEval 76.91); see evaluation-base.md. The report gives no stage-by-stage Lightning results (no before/after distillation numbers, no MTP warm-up acceptance lengths).

# Repo Recipe vs Report (`nemotron lightning35 pretrain`)
The repo stage-0 recipe runs Megatron-Bridge's `nemotron_3_5_lightning_pretrain_config` (which extends the Nemotron 3 Nano pretraining config) on open data only. Values from `docs/nemotron/lightning35/pretrain.md` and the recipe configs:

| Parameter | Repo recipe | Report |
|---|---|---|
| Starting point | random initialization | continues from Nano 3 Base + distillation |
| Objective | plain LM pretraining on the open blend | top-128 KD from Super 3 Base, KL 0.9 + CE 0.1 |
| Global batch size | 512 sequences | "match Nano 3 Base" (value not restated) |
| Sequence length | 8,192 | not stated |
| Parallelism | TP 1, PP 1, EP 8 (HybridEP dispatch), CP 2 | not stated for Lightning |
| MTP | repeated-layer module, depth 2, pattern `"*E"`, **loss scaling 0.3** | 2-step recursion; **MTP loss scaling 0.1** during distillation |
| MoE balancing | `seq_aux_loss` (coefficient 1e-4), sigmoid router score, expert bias | MoE sequence load-balancing coefficient 1e-4 |
| Data | single mix: 23 open subsets (Nemotron-CC-Math-v1, CC-v2, CC-v2.1, Pretraining-Specialized-v1), all weight 1.0; no phase curriculum | 1T tokens of Ultra 3 Phase 2 data |
| LR schedule, iterations | inherited from Megatron-Bridge's Nano 3 pretraining config | 1e-5 -> 1e-4 -> 1e-5 over 1T tokens |

The docs state that the recipe does not reproduce the MTP warm-up, the distillation, or the long-context stage. The recipe hardware note says CP=2 is used so that each MTP head's vocab-loss workspace fits on an 80 GiB H100.

# Caveats
- **Report vs Base model card on how the base was made.** Report: Nano 3 Base + 100B MTP warm-up + 1T distillation from Super 3 Base + long context. Base model card: >20T tokens with an NVFP4 recipe, MTP continued pretraining, "no ... distillation". Unreconciled; the repo docs follow the report and note the card. (Earlier repo docs described a 23.5T + 1.5T two-phase from-scratch curriculum with aux-loss-free balancing; that text was Nano 3's and has been removed.)
- **MTP loss scale:** report 0.1 (distillation) vs repo recipe 0.3 (pretraining config and RL). The docs state the difference; they describe different runs.
- **Load balancing:** report (sequence LB loss coefficient 1e-4) and the recipe (`seq_aux_loss`, coefficient 1e-4, plus router expert bias) agree.
- The 200B/800B schedule split, top-128, sink token, 0.9/0.1 weights, and MTP scale 0.1 are stated **for Lightning**. Super reuses them. The 10% peak-LR ratio and 1e-4 LB coefficient are Lightning-only. Super's LRs are 4.5e-6 / 4.5e-5.
- The warm-up and distillation order: §1 says "We begin by adapting the MTP architecture ... Both models undergo a dedicated 100 billion token MTP warm-up ... We subsequently distill both models", so the warm-up precedes distillation.
