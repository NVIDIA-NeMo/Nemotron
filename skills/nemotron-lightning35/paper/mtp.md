---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-lightning35"
section: "mtp"
paper_sections: ["1", "2.1", "2.2", "3.7", "4.7", "Table 8", "Table 9"]
title: "Nemotron 3.5 Lightning MTP: Head Initialization from Super 3, Warm-Up, 48k-Step Boosting, Acceptance Lengths"
summary: |
  Lightning's Multi-Token Prediction (MTP) drafter across the report and the repo. Report: Lightning's
  head is initialized from Nemotron 3 Super Base's MTP head by fusing latent-MoE projections and pruning
  (hidden 4096->2688, MoE intermediate 2688->1856, routed experts 512->128). It gets a 100B-token
  frozen-backbone warm-up with standard 2-step recursive MTP training (no hidden-state mixing, which is
  Super-only there), MTP loss scale 0.1 during 1T-token distillation, and, after post-training, 48k
  steps of MTP boosting with no SFT-stage warm start. The MTP module is not quantized in the NVFP4
  checkpoint. The report gives no Lightning acceptance lengths: Tables 8/9 and Figure 6 are Super-only.
  The repo adds MTP depth/loss settings for pretraining and RL, and serving draft lengths.
key_facts:
  - "§2.1: Lightning and Super both use the recursively shared MTP formulation of NVIDIA (2026b); the MTP head is an internal drafter verified in one backbone pass via rejection sampling or exact token matching."
  - "§2.1 initialization: Lightning's MTP head comes from Nemotron 3 Super Base's MTP head. Latent MoE projections are fused into the experts' up/down projections, then hidden size is pruned 4096 -> 2688, MoE intermediate size 2688 -> 1856, and routed experts 512 -> 128 to match Lightning. (§1 says this is done 'through architecture-aware projection fusion and pruning'.)"
  - "§2.1 warm-up: 100B tokens, backbone frozen, MTP head only. For Lightning 'we applied the standard 2-step recursive MTP training'. The hidden-state-mixing fix for train-inference conditioning mismatch is described for Super only."
  - "§2.2: during Lightning 3.5 Base distillation (1T tokens from Super 3 Base) the MTP loss scaling factor was 0.1."
  - "§3.7: 'For Nemotron 3.5 Lightning, we skip the warm start and apply only MTP boosting.' Backbone frozen, only MTP head parameters updated. 'For Nemotron 3.5 Lightining [sic], we ran MTP boosting for 48k steps against the final backbone' (Super: 4k-step warm start + 25k boosting steps)."
  - "§3.7 boosting recipe (stated generally, then detailed for Super): hidden-state mixing plus a knowledge-distillation loss with the frozen backbone as teacher, the LK loss (Samarin et al., 2026) in place of Ultra's temperature-scaled forward KL. Data: on-policy rollouts from the corresponding backbone at temp 1, 8K-token cap, loss on assistant tokens only, seed prompts from Nemotron-Post-Training-Dataset-v2 and Nemotron-RL-Super-Training-Blends. The report does not say separately whether Lightning used the same data and loss."
  - "Acceptance lengths for Lightning: not reported. Table 8, Table 9 (SPEED-Bench) and Figure 6 cover Super 3.5 only; Super's 4.287 -> 4.718 (greedy) and 3.787 -> 4.612 (temp=1) must not be quoted for Lightning."
  - "§4.7: 'This applies only to Nemotron 3.5 Super; the MTP module of Nemotron 3.5 Lightning is not quantized.' Lightning has no MTP-quantization AL table (Table 20 is Super)."
  - "Repo docs / configs: pretraining uses a repeated-layer MTP module of depth 2 with loss scaling 0.3 (docs/nemotron/lightning35/pretrain.md); RL keeps 5 repeated MTP layers training with loss scale 0.3 and detached heads (stage2_rl config). The recipes README summarizes this as 'depth 2 in pretraining, 5 in RL'."
  - "MTP loss scaling: report §2.2 gives 0.1 for Lightning distillation; the repo pretraining recipe uses 0.3. docs pretrain.md states the difference: they are different runs (report: distillation of the released model; repo: the open pretraining recipe). MTP boosting (§3.7) is not part of the repo recipes; RL only keeps the MTP heads training."
  - "Serving (repo): eval config serves with vLLM speculative_config {method: mtp, num_speculative_tokens: 5}. The vLLM usage cookbook uses MTP with 3 draft tokens on H100 and GB200 and 1 on DGX Spark, and recommends DSpark (separate draft model, 3 draft tokens) on 1x H100 NVFP4 for interactive use. The repo also lists DSpark and DFlash draft models. No acceptance-length or speedup numbers are given in the repo."
related_steps: []
currency: "frozen"
---

# Scope

- How was Lightning's MTP head created and trained?
- How does Lightning's MTP boosting differ from Super's?
- What acceptance lengths does Lightning reach? (Not reported.)
- Is Lightning's MTP quantized, and what draft lengths do the repo's serving configs use?

# Lifecycle of the Lightning MTP head (report)

| Phase | What happens | Section |
| --- | --- | --- |
| Initialization | Copy Nemotron 3 Super Base's MTP head; fuse latent-MoE projections into expert up/down projections; prune hidden 4096 -> 2688, MoE intermediate 2688 -> 1856, routed experts 512 -> 128 | §2.1 |
| Warm-up | 100B tokens, backbone frozen, MTP head only, standard 2-step recursive MTP training | §2.1 |
| Distillation (1T tokens) | MTP layers present; MTP loss scaling 0.1 | §2.2 |
| Post-training (SFT, RLVR) | Not stated whether the MTP head trains | §3 |
| SFT-stage warm start | **Skipped** for Lightning | §3.7 |
| MTP boosting | 48k steps against the final backbone, backbone frozen | §3.7 |
| NVFP4 | MTP module **not quantized** | §4.7 |

## Lightning vs Super

| Item | Lightning | Super |
| --- | --- | --- |
| Init source | Super 3 Base MTP head, fused and pruned | First of Super 3 Base's two MTP heads; second discarded |
| Warm-up recipe | Standard 2-step recursive training | Hidden-state mixing |
| SFT-stage warm start | None | 4k steps |
| Final boosting | 48k steps | 25k steps (initialized from the warm-start head) |
| Reported AL | **None** | Tables 8, 9; Figure 6 |
| MTP quantized in NVFP4 | No | Yes (NVFP4 W4A4 experts; Table 20) |

## Boosting recipe

§3.7 describes the recipe with two components: (1) "continue applying" hidden-state mixing from §2.1, and
(2) a knowledge-distillation loss (LK loss) against the frozen backbone in place of the cross-entropy
used in earlier MTP stages. The data paragraph (on-policy rollouts, temp 1, 8K cap,
Nemotron-Post-Training-Dataset-v2 + Nemotron-RL-Super-Training-Blends seeds) and the "Training
Procedure" paragraph are written around Super; the only Lightning-specific statements are "skip the warm
start" and "48k steps".

Note: §2.1 says hidden-state mixing was used for Super's warm-up and that Lightning used standard 2-step
training. "Continue applying" in §3.7 therefore does not clearly say whether Lightning's boosting used
hidden-state mixing. Record this as unresolved.

# Acceptance lengths

Not reported for Lightning. Table 8 ("Effect of MTP Boosting") lists only Super checkpoints. Table 9
(SPEED-Bench qualitative split, draft length 7) compares Super Native / Boosted with Gemma-4 31B,
Qwen3.5 122B-A10B and Qwen3.8-Flash-Next. Figure 6 is Super. The headline "4.287 -> 4.718 greedy /
3.787 -> 4.612 probabilistic" in §1 and the 10.1% / 21.8% gains in §6 are Super numbers.

# Repo facts (model card / repo docs)

| Item | Value | Source |
| --- | --- | --- |
| MTP module type | Repeated-layer MTP | recipes README, evaluate.md ("the released checkpoint ships a repeated-layer MTP module") |
| Pretraining recipe | depth 2 (`mtp_hybrid_override_pattern: "*E"`), loss scaling 0.3 | docs/nemotron/lightning35/pretrain.md |
| RL recipe | `mtp_num_layers: 5`, `mtp_use_repeated_layer: true`, `mtp_loss_scaling_factor: 0.3`, `mtp_detach_heads: true` | stage2_rl config |
| Eval serving | vLLM `{"method":"mtp","num_speculative_tokens":5}` | stage3_eval config |
| Cookbook serving (vLLM) | MTP: 3 draft tokens on H100 and GB200 (MoE via Triton on Blackwell), 1 on DGX Spark. Recommended 1x H100 NVFP4 setup uses DSpark with 3 draft tokens instead | usage-cookbook vLLM notebook |
| Other drafters | DSpark and DFlash draft models ship with the checkpoints; docs say MTP suits medium/high concurrency and DSpark suits DGX Spark / low concurrency | docs/nemotron/lightning35/README.md |

# Contradictions and open points

- **MTP loss scale**: report §2.2 = 0.1 (Lightning distillation); repo pretraining recipe = 0.3. docs pretrain.md notes the difference; they are different runs.
- **"Depth"**: the report describes a recursively shared head trained with 2-step recursion; the repo describes depth 2 (pretrain) and 5 repeated layers (RL). These are consistent with a single shared layer unrolled several times, but the report does not mention 5-step training for Lightning.
- **Hidden-state mixing in Lightning boosting**: not clearly stated (see above).
- **Draft length**: the report evaluates Super at draft length 7; the repo serves Lightning at 5 (eval) or 3/1 (cookbook). No Lightning draft length is evaluated in the report.

# Caveats

- No serving speedup from MTP is reported for Lightning in the report or the repo.
- The Lightning MTP head keeps full precision in the NVFP4 checkpoint (§4.7); the report does not name the precision (BF16 is likely but not stated).
