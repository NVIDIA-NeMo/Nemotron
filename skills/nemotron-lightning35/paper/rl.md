---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-lightning35"
section: "rl"
paper_sections: ["1", "3", "3.2", "3.6", "3.9", "6"]
title: "Nemotron 3.5 Lightning RL: RLVR Only (Report) vs the Repo's GRPO Recipe"
summary: |
  The report says only that Lightning's post-training ends with RLVR after text-only SFT, with no
  MOPD. It gives no Lightning RL environments, data, hyperparameters or step counts; §3.2's RL
  description (joint RLVR + RLHF, router replay, FP32 LM head, competitive-programming and finance-agent
  environments, GBS 8,192, 16 rollouts, 80K->128K generation) is written for Super. The BF16 model card
  describes multi-environment GRPO in NeMo RL / NeMo Gym with an asynchronous architecture that uses MTP
  to speed rollouts. The repo's Lightning RL recipe is one synchronous GRPO run over six NeMo Gym reward
  environments, 512 prompts x 16 generations, 73,728-token sequences, LR 4e-6, with MTP heads kept
  training; it has no RLHF, reward-model, or DPO stage and no MTP boosting.
key_facts:
  - "§1: 'Nemotron 3.5 Lightning follows a lighter post-training pipeline consisting of single-stage text-only SFT and RLVR.' §3 repeats this: 'Lightning goes through a lighter pipeline, with text-only SFT, and RLVR.'"
  - "§3.6: 'Nemotron 3.5 Lightning, however did not have an MOPD stage.' No specialist teachers are described for Lightning."
  - "Not reported for Lightning: RL environments, data blend, algorithm details, batch size, rollouts per prompt, generation length, LR, number of steps, whether RLHF/GenRM or DPO was used, reasoning-effort modes, router replay / FP32 LM head use."
  - "§3.2 (Super): joint RLVR + RLHF; router replay and FP32 LM head cut train-inference KL error by >10x; competitive-programming env (2,847 train / 372 validation problems); finance-agent env; GBS 8,192, 16 rollouts/sample, max generation 80K then 128K. The section names only Super; do not attribute these to Lightning."
  - "Repo docs / config (model card / repo docs): GRPO in NeMo-RL with NeMo Gym; dataset Nemotron-3.5-Lightning-RL-Training-Blend (placeholder records resolved from DAPO-Math-17k and Skywork-OR1-RL-Data); docs also name the Nemotron-RL Agentic Terminal Pivot dataset as used for some coding-agent capabilities."
  - "Repo config hyperparameters: 512 prompts/step x 16 generations = train GBS 8,192; max_total_sequence_length 73,728; LR 4e-6 constant (min 4e-6, 10 warm-up iters from 4e-7); AdamW beta1 0.9, beta2 0.999, eps 1e-8, wd 0.0, grad clip 1.0; ratio clip 0.2 / 0.28; on-policy KL approximation and importance-sampling correction on; KL penalty 0; token-level loss; leave-one-out baseline; reward normalization; 1 epoch; val every 5 steps; 32 nodes x 8 GPUs."
  - "Repo config MTP during RL: mtp_num_layers 5, repeated-layer MTP, mtp_loss_scaling_factor 0.3, mtp_detach_heads true. The report does not say whether Lightning's MTP head trained during RL."
  - "Repo config environments (6 NeMo Gym resource servers, plus the vllm_model_for_training policy server): math_with_judge (judge disabled), code_gen, workplace_assistant, mcqa, instruction_following, structured_outputs_json. Single-turn rollouts (max_rollout_turns 1). Synchronous (grpo.async_grpo.enabled false). Overlong filtering disabled."
  - "BF16 model card: GRPO across math, code, science, instruction following, multi-step tool use, multi-turn conversations, and structured output environments; asynchronous RL architecture that decouples training from inference and uses MTP to accelerate rollouts. Reasoning on/off via enable_thinking in the chat template; no description of how reasoning-control data was prepared."
  - "Earlier repo docs versions described 'RLVR + RLHF with GenRM + DPO', a DPO hallucination table (8.33% -> 0.7%), reasoning-control percentages (10% / 3%), and '7 reward environments'. That text was copied from Nano 3 and has been removed; treat any quote of it as Nano 3 content."
related_steps: []
currency: "frozen"
---

# Scope

- What RL did the released Lightning model get?
- Which environments, data and hyperparameters were used?
- How does the repo's GRPO recipe relate to the report?

# What the report says (Lightning)

| Statement | Section |
| --- | --- |
| Post-training = single-stage text-only SFT + RLVR | §1, §3 |
| No MOPD stage | §3.6 |
| Lighter, text-only pipeline "optimized for efficient deployment" | §6 |
| MTP boosting runs on the final backbone after RL (48k steps) | §3.7 (see mtp.md) |

Not reported for Lightning: environments, rewards, data, algorithm, any hyperparameter, step count, RLHF,
DPO, reasoning-effort modes, decontamination specifics.

## Super-only RL facts to keep out of Lightning answers

§3.2 never names Lightning. Its contents (joint RLVR + RLHF, refreshed RL data with reward profiling, router
replay, FP32 LM head, the Olympiad competitive-programming environment with 2,847/372 problems, the
SEC-filing finance-agent environment, GBS 8,192 with 16 rollouts, 80K -> 128K max generation) describe
Super. §3.9's medium-effort mode (about 4% of RL prompts) is also stated for Super only.

The repo config happens to use GBS 8,192 with 16 generations per prompt too. That is a repo setting, not
a report statement about Lightning.

# Repo recipe (model card / repo docs)

Sources: `docs/nemotron/lightning35/rl.md`, `src/nemotron/recipes/lightning35/stage2_rl/config/default.yaml`,
`.../config/data_prep/data_blend_raw.json`.

## Data

- Single blend: `nvidia/Nemotron-3.5-Lightning-RL-Training-Blend` (weight 1.0).
- Placeholder records are filled in from BytedTsinghua-SIA/DAPO-Math-17k and Skywork/Skywork-OR1-RL-Data.
- Splits: data prep (`src/nemotron/data_prep/recipes/rl.py`) writes one JSONL per split the HF dataset publishes (`train`, `validation`, `test` become `train`, `val`, `test`); no split ratio is applied. Output goes to `output/stage2_rl_resolved/` with a `manifest.json` that also lists `source_splits`. Training reads `train` and `val`.
- Docs: the release also includes **Nemotron-RL Agentic Terminal Pivot**, "used to train some of the
  coding-agent capabilities" (from the release blog). It is not in the repo RL blend.
- The open blend is "a subset of the full data used to train the released model".

## Environments (config)

| NeMo Gym resource server | Docs description |
| --- | --- |
| math_with_judge (judge disabled in config) | math correctness (DAPO, Skywork) |
| code_gen (1,024 processes, 10 s unit-test timeout) | unit-test pass rate |
| mcqa | STEM multiple choice |
| instruction_following | instruction constraint compliance |
| workplace_assistant | agentic tool use |
| structured_outputs_json | JSON schema adherence |

`max_rollout_turns: 1`. The docs README and rl.md both list these 6.

## GRPO hyperparameters (config)

| Parameter | Value |
| --- | --- |
| Prompts per step x generations | 512 x 16 (train GBS 8,192) |
| Max total sequence length | 73,728 |
| Generation | vLLM, temperature 1.0, top_p 1.0, TP=4, gpu_memory_utilization 0.5, Mamba SSM cache float32 |
| Parsers | tool `qwen3_coder`, reasoning `nano_v3` |
| Loss | ratio clip 0.2 / 0.28; on-policy KL approximation; importance-sampling correction (truncated IS, ratio 5, min 0.2); token-level loss; KL penalty 0 |
| Advantage | leave-one-out baseline; rewards normalized |
| Optimizer | Adam(W), LR 4e-6 constant (min 4e-6), warm-up 10 iters from 4e-7, wd 0.0, betas 0.9 / 0.999, eps 1e-8, grad clip 1.0 |
| MoE | router frozen, router dtype FP32, load balancing "none", aux loss 0 |
| MTP | 5 repeated layers, loss scale 0.3, heads detached |
| Parallelism | TP=4, CP=4, EP=16, PP=1, sequence parallel; 32 nodes x 8 GPUs |
| Schedule | 1 epoch, max steps 1,000,000 (effectively unbounded), validate every 5 steps, checkpoint every 10 |
| Overlong filtering | **false** (truncated generations stay in the loss) |
| Async GRPO | **false** (synchronous) |

# Report vs model card vs recipe

| Topic | Report | BF16 model card | Repo recipe |
| --- | --- | --- | --- |
| RL stages | RLVR only, no MOPD (§1, §3, §3.6) | multi-environment GRPO | one GRPO (RLVR) run; no RLHF/GenRM/DPO |
| Environments | not reported | math, code, science, IF, multi-step tool use, multi-turn, structured output | 6 NeMo Gym servers (above) |
| Async / MTP rollouts | not reported | asynchronous, MTP accelerates rollouts | synchronous; MTP heads trained alongside GRPO |
| MTP boosting | 48k steps after RL (§3.7) | not described | not a recipe stage |
| Reasoning control | not described for Lightning | on/off via `enable_thinking` | no reasoning-control step |

# Caveats

- No catalog step in `src/nemotron/steps` lists Lightning (the RLVR, RLHF and DPO steps do not), so `related_steps` is empty.
- Nothing in the report measures the effect of Lightning RL (no pre/post-RL table).
