# Nemotron 3 Ultra Knowledge Map

This skill is organized around four source layers:

1. **Skill routing** — how `/nemotron-ultra` should locate and cite sources
2. **Model card** — release identity, staged availability, intended use, variants, deployment facts
3. **Paper chunks** — architecture, training, MOPD, evaluation, quantization, inference, safety
4. **Recipe tracker** — release status only; public Ultra recipe stages are not yet released

---

## Start here

If you need a fast answer:

- `context/quick-reference.md` — compact reference

If you need the authoritative route:

- `SKILL.md` — invocation rules, source priority, caveats, and handoff boundaries
- `model-card.md` — release-facing facts and staged availability
- `paper/` — research and benchmark content
- `recipes/overview.md` — release tracker; **not** a runnable recipe path yet

---

## Question → file map

| Question type | Primary file(s) | Notes |
|---|---|---|
| What is Nemotron 3 Ultra? | `model-card.md`, `paper/_overview.md` | identity, variants, release status, headline claims |
| What are the main contributions? | `paper/_overview.md` | abstract, positioning, contributions, open-sourced checkpoints/datasets |
| How is the architecture built? | `paper/architecture.md` | hybrid Mamba-Attention MoE, LatentMoE, MTP, Table 1 dimensions |
| What does “LatentMoE” mean here? | `paper/architecture.md` | same architectural family as Super3, scaled to 550B/55B |
| How was Ultra pretrained? | `paper/pretraining.md` | NVFP4 recipe, WSD schedule, 20T horizon, long context, stability |
| What data was used? | `paper/data.md` | new/refreshed pretraining datasets, ablations, two-phase mixture |
| How does SFT work? | `paper/sft.md` | SFT hyperparameters, data domains, packing, post-training stage order |
| What is RLVR? | `paper/mopd/overview.md` | unified verifiable-reward RL stage before MOPD |
| What is MOPD? | `paper/mopd/overview.md` | asynchronous multi-teacher on-policy distillation algorithm |
| Which teachers feed MOPD? | `paper/mopd/teachers.md` | specialized teacher roster, training recipes, data sources |
| Why is MOPD warmup needed? | `paper/mopd/warmup-results.md` | teacher/student distribution mismatch and warmup ablations |
| What results does MOPD get? | `paper/mopd/warmup-results.md` | Tables 4–5, recovery rates, limitations |
| What is MTP Boosting? | `paper/mopd/mtp-reasoning.md` | head-only KL distillation for speculative-decoding acceptance |
| How does reasoning effort control work? | `paper/mopd/mtp-reasoning.md` | reasoning-off, regular, medium-effort, budget control |
| What post-training infrastructure was used? | `paper/infrastructure.md` | RL rollout acceleration, GB200/Slurm/Ray scaling, infra bottlenecks |
| What results does Ultra achieve? | `paper/evaluation.md` | base-model and post-trained BF16 benchmark tables; keep separated |
| How does quantization work? | `paper/quantization.md` | NVFP4 PTQ, mixed precision, BPE, SSM cache, W4A16/W4A4 checkpoint |
| How does inference/serving work? | `paper/inference.md` | standalone serving regimes, throughput, MTP, TP/EP, disaggregation |
| Where does safety show up? | `paper/safety.md`, `model-card.md` | SFT safety blend, RLVR safety domain, Agentic Safety teacher, GenRM |
| How do I run Ultra recipes? | `recipes/overview.md` | no public `src/nemotron/recipes/ultra/` tree yet; release tracker only |
| Where is the actual Ultra code? | `recipes/overview.md` | confirms Ultra stage recipe code is absent at time of writing |

---

## Source map

| File | Kind | One-line summary |
|---|---|---|
| `SKILL.md` | Skill router | Reference desk for NVIDIA Nemotron 3 Ultra (550B-A55B): architecture, NVFP4 pretraining, SFT, MOPD, MTP boosting, quantization, and inference. |
| `context/quick-reference.md` | Compact reference | Fast recall for identity, architecture, pretraining, long context, SFT, MOPD, evaluation, quantization, inference, and best next files. |
| `model-card.md` | Model card | Release identity and deployment facts for the NVIDIA Nemotron 3 Ultra 550B-A55B model family. |
| `recipes/overview.md` | Recipe tracker | Release status tracker: no public runnable Ultra recipe stages exist yet. |

---

## Paper map

| File | Main coverage | Best for |
|---|---|---|
| `paper/_overview.md` | 550B/55B MoE overview, 20T pretraining, 1M context, SFT/RL/MOPD, throughput claims, released checkpoints/datasets | quick orientation and headline claims |
| `paper/architecture.md` | Hybrid Mamba-Attention MoE, LatentMoE, two shared-weight MTP heads, 108-layer / 8192-dim Table 1 configuration | architecture questions |
| `paper/pretraining.md` | NVFP4 pretraining recipe, 20T WSD schedule, hyperparameters, long-context extension, training divergences and diagnostics | pretraining and stability questions |
| `paper/data.md` | New/refreshed pretraining datasets, validation ablations, multilingual coverage, two-phase data mixture | data provenance and curriculum questions |
| `paper/sft.md` | Ultra post-training stage order, SFT hyperparameters, SFT domains, length-aware best-fit packing | SFT and post-training setup questions |
| `paper/mopd/overview.md` | Unified RLVR stage plus MOPD objective, async stabilization, dense token-level teacher guidance, training settings | MOPD algorithm questions |
| `paper/mopd/teachers.md` | More-than-ten specialized teachers, their SFT/RL/PivotRL/RLHF recipes, General Reasoning teacher data and scores | teacher-roster and specialization questions |
| `paper/mopd/warmup-results.md` | MOPD warmup, Tables 4–5, recovery rates, discussion, limitations and open problems | MOPD results and caveats |
| `paper/mopd/mtp-reasoning.md` | MTP Boosting, SPEED-Bench acceptance lengths, speculative-decoding improvements, reasoning modes and budget control | MTP and reasoning-effort questions |
| `paper/infrastructure.md` | RL rollout acceleration, GB200/Slurm/Ray scaling, topology/NUMA/checkpoint/JIT/container optimizations | post-training systems questions |
| `paper/evaluation.md` | Base model evaluations and post-trained BF16 evaluations, test-time scaling, harness robustness | benchmark questions |
| `paper/quantization.md` | NVFP4 PTQ via Model-Optimizer, 5.03 BPE selection, FP4 algorithms, SSM cache, single W4A16/W4A4 checkpoint | quantization/deployment questions |
| `paper/inference.md` | Prefill/decode serving regimes, throughput vs frontier MoEs, MTP speculative decoding, TP/EP, disaggregation, all-to-all | inference and serving questions |
| `paper/safety.md` | Safety SFT blend, RLVR safety domain, Agentic Safety teacher, GenRM/RLHF alignment, not-reported safety gaps | safety and limitations |

---

## Common questions → best files

| Common question | Best file(s) | Reminder |
|---|---|---|
| “Is Ultra just Super3 scaled up?” | `paper/architecture.md`, `SKILL.md` | Same architecture family, but Ultra changes scale, post-training, and inference emphasis. |
| “What is the exact training pipeline?” | `paper/sft.md`, `paper/mopd/overview.md`, `paper/mopd/mtp-reasoning.md` | SFT → RLVR → MOPD warmup → MOPD (×N) → MTP Boosting. |
| “How is MOPD different from RLHF?” | `paper/mopd/overview.md`, `paper/mopd/teachers.md`, `paper/safety.md` | MOPD is dense token-level teacher distillation on student rollouts; RLHF appears through the Chat/GenRM teacher. |
| “Which benchmark table should I cite?” | `paper/evaluation.md`, `paper/quantization.md`, `paper/inference.md` | Label base, post-trained BF16, NVFP4, or throughput setting every time. |
| “Can I reproduce the Ultra stages?” | `recipes/overview.md`, then paper chunks | Public Ultra recipe code is absent; do not invent configs or commands. |
| “What is released today?” | `model-card.md`, `recipes/overview.md` | Release is staged: base first; full post-trained/NVFP4 release expected 1H 2026 per repo-card framing. |
| “What does NVFP4 mean here?” | `paper/pretraining.md`, `paper/quantization.md` | Pretraining in NVFP4 and post-training quantization to NVFP4 are separate topics. |
| “Why is inference fast?” | `paper/inference.md`, `paper/quantization.md`, `paper/architecture.md` | Cite the measured serving setting; do not generalize a single multiplier. |
| “Where are safety results?” | `paper/safety.md`, `paper/evaluation.md` | Safety mechanisms are described, but dedicated content-safety/jailbreak benchmark numbers are not reported. |

---

## Important distinctions

### 1. Ultra vs Super3

Ultra reuses the hybrid Mamba-Attention + LatentMoE + shared-weight MTP architecture lineage, but it is not merely “Super3 scaled up.” Ultra is **550B total / 55B active**, uses a redesigned post-training stack centered on **MOPD**, and has a standalone inference section focused on serving regimes and throughput at Ultra scale.

### 2. MOPD is first-class and lives under `paper/mopd/`

Ultra's post-training center of gravity is:

- `paper/mopd/overview.md` — RLVR framing and MOPD algorithm
- `paper/mopd/teachers.md` — specialized teachers
- `paper/mopd/warmup-results.md` — warmup, results, limitations
- `paper/mopd/mtp-reasoning.md` — MTP Boosting and reasoning control

There is no Super3-style `paper/rl/` subtree in this skill.

### 3. Paper vs release tracker

The paper describes the full reported training/evaluation program.
`recipes/overview.md` is **not** a runnable implementation map; it explicitly says no public `src/nemotron/recipes/ultra/` tree exists yet.

Use the paper for:

- claims about why the method works
- benchmark numbers
- internal stage structure
- algorithmic rationale

Use `recipes/overview.md` for:

- release status
- absence of public Ultra recipe stages
- handoff guidance to Super3 analogs or `/nemotron-customize`

### 4. Base vs post-trained BF16 vs NVFP4 vs throughput

Do not mix these:

- **Base** = pretrained / long-context-extended checkpoint before post-training
- **Post-trained BF16** = SFT + RLVR + MOPD + MTP Boosting model evaluated in BF16
- **NVFP4 / W4A4 / W4A16** = post-training quantized deployment checkpoint and hardware-specific serving modes
- **Throughput numbers** = condition-specific serving measurements, often GB200 NVL72, NVFP4, with fixed ISL/OSL settings

### 5. Staged availability

The model card reconciles the report and repo-card framing as a staged release: base checkpoint first, full post-trained / NVFP4 / GenRM release expected 1H 2026. Do not imply every checkpoint or recipe is publicly runnable today.

---

## Suggested read sequences

### For architecture

1. `model-card.md`
2. `paper/architecture.md`
3. `paper/pretraining.md` (if NVFP4 training, long context, or stability is relevant)

### For training methodology

1. `paper/_overview.md`
2. `paper/pretraining.md`
3. `paper/sft.md`
4. `paper/mopd/overview.md`
5. `paper/mopd/teachers.md`
6. `paper/mopd/warmup-results.md`

### For MOPD deep dive

1. `paper/mopd/overview.md`
2. `paper/mopd/teachers.md`
3. `paper/mopd/warmup-results.md`
4. `paper/mopd/mtp-reasoning.md`
5. `paper/infrastructure.md` (if rollout/scaling infrastructure matters)

### For deployment and efficiency

1. `model-card.md`
2. `paper/quantization.md`
3. `paper/inference.md`
4. `paper/evaluation.md`

### For reproduction / public code status

1. `recipes/overview.md`
2. relevant paper chunk for methodology
3. `/nemotron-customize` for procedural/build work

---

## One-line summaries

- **Nemotron 3 Ultra**: 550B total / 55B active hybrid Mamba-Attention LatentMoE model with 1M context.
- **Pretraining**: 20T text tokens under a WSD schedule, trained with NVFP4 recipe and extended to 1M context.
- **SFT**: 204,800 packed samples across long-context, safety, search, terminal/tool use, SWE, math/proof, code, CUDA, RTL, multilingual, and chat domains.
- **MOPD**: RLVR student distilled from more-than-ten specialized teachers via asynchronous dense token-level on-policy guidance.
- **MTP / reasoning**: MTP Boosting improves speculative-drafter acceptance; reasoning modes include reasoning-off, regular, and medium-effort.
- **Quantization / inference**: 5.03 BPE NVFP4 operating point, one W4A4/W4A16 checkpoint, dedicated Ultra-scale serving guidance.
- **Recipes**: public Ultra recipe stages are not yet released; `recipes/overview.md` is a tracker, not runnable instructions.
