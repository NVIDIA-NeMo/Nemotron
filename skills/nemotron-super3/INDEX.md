# Nemotron 3 Super Knowledge Map

This skill is organized around three source layers:

1. **Model card** — identity, release metadata, intended use, supported languages
2. **Paper chunks** — architecture, training, evaluation, quantization
3. **Recipe summaries** — how the released Nemotron repo maps those ideas to runnable stages

---

## Start here

If you need a fast answer:

- `context/quick-reference.md` — compact reference

If you need the authoritative route:

- `model-card.md` — release-facing facts
- `paper/` — research content
- `recipes/` — runnable implementation surface

---

## Question → file map

| Question type | Primary file(s) | Notes |
|---|---|---|
| What is Nemotron 3 Super? | `model-card.md` | identity, variants, release date, languages |
| What are the main contributions? | `paper/_overview.md` | concise summary of the report |
| How is the architecture built? | `paper/architecture.md` | hybrid stack, LatentMoE, MTP |
| What does “LatentMoE” mean here? | `paper/architecture.md` | includes design principles and dimensions |
| How was Super3 pretrained? | `paper/pretraining.md` | phases, hyperparams, NVFP4, checkpoint merging |
| What data was used? | `paper/data.md` | pretraining, SFT, RL datasets and released subsets |
| How does SFT work? | `paper/sft.md` | two-stage loss, reasoning control, stage-1/stage-2 blend |
| What are the RL stages? | `paper/rl/overview.md` | pipeline overview |
| What is RLVR? | `paper/rl/rlvr.md` | multi-environment RL from verifiable rewards |
| Why is SWE-RL separate? | `paper/rl/swe.md` | long-horizon SWE environments and isolation |
| What is the RLHF stage doing? | `paper/rl/rlhf.md` | GenRM, principle-following alignment |
| What results does it achieve? | `paper/evaluation.md` | base, post-trained, and evaluation setup |
| How does quantization work? | `paper/quantization.md` | FP8, NVFP4, AutoQuantize, QAD, Mamba state cache |
| Where does safety show up? | `paper/safety.md`, `model-card.md` | safety data + alignment + ethical considerations |
| How do I run stage 0/1/2/3? | `recipes/overview.md` + matching stage file | released commands, configs, source paths |
| Where is the actual code? | matching `recipes/*.md` | each summary points to `src/nemotron/recipes/super3/...` |

---

## Paper map

| File | Main coverage | Best for |
|---|---|---|
| `paper/_overview.md` | abstract, positioning, headline contributions | quick orientation |
| `paper/architecture.md` | hybrid Mamba-attention stack, LatentMoE, MTP | architecture questions |
| `paper/pretraining.md` | NVFP4 pretraining, data phases, hyperparams, long context | pretraining questions |
| `paper/data.md` | new synthetic corpora, SFT/RL data domains, open-data caveats | data provenance and composition |
| `paper/sft.md` | two-stage SFT loss, reasoning modes, stage-1/stage-2 blend | alignment-before-RL questions |
| `paper/rl/overview.md` | overall RL pipeline and algorithmic framing | RL stage map |
| `paper/rl/rlvr.md` | multi-environment RLVR stage | verifiable-reward training |
| `paper/rl/swe.md` | SWE pivot + SWE-bench RL | software-engineering RL |
| `paper/rl/rlhf.md` | GenRM-backed RLHF | behavioral alignment |
| `paper/evaluation.md` | benchmark setup and results | benchmark questions |
| `paper/quantization.md` | PTQ, NVFP4 recipe, AutoQuantize, QAD | deployment/serving questions |
| `paper/safety.md` | safety-relevant SFT/RL/RLHF content | safety and limitations |

---

## Recipe map

| File | What it summarizes | Underlying source |
|---|---|---|
| `recipes/overview.md` | stage graph and handoff points | `src/nemotron/recipes/super3/` |
| `recipes/stage0_pretrain.md` | 4-phase pretraining curriculum | `stage0_pretrain/` |
| `recipes/stage1_sft.md` | chat → packed parquet → finetune | `stage1_sft/` |
| `recipes/stage2_rl.md` | RL hub and sub-stage chain | `stage2_rl/` |
| `recipes/stage2_rl_rlvr.md` | RLVR config and train path | `stage2_rl/stage1_rlvr/` |
| `recipes/stage2_rl_swe1.md` | SWE-pivot | `stage2_rl/stage2_swe1/` |
| `recipes/stage2_rl_swe2.md` | full SWE-bench | `stage2_rl/stage2_swe2/` |
| `recipes/stage2_rl_rlhf.md` | RLHF config and GenRM plumbing | `stage2_rl/stage3_rlhf/` |
| `recipes/stage3_eval.md` | evaluator config and command flow | `stage3_eval/` |

---

## Important distinctions

### 1. Paper vs released recipe

The paper describes the full internal training program.  
The recipes in this repo are the released implementation surface.

Use the paper for:

- claims about why the method works
- benchmark numbers
- internal phase structure
- algorithmic rationale

Use the recipe summaries for:

- commands
- config names
- source file paths
- containers
- artifact names

### 2. Base vs post-trained vs quantized

Do not mix these:

- **Base** = pretrained checkpoint before SFT/RL
- **Post-trained BF16** = aligned model after SFT + RL
- **FP8 / NVFP4** = post-training quantized deployment variants

### 3. Open-source reproducibility

The docs explicitly note that released recipes use open-source data subsets and should be treated as **reference implementations**.

If the user asks whether they can reproduce the exact paper numbers, point them to:

- `paper/data.md`
- `recipes/overview.md`
- `model-card.md`

---

## Suggested read sequences

### For architecture

1. `model-card.md`
2. `paper/architecture.md`
3. `paper/pretraining.md` (if context length or NVFP4 is relevant)

### For training methodology

1. `paper/_overview.md`
2. `paper/pretraining.md`
3. `paper/sft.md`
4. `paper/rl/overview.md`

### For reproduction

1. `recipes/overview.md`
2. relevant stage file
3. only then the raw source file it names

### For deployment and efficiency

1. `model-card.md`
2. `paper/quantization.md`
3. `paper/evaluation.md`

---

## One-line summaries

- **Nemotron 3 Super**: 120.6B total / 12.7B active hybrid Mamba-attention LatentMoE model with MTP and 1M context.
- **Pretraining**: 25T tokens across a two-phase curriculum, trained in NVFP4, then extended to 1M context.
- **SFT**: 7M+ sample blend with a two-stage loss and three reasoning modes.
- **RL**: RLVR → SWE-RL → RLHF → MTP healing.
- **Quantization**: FP8 for Hopper, NVFP4 for Blackwell, with mixed-precision search to preserve quality.

