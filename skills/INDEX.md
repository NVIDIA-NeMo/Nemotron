# Skills index

Agent entry points for the Nemotron stack. Each skill is self-contained with
its own `SKILL.md` (frontmatter + body) and lives in a sibling directory.

## When to use which skill

| Skill | Use when |
|---|---|
| **[/nemotron-customize](nemotron-customize/SKILL.md)** | The user wants to **build a pipeline** — fine-tune, pretrain, align, evaluate, optimize a model end-to-end. Composes steps from [src/nemotron/steps/](../src/nemotron/steps/) into a runnable Python project. |
| **[/nemotron-nano3](nemotron-nano3/SKILL.md)** | The user wants **facts about Nano3** — architecture, training data, recipe details, eval scores, deployment notes. Reference, not generation. |
| **[/nemotron-super3](nemotron-super3/SKILL.md)** | Same as above, for **Super3**. |
| **[/nemotron-add-step](nemotron-add-step/SKILL.md)** | A contributor wants to **add a new step** to the catalog under [src/nemotron/steps/](../src/nemotron/steps/). |
| **[/nemotron-add-pattern](nemotron-add-pattern/SKILL.md)** | A contributor wants to **encode a cross-cutting decision rule** (tokenizer lock, eval bookends, etc.) under [src/nemotron/steps/patterns/](../src/nemotron/steps/patterns/). |
| **[/nemotron-add-model](nemotron-add-model/SKILL.md)** | A contributor wants to **onboard a new model family** so downstream skills can route to it. |

## Layering

```
skills/                            ← workflow & reference skills (this directory)
└── nemotron-customize/            ← e.g. pipeline-builder skill
    ├── SKILL.md                   ← agent entry point (Orient/Plan/Act/Verify)
    ├── references/
    │   ├── act/                   ← codegen rules loaded during Act phase
    │   │   ├── PROJECT.md         ← project-scaffold rules (R1–R10)
    │   │   └── STAGE.md           ← per-stage rules (R1–R5, dry-run, W&B)
    │   └── context/               ← authored library API extracts for codegen
    │       ├── index.toml         ← (step_id, intent) → pack file
    │       └── README.md          ← provenance notes

src/nemotron/steps/                ← step library (the catalog skills route into)
├── SKILL.md                       ← per-category routing
├── STEPS.md                       ← auto-generated catalog
├── PATTERNS.md                    ← auto-generated pattern index
├── types.toml                     ← artifact-type graph
├── patterns/                      ← decision rules
└── <category>/<step>/             ← each step
    ├── step.toml                  ← machine contract (consumes/produces/params/strategies/errors)
    ├── SKILL.md                   ← agent prose: when/why/gotchas (per-step)
    ├── step.py                    ← runspec + entry point
    └── config/                    ← one or more named configs
```

**Rule of thumb:**

- **Workflow** ("build me a pipeline") → top-level skill in `skills/`.
- **Catalog** (which step does X?) → step library under `src/nemotron/steps/`.
- **Implicit constraints** (tokenizer must match across steps, eval bookends, LoRA merge before eval) → [src/nemotron/steps/patterns/](../src/nemotron/steps/patterns/).

## Validation

Every `SKILL.md` requires a YAML frontmatter block:

```markdown
---
name: <skill-name>
description: <one-line "when to use" hook>
---
```

The files under `nemotron-customize/references/context/*.txt` are short
curated context packs for the Nemotron-stack libraries (Megatron-Bridge,
AutoModel, Curator, NeMo-RL, Evaluator, ModelOpt, Data Designer). They are
read-only reference material for grounding agent changes in the real library
APIs, not runtime code paths.
