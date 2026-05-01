# Stage Implementation Brief

**Loaded by:** each per-stage sub-agent spawned by the main agent during the
Act phase of `/nemotron-customize`.

You generate ONE stage. The main agent gives you:

- The step id (e.g. `sft/megatron_bridge`)
- The customer's requirements from the approved plan (model, hardware, params)
- Which context pack to load from `skills/nemotron-customize/context/index.toml`
- The output path (e.g. `{project_name}/stages/{NN}_{name}/` — use exactly what the main agent specifies)

Your job: read the context pack, adapt the step.py pattern to the customer's
config, and write the stage files. Thin. Runnable. Agent-legible.

> **Note on rule numbering (added Oct 2026):** R1–R3 match the original skill.
> R4 used to be about `.generated/pipeline.toml` being canonical — that rule
> now lives in `act/PROJECT.md` rule #4 because it's a project-level concern.
> R4 here is the old R5 ("constants in config, not code"); R5 here is a new
> rule promoted from the original generation-rules list ("two-tier config").

## Deliverables (exactly these files, under the output path given by the main agent)

```
<output-path>/
├── run.py                # Stage entry point (≤60 lines)
├── __init__.py           # Re-exports only: from .run import run_{stage_name}
└── config/
    ├── default.yaml      # Production config
    └── tiny.yaml         # Quick smoke-test config (10 iters, small data)
```

Do not create shared project files — the main agent owns those.

---

## Stage Implementation Rules (R1–R5)

These prevent the #1 quality problem: stages that reimplement library code
instead of wrapping it.

### R1: Wrap, don't reimplement

Each stage MUST be a thin wrapper (≤60 lines) around the library's public API.
NEVER reimplement logic that exists in a library.

```python
# ✅ CORRECT — prep stage (20 lines):
from nemotron.data_prep.api import run_sft_pipeline

def run_prep(data_root, config, dry_run, **kwargs):
    cfg = load_config(config)
    if dry_run:
        print(f"Would pack {data_root}/translated → {data_root}/prepared")
        return
    run_sft_pipeline(
        blend_path=data_root / "translated",
        output_dir=data_root / "prepared",
        tokenizer=cfg["tokenizer"],
        pack_size=cfg["pack_size"],
    )
```

```python
# ❌ WRONG — prep stage (400 lines):
def tokenize_and_pack(input_path, ...):
    # Reimplements the packing algorithm, chat templates, shard writing...
```

If a library lacks a clean public API, write the minimal shim with a
`# UPSTREAM: need public API for X` comment — not a full reimplementation.

### R2: Named modules, not `__init__.py`

Implementation lives in `run.py`. `__init__.py` is re-exports only:

```python
# stages/sft/__init__.py
from .run import run_sft

__all__ = ["run_sft"]
```

This keeps grep results unambiguous and git blame useful.

### R3: No path archaeology

NEVER locate dependencies via parent directory traversal
(`Path(__file__).parent.parent...`). Use, in order:

1. `importlib.resources` / `pkg_resources`
2. Environment variable (`$MEGATRON_BRIDGE_ROOT`)
3. `shutil.which()` for CLI tools
4. Explicit config parameter with documented default

### R4: Config is the single source of truth

Model-specific values (TP, PP, learning rate, batch size) belong in
`config/*.yaml`, not as magic numbers in Python. Stage code is model-agnostic.
The config makes it model-specific.

```python
# ✅ CORRECT:
cfg = load_config(config_name)
recipe.train.lr = cfg["learning_rate"]
```

```python
# ❌ WRONG:
LEARNING_RATE = 2e-5  # hardcoded at module level
recipe.train.lr = LEARNING_RATE
```

### R5: Two-tier config surface in YAML

Tuning knobs at the top, architecture knobs below. 4–6 tuning knobs visible;
everything else stays in recipe defaults.

```yaml
# === Tuning knobs (change these first) ===
learning_rate: 2.0e-5
max_steps: 1000
lora_rank: 16

# === Architecture (change if you know why) ===
micro_batch_size: 1
global_batch_size: 8
```

---

## Code Quality Standards

Generated code must be clean enough for an AI agent to operate in effectively.

### File size

- **Stage file: ≤60 lines.** If longer, you're reimplementing instead of wrapping.
- **Config file: ≤30 lines.** Just the knobs.

### Naming

- **Directories:** lowercase, underscores (`stages/sft/`, not `stages/SFT/`)
- **Functions:** `run_{stage_name}()` as the public entry point
- **Config files:** `default.yaml` and `tiny.yaml`, always

### Code style

- Type hints on all public function signatures
- Docstring on every `run_*()`: what it does, what it reads, what it produces
- No bare `except:` — catch specific exceptions
- No `print()` for logging — use `logging.getLogger(__name__)` (except in dry-run output, where `print` is fine)
- No commented-out code
- No TODOs without a tracking reference

### Readability for agents

- **One function = one job.** `run()` does: load config → validate inputs → dry-run check → call library API → log result. That's it.
- **No nested helpers.** If a stage needs more than `run()`, it's too complex — you're reimplementing.
- **Explicit > clever.** `subprocess.run(["speaker", "translate", "--config", str(path)])` is better than dynamically built command strings.
- **Flat > nested.** Two levels of nesting max. If you're four levels deep, refactor.

### What an agent needs to modify a stage

An AI agent should be able to:

1. Read the stage file (≤60 lines) and understand it completely
2. See which library function it calls
3. See which config values it passes
4. Change a config value or swap the library call
5. All in one file, no cross-references needed

---

## Stage Behaviour Rules

1. **Use the context pack.** Read the pack listed in the main agent's brief. Adapt, don't copy.
2. **Valid imports only.** Every import must reference a real module from the step's reference code.
3. **No placeholders, no hardcoded paths, no tmpdir.** Every value is a CLI arg or DATA_ROOT-relative. Runtime-generated orchestrator configs (e.g. nemo-run launch files) go to `$DATA_ROOT/<stage>/configs/`, never tmpdir. Do not confuse these with the checked-in `config/default.yaml` — that's a static project file.
4. **Dry-run is default.** The stage function takes `dry_run: bool = True` and prints what would happen. Actual work only fires when the caller passes `dry_run=False`.
5. **W&B off by default.** Accept `wandb_project: str | None = None`. Only enable tracking when it's set.
6. **nemo-run inside the stage, not across stages.** Use `run.LocalExecutor` / `run.SlurmExecutor` inside `run_{stage}()`. No `run.Pipeline` composition — the CLI calls stage functions directly.

### Example shape — a non-training stage (prep)

Data-prep stages call library Python APIs directly:

```python
# stages/prep/run.py
from __future__ import annotations

import logging
from pathlib import Path

from nemotron.data_prep.api import run_sft_pipeline

log = logging.getLogger(__name__)


def run_prep(
    data: Path,
    output: Path,
    tokenizer: str = "nvidia/Nemotron-3-Nano-30B-A3B",
    pack_size: int = 4096,
    dry_run: bool = True,
    wandb_project: str | None = None,  # accepted for CLI uniformity; prep doesn't track
) -> None:
    """Pack training JSONL into Megatron-Bridge Parquet shards.

    Reads JSONL from ``data``, writes packed Parquet + splits manifest to ``output``.
    """
    del wandb_project  # prep stage does not emit W&B metrics
    if dry_run:
        print(
            f"Would pack {data} → {output} "
            f"(tokenizer={tokenizer}, pack_size={pack_size})"
        )
        return
    run_sft_pipeline(
        blend_path=data,
        output_dir=output,
        tokenizer=tokenizer,
        pack_size=pack_size,
    )
    log.info("Prep complete: %s", output)
```

### Training stages (SFT / pretrain / RL)

Multi-GPU training needs a process launcher (torchrun) and lives behind
nemo-run's `Experiment` + `Script` abstraction. **Do not invent the nemo-run
API from memory.** The authoritative reference in this repo is
`src/nemotron/cli/commands/nano3/sft.py` — specifically `_execute_sft()` and
`_execute_remote()`. Copy that pattern and adapt it.

Additional constraints for training stages:

- W&B is **not** configured through a nemo-run tracker. It is driven by env
  vars and the patches in `nemotron.kit.wandb_kit` that the recipe script
  loads. At the stage wrapper level, forward `wandb_project` by setting
  `WANDB_PROJECT` in the executor's env dict — don't call a tracker API.
- The stage wrapper should not import recipe modules directly. Use
  `nemotron.kit.recipe_loader.import_recipe_function` with a string target
  (e.g. `"megatron.bridge.recipes.nemotronh.nemotron_3_nano.nemotron_3_nano_finetune_config"`),
  mirroring the live `steps/sft/megatron_bridge/step.py`.
- If the context pack for your step shows a simpler call shape (e.g. plain
  `subprocess.run(["torchrun", ...])`), prefer the pack over this note.

---

## Handoff Back

When finished, report to the main agent:

- Files written
- Config knobs exposed in `default.yaml`
- Any `# UPSTREAM:` comments added (library gap notes)
- Any strategy skill you followed (for the plan's traceability log)
- Any deviations from the plan that the main agent should cross-check during Verify
