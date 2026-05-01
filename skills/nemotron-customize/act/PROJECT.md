# Project Scaffold Brief

**Loaded by:** the main agent, during the Act phase of `/nemotron-customize`,
after plan approval.

You generate the shared project files that wire all stages together. Per-stage
implementations are delegated to sub-agents — you do NOT write stage code here.

## Deliverables

```
{project-name}/                  # kebab-case directory name
├── pyproject.toml               # Dependencies, metadata, ruff config
├── .python-version              # Pin Python 3.12
├── README.md                    # Mermaid diagram, usage, stage table
├── env.toml.example             # Cluster config template
├── {project_name}/              # snake_case Python package (valid identifier)
│   ├── __init__.py
│   ├── __main__.py              # `from .cli import app; app()`
│   ├── cli.py                   # Typer CLI: one command per stage + `all`
│   └── stages/                  # Directory only — populated by sub-agents
└── .generated/
    ├── pipeline.toml            # Canonical stage graph (single source of truth)
    ├── SKILL.md                 # Pipeline-specific skill (invocable as /{project-name})
    └── plugin.json              # .claude-plugin manifest for the generated skill
```

**Naming convention:**
- **`{project-name}`** (kebab-case): top-level directory, skill invocation (`/{project-name}`), Airflow/Kubeflow DAG name.
- **`{project_name}`** (snake_case, valid Python identifier): Python package name used in `python -m {project_name}.cli` and all imports.

If deploy target ≠ local-only, also produce:

- **Airflow**: `deploy/dag.py` — imports stage functions, wires as Airflow tasks
- **Kubeflow**: `deploy/pipeline.py` — KFP components, one per stage

## Rules

### 1. CLI with Typer, dry-run default

One command per stage, plus `all`. Dry-run is the default to prevent accidental
GPU launches.

```bash
python -m {project_name}.cli sft              # prints what would happen (dry-run)
python -m {project_name}.cli sft --run        # actually launches
python -m {project_name}.cli all --run        # all stages sequentially
```

Each CLI command imports and calls the stage function — never subprocess.
`cli.py` is ≤200 lines with no business logic.

**Flag convention:** `--run` is a bool opt-in flag that inverts the
`dry_run=True` default inside each stage function. Implement in Typer as:

```python
@app.command()
def sft(run: bool = typer.Option(False, "--run", help="Execute (default is dry-run)")):
    run_sft(..., dry_run=not run)
```

Do not use Typer's auto-generated `--dry-run / --no-dry-run` pair — the single
`--run` flag is the convention across all generated projects.

### 2. DATA_ROOT + explicit paths

No `${art:}` resolvers. Each stage reads from its predecessor's output directory:

```
$DATA_ROOT/
├── raw/           <- user places input data here
├── translated/    <- stage 1 output = stage 2 input
├── prepared/      <- stage 2 output = stage 3 input
├── sft/           <- stage 3 output
├── eval/          <- stage 4 output
└── converted/     <- stage 5 output (HF model)
```

Document the layout in README. The filesystem IS the artifact graph.

### 3. Tooling is mandatory

- `.python-version`: `3.12`
- `pyproject.toml` includes `[tool.ruff]`
- README uses `uv sync` / `uv run` throughout
- Every dependency imported anywhere in the project must appear in `pyproject.toml`

### 4. `.generated/pipeline.toml` is the canonical stage graph

Records which cookbook steps built this project:

```toml
[[stages]]
id = "01_translate"
step = "translate/translation"
consumes = "filtered_jsonl"
produces = "translated_jsonl"

[[stages]]
id = "02_prep"
step = "prep/sft_packing"
consumes = "translated_jsonl"
produces = "packed_parquet"
```

Don't duplicate this as Python dicts. `cli.py` derives the stage registry at
import time:

```python
import tomllib
_pipeline = tomllib.loads(Path(".generated/pipeline.toml").read_bytes())
STAGES = [s["id"] for s in _pipeline["stages"]]
```

### 5. Generated skill + plugin

`.generated/SKILL.md` + `.generated/plugin.json` make the project invocable as
`/{project-name}`, so the user can run, debug, and iterate via Claude Code.

Keep the generated SKILL narrow: "here's what this pipeline does, here's how to
run each stage, here's the README layout." Don't duplicate nemotron-customize
content.

### 6. `__main__.py` for zero-install runs

```python
from .cli import app

app()
```

Enables `python -m {project_name}` without installing the package.

### 7. W&B off by default, one-line opt-in

The CLI exposes `--wandb-project` on every stage command. First run works with
just DATA_ROOT:

```bash
python -m {project_name}.cli sft --run                         # no tracking
python -m {project_name}.cli sft --run --wandb-project my-exp  # W&B on
```

### 8. Container images live in runspec, not YAML

Training step images go in `[tool.runspec]` and `env.toml`, never hardcoded in
stage YAML.

### 9. Cite patterns in README

One line per influential pattern:

```
This design follows the eval-bookends pattern (eval before and after training).
Packing follows pack-variable-length for heterogeneous SFT data.
```

### 10. Deploy targets share `stages/`

CLI and deploy files share the same `stages/` directory. Neither imports from
the other. README documents both invocation paths:

```
## Run locally
python -m {project_name}.cli all           # dry-run
python -m {project_name}.cli sft --run

## Deploy to Airflow
cp deploy/dag.py $AIRFLOW_DAGS/
airflow dags trigger {project-name}
```

---

## Delegating Stages

After the scaffold is written, spawn one sub-agent per stage. Each sub-agent:

- Loads `skills/nemotron-customize/act/STAGE.md` for the implementation contract
- Loads the correct context pack from `skills/nemotron-customize/context/index.toml`
- Receives from you: the step id, the customer's requirements (model, hardware, params), and the output path

**Brief template** — pass this to each sub-agent:

```
You are implementing stage {NN}_{name} = {step_id}.

Load:
  - skills/nemotron-customize/act/STAGE.md  (implementation contract)
  - {context_pack_path}                     (from context/index.toml)

Plan requirements:
  - Model: {model}
  - Hardware: {gpus}
  - Key params: {params from approved plan}

Output path (absolute or repo-relative): {project_name}/stages/{NN}_{name}/

Deliverables (exactly these files, all under the output path):
  - run.py
  - __init__.py
  - config/default.yaml
  - config/tiny.yaml

Report back: files written, config knobs exposed, UPSTREAM notes (if any),
strategies followed (for the plan's traceability log).
```

Stages can be generated in parallel — they're independent directories.
