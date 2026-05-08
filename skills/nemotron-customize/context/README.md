# Context packs

Per-step extracts of upstream library documentation. Loaded by Act-phase
sub-agents in `/nemotron-customize` to ground code generation in the real
library API.

## Lookup

[index.toml](index.toml) maps `(step_id, intent)` → pack file. The Act phase
reads this once and dispatches packs to per-stage sub-agents.

## Provenance

Each `*.txt` file is a snapshot of upstream docs + selected source files from
one of:

| Pack file | Upstream | Env var (sanitized) |
|---|---|---|
| `mbridge-*.txt` | NVIDIA-NeMo/Megatron-Bridge | `$MBRIDGE_ROOT` |
| `automodel-*.txt` | NVIDIA-NeMo/Automodel | `$AUTOMODEL_ROOT` |
| `curator-*.txt` | NVIDIA-NeMo/Curator | `$CURATOR_ROOT` |
| `eval-*.txt` | NVIDIA-NeMo/Evaluator | `$EVALUATOR_ROOT` |
| `nemo-rl-alignment.txt` | NVIDIA-NeMo/RL | (linked via URL) |
| `speaker-translation-faith.txt` | NVIDIA Speaker | `$SPEAKER_ROOT` |
| `modelopt-optimization.txt` | NVIDIA Model Optimizer | (linked via URL) |
| `data-designer-sdg.txt` | NVIDIA Data Designer | (linked via URL) |
| `nemotron-data-prep.txt` | NVIDIA-NeMo/Nemotron (this repo) | `$NEMOTRON_ROOT` |

Original contributor-host absolute paths have been replaced with the env vars
listed above.

## nv-base findings

Several packs trigger nv-base SECURITY and PII rules because they contain
upstream library code samples (env-var-based credential setup, install steps
that mention root, optimizer hyperparameter literals, etc.). These are
**documentation excerpts**, not executable paths in this repo. The packs are
read-only docs consumed by the Act phase — no code path here invokes them.

When triaging an nv-base run, file findings under `context/*.txt` separately
from findings under any `SKILL.md` / `act/*.md`.
