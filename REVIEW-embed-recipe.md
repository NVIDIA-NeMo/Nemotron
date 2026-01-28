# Embed Recipe: Remaining Items

**Updated:** 2025-02-13
**Scope:** `src/nemotron/cli/embed/`, `src/nemotron/recipes/embed/`

---

## Summary

All P0 (critical) items from the original review have been resolved. The recipe now has:
- `extra="forbid"` on all Pydantic config models
- Field-level validation constraints (bounds, Literal types, model validators)
- Input file existence checks on all 6 stages
- Subprocess output captured in stage 1
- Config defaults aligned across stages 4/5 (ONNX, port 8000)
- .venv directories not tracked in VCS
- CLI commands flattened to uniform 2-word depth
- Mining prefixes configurable
- 110-test suite covering config validation, YAML/model compat, module exports, CLI + dry-run
- Sample corpus bundled for out-of-box testing
- Comprehensive error handling in stage 0 (SDG)
- README accuracy fixes (ports, defaults, troubleshooting)

---

## Remaining: P1 — Before production deployment

| # | Issue | Effort | Notes |
|---|-------|--------|-------|
| 1 | **Replace `print()` with `logging`** across all stages | ~4 hours | No log levels, timestamps, or file output. Impacts Slurm debugging. |
| 2 | **Add version guard to `torch.onnx.export` monkey-patch** | ~30 min | `stage4_export/export.py` — global patch with no version check. Low risk for EA since export runs in isolated env. |

## Remaining: P2 — Production maturity

| # | Issue | Effort | Notes |
|---|-------|--------|-------|
| 3 | Add intermediate checkpointing in stage 1 | ~2 hours | Skip convert if `train.json` exists; skip mining if `train_mined.automodel.json` exists. |
| 4 | Add `nemotron embed status` command | ~4 hours | Scan output directory, report which stages completed with timestamps/sizes. |
| 5 | Add `nemotron embed run-all` pipeline command | ~4 hours | Sequential invocation of all 6 stages. |
| 6 | Build integration smoke test on small mock data | ~8 hours | End-to-end test through stages 1-2 with minimal synthetic data. |
| 7 | Port availability check in stage 5 deploy | ~30 min | Check if `host_port` is in use before Docker run. |
| 8 | Document why deploy container runs as root (`-u root`) | ~10 min | Add comment in code + note in README. |
