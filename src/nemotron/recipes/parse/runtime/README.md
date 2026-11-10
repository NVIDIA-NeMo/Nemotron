# Parse training runtime

The Parse recipe uses an isolated environment because its validated Torch and
Transformers versions differ from the root Nemotron launcher dependencies.
The CLI launches stages with this environment's Python and the checked-out
Nemotron `src` directory on `PYTHONPATH`. The root package is not installed into
the training environment, and dependency resolution is not bypassed.

From the Nemotron checkout on a Linux GPU host:

```bash
python3 -m pip install uv==0.13.1
PYTHONPATH=src python3 -m nemotron.recipes.parse.runtime.setup \
  --runtime-dir "$PWD/.nemotron/parse-runtime" --torch-backend cu132
```

On a Slurm cluster, perform installation and all runtime checks inside a compute
allocation. The tested profile uses Python 3.13.16, Torch 2.14.0+cu132,
Transformers 5.16.1 and vLLM 0.33.0. It was validated on H200 with driver
580.126.20. `--torch-backend auto` is available for explicit compatibility
experiments; use a visible target GPU when relying on automatic selection.
The vLLM wheel uses CUDA 13, so selecting a CUDA 12 Torch build alone does not
make this environment CUDA 12 compatible.

The default runtime directory is `.nemotron/parse-runtime` in the current
project, or `NEMOTRON_PARSE_RUNTIME` when set. Sources, the venv, and default uv
cache/Python downloads live below it; explicit existing cache settings are
respected. An explicit directory is recommended for cluster jobs. The installer
records `requested.json`, `runtime.json` and `environment.txt`. Repeating setup
with identical inputs verifies the installed packages and patches without
upgrading them. Changed settings, source edits or package drift require a new
runtime directory. Existing runtimes are never deleted or reset.

The training requirements and both patch files are copied unchanged from the
Parse recipe at `2c2be873c05d904de38c80e55ef5ac754aea180d`. The framework revisions
remain NeMo RL `b0b7dd5597c4bc5c9a728d6d31e2058c698b7423` and Automodel
`b916107a5ce8e77a3f18682f8b0eaa46a4b54389`. The added
`launcher-requirements.txt` declares the small dependency set needed for the
Nemotron Parse CLI and configuration. Package resolution remains fully checked
by `uv pip check`; the installed package freeze is recorded, but compatible
version ranges in the upstream requirements are not a transitive lockfile.

Stage launchers set `VLLM_USE_FLASHINFER_SAMPLER=0` before importing vLLM and
propagate it to workers. This preserves the original notebook kernel's native
sampler choice without depending on a Jupyter kernel installation. It disables
FlashInfer sampling, not every FlashAttention/SDPA training kernel.
