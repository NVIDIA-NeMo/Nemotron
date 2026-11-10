# Parse training runtime

The Parse recipe uses an isolated environment because its validated Torch and
Transformers versions differ from the root Nemotron launcher dependencies.
The CLI launches stages with this environment's Python and the checked-out
Nemotron `src` directory on `PYTHONPATH`. The root package is not installed into
the training environment, and dependency resolution is not bypassed.

This v1 requires a source checkout or editable install, including the root
`pyproject.toml` and both packages under `src/`. A standalone wheel installation
is rejected: adding its entire host `site-packages` directory to worker
`PYTHONPATH` could import incompatible launcher dependencies into the runtime.

From the Nemotron checkout on a Linux GPU host, use Python 3.11 or newer to
bootstrap the installer. It creates the training environment with Python 3.13.16.

```bash
python3 -m pip install uv==0.13.1
PYTHONPATH=src python3 -m nemotron.recipes.parse.runtime.setup \
  --runtime-dir "$PWD/.nemotron/parse-runtime"
```

On a Slurm cluster, perform installation and all runtime checks inside a compute
allocation with the target GPU visible. The installer preserves the original
`--torch-backend auto` behavior. The reference H200 run resolved Python 3.13.16,
Torch 2.14.0+cu132, Transformers 5.16.1 and vLLM 0.33.0 with driver 580.126.20.
Automatic selection also allowed the required torchaudio 2.11.0+cpu package.
Forcing `--torch-backend cu132` instead prevents that mixed package resolution
and was rejected by the dependency resolver before training. Explicit backend
overrides are available for compatibility experiments, not as equivalent
reproductions of automatic selection. Actual package builds are recorded in
`runtime.json` and the full freeze; automatic selection is not a transitive lock.
The vLLM wheel uses CUDA 13, so selecting a CUDA 12 Torch build alone does not
make this environment CUDA 12 compatible.

The installer's default runtime directory is `.nemotron/parse-runtime` in the
current project, or `NEMOTRON_PARSE_RUNTIME` when set. Bundled stage profiles
explicitly use `.nemotron/parse-runtime`; when installing elsewhere, pass
`runtime_dir=/absolute/runtime/path` to every `nemotron parse` stage command.
Setting the environment variable alone does not override those stage profiles.
Sources, the venv, and default uv cache/Python downloads live below the selected
directory; explicit existing cache settings are respected. An explicit directory
is recommended for cluster jobs. The installer
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
