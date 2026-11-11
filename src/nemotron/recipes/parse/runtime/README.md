# Parse runtime setup

Parse uses a separate Python environment to keep its training and inference
dependencies independent of the main Nemotron environment.

## Requirements

- Linux with an NVIDIA GPU and a driver compatible with CUDA 13, used by the
  current runtime dependencies.
- Python 3.11 or newer to run the installer, Git, and network access for
  dependency and framework downloads.
- A full Nemotron source checkout. An editable install from that checkout also
  works; a standalone wheel installation is not supported.

## Install

Run these commands from the Nemotron repository root with the GPU visible:

```bash
python3 -m pip install uv==0.13.1
export PYTHONPATH="$PWD/src"
PARSE_RUNTIME="$PWD/.nemotron/parse-runtime"

python3 -m nemotron.recipes.parse.runtime.setup \
  --runtime-dir "$PARSE_RUNTIME"
```

The installer creates the environment, prepares the required framework sources
and patches, and checks dependency compatibility. Leave `--torch-backend` at its
`auto` default so it can select package builds for the available GPU.

## Use the runtime

In each new terminal, set these paths from the repository root. You can preview
a stage configuration before starting training:

```bash
export PYTHONPATH="$PWD/src"
PARSE_RUNTIME="$PWD/.nemotron/parse-runtime"
PARSE_PYTHON="$PARSE_RUNTIME/venv/bin/python"

"$PARSE_PYTHON" -m nemotron parse sft --dry-run \
  runtime_dir="$PARSE_RUNTIME"
```

Follow the [recipe commands](../README.md#run-the-dental-worked-example) for data
preparation, SFT, RLVR and evaluation. Pass `runtime_dir="$PARSE_RUNTIME"` to
each stage. The launcher starts stages in separate processes using this runtime.

To install elsewhere, choose a different `PARSE_RUNTIME` before setup and pass
that same path to every stage command. Bundled profiles otherwise use
`.nemotron/parse-runtime` relative to the current directory.

## Reuse or update an environment

Repeat the setup command with the same inputs and directory to verify an
existing installation. If you change runtime settings, framework sources or
installed packages, use a new runtime directory.

Keep `runtime.json` and `environment.txt` with your experiment results. They
record the framework and package versions actually installed; fresh installations
can resolve different compatible dependency versions.

For optional metric experiments, use the separate environment described in the
[document scorer guide](../document_rewards/README.md), preserving the training
runtime's recorded dependencies.
