# Set Up Parse Execution

Run Parse stages in separate uv environments locally or submit them with
NeMo-Run using prebuilt container images. Each stage declares its dependencies
in `pyproject.toml` and provides a `uv.lock`. Stage commands prepare the
required environment automatically.
Local runs, data containers, and RLVR containers apply the stage lockfiles.
SFT and evaluation containers retain the image's framework stack and resolve
additional packages from `pyproject.toml` without applying the full stage lockfile.

## Requirements

- Linux and one NVIDIA GPU per training or evaluation stage, with a driver
  compatible with the selected stage's CUDA dependencies.
- The Nemotron CLI installed from this source checkout, uv 0.12.12 or later,
  Git, and network access for model and dependency downloads.
- Docker with NVIDIA GPU support for Docker execution, or a configured
  single-task Slurm environment for cluster execution.

Use a GPU with at least 80 GB of memory for the full defaults. The batch-size-four
SFT configuration exceeded the 48 GB capacity of an NVIDIA RTX A6000 with both
the default and expandable CUDA allocator settings. The 80 GB recommendation
awaits a completed validation run of the current defaults; it is not a measured
minimum. The `tiny` profiles limit update counts but retain the default SFT
batch size. For lower-memory smoke checks, reduce batch sizes and input sizes
separately. Memory use depends on image dimensions, target lengths, and
training settings.

The controller and stage environments are separate. uv selects each local
stage's Python version; RLVR requires Python 3.13.14 or later in the 3.13 series.
No separate NeMo RL checkout or framework setup command is required: uv fetches
the archive of the pinned commit and builds its wheel automatically.

## Dental Rendering Prerequisites

The dental data stage and RLVR candidate generator require Pillow with Raqm
layout support. On Debian or Ubuntu, install `libfribidi0` and use a Pillow build
with Raqm support. The default RLVR container image provides FriBidi.

From the repository root, check both stage environments before generating data
or starting RLVR:

```bash
uv run --locked --project src/nemotron/recipes/parse/stage0_data \
  python -c 'from PIL import features; assert features.check_feature("raqm")'
uv run --locked --project src/nemotron/recipes/parse/stage2_rlvr \
  python -c 'from PIL import features; assert features.check_feature("raqm")'
```

These commands install the locked stage dependencies if needed. A successful
check exits without an assertion error. Restart Python after installing a
missing native library. The renderer fails if Raqm is unavailable because
Pillow's BASIC fallback changes the training images.

When comparing local and container runs, compare candidate image hashes as well
as manifests. Checkpoint reload checks do not establish rendering parity.

## Configure Local Execution

From the Nemotron repository root, install and activate the CLI environment:

```bash
uv sync --locked --python 3.12
source .venv/bin/activate
nemotron parse sft --dry-run
nemotron parse rlvr -c tiny --dry-run
```

A dry run prints the resolved configuration without installing stage
dependencies or starting training. Run the
[worked example](workflow.md#run-the-dental-worked-example) to prepare data and
train. Local stages use `uv run --locked` with their own project environments.
Parse ignores an ambient `UV_PROJECT_ENVIRONMENT` when selecting those stage
environments and preserves `PYTHONPATH` for external task modules.

An environment profile with `executor = "local"` also uses local uv execution,
including when selected with `--run` or `--batch`. Other profiles support one
node, one task, and one GPU per stage. The data stage also accepts zero GPUs.
The CLI rejects unsupported resource settings before submission.

## NVIDIA DGX Spark and Unified Memory

For local execution on NVIDIA DGX Spark, make the CUDA compiler discoverable
before starting a stage. FlashInfer uses it for just-in-time compilation.
The RLVR and evaluation stage environments supply `ninja`.

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
```

On unified-memory systems where NVML free-memory queries return
`NVMLError_NotSupported`, append `training.policy.refit_buffer_size_gb=1`
to the `nemotron parse rlvr` command. The same override applies with
`--run parse-docker`. It sets a 1 GiB weight-transfer buffer instead of querying
free GPU memory. This exceeds the default Parse model's largest FP32 tensor
(approximately 297 MB). The setting controls allocation and leaves training
batches, update counts, and sampling budgets unchanged. Recheck the buffer size
if you change the model.

## Configure Container Execution

Run remote stage commands from the Nemotron checkout root. The CLI rejects
other working directories before packaging to keep unrelated project files out
of the job archive.

The stage runspecs pin these prebuilt images by multiarchitecture index digest.
Docker selects the pinned `linux/amd64` or `linux/arm64` image for the host:

| Stage | Default image | Python dependencies |
| --- | --- | --- |
| Data | Astral uv with Python 3.12 | Complete locked stage environment |
| SFT | NVIDIA NeMo AutoModel 26.10.00 | Image training stack plus stage dependencies |
| RLVR | vLLM 0.33.0 | Complete locked stage environment |
| Evaluation | vLLM 0.33.0 | Image inference stack plus stage dependencies |

Data and RLVR install their complete environments separately from the images'
Python packages. SFT and evaluation retain their image's interpreter and selected
packages. The shared runner prepares these environments; no custom image build
is required. Allow space for image downloads, extraction, uv caches, and model
checkpoints. On `linux/amd64`, the three distinct image sizes total approximately
88 GB before layer sharing.

Create a persistent directory and add a profile to `env.toml`:

```bash
PARSE_RUN="$PWD/.nemotron/parse"
mkdir -p "$PARSE_RUN/home"
```

```toml
[parse-docker]
executor = "docker"
gpus_per_node = 1
ipc_mode = "host"
mounts = ["/absolute/path/parse-run:/absolute/path/parse-run"]

[parse-docker.env_vars]
HOME = "/absolute/path/parse-run/home"
UV_CACHE_DIR = "/absolute/path/parse-run/uv-cache"
NEMOTRON_VENV_ROOT = "/absolute/path/parse-run/venvs"
USER = "your-user-name"

[parse-docker.additional_kwargs]
user = "1000:1000"
group_add = ["65532"]
```

On hosts without a registered `nvidia` Docker runtime, set `runtime = "runc"`
under `[parse-docker]`. This supports DGX Spark installations that use the
default runtime with GPU device requests. NVIDIA Container Toolkit must still
be installed to provide GPU access.

Replace `/absolute/path/parse-run` with the absolute value of `PARSE_RUN`,
`1000:1000` with your host UID and GID (`id -u` and `id -g`), and
`your-user-name` with `id -un`. The AutoModel image requires group `65532` to read
its installed SDK files. Use a writable home directory for the selected user.
Omit `container` to use each stage's default image.

Add `--run parse-docker` to each command in the
[worked example](workflow.md#run-the-dental-worked-example). For example:

```bash
nemotron parse sft --run parse-docker -c tiny \
  data_dir="$PARSE_RUN/data" output_dir="$PARSE_RUN/sft"
```

Use attached Docker execution with `--run`; detached Docker `--batch` is
unavailable with the pinned NeMo-Run version. Parse checks the stage's saved
exit code and reports a failed container stage as a command failure.

## Keep Artifacts Available Between Jobs

Container and Slurm jobs require absolute artifact paths beneath profile
mounts. Keep the same absolute paths inside and outside the container because
artifacts record dataset, checkpoint, and processor paths. Output mounts must
be writable. The CLI rejects relative or unmounted artifact paths before
submission; packaged job directories do not provide persistent stage handoffs.
Docker mounts used for artifacts must include an explicit source and destination,
such as `/data:/data`. A destination-only mount creates an anonymous volume that
is not shared between stage containers.

Keep custom training YAML files on a mounted path and select them with an
absolute `training_config` path. Mounted configurations retain that path even
when they are stored in the checkout's Git-ignored run directory.

Use one node and one task for Slurm profiles. Submit with `--run profile` for
attached execution or `--batch profile` for detached execution, and wait for each
stage to finish before starting its consumer. Keep referenced dataset and model
paths available throughout training and comparison.

For stage dependency handling and framework versions, refer to the
[environment reference](environments.md).
For optional metric experiments, use the separate environment in the
[document scorer guide](rewards.md).
