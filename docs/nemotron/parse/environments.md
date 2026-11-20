# Parse Environment Reference

Use this reference to inspect stage dependencies, understand container package
selection, and customize environments for your task. For installation commands,
GPU requirements, and container profiles, follow [Set Up Parse Execution](runtime.md).

## Execution Modes

Each stage declares dependencies in its own `pyproject.toml` and includes a
`uv.lock`. Local execution starts the stage through `uv run --locked`.
Docker and Slurm execution use NeMo-Run to package the checkout and invoke the
shared `nemotron.kit.run_uv` runner inside the selected image.

Container dependency handling differs by stage, as shown in the following table.

| Stage | Default image | Container dependency handling |
| --- | --- | --- |
| Data preparation | Astral uv with Python 3.12 | Installs the complete stage environment from its lockfile. |
| Supervised fine-tuning (SFT) | NVIDIA NeMo AutoModel 26.10.00 | Retains the image's training packages and installs additional stage dependencies. |
| Reinforcement learning with verifiable rewards (RLVR) | vLLM 0.33.0 | Installs the complete stage environment from its lockfile. |
| Evaluation | vLLM 0.33.0 | Retains the image's inference packages and installs additional stage dependencies. |

The stage metadata pins images by multiarchitecture index digest. Docker selects
the `linux/amd64` or `linux/arm64` image for the host. These image variants do not
establish equivalent results across hardware architectures.

Data preparation and RLVR use `container-mode = "isolated"`. Their Python
packages come from the stage lockfile rather than the image's environment.
The RLVR image also provides native libraries required for dental rendering,
including FriBidi.

SFT and evaluation use the default `inherit` mode. The runner retains the image's
Python interpreter and exposes its installed packages to the stage environment.
It excludes packages listed in `container-exclude-dependencies` from the stage's
dependency resolution. These stages resolve additional dependencies without
applying the complete stage lockfile. The evaluation project also declares
`container-requires` constraints for the image's PyTorch and vLLM versions.

Local execution isolates stages from an ambient `UV_PROJECT_ENVIRONMENT` while
preserving `PYTHONPATH` for external task modules. All modes currently require
one controller per stage and support single-GPU training and evaluation.

## Framework Dependencies

The SFT project selects NeMo AutoModel 0.7.0. The RLVR project selects fixed
commit archives for NeMo RL, NeMo AutoModel, and NeMo Lens. uv downloads these
archives and builds the required Python packages without a separate framework
checkout.

NeMo RL is pinned to `5a21699af75913a0ff827cb5481bebacefad19c1`, which includes
the packaging changes proposed in [NeMo RL PR #5246](https://github.com/NVIDIA-NeMo/RL/pull/5246).
The stage can install the training modules from a wheel without reading the
framework's example directory. The recipe bundles the upstream configuration
files and keeps their `defaults` inheritance chain. Refer to the
[training configuration guide](workflow.md#sft-rlvr-and-artifact-flow) for
configuration inheritance and task overrides.

The RLVR project uses explicit dependency overrides for its backend stack.
The configured versions are listed in the following table.

| Package | RLVR override |
| --- | --- |
| PyTorch | 2.14.0 |
| torchvision | 0.29.0 |
| Transformers | 5.16.1 |
| quack-kernels | 0.6.5 |
| cuDNN | 9.24.0.43 |
| NVSHMEM | 3.4.5 |

TransferQueue uses a source override that replaces its Git URL with an archive
of the same commit. Triton uses the explicit PyTorch CUDA 13.0 package index.
These sources are declared in the RLVR stage's `pyproject.toml` and recorded in
its `uv.lock`.

The recipe's `NativeParseGenerationWorker` adapts the pinned NeMo RL package to
native vLLM 0.33.0. It replaces the worker's source-patching initialization path
while preserving environment-variable forwarding. It checks dependency versions,
the model architecture, and the upstream initializer before starting. The
adapter supports synchronous, single-GPU generation. Changes to these assumptions
require validation of training, weight transfer, and checkpoint reload.

Ray workers use the same stage environment through
`NEMO_RL_PY_EXECUTABLES_SYSTEM=1`. The recipe does not modify installed framework
source files or require a custom container build.

## Customize an Environment

Add task dependencies to the relevant stage projects and regenerate their
lockfiles. A local run or an isolated container uses the updated lockfile.
For an inherited container environment, also check its dependency exclusions and
image package versions.

Put external task code on `PYTHONPATH`. For Docker or Slurm, mount the code and
set its path inside the container in the profile's `env_vars`. Both the task and
reward environment must be importable by Ray workers. Refer to
[Adapt Parse to Your Domain](adaptation.md) for the task contract.

Keep uv caches on a filesystem owned by the execution user. If shared storage
maps ownership to another user ID, use a local cache. Refer to the
[container profile instructions](runtime.md#configure-container-execution) for
cache locations, user IDs, and persistent mounts.

Use fresh output directories when repeating a stage. Failed outputs remain
available for inspection, and stage-level resume is not implemented.
