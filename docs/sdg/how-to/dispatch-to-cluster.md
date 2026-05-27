<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

(sdg-dispatch-to-cluster)=
# Dispatch SDG to a Cluster

This guide covers configuring an env.toml profile and running `sdg/data_designer` on Lepton or Slurm. Generation is CPU-only (no GPUs needed) and calls a remote LLM endpoint, so the step fits naturally on a CPU node with outbound network access.

## env.toml Profile Shape

Add a profile to `env.toml` (repository root). The example below targets a Lepton CPU node:

```toml
[lepton_sdg_data_designer]
executor = "lepton"
container_image = "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
nemo_run_dir = "/mnt/shared/nemo-run"
nodes = 1
gpus_per_node = 0
resource_shape = "cpu.large"
node_group = "your-node-group"
shared_memory_size = 1024
can_be_preempted = true
queue_priority = "mid-4000"
startup_commands = [
    "python -m pip install --quiet --break-system-packages 'data-designer==0.5.5'"
]
mounts = [
    { path = "/your-nfs-source", mount_path = "/mnt/shared", from = "node-nfs:your-nfs-id" }
]

[lepton_sdg_data_designer.env_vars]
NVIDIA_API_KEY = "${oc.env:NVIDIA_API_KEY}"
```

## Run

```console
$ uv run --no-sync nemotron steps run sdg/data_designer -c default --batch lepton_sdg_data_designer num_records=1000
```

Use `--run` instead of `--batch` to stream logs interactively.

## Known Gotchas

These are the failure modes that commonly affect first-time cluster SDG runs.

### `data-designer` is not pre-installed in the container

The NeMo container image does not include `data-designer`. Install it at startup via `startup_commands`:

```toml
startup_commands = [
    "python -m pip install --quiet --break-system-packages 'data-designer==0.5.5'"
]
```

Do not omit `--break-system-packages` — without it pip refuses to install into the system Python on recent NeMo images.

### Default `shared_memory_size` crashes a CPU node

The runspec default for `shared_memory_size` is 65536 MB (64 GB), which exceeds the RAM of most CPU node types and causes the job to be rejected or OOM-killed immediately. Set it to a small value; this step makes no use of shared memory:

```toml
shared_memory_size = 1024
```

### `nemo_run_dir` must be on shared storage

`nemo-run` uses a busybox data-mover sidecar to stage the launch script into `nemo_run_dir`. If this path is not visible to both the data-mover and the main container — specifically if it is local to one node — the main container never finds the script and the job fails with `No such file or directory`.

Set `nemo_run_dir` to a path on the shared NFS mount and include the mount in your profile:

```toml
nemo_run_dir = "/mnt/shared/nemo-run"
mounts = [
    { path = "/your-nfs-source", mount_path = "/mnt/shared", from = "node-nfs:your-nfs-id" }
]
```

:::{note}
In the `mounts` table, `path` is the NFS **source** path on the NFS server — not the in-container destination. `mount_path` is the in-container path.
:::

### `NVIDIA_API_KEY` is not forwarded automatically

Unlike `HF_TOKEN` and `WANDB_API_KEY`, `NVIDIA_API_KEY` is not automatically forwarded to the container. Declare it explicitly in the `env_vars` section:

```toml
[lepton_sdg_data_designer.env_vars]
NVIDIA_API_KEY = "${oc.env:NVIDIA_API_KEY}"
```

Set it in your local shell before submitting the job:

```console
$ export NVIDIA_API_KEY="your-api-key"
$ uv run --no-sync nemotron steps run sdg/data_designer -c default --batch lepton_sdg_data_designer num_records=1000
```

### Container image: always look up, never guess

Do not invent image tags. `nemo:latest` does not exist on `nvcr.io`. Check `src/nemotron/steps/sdg/data_designer/step.py` header comments or `src/nemotron/steps/env/env_toml/config/lepton.yaml` for known-good image references before setting `container_image`.

### Preemption and queue-priority fields were not wired (now fixed)

`can_be_preempted`, `can_preempt`, and `queue_priority` are now forwarded from env.toml to `LeptonExecutor`. If you are on an older version of the repo where these were silently ignored, upgrade before expecting preemption scheduling to take effect.

## Slurm Profile

For Slurm, replace the Lepton-specific fields with Slurm equivalents. The `startup_commands` and `env_vars` gotchas apply equally:

```toml
[slurm-sdg]
executor = "slurm"
container_image = "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
nemo_run_dir = "/lustre/team/nemo-run"
nodes = 1
gpus_per_node = 0
run_partition = "cpu"
batch_partition = "cpu"
startup_commands = [
    "python -m pip install --quiet --break-system-packages 'data-designer==0.5.5'"
]

[slurm-sdg.env_vars]
NVIDIA_API_KEY = "${oc.env:NVIDIA_API_KEY}"
```

:::{tip}
On clusters where the default partition requires GPUs (for example, NVIDIA's `dlw` cluster), set `run_partition` and `batch_partition` to a CPU-capable partition. `gpus_per_node = 0` alone is not sufficient — the partition itself must accept zero-GPU jobs.
:::

## Verify Before Scaling

Run a preview via the cluster profile before a large batch:

```console
$ uv run --no-sync nemotron steps run sdg/data_designer -c default --run lepton_sdg_data_designer preview=true num_records=2
```

Confirm the job reaches `Running`, the model alias check succeeds, and two records are returned before submitting the full job.

## Next Steps

- **env.toml reference**: `docs/nemo_runspec/nemo-run.md` — full profile field reference.
- **CLI flags**: {doc}`../reference/cli-reference`.
- **Troubleshooting**: {doc}`../reference/troubleshooting` — full failure-mode reference.
