# Customizing Nemotron 3.5 Lightning on Two DGX Stations with GRPO

This guide runs full-weight GRPO post-training for
[NVIDIA Nemotron 3.5 Lightning 30B-A3B BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
on two DGX Station GB300 systems. It leverages the B300 GPU from each Station, a
two-node Ray cluster, NeMo RL, NeMo Gym, and the
[`python_inductive`](https://huggingface.co/datasets/nvidia/Nemotron-RL-ARC-AGI-v1)
split of the Nemotron RL ARC-AGI dataset to customize Nemotron 3.5 Lightning.

For other customization workflows on this topology, see the
[DGX Station recipe index](README.md). For supervised fine-tuning with NeMo
AutoModel, go directly to the [two-Station SFT recipe](sft.md).

> [!NOTE]
> The instructions use one Station as the **head** and the other as the
> **worker**. Keep those roles unchanged for the entire run. Commands marked
> **both Stations** must be run separately on each system; commands marked
> **head only** or **worker only** run on that system only.

> [!IMPORTANT]
> This recipe pins NeMo RL source commit
> [`7fa6e55`](https://github.com/NVIDIA-NeMo/RL/commit/7fa6e55192530ff1346d670ce74f9c70cab8f75b)
> from August 18, 2026. That commit adds fractional optimizer CPU offload. The
> recipe also applies a separate Nemotron-H MoE refit fix and a local vLLM
> memory patch. Do not substitute a newer `main` revision without revalidating
> the configuration and both patches.

## What this recipe does

| Component | Configuration |
| --- | --- |
| Hardware | Two DGX Station GB300 systems [connected via InfiniBand](https://build.nvidia.com/station/connect-two-stations/overview) |
| Training | Full-weight synchronous GRPO with Megatron Core |
| Parallelism | Expert parallel size 2 across the two GPUs |
| Generation | Colocated vLLM, one tensor-parallel rank per GPU |
| Memory strategy | Activation checkpointing and 75% optimizer-state CPU offload |
| Sequence length | 16,536 tokens
| Environment | NeMo Gym's `nvarc` Python-inductive ARC-AGI verifier |
| Dataset | `python_inductive` split of [Nemotron-RL-ARC-AGI-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-ARC-AGI-v1) |

This is provided as a recipe to showcase customization of Nemotron models using DGX Stations, not a reproduction of the model's complete post-training program or published evaluation results.

## Prerequisites

Before starting, make sure you have:

- Two DGX Station GB300 systems with current DGX software, NVIDIA drivers,
  Docker, and NVIDIA Container Toolkit. The
  [DGX Station software stack](https://docs.nvidia.com/dgx/dgx-station-development-guide/porting/software-requirements.html)
  includes these components.
- The two Stations connected and validated for distributed GPU workloads.
  Complete NVIDIA's
  [Connect Two DGX Stations for Distributed Workloads](https://build.nvidia.com/station/connect-two-stations/overview)
  playbook, including its RDMA/GPU-memory tests, before continuing.
- A Linux user with `sudo` access on both Stations.
- Internet access to `nvcr.io`, Hugging Face, GitHub, and, if enabled, Weights
  & Biases.
- Shared storage visible from both Stations at the same host path. The optional
  NFS section below creates a simple demonstration share if one is not
  available.

Weights & Biases is enabled in the final configuration. You can either log in
before the run or disable it with a command-line override shown later.

## 1. Prepare both Stations

### Enable non-root Docker access

First, add non-root Docker access for your user if not already enabled.

Run on **both Stations**:

```bash
sudo usermod -aG docker "${USER}"
newgrp docker
docker version
```

Membership in the `docker` group grants root-equivalent access to the host.

### Verify the NVIDIA Drivers

Verify the NVIDIA driver was installed correctly and your GPU is available:

```bash
nvidia-smi -L
```

The command should show a single B300 GPU in the list. If a display-only graphics card is included on the Station, that may also appear in the list.

### Record the network values

Run on **both Stations** to list the IPv4 addresses and Linux network-device
names:

```bash
hostname
ip -4 -br address show
ip -4 route
```

Choose the high-speed interface used for traffic between the Stations. If you
followed the two-Station networking playbook, map the `mlx5_0` HCA to its Linux
network device with:

```bash
ibdev2netdev
```

`COMM_IFACE` must be the Linux network-device name (for example,
`enP3s3f1np1`), not the HCA name `mlx5_0`. The final NeMo RL configuration is
shared by both nodes, so this recipe assumes that the chosen Linux interface
has the same name on both Stations.

Record these values and export the same values in the host shell on **both
Stations**:

```bash
export HEAD_IP=<HEAD_STATION_IPV4>
export WORKER_IP=<WORKER_STATION_IPV4>
export COMM_IFACE=<LINUX_NETWORK_INTERFACE>
```

Verify the path in both directions:

```bash
# Run on the head.
ping -c 4 "${WORKER_IP}"

# Run on the worker.
ping -c 4 "${HEAD_IP}"
```

Do not continue until the pings and the prerequisite fabric tests succeed.
The Stations must also permit trusted node-to-node traffic for Ray, NeMo Gym,
NCCL, Gloo, and the configured port ranges.

## 2. Configure shared storage

If both Stations already mount shared storage at the same host path, export that path on **both Stations** and skip the remainder of this section:

```bash
export SHARED_HOST_PATH=<SHARED_STORAGE_HOST_PATH>
```

Verify that a file written from either Station appears on the other before
launching the containers.

### Optional: create an NFS export on the head

If you do not already have a shared storage system on both Stations, the steps
below will walk through installing and configuring an NFS server on the head
node that will be shared between systems.

The following setup is intentionally simple and is suitable for a trusted,
point-to-point demonstration network. It is not a hardened or
performance-tuned production storage design.

Install and start the NFS server on the **head only**:

```bash
sudo apt-get update
sudo apt-get install -y nfs-kernel-server
sudo systemctl enable --now nfs-kernel-server
```

Create the export directory:

```bash
sudo mkdir -p /mnt/nfs_share
sudo chown -R "${USER}:$(id -gn)" /mnt/nfs_share
sudo chmod 0777 /mnt/nfs_share
```

[!NOTE]
> Mode `0777` allows root processes in a root-squashed containerized NFS client
> to write this demonstration share. Restrict the client network as narrowly as
> possible, and replace these permissions with your site's identity and storage
> policy for anything beyond an isolated lab setup.

Determine the network CIDR that contains both selected Station IPs. For a
direct link this is commonly a `/30`, such as `192.168.240.0/30`. Export it on
the **head only**:

```bash
export CLIENT_NETWORK_CIDR=<NETWORK_CIDR_FOR_BOTH_STATIONS>

printf '/mnt/nfs_share %s(rw,sync,no_subtree_check)\n' \
  "${CLIENT_NETWORK_CIDR}" \
  | sudo tee /etc/exports.d/nemotron.exports

sudo exportfs -ra
sudo exportfs -v
```

Confirm that `/mnt/nfs_share` is listed before continuing.

### Optional: mount the NFS export on both Stations

Mount the newly-created NFS server on **both Stations**, including the head,
so the container mount path is identical:

```bash
sudo apt-get update
sudo apt-get install -y nfs-common
sudo mkdir -p /mnt/nfs
sudo mount "${HEAD_IP}:/mnt/nfs_share" /mnt/nfs

mountpoint /mnt/nfs
df -hT /mnt/nfs
export SHARED_HOST_PATH=/mnt/nfs
```

For a persistent mount, add an `_netdev` NFS entry to `/etc/fstab` according
to your site's boot and network policy.

## 3. Launch the NeMo RL container

If your NGC organization requires authentication, log in on **both Stations**
before pulling the image:

```bash
docker login nvcr.io
```

Use `$oauthtoken` as the username and an NGC API key as the password.

Export the image and confirm all required values on **both Stations**:

```bash
export NEMO_RL_IMAGE=nvcr.io/nvidia/nemo-rl:v0.7.0

printf 'HEAD_IP=%s\nWORKER_IP=%s\nCOMM_IFACE=%s\nSHARED_HOST_PATH=%s\n' \
  "${HEAD_IP}" "${WORKER_IP}" "${COMM_IFACE}" "${SHARED_HOST_PATH}"

test -d "${SHARED_HOST_PATH}"
docker pull "${NEMO_RL_IMAGE}"
```

Start one container on **each Station**:

```bash
docker run -it \
  --name nemo-rl \
  --gpus all \
  --network host \
  --device=/dev/infiniband \
  --shm-size=128g \
  --ulimit nofile=65535:65535 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HEAD_IP="${HEAD_IP}" \
  -e WORKER_IP="${WORKER_IP}" \
  -e COMM_IFACE="${COMM_IFACE}" \
  -e HF_HOME=/shared/.cache/huggingface \
  -e NCCL_IB_HCA=mlx5_0,mlx5_1 \
  -v "${SHARED_HOST_PATH}:/shared" \
  "${NEMO_RL_IMAGE}"
```

`--device=/dev/infiniband` exposes the two CX8 RDMA devices to the container,
and `NCCL_IB_HCA` selects both validated rails. If your site uses different HCA
names, replace `mlx5_0,mlx5_1` with the names reported by `ibdev2netdev`.

The rest of the guide runs inside these two container shells unless a step
says otherwise. Keep both shells running for the duration of training. The
containers are intentionally retained if a shell exits; re-enter one with:

```bash
docker start -ai nemo-rl
```

Inside both containers, verify the environment:

```bash
nvidia-smi -L
df -hT /shared
ulimit -Sn
test -d /dev/infiniband
ls -1 /dev/infiniband
```

A single B300 GPU should be visible in each container. The soft open-file limit
should be `65535`, and `/shared` must resolve to the same storage from both
containers. If `/dev/infiniband` is missing, NCCL will fall back to a slower
socket transport or fail.

## 4. Pin and patch NeMo RL

In order to fit full-weight GRPO on two Stations, the optimizer's weights need
to be partially offloaded to system memory to prevent CUDA OOM errors. This
isn't available in the v0.7.0 NeMo-RL container but it can be patched from the
upstream repository.

The image contains the NeMo RL working tree at `/opt/nemo-rl`. The following
changes are made inside each retained container, so repeat this entire section
on **both Stations**.

Update `uv` and check out the pinned source revision:

```bash
cd /opt/nemo-rl
uv self update

export NEMO_RL_BASE_COMMIT=7fa6e55192530ff1346d670ce74f9c70cab8f75b

git fetch --no-recurse-submodules origin "${NEMO_RL_BASE_COMMIT}"
git checkout -B dgx-station-lightning "${NEMO_RL_BASE_COMMIT}"
git submodule sync --recursive
git submodule update --init --recursive
git --no-pager log -1 --oneline
```

The last command must start with `7fa6e55` and show
`feat(megatron): support fractional optimizer CPU offload (#3628)`.
Refreshing the submodules is required because this source revision pins newer
NeMo Gym and Megatron components than the v0.7.0 container image.

Configure an identity for the local cherry-pick, then apply the Nemotron-H MoE
refit correction:

```bash
git config user.name "<YOUR_NAME>"
git config user.email "<YOUR_EMAIL>"

git fetch --no-recurse-submodules \
  origin ruit/fix-nemotronh-moe-refit-shard-dim
git cherry-pick 6bb55bc03adbdc4943fba5c9e452586c04afee88

git --no-pager log -2 --oneline
```

The newest commit should be `fix(vllm): correct MoE refit shard_dim for
grouped/3D experts (NemotronH)`.

Patch async vLLM sleep to release the generation weights before policy
training. The pinned source contains exactly one matching line:

```bash
export VLLM_ASYNC_WORKER=nemo_rl/models/generation/vllm/vllm_worker_async.py

test "$(grep -c 'await self.llm.sleep(level=1)' "${VLLM_ASYNC_WORKER}")" -eq 1
sed -i \
  's/await self\.llm\.sleep(level=1)/await self.llm.sleep(level=2)/' \
  "${VLLM_ASYNC_WORKER}"

grep -n 'await self.llm.sleep' "${VLLM_ASYNC_WORKER}"
git diff --check
git status --short
```

The `grep` output must show `await self.llm.sleep(level=2)`. `git status`
should show only `vllm_worker_async.py` as modified. If the pre-patch `test`
fails, stop: the source revision is not the revision this recipe expects.

Finally, ensure the active shell has the required limit:

```bash
ulimit -Sn 65535
```

## 5. Start the two-node Ray cluster

NeMo-RL leverages Ray for serving the policy and inference engines and
coordinating communication between the hosts. A Ray cluster needs to be started
inside the containers for NeMo-RL to run in.

### Start Ray on the head node

Run inside the container on the **head only**:

```bash
cd /opt/nemo-rl
uv run ray start \
  --head \
  --node-ip-address="${HEAD_IP}" \
  --port=6379 \
  --num-gpus=1

ray status
```

At this point Ray should report one node and `1.0 GPU` total.

### Join the Ray cluster on the worker

Run inside the container on the **worker only**:

```bash
cd /opt/nemo-rl
uv run ray stop --force
uv run ray start \
  --address="${HEAD_IP}:6379" \
  --node-ip-address="${WORKER_IP}" \
  --num-gpus=1

ray status
```

The worker command must report a successful connection. Keep this container
shell open.

Back inside the container on the **head**, verify both nodes and both GPUs:

```bash
cd /opt/nemo-rl
ray status
```

The output should indicate that two GPUs are available in the cluster:

```bash
...
Resources
---------------------------------------------------------------
Total Usage:
 0.0/144.0 CPU
 0.0/2.0 GPU
...
```

## 6. Authenticate and populate the shared cache

To make training startup more efficient on both systems, pre-cache the
Nemotron 3.5 Lightning model on the shared storage so it is available for both
systems to pull from, avoiding a cold-pull to Hugging Face at the beginning of
training.

Run on the **head only**, inside the container:

```bash
mkdir -p /shared/.cache/huggingface
export HF_HOME=/shared/.cache/huggingface
hf auth login

export MODEL_ID=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
hf download "${MODEL_ID}"
```

The Hugging Face token and model cache are now available at the same path from
both containers.

If you want Weights and Biases (W&B) logging, authenticate on the **head only**:

```bash
wandb login
```

When prompted, enter your W&B API key to authenticate with their servers for
the training process.

## 7. Prepare the ARC-AGI dataset

We will prepare the ARC-AGI dataset which includes several RL-focused data
samples specializing in advanced mathematical operations.

To prepare the dataset, run on the **head only** inside the container:

```bash
mkdir -p /shared/data
cd /shared/data
```

Create `/shared/data/prep.py` with the following contents:

```python
from datasets import load_dataset


dataset = load_dataset(
    "nvidia/Nemotron-RL-ARC-AGI-v1",
    "python_inductive",
)

dataset["train"].to_json("train.jsonl")
dataset["validation"].to_json("validation.jsonl")
dataset["train"].select(range(min(4, len(dataset["train"])))).to_json(
    "smoke_train.jsonl"
)
```

Run the converter and verify all three outputs:

```bash
python3 prep.py
wc -l train.jsonl validation.jsonl smoke_train.jsonl

python3 - <<'PY'
import json
from pathlib import Path

for name in ("train.jsonl", "validation.jsonl", "smoke_train.jsonl"):
    path = Path(name)
    assert path.stat().st_size > 0, f"{name} is empty"
    with path.open() as stream:
        row = json.loads(stream.readline())
    assert "responses_create_params" in row
    assert row["agent_ref"]["name"] == "nvarc_inductive_simple_agent"

print("ARC-AGI NeMo Gym JSONL files are ready.")
PY
```

## 8. Create the two-Station GRPO configuration

Copy the config file at [grpo_lightning35_station.yaml](grpo_lightning35_station.yaml) in this repository and
save it on the **head node**  inside the container at
`/opt/nemo-rl/examples/nemo_gym/grpo_lightning35_station.yaml`.

## 9. Launch the full distributed run

Launch the full training run on the **head node**:

```bash
cd /opt/nemo-rl

NRL_FORCE_REBUILD_VENVS=true \
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/nemo_gym/grpo_lightning35_station.yaml \
  ++policy.megatron_cfg.env_vars.GLOO_SOCKET_IFNAME="${COMM_IFACE}" \
  ++policy.megatron_cfg.env_vars.NCCL_SOCKET_IFNAME="${COMM_IFACE}"
```

To run without W&B, add this override before the two interface overrides:

```text
logger.wandb_enabled=false
```

The driver remains attached to the head terminal. Leave the worker container
and Ray process running until training ends.

## 10. Monitor artifacts and cluster health

From another shell, enter the retained container on either Station and inspect
Ray and GPU health:

```bash
docker exec -it nemo-rl bash
cd /opt/nemo-rl
uv run ray status
nvidia-smi
```

Depending on which stage of the training process is currently active, you
should see tens to a couple hundred of GiBs of GPU memory allocated and GPU
activity near 100% utilization, indicating training is running.

Inspect the shared storage for logs and checkpoints. These directories will
become populated as training completes a few steps:

```bash
find /shared/logs -maxdepth 2 -type f | sort | tail -n 20
find /shared/results/grpo -maxdepth 2 -type f | sort | tail -n 20
df -h /shared
```

Monitor free storage throughout the run, especially if retaining all
checkpoints.

If you logged results to W&B, login to your account in a web browser and view
the link to your latest run. In general for GRPO workloads, you will want to
see your reward signal increase over time, indicating your policy is learning
how to achieve more desirable responses for your specific environment.

## 11. Stopping the run and cleaning up

If you need to stop the run, connect to the terminal process where training
is running and hit `Ctrl+C` and allow the process to shut down its actors.

To stop the Ray cluster, run the following in **both containers**:

```bash
cd /opt/nemo-rl
uv run ray stop --force
exit
```

The containers remain available for inspection or restart. Remove them only
when their local patches and logs are no longer needed:

```bash
docker rm nemo-rl
```

Training data, model cache, logs, and checkpoints under `/shared` are not
removed with the container.

## Troubleshooting

### Ray reports only one GPU

- Run `nvidia-smi -L` inside both containers.
- Confirm that the worker used `--address="${HEAD_IP}:6379"` and printed a
  successful connection.
- Confirm that `HEAD_IP` and `WORKER_IP` belong to the selected
  `COMM_IFACE` network.
- Stop stale Ray processes on both nodes, restart the head, then rejoin the
  worker.

### NCCL or Gloo cannot connect

- Confirm that `COMM_IFACE` is the Linux device name present on both Stations,
  not `mlx5_0`.
- Confirm that `/dev/infiniband` exists inside both containers and that
  `NCCL_IB_HCA` names the HCAs reported by `ibdev2netdev`.
- Re-run the two-direction ping and the NVIDIA two-Station fabric validation.
- Check host firewall rules against the port table in the Ray section, and
  confirm that no other process is using those ports.
- For the next diagnostic run, export `NCCL_DEBUG=INFO` in both container
  shells before starting Ray. A healthy RDMA run reports `NET/IB` and the
  selected CX8 HCAs; `NET/Socket` indicates a TCP fallback.

### CUDA runs out of memory during policy training

- Confirm the config contains `optimizer_cpu_offload: true` and
  `optimizer_offload_fraction: 0.75`.
- Confirm `grep -n 'self.llm.sleep'` shows `level=2` in
  `vllm_worker_async.py` on both Stations.
- Confirm Ray advertises exactly one GPU per Station and stop other GPU
  workloads.
- Do not increase sequence length, rollout count, or vLLM memory utilization
  until the baseline run succeeds.

### vLLM fails during Nemotron-H MoE weight refit

Run `git log -1 --format=%s` on both Stations and verify that the subject is
`fix(vllm): correct MoE refit shard_dim for grouped/3D experts (NemotronH)`.
That local commit was cherry-picked from `6bb55bc`. Recreate the pinned patch
stack if either working tree differs.

### `/shared` is read-only or differs between nodes

- Compare `df -hT /shared` and a test file from both containers.
- For the demonstration NFS setup, confirm `exportfs -v` on the head and
  `mountpoint /mnt/nfs` on both hosts.
- Resolve UID/GID, root-squash, and permissions through your storage
  administrator rather than disabling security controls on a shared network.

### A source patch no longer applies

Do not work around it by blindly changing line numbers. Verify that the base
commit is exactly `7fa6e55`, recreate the retained container if needed, and
repeat the patch section. A different NeMo RL revision requires recipe
revalidation.

## References

- [NeMo RL repository](https://github.com/NVIDIA-NeMo/RL)
- [NeMo RL v0.7.0 documentation](https://docs.nvidia.com/nemo/rl/0.7.0/index.html)
- [Nemotron 3.5 Lightning model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
- [Nemotron RL ARC-AGI dataset card](https://huggingface.co/datasets/nvidia/Nemotron-RL-ARC-AGI-v1)
- [DGX Station software stack](https://docs.nvidia.com/dgx/dgx-station-development-guide/porting/software-requirements.html)
- [Connect Two DGX Stations for Distributed Workloads](https://build.nvidia.com/station/connect-two-stations/overview)
