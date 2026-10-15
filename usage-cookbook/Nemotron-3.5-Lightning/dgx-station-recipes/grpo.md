# Customizing Nemotron 3.5 Lightning on Two DGX Stations with GRPO

This guide runs full-weight GRPO post-training for
[NVIDIA Nemotron 3.5 Lightning 30B-A3B BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
on two DGX Station GB300 systems. It leverages the B300 GPU from each Station, a
two-node Ray cluster, NeMo RL, NeMo Gym, and the
[`python_inductive`](https://huggingface.co/datasets/nvidia/Nemotron-RL-ARC-AGI-v1)
split of the Nemotron RL ARC-AGI dataset to customize Nemotron 3.5 Lightning.

For other customization workflows, see the
[DGX Station recipe index](README.md). For supervised tuning with NeMo
AutoModel, use the [single-Station LoRA recipe](lora.md) or the
[two-Station full-weight SFT recipe](sft.md).

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
| Sequence length | 16,384 tokens |
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
- Shared storage visible from both containers at `/shared`. The optional
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
followed the two-Station networking playbook, map each HCA to its Linux
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
export NCCL_IB_HCA=<VALIDATED_HCA_NAMES>
```

Use fabric IPs on `COMM_IFACE`, not SSH management IPs on a slower NIC.
Check link speed with `sudo ethtool "${COMM_IFACE}"` and verify that the
selected HCAs are active. Do not include disconnected rails. Configure IPs
through your network manager; transient `ip address add` settings can be
removed by NetworkManager DHCP retries.

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

### Inspect NUMA memory placement before CPU offload

On **both hosts**, record the topology with `numactl --hardware` (install
`numactl` if needed). Identify the CPU-attached memory node; do not assume
all GB300 systems expose the same node layout. CPU-offloaded optimizer and
model buffers must use CPU-attached memory, not GPU-backed memory exposed
through the host's NUMA topology. Select a node with CPU-attached memory and
verify process residency during offload and subsequent vLLM weight wake-up.

```bash
numactl --hardware
export CPU_MEM_NODE=<CPU_MEMORY_NUMA_NODE>
```

Set `CPU_MEM_NODE` independently on each host. The container below adds
`SYS_NICE` to permit memory-policy operations; this is a capability increase,
so use it only on trusted training containers. Bind Ray daemons and the
training driver to the selected CPU memory node, not just one of them.

## 2. Configure shared storage

If shared storage is already available, export its host path separately on
**each Station** and skip the optional NFS setup. Host paths may differ, but
both containers must see the same files at `/shared`:

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

> [!NOTE]
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

### Optional: use the local export on the head and NFS on the worker

On the **head**, bind the export directory directly into the container:

```bash
export SHARED_HOST_PATH=/mnt/nfs_share
```

On the **worker only**, mount the export:

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

### Create an isolated run and establish cross-node permissions

Choose a new identifier on **both host shells**. Do not reuse it for a new
experiment: NeMo RL automatically resumes the latest complete checkpoint.

```bash
export RUN_ID=<UNIQUE_RUN_ID>
export RUN_LOG_DIR=/shared/logs/${RUN_ID}
export RUN_CHECKPOINT_DIR=/shared/results/grpo/${RUN_ID}
export NRL_MEGATRON_CHECKPOINT_DIR=/shared/models/megatron/${RUN_ID}
```

Container root on the head's local bind mount is UID 0, but root on an NFS
client is normally mapped to the export's anonymous identity. A writable
share root does **not** make newly created root-owned `0755` subdirectories
writable by the worker. Verify inherited permissions before training.

For the optional trusted-lab NFS export, one approach is to grant the
anonymous identity access through **default ACLs on new run directories**.
Run on the **head host**, using the UID configured by your NFS export
(default Linux `anonuid` is 65534):

```bash
sudo apt-get install -y acl
export NFS_ANON_UID=65534
export RUN_OWNER_UID=$(id -u)
for relative in "logs/${RUN_ID}" "results/grpo/${RUN_ID}" "models/megatron/${RUN_ID}"; do
  target="${SHARED_HOST_PATH}/${relative}"
  test ! -e "${target}" || { printf 'Choose a fresh RUN_ID: %s\n' "${target}"; exit 1; }
  sudo mkdir -p "${target}"
  sudo chown "$(id -u):$(id -g)" "${target}"
  sudo chmod 0700 "${target}"
  sudo setfacl -m "u:${NFS_ANON_UID}:rwx,m::rwx" "${target}"
  sudo setfacl -d -m "u::rwx,u:${RUN_OWNER_UID}:rwx,u:${NFS_ANON_UID}:rwx,g::---,m::rwx,o::---" "${target}"
done
```

This grants all root-squashed clients allowed by the export access to these
run directories. Restrict the export to trusted clients. For production or
existing storage, use your administrator's UID/GID and ACL policy instead;
do not disable root-squash or recursively change existing experiments.
Default ACLs can be limited by applications' explicit creation modes, so
successful host writes are not enough: perform the nested container test
below and validate checkpoint saving during the smoke run.

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
  --cap-add=SYS_NICE \
  --device=/dev/infiniband \
  --shm-size=128g \
  --ulimit nofile=65535:65535 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HEAD_IP="${HEAD_IP}" \
  -e WORKER_IP="${WORKER_IP}" \
  -e COMM_IFACE="${COMM_IFACE}" \
  -e CPU_MEM_NODE="${CPU_MEM_NODE}" \
  -e RUN_ID="${RUN_ID}" \
  -e RUN_LOG_DIR="${RUN_LOG_DIR}" \
  -e RUN_CHECKPOINT_DIR="${RUN_CHECKPOINT_DIR}" \
  -e NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR}" \
  -e HF_HOME=/shared/.cache/huggingface \
  -e NCCL_IB_HCA="${NCCL_IB_HCA}" \
  -e NCCL_IB_DATA_DIRECT=0 \
  -v "${SHARED_HOST_PATH}:/shared" \
  "${NEMO_RL_IMAGE}"
```

`--device=/dev/infiniband` exposes the RDMA devices. `NCCL_IB_HCA` selects
only the active rails that passed fabric validation.

`NCCL_IB_DATA_DIRECT=0` disables the Data Direct registration path, not RDMA.
It avoids the `mlx5dv_reg_dmabuf_mr` error 524 associated with an unsupported
Data Direct mapping topology. Verify GPU all-reduce and object gathering
with this setting, and confirm that GPU-direct RDMA remains active. Do not
assume the same transport behavior on every driver/NCCL combination.

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

### Verify NUMA binding inside both containers

```bash
apt-get update -qq && apt-get install -y numactl
numactl --hardware
numactl --membind="${CPU_MEM_NODE}" true
```

Stop if binding fails. Verify the node exists and has CPU-attached memory;
if the error is `Operation not permitted`, check `SYS_NICE` and your site's
seccomp/capability policy. Do not disable all container security controls.

### Validate nested writes from both containers

Confirm `RUN_ID`, `RUN_LOG_DIR`, `RUN_CHECKPOINT_DIR`, and
`NRL_MEGATRON_CHECKPOINT_DIR` were passed into both containers. First, on the
**head container**:

```bash
mkdir -p "${RUN_CHECKPOINT_DIR}/permission-probe/policy"
printf 'head\n' > "${RUN_CHECKPOINT_DIR}/permission-probe/policy/head.txt"
```

Then, on the **worker container**:

```bash
test -s "${RUN_CHECKPOINT_DIR}/permission-probe/policy/head.txt"
printf 'worker\n' > "${RUN_CHECKPOINT_DIR}/permission-probe/policy/worker.txt"
mkdir -p "${RUN_CHECKPOINT_DIR}/permission-probe/policy/worker-nested"
```

Back on the **head container**, verify `worker.txt` and `worker-nested`.
Repeat this probe for the log and conversion directories. Stop if any
operation fails; this reproduces the nested writes needed for checkpointing.
Run bounded concurrent write/fsync tests sized for checkpoint shards before
training, and verify readback from the other Station. Do not fill the disk.

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

The `grep` output must show `await self.llm.sleep(level=2)`. Inspect the
targeted patch with `git diff -- "${VLLM_ASYNC_WORKER}"`. The container can
ship with tracked functional tests deleted and an untracked `NOTICES.txt`,
so a clean overall Git status is not required. Do not discard unrelated
changes. If the pre-patch `test` fails, stop: the source revision is not the
revision this recipe expects.

Finally, ensure the active shell has the required limit:

```bash
ulimit -Sn 65535
```

### Budget memory and storage for checkpoint save and restore

Checkpointing requires CPU memory and disk space in addition to the policy
and optimizer's training footprint. With asynchronous saves, monitor
`/dev/shm` as well as host RAM. Synchronous saving removes asynchronous
staging but can still exhaust CPU memory; it is not a complete OOM remedy.

To test synchronous saving, add this override to the smoke and full-run
commands:

```text
policy.megatron_cfg.checkpoint.async_save=false
```

Keep `checkpointing.enabled=true` and `checkpointing.save_optimizer=true`.
Measure peak host RAM, GPU memory, and shared-memory usage while saving,
restoring, and completing another optimizer update. A run that skips
checkpointing does not validate checkpoint memory requirements.

Reserve space for retained checkpoints, one in-flight save, model cache,
conversion, data, logs, and exports. Measure the first completed checkpoint's
size before setting retention. The base YAML retains effectively unlimited
checkpoints; reduce `checkpointing.keep_top_k` to a bounded value that fits
your storage budget. The smoke command below retains two checkpoints.

Do not enable custom checkpoint, optimizer-repair, or THP environment hooks
unless their implementation is included in the installed source and has
been reviewed for that revision. Unrecognized environment flags do not
install or activate a fix.

## 5. Start the two-node Ray cluster

NeMo-RL leverages Ray for serving the policy and inference engines and
coordinating communication between the hosts. A Ray cluster needs to be started
inside the containers for NeMo-RL to run in.

### Node-to-node ports and pinned-source limitations

Use an isolated, trusted fabric; Ray and Gym services are not intended to
be exposed to untrusted networks. This pinned stack has the following
connectivity requirements:

| Component | TCP ports / allocation |
| --- | --- |
| Ray GCS | Head port 6379 |
| Ray client | Head port 10001 by default |
| Ray workers | Default range 10002–19999 |
| Ray node/object managers and agents | Dynamically allocated unless explicitly pinned at Ray startup |
| NeMo RL master/TCPStore | Default 1400–1999 |
| vLLM HTTP | **Ephemeral ports** in this recipe's deferred-loading path |
| NeMo Gym HTTP | **5000–5999** in this pinned revision |
| NCCL/Gloo bootstrap | Additional dynamically allocated sockets |
| Optional NFSv4 | Server port 2049; NFS configuration may require additional services |

The YAML's generation range `11001–15000` and Gym range `15001–20000`
are **not honored by this pinned source path**. Deferred vLLM initialization
reserves a socket with `bind(("", 0))`; Gym setup omits the top-level range
fields when constructing `NemoGymConfig`. Allow trusted peer connectivity
or fix/revalidate those upstream paths before enforcing a restricted port
allowlist. Inspect actual listeners rather than relying on the YAML alone.

Keep all Ray commands in the same `uv run` environment on both Stations.
After pinning source, run a two-node NCCL collective test inside the policy
runtime as well as the hardware playbook's perftest; the latter alone does
not exercise NCCL's Data Direct path.

### Start Ray on the head node

Run inside the container on the **head only**:

```bash
cd /opt/nemo-rl
numactl --membind="${CPU_MEM_NODE}" uv run ray start \
  --head \
  --node-ip-address="${HEAD_IP}" \
  --port=6379 \
  --disable-usage-stats \
  --num-gpus=1

uv run ray status
```

At this point Ray should report one node and `1.0 GPU` total.

### Join the Ray cluster on the worker

Run inside the container on the **worker only**:

```bash
cd /opt/nemo-rl
uv run ray stop --force
numactl --membind="${CPU_MEM_NODE}" uv run ray start \
  --address="${HEAD_IP}:6379" \
  --node-ip-address="${WORKER_IP}" \
  --num-gpus=1

uv run ray status
```

The worker command must report a successful connection. Keep this container
shell open.

Back inside the container on the **head**, verify both nodes and both GPUs:

```bash
cd /opt/nemo-rl
uv run ray status
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

The Python-inductive ARC-AGI examples ask the model to infer grid
transformations and implement them in Python; NeMo Gym verifies the outputs.

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
dataset["train"].select(range(min(32, len(dataset["train"])))).to_json(
    "smoke_train.jsonl"
)
dataset["validation"].select(range(min(4, len(dataset["validation"])))).to_json(
    "smoke_validation.jsonl"
)
```

Run the converter and verify all four outputs:

```bash
python3 prep.py
wc -l train.jsonl validation.jsonl smoke_train.jsonl smoke_validation.jsonl

python3 - <<'PY'
import json
from pathlib import Path

for name in ("train.jsonl", "validation.jsonl", "smoke_train.jsonl", "smoke_validation.jsonl"):
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

## 9. Run a bounded smoke test, then launch training

### Smoke test and checkpoint resume

Use a separate `RUN_ID` and prepared permission-checked directories for the
smoke test. Keep the recipe's 16 prompts × 16 generations and 16,384-token
limit to exercise its baseline memory footprint. The converter creates 32
smoke training rows, enough for two full batches. Four rows with the default
16-prompt loader would produce zero batches because it drops incomplete batches.

Run on the **head container**:

```bash
cd /opt/nemo-rl
export RAY_ADDRESS="${HEAD_IP}:6379"
NRL_FORCE_REBUILD_VENVS=true \
numactl --membind="${CPU_MEM_NODE}" \
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/nemo_gym/grpo_lightning35_station.yaml \
  data.train.data_path=/shared/data/smoke_train.jsonl \
  data.validation.data_path=/shared/data/smoke_validation.jsonl \
  grpo.max_num_steps=2 grpo.val_period=1 \
  grpo.val_num_generations_per_prompt=2 \
  checkpointing.save_period=1 checkpointing.keep_top_k=2 \
  logger.wandb_enabled=false \
  logger.log_dir="${RUN_LOG_DIR}" \
  checkpointing.checkpoint_dir="${RUN_CHECKPOINT_DIR}" \
  ++policy.megatron_cfg.env_vars.NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR}" \
  '++policy.megatron_cfg.env_vars.NCCL_IB_DATA_DIRECT="0"' \
  ++policy.megatron_cfg.env_vars.GLOO_SOCKET_IFNAME="${COMM_IFACE}" \
  ++policy.megatron_cfg.env_vars.NCCL_SOCKET_IFNAME="${COMM_IFACE}"
```

Do not set `grpo.max_val_samples`: the Gym driver rejects non-null overrides.
It uses the supplied validation file directly and sets validation batch size
from that file. Do not interpret this small validation set as a benchmark.

Require both updates, finite training metrics, verifier rewards, successful
weight refit, and completed `step_1`/`step_2` checkpoints. `tmp_step_*` alone
is an incomplete checkpoint. Saving optimizer state is enabled in the YAML.
For resume validation, rerun the same command with `grpo.max_num_steps=3`
and `grpo.max_num_epochs=2`, keeping its directories and data unchanged;
verify that the driver restores step 2 and completes another update. The
extra epoch allows this 32-row smoke dataset to supply another full batch.

> [!IMPORTANT]
> Complete the entire smoke test and resume check before launching a long
> run. Both updates, checkpoint finalization, and a resumed optimizer update
> must pass. An initialized model or a completed training pass alone does
> not establish that checkpoint permissions and memory budgets are correct.

### Full distributed run

Choose a fresh full-run `RUN_ID` and permission-check its directories. Keep
the verified converted-model directory if reusing the same model/configuration;
do not reuse an incomplete conversion. Export the run variables inside the
head container before launching.


Launch the full training run on the **head node**:

```bash
cd /opt/nemo-rl

NRL_FORCE_REBUILD_VENVS=true \
numactl --membind="${CPU_MEM_NODE}" \
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/nemo_gym/grpo_lightning35_station.yaml \
  '++policy.megatron_cfg.env_vars.NCCL_IB_DATA_DIRECT="0"' \
  ++policy.megatron_cfg.env_vars.NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR}" \
  logger.log_dir="${RUN_LOG_DIR}" \
  checkpointing.checkpoint_dir="${RUN_CHECKPOINT_DIR}" \
  ++policy.megatron_cfg.env_vars.GLOO_SOCKET_IFNAME="${COMM_IFACE}" \
  ++policy.megatron_cfg.env_vars.NCCL_SOCKET_IFNAME="${COMM_IFACE}"
```

To run without W&B, add this override to the launch command:

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
activity during rollout or training. Initialization, refit, offload, and
checkpointing can have low utilization. GPU memory allocation alone does not
prove that an optimizer update completed.

Inspect the shared storage for logs and checkpoints. These directories will
become populated as training completes a few steps:

```bash
find "${RUN_LOG_DIR}" -maxdepth 2 -type f | sort | tail -n 20
find "${RUN_CHECKPOINT_DIR}" -maxdepth 3 -type f | sort | tail -n 20
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

### NCCL DMA-BUF registration fails with error 524

Check kernel logs for `topology not supported for mapping type FORCE_PCIE`.
Changing the RDMA plugin alone may not avoid the unsupported registration
path. Confirm `NCCL_IB_DATA_DIRECT=0` is present on
both Ray nodes and in `policy.megatron_cfg.env_vars`; numeric Hydra values
must be quoted as strings as shown in the launch commands. Test GPU
all-reduce and object gathering before retrying. Confirm the logs still show
RDMA/GDRDMA rather than `NET/Socket`; do not label TCP fallback a fabric pass.

### Initial model conversion was interrupted

This revision tests whether `iter_0000000` exists, not whether conversion
finished. An empty or partially written directory can suppress reconversion
and produce a misleading `run_config.yaml` missing/shared-mount error.
Inspect the conversion directory on both nodes. Preserve incomplete output
and use a fresh `NRL_MEGATRON_CHECKPOINT_DIR` (exported on both nodes and
passed in policy `env_vars`) or an administrator-approved quarantine step.
Do not delete the Hugging Face model cache or a complete prior experiment.

### Checkpoint saving fails with Permission denied

Inspect numeric ownership and ACLs of the **nested** `tmp_step_*/policy`
path on both hosts. A head-local root-owned `0755` parent is not writable by
root-squashed worker processes, even when `/shared` itself is `0777`.
Recheck inherited ACLs and rerun the nested-write probe. Do not work around
this by disabling checkpointing or root-squash. After correcting permissions,
repeat save and resume validation; a completed training pass is not a saved run.

### Shared storage stalls or the process exits with 137

Exit 137 indicates SIGKILL, not proof of OOM. Inspect host kernel logs,
container state, and cgroup `memory.events`. If NFS reports `server not
responding`, test storage health and inspect server threads before restarting
training. Blocked filesystem writes or journal commits require administrator
investigation. Keep driver console output on a local filesystem, for example
with `set -o pipefail` and `tee /var/tmp/grpo-${RUN_ID}.log` appended to the
launch command. Preserve incomplete conversion/checkpoint artifacts.

Use the head's direct local export path and the worker's NFS mount rather
than mounting the head's own export back onto itself. This avoids an
unnecessary NFS client path; it does not eliminate all filesystem or storage
failure modes. Revalidate sustained write/fsync and cross-node readback
after recovery.

### Ray asserts during shutdown after another failure

Ray's `The process is already initialized for core worker` assertion can
appear during cleanup after another exception. Diagnose the first exception
in the log before treating the cleanup assertion as the original training
failure. Preserve logs and verify actors terminate before starting another run.

### CUDA runs out of memory during policy training

- Confirm the config contains `optimizer_cpu_offload: true` and
  `optimizer_offload_fraction: 0.75`.
- Confirm `grep -n 'self.llm.sleep'` shows `level=2` in
  `vllm_worker_async.py` on both Stations.
- Confirm Ray advertises exactly one GPU per Station and stop other GPU
  workloads.
- Do not increase sequence length, rollout count, or vLLM memory utilization
  until the baseline run succeeds.

### Second vLLM wake-up, checkpoint save, or post-resume OOM

First distinguish CUDA OOM, host OOM, shared-memory exhaustion, and Ray's
memory guard from permission failures and filesystem stalls. Record
`nvidia-smi`, `df -h /dev/shm`, process RSS, `numastat -p <PID>`, and
`/proc/<PID>/smaps_rollup` at each transition. Use the actual worker PID,
not just the driver's PID. Verify Ray and the driver inherited the intended
NUMA memory policy. Check `AnonHugePages` when investigating gradual RSS
growth; increasing RSS alone does not establish THP as the cause. Do not
globally disable THP or modify optimizer master buffers without isolating
the cause and reviewing a revision-compatible fix.

Keep activation checkpointing, 75% optimizer CPU offload, microbatch size 1,
and the exact vLLM sleep-level-2 patch. If the issue occurs during checkpoint
save/restore or later RSS growth, inspect checkpoint staging allocations and
retained optimizer buffers, then reproduce save and restore in a bounded
run. These symptoms are not addressed by the sleep patch alone. If the
installed source cannot save and restore within the available memory budget,
stop the long-run workflow until a reviewed fix is available.

### vLLM fails during Nemotron-H MoE weight refit

Run `git log -1 --format=%s` on both Stations and verify that the subject is
`fix(vllm): correct MoE refit shard_dim for grouped/3D experts (NemotronH)`.
That local commit was cherry-picked from `6bb55bc`. Recreate the pinned patch
stack if either working tree differs.

### `/shared` is read-only or differs between nodes

- Compare `df -hT /shared` and a test file from both containers.
- For the demonstration NFS setup, confirm `exportfs -v` on the head and
  `mountpoint /mnt/nfs` on the worker and the local export path on the head.
- Resolve UID/GID, root-squash, and permissions through your storage
  administrator rather than disabling security controls on a shared network.

### Dependency-build failures after changing revisions

Keep the pinned source/submodules. If dependency resolution on aarch64 fails
on an x86_64 split or conflicting `transformers` pins, capture the exact
resolver error and compare the checked-out source with its lockfile. Inspect
the actual `uv sync` and `uv run` calls before applying any patch; require an
exact-match assertion so a no-op substitution cannot falsely claim a fix.
Review revision-specific dependency changes before rebuilding.
`--frozen` skips lockfile freshness validation; it does not prove that a stale
lockfile matches changed source.

Do not overwrite a conflicted `patches.py` with `git checkout --theirs` merely
because the original change was additive: this can discard unrelated fixes.
Recreate the exact pinned patch stack or review the conflict manually. Do not
regenerate the container fingerprint just to hide a mismatch. Rebuild and
verify dependencies first; retain `NRL_FORCE_REBUILD_VENVS=true` for the first
launch after changing source or installing hooks.

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
- [Single-Station LoRA customization recipe](lora.md)
- [Two-Station SFT customization recipe](sft.md)
