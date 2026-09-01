# Customizing Nemotron 3.5 Lightning on Two DGX Stations with SFT

This guide runs full-weight supervised fine-tuning (SFT) for
[NVIDIA Nemotron 3.5 Lightning 30B-A3B BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
on two DGX Station GB300 systems. It uses one B300 GPU from each Station,
NeMo AutoModel for distributed model training, and the `interactive_agent`
subset of
[Nemotron-SFT-Agentic-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2)
to customize the model for multi-turn agentic tool use.

For other customization workflows on this topology, see the
[DGX Station recipe index](README.md). For reinforcement learning, go directly
to the [two-Station GRPO recipe](grpo.md).

> [!NOTE]
> The instructions use one Station as the **head** and the other as the
> **worker**. Keep those roles unchanged for the entire run. Commands marked
> **both Stations** must be run separately on each system; commands marked
> **head only** or **worker only** run on that system only.

## What this recipe does

| Component | Configuration |
| --- | --- |
| Hardware | Two DGX Station GB300 systems [connected for distributed workloads](https://build.nvidia.com/station/connect-two-stations/overview) |
| Training | Full-weight SFT with NeMo AutoModel and FSDP2 |
| Parallelism | Context parallel size 2 and expert parallel size 2 across two GPUs |
| Sequence length | 32,768 tokens |
| Dataset | `interactive_agent` subset of [Nemotron-SFT-Agentic-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2) |

The configuration runs 100 optimizer steps as an end-to-end demonstration. It
is a customization starting point, not a reproduction of the model's complete
post-training program or published evaluation results.

## Prerequisites

Before starting, make sure you have:

- Two DGX Station GB300 systems with current DGX Base OS, NVIDIA drivers,
  Docker, and NVIDIA Container Toolkit. The
  [DGX Station software stack](https://docs.nvidia.com/dgx/dgx-station-development-guide/porting/software-requirements.html)
  includes these components.
- The two Stations connected and validated for distributed GPU workloads.
  Complete NVIDIA's
  [Connect Two DGX Stations for Distributed Workloads](https://build.nvidia.com/station/connect-two-stations/overview)
  playbook, including its RDMA and GPU-memory tests, before continuing.
- A Linux user with `sudo` access on both Stations.
- Internet access to `nvcr.io` and Hugging Face.
- Shared storage visible from both Stations at the same host path. The optional
  NFS section below creates a simple demonstration share if one is not
  available.

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

### Verify the NVIDIA driver

Run on **both Stations**:

```bash
nvidia-smi -L
```

The command should show a single B300 GPU in the list. If a display-only graphics card is included on the Station, that may also appear in the list.

### Record the network values

Run on **both Stations** to list IPv4 addresses and Linux network-device names:

```bash
hostname
ip -4 -br address show
ip -4 route
```

Choose the high-speed interface used for traffic between the Stations. If you
followed the two-Station networking playbook, map each ConnectX HCA to its
Linux network device with:

```bash
ibdev2netdev
```

`COMM_IFACE` must be the Linux network-device name, such as `enP3s3f1np1`,
not an HCA name such as `mlx5_0`. This guide assumes the selected interface has
the same name on both Stations.

Record the values and export the same values in the host shell on **both
Stations**:

```bash
export HEAD_IP=<HEAD_STATION_IPV4>
export WORKER_IP=<WORKER_STATION_IPV4>
export COMM_IFACE=<LINUX_NETWORK_INTERFACE>
```

Verify the selected path in both directions:

```bash
# Run on the head.
ping -c 4 "${WORKER_IP}"

# Run on the worker.
ping -c 4 "${HEAD_IP}"
```

Do not continue until the pings and the prerequisite fabric tests succeed.
The Stations must permit trusted node-to-node traffic on TCP port `29500` for
the PyTorch rendezvous and on the ports selected by NCCL for data transfer.

## 2. Configure shared storage

If both Stations already mount shared storage at the same host path, export
that path on **both Stations** and skip the remainder of this section:

```bash
export SHARED_HOST_PATH=<SHARED_STORAGE_HOST_PATH>
```

Verify that a file written from either Station appears on the other before
launching the containers.

### Optional: create an NFS export on the head

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
> Mode `0777` lets root processes in a root-squashed containerized NFS client
> write to this demonstration share. Restrict the client network as narrowly
> as possible, and replace these permissions with your site's identity and
> storage policy outside an isolated lab setup.

Determine the network CIDR containing both selected Station IPs. For a direct
link this is commonly a `/30`, such as `192.168.240.0/30`. Export it on the
**head only**:

```bash
export CLIENT_NETWORK_CIDR=<NETWORK_CIDR_FOR_BOTH_STATIONS>

printf '/mnt/nfs_share %s(rw,sync,no_subtree_check)\n' \
  "${CLIENT_NETWORK_CIDR}" \
  | sudo tee /etc/exports.d/nemotron-sft.exports

sudo exportfs -ra
sudo exportfs -v
```

Confirm that `/mnt/nfs_share` is listed before continuing.

### Optional: mount the NFS export on both Stations

Mount the newly created NFS server on **both Stations**, including the head,
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

## 3. Launch the NeMo AutoModel container

If your NGC organization requires authentication, log in on **both Stations**:

```bash
docker login nvcr.io
```

Use `$oauthtoken` as the username and an NGC API key as the password.

This recipe pins the public, stable NeMo AutoModel 26.08 container. Export the
image and confirm all required values on **both Stations**:

```bash
export AUTOMODEL_IMAGE=nvcr.io/nvidia/nemo-automodel:26.08

printf 'HEAD_IP=%s\nWORKER_IP=%s\nCOMM_IFACE=%s\nSHARED_HOST_PATH=%s\n' \
  "${HEAD_IP}" "${WORKER_IP}" "${COMM_IFACE}" "${SHARED_HOST_PATH}"

test -d "${SHARED_HOST_PATH}"
docker pull "${AUTOMODEL_IMAGE}"
```

Start one retained container on **each Station**:

```bash
docker run -it \
  --name nemo-automodel-sft \
  --gpus all \
  --network host \
  --device=/dev/infiniband \
  --shm-size=128g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HEAD_IP="${HEAD_IP}" \
  -e WORKER_IP="${WORKER_IP}" \
  -e COMM_IFACE="${COMM_IFACE}" \
  -e GLOO_SOCKET_IFNAME="${COMM_IFACE}" \
  -e NCCL_SOCKET_IFNAME="${COMM_IFACE}" \
  -e NCCL_IB_HCA=mlx5_0,mlx5_1 \
  -e HF_HOME=/shared/.cache/huggingface \
  -v "${SHARED_HOST_PATH}:/shared" \
  "${AUTOMODEL_IMAGE}"
```

`--device=/dev/infiniband` exposes the Station's RDMA devices to the
container. If your validated fabric uses different HCA names, replace
`mlx5_0,mlx5_1` with the names reported by `ibdev2netdev`.

The rest of the guide runs inside these two container shells unless a step
says otherwise. Keep both shells running for the duration of training. The
containers are retained if a shell exits; re-enter one with:

```bash
docker start -ai nemo-automodel-sft
```

Inside both containers, verify the environment and the SFT dataset adapter:

```bash
nvidia-smi -L
df -hT /shared
test -d /dev/infiniband
ls -1 /dev/infiniband

cd /opt/Automodel
python3 -c 'import nemo_automodel; print(nemo_automodel.__version__)'
python3 -c 'from nemo_automodel.components.datasets.llm.agent_chat import make_agent_chat_dataset; print("Agent SFT adapter is available.")'
test -f nemo_automodel/recipes/llm/train_ft.py
```

A single B300 GPU should be visible in each container, and `/shared` must
resolve to the same storage from both containers. If the agent SFT adapter is
missing, verify that the image tag is exactly `26.08.00`.

## 4. Authenticate and populate the shared model cache

Run on the **head only**, inside the container:

```bash
mkdir -p /shared/.cache/huggingface
export HF_HOME=/shared/.cache/huggingface
hf auth login

export MODEL_ID=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
hf download "${MODEL_ID}"
```

The Hugging Face token and model cache now reside on shared storage and are
available to both containers. Review and accept any applicable model or
dataset terms before downloading.

## 5. Prepare the agentic SFT dataset

The `interactive_agent` JSONL file contains heterogeneous nested schemas that
cannot be loaded directly through Arrow. The converter below downloads the
raw JSONL, normalizes messages one record at a time, renders tool calls in the
Nemotron 3.5 Lightning chat format, and serializes the tool definitions to
avoid Arrow schema inference.

Run this section on the **head only**, inside the container.

Create the data directory:

```bash
mkdir -p /shared/sft/data
cd /shared/sft/data
```

Create `/shared/sft/data/prepare.py` with the following contents:

```python
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

OUTPUT = Path("/shared/sft/data")
VALIDATION_SIZE = 256
DATASET_REVISION = "4fb69cd40dbf36da60c73321e094e093946e60e9"


def parse_json(value):
    return json.loads(value) if isinstance(value, str) else value


def text(value):
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def render_call(call):
    call = parse_json(call)
    function = parse_json(call.get("function", call))
    arguments = parse_json(function.get("arguments") or {})

    if not isinstance(arguments, dict):
        raise TypeError("Tool-call arguments must be a JSON object")

    parameters = []
    for name, value in arguments.items():
        rendered_value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        parameters.append(
            f"<parameter={name}>\n"
            f"{rendered_value}\n"
            f"</parameter>\n"
        )

    return (
        "<tool_call>\n"
        f"<function={function['name']}>\n"
        f"{''.join(parameters)}"
        "</function>\n"
        "</tool_call>"
    )


def normalize(row):
    raw_messages = parse_json(row["messages"])
    messages = []

    for raw_message in raw_messages:
        message = parse_json(raw_message)
        calls = parse_json(message.get("tool_calls") or [])
        if isinstance(calls, dict):
            calls = [calls]
        if message.get("function_call"):
            calls = [message["function_call"], *calls]

        rendered_calls = "\n".join(render_call(call) for call in calls)
        content = text(message.get("content"))

        messages.append(
            {
                "role": message["role"],
                "content": "\n".join(filter(None, [content.rstrip(), rendered_calls])),
                "reasoning_content": text(message.get("reasoning_content")),
            }
        )

    tools = parse_json(row.get("tools") or [])
    return {
        "messages": messages,
        # Encoding tools prevents Arrow from inferring a heterogeneous schema.
        "tools": json.dumps(tools, ensure_ascii=False),
    }


OUTPUT.mkdir(parents=True, exist_ok=True)

source = hf_hub_download(
    repo_id="nvidia/Nemotron-SFT-Agentic-v2",
    filename="data/interactive_agent.jsonl",
    repo_type="dataset",
    revision=DATASET_REVISION,
    local_dir=OUTPUT / "raw",
)

with (
    open(source, encoding="utf-8") as input_file,
    open(OUTPUT / "train.jsonl", "w", encoding="utf-8") as train_file,
    open(OUTPUT / "validation.jsonl", "w", encoding="utf-8") as validation_file,
):
    for index, line in enumerate(input_file):
        destination = validation_file if index < VALIDATION_SIZE else train_file
        destination.write(json.dumps(normalize(json.loads(line)), ensure_ascii=False) + "\n")
```

Run the converter from the AutoModel environment:

```bash
cd /opt/Automodel
python3 /shared/sft/data/prepare.py
```

The source file is several gigabytes, so the download and conversion can take
some time. Verify the record counts:

```bash
wc -l /shared/sft/data/train.jsonl /shared/sft/data/validation.jsonl
```

Expected output for the pinned dataset revision:

```text
  278624 /shared/sft/data/train.jsonl
     256 /shared/sft/data/validation.jsonl
  278880 total
```

Validate the converted schema before training:

```bash
test "$(wc -l < /shared/sft/data/train.jsonl)" -eq 278624
test "$(wc -l < /shared/sft/data/validation.jsonl)" -eq 256

python3 - <<'PY'
import json
from pathlib import Path

for name in ("train.jsonl", "validation.jsonl"):
    path = Path("/shared/sft/data") / name
    with path.open(encoding="utf-8") as stream:
        row = json.loads(stream.readline())
    assert row["messages"], f"{name} has no messages"
    assert all("role" in message and "content" in message for message in row["messages"])
    assert isinstance(json.loads(row["tools"]), list)

print("Agentic SFT JSONL files are ready.")
PY
```

## 6. Install the two-Station SFT configuration

Create the shared configuration directory on the **head only**:

```bash
mkdir -p /shared/sft/configs /shared/sft/checkpoints
```

Download the checked-in
[`sft_lightning35_station.yaml`](sft_lightning35_station.yaml) configuration:

```bash
curl -fL \
  https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/usage-cookbook/Nemotron-3.5-Lightning/dgx-station-recipes/sft_lightning35_station.yaml \
  -o /shared/sft/configs/sft_lightning35_station.yaml
```

If you are working from a local clone of this repository instead, copy that
file to the same shared destination.

The configuration has these important properties:

- FSDP2 provides the distributed training strategy. Context parallelism
  partitions each long sequence, while expert parallelism distributes the MoE
  experts across the two-rank world.
- Context and expert parallel sizes are both 2, matching the two-rank topology.
- `truncate_history: true` removes complete oldest exchanges to preserve the
  supervised final response within the 32,768-token limit.
- `train_on_last_turn_only: false` supervises assistant text and tool calls in
  every assistant turn. The adapter masks user and tool-response tokens.
- `drop_history_reasoning_content: true` removes hidden reasoning from earlier
  turns, while `mask_reasoning_content: false` trains on any reasoning content
  that remains after that history cleanup.
- `max_steps: 100` makes this a bounded demonstration. Because
  `ckpt_every_steps` is 1,000, the run saves its consolidated checkpoint at the
  end rather than producing periodic checkpoints.

Verify the shared files on **both Stations**:

```bash
test -s /shared/sft/data/train.jsonl
test -s /shared/sft/data/validation.jsonl
test -s /shared/sft/configs/sft_lightning35_station.yaml
test -w /shared/sft/checkpoints

python3 - <<'PY'
from pathlib import Path

import yaml

path = Path("/shared/sft/configs/sft_lightning35_station.yaml")
config = yaml.safe_load(path.read_text())
assert config["model"]["pretrained_model_name_or_path"].endswith("30B-A3B-BF16")
assert config["distributed"]["cp_size"] == 2
assert config["distributed"]["ep_size"] == 2
assert config["dataset"]["seq_length"] == 32768
assert "peft" not in config
print("Two-Station full-weight SFT configuration is ready.")
PY
```

## 7. Launch distributed training

AutoModel supports PyTorch's distributed launcher. Because this setup does not
use Slurm or Kubernetes, start one `torchrun` process directly on each Station.
The two commands must use the same head address and port, with unique node
ranks.

If you need another shell in either running container, open a new host terminal
on that Station and run:

```bash
docker exec -it nemo-automodel-sft bash
```

### Start the head process

Run inside the container on the **head only**:

```bash
cd /opt/Automodel

torchrun \
  --nnodes=2 \
  --nproc-per-node=1 \
  --node-rank=0 \
  --master-addr="${HEAD_IP}" \
  --master-port=29500 \
  nemo_automodel/recipes/llm/train_ft.py \
  -c /shared/sft/configs/sft_lightning35_station.yaml
```

The head waits for the worker at the rendezvous.

### Start the worker process

Promptly run inside the container on the **worker only**:

```bash
cd /opt/Automodel

torchrun \
  --nnodes=2 \
  --nproc-per-node=1 \
  --node-rank=1 \
  --master-addr="${HEAD_IP}" \
  --master-port=29500 \
  nemo_automodel/recipes/llm/train_ft.py \
  -c /shared/sft/configs/sft_lightning35_station.yaml
```

The worker must use `--node-rank=1` and must still use the **head** Station's IP
for `--master-addr`. Reusing rank 0 or substituting the worker IP prevents the
two processes from forming one distributed job.

## 8. Monitor training and artifacts

From another host shell on either Station, inspect GPU activity:

```bash
docker exec nemo-automodel-sft nvidia-smi
```

Training logs remain attached to the two `torchrun` terminals. Both terminals
should report ranks joining the same two-process job, followed by model and
dataset initialization and per-step loss output.

Inspect the shared checkpoint directory as the run completes:

```bash
find /shared/sft/checkpoints -maxdepth 3 -type f | sort | tail -n 40
df -h /shared
```

With the supplied 100-step configuration, expect the consolidated checkpoint
at the end of the run. Monitor free storage throughout training, especially
after increasing `max_steps` or enabling periodic checkpoint retention.

## 9. Stop or clean up

To stop training early, press `Ctrl+C` in one `torchrun` terminal, then stop the
other rank if it does not exit automatically. Preserve both terminal outputs
when diagnosing a distributed failure.

Exit each container when training is no longer running:

```bash
exit
```

The containers remain available for inspection or restart. Remove them on
**both Stations** only when they are no longer needed:

```bash
docker rm nemo-automodel-sft
```

The model cache, data, configuration, and checkpoints under `/shared` are not
removed with the containers.

## Troubleshooting

### The two ranks time out during rendezvous

- Confirm both commands use the head Station's address in `--master-addr`.
- Confirm the head uses `--node-rank=0` and the worker uses `--node-rank=1`.
- Re-run the bidirectional ping checks and confirm TCP port `29500` is open on
  the selected interface.
- Confirm no stale process is already listening on port `29500` with
  `ss -ltnp | grep 29500` on the head.

### NCCL hangs or uses the wrong interface

- Confirm `COMM_IFACE` is the Linux device name present on both Stations, not
  an HCA name such as `mlx5_0`.
- Confirm `/dev/infiniband` exists inside both containers and that
  `NCCL_IB_HCA` names the HCAs reported by `ibdev2netdev`.
- Re-run the NVIDIA two-Station fabric validation.
- For the next diagnostic run, export `NCCL_DEBUG=INFO` in both container
  shells. A healthy RDMA run reports `NET/IB`; `NET/Socket` indicates TCP
  fallback.

### AutoModel cannot import `agent_chat`

Verify the public container tag in both host shells:

```bash
docker inspect nemo-automodel-sft --format '{{.Config.Image}}'
```

It must report `nvcr.io/nvidia/nemo-automodel:26.08`. Recreate a container
started from another release rather than modifying its installed package in
place.

### Dataset preparation fails or produces different counts

- Confirm `prepare.py` sets `DATASET_REVISION` to
  `4fb69cd40dbf36da60c73321e094e093946e60e9`.
- Confirm the shared volume has enough free space for the 6.3 GB raw file and
  both converted JSONL files.
- Inspect the last processed line and the Python exception before rerunning;
  the line-by-line converter intentionally bypasses the dataset's heterogeneous
  Arrow schema.

### CUDA runs out of memory

- Stop other GPU workloads and confirm each Station exposes exactly one B300
  GPU to its container.
- Reduce `step_scheduler.local_batch_size` from 2 to 1 while leaving the global
  batch size at 8.
- If needed, reduce both dataset `seq_length` values from 32,768. This changes
  the training workload, so revalidate quality and throughput after doing so.
- Do not change context or expert parallel size independently; both are set to
  2 for this two-rank topology.

### `/shared` is read-only, slow, or differs between nodes

- Compare `df -hT /shared` and a test file from both containers.
- For the demonstration NFS setup, confirm `exportfs -v` on the head and
  `mountpoint /mnt/nfs` on both hosts.
- Resolve UID/GID, root-squash, and performance requirements through your
  storage administrator for production use.

## References

- [NeMo AutoModel container on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel)
- [NeMo AutoModel multi-turn agent SFT guide](https://docs.nvidia.com/nemo/automodel/latest/recipes-e2e-examples/agent-sft)
- [Nemotron 3.5 Lightning model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
- [Nemotron-SFT-Agentic-v2 dataset card](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2)
- [DGX Station software stack](https://docs.nvidia.com/dgx/dgx-station-development-guide/porting/software-requirements.html)
- [Connect Two DGX Stations for Distributed Workloads](https://build.nvidia.com/station/connect-two-stations/overview)
- [Two-Station GRPO customization recipe](grpo.md)
