# Customizing Nemotron 3.5 Lightning on One DGX Station with LoRA

This guide trains a low-rank adaptation (LoRA) adapter for
[NVIDIA Nemotron 3.5 Lightning 30B-A3B BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
on one DGX Station GB300. It uses the Station's B300 GPU, NeMo AutoModel, and
the `interactive_agent` subset of
[Nemotron-SFT-Agentic-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2)
to customize the model for multi-turn agentic tool use while updating only a
small set of adapter parameters.

For other customization workflows, see the
[DGX Station recipe index](README.md). Use the
[two-Station SFT recipe](sft.md) for full-weight supervised fine-tuning or the
[two-Station GRPO recipe](grpo.md) for reinforcement learning.

> [!NOTE]
> This recipe produces a LoRA adapter, not a standalone copy of the full
> model. Inference requires the original base model together with the adapter,
> unless you merge the adapter into the base model in a separate step.

## What this recipe does

| Component | Configuration |
| --- | --- |
| Hardware | One DGX Station GB300 using one B300 GPU |
| Training | Rank-8 LoRA with NeMo AutoModel and FSDP2 |
| Parallelism | Tensor, context, and expert parallel sizes all set to 1 |
| Sequence length | 32,768 tokens |
| Dataset | `interactive_agent` subset of [Nemotron-SFT-Agentic-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2) |
| Output | Safetensors LoRA adapter checkpoint |

The configuration runs 100 optimizer steps as an end-to-end demonstration. It
is a customization starting point, not a reproduction of the model's complete
post-training program or published evaluation results.

## Prerequisites

Before starting, make sure you have:

- One DGX Station GB300 with current DGX Base OS, NVIDIA drivers, Docker, and
  NVIDIA Container Toolkit. The
  [DGX Station software stack](https://docs.nvidia.com/dgx/dgx-station-development-guide/porting/software-requirements.html)
  includes these components.
- A Linux user with `sudo` access on the Station.
- Internet access to `nvcr.io` and Hugging Face.
- A Hugging Face account and token with access to the model and dataset.
- Persistent local or shared storage with approximately 200GB of available
  capacity for the base model, the source and converted datasets, and adapter
  checkpoints.

## 1. Prepare the Station

### Enable non-root Docker access

Add non-root Docker access for your user if it is not already enabled:

```bash
sudo usermod -aG docker "${USER}"
newgrp docker
docker version
```

Membership in the `docker` group grants root-equivalent access to the host.

### Verify the NVIDIA driver

```bash
nvidia-smi -L
```

The command should show the Station's B300 GPU. If the Station includes a
display-only graphics device, that device may also appear in the list.

## 2. Configure persistent storage

The container is disposable, so mount a persistent host directory at
`/shared` to retain downloads, converted data, configuration, and checkpoints.

If you already have suitable local or shared storage, export its host path:

```bash
export PERSISTENT_HOST_PATH=<PERSISTENT_STORAGE_HOST_PATH>
```

Otherwise, create `/workspace` on the Station:

```bash
sudo mkdir -p /workspace
sudo chown "${USER}:$(id -gn)" /workspace
export PERSISTENT_HOST_PATH=/workspace
```

Verify the directory before launching the container:

```bash
test -d "${PERSISTENT_HOST_PATH}"
test -w "${PERSISTENT_HOST_PATH}"
df -hT "${PERSISTENT_HOST_PATH}"
```

An NFS or other shared filesystem is also valid, but this recipe only needs a
single Station and shared storage is not required.

## 3. Launch the NeMo AutoModel container

If your NGC organization requires authentication, log in on the Station:

```bash
docker login nvcr.io
```

Use `$oauthtoken` as the username and an NGC API key as the password.

This recipe pins the public, stable NeMo AutoModel 26.08 container. Export the
image, check the persistent path, and pull the image:

```bash
export AUTOMODEL_IMAGE=nvcr.io/nvidia/nemo-automodel:26.08

printf 'PERSISTENT_HOST_PATH=%s\n' "${PERSISTENT_HOST_PATH}"
test -d "${PERSISTENT_HOST_PATH}"
docker pull "${AUTOMODEL_IMAGE}"
```

Start a retained container with the B300 GPU and persistent directory mounted:

```bash
docker run -it \
  --name nemo-automodel-lora \
  --gpus all \
  --network host \
  --shm-size=128g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_HOME=/shared/.cache/huggingface \
  -v "${PERSISTENT_HOST_PATH}:/shared" \
  "${AUTOMODEL_IMAGE}"
```

The rest of the guide runs inside this container shell unless a step says
otherwise. The container remains available if its shell exits; re-enter it
with:

```bash
docker start -ai nemo-automodel-lora
```

Inside the container, verify the GPU, persistent mount, AutoModel CLI, dataset
adapter, and LoRA support:

```bash
nvidia-smi -L
df -hT /shared

cd /opt/Automodel
command -v automodel
python3 -c 'import nemo_automodel; print(nemo_automodel.__version__)'
python3 -c 'from nemo_automodel.components.datasets.llm.agent_chat import make_agent_chat_dataset; print("Agent SFT adapter is available.")'
python3 -c 'from nemo_automodel.components._peft.lora import PeftConfig; print("LoRA support is available.")'
```

The B300 GPU must be visible, and `/shared` must resolve to the persistent host
directory. If either Python import fails, confirm that the container image tag
is `26.08`.

## 4. Authenticate and populate the model cache

Inside the container, authenticate to Hugging Face and download the model:

```bash
mkdir -p /shared/.cache/huggingface
export HF_HOME=/shared/.cache/huggingface
hf auth login

export MODEL_ID=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
hf download "${MODEL_ID}"
```

The Hugging Face token and model cache reside on persistent storage. Review
and accept any applicable model or dataset terms before downloading.

## 5. Prepare the agentic dataset

The `interactive_agent` JSONL file contains heterogeneous nested schemas that
cannot be loaded directly through Arrow. The converter below downloads the
raw JSONL, normalizes messages one record at a time, renders tool calls in the
Nemotron 3.5 Lightning chat format, and serializes the tool definitions to
avoid Arrow schema inference.

Create the data directory inside the container:

```bash
mkdir -p /shared/lora/data
cd /shared/lora/data
```

Create `/shared/lora/data/prepare.py` with the following contents:

```python
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

OUTPUT = Path("/shared/lora/data")
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
python3 /shared/lora/data/prepare.py
```

The source file is several gigabytes, so the download and conversion can take
some time. Verify the record counts:

```bash
wc -l /shared/lora/data/train.jsonl /shared/lora/data/validation.jsonl
```

Expected output for the pinned dataset revision:

```text
  278624 /shared/lora/data/train.jsonl
     256 /shared/lora/data/validation.jsonl
  278880 total
```

Validate the converted schema before training:

```bash
test "$(wc -l < /shared/lora/data/train.jsonl)" -eq 278624
test "$(wc -l < /shared/lora/data/validation.jsonl)" -eq 256

python3 - <<'PY'
import json
from pathlib import Path

for name in ("train.jsonl", "validation.jsonl"):
    path = Path("/shared/lora/data") / name
    with path.open(encoding="utf-8") as stream:
        row = json.loads(stream.readline())
    assert row["messages"], f"{name} has no messages"
    assert all("role" in message and "content" in message for message in row["messages"])
    assert isinstance(json.loads(row["tools"]), list)

print("Agentic LoRA JSONL files are ready.")
PY
```

If the SFT recipe already created the same pinned dataset under
`/shared/sft/data`, you can reuse it instead of converting it again. Update
both dataset paths in the LoRA configuration to `/shared/sft/data/...`.

## 6. Install the single-Station LoRA configuration

Create persistent configuration and checkpoint directories inside the
container:

```bash
mkdir -p /shared/lora/configs /shared/lora/checkpoints
```

Download the checked-in
[`lora_lightning35_station.yaml`](lora_lightning35_station.yaml)
configuration:

```bash
curl -fL \
  https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/usage-cookbook/Nemotron-3.5-Lightning/dgx-station-recipes/lora_lightning35_station.yaml \
  -o /shared/lora/configs/lora_lightning35_station.yaml
```

If you are working from a local clone of this repository instead, copy that
file to the same persistent destination.

The configuration has these important properties:

- The `peft` section enables rank-8 LoRA with alpha 32. It excludes
  `*.out_proj` because the Mamba layers pass that weight directly to custom
  kernels that LoRA cannot wrap.
- FSDP2 supplies the training strategy, with tensor, context, and expert
  parallel sizes all set to 1 for the single-GPU run.
- The single-GPU backend uses Transformer Engine attention and PyTorch linear,
  RMSNorm, MoE expert, and dispatcher implementations.
- `truncate_history: true` removes complete oldest exchanges to preserve the
  supervised final response within the 32,768-token limit.
- `train_on_last_turn_only: false` supervises assistant text and tool calls in
  every assistant turn. The adapter masks user and tool-response tokens.
- `drop_history_reasoning_content: true` removes hidden reasoning from earlier
  turns, while `mask_reasoning_content: false` trains on any reasoning content
  that remains after that history cleanup.
- `max_steps: 100` makes this a bounded demonstration. Because
  `ckpt_every_steps` is 1,000, the run saves its consolidated adapter at the
  end rather than producing periodic checkpoints.

Verify the input files and configuration:

```bash
test -s /shared/lora/data/train.jsonl
test -s /shared/lora/data/validation.jsonl
test -s /shared/lora/configs/lora_lightning35_station.yaml
test -w /shared/lora/checkpoints

python3 - <<'PY'
from pathlib import Path

import yaml

path = Path("/shared/lora/configs/lora_lightning35_station.yaml")
config = yaml.safe_load(path.read_text())
assert config["model"]["pretrained_model_name_or_path"].endswith("30B-A3B-BF16")
assert config["peft"]["dim"] == 8
assert config["peft"]["alpha"] == 32
assert config["peft"]["exclude_modules"] == ["*.out_proj"]
assert config["distributed"]["tp_size"] == 1
assert config["distributed"]["cp_size"] == 1
assert config["distributed"]["ep_size"] == 1
assert config["dataset"]["seq_length"] == 32768
print("Single-Station LoRA configuration is ready.")
PY
```

## 7. Launch LoRA training

No cross-node rendezvous or separate worker process is required. Run the
training command inside the container:

```bash
cd /opt/Automodel

automodel /shared/lora/configs/lora_lightning35_station.yaml \
  --nproc-per-node 1
```

`--nproc-per-node 1` explicitly launches one process for the Station's B300
GPU. AutoModel loads the frozen base model, attaches the LoRA modules, and
updates only the adapter parameters.

## 8. Monitor training and artifacts

From another host shell, inspect GPU activity while the run is active:

```bash
docker exec nemo-automodel-lora nvidia-smi
```

Training logs remain attached to the `automodel` terminal. Expect model and
dataset initialization, confirmation of the trainable adapter parameters, and
per-step loss output.

Inspect the persistent checkpoint directory as the run completes:

```bash
docker exec nemo-automodel-lora \
  bash -lc 'find /shared/lora/checkpoints -maxdepth 5 -type f | sort | tail -n 40'

docker exec nemo-automodel-lora df -h /shared
```

With the supplied 100-step configuration, expect a consolidated adapter at
the end of the run rather than an intermediate 1,000-step checkpoint. The
adapter output includes safetensors weights and adapter configuration metadata;
retain the exact base-model revision with it for inference or later merging.

## 9. Stop or clean up

To stop training early, press `Ctrl+C` in the `automodel` terminal. Preserve
the terminal output when diagnosing a failure.

Exit the container when training is no longer running:

```bash
exit
```

The retained container remains available for inspection or restart. Remove it
only when it is no longer needed:

```bash
docker rm nemo-automodel-lora
```

The model cache, data, configuration, and adapter checkpoints under the
persistent host path are not removed with the container.

## Troubleshooting

### AutoModel cannot import `agent_chat` or `PeftConfig`

Verify the container image in a host shell:

```bash
docker inspect nemo-automodel-lora --format '{{.Config.Image}}'
```

It must report `nvcr.io/nvidia/nemo-automodel:26.08`. Recreate a container
started from another release instead of modifying its installed package in
place.

### Dataset preparation fails or produces different counts

- Confirm `prepare.py` sets `DATASET_REVISION` to
  `4fb69cd40dbf36da60c73321e094e093946e60e9`.
- Confirm the persistent volume has enough free space for the raw file and
  both converted JSONL files.
- Inspect the last processed line and the Python exception before rerunning;
  the line-by-line converter intentionally bypasses the dataset's heterogeneous
  Arrow schema.

### CUDA runs out of memory

- Stop other GPU workloads and confirm the B300 GPU is visible in the
  container.
- Reduce `step_scheduler.local_batch_size` from 2 to 1 while leaving the
  global batch size at 8.
- If needed, reduce both dataset `seq_length` values from 32,768. This changes
  the training workload, so revalidate quality and throughput afterward.
- Keep tensor, context, and expert parallel sizes at 1 for this single-process
  recipe.

### Training tries to wrap a Mamba output projection

Confirm the configuration retains this exclusion:

```yaml
peft:
  exclude_modules: ["*.out_proj"]
```

Mamba layers use custom kernels that consume `out_proj.weight` directly, so
those projections are not compatible with this LoRA wrapping path.

### Checkpoints are missing after the run

- Confirm `/shared/lora/checkpoints` exists and is writable inside the
  container.
- Confirm the container was launched with the intended persistent host path
  mounted at `/shared`.
- Look for the final adapter below the checkpoint tree with
  `find /shared/lora/checkpoints -type f`.
- Remember that `ckpt_every_steps: 1000` does not create an intermediate
  checkpoint during the supplied 100-step run; `save_consolidated: final`
  writes the adapter when the run finishes normally.

## References

- [NeMo AutoModel container on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel)
- [NeMo AutoModel fine-tuning guide](https://docs.nvidia.com/nemo/automodel/latest/guides/llm/finetune.html)
- [NeMo AutoModel repository](https://github.com/NVIDIA-NeMo/Automodel)
- [Nemotron 3.5 Lightning model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
- [Nemotron-SFT-Agentic-v2 dataset card](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2)
- [DGX Station software stack](https://docs.nvidia.com/dgx/dgx-station-development-guide/porting/software-requirements.html)
- [DGX Station customization recipe index](README.md)
