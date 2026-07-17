# Deploying Nemotron 3 Ultra on a 2x DGX Station Cluster

A step-by-step guide to serving **NVIDIA Nemotron 3 Ultra** across two [DGX Stations](https://www.nvidia.com/en-us/products/workstations/dgx-station/) with [vLLM](https://docs.vllm.ai/) and Ray, then benchmarking the deployment with [NVIDIA AIPerf](https://github.com/ai-dynamo/aiperf).

**Jump to:**

- [Prerequisites](#prerequisites)
- [Prepare the Stations](#prepare-the-stations)
- [Launch vLLM](#launch-vllm)
- [Verify the Deployment](#verify-the-deployment)
- [Benchmark with AIPerf](#benchmark-with-aiperf)
- [Quick Troubleshooting](#quick-troubleshooting)
- [Stop the Deployment](#stop-the-deployment)

---

## What you're deploying

**Nemotron 3 Ultra**
(`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`) is a hybrid Mamba-Transformer Mixture-of-Experts (MoE) model with 550 billion total parameters and approximately 55 billion active parameters per token.

This guide uses:
- **Two GB300 GPUs** -- one GPU in each DGX Station.
- **Pipeline parallelism** -- vLLM uses tensor parallel size 1 and pipeline parallel size 2, placing one model stage on each Station.
- **Ray** -- the head container creates the Ray cluster and starts vLLM after the worker GPU joins.
- **Two CX8 RoCE rails** -- NCCL and UCX can use both validated high-speed links between the Stations.
- **256K context** -- the server uses a maximum model length of 262,144 tokens.
- **Agentic output parsing** -- vLLM parses Nemotron reasoning and OpenAI-compatible tool calls.
The resulting OpenAI-compatible API is served by `station-1` on port 8000.

Model card:
[nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4)

---

## Prerequisites

Before anything below will work,  2 DGX Stations need to be networked together. Follow [NVIDIA's Connect Two DGX Stations for Distributed Workloads](https://build.nvidia.com/station/connect-two-stations/instructions) setup guide first — this is not optional, and the later steps assume it's done:

You also need:

- Two GB300-based DGX Stations with current NVIDIA drivers and firmware.
- SSH access between the Stations. The examples use `station-1` and `station-2` as SSH host aliases.
- Docker and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on both Stations.
- Hugging Face access to the gated Nemotron 3 Ultra NVFP4 model.
- `hf`, `rsync`, `jq`, `ibdev2netdev`, and `show_gids` available on the hosts.
- Enough 350GB of storage on both Stations for a complete copy of the model cache.
- CX8 Interface should be enabled on both Stations.

Throughout this guide:

| Term | Meaning |
| :--- | :--- |
| **Head** | `station-1`: Ray head, vLLM API, and pipeline stage 0. |
| **Worker** | `station-2`: Ray worker and pipeline stage 1. |
| **Head IP** | The IPv4 address on `station-1`'s `mlx5_0` interface. |
| **Worker IP** | The IPv4 address on `station-2`'s `mlx5_0` interface. |

The examples in this guide use the following direct-attach network. Use your configured addresses if they differ.

| Rail | `station-1` | `station-2` | HCA |
| --- | --- | --- | --- |
| 0 | `192.168.240.1/30` | `192.168.240.2/30` | `mlx5_0` |
| 1 | `192.168.240.5/30` | `192.168.240.6/30` | `mlx5_1` |

---

## Prepare the Stations

### 1. Verify the CX8 fabric

Run these commands on **both Stations**:

```shell
ibdev2netdev
ip -br link show
ip -br address show
show_gids
```

Confirm that `mlx5_0` and `mlx5_1` map to active Ethernet interfaces and that both interfaces have the expected IPv4 addresses and MTU.

From `station-1`, verify both peer addresses with jumbo packets:

```shell
ping -c 4 -M do -s 8972 192.168.240.2
ping -c 4 -M do -s 8972 192.168.240.6
```

Do not continue until both pings and the prerequisite RDMA/NCCL tests pass.

### 2. Verify Docker and the GB300 GPU

Run on **both Stations**:

```shell
export IMAGE="vllm/vllm-openai:v0.25.1-aarch64"

nvidia-smi -L
free -h
df -h "${HOME}"

docker pull "${IMAGE}"
docker run --rm --gpus all --entrypoint nvidia-smi "${IMAGE}" -L
```

The container must list the GB300 GPU. If a Station also contains another GPU,
the launch steps below select the GB300 by UUID instead of using `--gpus all`.

### 3. Download and distribute the model cache

Run these on the **head node** (`station-1`).

Install [uv](https://docs.astral.sh/uv/), the fast Python package manager:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Download the model from Hugging Face:

>[!NOTE]

```shell
ssh station-1

export MODEL="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
export HF_HOME="${HOME}/.cache/huggingface"
uvx hf download "${MODEL}"
```

>[!NOTE]
> This step takes more than an hour depending on the internet speed so Grab a coffee. Once it's on the head node, you can copy it to the other node rather than re-downloading. And also requires minimum storage of 350GB on both stations.


Copy the model to the other node (`station-2`):

```shell
rsync -aH --info=progress2  "${HOME}/.cache/huggingface/" station-2:.cache/huggingface/
```

Verify the model snapshot from `station-1` on both hosts:

```shell
MODEL_CACHE="${HOME}/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"

test -d "${MODEL_CACHE}" && echo "station-1 cache: ready"
ssh station-2 \
  'test -d ~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 && echo "station-2 cache: ready"'
```

---

## Launch vLLM

Start the head container first. It creates the Ray head and waits for the worker GPU to join before it launches vLLM. Start the worker immediately after the head container is running.

### 1. Start the head on station-1

Connect to `station-1` and discover the deployment values:

```shell
ssh station-1

export HEAD_IFACE=$(ibdev2netdev | awk '$1 == "mlx5_0" {print $5; exit}')
export HEAD_IP=$(ip -4 -o address show dev "${HEAD_IFACE}" \
  | awk '{split($4, address, "/"); print address[1]; exit}')
export GPU_UUID_HEAD=$(nvidia-smi \
  --query-gpu=name,uuid --format=csv,noheader \
  | awk -F', ' '/GB300|B300/ {print $2; exit}')
export IMAGE="vllm/vllm-openai:v0.25.1-aarch64"
export MODEL="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
export SERVED_MODEL="nemotron-ultra"

printf 'HEAD_IFACE=%s\nHEAD_IP=%s\nGPU_UUID_HEAD=%s\n' \
  "${HEAD_IFACE}" "${HEAD_IP}" "${GPU_UUID_HEAD}"
```

Review the printed values, then start the head container:

```shell
sudo docker rm -f nemotron-ultra-head 2>/dev/null || true

sudo docker run -d --name nemotron-ultra-head \
  --restart unless-stopped --init \
  --network host --shm-size 16g \
  --gpus "device=${GPU_UUID_HEAD}" \
  --device=/dev/infiniband/uverbs0 \
  --device=/dev/infiniband/uverbs1 \
  --ulimit memlock=-1 \
  -e HEAD_IP="${HEAD_IP}" \
  -e MODEL="${MODEL}" \
  -e SERVED_MODEL="${SERVED_MODEL}" \
  -e VLLM_HOST_IP="${HEAD_IP}" \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e NCCL_IB_HCA=mlx5_0,mlx5_1 \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ROCE_VERSION_NUM=2 \
  -e NCCL_IB_TC=106 \
  -e NCCL_NET_GDR_LEVEL=PHB \
  -e NCCL_SOCKET_IFNAME="${HEAD_IFACE}" \
  -e GLOO_SOCKET_IFNAME="${HEAD_IFACE}" \
  -e TP_SOCKET_IFNAME="${HEAD_IFACE}" \
  -e NCCL_IB_QPS_PER_CONNECTION=4 \
  -e NCCL_IB_PCI_RELAXED_ORDERING=1 \
  -e UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1 \
  -e HF_HOME=/models/huggingface \
  -v "${HOME}/.cache/huggingface:/models/huggingface" \
  --entrypoint bash "${IMAGE}" -lc '
    set -euo pipefail
    python3 -m pip install --break-system-packages "ray[cgraph]"
    ray start --head --node-ip-address="${HEAD_IP}" --port=6379 --num-gpus=1

    python3 - <<"PY"
import time
import ray

ray.init(address="auto")
deadline = time.time() + 3600
while ray.cluster_resources().get("GPU", 0) < 2:
    if time.time() >= deadline:
        raise TimeoutError("station-2 GPU did not join Ray within 3600 seconds")
    time.sleep(5)
print(ray.cluster_resources())
PY

    exec vllm serve "${MODEL}" \
      --served-model-name "${SERVED_MODEL}" \
      --host 0.0.0.0 --port 8000 \
      --trust-remote-code \
      --tensor-parallel-size 1 \
      --pipeline-parallel-size 2 \
      --distributed-executor-backend ray \
      --kv-cache-dtype fp8 \
      --max-model-len 262144 \
      --gpu-memory-utilization 0.9 \
      --max-num-seqs 256 \
      --distributed-timeout-seconds 7200 \
      --enable-prefix-caching
  '
```

>[!NOTE]
> If you have plan to use **NVIDIA Nemotron 3 Ultra** with [NeMoClaw](https://build.nvidia.com/station/nemoclaw/instructions) add `--enable-auto-tool-choice` and `--tool-call-parser qwen3_coder` and `--reasoning-parser nemotron_v3` to vll serve command.
>
### 2. Start the worker on station-2

>[!NOTE]
> HEAD_IP value should be fetched from station-1 [Step 3](#3-download-and-distribute-the-model-cache) and should be set to the value printed on `station-1`.

Connect to `station-2` and discover its values:

```shell
ssh station-2

export WORKER_IFACE=$(ibdev2netdev | awk '$1 == "mlx5_0" {print $5; exit}')
export WORKER_IP=$(ip -4 -o address show dev "${WORKER_IFACE}" \
  | awk '{split($4, address, "/"); print address[1]; exit}')
export GPU_UUID_WORKER=$(nvidia-smi \
  --query-gpu=name,uuid --format=csv,noheader \
  | awk -F', ' '/GB300|B300/ {print $2; exit}')
export HEAD_IP="" # HEAD_IP from station-1
export IMAGE="vllm/vllm-openai:v0.25.1-aarch64"

printf 'WORKER_IFACE=%s\nWORKER_IP=%s\nGPU_UUID_WORKER=%s\nHEAD_IP=%s\n' \
  "${WORKER_IFACE}" "${WORKER_IP}" "${GPU_UUID_WORKER}" "${HEAD_IP}"
```

Start the worker:

```shell
sudo docker rm -f nemotron-ultra-worker 2>/dev/null || true

sudo docker run -d --name nemotron-ultra-worker \
  --restart unless-stopped --init \
  --network host --shm-size 16g \
  --gpus "device=${GPU_UUID_WORKER}" \
  --device=/dev/infiniband/uverbs0 \
  --device=/dev/infiniband/uverbs1 \
  --ulimit memlock=-1 \
  -e HEAD_IP="${HEAD_IP}" \
  -e WORKER_IP="${WORKER_IP}" \
  -e VLLM_HOST_IP="${WORKER_IP}" \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e NCCL_IB_HCA=mlx5_0,mlx5_1 \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ROCE_VERSION_NUM=2 \
  -e NCCL_IB_TC=106 \
  -e NCCL_NET_GDR_LEVEL=PHB \
  -e NCCL_SOCKET_IFNAME="${WORKER_IFACE}" \
  -e GLOO_SOCKET_IFNAME="${WORKER_IFACE}" \
  -e TP_SOCKET_IFNAME="${WORKER_IFACE}" \
  -e NCCL_IB_QPS_PER_CONNECTION=4 \
  -e NCCL_IB_PCI_RELAXED_ORDERING=1 \
  -e UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1 \
  -e HF_HOME=/models/huggingface \
  -v "${HOME}/.cache/huggingface:/models/huggingface" \
  --entrypoint bash "${IMAGE}" -lc '
    set -euo pipefail
    python3 -m pip install --break-system-packages "ray[cgraph]"
    exec ray start --address="${HEAD_IP}:6379" \
      --node-ip-address="${WORKER_IP}" --num-gpus=1 --block
  '
```

### Configuration notes

| Setting | Purpose |
| :--- | :--- |
| `vllm/vllm-openai:v0.25.1-aarch64` | Uses the same Arm64 vLLM runtime on both Stations. |
| `--tensor-parallel-size 1` | Keeps each pipeline stage local to one GPU. |
| `--pipeline-parallel-size 2` | Splits the model into two stages, one on each Station. |
| `--distributed-executor-backend ray` | Uses the two-node Ray cluster to place distributed workers. |
| `--max-model-len 262144` | Enables a maximum context length of 262,144 tokens. |
| `--kv-cache-dtype fp8` | Reduces KV-cache memory consumption. |
| `--enable-prefix-caching` | Reuses KV-cache blocks for requests with shared prompt prefixes. |
| `--tool-call-parser qwen3_coder` | Parses OpenAI-compatible tool calls from model output. |
| `--reasoning-parser nemotron_v3` | Separates Nemotron 3 reasoning from the final response. |
| `NCCL_IB_HCA=mlx5_0,mlx5_1` | Makes both validated CX8 rails available to NCCL. |
| `NCCL_SOCKET_IFNAME` | Selects the rail-0 interface for bootstrap and socket traffic. |
| `UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1` | Makes both CX8 devices available to UCX. |

---

## Verify the Deployment

### 1. Monitor startup

Use one terminal for each Station:

```shell
# station-1
sudo docker logs -f nemotron-ultra-head
```

```shell
# station-2
sudo docker logs -f nemotron-ultra-worker
```

The first startup downloads or resolves model files, loads weights, compiles kernels, builds the KV cache, autotunes kernels, and captures CUDA graphs. The containers can remain `Up` while port 8000 is not yet listening. Do not restart them while logs continue to advance.

Wait for this message in the head log:

```text
Application startup complete.
```

### 2. Check containers, Ray, and the model endpoint

Run from `station-1`:

```shell
test "$(sudo docker inspect --format '{{.State.Status}}' nemotron-ultra-head)" = running
test "$(ssh station-2 sudo docker inspect --format '{{.State.Status}}' nemotron-ultra-worker)" = running

sudo docker exec nemotron-ultra-head \
  ray status --address="${HEAD_IP}:6379"

curl -fsS http://127.0.0.1:8000/v1/models \
  | jq -e '.data[] | select(.id == "nemotron-ultra")'
```

Ray must report two GPUs and the model request must return `nemotron-ultra`.

### 3. Send a chat request

```shell
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nemotron-ultra",
    "messages": [
      {"role": "user", "content": "Reply with exactly: READY"}
    ],
    "max_tokens": 16,
    "temperature": 0,
    "stream": false
  }' | jq
```

### 4. Verify reasoning and tool-call parsing

This request asks the model to select a function. It does not execute the
function; it verifies that vLLM returns a structured `tool_calls` response.

```shell
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nemotron-ultra",
    "messages": [
      {"role": "user", "content": "Use the weather tool for Seattle."}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather for a city.",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string"}
            },
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "max_tokens": 256,
    "temperature": 0
  }' | jq '.choices[0].message | {
    reasoning: (.reasoning_content // .reasoning),
    tool_calls,
    content
  }'
```

---

## Benchmark with AIPerf

Run AIPerf on `station-1` only after all readiness checks pass. The workload
below uses synthetic prompts with a shared 32K-token prefix, a fixed 2,048-token
input sequence length (ISL), and a fixed 1,024-token output sequence length
(OSL). It uses streaming chat requests at concurrency levels 1, 4, 8, and 16.

```shell
export MODEL="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
export ENDPOINT="http://127.0.0.1:8000"
export RESULT_ROOT="${HOME}/aiperf-results/station-32k-prefix-2k1k-$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "${RESULT_ROOT}"

for BS in 1 4 8 16; do
  WARMUP=${BS}
  REQUESTS=$((BS * 2))
  if [ "${REQUESTS}" -lt 10 ]; then REQUESTS=10; fi

  uvx aiperf profile \
    --model nemotron-ultra \
    --endpoint-type chat \
    --streaming \
    --url "${ENDPOINT}" \
    --tokenizer "${MODEL}" \
    --tokenizer-trust-remote-code \
    --isl 2048 --isl-stddev 0 \
    --osl 1024 --osl-stddev 0 \
    --extra-inputs 'min_tokens:1024,ignore_eos:true' \
    --num-prefix-prompts 1 \
    --prefix-prompt-length 32768 \
    --warmup-request-count "${WARMUP}" \
    --concurrency "${BS}" \
    --request-count "${REQUESTS}" \
    --request-timeout-seconds 1200 \
    --wait-for-model-timeout 120 \
    --wait-for-model-mode models \
    --use-server-token-count \
    --output-artifact-dir "${RESULT_ROOT}/bs${BS}"
done
```

### Measured performance

The following single measured benchmark set uses the PP=2, TP=1 deployment and the fixed synthetic 32K shared-prefix, 2K ISL, and 1K OSL workload above. Each
row reports AIPerf's average metrics for one concurrency level. These are point-in-time results for this two-Station configuration, not guaranteed performance. Throughput and latency vary with software versions, firmware, power state, thermals, and server settings.

| Concurrency (BS) | Output throughput (tok/s) | Per-user throughput (tok/s) | Average TTFT (ms) | Average ITL (ms) | Average request latency (ms) | Request throughput (req/s) | Duration (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 82.22 | 84.96 | 409.65 | 11.77 | 12,450.51 | 0.08 | 74.73 |
| 4 | 231.61 | 60.77 | 830.87 | 16.47 | 17,679.35 | 0.23 | 106.11 |
| 8 | 343.51 | 45.80 | 1,414.05 | 21.92 | 23,836.38 | 0.34 | 143.09 |
| 16 | 500.10 | 33.33 | 1,983.82 | 30.07 | 32,748.88 | 0.49 | 196.57 |

The measurements show the expected throughput/latency tradeoff: aggregate output throughput increases with concurrency, while TTFT, inter-token latency, and end-to-end request latency also increase. Keep the workload, image, model revision, parallelism, and warmup policy fixed when comparing future runs.


| Metric | Meaning |
| :--- | :--- |
| **Output throughput** | Aggregate output tokens generated per second. |
| **Per-user throughput** | Average generation rate observed by an individual request. |
| **TTFT** | Time to first output token. |
| **ITL** | Average time between output tokens after generation begins. |
| **Request latency** | Average end-to-end time for a complete request. |
| **Request throughput** | Completed requests per second. |
| **Duration** | Total elapsed time for the benchmark stage. |

---

## Quick Troubleshooting

| Symptom | What to check |
| :--- | :--- |
| Worker does not join Ray | Confirm `HEAD_IP`, peer ping, TCP 6379 reachability, and that both containers installed compatible Ray packages. |
| Ray reports only one GPU | Check the worker container, the selected GB300 UUID, and Docker GPU access on `station-2`. |
| NCCL reports a network or system error | Re-run both-rail MTU, GID, RDMA, and NCCL validation. Confirm both containers use the same HCA list and RoCE settings. |
| Containers are `Up`, but port 8000 is closed | Continue monitoring logs while weights, kernels, KV cache, autotuning, or CUDA graphs are progressing. |
| Model access is denied | Accept the model terms, run `hf auth whoami`, and verify the complete cache exists on both Stations. |
| Tool calls appear as plain text | Confirm `--enable-auto-tool-choice`, `--tool-call-parser qwen3_coder`, and Chat Completions are in use. |
| AIPerf fails and later requests return HTTP 500 | Stop the benchmark, save both container logs, check container restart counts, and verify a short non-streaming request before retrying. |

Useful diagnostics:

```shell
# station-1
sudo docker logs --tail 200 nemotron-ultra-head
sudo docker inspect --format 'status={{.State.Status}} restarts={{.RestartCount}}' \
  nemotron-ultra-head
sudo docker exec nemotron-ultra-head \
  ray status --address=192.168.240.1:6379
ss -ltnp | grep ':8000' || true
```

```shell
# station-2
sudo docker logs --tail 200 nemotron-ultra-worker
sudo docker inspect --format 'status={{.State.Status}} restarts={{.RestartCount}}' \
  nemotron-ultra-worker
nvidia-smi
```

Because the head uses host networking and listens on `0.0.0.0:8000`, restrict
the vLLM API and Ray control-plane ports to the intended management/client
network. Do not expose this unauthenticated endpoint directly to an untrusted
network.

---

## Stop the Deployment

Stop the head first, then the worker:

```shell
# station-1
sudo docker rm -f nemotron-ultra-head
```

```shell
# station-2
sudo docker rm -f nemotron-ultra-worker
```

Keep both Hugging Face caches unless reclaiming disk is intentional. Retaining
the cache avoids downloading the model again during the next deployment.

---

## Related resources

- [Single DGX Station deployment](../StationDeploymentGuide/README.md)
- [Four-node DGX Spark deployment](../SparkDeploymentGuide/README.md)
- [DGX Station Development Guide](https://docs.nvidia.com/dgx/dgx-station-development-guide/Intro.html)
- [NVIDIA AIPerf documentation](https://docs.nvidia.com/aiperf/)
- [NVIDIA Nemotron repository](https://github.com/NVIDIA-NeMo/Nemotron)
