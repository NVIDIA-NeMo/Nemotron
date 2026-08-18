# Deploying Nemotron 3 Ultra on DGX Station

This guide serves **NVIDIA Nemotron 3 Ultra** on a single GB300-based
[DGX Station](https://www.nvidia.com/en-us/products/workstations/dgx-station/)
with [vLLM](https://docs.vllm.ai/) 0.22.0. The recommended container image is
`vllm/vllm-openai:v0.22.0`. The result is an OpenAI-compatible API on
port 8000.

Unlike the [four-node DGX Spark deployment](../SparkDeploymentGuide/README.md),
this configuration uses one GPU and offloads selected Mixture-of-Experts (MoE)
weights into the DGX Station's coherent CPU memory.

## Prerequisites

- A GB300-based DGX Station running its current NVIDIA AI software stack. The
  system combines GPU HBM and CPU LPDDR5X in a coherent memory space over
  NVLink-C2C; see the
  [DGX Station Development Guide](https://docs.nvidia.com/dgx/dgx-station-development-guide/Intro.html).
- Docker with the NVIDIA Container Toolkit configured for GPU access.
- The recommended vLLM 0.22.0 image, `vllm/vllm-openai:v0.22.0`. It
  supports `--cpu-offload-params`, `--kernel_config`, and the
  `flashinfer-trtllm` NVFP4 GEMM backend used below.
- Access to
  [`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4)
  on Hugging Face and enough local storage for the model weights.

If the model download requires authentication, log in on the host before
starting the container. The launch command mounts this Hugging Face cache:

```shell
hf auth login
```

## Launch vLLM

Run the following commands on the DGX Station:

```shell
export IMAGE="vllm/vllm-openai:v0.22.0"
export HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p "$HF_CACHE_DIR"
docker pull "$IMAGE"

docker run --rm --name nemotron-ultra-vllm \
  --gpus all \
  --ipc=host \
  --network=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  -e VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1 \
  -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-trtllm \
  "$IMAGE" \
  nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 \
  --served-model-name nemotron-ultra \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --speculative-config '{"method": "nemotron_h_mtp", "num_speculative_tokens": 3}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --cpu-offload-gb 150 \
  --cpu-offload-params experts \
  --kernel_config '{"enable_flashinfer_autotune": false}' \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.9
```

The image entrypoint invokes `vllm serve`; the model ID and flags after
`"$IMAGE"` are passed directly to vLLM.

The initial model download and load can take several minutes. Keep this process
running while clients use the API.

### Configuration notes

| Setting | Purpose |
| :--- | :--- |
| `vllm/vllm-openai:v0.22.0` | Pins the recommended vLLM runtime and its dependencies. |
| `-v "$HF_CACHE_DIR:/root/.cache/huggingface"` | Reuses the host's model cache and Hugging Face authentication inside the container. |
| `VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1` | Avoids consuming GPU memory for pinned offload buffers on the coherent-memory system. |
| `VLLM_NVFP4_GEMM_BACKEND=flashinfer-trtllm` | Selects the FlashInfer TensorRT-LLM backend for NVFP4 matrix multiplication. |
| `--tensor-parallel-size 1` | Runs the model on the DGX Station's single integrated Blackwell Ultra GPU. |
| `--reasoning-parser nemotron_v3` | Parses Nemotron 3's reasoning output format. |
| `--enable-prefix-caching` | Reuses KV-cache blocks for requests that share a common prompt prefix. |
| `--enable-chunked-prefill` | Splits large prompt prefills into chunks so they can be scheduled alongside decode requests. |
| `--max-num-batched-tokens 8192` | Caps the total number of tokens processed in a scheduler iteration at 8,192. |
| `--cpu-offload-gb 150` | Makes up to 150 GiB of CPU memory available for weight offloading. |
| `--cpu-offload-params experts` | Restricts offloading to parameters whose name contains the exact `experts` segment. |
| `--kernel_config ...` | Disables FlashInfer autotuning for this launch configuration. |
| `--max-num-seqs 256` | Allows the scheduler to batch up to 256 sequences, subject to available cache memory. |
| `--gpu-memory-utilization 0.9` | Lets vLLM use up to 90 percent of GPU memory for model execution and cache. |

## Verify the API

In another terminal, confirm that the served model is available:

```shell
curl http://localhost:8000/v1/models
```

Then send a chat completion request:

```shell
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nemotron-ultra",
    "messages": [
      {"role": "user", "content": "Explain why selective expert offloading works well for a mixture-of-experts model."}
    ],
    "max_tokens": 256
  }'
```

Clients should use `nemotron-ultra` as the model name and
`http://<station-host>:8000/v1` as the OpenAI-compatible base URL. Restrict
network access to the service if it is reachable beyond the local machine.

## Troubleshooting

- **vLLM rejects an option or backend:** update the DGX Station software stack
  and use the recommended `vllm/vllm-openai:v0.22.0` image, which supports the
  offload and kernel settings used above.
- **The model download is denied:** accept the model terms on Hugging Face and
  authenticate with `hf auth login`.
- **The model runs out of memory while loading:** stop other GPU workloads and
  confirm that both vLLM environment variables are passed with `-e` as shown.
  If necessary, increase `--cpu-offload-gb` within available
  system memory or reduce `--gpu-memory-utilization`.
- **The API reports an unknown model:** send `nemotron-ultra`, the value passed
  to `--served-model-name`, rather than the Hugging Face repository name.
