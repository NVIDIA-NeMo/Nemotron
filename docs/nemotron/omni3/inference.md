# Nemotron 3 Nano Omni — Inference & Deployment

How to deploy and serve Nemotron 3 Nano Omni after training (or against the public release checkpoint). For training instructions see the [recipe overview](./README.md); for architectural background see [`architecture.md`](./architecture.md).

## TL;DR

| If you want… | Use |
|---|---|
| Drop-in agent endpoint | NIM at [`build.nvidia.com`](https://build.nvidia.com) |
| Production high-throughput serving | TensorRT-LLM (Hopper / Blackwell) with NVFP4 |
| Continuous batching + streaming | vLLM |
| Multi-agent + tool-calling, lightweight | SGLang |
| Disaggregated serving with intelligent routing | Dynamo |
| Local laptop inference | Ollama or `llama.cpp` (GGUF) |
| Local desktop UI | LM Studio |
| Managed cloud | Amazon SageMaker JumpStart, Oracle Cloud, Microsoft Azure (coming soon), Dell on-premises |
| Privacy-first video processing | NemoClaw sandbox (sandboxed video analysis with policy-bounded output) |

## Inference engines

| Engine | Strengths | Hardware | Quantization |
|---|---|---|---|
| **vLLM** | Continuous batching, streaming, broad ecosystem | Ampere / Hopper / Blackwell | FP8, NVFP4 |
| **SGLang** | Lightweight, strong multi-agent + tool-calling story | Ampere / Hopper / Blackwell | FP8, NVFP4 |
| **NVIDIA TensorRT-LLM** | Lowest-latency production serving, latent MoE kernels | Hopper / Blackwell | FP8, NVFP4 |
| **Dynamo** | Disaggregated serving, intelligent routing, multi-tier KV caching | Hopper / Blackwell | FP8, NVFP4 |

Pyxis-mounting the trained `omni3-sft.sqsh` from this repo's training pipeline produces a container compatible with all four engines — see the engine documentation for serving entry points. For the open release checkpoint, pull `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` from Hugging Face directly and let the engine handle weight loading + (optional) on-the-fly quantization.

## Quantization

The model supports two quantization paths out of the box:

- **FP8** — works on Hopper and Blackwell; ~2× memory reduction over BF16 with minimal quality loss for this model class
- **NVFP4** — Blackwell-only; produces the highest reported throughput among open omnimodal models per the release blog

Quantization is applied at the inference-engine level, not as separately published HF checkpoints. All four engines above support both formats. For Blackwell + NVFP4 specifically, the release blog reports the highest throughput of any open omnimodal model in this class.

## Hardware

| Family | Supported | Notes |
|---|---|---|
| Ampere (A100, A40, …) | ✅ | BF16 / FP8; older GPUs lack NVFP4 hardware |
| Hopper (H100, H200) | ✅ | All paths supported; FP8 most efficient |
| Blackwell (B100, B200, GB200) | ✅ | NVFP4 unlocks peak throughput |

## Cloud platforms

| Platform | Status | Path |
|---|---|---|
| Amazon SageMaker JumpStart | Available | Search for `Nemotron-3-Nano-Omni` in JumpStart catalog |
| Oracle Cloud | Available | OCI AI services |
| Microsoft Azure | Coming soon (per release blog) | Azure AI catalog |
| Dell Technologies | Available (on-premises / hybrid) | Dell AI Factory |

## Inference providers

The model is hosted by:

- **Baseten**, **Clarifai**, **DeepInfra**, **Fireworks AI**, **Together AI**, **Vultr**, **Bitdeer**, **Crusoe**, **DigitalOcean**, **Lightning AI**, **Nebius**

These provide managed endpoints if you don't want to run inference yourself.

## Local / edge

| Format | Tool | Notes |
|---|---|---|
| GGUF | **`llama.cpp`** | Quantized weights for CPU / Apple Silicon / consumer GPU |
| GGUF | **Ollama** | Wraps `llama.cpp`; one-line model pull + chat |
| Native | **LM Studio** | Desktop UI for local inference |
| Inference Snaps | NVIDIA Inference Snaps | Pre-packaged endpoints for edge deployment |

These are best for local development, offline analysis, or edge inference where latency and privacy matter more than peak throughput.

## NemoClaw — privacy-first video processing

For workloads where video frames can't leave a sandboxed environment (compliance, healthcare, classified content), the model integrates with **NemoClaw**, NVIDIA's sandboxed video-processing layer. The sandbox runs the perception model, then exports only policy-bounded outputs (transcripts, summaries, structured tags) — raw frames stay inside.

This is a distinguishing capability for the model's "perception sub-agent" framing: a downstream agent can invoke the omni model on private video content and get usable summaries back without ever handling raw frames itself.

## Tuning notes

- **Per-user interactivity**: the release blog reports throughput numbers at a fixed per-user tokens-per-second floor; aggregate throughput scales by holding user-level interactivity constant and adding concurrency.
- **EVS configuration**: for video-heavy workloads, the EVS layer's frame-compression ratio is the most important throughput knob (see [`architecture.md` §Video pipeline](./architecture.md#video-pipeline--3d-convolutions--efficient-video-sampling-evs) for the why). Lower compression → higher quality, lower throughput; tune per workload.
- **MoE expert dispatch**: the 30B-A3B router is sensitive to batch composition. For peak throughput, group requests with similar modality mix.

## Deployment from this recipe's training output

The training pipeline in this repo produces `${build_cache_dir}/containers/omni3-sft.sqsh`, a squashfs container that pyxis mounts directly. To use it for inference:

```bash
# Verify the squashfs is in place
ls -lh ${build_cache_dir}/containers/omni3-sft.sqsh

# Submit an inference job (example with vLLM)
srun --container-image=${build_cache_dir}/containers/omni3-sft.sqsh \
     --container-mounts=/lustre:/lustre,/path/to/checkpoint:/checkpoint \
     bash -lc 'vllm serve /checkpoint --tensor-parallel-size 1 --gpu-memory-utilization 0.9'
```

For benchmark evaluation against a deployed checkpoint, run `nemotron omni3 model eval` (a dedicated `nemotron omni3 eval` stage will land in a follow-up release).

## References

- **[Release blog](https://developer.nvidia.com/blog/nvidia-nemotron-3-nano-omni-powers-multimodal-agent-reasoning-in-a-single-efficient-open-model/)** — canonical source for the throughput claims, provider list, and deployment paths
- **[NIM at build.nvidia.com](https://build.nvidia.com)**
- **[Model weights](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)**
- [`architecture.md`](./architecture.md) — the *why* behind the inference characteristics
