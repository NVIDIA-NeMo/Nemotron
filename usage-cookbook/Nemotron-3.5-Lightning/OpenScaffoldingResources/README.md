# Nemotron 3.5 Lightning with Agentic Coding Tools

Nemotron 3.5 Lightning is a 30B total / 3B active-parameter mixture-of-experts
model built for fast reasoning, coding, and agentic workflows. This guide covers
config-based setup for **OpenCode**, **OpenClaw**, **Kilo Code CLI**,
**OpenHands CLI**, **Hermes Agent**, and **Pi**.

Use either of these access paths:

- [NVIDIA-hosted API](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b):
  `nvidia/nemotron-3.5-lightning-30b-a3b`
- Local vLLM: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`

## Why Nemotron 3.5 Lightning for Agentic Coding

Agentic coding tools need strong tool use, visible reasoning, long context, and
enough generation speed to keep multi-step workflows moving. Nemotron 3.5
Lightning packages those capabilities into a model sized for workstation-class
deployment.

**30B total / 3B active parameters.** LatentMoE routing activates a compact
expert subset for each token, combining broad model capacity with efficient
generation.

**Hybrid Mamba-2 and attention architecture.** Mamba-2 layers carry state
efficiently across long sequences, while attention layers preserve precise
reasoning and tool-use behavior.

**NVFP4 deployment.** The NVFP4 checkpoint brings the full target model onto a
single DGX Spark with room for KV cache and agent tooling.

**DSpark speculative decoding.** The companion DSpark checkpoint drafts
multiple future tokens so vLLM can accelerate generation while preserving the
target model's output distribution.

**Reasoning and native tool calls.** The model's chat template supports
thinking traces and structured tool calls used directly by modern coding-agent
harnesses.

**Up to 1M-token context.** Lightning is trained for long repositories, tool
traces, and extended agent sessions.

## Shared Setup

For tools using the NVIDIA-hosted API:

```bash
export NVIDIA_API_KEY="nvapi-..."
export NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1"
export NVIDIA_MODEL="nvidia/nemotron-3.5-lightning-30b-a3b"
```

For the local vLLM deployment:

```bash
export NEMOTRON_BASE_URL="http://127.0.0.1:8001/v1"
export NEMOTRON_API_KEY="local-vllm"
export NEMOTRON_MODEL="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
```

## Local vLLM Deployment on DGX Spark

This vLLM `v0.27.0` recipe serves the NVFP4 target with DSpark speculative
decoding on one DGX Spark. It combines Marlin NVFP4, FP8 KV cache, FlashInfer
Mamba kernels, prefix caching, and CUDA graphs with a 16,384-token batching
budget.

**Prerequisites**

- Docker with NVIDIA Container Toolkit support
- Hugging Face access to the target and DSpark checkpoints
- vLLM `v0.27.0`, or `vllm/vllm-openai:v0.27.0` for the container launcher

**Select the checkpoints**

```bash
hf auth login

export MODEL_CKPT=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
export DSPARK_CKPT=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark
```

**Run with vLLM**

```bash
vllm serve --model "$MODEL_CKPT" \
  --host 0.0.0.0 \
  --port 8001 \
  --served-model-name nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --api-key local-vllm \
  --moe-backend marlin \
  --kv-cache-dtype fp8 \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \
  --quantization modelopt_fp4 \
  --compilation_config.cudagraph_capture_sizes '[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 1024, 2048, 4096, 8192]' \
  --speculative_config.num_speculative_tokens 3 \
  --mamba-backend flashinfer \
  --mamba-cache-mode align \
  --reasoning-parser nemotron_v3 \
  --speculative_config.method dspark \
  --speculative_config.model "$DSPARK_CKPT" \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice
```

**Run with the container launcher**

```bash
./scripts/serve-local-vllm.sh
```

The launcher uses the same recipe with `vllm/vllm-openai:v0.27.0`, persists
the Hugging Face and vLLM caches, binds to `0.0.0.0:8001`, and holds the DGX
Spark runtime lock until the container exits. `MODEL_CKPT` and `DSPARK_CKPT`
also accept downloaded checkpoint directories. Use the current GA checkpoint
revisions to expose the full 1,048,576-token window.

**Verify the endpoint**

```bash
./scripts/verify-endpoint.py
```

The verifier checks the served model ID, incremental SSE reasoning, final
content, and a native function call with parsed JSON arguments.

## OpenCode

[OpenCode](https://opencode.ai) is an open-source terminal coding agent with
support for custom OpenAI-compatible providers.

**Install**

```bash
npm install -g opencode-ai
```

**Configure with the NVIDIA-hosted API**

```bash
export OPENCODE_CONFIG="$PWD/configs/opencode-nvidia.json"
opencode
```

The config registers `https://integrate.api.nvidia.com/v1` as an
OpenAI-compatible provider, reads `NVIDIA_API_KEY`, and selects
`nvidia/nemotron-3.5-lightning-30b-a3b`.

**Configure with local vLLM**

```bash
export NEMOTRON_BASE_URL="http://127.0.0.1:8001/v1"
export NEMOTRON_API_KEY="local-vllm"
export OPENCODE_CONFIG="$PWD/configs/opencode.json"
```

**Run**

```bash
cd /path/to/project
opencode
/init
```

For a headless run with thinking visible:

```bash
opencode run --thinking --format json --auto \
  "Inspect the project, make the requested change, and run its tests."
```

## OpenClaw

[OpenClaw](https://docs.openclaw.ai) is a persistent autonomous agent that can
run locally or as a daemon.

**Install**

OpenClaw requires Node.js 22.22.3 or newer.

```bash
npm install -g openclaw@latest
```

**Configure with the NVIDIA-hosted API**

```bash
mkdir -p ~/.openclaw
cp configs/openclaw-nvidia.json ~/.openclaw/openclaw.json
```

The included provider mapping uses the NVIDIA OpenAI-compatible endpoint and
selects `nvidia/nemotron-3.5-lightning-30b-a3b`.

**Configure with local vLLM**

```bash
mkdir -p ~/.openclaw
cp configs/openclaw.json ~/.openclaw/openclaw.json
```

For an existing profile, merge the selected `agents.defaults.model` and
provider mapping into the current configuration.

**Verify and run**

```bash
openclaw doctor
openclaw models list
openclaw tui
```

## Kilo Code CLI

[Kilo Code CLI](https://kilo.ai/docs/code-with-ai/platforms/cli) is a terminal
coding agent with custom OpenAI-compatible provider support.

**Install**

```bash
npm install -g @kilocode/cli
```

**Configure with the NVIDIA-hosted API**

```bash
export KILO_CONFIG="$PWD/configs/kilo-nvidia.json"
kilo
```

The config registers the NVIDIA endpoint, reads `NVIDIA_API_KEY`, and selects
Nemotron 3.5 Lightning. Use `/models` to verify the selection.

**Configure with local vLLM**

```bash
export KILO_CONFIG="$PWD/configs/kilo.json"
```

**Run**

```bash
cd /path/to/project
kilo
```

For a headless run:

```bash
kilo run --thinking --format json --auto \
  "Inspect the project, make the requested change, and run its tests."
```

## OpenHands CLI

[OpenHands CLI](https://openhands.dev/product/cli) uses LiteLLM-compatible
provider configuration and custom OpenAI-compatible endpoints.

**Install**

```bash
uv tool install openhands --python 3.12
```

**Configure with the NVIDIA-hosted API**

```bash
set -a
source configs/openhands-nvidia.env
set +a
```

**Configure with local vLLM**

```bash
set -a
source configs/openhands.env
set +a
```

**Run**

```bash
openhands --override-with-envs
```

For automation:

```bash
openhands --override-with-envs --headless --json \
  --task "Inspect the project, make the requested change, and run its tests."
```

## Hermes Agent

[Hermes Agent](https://hermes-agent.nousresearch.com/docs/integrations/) is a
terminal-native autonomous agent with persistent memory and multi-provider
model routing, including native NVIDIA Build support.

**Configure with the NVIDIA-hosted API**

```bash
hermes config set NVIDIA_API_KEY "$NVIDIA_API_KEY"
mkdir -p ~/.hermes
cp configs/hermes-nvidia.yaml ~/.hermes/config.yaml
```

**Configure with local vLLM**

```bash
mkdir -p ~/.hermes
cp configs/hermes-config.yaml ~/.hermes/config.yaml
```

**Run**

```bash
cd /path/to/project
hermes
```

## Pi

[Pi](https://pi.dev) is a terminal coding agent with provider and model defaults
stored under `~/.pi/agent`.

**Install**

```bash
npm install -g @mariozechner/pi-coding-agent
```

**Configure with the NVIDIA-hosted API**

```bash
mkdir -p ~/.pi/agent
cp configs/pi-models-nvidia.json ~/.pi/agent/models.json
cp configs/pi-settings-nvidia.json ~/.pi/agent/settings.json
```

Pi reads `NVIDIA_API_KEY` from the environment.

**Configure with local vLLM**

```bash
mkdir -p ~/.pi/agent
cp configs/pi-models.json ~/.pi/agent/models.json
cp configs/pi-settings.json ~/.pi/agent/settings.json
```

**Run**

```bash
cd /path/to/project
pi --thinking high
```

For a non-interactive NVIDIA-hosted run:

```bash
pi --provider nvidia --model "$NVIDIA_MODEL" --thinking high --print \
  "Inspect this project and report what its tests do."
```

## Context and Generation Settings

Lightning supports a 1,048,576-token context. The v0.27.0 server reads that
limit from the checkpoint. The local OpenCode, Kilo, OpenClaw, and Pi configs
use the same context with a 32K output window; Hermes declares the full context.
NVIDIA-hosted configs with explicit model metadata mirror those values.

The model's generation defaults are `temperature=1.0` and `top_p=0.95`.
Thinking is enabled by the chat template and surfaced by each harness.

## License

The model weights are governed by the
[OpenMDW License Agreement, version 1.1](https://openmdw.ai/license/1-1/).
The scripts and documentation in this repository are Apache-2.0 licensed.
