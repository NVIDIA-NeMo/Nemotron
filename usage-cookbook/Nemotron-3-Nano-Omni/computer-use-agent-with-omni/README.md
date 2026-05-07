# Computer Use Agent with Nemotron-3 Nano Omni & Holotron-3-Nano

A self-contained, reproducible demo showing how to run **NVIDIA Nemotron-3 Nano Omni 30B** or **H Company Holotron-3-Nano** as a Computer Use Agent (CUA) that autonomously drives a live desktop environment via screenshots -> reasoning -> pyautogui actions.

Both models share the same Omni-Nano-v3 backbone and run on the same vLLM TP=2 setup, but use **different prompt/output contracts**:

| | Nemotron-3 Nano Omni (default) | Holotron-3-Nano |
|---|---|---|
| Output | Free-text `## Action: … ## Code: <pyautogui>` | Strict JSON `{note, thought, tool_call: {…}}` |
| Coordinates | Floats in `[0, 1]` projected to pixels | Integers in `[0, 1000]` scaled to pixels |
| Constraint | Reasoning parser | vLLM `structured_outputs` JSON schema |
| Tools | `pyautogui.*` + `computer.wait`/`computer.terminate` | 12-tool union (click_desktop, write_desktop, drag_to_desktop, scroll_desktop, hotkey_desktop, update_plan, answer, …) |
| Selection | `MODEL_FAMILY=nemotron` (default) | `MODEL_FAMILY=holotron` |

Switch between them with the `MODEL_FAMILY` env var; the FastAPI server picks the right agent class at startup. **Only one vLLM container can serve at a time** on a 2-GPU host — stop the active vLLM and start the sibling for the other model. See [Launch vLLM](#launch-vllm) below.

Inference currently uses an OpenAI-compatible **vLLM** endpoint only. You can run vLLM locally on a GPU machine or point the demo at a remote vLLM server. The hosted build.nvidia.com / NVIDIA Build / NIM path is intentionally disabled for now because current hosted endpoint behavior differs from the local vLLM server and breaks the computer-use agent loop.

```
 Your browser (http://localhost:8000)
 ┌──────────────────────────────────────────────────────────────────────┐
 │  ┌───────────────────────────┬──────────────────────────────────────┐│
 │  │     Live Desktop          │  Side Panel                          ││
 │  │     (KasmVNC iframe)      │  ┌─ Desktop ─────────────────────┐  ││
 │  │                           │  │  Ready — model name [Restart] │  ││
 │  │                           │  └────────────────────────────────┘  ││
 │  │                           │  ┌─ Task ────────────────────────┐  ││
 │  │                           │  │  "Open Chrome, go to           │  ││
 │  │  ┌─────────────────────┐  │  │   amazon.com, search keyboard" │  ││
 │  │  │ Ubuntu GNOME Desktop│  │  │  [▶ Run]  [■ Stop]             │  ││
 │  │  │ Chrome, Firefox,    │  │  └────────────────────────────────┘  ││
 │  │  │ VS Code, LibreOffice│  │  ┌─ Live Agent Trace ────────────┐  ││
 │  │  │ GIMP, VLC, Mail     │  │  │  step 1: 💭 I see a desktop…  │  ││
 │  │  │                     │  │  │  → click(0.5, 0.9)            │  ││
 │  │  │ Agent controls this │  │  │  step 2: 💭 Chrome opened…    │  ││
 │  │  │ in real-time ←──────│──│──│──── streaming reasoning        │  ││
 │  │  └─────────────────────┘  │  └────────────────────────────────┘  ││
 │  └───────────────────────────┴──────────────────────────────────────┘│
 └──────────────────────────────────────────────────────────────────────┘

         ┌──────────────────────────────┐
         │  vLLM OpenAI endpoint        │
         │  http://.../v1               │
         │  Nemotron-3 Nano Omni 30B    │
         │  or Hcompany/Holotron-3-Nano │
         │  - Vision + Reasoning        │
         │  - Streaming response        │
         └──────────────────────────────┘
```

## Quick Start

```bash
# 1. Start vLLM in a separate terminal, or point .env at an existing vLLM URL.
# See "Launch vLLM" below.

# 2. Clone and configure the demo
git clone https://github.com/NVIDIA-NeMo/Nemotron.git
cd Nemotron/usage-cookbook/Nemotron-3-Nano-Omni/computer-use-agent-with-omni
cp .env.example .env

# 3. Start the desktop + web server
docker compose up -d

# 4. Open the demo
open http://localhost:8000
```

The default `.env.example` expects vLLM on the host at `http://127.0.0.1:8001/v1`; the server container reaches that same endpoint as `http://host.docker.internal:8001/v1`.

The web UI shows:
- **Left pane**: Live KasmVNC desktop — you see exactly what the agent sees and does
- **Desktop Environment card**: Readiness status plus a **Restart** button for the OS container
- **Task Instruction card**: Type a task, **Run** the agent, or **Stop** the current inference/action loop
- **Live Agent Trace**: Streaming model reasoning, parsed actions, execution output, retry notices, and errors

## Prerequisites

| Requirement | Notes |
|---|---|
| Docker + Docker Compose | Desktop or server, any OS |
| NVIDIA GPU machine or remote vLLM endpoint | Required for the default `INFERENCE_PROVIDER=vllm` path |
| Hugging Face access | Required if your vLLM server downloads the model from Hugging Face |
| ~12 GB disk | For the desktop container image and build cache |

Nemotron-3 Nano Omni BF16 model weights are about 62 GB. Holotron-3-Nano weights are similar (also a 30B-class A3B model). Plan GPU memory and disk accordingly, or use a quantized variant supported by your vLLM setup.

## Launch vLLM

The demo only needs an OpenAI-compatible `/v1/chat/completions` endpoint. The default config assumes this endpoint is available on the host at `http://127.0.0.1:8001/v1`.

### Option A: Docker — Nemotron-3 Nano Omni (default)

This starts vLLM with the BF16 Hugging Face model ID and serves it as `vllm_local`, which matches `.env.example`.

```bash
docker pull vllm/vllm-openai:v0.20.0

docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  -p 8001:8001 \
  --name nano-omni-vllm \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.20.0 -lc '
    pip install "vllm[audio]" &&
    vllm serve nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
      --served-model-name vllm_local \
      --host 0.0.0.0 \
      --port 8001 \
      --trust-remote-code \
      --max-model-len 131072 \
      --gpu-memory-utilization 0.9 \
      --enable-prefix-caching \
      --max-num-seqs 8 \
      --allowed-local-media-path / \
      --reasoning-parser nemotron_v3
  '
```

If your Hugging Face account is required to access the model, export `HF_TOKEN` and add `-e HF_TOKEN` to the `docker run` command.

The agent sends `truncate_history_thinking=false` in `chat_template_kwargs` by default, so vLLM preserves previous-step thinking traces when rendering multi-turn GUI history. This is an inference request setting, not a `vllm serve` launch flag.

Verify vLLM before starting the demo:

```bash
curl -sS http://127.0.0.1:8001/v1/models | python3 -m json.tool
```

If vLLM runs on another machine, update `.env`:

```bash
INFERENCE_PROVIDER=vllm
VLLM_API_BASE=http://YOUR_VLLM_HOST:8001/v1
VLLM_API_KEY=EMPTY
VLLM_MODEL=vllm_local
```

When the FastAPI server runs inside Docker Compose and vLLM runs on the same host, keep `VLLM_API_BASE=http://host.docker.internal:8001/v1`.

### Option B: Docker — Hcompany/Holotron-3-Nano

To run the H Company Holotron model instead, stop the Nemotron container and start a sibling. Holotron uses the Qwen-3 reasoning parser:

```bash
docker stop nano-omni-vllm 2>/dev/null && docker rm nano-omni-vllm 2>/dev/null

docker run -d \
  --gpus '"device=0,1"' \
  --ipc=host \
  --shm-size=16g \
  -p 8011:8011 \
  --name nano-omni-vllm-holotron \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.20.0 -lc '
    pip install "vllm[audio]" &&
    exec vllm serve Hcompany/Holotron-3-Nano \
      --served-model-name holotron_local \
      --host 0.0.0.0 \
      --port 8011 \
      --tensor-parallel-size 2 \
      --trust-remote-code \
      --max-model-len 131072 \
      --gpu-memory-utilization 0.9 \
      --enable-prefix-caching \
      --max-num-seqs 8 \
      --allowed-local-media-path / \
      --reasoning-parser nemotron_v3
  '
```

Then point `.env` at this endpoint and select the matching model family:

```bash
INFERENCE_PROVIDER=vllm
VLLM_API_BASE=http://host.docker.internal:8011/v1
VLLM_API_KEY=EMPTY
VLLM_MODEL=holotron_local
MODEL_FAMILY=holotron
```

Restart the demo server to pick up the new agent class:

```bash
docker compose up -d --build server
curl -s http://localhost:8000/health    # → "model_family": "holotron"
```

The Holotron agent uses the official H Company `holo-nano` agent loop:
- 12-tool JSON schema (click/double_click/move_to/drag_to/scroll/write/key_down/key_up/hotkey/hold_and_tap_key/update_plan/answer)
- vLLM `structured_outputs` constrained decoding so the model can never emit malformed JSON
- `<observation>` / `<tool_output>` / `<error>` user-message wrappers around each turn
- Image budget: oldest screenshots are demoted to `[Image omitted by context cleaning]` text placeholders, last 3 kept

### Option C: Native vLLM

Use the same `vllm serve` flags from Options A or B after installing vLLM 0.20.0 in your Python environment. Keep the served model name aligned with `VLLM_MODEL`.

## Switching Between Models

The demo supports both Nemotron-3 Nano Omni and Hcompany/Holotron-3-Nano, but a typical 2× A6000 host has only enough GPU memory to serve **one model at a time**. Switching is a 3-step ritual: (1) replace the vLLM container, (2) point `.env` at the new endpoint and family, (3) rebuild the FastAPI server container so it picks up the new env vars.

### From Nemotron → Holotron

```bash
# 1. Replace the vLLM container
docker stop nano-omni-vllm 2>/dev/null && docker rm nano-omni-vllm 2>/dev/null

docker run -d \
  --gpus '"device=0,1"' --ipc=host --shm-size=16g \
  -p 8011:8011 --name nano-omni-vllm-holotron \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint /bin/bash vllm/vllm-openai:v0.20.0 -lc '
    pip install "vllm[audio]" &&
    exec vllm serve Hcompany/Holotron-3-Nano \
      --served-model-name holotron_local \
      --host 0.0.0.0 --port 8011 \
      --tensor-parallel-size 2 --trust-remote-code \
      --max-model-len 131072 --gpu-memory-utilization 0.9 \
      --enable-prefix-caching --max-num-seqs 8 \
      --allowed-local-media-path / --reasoning-parser nemotron_v3
  '

# Wait for vLLM to be ready (first launch downloads weights):
until curl -sf http://127.0.0.1:8011/v1/models >/dev/null; do sleep 5; done

# 2. Update .env (these four lines must match the running vLLM)
sed -i \
  -e 's|^VLLM_API_BASE=.*|VLLM_API_BASE=http://host.docker.internal:8011/v1|' \
  -e 's|^VLLM_MODEL=.*|VLLM_MODEL=holotron_local|' \
  -e 's|^MODEL_FAMILY=.*|MODEL_FAMILY=holotron|' \
  .env

# 3. Rebuild the demo server container (env vars are baked in at build time)
docker compose up -d --build server

# Verify
curl -s http://localhost:8000/health | python -m json.tool   # model_family: holotron
```

### From Holotron → Nemotron

```bash
# 1. Replace the vLLM container
docker stop nano-omni-vllm-holotron 2>/dev/null && docker rm nano-omni-vllm-holotron 2>/dev/null

docker run -d \
  --gpus '"device=0,1"' --ipc=host --shm-size=16g \
  -p 8001:8001 --name nano-omni-vllm \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint /bin/bash vllm/vllm-openai:v0.20.0 -lc '
    pip install "vllm[audio]" &&
    exec vllm serve nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
      --served-model-name vllm_local \
      --host 0.0.0.0 --port 8001 \
      --tensor-parallel-size 2 --trust-remote-code \
      --max-model-len 131072 --gpu-memory-utilization 0.9 \
      --enable-prefix-caching --max-num-seqs 8 \
      --allowed-local-media-path / --reasoning-parser nemotron_v3
  '

until curl -sf http://127.0.0.1:8001/v1/models >/dev/null; do sleep 5; done

# 2. Update .env
sed -i \
  -e 's|^VLLM_API_BASE=.*|VLLM_API_BASE=http://host.docker.internal:8001/v1|' \
  -e 's|^VLLM_MODEL=.*|VLLM_MODEL=vllm_local|' \
  -e 's|^MODEL_FAMILY=.*|MODEL_FAMILY=nemotron|' \
  .env

# 3. Rebuild the demo server
docker compose up -d --build server

curl -s http://localhost:8000/health | python -m json.tool   # model_family: nemotron
```

### Switching matrix

`MODEL_FAMILY` must match what the vLLM container is serving — running Holotron weights with `MODEL_FAMILY=nemotron` (or vice versa) produces immediate parse errors because the prompt contracts are incompatible.

| Model | `VLLM_API_BASE`                         | `VLLM_MODEL`     | `MODEL_FAMILY` | vLLM port |
|---|---|---|---|---|
| Nemotron-3 Nano Omni | `http://host.docker.internal:8001/v1`  | `vllm_local`     | `nemotron`     | 8001      |
| Hcompany/Holotron-3-Nano | `http://host.docker.internal:8011/v1`  | `holotron_local` | `holotron`     | 8011      |

The two families use different ports by convention (8001 vs 8011) so you can leave both `.env` blocks ready and only flip `MODEL_FAMILY` + the matching `VLLM_*` lines. The web UI's `/health` endpoint always reports the live `model_family` so you can confirm which model is currently driving the demo.

## How It Works

1. A **Docker container** (`desktop/`) runs a full Ubuntu 22.04 GNOME desktop with:
   - **KasmVNC** for browser-accessible live desktop viewing
   - A minimal Flask API for screenshots, screen size, health, and command execution
   - Desktop apps: Chrome, Firefox, LibreOffice, GIMP, VLC, VS Code, Thunderbird, Files, and Terminal
   - A clean desktop canvas with launchers kept in the dock/app grid instead of desktop icons
   - Based on the [ProRL-Agent-Server desktop recipe](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server/blob/docker_osworld/osworld-docker/Dockerfile), trimmed for this demo

2. The **FastAPI backend** (`server/`) orchestrates the agent loop:
   - Takes a screenshot from the desktop API
   - Sends screenshot + instruction + history through the configured vLLM endpoint
   - Parses the model's response — `## Action` / `## Code` for Nemotron, or a JSON `tool_call` for Holotron — into a uniform `ParsedStep`
   - Executes the resulting pyautogui commands inside the desktop container
   - Feeds tool execution results back into the conversation as `<tool_output>` for Holotron's next turn
   - Cancels active inference/tasks immediately when **Stop** is clicked
   - Retries model calls with configurable per-attempt timeout settings
   - Restarts the desktop container through the Docker Engine API when **Restart** is clicked
   - Repeats until the model calls `computer.terminate(status="success")` (Nemotron) or `answer` (Holotron)

3. The **web frontend** (`web/`) shows the live KasmVNC desktop in an iframe, exposes Run/Stop/Restart controls, and streams the agent's reasoning tokens in real-time via SSE.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  docker compose                                                      │
│                                                                      │
│  ┌──────────────────────────────┐    ┌───────────────────────────┐  │
│  │  server (FastAPI :8000)      │    │  desktop (GNOME :6901)    │  │
│  │  ├─ /web/*       static UI  │    │  ├─ KasmVNC              │  │
│  │  ├─ /vnc/*       proxy ─────│────│──│  (browser desktop)     │  │
│  │  ├─ /agent/*     loop       │    │  ├─ Desktop API :5000     │  │
│  │  │    ├─ screenshot ────────│────│──│── /screenshot          │  │
│  │  │    └─ execute ───────────│────│──│── /execute             │  │
│  │  ├─ /env/restart ───────────│────│──┤  restart container     │  │
│  │  └──────────────────────────┘    │  ├─ Desktop apps          │  │
│  │         │                         │  │  Chrome, Code, Office │  │
│  │         │ vLLM /v1 endpoint       │  └───────────────────────│  │
│  │         └─────────────────────────│──────────────────────────┘  │
│  └───────────────────────────────────────────────────────────────── │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
  OpenAI-compatible vLLM endpoint
  (Nemotron-3 Nano Omni 30B  or  Hcompany/Holotron-3-Nano)
```

## Configuration

All settings go in `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `INFERENCE_PROVIDER` | `vllm` | Must be `vllm`; other values are rejected by the server |
| `VLLM_API_BASE` | `http://host.docker.internal:8001/v1` | OpenAI-compatible vLLM base URL for Docker Compose |
| `VLLM_API_KEY` | `EMPTY` | Bearer token for vLLM |
| `VLLM_MODEL` | `vllm_local` | vLLM served model name (use `holotron_local` for the Holotron container) |
| `MODEL_FAMILY` | `nemotron` | Prompt/output contract: `nemotron` or `holotron`. Must match the model the vLLM container is serving. |
| `ENABLE_THINKING` | `true` | Enable reasoning mode (`<think>` tags). Honored by both families. |
| `TRUNCATE_HISTORY_THINKING` | `false` | Keep previous-step thinking traces in chat-template history |
| `MAX_STEPS` | `150` | Maximum agent steps |
| `MAX_IMAGE_HISTORY` | `3` | Screenshots kept in context window |
| `MODEL_MAX_TOKENS` | `20480` | Maximum generated tokens per model attempt |
| `REASONING_BUDGET` | `16384` | Max reasoning tokens |
| `REASONING_GRACE_TOKENS` | `1024` | Extra thinking-token allowance before final answer generation |
| `MODEL_ATTEMPT_TIMEOUT` | `120` | Seconds before one model attempt is timed out |
| `MODEL_MAX_RETRIES` | `3` | Maximum model attempts per agent step |
| `MODEL_RETRY_SLEEP` | `5` | Seconds to wait between model retry attempts |
| `COMPUTER_WAIT_SECONDS` | `3` | Duration for explicit `computer.wait` actions generated by the model |
| `DEMO_PORT` | `8000` | Port for the web UI |
| `DOCKER_SOCKET` | `/var/run/docker.sock` | Docker Engine socket used by `/env/restart` |
| `DESKTOP_CONTAINER_SERVICE` | `desktop` | Compose service name to restart |
| `DOCKER_RESTART_TIMEOUT` | `10` | Docker stop timeout, in seconds, during restart |
| `DESKTOP_API_PORT` | `5000` | Desktop API port inside the Compose network |
| `DESKTOP_VNC_PORT` | `6901` | KasmVNC port inside the Compose network |
| `DESKTOP_PASSWORD` | `password` | Desktop login password |
| `SCREEN_RESOLUTION` | `1920x1080` | Desktop resolution |

The server container mounts `/var/run/docker.sock` so the **Restart** button can restart only the Compose desktop service. Treat Docker socket access as host-level administrative access and expose this demo only in trusted development environments.

## NVIDIA Build / NIM

Hosted build.nvidia.com / NVIDIA Build / NIM inference is not supported by this demo at the moment. The computer-use agent loop is currently validated only against an OpenAI-compatible vLLM server because the hosted endpoint has behavior differences that affect action generation and parsing. Keep `INFERENCE_PROVIDER=vllm`.

## Project Structure

```
computer-use-agent-with-omni/
├── README.md
├── .env.example              # Template — copy to .env and configure inference
├── docker-compose.yml        # One-command setup
├── Dockerfile.server         # FastAPI server container
├── requirements.txt          # Python deps for the server
├── desktop/                  # Desktop container build context
│   ├── Dockerfile            # Ubuntu GNOME + KasmVNC + minimal desktop API
│   ├── requirements-desktop-api.txt # Python deps for the desktop API
│   ├── desktop-api/          # Flask API server (health, screenshot, execute)
│   │   └── main.py
│   ├── gnome-config/         # GNOME session file
│   ├── startup-scripts/      # Custom GNOME startup
│   ├── logind-mock.py        # Mock systemd-logind for container
│   └── kasmvnc-entrypoint.sh # Container entrypoint
├── server/
│   ├── __init__.py
│   ├── main.py               # FastAPI app (REST + SSE + VNC proxy); MODEL_FAMILY dispatch
│   ├── agent.py              # NemotronAgent: ## Action / ## Code prompt + parsing + coord projection
│   ├── holotron_agent.py     # HolotronAgent: H Company agent-loop (12-tool JSON schema, structured_outputs)
│   ├── nvinference.py        # Reserved inactive hosted inference path
│   ├── vllm_inference.py     # vLLM OpenAI-compatible inference path (both families)
│   ├── agent_runner.py       # Async screenshot→model→action loop (model-family agnostic)
│   └── desktop_client.py     # HTTP client for the desktop container API
└── web/
    ├── index.html            # Two-pane UI (VNC iframe + side panel)
    ├── sidepanel.js          # SSE + REST glue
    └── style.css             # Dark theme
```

## Development

```bash
# Run the server outside Docker (desktop container must be running and reachable)
docker compose up -d desktop
pip install -r requirements.txt
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

The default Compose file does not publish desktop ports to the host. For host-side server development, add a local Compose override that maps `127.0.0.1:5000:5000` and `127.0.0.1:6901:6901`, then set `DESKTOP_HOST=localhost`. When running the server outside Docker, `/env/restart` uses the host Docker socket path from `DOCKER_SOCKET`; the default `/var/run/docker.sock` works on typical Linux Docker installs.

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Redirects to web UI |
| `/health` | GET | Health check + desktop status |
| `/env/screenshot` | GET | Live PNG screenshot |
| `/env/restart` | POST | Stop active jobs, restart the desktop container, wait for readiness |
| `/agent/start` | POST | Start agent task `{instruction, max_steps?}` |
| `/agent/{job_id}/stop` | POST | Cancel a running task/inference |
| `/agent/{job_id}/status` | GET | Job status |
| `/agent/{job_id}/events` | GET | SSE stream of reasoning + actions |
| `/vnc/*` | * | KasmVNC reverse proxy |

## License

Apache 2.0
