# Nemotron 3 Ultra with Agentic Coding Tools

Nemotron 3 Ultra is a 550B total / 55B active-parameter mixture-of-experts
model built for long-running agentic workflows. This guide covers config-based
setup for **OpenCode**, **OpenClaw**, **Kilo Code CLI**, **OpenHands CLI**,
**Hermes Agent**, **Pi**, and **VT Code**.

The hosted model ID is:

```text
nvidia/nemotron-3-ultra-550b-a55b
```

Use one of these hosted access paths:

- NVIDIA NIM on build.nvidia.com: `https://integrate.api.nvidia.com/v1`
- OpenRouter: `nvidia/nemotron-3-ultra-550b-a55b`

## Why Nemotron 3 Ultra for Agentic Coding

Agentic coding tools need reliable behavior across many sequential tool calls,
large project contexts, and long-running state. Nemotron 3 Ultra is post-trained
for leading agent harnesses and is designed for coding, research, and enterprise
workflows that require stronger reasoning than a single-GPU model tier.

**550B total / 55B active parameters.** Ultra uses sparse MoE routing so each
token activates a smaller expert subset while retaining frontier-scale capacity.

**Hybrid Mamba-Transformer architecture.** Mamba layers help sustain long
contexts efficiently, while transformer layers preserve strong reasoning and
tool-use behavior.

**1M-token context.** Ultra can keep large repository, trace, and tool-output
contexts live across extended agent sessions.

**MTP support.** Multi-token prediction is designed to improve structured
generation and accelerate inference through speculative decoding.

## Benchmark Performance

On [PinchBench](https://pinchbench.com/model/nvidia/nvidia/nemotron-3-ultra-550b-a55b),
which evaluates models as the brain of an OpenClaw agent, Nemotron 3 Ultra is
listed at:

| Model | Best Score | Average Score | Median Score |
| --- | ---: | ---: | ---: |
| `nvidia/nemotron-3-ultra-550b-a55b` | 90.6% | 89.9% | 90.0% |

## Shared Setup

For OpenRouter-backed tools:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

For direct NVIDIA NIM access through build.nvidia.com:

```bash
export NVIDIA_API_KEY="nvapi-..."
export NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1"
export NVIDIA_MODEL="nvidia/nemotron-3-ultra-550b-a55b"
```

## OpenCode

[OpenCode](https://opencode.ai) is an open-source terminal coding agent. Configure
it via `~/.config/opencode/opencode.json`.

**Install**

```bash
npm install -g opencode-ai
```

**Configure**

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "nvidia/nemotron-3-ultra-550b-a55b",
  "provider": {
    "nvidia": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "NVIDIA NIM",
      "options": {
        "baseURL": "{env:NVIDIA_BASE_URL}",
        "apiKey": "{env:NVIDIA_API_KEY}"
      },
      "models": {
        "nemotron-3-ultra-550b-a55b": {
          "name": "nvidia/nemotron-3-ultra-550b-a55b",
          "limit": {
            "context": 1000000,
            "output": 32768
          }
        }
      }
    }
  },
  "agent": {
    "build": {
      "temperature": 0.6,
      "top_p": 0.95,
      "max_tokens": 32000
    },
    "plan": {
      "temperature": 0.6,
      "top_p": 0.95,
      "max_tokens": 32000
    }
  }
}
```

**Run**

```bash
cd /path/to/project
opencode
/init
```

## OpenClaw

[OpenClaw](https://docs.openclaw.ai) is a persistent autonomous agent that can
run locally or as a daemon. OpenRouter model refs use
`openrouter/<provider>/<model>`.

**Install**

Requires Node.js >= 22.

```bash
npm install -g openclaw@latest
openclaw onboard --install-daemon \
  --auth-choice apiKey \
  --token-provider openrouter \
  --token "$OPENROUTER_API_KEY"
```

**Configure**

Edit `~/.openclaw/openclaw.json`:

```json
{
  "agents": {
    "defaults": {
      "model": {
        "primary": "openrouter/nvidia/nemotron-3-ultra-550b-a55b"
      }
    }
  }
}
```

**Verify and run**

```bash
openclaw doctor
openclaw status
openclaw tui
```

## Kilo Code CLI

[Kilo Code CLI](https://kilo.ai/docs/cli) is built from the same engine as the
Kilo IDE extension and supports config-file model registration.

**Install**

```bash
npm install -g @kilocode/cli
```

**Configure**

Create or edit `~/.config/kilo/opencode.json`:

```json
{
  "$schema": "https://app.kilo.ai/config.json",
  "model": "nvidia/nemotron-3-ultra-550b-a55b",
  "provider": {
    "nvidia": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "NVIDIA NIM",
      "options": {
        "baseURL": "{env:NVIDIA_BASE_URL}",
        "apiKey": "{env:NVIDIA_API_KEY}"
      },
      "models": {
        "nemotron-3-ultra-550b-a55b": {
          "name": "nvidia/nemotron-3-ultra-550b-a55b",
          "limit": {
            "context": 1000000,
            "output": 32768
          }
        }
      }
    }
  }
}
```

**Run**

```bash
cd /path/to/project
kilo
```

## OpenHands

[OpenHands CLI](https://openhands.dev/product/cli) uses LiteLLM-compatible LLM
configuration. For an OpenAI-compatible endpoint such as NVIDIA NIM, prefix the
model with `openai/`.

**Install**

```bash
uv tool install openhands --python 3.12
```

**Configure**

Set environment variables before starting OpenHands:

```bash
export LLM_MODEL="openai/nvidia/nemotron-3-ultra-550b-a55b"
export LLM_BASE_URL="https://integrate.api.nvidia.com/v1"
export LLM_API_KEY="$NVIDIA_API_KEY"
export LLM_MAX_INPUT_TOKENS=1000000
export LLM_MAX_OUTPUT_TOKENS=32768
```

Or edit `~/.openhands/config.toml`:

```toml
[llm]
model = "openai/nvidia/nemotron-3-ultra-550b-a55b"
base_url = "https://integrate.api.nvidia.com/v1"
api_key = "nvapi-..."
max_input_tokens = 1000000
max_output_tokens = 32768
```

**Run**

```bash
openhands
```

## Hermes Agent

[Hermes Agent](https://openrouter.ai/docs/cookbook/coding-agents/hermes-integration)
is a terminal-native autonomous coding and task agent with persistent memory and
multi-provider model routing.

**Configure**

Add the OpenRouter key to `~/.hermes/.env`:

```bash
OPENROUTER_API_KEY=sk-or-...
```

Then edit `~/.hermes/config.yaml`:

```yaml
model:
  provider: openrouter
  default: nvidia/nemotron-3-ultra-550b-a55b
```

**Run**

```bash
hermes
```

## Pi

[Pi](https://pi.dev) is a terminal coding agent with provider and model defaults
stored under `~/.pi/agent`.

**Install**

```bash
npm install -g @mariozechner/pi-coding-agent
```

**Configure**

Create `~/.pi/agent/auth.json`:

```json
{
  "openrouter": {
    "type": "api_key",
    "key": "sk-or-..."
  }
}
```

Create or edit `~/.pi/agent/settings.json`:

```json
{
  "defaultProvider": "openrouter",
  "defaultModel": "nvidia/nemotron-3-ultra-550b-a55b"
}
```

**Run**

```bash
cd /path/to/project
pi
```

## VT Code

[VT Code](https://github.com/vinhnx/vtcode) is a Rust-native terminal coding
agent with safe workspace tools and multi-provider LLM support. Configure it
via `vtcode.toml` in your project root.

**Install**

```bash
cargo install vtcode
```

**Configure**

Via HuggingFace:

```bash
export HF_TOKEN="hf_..."
```

```toml
[agent]
provider = "huggingface"
default_model = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4:together"
```

[agent]
provider = "openrouter"
default_model = "nvidia/nemotron-3-ultra-550b-a55b"

**Run**

```bash
cd /path/to/project
vtcode
```
