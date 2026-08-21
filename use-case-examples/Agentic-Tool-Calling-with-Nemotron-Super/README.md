# Agentic Tool-Calling with Nemotron 3 Super

Build multi-step AI agents that plan, call tools, and synthesize results using Nemotron 3 Super's structured function-calling capabilities.

## Overview

This example demonstrates how to build agentic workflows with Nemotron 3 Super, progressing from simple tool calls to a fully autonomous agent loop:

1. **Single Tool Call** - Model selects and invokes one function
2. **Multi-Turn Tool Calling** - Model chains tool results across conversation turns
3. **Autonomous Agent Loop** - Model plans a strategy, executes multiple tools, and synthesizes a final report
4. **Reasoning Modes** - Compare `reasoning-off`, `regular`, and `low-effort` modes with tool calling

## Models Used

| Component | Model | Parameters | Deployment |
|-----------|-------|------------|------------|
| **Reasoning + Tool Calling** | `nvidia/nemotron-3-super-120b-a12b` | 120B total / 12B active | NVIDIA API or self-hosted (vLLM) |

## Why Nemotron 3 Super for Agents?

- **85.6% on PinchBench** - Best open model for agentic tasks
- **Trained on 21 RL environments** including TerminalBench, TauBench V2, and SWE-Bench
- **Structured tool calling** with JSON schema support via OpenAI-compatible API
- **Three reasoning modes** for balancing speed vs. depth in tool-calling scenarios
- **Hybrid Mamba-Transformer MoE** architecture delivers high throughput at inference time

## Requirements

- Python 3.10+
- NVIDIA API Key ([get one here](https://build.nvidia.com/))

## Quick Start

```bash
# Install dependencies
pip install openai

# Set your API key
export NVIDIA_API_KEY="your-key-here"

# Run the notebook
jupyter notebook agentic_tool_calling_tutorial.ipynb
```

## What You'll Learn

- How to define tools with JSON schema for Nemotron 3 Super
- Building a tool-calling conversation loop with proper message threading
- Implementing an autonomous agent that plans and executes multi-step tasks
- Choosing the right reasoning mode for different agentic scenarios
- Best practices for system prompts, error handling, and tool result formatting
