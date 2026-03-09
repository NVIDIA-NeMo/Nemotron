# NVIDIA Nemotron 3 Nano - Simple Usage Guide

A comprehensive guide to using **NVIDIA Nemotron 3 Nano** for basic inference, reasoning modes, and building intelligent agents with LangChain.

## Overview

This notebook provides a streamlined walkthrough of NVIDIA Nemotron 3 Nano capabilities, from simple API calls to orchestrating multi-agent research systems.

## Models Used

- **LLM**: `nvidia/nemotron-3-nano-30b-a3b` (via OpenRouter)

## Key Features

- 🔌 **OpenAI-Compatible API** via OpenRouter for easy integration
- 🧠 **Reasoning Modes** - Toggle chain-of-thought thinking ON/OFF
- 🔍 **Web Search Agent** with DuckDuckGo integration
- 💬 **Conversation Memory** using LangGraph's InMemorySaver
- 🤝 **Multi-Agent System** with specialized agents (Search, Report Writer, Quality Reviewer)

## What's Covered

| Part | Topic | Description |
|------|-------|-------------|
| **1** | Basic Usage | Simple API calls, streaming responses |
| **2** | Reasoning Modes | Enable/disable chain-of-thought reasoning |
| **3** | LangChain Agent | Research assistant with web search and memory |
| **4** | Multi-Agent System | Coordinator orchestrating specialist agents |

## Multi-Agent Architecture

```
┌─────────────────┐
│   Coordinator   │ ◄── User Query
│   (Supervisor)  │
└────────┬────────┘
         │
    ┌────┴────┬──────────┐
    ▼         ▼          ▼
┌───────┐ ┌────────┐ ┌──────────┐
│Search │ │Report  │ │Quality   │
│Agent  │ │Writer  │ │Reviewer  │
└───────┘ └────────┘ └──────────┘
```

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenRouter API key ([get one here](https://openrouter.ai/settings/keys))

## Quick Start

1. Clone the repository
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Run the notebook and enter your OpenRouter API key when prompted

## Resources

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [LangChain v1.0 Release Notes](https://docs.langchain.com/oss/python/releases/langchain-v1)
- [Multi-Agent Patterns](https://docs.langchain.com/oss/python/langchain/multi-agent)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

