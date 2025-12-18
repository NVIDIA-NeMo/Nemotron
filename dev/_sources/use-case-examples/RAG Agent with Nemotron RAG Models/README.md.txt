# RAG Agent with Nemotron RAG Models

A production-ready RAG (Retrieval-Augmented Generation) agent demonstrating a hybrid approach using local Hugging Face models for embeddings/reranking and NVIDIA AI Endpoints for LLM inference.

## Overview

This notebook builds an IT help desk support agent that can answer questions from an internal knowledge base using state-of-the-art retrieval and generation techniques.

## Models Used

- **Embedding**: `nvidia/llama-3.2-nv-embedqa-1b-v2` (Hugging Face)
- **Reranking**: `nvidia/llama-3.2-nv-rerankqa-1b-v2` (Hugging Face)
- **LLM**: `nvidia/nvidia-nemotron-nano-9b-v2` (NVIDIA AI Endpoints)

## Key Features

- 🏠 **Local Models** for embedding and reranking (privacy, performance, cost-effective)
- ☁️ **NVIDIA AI Endpoints** for LLM (managed service, latest models)
- 🤖 **LangGraph ReAct Agent** with tool integration
- 🔍 **Advanced Retrieval** with FAISS vector search and reranking
- ⚡ **GPU Acceleration** for local models (CPU fallback supported)

## Requirements

- Python 3.8+
- NVIDIA API key ([get one here](https://build.nvidia.com))
- GPU recommended (falls back to CPU if unavailable)
- Required packages: `transformers`, `langchain`, `langgraph`, `langchain-nvidia-ai-endpoints`, `faiss-cpu`, `torch`

