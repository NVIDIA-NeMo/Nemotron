# Long-Context Document Analysis with Nemotron 3 Super

Analyze large document collections in a single context window using Nemotron 3 Super's native 1M-token context capability - no chunking, no RAG pipeline required.

## Overview

This example demonstrates how to leverage Nemotron 3 Super's 1M native context window for practical document analysis tasks, progressing from single-document summarization to full cross-document synthesis:

1. **Document Corpus Construction** - Build a realistic multi-document corpus for analysis
2. **Single-Document Analysis** - Summarize and extract key findings from individual documents
3. **Multi-Document Q&A** - Answer questions that require synthesizing information across multiple documents
4. **Cross-Document Synthesis** - Identify themes, contradictions, and connections across the full corpus
5. **Context Length Scaling** - Compare quality and latency at different context sizes (32K, 128K, 256K)

## Models Used

| Component | Model | Parameters | Context Window | Deployment |
|-----------|-------|------------|----------------|------------|
| **Document Analysis** | `nvidia/nemotron-3-super-120b-a12b` | 120B total / 12B active | **1M tokens (native)** | NVIDIA API or self-hosted (vLLM) |

## Why Nemotron 3 Super for Long-Context Tasks?

- **1M native context window** - Trained with long-context extension via CPT methodology
- **Outperforms on RULER at 1M** - Higher accuracy than GPT-OSS and Qwen3.5 at maximum context
- **8x the context of Qwen 3.5** (128K) - Process entire codebases, legal corpora, or research collections
- **Hybrid Mamba-Transformer MoE** architecture maintains quality across long sequences
- **5x throughput improvement** - Process large documents efficiently

## Requirements

- Python 3.10+
- NVIDIA API Key ([get one here](https://build.nvidia.com/))

## Quick Start

```bash
# Install dependencies
pip install openai tiktoken

# Set your API key
export NVIDIA_API_KEY="your-key-here"

# Run the notebook
jupyter notebook long_context_analysis_tutorial.ipynb
```

## What You'll Learn

- How to structure prompts for effective long-context document analysis
- Building multi-document corpora that fit within context windows
- Techniques for cross-document synthesis without RAG
- How context length affects response quality and latency
- Best practices for instruction placement in long contexts
