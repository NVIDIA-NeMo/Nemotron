# 🎙️ Voice-Powered RAG Agent with NVIDIA Nemotron Models

Build a complete end-to-end AI agent that accepts voice input, retrieves multimodal context, reasons with long-context models, and enforces safety guardrails—all using the latest NVIDIA Nemotron open models.

## 🌟 Features

- **Voice Input**: Nemotron Speech ASR for real-time speech-to-text
- **Multimodal RAG**: Embed and retrieve both text and document images
- **Smart Reranking**: Improve retrieval accuracy by 6-7% with cross-encoder reranking
- **Image Understanding**: Describe visual content in context using vision-language models
- **Long-Context Reasoning**: Generate responses with 1M token context window
- **Safety Guardrails**: PII detection and content moderation in 20+ languages

## 📦 Models Used

| Component | Model | Parameters | Deployment |
|-----------|-------|------------|------------|
| **Speech-to-Text** | `nvidia/nemotron-speech-streaming-en-0.6b` | 600M | Self-hosted (NeMo) |
| **Embeddings** | `nvidia/llama-nemotron-embed-vl-1b-v2` | 1.7B | Self-hosted (Transformers) |
| **Reranking** | `nvidia/llama-nemotron-rerank-vl-1b-v2` | 1.7B | Self-hosted (Transformers) |
| **Vision-Language** | `nvidia/nemotron-nano-12b-v2-vl` | 12B | NVIDIA API |
| **Reasoning** | `nvidia/nemotron-3-nano-30b-a3b` | 30B | NVIDIA API |
| **Safety** | `nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3` | 8B | Self-hosted (Transformers) |

## 🔧 Requirements

### Hardware
- **GPU**: NVIDIA GPU with at least 24GB VRAM recommended (for self-hosted models)
- **CUDA**: 11.8 or later

### Software
- Python 3.10+
- PyTorch 2.0+
- NVIDIA API Key (for cloud-hosted models)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/NVIDIA/Nemotron_MultiModalRAGAgent.git
cd Nemotron_MultiModalRAGAgent
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key

```bash
export NVIDIA_API_KEY="your-nvidia-api-key"
```

Get your API key from [NVIDIA NGC](https://ngc.nvidia.com/).

### 4. Run the Tutorial

```bash
jupyter notebook voice_rag_agent_tutorial.ipynb
```

## 📁 Project Structure

```
Nemotron_MultiModalRAGAgent/
├── voice_rag_agent_tutorial.ipynb  # Main tutorial notebook
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
└── BlogSkeleton/                    # Blog content and model docs
    ├── BLOG.md
    ├── BLOG_UPDATED.md
    ├── Code Snippets/
    └── Model Information/
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Voice-Powered RAG Agent                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  🎤 Voice Input → Nemotron Speech ASR → Text Query                 │
│                           ↓                                         │
│  📚 Multimodal RAG                                                  │
│     ├── Embed (llama-nemotron-embed-vl-1b-v2)                      │
│     ├── Vector Search (FAISS)                                       │
│     └── Rerank (llama-nemotron-rerank-vl-1b-v2)                    │
│                           ↓                                         │
│  🖼️ Image Description (nemotron-nano-12b-v2-vl) [if images found]  │
│                           ↓                                         │
│  🧠 Response Generation (nemotron-3-nano-30b-a3b)                  │
│                           ↓                                         │
│  🛡️ Safety Check (Llama-3.1-Nemotron-Safety-Guard-8B-v3)           │
│                           ↓                                         │
│  📝 Safe Text Output                                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 📖 Tutorial Steps

1. **Environment Setup**: Install dependencies and configure API keys
2. **Multimodal RAG**: Build embeddings and vector store for text + images
3. **Speech Input**: Add real-time speech transcription
4. **Safety Guardrails**: Implement PII detection and content moderation
5. **Reasoning**: Generate context-aware responses
6. **LangGraph Workflow**: Wire everything into a complete agent

## 🎯 Use Cases

- **Enterprise Q&A**: Answer questions over documents with charts, tables, and images
- **Voice Assistants**: Build conversational AI with voice input
- **Compliance**: Detect PII and enforce content policies
- **Research**: Query scientific papers with visual content

## 📚 Resources

- [NVIDIA Nemotron Models](https://huggingface.co/nvidia)
- [NVIDIA NIM](https://developer.nvidia.com/nim)
- [NVIDIA NeMo Framework](https://github.com/NVIDIA/NeMo)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://docs.langchain.com/)

## 📄 License

This project uses NVIDIA open models. Each model is governed by its respective license:
- [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
- [Llama 3.1 Community License](https://www.llama.com/llama3_1/license/)

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## 📬 Support

- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
- [GitHub Issues](https://github.com/NVIDIA/Nemotron_MultiModalRAGAgent/issues)

