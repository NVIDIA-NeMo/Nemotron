# NVIDIA Nemotron Developer Repository

**Open and efficient models for agentic AI.** Training recipes, deployment guides, and use-case examples for the Nemotron family.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Docs](https://img.shields.io/badge/docs-dev-76B900.svg)](https://nvidia-nemo.github.io/Nemotron/dev/)

<div align="center">

[![Watch the Nemotron Overview](https://img.youtube.com/vi/_y9SEtn1lU8/hqdefault.jpg)](https://www.youtube.com/watch?v=_y9SEtn1lU8)

**[Watch: Nemotron Overview](https://www.youtube.com/watch?v=_y9SEtn1lU8)**

</div>

---

## Why Nemotron?

| | |
|---|---|
| **Open Models** | Fully transparent training data, techniques, and weights for community innovation |
| **Compute Efficiency** | Model pruning and optimization enabling higher throughput via TensorRT-LLM |
| **High Accuracy** | Built on frontier open models with human-aligned reasoning for agentic workflows |
| **Flexible Deployment** | Deploy anywhere: edge, single GPU, or data center with NIM microservices |

---

## Repository Overview

```
nemotron/
│
├── src/nemotron/recipes/    Training recipes (complete, reproducible pipelines)
│
├── usage-cookbook/          Usage cookbooks (deployment and model usage guides)
│
└── use-case-examples/       Examples of leveraging Nemotron in agentic workflows
```

---

## What is Nemotron?

[NVIDIA Nemotron](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/) is a family of open, high-efficiency multimodal models purpose-built for agentic AI.

**Model Tiers:**

- **Nano** — Optimized for edge and PC deployments
- **Super** — Single GPU deployment with highest throughput
- **Ultra** — Multi-GPU datacenter applications

Nemotron models excel at coding, math, scientific reasoning, tool calling, instruction following, and visual reasoning. Deploy across edge, single GPU, or data center environments with support for NeMo, TensorRT-LLM, vLLM, SGLang, and NIM microservices.

---

## Training Recipes

The Nemotron respository provides reproducible training pipelines from raw data to deployment-ready models. These implementations reflect how large language models are actually trained: careful experimentation, validation gates, and systematic optimization.

### Why Complete Pipelines?

Training a production model involves interconnected components. Isolated examples miss how stages interact. Complete pipelines show:

- **How data quality affects downstream performance** across pretraining, SFT, and RL
- **Which training techniques actually work together**, not just in theory
- **Where validation gates prevent failures** and maintain reproducibility
- **How to balance competing objectives** across stages

Because these are complete systems, you can extract specific techniques with confidence. Each component has been proven to work in context.

### Each Recipe Includes

- 🎨 **Synthetic Data Generation** - Scripts to generate synthetic datasets using [NVIDIA-NeMo/DataDesigner](https://github.com/NVIDIA-NeMo/DataDesigner)
- 🗂️ **Data Curation** - Scripts to prepare training data using [NVIDIA NeMo Curator](https://github.com/NVIDIA/NeMo-Curator) for scalable data processing, filtering, and quality enhancement
- 🔁 **Training** - Complete training loops with hyperparameters using:
  - [NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main) for Megatron models
  - [NVIDIA-NeMo/Automodel](https://github.com/NVIDIA-NeMo/Automodel) for HuggingFace models
  - [NVIDIA-NeMo/NeMo-RL](https://github.com/NVIDIA-NeMo/RL/tree/main) when RL is needed
  - Includes GPU-accelerated last-mile data processing (tokenization + optional sequence packing) for optimal training efficiency
- 📊 **Evaluation** - Benchmark evaluation on standard suites using [NVIDIA NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator)
- 📖 **Documentation** - Detailed explanations of each stage

### Available Recipes

| Model | Description | Stages | Guide |
|-------|-------------|--------|-------|
| **[Nemotron 3 Nano](docs/nemotron/nano3/README.md)** | 3.6B active / 31.6B total MoE Hybrid Mamba-Transformer for agentic reasoning | Pretrain → SFT → RL | [Training Guide](docs/nemotron/nano3/README.md) |

### Nemotron 3 Nano

A complete training recipe for the open, efficient Mixture-of-Experts hybrid Mamba-Transformer model optimized for agentic reasoning.

> **Open-Source Data Only**: These recipes train exclusively on the open-sourced subset of training data. Results will differ from the tech report benchmarks, which used additional proprietary data. Use these recipes as reference implementations to apply the methodology with your own data.

**Model Specifications**:
- 31.6B total parameters, 3.6B active per forward pass
- 25 trillion pretraining tokens with curriculum learning
- Up to 1M context length
- 3.3x higher inference throughput than similarly sized models

**What You Can Extract**:
- Curriculum-based pretraining with two-phase data mixture
- Long-context extension via CPT methodology
- Multi-domain SFT with 12+ data sources
- InfinityByte cross-domain code synthesis
- Tool-calling fine-tuning and budget-controlled reasoning
- Multi-environment RLVR with GRPO
- GenRM reward modeling with circular comparison
- DPO for tool hallucination reduction

**Resources**:
- [Training Guide](docs/nemotron/nano3/README.md)
- [Tech Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf)
- [Model Weights (Base)](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16)
- [Model Weights (Instruct)](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- [Model Weights (FP8)](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8)

---

## Usage Cookbooks

Practical deployment and model usage guides for Nemotron models.

| Model | Best For | Key Features | Resources |
|-------|----------|--------------|-----------|
| [**Llama-3.3-Nemotron-Super-49B-v1.5**](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5) | Production deployments needing strong reasoning | 128K context, single H200 GPU, RAG & tool calling | [Cookbooks](./usage-cookbook/Llama-Nemotron-Super-49B-v1.5/) |
| [**NVIDIA-Nemotron-Nano-9B-v2**](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) | Resource-constrained environments | 9B params, hybrid Mamba-2, controllable reasoning | [Cookbooks](./usage-cookbook/Nemotron-Nano-9B-v2/) |
| [**NVIDIA-Nemotron-Nano-12B-v2-VL**](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL) | Document intelligence and video understanding | 12B VLM, video reasoning, Efficient Video Sampling | [Cookbooks](./usage-cookbook/Nemotron-Nano2-VL/) |
| [**Llama-3.1-Nemotron-Safety-Guard-8B-v3**](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3) | Multilingual content moderation | 9 languages, 23 safety categories | [Cookbooks](./usage-cookbook/Llama-3.1-Nemotron-Safety-Guard-V3/) |
| **Nemotron-Parse** | Document parsing for RAG and AI agents | Table extraction, semantic segmentation | [Cookbooks](./usage-cookbook/Nemotron-Parse-v1.1/) |

---

## Use Case Examples

End-to-end examples demonstrating practical applications in the [`use-case-examples/`](./use-case-examples/) directory:

- **Agentic Workflows** — Multi-step AI agents with planning, context management, and external tools
- **RAG Systems** — Pipelines combining retrieval with Nemotron models for grounded outputs
- **Tool Integration** — Structured tool calling, function execution, and data enrichment
- **Production Patterns** — Scalability, monitoring, and deployment architectures

---

## Nemotron Open Datasets

More than just weights, recipes, and libraries: Nemotron is committed to opening data across many domains, training phases, and use cases.

<details>
<summary><strong>Nemotron Data Catalogue</strong></summary>

*A comprehensive collection of NVIDIA Nemotron datasets spanning pre-training, post-training, reinforcement learning, multimodal, safety, and domain-specific applications. These openly available datasets power the Nemotron family of models for agentic AI development.*

---

<details>
<summary><strong>Code</strong></summary>

*Datasets for training code generation, competitive programming, and software engineering capabilities across multiple programming languages.*

| Dataset | Usage | License | Model(s) | Description |
|---------|-------|---------|----------|-------------|
| [Nemotron-CC-Code-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Code-v1) | Pre-training | NVIDIA Data Agreement | Nemotron 3 Nano | 427.9B tokens from Common Crawl code pages using Lynx + LLM pipeline |
| [Nemotron-Pretraining-Code-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v1) | Pre-training | NVIDIA Data Agreement | Nemotron Nano 2 | GitHub-sourced code corpus for Nemotron Nano 2 |
| [Nemotron-Pretraining-Code-v2](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v2) | Pre-training | NVIDIA Data Agreement | Nemotron 3 Nano | Updated GitHub code + synthetic QA with STEM reasoning |
| [Nemotron-Cascade-RL-SWE](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-RL-SWE) | RL Training | CC-BY-4.0 | Nemotron 3 | SWE code repair from SWE-Bench, SWE-Smith, R2E-Gym |
| [Nemotron-Competitive-Programming-v1](https://huggingface.co/datasets/nvidia/Nemotron-Competitive-Programming-v1) | SFT | CC-BY-4.0 | Nemotron 3 | 2M+ Python and 1M+ C++ samples across 34K competitive programming questions |
| [OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) | SFT | CC-BY-4.0 | OpenCode-Nemotron | 735K Python samples across 28K competitive programming questions |
| [OpenCodeReasoning-2](https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2) | SFT | CC-BY-4.0 | OpenCode-Nemotron | 2.5M samples (1.4M Python, 1.1M C++) with code completion and critique |
| [Scoring-Verifiers](https://huggingface.co/datasets/nvidia/Scoring-Verifiers) | Evaluation | CC-BY-4.0 | — | Benchmark for test case generation and code reward models |

</details>

---

<details>
<summary><strong>Math</strong></summary>

*Mathematical reasoning datasets ranging from pre-training corpora to advanced problem-solving with chain-of-thought and tool-integrated reasoning. Includes the AIMO-2 competition winning dataset.*

| Dataset | Usage | License | Model(s) | Description |
|---------|-------|---------|----------|-------------|
| [Nemotron-CC-Math-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1) | Pre-training | NVIDIA Data Agreement | Nemotron Nano 2, Nemotron 3 Nano | 133B-token math dataset from Common Crawl using Lynx + LLM pipeline |
| [Nemotron-Math-Proofs-v1](https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v1) | SFT | CC-BY-4.0 | Nemotron 3 Nano | Mathematical proofs dataset for Nemotron 3 post-training |
| [Nemotron-Math-v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2) | SFT | CC-BY-4.0 | Nemotron 3 | 347K samples and 7M reasoning trajectories for Deeper Math Reasoning |
| [Nemotron-CrossThink](https://huggingface.co/datasets/nvidia/Nemotron-CrossThink) | RL Training | CC-BY-4.0 | Nemotron 3 | Multi-domain QA with MCQ and open-ended formats for verifiable rewards |
| [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) | SFT | CC-BY-4.0 | OpenMath-Nemotron | 5.68M samples, 306K problems from AoPS with CoT/TIR (AIMO-2 winner) |

</details>

---

<details>
<summary><strong>Science / STEM</strong></summary>

*Scientific reasoning datasets covering chemistry, physics, and general STEM domains for training models on scientific question answering and reasoning.*

| Dataset | Usage | License | Model(s) | Description |
|---------|-------|---------|----------|-------------|
| [Nemotron-Science-v1](https://huggingface.co/datasets/nvidia/Nemotron-Science-v1) | SFT | CC-BY-4.0 | Nemotron 3 Nano | Synthetic science reasoning (MCQA + chemistry RQA) |

</details>

---

<details>
<summary><strong>General / Web</strong></summary>

*Large-scale web-crawled and curated datasets for pre-training and post-training, including multilingual data and general instruction-following capabilities.*

| Dataset | Usage | License | Model(s) | Description |
|---------|-------|---------|----------|-------------|
| [Nemotron-CC-v2.1](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1) | Pre-training | NVIDIA Data Agreement | Nemotron 3 Nano | 2.5T tokens English web data with synthetic rephrases and translations |
| [Nemotron-CC-v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) | Pre-training | NVIDIA Data Agreement | Nemotron Nano 2 | 6.6T tokens quality-filtered Common Crawl with multilingual Q&A |
| [Nemotron-Pretraining-Dataset-sample](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample) | Pre-training (Sample) | NVIDIA Data Agreement | — | Sample subset of Nemotron pre-training corpus for experimentation |
| [Llama-Nemotron-Post-Training-Dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) | SFT + RL | CC-BY-4.0 | Llama-Nemotron Ultra/Super/Nano | Math, code, reasoning data (2.2M math, 500K code) |
| [Nemotron-Post-Training-Dataset-v1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1) | SFT | CC-BY-4.0 | Llama-3.3-Nemotron-Super-49B-v1.5 | Math, code, STEM, tool calling |
| [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) | SFT + RL | CC-BY-4.0 | Llama-Nemotron | Multilingual extension (Spanish, French, German, Italian, Japanese) |
| [Nemotron-3-Nano-RL-Training-Blend](https://huggingface.co/datasets/nvidia/Nemotron-3-Nano-RL-Training-Blend) | RL Training | CC-BY-4.0 | Nemotron-3-Nano-30B-A3B | Curated multi-domain blend for Nemotron 3 Nano |
| [Nemotron-RL-knowledge-web_search-mcqa](https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-web_search-mcqa) | RL Training | ODC-BY-1.0 | Nemotron 3 | Web search and multiple-choice QA tasks for NeMo Gym |

</details>

---

<details>
<summary><strong>Chat / Instruction Following</strong></summary>

*Datasets for training conversational AI with strong instruction-following capabilities, structured output generation, and multi-turn dialogue.*

| Dataset | Usage | License | Model(s) | Description |
|---------|-------|---------|----------|-------------|
| [Nemotron-Instruction-Following-Chat-v1](https://huggingface.co/datasets/nvidia/Nemotron-Instruction-Following-Chat-v1) | SFT | CC-BY-4.0 | Nemotron 3 Nano | Multi-turn chat and structured output generation |
| [Nemotron-RL-instruction_following](https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following) | RL Training | ODC-BY-1.0 | Nemotron 3 | Verifiable instruction adherence from WildChat-1M + Open-Instruct |
| [Nemotron-RL-instruction_following-structured_outputs](https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following-structured_outputs) | RL Training | ODC-BY-1.0 | Nemotron 3 | JSON schema-constrained output formatting tests |
| [Nemotron-Cascade-RL-Instruction-Following](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-RL-Instruction-Following) | RL Training | ODC-BY-1.0 | Nemotron 3 | 108K samples for instruction-following RL |

</details>

---

<details>
<summary><strong>Agentic / Tool Use</strong></summary>

*Datasets for training AI agents with tool calling, multi-step workflows, and agentic reasoning capabilities.*

| Dataset | Usage | License | Model(s) | Description |
|---------|-------|---------|----------|-------------|
| [Nemotron-Agentic-v1](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1) | SFT | CC-BY-4.0 | Nemotron 3 Nano | Multi-turn trajectories for conversational tool use and agentic workflows |
| [Nemotron-RL-agent-workplace_assistant](https://huggingface.co/datasets/nvidia/Nemotron-RL-agent-workplace_assistant) | RL Training | ODC-BY-1.0 | Nemotron 3 | Workplace assistant agent tasks for NeMo Gym |

</details>

---

<details>
<summary><strong>Alignment / Reward Modeling</strong></summary>

*Human preference and reward modeling datasets for RLHF, SteerLM training, and model alignment. Powers top-performing reward models on RM-Bench and JudgeBench.*

| Dataset | Usage | License | Model(s) | Description |
|---------|-------|---------|----------|-------------|
| [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) | Reward Modeling | CC-BY-4.0 | Nemotron 3 Nano, Llama-Nemotron Super 49B | 40K+ samples; top on RM-Bench/JudgeBench with preference, feedback, edit-quality |
| [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | Reward Modeling | CC-BY-4.0 | Nemotron-4-340B-Reward, Llama-3.1-Nemotron-70B-Reward | 21K samples with 5 attributes |
| [HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) | SteerLM Training | CC-BY-4.0 | Nemotron-4 SteerLM | 37K samples (helpfulness, correctness, coherence, complexity, verbosity) |
| [Daring-Anteater](https://huggingface.co/datasets/nvidia/Daring-Anteater) | SFT/RLHF | CC-BY-4.0 | Nemotron-4-340B-Instruct | Instruction tuning dataset; synthetic subsets + FinQA, wikitablequestions |
| [sft_datablend_v1](https://huggingface.co/datasets/nvidia/sft_datablend_v1) | SFT | CC-BY-4.0 | — | SFT data blend for RLHF pipeline |

</details>

---

<details>
<summary><strong>Vision-Language / Multimodal</strong></summary>

*High-quality VLM training data for document intelligence, OCR, image reasoning, video QA, and chain-of-thought visual understanding.*

| Dataset | Usage | License | Model(s) | Description |
|---------|-------|---------|----------|-------------|
| [Nemotron-VLM-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-VLM-Dataset-v2) | VLM Training | CC-BY-4.0 (some CC-BY-SA-4.0) | Nemotron VLM | 8M samples for OCR, image reasoning, video QA with chain-of-thought |
| [Llama-Nemotron-VLM-Dataset-v1](https://huggingface.co/datasets/nvidia/Llama-Nemotron-VLM-Dataset-v1) | VLM Training | CC-BY-4.0 (some CC-BY-SA-4.0) | Llama-3.1-Nemotron-Nano-VL-8B | 3M samples for visual question answering and captioning |

</details>

---

<details>
<summary><strong>Physical AI / Robotics</strong></summary>

*Datasets for embodied reasoning, physical common sense, and robotic manipulation. Powers Cosmos-Reason1 for physical AI applications.*

| Dataset | Usage | License | Model(s) | Description |
|---------|-------|---------|----------|-------------|
| [Cosmos-Reason1-SFT-Dataset](https://huggingface.co/datasets/nvidia/Cosmos-Reason1-SFT-Dataset) | SFT | CC-BY-4.0 | Cosmos-Reason1-7B | Video-text pairs for robotics, ego-centric demos, AV reasoning |
| [Cosmos-Reason1-RL-Dataset](https://huggingface.co/datasets/nvidia/Cosmos-Reason1-RL-Dataset) | RL Training | CC-BY-4.0 | Cosmos-Reason1-7B | RL data for physical common sense and embodied reasoning |
| [Cosmos-Reason1-Benchmark](https://huggingface.co/datasets/nvidia/Cosmos-Reason1-Benchmark) | Evaluation | CC-BY-4.0 | — | Benchmark for embodied reasoning (robotics, HoloAssist, AV) |
| [PhysicalAI-Robotics-Manipulation-Augmented](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Augmented) | Training | CC-BY-4.0 | — | 1K Franka Panda demos with Cosmos Transfer1 domain augmentation |

</details>

---

<details>
<summary><strong>Autonomous Vehicles</strong></summary>

*Multi-sensor driving data and synthetic scenarios for training and validating autonomous vehicle systems.*

| Dataset | Usage | License | Model(s) | Description |
|---------|-------|---------|----------|-------------|
| [PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) | Training | NVIDIA AV Dataset License | — | 1,700 hours multi-sensor data from 25 countries, 306K clips |
| [PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams) | SDG | CC-BY-4.0 | Cosmos | 81K synthetic videos with LiDAR and HD-map annotations |
| [PhysicalAI-Autonomous-Vehicle-Cosmos-Synthetic](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Synthetic) | SDG | CC-BY-4.0 | Cosmos | Cosmos-generated synthetic driving scenarios |
| [PhysicalAI-Autonomous-Vehicles-NuRec](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) | Reconstruction | NVIDIA AV Dataset License | — | NuScenes-based reconstruction data |

</details>

---

<details>
<summary><strong>Synthetic Personas / Data Generation</strong></summary>

*Privacy-safe synthetic personas grounded in real-world demographics for sovereign AI development and synthetic data generation pipelines.*

| Dataset | Usage | License | Model(s) | Description |
|---------|-------|---------|----------|-------------|
| [Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA) | SDG | CC-BY-4.0 | NeMo Data Designer | 1M US personas grounded in Census demographics |
| [Nemotron-Personas-Japan](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Japan) | SDG | CC-BY-4.0 | NeMo Data Designer | 1M Japanese personas aligned with regional statistics |
| [Nemotron-Personas-India](https://huggingface.co/datasets/nvidia/Nemotron-Personas-India) | SDG | CC-BY-4.0 | NeMo Data Designer | 3M Indian personas for sovereign AI development |
| [Nemotron-Personas](https://huggingface.co/datasets/nvidia/Nemotron-Personas) | SDG | CC-BY-4.0 | NeMo Data Designer | 100K US personas with 22 fields aligned to Census data |

</details>

---

<details>
<summary><strong>Privacy / PII Detection</strong></summary>

*Synthetic datasets for training named entity recognition models to detect and redact personally identifiable information.*

| Dataset | Usage | License | Model(s) | Description |
|---------|-------|---------|----------|-------------|
| [Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) | NER Training | CC-BY-4.0 | GLiNER-PII | 100K synthetic records with 55+ PII/PHI entity types |

</details>

---

<details>
<summary><strong>Safety / Content Moderation</strong></summary>

*Content safety datasets for training guardrail models covering comprehensive risk taxonomies. Powers NemoGuard content safety models.*

| Dataset | Usage | License | Model(s) | Description |
|---------|-------|---------|----------|-------------|
| [Aegis-AI-Content-Safety-Dataset-1.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0) | Content Moderation | CC-BY-4.0 | NemoGuard Permissive/Defensive | 11K annotated interactions covering 13 risk categories |
| [Aegis-AI-Content-Safety-Dataset-2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) | Content Moderation | CC-BY-4.0 | Llama-3.1-NemoGuard-8B-ContentSafety | Extended safety dataset with 23 violation categories |
| [Nemotron-Content-Safety-Audio-Dataset](https://huggingface.co/datasets/nvidia/Nemotron-Content-Safety-Audio-Dataset) | Audio Safety | CC-BY-4.0 | — | 1.9K audio files from Aegis 2.0 with accent diversity |

</details>

---

<details>
<summary><strong>RAG / Conversational QA</strong></summary>

*Training and evaluation data for retrieval-augmented generation and conversational question answering. Powers ChatQA models.*

| Dataset | Usage | License | Model(s) | Description |
|---------|-------|---------|----------|-------------|
| [ChatRAG-Bench](https://huggingface.co/datasets/nvidia/ChatRAG-Bench) | Evaluation | Other (derived) | — | Benchmark across 10 datasets for document QA and unanswerable detection |
| [ChatQA-Training-Data](https://huggingface.co/datasets/nvidia/ChatQA-Training-Data) | SFT | Other (derived) | ChatQA-1.5 | Training data for ChatQA models from multiple sources |
| [ChatQA2-Long-SFT-data](https://huggingface.co/datasets/nvidia/ChatQA2-Long-SFT-data) | SFT | Other (derived) | ChatQA-2 | 128K long-context training data for ChatQA-2 |

</details>

---

<details>
<summary><strong>Biology / Drug Discovery</strong></summary>

*Protein sequence data for training biological foundation models.*

| Dataset | Usage | License | Model(s) | Description |
|---------|-------|---------|----------|-------------|
| [esm2_uniref_pretraining_data](https://huggingface.co/datasets/nvidia/esm2_uniref_pretraining_data) | Pre-training | CC-BY-4.0 | ESM2-nv | 188M protein sequences for ESM2 |

</details>

---

<details>
<summary><strong>3D / Spatial Intelligence</strong></summary>

*Testing and synthetic data for 3D reconstruction, video generation, and spatial understanding models.*

| Dataset | Usage | License | Model(s) | Description |
|---------|-------|---------|----------|-------------|
| [Lyra-Testing-Example](https://huggingface.co/datasets/nvidia/Lyra-Testing-Example) | Evaluation | CC-BY-4.0 | Lyra | Testing examples for Lyra generative 3D reconstruction |
| [PhysicalAI-SpatialIntelligence-Lyra-SDG](https://huggingface.co/datasets/nvidia/PhysicalAI-SpatialIntelligence-Lyra-SDG) | SDG | CC-BY-4.0 | Lyra | Synthetic data for spatial intelligence models |
| [GEN3C-Testing-Example](https://huggingface.co/datasets/nvidia/GEN3C-Testing-Example) | Evaluation | CC-BY-4.0 | GEN3C | Testing examples for GEN3C video generation |
| [ChronoEdit-Example-Dataset](https://huggingface.co/datasets/nvidia/ChronoEdit-Example-Dataset) | Evaluation | CC-BY-4.0 | ChronoEdit | Temporal reasoning examples for image editing |

</details>

</details>

---

## 💡 Feature Requests & Ideas

Have an idea for improving Nemotron models? Create a [Discussion](https://github.com/NVIDIA-NeMo/Nemotron/discussions) topic for it!

If you have a feature request, feel free to open an [Issue](https://github.com/NVIDIA-NeMo/Nemotron/issues) and tag it as `enhancement`.

Your feedback helps shape the future of Nemotron models!

---

## Documentation

- [Nemotron 3 Nano Training Guide](docs/nemotron/nano3/README.md) – training recipe
- [NeMo-Run Configuration](docs/nemo_runspec/nemo-run.md) – execution profiles and job orchestration
- [Data Preparation](docs/nemotron/data-prep.md) – data preparation module
- [Contributing Guidelines](CONTRIBUTING.md) – how to contribute
- [Changelog](CHANGELOG.md) – version history

---

## Contributing

We welcome contributions: examples, recipes, or other tools. Please read the [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

---

## Security

To report any vulnerabilities, please reach out to [security@nvidia.com](mailto:security@nvidia.com)

---

## License

Apache 2.0 License — see [LICENSE](LICENSE) for details.

---

**NVIDIA Nemotron** — Open and efficient models for agentic AI.
