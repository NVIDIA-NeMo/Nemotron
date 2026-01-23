# Nemotron-Nano2-VL Notebooks

A collection of notebooks demonstrating the capabilities of **NVIDIA Nemotron Nano 2 VL**, a 12B parameter model that unifies visual and textual understanding for advanced multimodal agentic workflows.

## Overview

These notebooks show how to use **NVIDIA Nemotron Nano 2 VL** to build applications that can see, read, and reason across diverse media. The model can extract, understand, and act on information from text, images, and videos, making it a powerful tool for next-generation AI agents.

## Models

- **VLM (NIM)**: `nvidia/nemotron-nano-2-vl` (Available soon on [NVIDIA AI Endpoints](https://build.nvidia.com))
- **VLM (Hugging Face)**: `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8` ([link](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8))
- **VLM (Hugging Face)**: `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` ([link](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16))

## Key Features

- **Agentic Multimodal Reasoning**: Unifies visual and textual understanding to extract, reason, and act on information.
- **Versatile Inputs**: Natively handles text prompts, image URLs, and video URLs in a single request.
- **Controllable Reasoning**: Use the `/think` system prompt to enable detailed reasoning steps and `/no_think` for direct answers.
- **Multi-Image Understanding**: Capable of reasoning across multiple images, such as different pages of a PDF, to answer complex questions.
- **Advanced Video Analysis**: Performs dense captioning and summarization of video content.
- **Efficient Video Sampling (EVS)**: Automatically prunes redundant video frames to enable efficient long-context reasoning.
- **Hybrid Mamba-Transformer Architecture**: Delivers high accuracy with superior throughput and lower latency.

## Requirements

- NVIDIA API key ([get one here](https://build.nvidia.com))
- GPU recommended for local deployment (e.g., single H100)
