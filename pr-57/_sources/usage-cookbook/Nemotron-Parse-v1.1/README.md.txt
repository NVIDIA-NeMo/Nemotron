# Nemotron-Parse-v1.1 Notebooks

A notebook demonstrating **NVIDIA Nemotron-Parse-v1.1**, a specialized VLM for high-accuracy document ingestion.

## Overview

These notebooks provide examples of using **NVIDIA Nemotron-Parse-v1.1**, a specialized Transformer-based VLM that functions as the "ingestion backbone" for AI agents. It excels at turning messy, unstructured documents (like PDFs) into clean, structured, and agent-ready data formats, including JSON, LaTeX, and Markdown.

## Models

- **Document VLM (NIM)**: `nvidia/nemotron-parse` (Available on [NVIDIA AI Endpoints](https://build.nvidia.com/nvidia/nemotron-parse))
- **Document VLM (Hugging Face)**: TBD

## Key Features

- **Structured Data Extraction**: Converts complex PDFs into structured JSONL, tables into LaTeX, and full pages into clean Markdown.
- **High-Accuracy Parsing**: Specialized for document intelligence, achieving industry-leading performance on benchmarks like PubTables-1M.
- **Reading Order Preservation**: Intelligently extracts text, lists, and formulas in the correct semantic reading order.
- **Precise Bounding Boxes**: Returns accurate, normalized bounding boxes for every extracted element (titles, text, figures, etc.), ideal for grounding.
- **9K Token Context**: Features an extended context window for improved cross-page coherence and parsing of large, complex tables.
- **Agent-Ready Data**: Drastically reduces post-processing and hallucinations by providing reliable, structured output for RAG and agent pipelines.

## Requirements

- NVIDIA API key ([get one here](https://build.nvidia.com))
- GPU recommended for local deployment (e.g., single H100)
