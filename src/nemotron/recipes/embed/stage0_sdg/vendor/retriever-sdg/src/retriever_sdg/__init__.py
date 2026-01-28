"""
SDG Pipeline for Retriever Evaluation Dataset Generation

This package provides tools for generating synthetic queries for retriever evaluation
using the NeMo Data Designer library.
"""

__version__ = "0.1.0"

from retriever_sdg.pipeline import (
    load_positive_docs_with_modality,
    load_text_files_from_directory,
    build_qa_generation_pipeline,
    postprocess_retriever_data,
)

__all__ = [
    "load_positive_docs_with_modality",
    "load_text_files_from_directory",
    "build_qa_generation_pipeline",
    "postprocess_retriever_data",
]

