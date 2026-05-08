"""
Retriever Evaluation Dataset Generation Pipeline

This script generates synthetic queries from text files for retriever evaluation.
It takes a directory of text files (with optional subdirectories) and generates
queries without requiring the full BEIR format. The pipeline extracts facts from
documents and creates queries that require those facts to answer correctly.

Usage:
    # Generate synthetic queries (preview mode)
    retriever-sdg generate --input-dir ./my_documents \
        --output-dir ./generated_queries --preview

    # Generate synthetic queries from text files
    retriever-sdg generate --input-dir ./my_documents \
        --output-dir ./generated_queries \
        --num-pairs 7 \
        --num-files 50
    
    # Generate with custom file extensions
    retriever-sdg generate --input-dir ./my_documents \
        --output-dir ./generated_queries \
        --file-extensions .txt .md .rst
    
    # Generate with custom maximum artifacts per type (default is 2)
    retriever-sdg generate --input-dir ./my_documents \
        --output-dir ./generated_queries \
        --max-artifacts-per-type 5

    # Generate with a different model for QA generation
    retriever-sdg generate --input-dir ./my_documents \
        --output-dir ./generated_queries \
        --qa-generation-model nvidia/llama-3.1-nemotron-ultra-253b-v1

For custom models and providers (including self-hosted endpoints), see:
    https://nvidia-nemo.github.io/DataDesigner/0.3.4/concepts/models/custom-model-settings/

Requirements:
    - Set NVIDIA_API_KEY environment variable with your API key
"""

import json
import hashlib
import math
import time
import numpy as np
import pandas as pd
import sys
import yaml
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Literal, Any
from tyro.extras import SubcommandApp
from pydantic import BaseModel, Field
import re
import nltk
from nltk.tokenize import sent_tokenize

import data_designer.config as dd
from data_designer.interface import DataDesigner
from data_designer.logging import LoggerConfig, LoggingConfig, OutputConfig, configure_logging
from .deduplication import DDRetrievalDedupConfig

app = SubcommandApp()

def custom_model_config(
    artifact_extraction_model: str = "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    artifact_extraction_provider: str = "nvidia",
    qa_generation_model: str = "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    qa_generation_provider: str = "nvidia",
    quality_judge_model: str = "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    quality_judge_provider: str = "nvidia",
    embed_model: str = "nvidia/llama-3.2-nv-embedqa-1b-v2",
    embed_provider: str = "nvidia",
    max_parallel_requests_for_gen: Optional[int] = None,
) -> Tuple[List[dd.ModelConfig], Dict[str, str]]:
    """Configure the model suite for this generation job.
    
    Each pipeline role (artifact extraction, QA generation, quality judge, embedding)
    can be pointed at a different model+provider via CLI arguments. When multiple
    roles share the same model+provider, a single ModelConfig is created and the
    roles share its alias to avoid duplicate registrations.
    
    Args:
        artifact_extraction_model: Model name for artifact extraction.
        artifact_extraction_provider: Provider for artifact extraction model.
        qa_generation_model: Model name for QA generation.
        qa_generation_provider: Provider for QA generation model.
        quality_judge_model: Model name for quality judge.
        quality_judge_provider: Provider for quality judge model.
        embed_model: Model name for embeddings.
        embed_provider: Provider for embedding model.
        max_parallel_requests_for_gen: Maximum parallel requests for chat completion models.
                                       If None, uses the underlying default.
    
    Returns:
        Tuple of (model_configs, role_aliases) where role_aliases maps each role
        name ("artifact_extraction", "qa_generation", "quality_judge", "embed")
        to the ModelConfig alias it should reference.
    """
    configs: List[dd.ModelConfig] = [
        dd.ModelConfig(
            alias="embed",
            model=embed_model,
            inference_parameters=dd.EmbeddingInferenceParams(
                max_parallel_requests=8,
                extra_body={"input_type": "query", "truncate": "NONE"},
            ),
            provider=embed_provider,
        ),
    ]

    role_aliases: Dict[str, str] = {"embed": "embed"}

    # Deduplicate chat models: roles sharing the same (model, provider) reuse
    # a single ModelConfig so the framework only registers and health-checks it once.
    chat_roles = [
        ("artifact_extraction", artifact_extraction_model, artifact_extraction_provider),
        ("qa_generation", qa_generation_model, qa_generation_provider),
        ("quality_judge", quality_judge_model, quality_judge_provider),
    ]

    seen: Dict[Tuple[str, str], str] = {}  # (model, provider) -> alias

    for role_name, model, provider in chat_roles:
        key = (model, provider)
        if key not in seen:
            seen[key] = role_name
            configs.append(
                dd.ModelConfig(
                    alias=role_name,
                    model=model,
                    provider=provider,
                    inference_parameters=dd.ChatCompletionInferenceParams(
                        temperature=0.6,
                        top_p=0.95,
                        timeout=120,
                        **({"max_parallel_requests": max_parallel_requests_for_gen} if max_parallel_requests_for_gen is not None else {}),
                    ),
                )
            )
        role_aliases[role_name] = seen[key]

    return configs, role_aliases


# =============================================================================
# Multi-Document Bundling Support
# =============================================================================

def _load_multi_doc_manifest(manifest_path: Optional[Path]) -> List[List[str]]:
    """Load and parse a multi-doc manifest file.
    
    Supports JSON or YAML format:
    - List of lists: [["doc1.txt", "doc2.txt"], ["doc3.txt", "doc4.txt"]]
    - Dict with bundles key: {"bundles": [{"docs": ["doc1.txt", "doc2.txt"]}, ...]}
    
    Args:
        manifest_path: Path to the manifest file (JSON or YAML)
    
    Returns:
        List of bundles, where each bundle is a list of file paths
    """
    if not manifest_path:
        return []
    
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"Warning: Unable to read multi_doc_manifest at {manifest_path}: {exc}")
        return []
    
    data: Any = None
    try:
        data = json.loads(manifest_text)
    except json.JSONDecodeError:
        try:
            data = yaml.safe_load(manifest_text)
        except Exception as exc:
            print(f"Warning: Failed to parse multi_doc_manifest: {exc}")
            return []
    
    bundles: List[List[str]] = []
    
    # Handle dict with "bundles" key
    if isinstance(data, dict) and "bundles" in data:
        data = data["bundles"]
    
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and "docs" in entry:
                docs = entry["docs"]
            elif isinstance(entry, list):
                docs = entry
            else:
                docs = []
            clean_docs = [str(doc) for doc in docs if doc]
            if clean_docs:
                bundles.append(clean_docs)
    else:
        print("Warning: multi_doc_manifest must be a list or dict with 'bundles'")
    
    return bundles


def _is_traditional_extension(suffix: str) -> bool:
    """Check if a suffix looks like a traditional file extension.
    
    Traditional extensions are short (1-10 chars), start with a period,
    and contain only alphanumeric characters. Examples: .txt, .md, .json
    
    Non-traditional suffixes (like ".com_publication_2001-08_user-programmable")
    from filenames with periods in them should not be treated as extensions.
    
    Args:
        suffix: The file suffix from Path.suffix (includes leading period)
    
    Returns:
        True if this looks like a real file extension
    """
    if not suffix:
        return False  # Empty string means no extension
    if not suffix.startswith("."):
        return False
    ext_part = suffix[1:]  # Remove leading period
    # Traditional extensions are short and alphanumeric (allowing numbers like .mp3)
    return len(ext_part) <= 10 and ext_part.replace("_", "").isalnum()


def _file_matches_extensions(file_path: Path, file_extensions: List[str]) -> bool:
    """Check if a file matches the allowed extensions list.
    
    Handles the edge case where filenames contain periods but don't have
    traditional extensions (e.g., "research.nvidia.com_publication_...").
    
    Args:
        file_path: Path to the file
        file_extensions: List of allowed extensions (e.g., [".txt", ".md", ""])
    
    Returns:
        True if the file matches one of the allowed extensions
    """
    suffix = file_path.suffix.lower()
    
    # If suffix looks like a traditional extension, check directly
    if _is_traditional_extension(suffix):
        return suffix in file_extensions
    
    # Suffix doesn't look traditional (e.g., ".com_publication_..." or "")
    # Treat this file as having no extension
    return "" in file_extensions


def _build_bundle_id(bundle_members: List[str]) -> str:
    """Generate a unique bundle ID from member paths.
    
    Args:
        bundle_members: List of file paths in the bundle
    
    Returns:
        MD5 hash of sorted, normalized paths
    """
    if not bundle_members:
        return ""
    normalized = "||".join(sorted(str(Path(member).resolve()) for member in bundle_members))
    return hashlib.md5(normalized.encode()).hexdigest()


def _build_bundles(
    file_paths: List[Path],
    bundle_size: int = 2,
    max_docs_per_bundle: int = 3,
    manifest_bundles: Optional[List[List[str]]] = None,
    input_dir: Optional[Path] = None
) -> List[List[Path]]:
    """Build document bundles from file paths.
    
    Manifest-defined bundles take priority. Remaining documents are grouped
    sequentially according to bundle_size.
    
    Args:
        file_paths: List of file paths to bundle
        bundle_size: Number of documents per automatic bundle (default: 2)
        max_docs_per_bundle: Maximum docs allowed per bundle (default: 3)
        manifest_bundles: Pre-defined bundles from manifest file
        input_dir: Input directory for resolving relative paths in manifest
    
    Returns:
        List of bundles, where each bundle is a list of Path objects
    """
    if not file_paths:
        return []
    
    resolved_paths = [path.resolve() for path in file_paths]
    seen: set = set()
    bundles: List[List[Path]] = []
    
    # Process manifest-defined bundles first (they take priority)
    if manifest_bundles:
        for entry in manifest_bundles:
            resolved_bundle: List[Path] = []
            for raw_doc in entry:
                candidate = Path(raw_doc)
                if not candidate.is_absolute() and input_dir:
                    candidate = (input_dir / raw_doc).resolve()
                candidate = candidate.resolve()
                if candidate in resolved_paths and candidate not in seen:
                    resolved_bundle.append(candidate)
                    seen.add(candidate)
            if resolved_bundle:
                bundles.append(resolved_bundle)
    
    # Build bundles from remaining documents (sequential strategy)
    remaining = [p for p in resolved_paths if p not in seen]
    
    for start in range(0, len(remaining), bundle_size):
        bundle = remaining[start:start + bundle_size]
        if bundle:
            bundles.append(bundle)
    
    # Validate bundle sizes against max_docs_per_bundle
    for i, bundle in enumerate(bundles):
        if len(bundle) > max_docs_per_bundle:
            raise ValueError(
                f"Bundle {i} has {len(bundle)} documents, which exceeds "
                f"max_docs_per_bundle={max_docs_per_bundle}. "
                f"Either reduce the bundle size in your manifest or increase max_docs_per_bundle."
            )
    
    return [b for b in bundles if b]


def _group_chunks_by_doc(chunks: List[dict]) -> Dict[str, List[Tuple[int, dict]]]:
    """Group chunks by their document ID.
    
    Args:
        chunks: List of chunk dicts with 'doc_id' field
    
    Returns:
        Dict mapping doc_id to list of (global_index, chunk) tuples
    """
    grouped: Dict[str, List[Tuple[int, dict]]] = defaultdict(list)
    for idx, chunk in enumerate(chunks):
        doc_id = chunk.get('doc_id', 'default')
        grouped[doc_id].append((idx, chunk))
    return dict(grouped)


def _format_section_chunks(
    indexed_chunks: List[Tuple[int, dict]],
    section_number: int
) -> str:
    """Format a list of indexed chunks into a section string.
    
    Args:
        indexed_chunks: List of (global_index, chunk) tuples
        section_number: Section number for the header
    
    Returns:
        Formatted section string
    """
    section_lines = []
    for _, chunk in indexed_chunks:
        text = chunk.get('text', '').strip()
        if not text:
            continue
        
        segment_id = chunk.get('chunk_id', 1)
        doc_id = chunk.get('doc_id', '')
        
        # Include doc_id in segment info for multi-doc bundles
        start_time = "00:00:00"
        end_time = "00:00:00"
        
        if doc_id:
            segment_info = f"Segment {segment_id} [Doc: {doc_id}] ({start_time} - {end_time}): {text}"
        else:
            segment_info = f"Segment {segment_id} ({start_time} - {end_time}): {text}"
        section_lines.append(segment_info)
    
    if section_lines:
        return f"=== Section {section_number} ===\n" + '\n'.join(section_lines)
    return ""


def chunks_to_sections_sequential(chunks: List[dict], num_sections: int = 1) -> List[str]:
    """Split chunks into sections maintaining original document order.
    
    This is the default 'sequential' strategy - chunks maintain their natural
    order within each section.
    
    Args:
        chunks: List of chunk dictionaries
        num_sections: Number of sections to divide chunks into
    
    Returns:
        List of formatted section strings
    """
    total = len(chunks)
    if total == 0:
        return []
    
    section_size = max(1, total // num_sections)
    formatted_sections = []
    
    for i in range(num_sections):
        start_idx = i * section_size
        end_idx = (i + 1) * section_size if i < num_sections - 1 else total
        
        indexed_chunks = [(j, chunks[j]) for j in range(start_idx, end_idx)]
        section_text = _format_section_chunks(indexed_chunks, i + 1)
        if section_text:
            formatted_sections.append(section_text)
    
    return formatted_sections


def chunks_to_sections_doc_balanced(chunks: List[dict], num_sections: int = 1) -> List[str]:
    """Split chunks ensuring each section has proportional representation from each document.
    
    This 'doc_balanced' strategy ensures every section contains chunks from all
    documents in the bundle, useful for generating comparison questions.
    
    Args:
        chunks: List of chunk dictionaries with 'doc_id' field
        num_sections: Number of sections to divide chunks into
    
    Returns:
        List of formatted section strings
    """
    if not chunks:
        return []
    
    grouped = _group_chunks_by_doc(chunks)
    
    # If only one document, fall back to sequential
    if len(grouped) <= 1:
        return chunks_to_sections_sequential(chunks, num_sections)
    
    # Calculate chunk sizes per document per section
    chunk_sizes = {
        doc_id: max(1, math.ceil(len(entries) / num_sections))
        for doc_id, entries in grouped.items()
    }
    
    sections: List[List[Tuple[int, dict]]] = []
    for part_idx in range(num_sections):
        part_entries: List[Tuple[int, dict]] = []
        for doc_id, entries in grouped.items():
            chunk_size = chunk_sizes[doc_id]
            start = part_idx * chunk_size
            end = min(len(entries), start + chunk_size)
            if start < len(entries):
                part_entries.extend(entries[start:end])
        if part_entries:
            sections.append(part_entries)
    
    formatted_sections = []
    for i, indexed_chunks in enumerate(sections):
        section_text = _format_section_chunks(indexed_chunks, i + 1)
        if section_text:
            formatted_sections.append(section_text)
    
    return formatted_sections


def chunks_to_sections_interleaved(chunks: List[dict], num_sections: int = 1) -> List[str]:
    """Split chunks with round-robin interleaving from each document.
    
    This 'interleaved' strategy alternates chunks from different documents
    before splitting into sections, maximizing cross-document connections.
    
    Args:
        chunks: List of chunk dictionaries with 'doc_id' field
        num_sections: Number of sections to divide chunks into
    
    Returns:
        List of formatted section strings
    """
    if not chunks:
        return []
    
    grouped = _group_chunks_by_doc(chunks)
    
    # If only one document, fall back to sequential
    if len(grouped) <= 1:
        return chunks_to_sections_sequential(chunks, num_sections)
    
    # Create deques for round-robin iteration
    doc_iterators = {doc_id: deque(entries) for doc_id, entries in grouped.items()}
    doc_order = list(grouped.keys())
    interleaved: List[Tuple[int, dict]] = []
    
    # Round-robin interleave
    while True:
        added = False
        for doc_id in doc_order:
            doc_queue = doc_iterators[doc_id]
            if doc_queue:
                interleaved.append(doc_queue.popleft())
                added = True
        if not added:
            break
    
    if not interleaved:
        return []
    
    # Split interleaved chunks into sections
    total = len(interleaved)
    section_size = max(1, total // num_sections)
    formatted_sections = []
    
    for i in range(num_sections):
        start_idx = i * section_size
        end_idx = (i + 1) * section_size if i < num_sections - 1 else total
        indexed_chunks = interleaved[start_idx:end_idx]
        section_text = _format_section_chunks(indexed_chunks, i + 1)
        if section_text:
            formatted_sections.append(section_text)
    
    return formatted_sections


def chunks_to_sections_with_strategy(
    chunks: List[dict],
    num_sections: int = 1,
    strategy: Literal["sequential", "doc_balanced", "interleaved"] = "sequential"
) -> List[str]:
    """Split chunks into sections using the specified strategy.
    
    Args:
        chunks: List of chunk dictionaries
        num_sections: Number of sections to divide chunks into
        strategy: One of 'sequential', 'doc_balanced', or 'interleaved'
    
    Returns:
        List of formatted section strings
    """
    if strategy == "doc_balanced":
        return chunks_to_sections_doc_balanced(chunks, num_sections)
    elif strategy == "interleaved":
        return chunks_to_sections_interleaved(chunks, num_sections)
    else:  # sequential (default)
        return chunks_to_sections_sequential(chunks, num_sections)


# =============================================================================
# Text Processing Functions
# =============================================================================

def text_to_sentence_chunks(
    text: str,
    sentences_per_chunk: int = 5,
    doc_id: Optional[str] = None,
    doc_path: Optional[str] = None,
    chunk_id_offset: int = 0
) -> List[dict]:
    """
    Chunk text by sentences for more natural boundaries.
    
    Args:
        text: The input text to chunk
        sentences_per_chunk: Number of sentences per chunk (default: 5)
        doc_id: Optional document identifier for multi-doc bundles
        doc_path: Optional document path for multi-doc bundles
        chunk_id_offset: Offset for chunk IDs when aggregating multiple documents
    
    Returns:
        List of chunk dictionaries containing text, start/end positions, and metadata
    """
    # Ensure NLTK punkt tokenizer is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    
    # First split by paragraphs (multiple newlines or return characters)
    # This preserves paragraph boundaries in the original text
    paragraphs = re.split(r'\n\s*\n+', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Then use NLTK to properly tokenize sentences within each paragraph
    sentences = []
    for paragraph in paragraphs:
        paragraph_sentences = sent_tokenize(paragraph)
        sentences.extend(paragraph_sentences)

    chunks = []
    word_position = 0
    doc_chunk_index = 0  # Track chunk index within this document

    for i in range(0, len(sentences), sentences_per_chunk):
        chunk_sentences = sentences[i:i + sentences_per_chunk]
        chunk_text = '. '.join(chunk_sentences)
        if chunk_text and not chunk_text.endswith('.'):
            chunk_text += '.'

        chunk_words = chunk_text.split()
        start_word_pos = word_position
        end_word_pos = word_position + len(chunk_words)
        word_position = end_word_pos
        doc_chunk_index += 1

        chunk_data = {
            'text': chunk_text,
            'start': start_word_pos,
            'end': end_word_pos,
            'sentence_count': len(chunk_sentences),
            'word_count': len(chunk_words),
            'chunk_id': chunk_id_offset + len(chunks) + 1,  # Global chunk ID
            'doc_chunk_index': doc_chunk_index,  # Chunk index within document
        }
        
        # Add document tracking for multi-doc bundles
        if doc_id is not None:
            chunk_data['doc_id'] = doc_id
        if doc_path is not None:
            chunk_data['doc_path'] = doc_path
        
        chunks.append(chunk_data)

    return chunks


def chunks_to_sections_structured(
    chunks: List[dict],
    num_sections: int = 1,
    strategy: Literal["sequential", "doc_balanced", "interleaved"] = "sequential"
) -> List[str]:
    """
    Split chunks into sections with structured formatting.
    Each section includes a header and formatted segment information with IDs and timestamps.
    
    Args:
        chunks: List of chunk dictionaries
        num_sections: Number of sections to divide chunks into (default: 1)
        strategy: Splitting strategy - 'sequential', 'doc_balanced', or 'interleaved'
    
    Returns:
        List of formatted section strings, each with header and segment details
    """
    return chunks_to_sections_with_strategy(chunks, num_sections, strategy)


def load_text_files_from_directory(
    input_dir: Path,
    file_extensions: List[str] = [".txt", ".md", ".text", ""],
    min_text_length: int = 0,
    sentences_per_chunk: int = 5,
    num_sections: int = 1,
    num_files: Optional[int] = None,
    multi_doc: bool = False,
    bundle_size: int = 2,
    bundle_strategy: Literal["sequential", "doc_balanced", "interleaved"] = "sequential",
    max_docs_per_bundle: int = 3,
    multi_doc_manifest: Optional[Path] = None
) -> pd.DataFrame:
    """Load text files from a directory (including subdirectories) into a DataFrame.
    
    Supports two modes:
    - Single-doc mode (default): Each file becomes one DataFrame row
    - Multi-doc mode: Files are grouped into bundles, each bundle becomes one row
    
    Args:
        input_dir: Root directory containing text files
        file_extensions: List of file extensions to include (default: .txt, .md, .text, and files with no extension)
        min_text_length: Minimum text length for documents to include (default: 0)
        sentences_per_chunk: Number of sentences per chunk for text chunking (default: 5)
        num_sections: Number of sections to divide chunks into (default: 1)
        num_files: Maximum number of files to process (default: None, processes all files)
        multi_doc: Enable multi-document bundling (default: False)
        bundle_size: Number of documents per bundle when multi_doc=True (default: 2)
        bundle_strategy: Segment splitting strategy - 'sequential', 'doc_balanced', or 'interleaved'
        max_docs_per_bundle: Maximum documents per bundle (default: 3)
        multi_doc_manifest: Optional path to manifest file defining explicit bundles
    
    Returns:
        DataFrame with columns: file_name, text, chunks, sections_structured
        Additional columns for multi-doc: bundle_id, bundle_members, is_multi_doc
    """
    # Collect all valid file paths first
    all_file_paths: List[Path] = []
    
    for file_path in input_dir.rglob("*"):
        if num_files is not None and len(all_file_paths) >= num_files:
            break
        
        if file_path.is_file() and _file_matches_extensions(file_path, file_extensions):
            try:
                # Read file to check length
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if min_text_length > 0 and len(content) < min_text_length:
                    continue
                
                all_file_paths.append(file_path)
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
                continue
    
    if not all_file_paths:
        raise ValueError(f"No text files found in {input_dir} with extensions {file_extensions}")
    
    documents = []
    
    # Resolve input_dir for consistent path handling
    resolved_input_dir = input_dir.resolve()
    
    if multi_doc:
        # Multi-doc mode: Group files into bundles
        manifest_bundles = _load_multi_doc_manifest(multi_doc_manifest)
        bundles = _build_bundles(
            all_file_paths,
            bundle_size=bundle_size,
            max_docs_per_bundle=max_docs_per_bundle,
            manifest_bundles=manifest_bundles,
            input_dir=resolved_input_dir
        )
        
        print(f"Multi-doc mode: Created {len(bundles)} bundles from {len(all_file_paths)} files")
        
        for bundle in bundles:
            bundle_texts = []
            bundle_chunks = []
            bundle_members = []
            chunk_id_offset = 0
            
            for doc_idx, file_path in enumerate(bundle):
                relative_path = file_path.relative_to(resolved_input_dir)
                doc_id = str(relative_path)
                bundle_members.append(doc_id)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    bundle_texts.append(content)
                    
                    # Create chunks with document tracking
                    doc_chunks = text_to_sentence_chunks(
                        content,
                        sentences_per_chunk=sentences_per_chunk,
                        doc_id=doc_id,
                        doc_path=str(file_path),
                        chunk_id_offset=chunk_id_offset
                    )
                    
                    bundle_chunks.extend(doc_chunks)
                    chunk_id_offset += len(doc_chunks)
                    
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")
                    continue
            
            if not bundle_chunks:
                continue
            
            # Combine texts with document separators
            combined_text = "\n\n=== Document Boundary ===\n\n".join(bundle_texts)
            
            # Create sections using the specified strategy
            sections_structured = chunks_to_sections_structured(
                bundle_chunks,
                num_sections=num_sections,
                strategy=bundle_strategy
            )
            
            # Generate bundle ID
            bundle_id = _build_bundle_id(bundle_members)
            
            documents.append({
                'file_name': bundle_members,  # All files in the bundle
                'text': combined_text,
                'chunks': bundle_chunks,
                'sections_structured': sections_structured,
                'bundle_id': bundle_id,
                'bundle_members': bundle_members,
                'is_multi_doc': True
            })
    
    else:
        # Single-doc mode: Each file becomes one row
        for file_path in all_file_paths:
            relative_path = file_path.relative_to(input_dir)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create chunks from the text
                chunks = text_to_sentence_chunks(content, sentences_per_chunk=sentences_per_chunk)
                
                # Create structured formatted sections
                sections_structured = chunks_to_sections_structured(
                    chunks,
                    num_sections=num_sections,
                    strategy=bundle_strategy  # Can still use strategy for single-doc
                )
                
                documents.append({
                    'file_name': [str(relative_path)],
                    'text': content,
                    'chunks': chunks,
                    'sections_structured': sections_structured,
                    'bundle_id': '',
                    'bundle_members': [str(relative_path)],
                    'is_multi_doc': False
                })
            except Exception as e:
                print(f"Warning: Could not read {relative_path}: {e}")
                continue
    
    if not documents:
        raise ValueError(f"No valid documents created from {input_dir}")
    
    df = pd.DataFrame(documents)
    
    if multi_doc:
        print(f"Created {len(df)} bundles from {len(all_file_paths)} files")
        avg_docs_per_bundle = sum(len(m) for m in df['bundle_members']) / len(df) if len(df) > 0 else 0
        print(f"Average documents per bundle: {avg_docs_per_bundle:.1f}")
    else:
        print(f"Loaded {len(df)} text files from {input_dir}")
    
    if min_text_length > 0:
        print(f"Filtered to documents with at least {min_text_length} characters")
    
    # Print chunking statistics
    total_chunks = sum(len(chunks) for chunks in df['chunks'])
    avg_chunks_per_row = total_chunks / len(df) if len(df) > 0 else 0
    row_type = "bundle" if multi_doc else "document"
    print(f"Created {total_chunks} total chunks ({avg_chunks_per_row:.1f} chunks per {row_type})")
    
    # Print section statistics
    total_sections = sum(len(sections) for sections in df['sections_structured'])
    avg_sections_per_row = total_sections / len(df) if len(df) > 0 else 0
    avg_chunks_per_section = total_chunks / total_sections if total_sections > 0 else 0
    print(f"Organized into {total_sections} sections ({avg_sections_per_row:.1f} sections per {row_type}, {avg_chunks_per_section:.1f} chunks per section)")
    print(f"Bundle strategy: {bundle_strategy}")
    
    return df


def load_positive_docs_with_modality(
    test_tsv_path: Path,
    corpus_jsonl_path: Path,
    split_json_path: Path,
    min_text_length: int = 0
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load positive documents and map them to their modalities.
    Returns:
        - DataFrame with positive docs and their modalities
        - Dict mapping doc_id to modality
    """
    # Load test.tsv
    qrels_df = pd.read_csv(test_tsv_path, sep='\t')

    # Load split.json
    with open(split_json_path, encoding='utf-8') as f:
        splits = json.load(f)

    # Create query to modality mapping
    query_to_modality = {}
    for modality, query_ids in splits.items():
        for query_id in query_ids:
            query_to_modality[query_id] = modality

    # Map corpus-id to modality through queries
    doc_to_modality = defaultdict(set)
    for _, row in qrels_df.iterrows():
        query_id = row['query-id']
        corpus_id = row['corpus-id ']  # Note: column has trailing space
        if query_id in query_to_modality:
            doc_to_modality[corpus_id].add(query_to_modality[query_id])

    # Resolve docs that appear in multiple modalities by taking the most frequent
    doc_to_modality_final = {}
    for doc_id, modalities in doc_to_modality.items():
        if len(modalities) == 1:
            doc_to_modality_final[doc_id] = list(modalities)[0]
        else:
            # Count occurrences per modality
            modality_counts = defaultdict(int)
            for _, row in qrels_df[qrels_df['corpus-id '] == doc_id].iterrows():
                query_id = row['query-id']
                if query_id in query_to_modality:
                    modality_counts[query_to_modality[query_id]] += 1
            # Take the most frequent
            doc_to_modality_final[doc_id] = max(modality_counts, key=modality_counts.get)

    # Get unique group IDs we need (test.tsv uses group_id, not _id)
    unique_group_ids = set(doc_to_modality_final.keys())

    # Load corpus documents by group_id
    # Note: Multiple docs can have the same group_id, we'll take the first one per group
    corpus_docs_by_group = {}
    with open(corpus_jsonl_path, encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            group_id = doc.get('group_id', doc['_id'])  # Use _id as fallback if no group_id
            if group_id in unique_group_ids and group_id not in corpus_docs_by_group:
                corpus_docs_by_group[group_id] = doc

    # Create DataFrame with positive docs and their modalities
    positive_docs_data = []
    for group_id, modality in doc_to_modality_final.items():
        if group_id in corpus_docs_by_group:
            doc = corpus_docs_by_group[group_id]
            positive_docs_data.append({
                'doc_id': doc['_id'],  # Store the actual doc _id
                'text': doc['text'],
                'title': doc.get('title', ''),
                'modality': modality,
                'group_id': group_id  # Store the group_id (which is what test.tsv references)
            })

    positive_docs_df = pd.DataFrame(positive_docs_data)

    # Filter by minimum text length if specified
    if min_text_length > 0 and len(positive_docs_df) > 0:
        original_count = len(positive_docs_df)
        positive_docs_df = positive_docs_df[positive_docs_df['text'].str.len() >= min_text_length]
        filtered_count = original_count - len(positive_docs_df)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} documents shorter than {min_text_length} characters")

    return positive_docs_df, doc_to_modality_final


def build_qa_generation_pipeline(
        seed_dataset: pd.DataFrame,
        start_index: int = 0,
        end_index: int = 199,
        max_artifacts_per_type: int = 2,
        num_pairs: int = 5,
        query_counts: Optional[Dict[str, int]] = None,
        min_hops: int = 2,
        max_hops: int = 3,
        reasoning_counts: Optional[Dict[str, int]] = None,
        min_complexity: int = 4,
        max_parallel_requests_for_gen: Optional[int] = None,
        artifact_extraction_model: str = "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        artifact_extraction_provider: str = "nvidia",
        qa_generation_model: str = "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        qa_generation_provider: str = "nvidia",
        quality_judge_model: str = "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        quality_judge_provider: str = "nvidia",
        embed_model: str = "nvidia/llama-3.2-nv-embedqa-1b-v2",
        embed_provider: str = "nvidia",
    ) -> dd.DataDesignerConfigBuilder:
    """Create a simple query generation config for text files.
    
    This is a simplified version that works with plain text files without modality information.
    
    Args:
        seed_dataset: DataFrame with 'file_name' and 'text' columns
        data_designer: DataDesigner instance
        start_index: Start index (inclusive) for the index range selection (default: 0)
        end_index: End index (inclusive) for the index range selection (default: 199)
        max_artifacts_per_type: Maximum number of artifacts to extract per type (default: 2)
        num_pairs: Number of question-answer pairs to generate (default: 5)
        query_counts: Dictionary with counts for query types: multi_hop, structural, contextual (default: None)
        min_hops: Minimum number of hops for multi-hop questions (default: 2)
        max_hops: Maximum number of hops for multi-hop questions (default: 3)
        reasoning_counts: Dictionary with counts for reasoning types (default: None)
        min_complexity: Minimum complexity level for questions (default: 3)
        max_parallel_requests_for_gen: Maximum parallel requests for generation models (default: None, uses underlying default)
        artifact_extraction_model: Model name for artifact extraction.
        artifact_extraction_provider: Provider for artifact extraction model.
        qa_generation_model: Model name for QA generation.
        qa_generation_provider: Provider for QA generation model.
        quality_judge_model: Model name for quality judge.
        quality_judge_provider: Provider for quality judge model.
        embed_model: Model name for embeddings.
        embed_provider: Provider for embedding model.
    """
    # Set default values for query_counts and reasoning_counts
    if query_counts is None:
        query_counts = {"multi_hop": 3, "structural": 2, "contextual": 2}

    if reasoning_counts is None:
        reasoning_counts = {
            "factual": 1,
            "relational": 1,
            "inferential": 1,
            "temporal": 1,
            "procedural": 1,
            "causal": 1,
            "visual": 1,
        }

    # Create config builder with model configs (deduplicated)
    model_configs, role_aliases = custom_model_config(
        artifact_extraction_model=artifact_extraction_model,
        artifact_extraction_provider=artifact_extraction_provider,
        qa_generation_model=qa_generation_model,
        qa_generation_provider=qa_generation_provider,
        quality_judge_model=quality_judge_model,
        quality_judge_provider=quality_judge_provider,
        embed_model=embed_model,
        embed_provider=embed_provider,
        max_parallel_requests_for_gen=max_parallel_requests_for_gen,
    )
    config_builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)

    df_seed_source = dd.DataFrameSeedSource(df=seed_dataset)

    # Add seed dataset with ordered sampling
    config_builder.with_seed_dataset(
        df_seed_source,
        sampling_strategy=dd.SamplingStrategy.ORDERED,
        selection_strategy=dd.IndexRange(start=start_index, end=end_index),
    )

    class ArtifactItem(BaseModel):
        """A single artifact item with text, description, and importance"""
        text: str = Field(description="The artifact text or name")
        description: str = Field(
            description="Detailed description of the artifact")
        importance: str = Field(description="Why this artifact is important")

    class DocumentArtifacts(BaseModel):
        """Semantic artifacts extracted from the document"""
        key_concepts: list[ArtifactItem] = Field(
            default_factory=list, description="Key concepts in the document")
        relationships: list[ArtifactItem] = Field(
            default_factory=list, description="Relationships between concepts")
        themes: list[ArtifactItem] = Field(default_factory=list,
                                           description="Main themes")
        entities: list[ArtifactItem] = Field(default_factory=list,
                                             description="Entities mentioned")
        processes: list[ArtifactItem] = Field(
            default_factory=list, description="Processes described")
        insights: list[ArtifactItem] = Field(default_factory=list,
                                             description="Key insights")
        technical_terms: list[ArtifactItem] = Field(
            default_factory=list, description="Technical terms")
        contextual_factors: list[ArtifactItem] = Field(
            default_factory=list, description="Contextual factors")

    # Extract semantic artifacts from document
    config_builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="document_artifacts",
            system_prompt=
            "You are an expert at analyzing documents and extracting semantic artifacts.",
            prompt=f"""\
Analyze the following content and extract semantic artifacts that would be valuable for generating high-quality question-answer pairs.

Note: The content may contain multiple documents bundled together (separated by "=== Document Boundary ==="). 
If multiple documents are present, identify cross-document relationships and connections.

CONTENT:
{{{{ text }}}}

ARTIFACT TYPES TO EXTRACT:
- key_concepts: Core ideas and concepts discussed in the document(s)
- relationships: Connections and relationships between different concepts (including cross-document relationships)
- themes: Overarching themes and topics
- entities: Specific entities, people, organizations, or items mentioned
- processes: Processes, workflows, or procedures described
- insights: Key insights, conclusions, or findings
- technical_terms: Technical terminology and specialized vocabulary
- contextual_factors: Contextual information that provides background

INSTRUCTIONS:
1. Extract up to {max_artifacts_per_type} artifacts for each relevant type
2. Focus on the most significant and informative elements
3. Provide clear, concise descriptions for each artifact
4. Include context about why each artifact is important
5. Ensure artifacts are specific and actionable for Q&A generation
6. For multi-document bundles, pay special attention to relationships and comparisons between documents
""",
            output_format=DocumentArtifacts,
            model_alias=role_aliases["artifact_extraction"],
        ))

    # Define data models for hard question-answer generation
    class HopContext(BaseModel):
        """Context for a single hop in a multi-hop question"""
        hop_number: int = Field(description="The hop number (1-indexed)")
        segment_ids: List[int] = Field(description="Segment IDs for this hop")
        summary: str = Field(
            description="Summary of the supporting segments for this hop")

    class QuestionAnswerPair(BaseModel):
        """A single question-answer pair with metadata"""
        question: str = Field(
            description=
            "The question requiring understanding of contexts without explicitly referencing them"
        )
        answer: str = Field(
            description=
            "Comprehensive answer from the contexts without explicitly referencing them"
        )
        question_complexity: int = Field(
            description="Numeric score from min_complexity to 5")
        query_type: Literal["multi_hop", "structural", "contextual"] = Field(
            description=
            "Type of query, one of multi_hop, structural, or contextual")
        reasoning_type: Literal[
            "factual", "relational", "inferential", "temporal", "procedural",
            "visual", "causal"] = Field(
                description=
                "Type of reasoning required, one of factual, relational, inferential, temporal, procedural, visual, or causal"
            )
        segment_ids: List[int] = Field(
            description=
            "List of segment IDs that are source material for this question")
        hop_count: int = Field(
            description=
            "Number of hops (min_hops to max_hops) for multi_hop questions, or 1 for non-multi-hop"
        )
        hop_contexts: List[HopContext] = Field(
            description="Array of hop detail objects")

    class QuestionAnswerPairs(BaseModel):
        """Collection of question-answer pairs"""
        pairs: List[QuestionAnswerPair] = Field(
            description="List of question-answer pairs")

    # Generate question-answer pairs based on document artifacts and context
    config_builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="qa_generation",
            system_prompt=
            "You are an expert at extracting question and answer pairs from provided context/transcript/segments.",
            prompt="""\
You are an expert at extracting question and answer pairs from provided context/transcript/segments.

<document_facts_block>:
{{%- if document_artifacts.key_concepts %}}
<key_concepts>
{{%- for item in document_artifacts.key_concepts %}}
- {{{{ item.text }}}}: {{{{ item.description }}}}
{{%- endfor %}}
</key_concepts>
{{%- endif %}}

{{%- if document_artifacts.relationships %}}
<relationships>
{{%- for item in document_artifacts.relationships %}}
- {{{{ item.text }}}}: {{{{ item.description }}}}
{{%- endfor %}}
</relationships>
{{%- endif %}}

{{%- if document_artifacts.themes %}}
<themes>
{{%- for item in document_artifacts.themes %}}
- {{{{ item.text }}}}: {{{{ item.description }}}}
{{%- endfor %}}
</themes>
{{%- endif %}}

{{%- if document_artifacts.entities %}}
<entities>
{{%- for item in document_artifacts.entities %}}
- {{{{ item.text }}}}: {{{{ item.description }}}}
{{%- endfor %}}
</entities>
{{%- endif %}}

{{%- if document_artifacts.processes %}}
<processes>
{{%- for item in document_artifacts.processes %}}
- {{{{ item.text }}}}: {{{{ item.description }}}}
{{%- endfor %}}
</processes>
{{%- endif %}}

{{%- if document_artifacts.insights %}}
<insights>
{{%- for item in document_artifacts.insights %}}
- {{{{ item.text }}}}: {{{{ item.description }}}}
{{%- endfor %}}
</insights>
{{%- endif %}}

{{%- if document_artifacts.technical_terms %}}
<technical_terms>
{{%- for item in document_artifacts.technical_terms %}}
- {{{{ item.text }}}}: {{{{ item.description }}}}
{{%- endfor %}}
</technical_terms>
{{%- endif %}}

{{%- if document_artifacts.contextual_factors %}}
<contextual_factors>
{{%- for item in document_artifacts.contextual_factors %}}
- {{{{ item.text }}}}: {{{{ item.description }}}}
{{%- endfor %}}
</contextual_factors>
{{%- endif %}}
</document_facts_block>

<context_block>:
{{%- for section in sections_structured %}}
{{{{ section }}}}

{{%- endfor %}}
</context_block>

Guidelines:
1. Generate questions with varying complexity levels between 1 (simple) and 5 (complex):
   - All questions MUST require understanding connections between different parts of the context/transcript/segments
   - Questions should test deep understanding, not simple facts
   - Do not mention the existence of a context/transcript in the generated question like "in the transcript", "from the given context", or "in Segment 148". Produce a natural, standalone question.
   - Only use facts present in the provided context/transcript; if missing, say you cannot generate a question.
   - Example: "How does the speaker's initial explanation of X relate to the later implementation of Y?"

2. Question Types to Generate (for the "query_type" field - ONLY these 3 values allowed):
   - "multi_hop" ({query_counts_multi_hop} questions): Connect {min_hops}-{max_hops} separated segments
   - "structural" ({query_counts_structural} questions): Focus on relationships between concepts
   - "contextual" ({query_counts_contextual} questions): Require surrounding context to understand
   - Use the cross-part context snippets to connect evidence that lives outside the current transcript section

3. Reasoning Types to Include (for the "reasoning_type" field - ONLY these 7 values allowed):
   - "factual" ({reasoning_counts_factual} questions): Ask for complex facts that require synthesizing multiple pieces of information (NOT simple lookups)
   - "relational" ({reasoning_counts_relational} questions): Ask how data points compare or correlate across different segments
   - "inferential" ({reasoning_counts_inferential} questions): Ask about conclusions or implications requiring synthesis
   - "temporal" ({reasoning_counts_temporal} questions): Ask about changes or events over time across segments
   - "procedural" ({reasoning_counts_procedural} questions): Ask about complex multi-step processes or guidelines
   - "visual" ({reasoning_counts_visual} questions): Ask about visual details requiring cross-reference
   - "causal" ({reasoning_counts_causal} questions): Ask about cause-effect chains spanning segments
   
   Example COMPLEX questions by reasoning type:
   - Factual: "What is the total combined budget allocation across all departmental initiatives mentioned, and how does it relate to the overall fiscal year target?"
   - Relational: "How does the performance metric achieved in Q2 compare to both the initial baseline and the revised targets that were set?"
   - Inferential: "Based on the challenges outlined and the proposed solutions, what unstated assumptions underlie the strategic pivot?"
   - Temporal: "How did the implementation timeline evolve from the initial proposal through the mid-year review to the final execution phase?"
   - Procedural: "What is the complete approval workflow including standard requirements, exceptions, and escalation processes?"
   - Visual: "How do the visual elements presented relate to the verbal descriptions provided, and what discrepancies exist between them?"
   - Causal: "What chain of events, starting from the initial decision, led through various complications to the final outcome?"

4. IMPORTANT - Orthogonal Distributions (query_type and reasoning_type are SEPARATE fields):
   - Each question must have BOTH a query_type (multi_hop/structural/contextual) AND a reasoning_type (factual/relational/inferential/temporal/procedural/visual/causal)
   - These are TWO DIFFERENT fields - do NOT put reasoning types in the query_type field!
   - For example: A question can be query_type="multi_hop" with reasoning_type="procedural"
   - Ensure the final distribution matches both specified percentages

5. **IMPORTANT - Segment Identification**:
   - The content below contains segments formatted as "Segment N (HH:MM:SS - HH:MM:SS): text" or "Segment N [Doc: doc_id] (HH:MM:SS - HH:MM:SS): text" where N starts from 1
   - The "[Doc: doc_id]" tag indicates which document the segment belongs to (for multi-document bundles)
   - For each question-answer pair you generate, identify ALL segment numbers FROM which the question is derived
   - These segments are the source material that should be retrieved when someone asks this question
   - Record these segment numbers in the "segment_ids" field as a list of integers (e.g., [1, 4, 8])
   - For multi-document bundles, prefer questions that span multiple documents to maximize cross-document reasoning
   - For multi-hop questions:
     * The top-level "segment_ids" should be the UNION of all segment IDs across all hops
     * Each hop in "hop_contexts" should specify its own "segment_ids" list
     * Example: If hop 1 uses [1, 3] and hop 2 uses [6, 8], then top-level segment_ids should be [1, 3, 6, 8]
     * For multi-document bundles, try to have different hops reference different documents

6. For Each Question:
   - Must have complexity level {min_complexity} or higher
   - Generate the question FROM the identified segments (these segments are the source material)
   - Multi-hop questions must specify hop_count ({min_hops}-{max_hops})
   - Provide hop_contexts: a list where each hop includes "hop_number", "segment_ids" (the source segments for this hop), and "summary" (a concise summary describing the supporting segments).

7. Generate {num_pairs} distinct question and answer pairs.

The output should be a JSON object with a "pairs" field containing an array of {num_pairs} objects, where each object contains:
  - "question": the question, requiring understanding of the contexts/transcripts/segments without explicitly referencing the context/transcript/segments in the question
  - "answer": comprehensive answer from the contexts/transcripts/segments without explicitly referencing the context/transcript/segments in the answer
  - "question_complexity": numeric score {min_complexity}-5
  - "query_type": MUST be exactly one of these three values: "multi_hop", "structural", or "contextual" (NO other values allowed - do NOT use reasoning types here)
  - "reasoning_type": MUST be exactly one of these seven values: "factual", "relational", "inferential", "temporal", "procedural", "visual", or "causal" (this is DIFFERENT from query_type)
  - "segment_ids": list of segment numbers (e.g., [1, 4, 8]) that are the source material for this question (these should be retrieved when the question is asked)
  - "hop_count": number of hops ({min_hops}-{max_hops}) for multi_hop questions, or 1 for non-multi-hop questions
  - "hop_contexts": array of hop detail objects with "hop_number", "segment_ids", "summary"

CRITICAL: "query_type" and "reasoning_type" are TWO SEPARATE FIELDS with different allowed values. Do NOT mix them up:
  - query_type can ONLY be: "multi_hop", "structural", "contextual"
  - reasoning_type can ONLY be: "factual", "relational", "inferential", "temporal", "procedural", "visual", "causal"
""".format(query_counts_multi_hop=query_counts.get("multi_hop", 0),
           query_counts_structural=query_counts.get("structural", 0),
           query_counts_contextual=query_counts.get("contextual", 0),
           reasoning_counts_factual=reasoning_counts.get("factual", 0),
           reasoning_counts_relational=reasoning_counts.get("relational", 0),
           reasoning_counts_inferential=reasoning_counts.get("inferential", 0),
           reasoning_counts_temporal=reasoning_counts.get("temporal", 0),
           reasoning_counts_procedural=reasoning_counts.get("procedural", 0),
           reasoning_counts_visual=reasoning_counts.get("visual", 0),
           reasoning_counts_causal=reasoning_counts.get("causal", 0),
           min_hops=min_hops,
           max_hops=max_hops,
           min_complexity=min_complexity,
           num_pairs=num_pairs),
            output_format=QuestionAnswerPairs,
            model_alias=role_aliases["qa_generation"],
        ))

    config_builder.add_column(
        DDRetrievalDedupConfig(
            name="deduplicated_qa_pairs",
            qa_pairs_column="qa_generation",
            embedding_alias="embed",
            dedupe_similarity_threshold=0.9,
        )
    )

    # Add QA quality evaluation for each pair
    # First, define the evaluation data model
    class QAEvaluationCriterion(BaseModel):
        """Evaluation criterion with score and justification"""
        score: int = Field(description="Score from 1-10")
        justification: str = Field(
            description="Brief justification for the score")

    class QAOverallEvaluation(BaseModel):
        """Overall evaluation with score and assessment"""
        score: float = Field(description="Overall score from 1-10")
        assessment: str = Field(description="Final assessment of the QA pair")

    class QAEvaluation(BaseModel):
        """Evaluation of a single QA pair"""
        relevance: QAEvaluationCriterion = Field(
            description="Relevance of question to context")
        accuracy: QAEvaluationCriterion = Field(
            description="Factual accuracy of answer")
        context_support: QAEvaluationCriterion = Field(
            description="How well answer is supported by context")
        clarity: QAEvaluationCriterion = Field(
            description="Clarity and unambiguity of question")
        overall: QAOverallEvaluation = Field(description="Overall evaluation")
        improvements: str = Field(
            description="Suggestions for improving this QA pair")

    class QAPairEvaluations(BaseModel):
        """Evaluations for all QA pairs in a document"""
        evaluations: List[QAEvaluation] = Field(
            description=
            "List of evaluations, one per QA pair, in the same order as the QA pairs"
        )

    # LLM column to evaluate each QA pair
    # Extract just question and answer (no context for now)
    config_builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="qa_evaluations",
            system_prompt="You are an expert evaluator of question-answer pairs.",
            prompt="""\
You are an expert evaluator of question-answer pairs.

You will evaluate multiple question-answer pairs from a document.

{% for qa_pair in deduplicated_qa_pairs %}
=== QA Pair {{ loop.index }} ===

QUESTION: {{ qa_pair.question }}

ANSWER: {{ qa_pair.answer }}

CONTEXT (Relevant Segment IDs): {{ qa_pair.segment_ids }}

{% endfor %}

<segments>
{% for chunk in chunks %}
- Segment {{ chunk.chunk_id }}: {{ chunk.text }}
{% endfor %}
</segments>

Evaluate EACH of the {{ deduplicated_qa_pairs | length }} QA pairs above.
""",
            output_format=QAPairEvaluations,
            model_alias=role_aliases["quality_judge"],
        ))

    return config_builder


def postprocess_retriever_data(
    generated_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    """Post-process generated data into BEIR format.
    
    Flattens the list of question-answer pairs from deduplicated_qa_pairs column.
    Adds metadata fields: query_type, reasoning_type, question_complexity, hop_count, etc.
    
    Returns:
        - queries_df: DataFrame for queries.jsonl
        - qrels_df: DataFrame for test.tsv
        - splits: Dict for split.json
    """
    print(f"Processing {len(generated_df)} generated records...")

    queries_data = []
    qrels_data = []
    splits = defaultdict(list)

    query_counter = 0

    # Track reasoning type distribution
    reasoning_types = []
    query_types = []

    for _, row in generated_df.iterrows():
        # Check if we have the required columns
        if 'file_name' not in row:
            print("Warning: Skipping row without file_name")
            continue

        file_name = row['file_name']

        # Get the deduplicated_qa_pairs (fallback to qa_generation for backward compatibility)
        if 'deduplicated_qa_pairs' in row and row['deduplicated_qa_pairs'] is not None:
            qa_pairs = row['deduplicated_qa_pairs']
        elif 'qa_generation' in row:
            qa_generation = row.get('qa_generation')
            # Handle case where qa_generation might be None
            if qa_generation is None:
                print(f"Warning: Skipping {file_name} - qa_generation is None")
                print(f"Full record for debugging: {row.to_dict()}")
                continue

            # Extract the pairs list from the qa_generation object
            if isinstance(qa_generation, dict):
                qa_pairs = qa_generation.get('pairs', [])
            else:
                # Handle Pydantic model objects
                qa_pairs = getattr(qa_generation, 'pairs', [])
        else:
            print(f"Warning: Skipping {file_name} - no qa_generation or deduplicated_qa_pairs found")
            print(f"Full record for debugging: {row.to_dict()}")
            continue

        # Convert numpy array to list if needed (parquet stores as numpy arrays)
        if isinstance(qa_pairs, np.ndarray):
            qa_pairs = qa_pairs.tolist()

        # Handle case where pairs is not a list or is empty
        if not isinstance(qa_pairs, (list, np.ndarray)) or len(qa_pairs) == 0:
            print(f"Warning: Skipping {file_name} - no valid pairs found")
            print(f"Type of qa_pairs: {type(qa_pairs)}")
            print(f"Content of qa_pairs: {qa_pairs}")
            continue

        # Process each QA pair in the list
        for qa_pair in qa_pairs:
            # Handle both dict and object formats
            if isinstance(qa_pair, dict):
                question = qa_pair.get('question', '')
                answer = qa_pair.get('answer', '')
                query_type = qa_pair.get('query_type', '')
                reasoning_type = qa_pair.get('reasoning_type', '')
                question_complexity = qa_pair.get('question_complexity', 0)
                segment_ids = qa_pair.get('segment_ids', [])
                hop_count = qa_pair.get('hop_count', 1)
                hop_contexts = qa_pair.get('hop_contexts', [])
            else:
                # Handle Pydantic model objects
                question = getattr(qa_pair, 'question', '')
                answer = getattr(qa_pair, 'answer', '')
                query_type = getattr(qa_pair, 'query_type', '')
                reasoning_type = getattr(qa_pair, 'reasoning_type', '')
                question_complexity = getattr(qa_pair, 'question_complexity', 0)
                segment_ids = getattr(qa_pair, 'segment_ids', [])
                hop_count = getattr(qa_pair, 'hop_count', 1)
                hop_contexts = getattr(qa_pair, 'hop_contexts', [])

            if not question or not isinstance(question, str):
                continue

            # Convert numpy arrays to lists for serialization
            if isinstance(segment_ids, np.ndarray):
                segment_ids = segment_ids.tolist()
            if isinstance(hop_contexts, np.ndarray):
                hop_contexts = hop_contexts.tolist()

            query_id = f"q{query_counter:08d}"
            query_counter += 1

            # Track distributions
            reasoning_types.append(reasoning_type)
            query_types.append(query_type)

            # Build metadata
            metadata = {
                'query_type': query_type,
                'reasoning_type': reasoning_type,
                'question_complexity': question_complexity,
                'hop_count': hop_count,
                'segment_ids': segment_ids,
                'source_file': file_name,
                'answer': answer,  # Include the answer in metadata
            }

            # Add hop contexts if available
            if hop_contexts:
                metadata['hop_contexts'] = hop_contexts

            # Add to queries
            queries_data.append({
                '_id': query_id,
                'metadata': metadata,
                'text': question
            })

            # Add to qrels (using file_name as corpus-id for now)
            # In a real scenario, you'd map this to actual document IDs
            qrels_data.append({
                'query-id': query_id,
                'corpus-id': file_name,
                'score': 1
            })

            # Add to splits (use 'text' as default modality)
            splits['text'].append(query_id)

    queries_df = pd.DataFrame(queries_data)
    qrels_df = pd.DataFrame(qrels_data)

    # Show final statistics
    total_queries = len(queries_df)
    if total_queries > 0:
        print(f"\nGenerated {total_queries} queries from {len(generated_df)} documents")

        # Show reasoning type distribution
        if reasoning_types:
            print("\nReasoning type distribution:")
            reasoning_dist = pd.Series(reasoning_types).value_counts()
            for rtype, count in reasoning_dist.items():
                pct = count/total_queries*100
                print(f"  {rtype}: {count} queries ({pct:.1f}%)")

        # Show query type distribution
        if query_types:
            print("\nQuery type distribution:")
            query_dist = pd.Series(query_types).value_counts()
            for qtype, count in query_dist.items():
                pct = count/total_queries*100
                print(f"  {qtype}: {count} queries ({pct:.1f}%)")
    else:
        print("\nWarning: No queries generated!")

    return queries_df, qrels_df, dict(splits)


def filter_qa_pairs_by_quality(
    generated_df: pd.DataFrame,
    quality_threshold: float = 7.0
) -> tuple[pd.DataFrame, list[dict]]:
    """Filter deduplicated_qa_pairs based on qa_evaluations scores.
    
    This function filters QA pairs from the generated data based on quality scores
    from the qa_evaluations column. Each QA pair's overall score is compared against
    the threshold, and only pairs meeting the threshold are returned.
    
    Files with data integrity issues (mismatched QA pairs and evaluations) are skipped.
    
    Args:
        generated_df: DataFrame with 'deduplicated_qa_pairs', 'qa_evaluations', and 'file_name' columns
        quality_threshold: Minimum overall quality score for QA pairs (default: 7.0)
    
    Returns:
        Tuple of:
        - DataFrame containing filtered QA pairs with columns from the original QA pairs,
          plus 'file_name' and 'quality_score' columns added.
        - List of dicts with 'file_name' and 'reason' for each skipped file.
    """
    print(f"Filtering QA pairs based on quality threshold: {quality_threshold}")

    total_pairs = 0
    filtered_pairs = 0
    all_filtered_qa_pairs = []
    skipped_files = []  # Track files skipped due to data integrity issues

    for _, row in generated_df.iterrows():
        file_name = row.get('file_name', 'unknown')

        # Get deduplicated QA pairs
        deduplicated_qa_pairs = row.get('deduplicated_qa_pairs')
        
        # Get QA evaluations
        qa_evaluations = row.get('qa_evaluations')

        # Handle case where deduplicated_qa_pairs might be None
        if deduplicated_qa_pairs is None:
            print(f"Warning: Skipping {file_name} - deduplicated_qa_pairs is None")
            continue

        # Convert numpy array to list if needed (parquet stores as numpy arrays)
        if isinstance(deduplicated_qa_pairs, np.ndarray):
            deduplicated_qa_pairs = deduplicated_qa_pairs.tolist()

        # Handle case where pairs is not a list or is empty
        if not isinstance(deduplicated_qa_pairs, (list, np.ndarray)) or len(deduplicated_qa_pairs) == 0:
            print(f"Warning: Skipping {file_name} - no valid deduplicated pairs found")
            continue

        # Extract evaluation scores
        evaluation_scores = []
        if qa_evaluations is not None:
            if isinstance(qa_evaluations, dict):
                evaluations_list = qa_evaluations.get('evaluations', [])
            else:
                evaluations_list = getattr(qa_evaluations, 'evaluations', [])
            
            # Convert numpy array to list if needed
            if isinstance(evaluations_list, np.ndarray):
                evaluations_list = evaluations_list.tolist()
            
            # Extract overall scores
            for eval_item in evaluations_list:
                if isinstance(eval_item, dict):
                    overall = eval_item.get('overall', {})
                    if isinstance(overall, dict):
                        score = overall.get('score', 0)
                    else:
                        score = getattr(overall, 'score', 0)
                else:
                    overall = getattr(eval_item, 'overall', None)
                    if overall is not None:
                        if isinstance(overall, dict):
                            score = overall.get('score', 0)
                        else:
                            score = getattr(overall, 'score', 0)
                    else:
                        score = 0
                evaluation_scores.append(score)

        # Validate that deduplicated_qa_pairs and qa_evaluations have the same length
        if len(evaluation_scores) != len(deduplicated_qa_pairs):
            reason = (
                f"deduplicated_qa_pairs has {len(deduplicated_qa_pairs)} items "
                f"but qa_evaluations has {len(evaluation_scores)} items"
            )
            print(f"Warning: Skipping {file_name} - data integrity error: {reason}")
            skipped_files.append({'file_name': file_name, 'reason': reason})
            continue

        # Filter QA pairs based on quality threshold
        for pair_idx, qa_pair in enumerate(deduplicated_qa_pairs):
            total_pairs += 1
            
            # Get quality score for this pair
            quality_score = evaluation_scores[pair_idx] if pair_idx < len(evaluation_scores) else 0
            
            # Filter based on quality threshold
            if quality_score >= quality_threshold:
                # Add file_name and quality_score to the QA pair
                if isinstance(qa_pair, dict):
                    qa_pair_with_metadata = qa_pair.copy()
                else:
                    # Handle Pydantic model objects by converting to dict
                    qa_pair_with_metadata = {
                        'question': getattr(qa_pair, 'question', ''),
                        'answer': getattr(qa_pair, 'answer', ''),
                        'query_type': getattr(qa_pair, 'query_type', ''),
                        'reasoning_type': getattr(qa_pair, 'reasoning_type', ''),
                        'question_complexity': getattr(qa_pair, 'question_complexity', 0),
                        'segment_ids': getattr(qa_pair, 'segment_ids', []),
                        'hop_count': getattr(qa_pair, 'hop_count', 1),
                        'hop_contexts': getattr(qa_pair, 'hop_contexts', []),
                    }
                qa_pair_with_metadata['file_name'] = file_name
                qa_pair_with_metadata['quality_score'] = quality_score
                all_filtered_qa_pairs.append(qa_pair_with_metadata)
            else:
                filtered_pairs += 1

    # Create dataframe from filtered QA pairs
    filtered_df = pd.DataFrame(all_filtered_qa_pairs)

    # Print statistics
    print(f"\nQuality Filtering Results:")
    print(f"  Total QA pairs: {total_pairs}")
    print(f"  Filtered out (score < {quality_threshold}): {filtered_pairs}")
    print(f"  Remaining high-quality pairs: {len(filtered_df)}")
    print(f"  Files skipped due to data issues: {len(skipped_files)}")
    if total_pairs > 0:
        print(f"  Retention rate: {len(filtered_df)/total_pairs*100:.1f}%")
    else:
        print("  Retention rate: 0%")

    return filtered_df, skipped_files


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


@app.command
def generate(
    input_dir: Path,
    output_dir: Path,
    min_text_length: int = 50,
    sentences_per_chunk: int = 5,
    num_sections: int = 1,
    max_artifacts_per_type: int = 2,
    num_pairs: int = 7,
    min_hops: int = 2,
    max_hops: int = 4,
    min_complexity: int = 4,
    preview: bool = False,
    file_extensions: Optional[List[str]] = None,
    artifact_path: Path = Path("./artifacts"),
    num_files: Optional[int] = None,
    batch_size: int = 200,
    start_batch_index: int = 0,
    end_batch_index: int = -1,
    # Multi-doc bundling options
    multi_doc: bool = False,
    bundle_size: int = 2,
    bundle_strategy: Literal["sequential", "doc_balanced", "interleaved"] = "sequential",
    max_docs_per_bundle: int = 3,
    multi_doc_manifest: Optional[Path] = None,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    max_parallel_requests_for_gen: Optional[int] = None,
    # Model configuration
    artifact_extraction_model: str = "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    artifact_extraction_provider: str = "nvidia",
    qa_generation_model: str = "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    qa_generation_provider: str = "nvidia",
    quality_judge_model: str = "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    quality_judge_provider: str = "nvidia",
    embed_model: str = "nvidia/llama-3.2-nv-embedqa-1b-v2",
    embed_provider: str = "nvidia",
) -> None:
    """Generate synthetic queries from a directory of text files.
    
    This is a simplified mode that takes a folder of text files (with optional subfolders)
    and generates queries without requiring the full BEIR format. The pipeline will generate
    one record per input document (or per bundle in multi-doc mode), processing in batches.
    
    Args:
        input_dir: Directory containing text files (can have subfolders)
        output_dir: Directory to save the generated queries
        min_text_length: Minimum text length for documents to include (default: 50)
        sentences_per_chunk: Number of sentences per chunk for text splitting (default: 5)
        num_sections: Number of sections to divide chunks into (default: 1)
        max_artifacts_per_type: Maximum number of artifacts to extract per type (default: 2)
        num_pairs: Number of question-answer pairs to generate per document (default: 5)
        min_hops: Minimum number of hops for multi-hop questions (default: 2)
        max_hops: Maximum number of hops for multi-hop questions (default: 3)
        min_complexity: Minimum complexity level for questions (default: 3)
        preview: Whether to preview the generation without running
        file_extensions: List of file extensions to include (default: [".txt", ".md", ".text"])
        artifact_path: Path to store Data Designer artifacts (default: ./artifacts)
        num_files: Maximum number of files to process (default: None, processes all files)
        batch_size: Number of records to process per batch (default: 200)
        start_batch_index: Batch index to start from, for resuming failed runs (default: 0)
        end_batch_index: Batch index to end at (exclusive), -1 means all batches (default: -1)
        multi_doc: Enable multi-document bundling mode (default: False)
        bundle_size: Number of documents per bundle in multi-doc mode (default: 2)
        bundle_strategy: Segment splitting strategy - 'sequential', 'doc_balanced', or 'interleaved'
        max_docs_per_bundle: Maximum documents allowed per bundle (default: 3)
        multi_doc_manifest: Path to manifest file defining explicit bundles (JSON/YAML)
        log_level: Logging level - DEBUG, INFO, WARNING, or ERROR (default: INFO)
        max_parallel_requests_for_gen: Maximum parallel requests for generation models.
            If not provided, uses the underlying default from the inference library.
        artifact_extraction_model: Model name for artifact extraction (default: nvidia/llama-3.3-nemotron-super-49b-v1.5)
        artifact_extraction_provider: Provider for artifact extraction model (default: nvidia)
        qa_generation_model: Model name for QA generation (default: nvidia/llama-3.3-nemotron-super-49b-v1.5)
        qa_generation_provider: Provider for QA generation model (default: nvidia)
        quality_judge_model: Model name for quality judge (default: nvidia/llama-3.3-nemotron-super-49b-v1.5)
        quality_judge_provider: Provider for quality judge model (default: nvidia)
        embed_model: Model name for embeddings (default: nvidia/llama-3.2-nv-embedqa-1b-v2)
        embed_provider: Provider for embedding model (default: nvidia)
    Examples:
        # Generate from text files (processes all in batches of 200)
        retriever-sdg generate \\
            --input-dir ./my_documents \\
            --output-dir ./generated_queries
        
        # Resume from batch 5 after a failure
        retriever-sdg generate \\
            --input-dir ./my_documents \\
            --output-dir ./generated_queries \\
            --start-batch-index 5
        
        # Preview mode
        retriever-sdg generate \\
            --input-dir ./my_documents \\
            --output-dir ./generated_queries \\
            --preview
        
        # Custom file extensions
        retriever-sdg generate \\
            --input-dir ./my_documents \\
            --output-dir ./generated_queries \\
            --file-extensions .txt .md .rst
        
        # Custom chunk and section sizes
        retriever-sdg generate \\
            --input-dir ./my_documents \\
            --output-dir ./generated_queries \\
            --sentences-per-chunk 10 \\
            --sections-per-document 5
        
        # Custom batch size
        retriever-sdg generate \\
            --input-dir ./my_documents \\
            --output-dir ./generated_queries \\
            --batch-size 100
    """
    if file_extensions is None:
        file_extensions = [".txt", ".md", ".text", ""]  # "" treats files with no extension as text files

    # Load text files (with optional multi-doc bundling)
    print(f"Loading text files from {input_dir}...")
    if multi_doc:
        print(f"Multi-doc mode enabled: bundle_size={bundle_size}, strategy={bundle_strategy}")
        if multi_doc_manifest:
            print(f"Using manifest file: {multi_doc_manifest}")
    
    text_files_df = load_text_files_from_directory(
        input_dir=input_dir,
        file_extensions=file_extensions,
        min_text_length=min_text_length,
        sentences_per_chunk=sentences_per_chunk,
        num_sections=num_sections,
        num_files=num_files,
        multi_doc=multi_doc,
        bundle_size=bundle_size,
        bundle_strategy=bundle_strategy,
        max_docs_per_bundle=max_docs_per_bundle,
        multi_doc_manifest=multi_doc_manifest
    )

    row_type = "bundles" if multi_doc else "text files"
    print(f"\nLoaded {len(text_files_df)} {row_type}")
    print(f"Sample files: {text_files_df['file_name'].head(5).tolist()}")

    # Configure logging
    configure_logging(
        LoggingConfig(
            logger_configs=[LoggerConfig(name="data_designer", level=log_level)],
            output_configs=[OutputConfig(destination=sys.stderr, structured=(log_level == "DEBUG"))],
            root_level=log_level,
        )
    )

    # Initialize Data Designer
    data_designer = DataDesigner(
        artifact_path=artifact_path,
    )
    data_designer.set_run_config(dd.RunConfig(disable_early_shutdown=True))

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate total records and number of batches
    total_records = len(text_files_df)
    num_batches = (total_records + batch_size - 1) // batch_size

    # Determine actual end batch index
    actual_end_batch = num_batches if end_batch_index == -1 else min(end_batch_index, num_batches)

    print(f"\nTotal records: {total_records}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {num_batches}")
    print(f"Starting from batch index: {start_batch_index}")
    print(f"Ending at batch index: {actual_end_batch} (exclusive)")

    # Common model config kwargs passed to build_qa_generation_pipeline
    model_kwargs = dict(
        max_parallel_requests_for_gen=max_parallel_requests_for_gen,
        artifact_extraction_model=artifact_extraction_model,
        artifact_extraction_provider=artifact_extraction_provider,
        qa_generation_model=qa_generation_model,
        qa_generation_provider=qa_generation_provider,
        quality_judge_model=quality_judge_model,
        quality_judge_provider=quality_judge_provider,
        embed_model=embed_model,
        embed_provider=embed_provider,
    )

    print(f"\nModel configuration:")
    print(f"  Artifact extraction: {artifact_extraction_model} ({artifact_extraction_provider})")
    print(f"  QA generation:       {qa_generation_model} ({qa_generation_provider})")
    print(f"  Quality judge:       {quality_judge_model} ({quality_judge_provider})")
    print(f"  Embedding:           {embed_model} ({embed_provider})")

    if preview:
        # For preview, just create config for first batch
        config_builder = build_qa_generation_pipeline(
            seed_dataset=text_files_df,
            start_index=0,
            end_index=min(batch_size - 1, total_records - 1),
            max_artifacts_per_type=max_artifacts_per_type,
            num_pairs=num_pairs,
            min_hops=min_hops,
            max_hops=max_hops,
            min_complexity=min_complexity,
            **model_kwargs,
        )
        print("\nPreviewing generation...")
        try:
            preview_result = data_designer.preview(config_builder, num_records=1)
            preview_result.display_sample_record()
        except Exception as e:
            print(f"Preview error: {e}")
        return

    # Process in batches
    input_basename = input_dir.name
    total_batches_to_run = actual_end_batch - start_batch_index
    batch_times: list[float] = []

    for batch_idx in range(start_batch_index, actual_end_batch):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size - 1, total_records - 1)
        num_records_in_batch = end_idx - start_idx + 1

        print(f"\n{'='*60}")
        print(f"Processing batch {batch_idx}/{num_batches - 1} (records {start_idx}-{end_idx})")
        print(f"{'='*60}")

        batch_start = time.monotonic()

        # Create config builder for this batch
        config_builder = build_qa_generation_pipeline(
            seed_dataset=text_files_df,
            start_index=start_idx,
            end_index=end_idx,
            max_artifacts_per_type=max_artifacts_per_type,
            num_pairs=num_pairs,
            min_hops=min_hops,
            max_hops=max_hops,
            min_complexity=min_complexity,
            **model_kwargs,
        )

        # Run generation for this batch
        dataset_name = f"{input_basename}_batch{batch_idx}_{start_idx}_{end_idx}"
        result = data_designer.create(
            config_builder,
            num_records=num_records_in_batch,
            dataset_name=dataset_name,
        )

        generated_df = result.load_dataset()

        # Save batch output to JSON with batch info in filename
        output_filename = f"generated_batch{batch_idx}_{start_idx}_{end_idx}.json"
        generated_df.to_json(output_dir / output_filename, orient='records', indent=2)

        batch_elapsed = time.monotonic() - batch_start
        batch_times.append(batch_elapsed)

        batches_done = batch_idx - start_batch_index + 1
        batches_remaining = actual_end_batch - batch_idx - 1

        print(f"Batch {batch_idx}/{num_batches - 1} done in {_format_duration(batch_elapsed)}")
        print(f"  Saved to {output_filename} ({len(generated_df)} records)")
        if batches_remaining > 0:
            avg_batch_time = sum(batch_times) / len(batch_times)
            eta_seconds = avg_batch_time * batches_remaining
            print(f"  Progress: {batches_done}/{total_batches_to_run} batches")
            print(f"  ETA: ~{_format_duration(eta_seconds)} remaining")

    print(f"\n{'='*60}")
    print(f"Generation complete! All batches saved to {output_dir}")
    print(f"Total batches processed: {actual_end_batch - start_batch_index}")
    print("\nOutput files:")
    print(f"  - generated_batch{{idx}}_{{start}}_{{end}}.json: Raw generation data per batch")


def entrypoint():
    app.cli(
        description="SDG Pipeline for Retriever Evaluation Dataset Generation"
    )


if __name__ == "__main__":
    entrypoint()
