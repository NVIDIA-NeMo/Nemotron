#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synthetic Data Generation for embedding fine-tuning.

Generates synthetic Q&A pairs from document corpus using retriever-sdg.
This step uses NVIDIA's LLM APIs to create high-quality training data.

Usage:
    # With default config
    nemotron embed sdg -c default

    # With custom config
    nemotron embed sdg -c /path/to/config.yaml

    # With CLI overrides
    nemotron embed sdg -c default corpus_dir=/path/to/docs
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import List

from pydantic import ConfigDict, Field, model_validator

from nemotron.kit.config_loader import RecipeSettings, load_config, parse_config_and_overrides

STAGE_PATH = Path(__file__).parent
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "default.yaml"

# Use NEMO_RUN_DIR for output when running via nemo-run
_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))


class SDGConfig(RecipeSettings):
    """Synthetic Data Generation configuration.

    Uses retriever-sdg to generate Q&A pairs from document corpus.
    All CLI arguments of the ``retriever-sdg generate`` command are
    configurable here.
    """

    model_config = ConfigDict(extra="forbid")

    # --- Core paths -----------------------------------------------------------
    corpus_id: str = Field(default="my_corpus", description="Identifier for your corpus (used in output naming).")
    corpus_dir: Path = Field(default_factory=lambda: Path("./data/corpus"), description="Path to directory containing document files (.txt, .md, etc.).")
    output_dir: Path = Field(default_factory=lambda: _OUTPUT_BASE / "output/embed/stage0_sdg", description="Output directory for generated synthetic data.")
    file_extensions: str | None = Field(default=None, description="Comma-separated list of file extensions to process.")
    nvidia_api_key: str | None = Field(default=None, description="NVIDIA API key for LLM access. If None, uses NVIDIA_API_KEY env var.")

    # --- Document processing ---------------------------------------------------
    min_text_length: int = Field(default=50, ge=0, description="Minimum text length (characters) for documents to include.")
    sentences_per_chunk: int = Field(default=5, gt=0, description="Number of sentences per chunk for text splitting.")
    num_sections: int = Field(default=1, gt=0, description="Number of sections to divide chunks into.")
    num_files: int | None = Field(default=None, gt=0, description="Maximum number of files to process. None means process all files.")

    # --- Generation parameters -------------------------------------------------
    max_artifacts_per_type: int = Field(default=2, gt=0, description="Maximum number of artifacts to extract per type.")
    num_pairs: int = Field(default=5, gt=0, description="Number of question-answer pairs to generate per document.")
    min_hops: int = Field(default=1, ge=1, description="Minimum number of hops for multi-hop questions.")
    max_hops: int = Field(default=3, ge=1, description="Maximum number of hops for multi-hop questions.")
    min_complexity: int = Field(default=2, ge=1, le=5, description="Minimum complexity level for questions (1-5).")

    @model_validator(mode="after")
    def _check_hops_order(self):
        if self.min_hops > self.max_hops:
            raise ValueError(f"min_hops ({self.min_hops}) must be <= max_hops ({self.max_hops})")
        return self

    # --- Batch processing ------------------------------------------------------
    batch_size: int = Field(default=200, gt=0, description="Number of records to process per batch.")
    start_batch_index: int = Field(default=0, ge=0, description="Batch index to start from (for resuming failed runs).")
    end_batch_index: int = Field(default=-1, description="Batch index to end at (exclusive). -1 means all batches.")

    # --- Multi-document bundling -----------------------------------------------
    multi_doc: bool = Field(default=False, description="Enable multi-document bundling mode.")
    bundle_size: int = Field(default=2, gt=0, description="Number of documents per bundle in multi-doc mode.")
    bundle_strategy: str = Field(default="sequential", description="Segment splitting strategy: 'sequential', 'doc_balanced', or 'interleaved'.")
    max_docs_per_bundle: int = Field(default=3, gt=0, description="Maximum documents allowed per bundle.")
    multi_doc_manifest: str | None = Field(default=None, description="Path to manifest file defining explicit bundles (JSON/YAML).")

    # --- Model configuration ---------------------------------------------------
    artifact_extraction_model: str = Field(default="nvidia/nemotron-3-nano-30b-a3b", description="Model name for artifact extraction.")
    artifact_extraction_provider: str = Field(default="nvidia", description="Provider for artifact extraction model.")
    qa_generation_model: str = Field(default="nvidia/nemotron-3-nano-30b-a3b", description="Model name for QA generation.")
    qa_generation_provider: str = Field(default="nvidia", description="Provider for QA generation model.")
    quality_judge_model: str = Field(default="nvidia/nemotron-3-nano-30b-a3b", description="Model name for quality judge.")
    quality_judge_provider: str = Field(default="nvidia", description="Provider for quality judge model.")
    embed_model: str = Field(default="nvidia/llama-3.2-nv-embedqa-1b-v2", description="Model name for embeddings.")
    embed_provider: str = Field(default="nvidia", description="Provider for embedding model.")
    max_parallel_requests_for_gen: int | None = Field(default=None, gt=0, description="Maximum parallel requests for generation models. None uses the library default.")

    # --- Runtime options -------------------------------------------------------
    artifact_path: Path = Field(default_factory=lambda: _OUTPUT_BASE / "output/embed/stage0_sdg/artifacts", description="Path to store Data Designer artifacts.")
    preview: bool = Field(default=False, description="Preview the generation without actually running.")
    log_level: str = Field(default="INFO", description="Logging level: DEBUG, INFO, WARNING, or ERROR.")


def _validate_corpus(
    corpus_dir: Path,
    file_extensions: List[str] | None,
    min_text_length: int,
    num_pairs: int,
    batch_size: int,
) -> None:
    """Validate the corpus directory and print a summary before generation.

    Scans files matching *file_extensions*, checks sizes against
    *min_text_length*, and prints a human-readable summary with warnings.
    Exits with an error if no usable files are found.
    """
    # Default extensions match those used by retriever-sdg's
    # load_text_files_from_directory when no explicit list is provided.
    if file_extensions is None:
        extensions = [".txt", ".md", ".text", ""]
    else:
        extensions = file_extensions

    ext_display = ", ".join(repr(e) for e in extensions if e) or "(any)"

    char_sizes: list[int] = []
    skipped_short: list[Path] = []
    empty_files: list[Path] = []

    for file_path in sorted(corpus_dir.rglob("*")):
        if not file_path.is_file():
            continue
        # Replicate the extension-matching logic used downstream.
        suffix = file_path.suffix.lower()
        # Simple check: either the suffix is in the list, or "" is in the
        # list and the suffix doesn't look like a traditional short extension.
        if suffix in extensions or ("" in extensions and (len(suffix) > 11 or not suffix)):
            pass  # matches
        else:
            continue

        try:
            size = file_path.stat().st_size
        except OSError:
            continue

        if size == 0:
            empty_files.append(file_path)
            skipped_short.append(file_path)
            continue

        # Read the file to get character count (matching downstream behaviour
        # which filters on character length, not byte size).
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception:
            continue

        char_len = len(text)
        if min_text_length > 0 and char_len < min_text_length:
            skipped_short.append(file_path)
            continue

        char_sizes.append(char_len)

    total_scanned = len(char_sizes) + len(skipped_short)

    # --- Hard error: nothing usable -------------------------------------------
    if not char_sizes:
        if total_scanned == 0:
            print(
                f"Error: No files found in {corpus_dir} matching extensions: {ext_display}",
                file=sys.stderr,
            )
        else:
            print(
                f"Error: All {total_scanned} files were skipped "
                f"(below min_text_length={min_text_length}).",
                file=sys.stderr,
            )
        sys.exit(1)

    # --- Gather size stats ----------------------------------------------------
    total_chars = sum(char_sizes)
    avg_chars = total_chars // len(char_sizes) if char_sizes else 0
    min_chars = min(char_sizes) if char_sizes else 0
    max_chars = max(char_sizes) if char_sizes else 0

    def _fmt_size(chars: int) -> str:
        if chars >= 1_000_000:
            return f"{chars / 1_000_000:,.1f} MB"
        if chars >= 1_000:
            return f"{chars / 1_000:,.0f} KB"
        return f"{chars:,} chars"

    num_docs = len(char_sizes)
    expected_pairs = num_docs * num_pairs
    num_batches = math.ceil(num_docs / batch_size)

    # --- Print summary --------------------------------------------------------
    print(f"\nCorpus summary:")
    print(f"  Files found:       {num_docs} (matching extensions: {ext_display})")
    if skipped_short:
        print(f"  Skipped:           {len(skipped_short)} (below min_text_length={min_text_length})")
    print(f"  Total text:        {_fmt_size(total_chars)}")
    print(f"  Avg file size:     {avg_chars:,} chars")
    print(f"  Size range:        {min_chars:,} - {max_chars:,} chars")
    print()
    print(f"Generation plan:")
    print(f"  Documents:         {num_docs}")
    print(f"  QA pairs/doc:      {num_pairs}")
    print(f"  Expected QA pairs: ~{expected_pairs:,}")
    print(f"  Batches:           {num_batches} (batch_size={batch_size})")
    print(f"  API stages/batch:  4 (artifact extraction -> QA generation -> dedup -> quality eval)")

    # --- Warnings -------------------------------------------------------------
    warnings_printed = False
    if num_docs < 20:
        print(f"\n  Warning: Very few documents ({num_docs}). Consider adding more for better training data.")
        warnings_printed = True
    if total_scanned > 0 and len(skipped_short) / total_scanned > 0.5:
        pct = len(skipped_short) * 100 // total_scanned
        print(
            f"\n  Warning: {pct}% of files were too short (skipped {len(skipped_short)}/{total_scanned}). "
            f"Check min_text_length={min_text_length} or file content."
        )
        warnings_printed = True
    if empty_files:
        print(f"\n  Warning: {len(empty_files)} empty file(s):")
        for ef in empty_files[:10]:
            print(f"    - {ef}")
        if len(empty_files) > 10:
            print(f"    ... and {len(empty_files) - 10} more")
        warnings_printed = True

    if warnings_printed:
        print()


def run_sdg(cfg: SDGConfig) -> Path:
    """Run synthetic data generation using vendored retriever-sdg.

    Args:
        cfg: SDG configuration.

    Returns:
        Path to output directory with generated data.
    """
    # Import from vendored retriever_sdg (installed via pyproject.toml)
    from retriever_sdg.pipeline import generate

    # Resolve all Path fields to absolute so downstream libraries (data-designer)
    # don't depend on CWD and error messages show full paths.
    corpus_dir = cfg.corpus_dir.resolve()
    output_dir = cfg.output_dir.resolve()
    artifact_path = cfg.artifact_path.resolve()

    # Validate input corpus directory
    if not corpus_dir.exists():
        print(f"Error: Corpus directory not found: {corpus_dir}", file=sys.stderr)
        sys.exit(1)
    if not any(corpus_dir.iterdir()):
        print(f"Error: Corpus directory is empty: {corpus_dir}", file=sys.stderr)
        sys.exit(1)

    # Get API key from config or environment
    api_key = cfg.nvidia_api_key or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("Error: NVIDIA_API_KEY not set. Please set it in the environment or config.")
        sys.exit(1)

    # Set API key in environment for data-designer
    os.environ["NVIDIA_API_KEY"] = api_key

    # Parse file extensions from comma-separated string to list
    file_extensions = None
    if cfg.file_extensions:
        file_extensions = [ext.strip() for ext in cfg.file_extensions.split(",") if ext.strip()]

    print(f"🚀 Starting synthetic data generation...")
    print(f"   Corpus ID: {cfg.corpus_id}")
    print(f"   Input:  {corpus_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Artifacts: {artifact_path}")
    print(f"   Model (QA generation): {cfg.qa_generation_model}")
    print(f"   Num pairs: {cfg.num_pairs}")
    print(f"   Batch size: {cfg.batch_size}")
    print()

    # Validate corpus and print summary before spending API credits
    _validate_corpus(
        corpus_dir=corpus_dir,
        file_extensions=file_extensions,
        min_text_length=cfg.min_text_length,
        num_pairs=cfg.num_pairs,
        batch_size=cfg.batch_size,
    )

    # Call generate function directly
    try:
        generate(
            input_dir=corpus_dir,
            output_dir=output_dir,
            min_text_length=cfg.min_text_length,
            sentences_per_chunk=cfg.sentences_per_chunk,
            num_sections=cfg.num_sections,
            max_artifacts_per_type=cfg.max_artifacts_per_type,
            num_pairs=cfg.num_pairs,
            min_hops=cfg.min_hops,
            max_hops=cfg.max_hops,
            min_complexity=cfg.min_complexity,
            preview=cfg.preview,
            file_extensions=file_extensions,
            artifact_path=artifact_path,
            num_files=cfg.num_files,
            batch_size=cfg.batch_size,
            start_batch_index=cfg.start_batch_index,
            end_batch_index=cfg.end_batch_index,
            multi_doc=cfg.multi_doc,
            bundle_size=cfg.bundle_size,
            bundle_strategy=cfg.bundle_strategy,
            max_docs_per_bundle=cfg.max_docs_per_bundle,
            multi_doc_manifest=cfg.multi_doc_manifest,
            log_level=cfg.log_level,
            max_parallel_requests_for_gen=cfg.max_parallel_requests_for_gen,
            artifact_extraction_model=cfg.artifact_extraction_model,
            artifact_extraction_provider=cfg.artifact_extraction_provider,
            qa_generation_model=cfg.qa_generation_model,
            qa_generation_provider=cfg.qa_generation_provider,
            quality_judge_model=cfg.quality_judge_model,
            quality_judge_provider=cfg.quality_judge_provider,
            embed_model=cfg.embed_model,
            embed_provider=cfg.embed_provider,
        )
    except FileNotFoundError as exc:
        print(f"\nError: SDG pipeline could not find an intermediate file.", file=sys.stderr)
        print(f"  Missing: {exc}", file=sys.stderr)
        print(f"  Artifacts dir: {artifact_path}", file=sys.stderr)
        print(f"\nThis usually means a previous step produced no output.", file=sys.stderr)
        print(f"Common causes:", file=sys.stderr)
        print(f"  - NVIDIA_API_KEY is invalid or expired", file=sys.stderr)
        print(f"  - The LLM API returned errors (check output above for 4xx/5xx)", file=sys.stderr)
        print(f"  - The corpus documents were too short after chunking", file=sys.stderr)
        print(f"\nCheck {artifact_path} for partial output to diagnose.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        exc_name = type(exc).__name__
        exc_str = str(exc)

        # Authentication / API key errors
        if "Authentication" in exc_name or "api key" in exc_str.lower():
            print(f"\nError: NVIDIA API key is invalid or expired.", file=sys.stderr)
            print(f"  Set a valid key via:  export NVIDIA_API_KEY=nvapi-...", file=sys.stderr)
            print(f"  Or in your config:    nvidia_api_key: nvapi-...", file=sys.stderr)
            sys.exit(1)

        # Generation errors (wraps auth errors and other data-designer failures)
        if "Generation" in exc_name:
            # Extract the inner message — data-designer wraps it with an emoji prefix
            inner = exc_str.removeprefix("🛑 Error generating dataset:").strip()
            print(f"\nError: SDG generation failed.", file=sys.stderr)
            print(f"  {inner}", file=sys.stderr)
            sys.exit(1)

        # Profiling errors (missing intermediate files)
        if "Profiling" in exc_name or "profiling" in exc_str.lower():
            print(f"\nError: SDG data profiling failed — intermediate files are missing.", file=sys.stderr)
            print(f"  {exc_name}: {exc}", file=sys.stderr)
            print(f"  Artifacts dir: {artifact_path}", file=sys.stderr)
            print(f"\nThis usually means the LLM generation step produced no output.", file=sys.stderr)
            print(f"Common causes:", file=sys.stderr)
            print(f"  - NVIDIA_API_KEY is invalid or expired", file=sys.stderr)
            print(f"  - The LLM API returned errors (check output above for 4xx/5xx)", file=sys.stderr)
            print(f"  - The corpus documents were too short after chunking", file=sys.stderr)
            print(f"\nCheck {artifact_path} for partial output to diagnose.", file=sys.stderr)
            sys.exit(1)

        raise

    print(f"\n✅ Synthetic data generation complete!")
    print(f"   Output: {output_dir}")

    return output_dir


def main(cfg: SDGConfig | None = None) -> Path:
    """Entry point for synthetic data generation.

    Args:
        cfg: Config from CLI framework, or None when run directly as script.

    Returns:
        Path to output directory.
    """
    if cfg is None:
        # Called directly as script - parse config ourselves
        config_path, cli_overrides = parse_config_and_overrides(
            default_config=DEFAULT_CONFIG_PATH
        )

        try:
            cfg = load_config(config_path, cli_overrides, SDGConfig)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    return run_sdg(cfg)


if __name__ == "__main__":
    main()
