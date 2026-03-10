"""Session-scoped fixtures for embed recipe integration tests.

Generates synthetic SDG output, runs stage 1 convert and unroll scripts
as subprocesses, and shares outputs across all test modules.

GPU-tier fixtures create isolated venvs via ``uv sync`` for each stage
and detect Docker + NVIDIA runtime availability.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBED_DIR = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "nemotron"
    / "recipes"
    / "embed"
)
SCRIPTS_DIR = EMBED_DIR / "stage1_data_prep" / "scripts"
CONVERT_SCRIPT = SCRIPTS_DIR / "convert_to_retriever_data.py"
UNROLL_SCRIPT = SCRIPTS_DIR / "unroll_pos_docs.py"

CORPUS_ID = "test_corpus"
NUM_SYNTHETIC_FILES = 15

# ---------------------------------------------------------------------------
# Stage metadata for GPU-tier tests
# ---------------------------------------------------------------------------
STAGES: dict[str, dict] = {
    "stage1_data_prep": {
        "key_imports": ["nemo_automodel", "sentence_transformers", "faiss", "pandas"],
        "container": "nvcr.io/nvidia/pytorch:25.12-py3",
        "entry_script": "data_prep.py",
    },
    "stage2_finetune": {
        "key_imports": ["nemo_automodel", "omegaconf"],
        "container": "nvcr.io/nvidia/pytorch:25.12-py3",
        "entry_script": "train.py",
    },
    "stage3_eval": {
        "key_imports": ["beir", "sentence_transformers", "torch"],
        "container": "nvcr.io/nvidia/pytorch:25.12-py3",
        "entry_script": "eval.py",
    },
}


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------
def _make_sdg_record(file_index: int) -> dict:
    """Return one synthetic SDG JSON record mimicking stage 0 output.

    Each record has:
    - 2 chunks with globally unique chunk_id values
    - 2 QA pairs: first references chunk 1, second references both chunks
    - 2 qa_evaluations with score 8.5 (above the default 7.0 threshold)
    """
    chunk_id_1 = file_index * 2 + 1
    chunk_id_2 = file_index * 2 + 2

    return {
        "file_name": [f"synthetic_doc_{file_index:03d}.txt"],
        "chunks": [
            {
                "chunk_id": chunk_id_1,
                "text": f"This is chunk {chunk_id_1} from document {file_index}. "
                f"It contains important information about topic A.",
            },
            {
                "chunk_id": chunk_id_2,
                "text": f"This is chunk {chunk_id_2} from document {file_index}. "
                f"It contains important information about topic B.",
            },
        ],
        "deduplicated_qa_pairs": [
            {
                "question": f"What is discussed in chunk {chunk_id_1} of doc {file_index}?",
                "answer": f"Chunk {chunk_id_1} discusses topic A.",
                "query_type": "factual",
                "reasoning_type": "extractive",
                "question_complexity": 1,
                "segment_ids": [chunk_id_1],
                "hop_count": 1,
                "hop_contexts": [],
            },
            {
                "question": f"How do topics A and B relate in doc {file_index}?",
                "answer": f"Topics A ({chunk_id_1}) and B ({chunk_id_2}) are complementary.",
                "query_type": "multi_hop",
                "reasoning_type": "analytical",
                "question_complexity": 2,
                "segment_ids": [chunk_id_1, chunk_id_2],
                "hop_count": 2,
                "hop_contexts": [
                    f"Context from chunk {chunk_id_1}",
                    f"Context from chunk {chunk_id_2}",
                ],
            },
        ],
        "qa_evaluations": {
            "evaluations": [
                {"overall": {"score": 8.5}},
                {"overall": {"score": 8.5}},
            ]
        },
    }


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def sdg_output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write all synthetic SDG records as ``generated_batch_0.json``."""
    d = tmp_path_factory.mktemp("sdg_output")
    records = [_make_sdg_record(i) for i in range(NUM_SYNTHETIC_FILES)]
    (d / "generated_batch_0.json").write_text(json.dumps(records, indent=2))
    return d


@pytest.fixture(scope="session")
def convert_output_dir(
    sdg_output_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Run *convert_to_retriever_data.py* and return its output directory."""
    out = tmp_path_factory.mktemp("convert_output")
    cmd = [
        sys.executable,
        str(CONVERT_SCRIPT),
        str(sdg_output_dir),
        "--corpus-id",
        CORPUS_ID,
        "--output-dir",
        str(out),
        "--seed",
        "42",
        "--quality-threshold",
        "7.0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, (
        f"convert script failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    return out


@pytest.fixture(scope="session")
def unroll_output_path(convert_output_dir: Path) -> Path:
    """Run *unroll_pos_docs.py* on train.json and return path to the unrolled file."""
    train_json = convert_output_dir / "train.json"
    assert train_json.exists(), f"train.json not found in {convert_output_dir}"

    cmd = [
        sys.executable,
        str(UNROLL_SCRIPT),
        str(train_json),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, (
        f"unroll script failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )

    output = convert_output_dir / "train_unrolled.json"
    assert output.exists(), f"train_unrolled.json not found in {convert_output_dir}"
    return output


# ---------------------------------------------------------------------------
# GPU-tier fixtures
# ---------------------------------------------------------------------------
_stage_venv_cache: dict[str, Path] = {}


@pytest.fixture(scope="session")
def stage_venv_factory(tmp_path_factory: pytest.TempPathFactory):
    """Factory that creates and caches a uv-synced venv for a given stage.

    Returns a callable ``(stage_name) -> Path`` where the path points to
    ``<venv>/bin/python``.  Venvs are session-scoped and reused across tests.
    """
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not found on PATH")

    def _get_or_create(stage_name: str) -> Path:
        if stage_name in _stage_venv_cache:
            return _stage_venv_cache[stage_name]

        stage_dir = EMBED_DIR / stage_name
        assert stage_dir.is_dir(), f"Stage directory not found: {stage_dir}"
        assert (stage_dir / "pyproject.toml").is_file(), (
            f"pyproject.toml not found in {stage_dir}"
        )

        venv_dir = tmp_path_factory.mktemp(f"venv_{stage_name}")
        env = {**subprocess.os.environ, "UV_PROJECT_ENVIRONMENT": str(venv_dir)}

        result = subprocess.run(
            [uv, "sync", "--frozen", "--project", str(stage_dir)],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        assert result.returncode == 0, (
            f"uv sync failed for {stage_name} (rc={result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )

        python = venv_dir / "bin" / "python"
        assert python.is_file(), f"Python not found at {python}"
        _stage_venv_cache[stage_name] = python
        return python

    return _get_or_create


@pytest.fixture(scope="session")
def docker_available() -> bool:
    """Return True if Docker with NVIDIA runtime is available."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return False
        # Check for NVIDIA runtime in docker info output
        return "nvidia" in result.stdout.lower()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
