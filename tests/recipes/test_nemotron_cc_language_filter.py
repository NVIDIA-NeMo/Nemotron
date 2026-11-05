# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Language selection accepts FastText labels from old and new Curator versions."""

import importlib.util
from inspect import signature
from pathlib import Path
from types import ModuleType

import pandas as pd
import pyarrow as pa
import pytest

curator_tasks = pytest.importorskip("nemo_curator.tasks")


@pytest.fixture(scope="module")
def download_extract() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "src/nemotron/recipes/data/curation/nemotron-cc/step_1-download_extract.py"
    )
    spec = importlib.util.spec_from_file_location("nemotron_cc_download_extract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("backend", ["pandas", "pyarrow"])
@pytest.mark.parametrize("language_field", ["language", "detected_language"])
@pytest.mark.parametrize(
    ("target_languages", "kept_rows"),
    [
        (["en"], [0, 1]),
        (["EN"], [0, 1]),
        (["en", "fR"], [0, 1, 2, 3]),
        (["es"], []),
        (None, [0, 1, 2, 3, 4]),
        ([], [0, 1, 2, 3, 4]),
    ],
)
def test_language_filter_preserves_matching_documents(
    download_extract: ModuleType,
    backend: str,
    language_field: str,
    target_languages: list[str] | None,
    kept_rows: list[int],
) -> None:
    labels = ["EN", "en", "FR", "fr", "de"]
    frame = pd.DataFrame(
        {
            "text": ["old English", "new English", "old French", "new French", "German"],
            language_field: [str([0.95, label]) for label in labels],
            "source": ["a", "b", "c", "d", "e"],
        },
        index=[3, 5, 8, 9, 12],
    )
    data = frame if backend == "pandas" else pa.Table.from_pandas(frame)
    # Curator 1.2 requires task_id as a constructor argument; Curator 1.3 does not.
    task_kwargs = {"task_id": "test"} if "task_id" in signature(curator_tasks.DocumentBatch).parameters else {}
    task = curator_tasks.DocumentBatch(dataset_name="test", data=data, _metadata={"snapshot": "test"}, **task_kwargs)
    expected = task.to_pandas().copy()
    expected[language_field] = labels
    expected = expected.iloc[kept_rows]

    result = download_extract.LanguageFilter(target_languages, language_field).process(task)

    if not kept_rows:
        assert result is None
    else:
        assert result is task
        assert result.dataset_name == "test"
        assert result._metadata == {"snapshot": "test"}
        pd.testing.assert_frame_equal(result.to_pandas(), expected)
