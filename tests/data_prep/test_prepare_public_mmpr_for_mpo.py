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

"""Tests for ``scripts/prepare_public_mmpr_for_mpo.py``.

We focus on the meta-rewrite logic — the only real judgment call in the
script. Zip extraction is delegated to ``zipfile`` and the directory-
shape glob is mirror-of-mmpr-tiny code already exercised in production.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "prepare_public_mmpr_for_mpo.py"


def _load_script_module():
    """Load the standalone script as a Python module for unit testing.

    The script lives outside any package (``scripts/`` is not a Python
    package) so we use importlib.util rather than a regular import.
    """
    spec = importlib.util.spec_from_file_location(
        "prepare_public_mmpr_for_mpo", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None, SCRIPT_PATH
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script_module()


class TestRelpathForMeta:
    def test_strips_petrelfs_prefix(self):
        result = SCRIPT._relpath_for_meta(
            "/mnt/petrelfs/share_data/wangweiyun/open_source_wwy/MMPR/annotations/openbmb_RLAIF-V-Dataset.jsonl"
        )
        assert str(result) == "MMPR/annotations/openbmb_RLAIF-V-Dataset.jsonl"

    def test_strips_windows_separators(self):
        result = SCRIPT._relpath_for_meta(
            r"\some\windows\path\with_backslash.jsonl"
        )
        assert str(result) == "MMPR/annotations/with_backslash.jsonl"

    def test_already_relative_passes_through_basename(self):
        result = SCRIPT._relpath_for_meta("MMPR/annotations/foo.jsonl")
        assert str(result) == "MMPR/annotations/foo.jsonl"


class TestRewriteMeta:
    def _make_meta(self) -> dict[str, Any]:
        return {
            "openbmb/RLAIF-V-Dataset": {
                "root": "MMPR/images/RLAIF-V",
                "annotation": (
                    "/mnt/petrelfs/share_data/wangweiyun/open_source_wwy/MMPR/"
                    "annotations/openbmb_RLAIF-V-Dataset.jsonl"
                ),
                "data_augment": False,
                "repeat_time": 1,
                "length": 83121,
            },
            "CLEVR_math_en_extracted_prefix_pair_sr0.5_wo_image": {
                "root": "MMPR/images/CLEVR",
                "annotation": (
                    "/mnt/petrelfs/share_data/wangweiyun/open_source_wwy/MMPR/"
                    "annotations/clevr_math_en.jsonl"
                ),
                "length": 12000,
            },
        }

    def _stage_annotations(self, ann_dir: Path, names: list[str]) -> None:
        ann_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            (ann_dir / name).write_text("{}\n", encoding="utf-8")

    def test_rewrites_paths_to_relative(self, tmp_path: Path):
        meta_in = tmp_path / "meta.json"
        ann_dir = tmp_path / "MMPR" / "annotations"
        meta = self._make_meta()
        meta_in.write_text(json.dumps(meta), encoding="utf-8")
        self._stage_annotations(
            ann_dir,
            ["openbmb_RLAIF-V-Dataset.jsonl", "clevr_math_en.jsonl"],
        )

        out_path = tmp_path / "meta_public.json"
        rewritten = SCRIPT.rewrite_meta(meta_in, ann_dir, out_path)

        assert (
            rewritten["openbmb/RLAIF-V-Dataset"]["annotation"]
            == "MMPR/annotations/openbmb_RLAIF-V-Dataset.jsonl"
        )
        # Other fields preserved verbatim.
        assert rewritten["openbmb/RLAIF-V-Dataset"]["length"] == 83121
        assert rewritten["openbmb/RLAIF-V-Dataset"]["repeat_time"] == 1
        assert rewritten["openbmb/RLAIF-V-Dataset"]["data_augment"] is False
        # ``root`` is left untouched (already relative-from-cache form).
        assert (
            rewritten["openbmb/RLAIF-V-Dataset"]["root"] == "MMPR/images/RLAIF-V"
        )

        # Output file is valid JSON.
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        assert loaded == rewritten

    def test_raises_on_missing_required_key(self, tmp_path: Path):
        meta_in = tmp_path / "meta.json"
        ann_dir = tmp_path / "MMPR" / "annotations"
        meta_in.write_text(
            json.dumps(
                {
                    "broken": {
                        "annotation": "/mnt/petrelfs/whatever.jsonl",
                        # missing ``root`` and ``length``
                    },
                }
            ),
            encoding="utf-8",
        )
        ann_dir.mkdir(parents=True, exist_ok=True)
        (ann_dir / "whatever.jsonl").write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="missing required keys"):
            SCRIPT.rewrite_meta(meta_in, ann_dir, tmp_path / "out.json")

    def test_raises_when_annotation_file_missing(self, tmp_path: Path):
        meta_in = tmp_path / "meta.json"
        ann_dir = tmp_path / "MMPR" / "annotations"
        ann_dir.mkdir(parents=True)
        meta_in.write_text(json.dumps(self._make_meta()), encoding="utf-8")
        # Stage only one of the two annotations referenced by meta.
        (ann_dir / "openbmb_RLAIF-V-Dataset.jsonl").write_text("{}\n")

        with pytest.raises(FileNotFoundError, match="Annotations missing"):
            SCRIPT.rewrite_meta(meta_in, ann_dir, tmp_path / "out.json")

    def test_raises_when_meta_isnt_dict(self, tmp_path: Path):
        meta_in = tmp_path / "meta.json"
        meta_in.write_text(json.dumps([{"not": "a dict"}]), encoding="utf-8")
        ann_dir = tmp_path / "MMPR" / "annotations"
        ann_dir.mkdir(parents=True)

        with pytest.raises(ValueError, match="Expected meta.json to be a dict"):
            SCRIPT.rewrite_meta(meta_in, ann_dir, tmp_path / "out.json")
