# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""VLM Preference Prep Stage — single-worker preference-cache builder.

Generic *shape*: take a media archive plus a tabular records file, extract
the archive into a cache layout, copy the records through, validate rows
against the extracted media, and write a ready sentinel. This is the
shape any image-text preference dataset shipped as
``(images.zip + records.parquet)`` has.

Current *implementation*: targets MMPR-Tiny's specific parquet schema
(``prompt`` / ``reward_model`` / ``images`` columns) and the
``MMPR-Tiny/images/`` cache layout that NeMo-RL's ``mmpr_tiny``
response dataset expects. The row-validation helpers below
(``_extract_question`` / ``_extract_answer`` / ``_normalize_cache_image_path``
/ ``_extract_first_image_path``) encode that schema knowledge inline.

When a second VLM preference dataset arrives with a different schema,
factor the row-validation helpers out into a per-dataset adapter
(callable on the work item or stage config). Until then "locality over
DRY" applies — the schema-specific code is small enough to live next
to the rest of the prep without harming reuse.

Flavors:

- ``flavor="tiny"`` — fully vendored from the upstream
  ``prepare_mmpr_tiny_for_vision_rl.py`` script. Reads raw HF download
  (``images.zip`` + ``mmpr_tiny.parquet``), extracts images into the
  cache layout NeMo-RL's ``mmpr_tiny`` response dataset expects, copies
  the parquet through, builds an inspection JSONL, writes the
  ``.mmpr_ready`` sentinel and a summary JSON.

- ``flavor="mpo"`` — TODO: vendor logic from upstream
  ``prepare_public_mmpr_for_mpo.py``. Until that script's logic can be
  brought in-tree, this flavor shells out via ``builder_command`` and
  verifies the expected ``meta_public.json`` exists. Framework
  consistency (run-hash, receipts, lineage) is preserved either way.

Why single-worker and not multi-worker fan-out? The MMPR-Tiny prep is
bounded by one zip extraction and one parquet validation — neither is
embarrassingly parallel at row level (the parquet is small) and the zip
is a single archive. The framework value here is *consistency* (receipts,
caching, observability), not throughput. When MMPR-full or a successor
dataset arrives that genuinely benefits from fan-out (e.g., multiple
per-subset archives), the work-item shape can be widened to
one-per-archive and ``num_workers`` lifted accordingly without changing
the recipe surface.
"""

from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import cosmos_xenna.pipelines.v1 as pipelines_v1

from nemotron.data_prep.core.receipt import ReceiptManager
from nemotron.data_prep.core.work_items import VlmPreferencePrepWorkItem
from nemotron.data_prep.stages.context import PipelineContext
from nemotron.data_prep.utils.filesystem import get_filesystem, read_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VlmPreferencePrepStageConfig:
    """Configuration for VlmPreferencePrepStage.

    Attributes:
        stage_cpus: CPU request per worker. Zip extraction is I/O-bound,
            parquet validation is single-threaded pandas — 1.0 is plenty.
        preview_name: Output filename for the inspection JSONL (tiny flavor).
        summary_name: Output filename for the prep summary JSON (tiny flavor).
    """

    stage_cpus: float = 1.0
    preview_name: str = "mmpr_tiny_preview.jsonl"
    summary_name: str = "prepare_mmpr_tiny_for_vision_rl_summary.json"

    def __post_init__(self) -> None:
        if self.stage_cpus <= 0:
            raise ValueError(f"stage_cpus must be positive, got {self.stage_cpus}")


# =============================================================================
# Vendored MMPR-Tiny prep helpers
# =============================================================================
#
# Faithful in-tree reproduction of OpenGVLab/MMPR-Tiny → NeMo-RL cache
# layout conversion. Source:
# https://huggingface.co/datasets/OpenGVLab/MMPR-Tiny + the upstream
# ``prepare_mmpr_tiny_for_vision_rl.py`` recipe shipped with NeMo-RL's
# nano-v3-omni-recipes branch. Owned in-tree to enable typing, tests,
# and parallel evolution alongside this framework.
#
# Behavior notes (preserved from upstream):
# - The extracted-images directory under the zip can vary in depth; the
#   helper falls back to globbing for ``images/`` and picking the
#   shallowest match, then renames into ``MMPR-Tiny/images/`` so the
#   GRPO loader's expected layout is canonical.
# - The validation pass writes a preview JSONL whose rows are NOT
#   consumed by the GRPO loader (which still reads the parquet directly).
#   The preview exists for inspection and CI sanity-checks; we keep
#   producing it because operators rely on the row-level diagnostics.
# - Image-path normalization handles backslashes, leading slashes, and
#   variants where the path is absolute, prefixed with ``MMPR-Tiny/``,
#   prefixed with ``images/``, or just a filename. All resolve into
#   ``MMPR-Tiny/images/<name>`` form.


def _normalize_text(value: Any) -> str:
    """Strip + str-coerce, treating pandas NA as empty string."""
    if value is None:
        return ""
    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except (ImportError, TypeError):
        pass
    return str(value).strip()


def _flatten_prompt_content(content: Any) -> str:
    """Pull text out of a chat-style content list or string."""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            if isinstance(item.get("text"), str):
                text = item["text"].strip()
            elif isinstance(item.get("content"), str):
                text = item["content"].strip()
            else:
                text = ""
        else:
            text = ""
        if text:
            text_parts.append(text)

    return "\n".join(text_parts).strip()


def _extract_question(prompt_value: Any) -> str:
    """Find the first user-role message and return its flattened text."""
    if not isinstance(prompt_value, list):
        return ""
    for message in prompt_value:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        question = _flatten_prompt_content(message.get("content"))
        if question:
            return question
    return ""


def _extract_answer(reward_model_value: Any) -> str:
    if not isinstance(reward_model_value, dict):
        return ""
    return _normalize_text(reward_model_value.get("ground_truth"))


def _normalize_cache_image_path(path_value: str) -> str:
    """Resolve any flavor of input image path into ``MMPR-Tiny/images/...``."""
    normalized = path_value.replace("\\", "/").strip().lstrip("/")
    if not normalized:
        return ""

    if "/MMPR-Tiny/" in normalized:
        normalized = "MMPR-Tiny/" + normalized.split("/MMPR-Tiny/", 1)[1]
    elif normalized.startswith("MMPR-Tiny/"):
        pass
    elif normalized.startswith("images/"):
        normalized = f"MMPR-Tiny/{normalized}"
    elif "/images/" in normalized:
        normalized = "MMPR-Tiny/images/" + normalized.split("/images/", 1)[1]
    else:
        normalized = f"MMPR-Tiny/images/{PurePosixPath(normalized).name}"

    return PurePosixPath(normalized).as_posix()


def _extract_first_image_path(images_value: Any) -> tuple[str | None, int]:
    """Return ``(normalized_path, total_image_count)`` from the parquet field."""
    if not isinstance(images_value, list) or not images_value:
        return None, 0

    raw_paths: list[str] = []
    for image_value in images_value:
        if isinstance(image_value, dict) and isinstance(image_value.get("path"), str):
            raw_paths.append(image_value["path"])
        elif isinstance(image_value, str):
            raw_paths.append(image_value)

    if not raw_paths:
        return None, 0

    normalized_path = _normalize_cache_image_path(raw_paths[0])
    if not normalized_path:
        return None, len(raw_paths)
    return normalized_path, len(raw_paths)


def _find_extracted_images_dir(temp_dir: Path) -> Path:
    """Locate the ``images/`` directory inside the extracted zip tree."""
    direct_candidates = [
        temp_dir / "images",
        temp_dir / "MMPR-Tiny" / "images",
    ]
    for candidate in direct_candidates:
        if candidate.is_dir():
            return candidate

    candidates = sorted(
        (path for path in temp_dir.rglob("images") if path.is_dir()),
        key=lambda path: (len(path.parts), str(path)),
    )
    if not candidates:
        raise FileNotFoundError("Could not locate extracted `images/` directory in zip.")
    return candidates[0]


def _extract_images_if_needed(images_zip_path: Path, output_dir: Path) -> Path:
    """Idempotent unzip into ``<output_dir>/MMPR-Tiny/images/``."""
    images_dir = output_dir / "MMPR-Tiny" / "images"
    if images_dir.exists():
        logger.info("Images already extracted at %s", images_dir)
        return images_dir

    temp_dir = output_dir / "_mmpr_tiny_extract_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting %s to %s", images_zip_path, images_dir)
    with zipfile.ZipFile(images_zip_path) as zf:
        zf.extractall(temp_dir)

    extracted_images_dir = _find_extracted_images_dir(temp_dir)
    images_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(extracted_images_dir), str(images_dir))
    shutil.rmtree(temp_dir, ignore_errors=True)
    logger.info("Finished extracting images.")
    return images_dir


def _copy_parquet(input_parquet_path: Path, output_dir: Path) -> Path:
    output_parquet_path = output_dir / "mmpr_tiny.parquet"
    shutil.copy2(input_parquet_path, output_parquet_path)
    return output_parquet_path


def _build_preview_jsonl(parquet_path: Path, output_dir: Path, preview_path: Path) -> dict[str, int]:
    """Iterate parquet rows, validate against on-disk images, write preview JSONL.

    The preview is for inspection only — the GRPO loader reads the parquet
    file directly. This pass surfaces row-level data-quality issues
    (missing question/answer fields, missing image files) via the returned
    stats dict, which is also persisted in the prep summary.
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    stats: Counter[str] = Counter()
    stats["total_rows"] = len(df)

    preview_df = df[["prompt", "reward_model", "images"]]
    with preview_path.open("w", encoding="utf-8") as fout:
        for prompt_value, reward_model_value, images_value in preview_df.itertuples(
            index=False, name=None
        ):
            question = _extract_question(prompt_value)
            if not question:
                stats["missing_question"] += 1

            answer = _extract_answer(reward_model_value)
            if not answer:
                stats["missing_answer"] += 1

            image_rel_path, image_count = _extract_first_image_path(images_value)
            if image_rel_path is None:
                stats["invalid_image_field"] += 1
                continue

            if image_count > 1:
                stats["multi_image_rows"] += 1

            image_abs_path = output_dir / image_rel_path
            if not image_abs_path.exists():
                stats["missing_image_file"] += 1
                continue

            preview_record = {
                "images": [image_rel_path],
                "question": question,
                "answer": answer,
                "task_name": "mmpr_tiny",
            }
            fout.write(json.dumps(preview_record, ensure_ascii=False) + "\n")
            stats["preview_rows"] += 1

    return dict(stats)


def _write_ready_marker(output_dir: Path) -> Path:
    ready_marker = output_dir / ".mmpr_ready"
    ready_marker.write_text("ready\n", encoding="utf-8")
    return ready_marker


# =============================================================================
# Stage
# =============================================================================


class VlmPreferencePrepStage(pipelines_v1.Stage[VlmPreferencePrepWorkItem, VlmPreferencePrepWorkItem]):
    """Build the VLM-preference cache layout NeMo-RL's response datasets consume.

    Single-worker, single-stage. Idempotent via ReceiptManager.

    For ``flavor="tiny"``: vendored logic, unzip + copy + validate + sentinel.
    For ``flavor="mpo"``: shells out to a configured ``builder_command`` and
    verifies the expected meta JSON appeared. Both paths share receipt
    semantics so the framework view is uniform.
    """

    def __init__(
        self,
        stage_config: VlmPreferencePrepStageConfig,
        pipeline_context: PipelineContext,
    ) -> None:
        self._cfg = stage_config
        self._ctx = pipeline_context
        self._output_fs = None
        self._receipts: ReceiptManager | None = None

    @property
    def stage_batch_size(self) -> int:
        """One work item per call — these are coarse, per-dataset units."""
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=self._cfg.stage_cpus, gpus=0)

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        self._output_fs, _ = get_filesystem(self._ctx.output_root)
        if self._ctx.run_hash is None:
            raise ValueError(
                "VlmPreferencePrepStage requires a non-None run_hash on "
                "PipelineContext (set via setup_rl_omni_run)."
            )
        self._receipts = ReceiptManager(self._output_fs, self._ctx.run_hash)

    def process_data(
        self, items: list[VlmPreferencePrepWorkItem]
    ) -> list[VlmPreferencePrepWorkItem]:
        for item in items:
            self._process_item(item)
        return items

    def _process_item(self, item: VlmPreferencePrepWorkItem) -> None:
        assert self._receipts is not None
        receipts = self._receipts
        rpath = receipts.receipt_path(item.receipts_dir, 0)
        output_dir = Path(item.output_dir)
        ready_marker = output_dir / ".mmpr_ready"
        meta_path = output_dir / item.meta_name

        def verify() -> bool:
            """Confirm the prior run's published outputs are still on disk.

            Receipt-only completion isn't enough: an operator could have
            partially cleaned up the cache after the previous run. We
            re-check every file the receipt recorded to make sure the
            cache is actually usable.
            """
            try:
                r = read_json(self._output_fs, rpath)
                files = r.get("files") or {}
                # Tiny flavor must have parquet + extracted images + sentinel.
                # MPO flavor must have meta_path. We read the receipt's flavor
                # rather than re-deriving from item to handle the case where
                # a previous run was a different flavor against the same dir.
                receipt_flavor = r.get("flavor") or item.flavor
                if receipt_flavor == "tiny":
                    required = [
                        files.get("ready_marker") or str(ready_marker),
                        files.get("parquet_path"),
                        files.get("images_dir"),
                    ]
                elif receipt_flavor == "mpo":
                    required = [files.get("meta_path") or str(meta_path)]
                else:
                    return False
                if any(p is None for p in required):
                    return False
                return all(self._output_fs.exists(p) for p in required)
            except Exception:
                return False

        if receipts.is_completed(rpath, item.plan_hash, verify_outputs=verify):
            logger.info(
                "VlmPreferencePrepStage: skipping completed prep at %s", output_dir
            )
            return

        meta = dict(plan_hash=item.plan_hash, shard_index=0, flavor=item.flavor)
        receipts.write_started(rpath, **meta)

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            if item.flavor == "tiny":
                stats, files = self._run_tiny(item, output_dir, ready_marker)
            elif item.flavor == "mpo":
                stats, files = self._run_mpo(item, output_dir, meta_path)
            else:
                raise ValueError(f"Unknown MMPR flavor: {item.flavor!r}")

            receipts.write_completed(rpath, stats=stats, files=files, **meta)
        except Exception as e:
            receipts.write_failed(rpath, error=e, **meta)
            raise

    def _run_tiny(
        self,
        item: VlmPreferencePrepWorkItem,
        output_dir: Path,
        ready_marker: Path,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Vendored MMPR-Tiny prep: extract + copy + validate + sentinel."""
        raw_dir = Path(item.raw_dir)
        images_zip = raw_dir / "images.zip"
        source_parquet = raw_dir / "mmpr_tiny.parquet"
        missing = [str(p) for p in (images_zip, source_parquet) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing required MMPR-Tiny inputs: {missing}")

        output_parquet = _copy_parquet(source_parquet, output_dir)
        images_dir = _extract_images_if_needed(images_zip, output_dir)

        preview_path = output_dir / self._cfg.preview_name
        stats = _build_preview_jsonl(output_parquet, output_dir, preview_path)
        marker_path = _write_ready_marker(output_dir)

        summary = {
            "input_dir": str(raw_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
            "parquet_path": str(output_parquet),
            "images_dir": str(images_dir),
            "preview_path": str(preview_path),
            "ready_marker": str(marker_path),
            "totals": stats,
        }
        summary_path = output_dir / self._cfg.summary_name
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Also write a stable ``metadata.json`` next to the published cache.
        # Omni3RLDataArtifact.get_wandb_files looks for this filename; without
        # it, vision artifacts contribute zero uploaded files to W&B.
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        files = {
            "parquet_path": str(output_parquet),
            "images_dir": str(images_dir),
            "preview_path": str(preview_path),
            "ready_marker": str(ready_marker),
            "summary_path": str(summary_path),
            "metadata_path": str(metadata_path),
        }
        return stats, files

    def _run_mpo(
        self,
        item: VlmPreferencePrepWorkItem,
        output_dir: Path,
        meta_path: Path,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """MPO escape hatch: shell out to the upstream prep script.

        TODO: vendor logic from ``prepare_public_mmpr_for_mpo.py`` once that
        script can be brought in-tree. When that lands, this branch becomes
        a vendored implementation matching ``_run_tiny``'s shape.
        """
        if not item.builder_command:
            raise ValueError(
                "VlmPreferencePrepWorkItem(flavor='mpo') requires a "
                "builder_command until the upstream prep logic is vendored."
            )
        command = item.builder_command.format(
            input_dir=str(Path(item.raw_dir).expanduser()),
            output_dir=str(output_dir.expanduser()),
            meta_name=item.meta_name,
        )
        # Use shlex.split rather than ``bash -lc`` so the subprocess does
        # not depend on operator shell state (.bashrc, login shell setup).
        # The command comes from a YAML config not user input, so we don't
        # need shell escape semantics either.
        logger.info("VlmPreferencePrepStage[mpo]: running builder command: %s", command)
        subprocess.check_call(shlex.split(command))
        if not meta_path.exists():
            raise FileNotFoundError(
                f"MPO builder command completed but {meta_path} was not produced."
            )
        files = {"meta_path": str(meta_path), "output_dir": str(output_dir)}
        # Stats are best-effort; we don't read the upstream's summary JSON.
        stats: dict[str, Any] = {"flavor": "mpo", "meta_name": item.meta_name}
        return stats, files


__all__ = [
    "VlmPreferencePrepStage",
    "VlmPreferencePrepStageConfig",
]
