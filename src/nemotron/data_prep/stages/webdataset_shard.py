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

"""
WebDataset Shard Stage — per-shard tar build for Energon WebDatasets.

Takes a pre-grouped batch of (video, audio, conversation) triples and writes
one Energon-compatible tar shard. Idempotent via ReceiptManager: a shard with
a completed receipt and a verifiable tar on disk is skipped on resume.

Requires the ``webdataset`` package (install via ``nemotron[audio]``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cosmos_xenna.pipelines.v1 as pipelines_v1

from nemotron.data_prep.core.receipt import ReceiptManager
from nemotron.data_prep.core.work_items import WebDatasetShardWorkItem
from nemotron.data_prep.stages.context import PipelineContext
from nemotron.data_prep.utils.filesystem import get_filesystem, read_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebDatasetShardStageConfig:
    """Configuration for WebDatasetShardStage.

    Attributes:
        stage_cpus: CPU request per worker. Tar building is I/O bound, not
            CPU bound — 1.0 is plenty. Default 1.0.
    """

    stage_cpus: float = 1.0

    def __post_init__(self) -> None:
        if self.stage_cpus <= 0:
            raise ValueError(f"stage_cpus must be positive, got {self.stage_cpus}")


def _shard_filename(split: str, shard_index: int) -> str:
    """Per-shard tar filename in the Energon flat layout."""
    return f"{split}-shard-{shard_index:06d}.tar"


class WebDatasetShardStage(pipelines_v1.Stage[WebDatasetShardWorkItem, WebDatasetShardWorkItem]):
    """Build one Energon WebDataset tar shard per work item.

    Each tar contains three files per sample:
        - conversation.json (ChatML conversation)
        - audio.wav         (16 kHz mono PCM)
        - video.mp4         (raw clip)

    Receipt lifecycle: started → completed (or failed). On resume, completed
    receipts whose tar still exists on disk are skipped without rebuilding.

    Args:
        stage_config: Stage-specific configuration (WebDatasetShardStageConfig).
        pipeline_context: Shared runtime context (PipelineContext).
    """

    def __init__(
        self,
        stage_config: WebDatasetShardStageConfig,
        pipeline_context: PipelineContext,
    ) -> None:
        self._cfg = stage_config
        self._ctx = pipeline_context
        self._output_fs = None
        self._receipts: ReceiptManager | None = None
        self._wds = None  # webdataset module, resolved in setup()

    @property
    def stage_batch_size(self) -> int:
        """One shard per call — shards are independent."""
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=self._cfg.stage_cpus, gpus=0)

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        self._output_fs, _ = get_filesystem(self._ctx.output_root)
        self._receipts = ReceiptManager(self._output_fs, self._ctx.run_hash)
        # Fail fast at worker init if the audio extras aren't installed,
        # rather than after a multi-hour audio pipeline finishes.
        try:
            import webdataset as wds
        except ImportError as e:
            raise ImportError(
                "WebDatasetShardStage requires the `webdataset` package. "
                "Install the audio extras: `pip install 'nemotron[audio]'`."
            ) from e
        self._wds = wds

    def process_data(self, tasks: list[WebDatasetShardWorkItem]) -> list[WebDatasetShardWorkItem]:
        for task in tasks:
            self._process_shard(task)
        return tasks

    def _process_shard(self, task: WebDatasetShardWorkItem) -> None:
        assert self._receipts is not None
        receipts = self._receipts
        rpath = receipts.receipt_path(task.receipts_dir, task.shard_index)
        shard_name = _shard_filename(task.split, task.shard_index)
        shard_path = f"{task.output_dir.rstrip('/')}/{shard_name}"

        def verify() -> bool:
            try:
                r = read_json(self._output_fs, rpath)
                expected = (r.get("files") or {}).get("shard_path")
                if not expected:
                    return False
                return self._output_fs.exists(expected)
            except Exception:
                return False

        if receipts.is_completed(rpath, task.plan_hash, verify_outputs=verify):
            return

        meta = dict(plan_hash=task.plan_hash, shard_index=task.shard_index, split=task.split)
        receipts.write_started(rpath, **meta)

        try:
            n_written = self._write_one_shard(shard_path, task.records)
            receipts.write_completed(
                rpath,
                stats={"num_samples": n_written, "split": task.split},
                files={"shard_path": shard_path, "shard_name": shard_name},
                **meta,
            )
        except Exception as e:
            receipts.write_failed(rpath, error=e, **meta)
            raise

    def _write_one_shard(self, shard_path: str, records: tuple[tuple[str, str, str], ...]) -> int:
        """Write a single Energon WebDataset tar to ``shard_path``.

        Each ``records`` entry is ``(video_path, audio_path, conversation_json)``.
        Returns the number of samples actually written (skips entries with
        missing files; per-sample failures are logged not raised).
        """
        # webdataset's TarWriter takes a local path; we rely on the output_fs
        # being a local filesystem (Energon datasets live on shared cluster
        # storage, not S3, so this is reasonable for the foreseeable future).
        from pathlib import Path as _Path

        local_path = shard_path
        if local_path.startswith("file://"):
            local_path = local_path[7:]
        _Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        assert self._wds is not None, "setup() must be called before _write_one_shard"
        written = 0
        with self._wds.TarWriter(local_path) as sink:
            for sample_index, (video_path, audio_path, conversation_json) in enumerate(records):
                try:
                    with open(video_path, "rb") as vf:
                        video_bytes = vf.read()
                    with open(audio_path, "rb") as af:
                        audio_bytes = af.read()
                except FileNotFoundError as e:
                    logger.warning("Skipping sample %d in %s: %s", sample_index, shard_path, e)
                    continue

                key = f"{sample_index:08d}"
                sink.write({
                    "__key__": key,
                    "conversation.json": conversation_json.encode("utf-8"),
                    "video.mp4": video_bytes,
                    "audio.wav": audio_bytes,
                })
                written += 1
        return written


__all__ = ["WebDatasetShardStage", "WebDatasetShardStageConfig"]
