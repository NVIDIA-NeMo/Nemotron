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
Audio Extract Stage — per-video ffmpeg fan-out.

Embarrassingly parallel: one video in, one WAV out, ffmpeg saturates one CPU.
Idempotent — skips work if the target audio already exists. Any per-video
failure is logged and the work item is dropped (downstream stages are robust
to missing audio entries).

The stage requires either a system ``ffmpeg`` on PATH or the
``imageio-ffmpeg`` package (which ships a static binary). Install via the
``nemotron[audio]`` extra to get the latter without needing system packages.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass

import cosmos_xenna.pipelines.v1 as pipelines_v1

from nemotron.data_prep.core.work_items import VideoExtractWorkItem
from nemotron.data_prep.stages.context import PipelineContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioExtractStageConfig:
    """Configuration for AudioExtractStage.

    Attributes:
        stage_cpus: CPU request per worker. ffmpeg saturates a single core,
            so 1.0 is the right answer; 0.5 lets two ffmpeg workers share a
            core if you want to oversubscribe. Default 1.0.
        target_sr: Target sample rate for the extracted audio. Default 16 kHz
            (Nemotron Omni's audio encoder expects 16 kHz mono).
        timeout_sec: Per-video ffmpeg timeout. Default 60s — clips are 10s,
            so anything longer is a hung process.
    """

    stage_cpus: float = 1.0
    target_sr: int = 16000
    timeout_sec: int = 60

    def __post_init__(self) -> None:
        if self.stage_cpus <= 0:
            raise ValueError(f"stage_cpus must be positive, got {self.stage_cpus}")
        if self.target_sr <= 0:
            raise ValueError(f"target_sr must be positive, got {self.target_sr}")
        if self.timeout_sec <= 0:
            raise ValueError(f"timeout_sec must be positive, got {self.timeout_sec}")


def _resolve_ffmpeg() -> str:
    """Return path to an ffmpeg executable.

    Prefers a system ffmpeg on PATH (faster startup, shared libs);
    falls back to imageio-ffmpeg's bundled static binary.
    """
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg
    except ImportError as e:
        raise RuntimeError(
            "AudioExtractStage requires either a system ffmpeg on PATH or the "
            "`imageio-ffmpeg` Python package. Install the audio extras: "
            "`pip install 'nemotron[audio]'`."
        ) from e
    return imageio_ffmpeg.get_ffmpeg_exe()


class AudioExtractStage(pipelines_v1.Stage[VideoExtractWorkItem, VideoExtractWorkItem]):
    """Run ffmpeg per video to produce 16 kHz mono WAV. CPU-bound, embarrassingly parallel.

    Idempotent: re-runs skip videos whose audio already exists. Per-video
    failures (corrupt video, ffmpeg timeout, etc.) are logged and the work
    item is dropped from the output list — downstream shard planning will
    drop any QA records that reference a video without audio.

    Args:
        stage_config: Stage-specific configuration (AudioExtractStageConfig).
        pipeline_context: Shared runtime context (PipelineContext).
    """

    def __init__(
        self,
        stage_config: AudioExtractStageConfig,
        pipeline_context: PipelineContext,
    ) -> None:
        self._cfg = stage_config
        self._ctx = pipeline_context
        self._ffmpeg: str | None = None

    @property
    def stage_batch_size(self) -> int:
        """One video per call — ffmpeg saturates one CPU, no batching benefit."""
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        """CPU-only stage; one ffmpeg process per worker."""
        return pipelines_v1.Resources(cpus=self._cfg.stage_cpus, gpus=0)

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        """Resolve the ffmpeg binary once per worker (sticky for the lifetime)."""
        self._ffmpeg = _resolve_ffmpeg()

    def process_data(self, items: list[VideoExtractWorkItem]) -> list[VideoExtractWorkItem]:
        """Extract audio for each work item; drop failures."""
        out: list[VideoExtractWorkItem] = []
        for item in items:
            from pathlib import Path

            audio = Path(item.audio_path)
            if audio.exists():
                out.append(item)
                continue

            audio.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                self._ffmpeg or _resolve_ffmpeg(),
                "-i", item.video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", str(self._cfg.target_sr),
                "-ac", "1",
                str(audio),
                "-y",
                "-loglevel", "error",
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=self._cfg.timeout_sec)
            except subprocess.TimeoutExpired:
                logger.warning("ffmpeg timed out for %s", item.youtube_id)
                continue
            except Exception as e:  # noqa: BLE001
                logger.warning("ffmpeg failed for %s: %s: %s", item.youtube_id, type(e).__name__, e)
                continue

            if result.returncode != 0 or not audio.exists():
                stderr = (result.stderr or result.stdout or "").strip()[:300]
                logger.warning("ffmpeg non-zero exit for %s: %s", item.youtube_id, stderr)
                continue

            out.append(item)
        return out


__all__ = ["AudioExtractStage", "AudioExtractStageConfig"]
