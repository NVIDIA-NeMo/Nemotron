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

"""Xenna stages for Nemotron data prep."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cosmos_xenna.pipelines.v1 as pipelines_v1
from cosmos_xenna.ray_utils.runtime_envs import RuntimeEnv
import numpy as np


def _get_hf_runtime_env() -> RuntimeEnv:
    """Create RuntimeEnv with HF_HOME and HF_TOKEN for worker processes."""
    env_vars = {}
    if os.environ.get("HF_HOME"):
        env_vars["HF_HOME"] = os.environ["HF_HOME"]
    if os.environ.get("HF_TOKEN"):
        env_vars["HF_TOKEN"] = os.environ["HF_TOKEN"]
    return RuntimeEnv(extra_env_vars=env_vars) if env_vars else RuntimeEnv()

from nemotron.data_prep.chat_sft_shard_core import (
    process_chat_sft_pack_from_spool_core,
    process_chat_sft_shard_core,
    process_chat_sft_spool_core,
)
from nemotron.data_prep.filesystem import ensure_dir, get_filesystem
from nemotron.data_prep.formats.transforms import resolve_hf_placeholders
from nemotron.data_prep.hf_placeholder import HFPlaceholderResolver
from nemotron.data_prep.jsonl_shard_core import process_jsonl_shard_core
from nemotron.data_prep.providers import create_tokenizer
from nemotron.data_prep.shard_processor import process_binidx_shard_core
from nemotron.data_prep.xenna.work_items import (
    ChatSftShardWorkItem,
    ChatSftSpoolWorkItem,
    JsonlShardWorkItem,
    ShardWorkItem,
)


class PretrainShardStage(pipelines_v1.Stage[ShardWorkItem, dict]):
    """Process bin/idx shards using Xenna."""

    def __init__(self, *, resolved_tokenizer: dict, output_root: str) -> None:
        self._resolved_tokenizer = resolved_tokenizer
        self._output_root = output_root
        self._tokenize = None
        self._output_fs = None

    @property
    def stage_batch_size(self) -> int:
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(gpus=0, cpus=1.0)

    @property
    def env_info(self) -> RuntimeEnv:
        return _get_hf_runtime_env()

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        self._tokenize = create_tokenizer(self._resolved_tokenizer)
        self._output_fs, _ = get_filesystem(self._output_root)

    def process_data(self, tasks: list[ShardWorkItem]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for task in tasks:
            stats = process_binidx_shard_core(
                tokenize=self._tokenize,
                text_field=task.text_field,
                min_doc_chars=task.min_doc_chars,
                max_doc_tokens=task.max_doc_tokens,
                dtype=task.dtype,
                max_rows=task.max_rows,
                shard_index=task.shard_index,
                assignment=task.assignment,
                plan_hash=task.plan_hash,
                output_dir=task.output_dir,
                receipts_dir=task.receipts_dir,
                output_fs=self._output_fs,
            )
            results.append(
                {
                    "dataset_name": task.dataset_name,
                    "shard_index": task.shard_index,
                    "stats": stats,
                }
            )
        return results


class JsonlShardStage(pipelines_v1.Stage[JsonlShardWorkItem, dict]):
    """Process JSONL shards using Xenna."""

    def __init__(
        self,
        *,
        output_root: str,
        text_field: str,
        transform,
        compression: str,
        max_rows: int | None,
        resolve_hf_placeholders: bool = False,
    ) -> None:
        self._output_root = output_root
        self._text_field = text_field
        self._transform = transform
        self._compression = compression
        self._max_rows = max_rows
        self._resolve_hf_placeholders = resolve_hf_placeholders
        self._output_fs = None

    @property
    def stage_batch_size(self) -> int:
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(gpus=0, cpus=0.5)

    @property
    def env_info(self) -> RuntimeEnv:
        return _get_hf_runtime_env()

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        self._output_fs, _ = get_filesystem(self._output_root)
        if self._resolve_hf_placeholders:
            resolver = HFPlaceholderResolver.create()
            self._transform = resolve_hf_placeholders(resolver=resolver)

    def process_data(self, tasks: list[JsonlShardWorkItem]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for task in tasks:
            stats = process_jsonl_shard_core(
                shard_index=task.shard_index,
                files=task.assignment.get("files", []),
                output_dir=task.output_dir,
                receipts_dir=task.receipts_dir,
                output_fs=self._output_fs,
                text_field=self._text_field,
                transform=self._transform,
                compression=self._compression,
                max_rows=self._max_rows,
            )
            results.append(
                {
                    "dataset_name": task.dataset_name,
                    "shard_index": task.shard_index,
                    "stats": stats,
                }
            )
        return results


class ChatSftShardStage(pipelines_v1.Stage[ChatSftShardWorkItem, dict]):
    """Process ChatSFT packed shards using Xenna."""

    def __init__(
        self,
        *,
        resolved_tokenizer: dict,
        output_root: str,
        messages_field: str,
        tools_field: str,
        pack_size: int,
        algorithm: str,
        dtype: str,
        chat_template: str | None,
        max_doc_tokens: int | None,
        max_rows: int | None,
        seed: int | None,
        used_in_filter: str | None,
        used_in_field: str,
    ) -> None:
        self._resolved_tokenizer = resolved_tokenizer
        self._output_root = output_root
        self._messages_field = messages_field
        self._tools_field = tools_field
        self._pack_size = pack_size
        self._algorithm = algorithm
        self._dtype = dtype
        self._chat_template = chat_template
        self._max_doc_tokens = max_doc_tokens
        self._max_rows = max_rows
        self._seed = seed
        self._used_in_filter = used_in_filter
        self._used_in_field = used_in_field
        self._tokenizer = None
        self._output_fs = None

    @property
    def stage_batch_size(self) -> int:
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(gpus=0, cpus=1.0)

    @property
    def env_info(self) -> RuntimeEnv:
        return _get_hf_runtime_env()

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._resolved_tokenizer["model"],
            revision=self._resolved_tokenizer.get("resolved_revision"),
            trust_remote_code=self._resolved_tokenizer.get("trust_remote_code", False),
            local_files_only=True,  # Use cached files to avoid HF rate limits
        )
        if self._chat_template:
            if self._chat_template == "nano3":
                template_path = Path(__file__).parent.parent / "templates" / "nano3.jinja"
                with open(template_path) as f:
                    self._tokenizer.chat_template = f.read()
            elif Path(self._chat_template).exists():
                with open(self._chat_template) as f:
                    self._tokenizer.chat_template = f.read()
            else:
                self._tokenizer.chat_template = self._chat_template

        self._output_fs, _ = get_filesystem(self._output_root)

    def process_data(self, tasks: list[ChatSftShardWorkItem]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for task in tasks:
            stats = process_chat_sft_shard_core(
                shard_index=task.shard_index,
                files=task.assignment.get("files", []),
                output_dir=task.output_dir,
                receipts_dir=task.receipts_dir,
                output_fs=self._output_fs,
                tokenizer=self._tokenizer,
                messages_field=self._messages_field,
                tools_field=self._tools_field,
                pack_size=self._pack_size,
                algorithm=self._algorithm,
                dtype=np.dtype(self._dtype),
                chat_template=None,
                max_doc_tokens=self._max_doc_tokens,
                max_rows=self._max_rows,
                seed=self._seed,
                used_in_filter=self._used_in_filter,
                used_in_field=self._used_in_field,
            )
            results.append(
                {
                    "dataset_name": task.dataset_name,
                    "shard_index": task.shard_index,
                    "stats": stats,
                }
            )
        return results


class ChatSftSpoolStage(pipelines_v1.Stage[ChatSftShardWorkItem, ChatSftSpoolWorkItem]):
    """Tokenize ChatSFT shards into SequenceSpool intermediates (no packing)."""

    def __init__(
        self,
        *,
        resolved_tokenizer: dict,
        output_root: str,
        messages_field: str,
        tools_field: str,
        pack_size: int,
        algorithm: str,
        dtype: str,
        chat_template: str | None,
        max_doc_tokens: int | None,
        max_rows: int | None,
        seed: int | None,
        used_in_filter: str | None,
        used_in_field: str,
    ) -> None:
        self._resolved_tokenizer = resolved_tokenizer
        self._output_root = output_root
        self._messages_field = messages_field
        self._tools_field = tools_field
        self._pack_size = pack_size
        self._algorithm = algorithm
        self._dtype = dtype
        self._chat_template = chat_template
        self._max_doc_tokens = max_doc_tokens
        self._max_rows = max_rows
        self._seed = seed
        self._used_in_filter = used_in_filter
        self._used_in_field = used_in_field
        self._tokenizer = None
        self._output_fs = None

    @property
    def stage_batch_size(self) -> int:
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(gpus=0, cpus=1.0)

    @property
    def env_info(self) -> RuntimeEnv:
        return _get_hf_runtime_env()

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._resolved_tokenizer["model"],
            revision=self._resolved_tokenizer.get("resolved_revision"),
            trust_remote_code=self._resolved_tokenizer.get("trust_remote_code", False),
            local_files_only=True,  # Use cached files to avoid HF rate limits
        )
        if self._chat_template:
            if self._chat_template == "nano3":
                template_path = Path(__file__).parent.parent / "templates" / "nano3.jinja"
                with open(template_path) as f:
                    self._tokenizer.chat_template = f.read()
            elif Path(self._chat_template).exists():
                with open(self._chat_template) as f:
                    self._tokenizer.chat_template = f.read()
            else:
                self._tokenizer.chat_template = self._chat_template

        self._output_fs, _ = get_filesystem(self._output_root)

    def process_data(self, tasks: list[ChatSftShardWorkItem]) -> list[ChatSftSpoolWorkItem]:
        out: list[ChatSftSpoolWorkItem] = []
        for task in tasks:
            shard_id = f"shard_{task.shard_index:06d}"
            spool_dir = f"{task.output_dir.rstrip('/')}/spool/{shard_id}"

            process_chat_sft_spool_core(
                shard_index=task.shard_index,
                files=task.assignment.get("files", []),
                output_dir=task.output_dir,
                receipts_dir=task.receipts_dir,
                spool_dir=spool_dir,
                output_fs=self._output_fs,
                tokenizer=self._tokenizer,
                messages_field=self._messages_field,
                tools_field=self._tools_field,
                pack_size=self._pack_size,
                algorithm=self._algorithm,
                dtype=np.dtype(self._dtype),
                chat_template=None,
                max_doc_tokens=self._max_doc_tokens,
                max_rows=task.max_rows if task.max_rows is not None else self._max_rows,
                seed=self._seed,
                used_in_filter=self._used_in_filter,
                used_in_field=self._used_in_field,
            )

            out.append(
                ChatSftSpoolWorkItem(
                    dataset_name=task.dataset_name,
                    shard_index=task.shard_index,
                    assignment=task.assignment,
                    output_dir=task.output_dir,
                    receipts_dir=task.receipts_dir,
                    spool_dir=spool_dir,
                    max_rows=task.max_rows,
                )
            )
        return out


class ChatSftCentralPackStage(pipelines_v1.Stage[ChatSftSpoolWorkItem, dict]):
    """Pack ChatSFT SequenceSpool intermediates into packed .npy shards (single-worker stage)."""

    def __init__(
        self,
        *,
        output_root: str,
        pack_size: int,
        algorithm: str,
        dtype: str,
        seed: int | None,
    ) -> None:
        self._output_root = output_root
        self._pack_size = pack_size
        self._algorithm = algorithm
        self._dtype = dtype
        self._seed = seed
        self._output_fs = None

    @property
    def stage_batch_size(self) -> int:
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(gpus=0, cpus=1.0)

    @property
    def env_info(self) -> RuntimeEnv:
        return _get_hf_runtime_env()

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        self._output_fs, _ = get_filesystem(self._output_root)

    def process_data(self, tasks: list[ChatSftSpoolWorkItem]) -> list[dict[str, Any]]:
        import gc

        results: list[dict[str, Any]] = []
        for task in tasks:
            stats = process_chat_sft_pack_from_spool_core(
                shard_index=task.shard_index,
                output_dir=task.output_dir,
                receipts_dir=task.receipts_dir,
                spool_dir=task.spool_dir,
                output_fs=self._output_fs,
                pack_size=self._pack_size,
                algorithm=self._algorithm,
                dtype=np.dtype(self._dtype),
                seed=self._seed,
            )
            results.append(
                {
                    "dataset_name": task.dataset_name,
                    "shard_index": task.shard_index,
                    "stats": stats,
                }
            )
            # Force garbage collection after each shard to prevent memory accumulation
            # across sequential tasks in this single-worker stage
            gc.collect()
        return results


class HfPredownloadStage(pipelines_v1.Stage[ShardWorkItem, ShardWorkItem]):
    """Pre-download HuggingFace files for a batch of shards."""

    def __init__(
        self,
        *,
        max_concurrent_downloads: int,
        output_root: str,
        download_timeout_sec: int = 300,
        max_retries: int = 3,
    ) -> None:
        self._max_concurrent_downloads = max_concurrent_downloads
        self._output_root = output_root
        self._download_timeout_sec = download_timeout_sec
        self._max_retries = max_retries
        self._progress_path = None
        self._output_fs = None

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        self._output_fs, base_path = get_filesystem(self._output_root)
        progress_dir = f"{base_path.rstrip('/')}/.xenna/downloads"
        ensure_dir(self._output_fs, progress_dir)
        self._progress_path = f"{progress_dir}/{worker_metadata.worker_id}.json"

    @property
    def stage_batch_size(self) -> int:
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(gpus=0, cpus=0.5)

    @property
    def env_info(self) -> RuntimeEnv:
        return _get_hf_runtime_env()

    def process_data(self, tasks: list[ShardWorkItem]) -> list[ShardWorkItem]:
        if not tasks:
            return []

        unique_files = _collect_unique_hf_files(tasks)
        if not unique_files:
            return tasks

        cache_dir = None
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            cache_dir = os.path.join(hf_home, "hub")

        total_files = len(unique_files)
        max_workers = min(self._max_concurrent_downloads, total_files)
        print(f"[Pre-download] Starting download of {total_files} unique files (max_concurrent={max_workers})")
        start_time = time.perf_counter()
        completed = 0
        last_report = start_time
        self._write_progress(completed, total_files, start_time)

        failed_downloads: list[tuple[dict[str, str], Exception]] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures_to_files = {
                executor.submit(
                    _download_hf_file,
                    file_info["repo_id"],
                    file_info["filename"],
                    file_info["revision"],
                    cache_dir,
                    self._download_timeout_sec,
                    self._max_retries,
                ): file_info
                for file_info in unique_files
            }

            for future in as_completed(futures_to_files):
                file_info = futures_to_files[future]
                try:
                    future.result()
                except Exception as exc:
                    failed_downloads.append((file_info, exc))
                completed += 1
                now = time.perf_counter()
                if now - last_report >= 5.0 or completed == total_files:
                    rate = completed / max(now - start_time, 0.001)
                    print(f"[Pre-download] {completed}/{total_files} files ({rate:.1f}/s)")
                    last_report = now
                    self._write_progress(completed, total_files, start_time)

        # Report and fail on download errors - don't let PretrainShardStage run on missing files
        if failed_downloads:
            print(f"[Pre-download] ERROR: {len(failed_downloads)} downloads failed:")
            for file_info, exc in failed_downloads[:10]:
                print(f"  - {file_info['repo_id']}/{file_info['filename']}: {type(exc).__name__}: {exc}")
            if len(failed_downloads) > 10:
                print(f"  ... and {len(failed_downloads) - 10} more")
            raise RuntimeError(
                f"Pre-download failed: {len(failed_downloads)} files could not be downloaded. "
                "Cannot proceed with tokenization - files would be missing from cache."
            )

        self._write_progress(completed, total_files, start_time)

        return tasks

    def _write_progress(self, completed: int, total: int, start_time: float) -> None:
        if self._output_fs is None or self._progress_path is None:
            return
        elapsed = time.perf_counter() - start_time
        rate = completed / max(elapsed, 0.001)
        payload = {
            "completed": completed,
            "total": total,
            "elapsed_sec": elapsed,
            "rate": rate,
            "updated_at": time.time(),
        }
        try:
            with self._output_fs.open(self._progress_path, "w") as f:
                json.dump(payload, f)
        except Exception:
            pass


def _collect_unique_hf_files(tasks: list[ShardWorkItem]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    unique_files: list[dict[str, str]] = []
    for task in tasks:
        files = task.assignment.get("files", [])
        for file_info in files:
            repo_id = file_info.get("hf_repo_id")
            if repo_id is None:
                continue
            filename = file_info.get("hf_filename")
            revision = file_info.get("hf_revision") or ""
            key = (repo_id, filename, revision)
            if key in seen:
                continue
            seen.add(key)
            unique_files.append(
                {
                    "repo_id": repo_id,
                    "filename": filename,
                    "revision": revision or None,
                }
            )
    return unique_files

def _download_hf_file(
    repo_id: str,
    filename: str,
    revision: str | None,
    cache_dir: str | None,
    timeout_sec: int,
    max_retries: int,
) -> None:
    from huggingface_hub import hf_hub_download

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                repo_type="dataset",
                local_files_only=False,
                cache_dir=cache_dir,
                etag_timeout=timeout_sec,
                download_timeout=timeout_sec,
            )
            return
        except TypeError:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                repo_type="dataset",
                local_files_only=False,
                cache_dir=cache_dir,
            )
            return
        except Exception as exc:
            last_error = exc
            time.sleep(min(5 * attempt, 20))
    if last_error:
        raise last_error
