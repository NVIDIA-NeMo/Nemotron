# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Append-only, hash-verified persistence for executable oracle episodes."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_contract import (
    ExecutableEpisode,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_errors import (
    ToolTraceCacheConflictError,
    ToolTraceCacheError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.tool_trace_contract import (
    TOOL_TRACE_CACHE_CONTRACT_VERSION,
    ToolTraceRequest,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

try:  # pragma: no cover - BFCL production targets are POSIX.
    import fcntl
except ImportError:  # pragma: no cover - Windows has no advisory file locking.
    fcntl = None  # type: ignore[assignment]

_THREAD_LOCK = threading.RLock()
_RECORD_TYPES = frozenset({"request", "completion"})
RecordKey = tuple[str, str]


def _sha256_json(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


class _Located(NamedTuple):
    offset: int
    length: int
    record_hash: str


class PublishedTurn(NamedTuple):
    """One candidate observation an episode retained, without its payload."""

    request_hash: str
    call_status: str
    response_hash: str | None


class PublishedEpisode(NamedTuple):
    """The identity and candidate observations of one complete episode."""

    candidate_alias: str
    task_id: str
    episode_hash: str
    turns: tuple[PublishedTurn, ...]


class ToolTraceCache:
    """Persist each complete executable episode without re-executing its calls.

    A cache hit replays the whole episode, not an individual call. Skipping only
    a cached mutating call would fail to reproduce oracle state for later calls,
    final-state capture, and assertions.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock_path = path.with_suffix(f"{path.suffix}.lock")
        self._located: dict[RecordKey, _Located] = {}
        self._requests: dict[str, ToolTraceRequest] = {}
        self._offset = 0
        self._records_seen = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _THREAD_LOCK, _file_lock(self._lock_path):
            self._sync()

    def get(self, request: ToolTraceRequest) -> ExecutableEpisode | None:
        """Return immutable evidence as replayed, or claim no observation exists."""
        with _THREAD_LOCK, _file_lock(self._lock_path):
            self._sync()
            located = self._located.get(("completion", request.trace_key))
            if located is None:
                if request.trace_key in self._requests:
                    raise ToolTraceCacheError(
                        f"tool_trace_cache[{request.trace_key}]",
                        "holds an unfinished executable request",
                        expected="either no observation or one completed executable episode",
                        recovery=(
                            "keep this cache as crash evidence and resume into a new "
                            "eval output directory"
                        ),
                    )
                return None
            record = self._read(located)
        episode = self._episode(record["payload"]["episode"], request.trace_key)
        if not request.accepts(episode):
            raise self._invalid(
                self._records_seen,
                "contains an episode belonging to a different executable request",
            )
        return episode.model_copy(update={"replayed": True})

    def put_request(self, request: ToolTraceRequest) -> bool:
        """Claim a logical episode; False means an identical claim already exists."""
        return self._append(
            "request",
            request.trace_key,
            {"request": request.as_document()},
        )

    def put_completion(
        self,
        request: ToolTraceRequest,
        episode: ExecutableEpisode,
    ) -> bool:
        """Commit one complete episode after checking it belongs to the request."""
        if not request.accepts(episode):
            raise ToolTraceCacheError(
                f"tool_trace_cache[{request.trace_key}]",
                "was given an episode from a different executable request",
                expected="the exact candidate, task, plan, source, oracle, and task spec",
                recovery="write the episode under the ToolTraceRequest used to drive it",
            )
        return self._append(
            "completion",
            request.trace_key,
            {"episode": episode.model_copy(update={"replayed": False}).as_document()},
        )

    def publication_evidence(self) -> Iterator[PublishedEpisode]:
        """Stream every complete episode, refusing unfinished or orphan evidence.

        A whole run's episodes do not fit in memory at publication time, so this
        proves completeness eagerly and then yields one episode identity at a
        time, holding only the record it is reading.
        """
        with _THREAD_LOCK, _file_lock(self._lock_path):
            self._sync()
            request_keys = set(self._requests)
            completion_keys = {
                trace_key
                for record_type, trace_key in self._located
                if record_type == "completion"
            }
            if request_keys != completion_keys:
                raise ToolTraceCacheError(
                    self.path.name,
                    "contains unfinished or orphan executable trace evidence",
                    actual={
                        "requests": len(request_keys),
                        "completions": len(completion_keys),
                    },
                    expected="exactly one completion for every tool-trace request",
                    recovery="preserve this cache as crash evidence and use a new output directory",
                )
            located = tuple(
                (trace_key, self._located[("completion", trace_key)])
                for trace_key in sorted(completion_keys)
            )
        return self._stream(located)

    def _stream(
        self,
        located: tuple[tuple[str, _Located], ...],
    ) -> Iterator[PublishedEpisode]:
        for trace_key, item in located:
            episode = self._episode(self._read(item)["payload"]["episode"], trace_key)
            yield PublishedEpisode(
                candidate_alias=episode.candidate_alias,
                task_id=episode.task_id,
                episode_hash=episode.episode_hash,
                turns=tuple(
                    PublishedTurn(
                        request_hash=turn.request_hash,
                        call_status=turn.call_status,
                        response_hash=turn.response_hash,
                    )
                    for turn in episode.observed
                ),
            )

    @property
    def content_hash(self) -> str | None:
        """Hash the durable cache bytes for an eval manifest."""
        with _THREAD_LOCK, _file_lock(self._lock_path):
            self._sync()
            if not self.path.exists():
                return None
            digest = hashlib.sha256()
            with self.path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return f"sha256:{digest.hexdigest()}"

    def _append(
        self,
        record_type: str,
        trace_key: str,
        payload: dict[str, Any],
    ) -> bool:
        body = {
            "schema_version": TOOL_TRACE_CACHE_CONTRACT_VERSION,
            "record_type": record_type,
            "trace_key": trace_key,
            "payload": payload,
        }
        record = {**body, "record_hash": _sha256_json(body)}
        encoded = (
            json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        ).encode("utf-8")
        with _THREAD_LOCK, _file_lock(self._lock_path):
            self._sync()
            existing = self._located.get((record_type, trace_key))
            if existing is not None:
                if existing.record_hash != record["record_hash"]:
                    raise ToolTraceCacheConflictError(
                        f"tool_trace_cache[{trace_key}]",
                        f"already contains a different {record_type} record",
                        actual=existing.record_hash,
                        expected=record["record_hash"],
                        recovery=(
                            "use a new eval output directory; immutable oracle "
                            "observations may not be replaced"
                        ),
                    )
                return False
            self._ingest(encoded[:-1].decode("utf-8"), offset=self._offset, length=len(encoded))
            descriptor = os.open(
                self.path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o600,
            )
            with os.fdopen(descriptor, "ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            self._offset += len(encoded)
            return True

    def _sync(self) -> None:
        if not self.path.exists():
            return
        size = self.path.stat().st_size
        if size == self._offset:
            return
        if size < self._offset:
            raise ToolTraceCacheError(
                self.path.name,
                "was truncated or replaced while this run was using it",
                expected="an append-only cache that only grows",
                recovery="preserve it as evidence and resume into a new output directory",
            )
        with self.path.open("rb") as handle:
            handle.seek(self._offset)
            tail = handle.read()
        try:
            text = tail.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise self._invalid(self._records_seen + 1, "contains non-UTF-8 bytes") from exc
        if not text.endswith("\n"):
            raise self._invalid(
                self._records_seen + 1,
                "ends without a complete JSONL record terminator",
            )
        offset = self._offset
        for line in text.split("\n")[:-1]:
            length = len(line.encode("utf-8")) + 1
            self._ingest(line, offset=offset, length=length)
            offset += length
        self._offset = offset

    def _ingest(self, line: str, *, offset: int, length: int) -> None:
        self._records_seen += 1
        number = self._records_seen
        if not line.strip():
            raise self._invalid(number, "contains an empty record")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise self._invalid(number, "is truncated or is not JSON") from exc
        if not isinstance(record, dict):
            raise self._invalid(number, "is not a JSON object")
        body = {key: value for key, value in record.items() if key != "record_hash"}
        record_hash = _sha256_json(body)
        if record.get("record_hash") != record_hash:
            raise self._invalid(number, "has an invalid record_hash")
        if record.get("schema_version") != TOOL_TRACE_CACHE_CONTRACT_VERSION:
            raise self._invalid(number, "declares an unsupported cache schema")
        record_type = record.get("record_type")
        trace_key = record.get("trace_key")
        if record_type not in _RECORD_TYPES or not isinstance(trace_key, str):
            raise self._invalid(number, "has no valid record type or trace key")
        key = (record_type, trace_key)
        existing = self._located.get(key)
        if existing is not None:
            if existing.record_hash != record_hash:
                raise ToolTraceCacheConflictError(
                    f"{self.path.name}:{number}",
                    "conflicts with an earlier immutable record",
                    expected="duplicate records to be byte-equivalent",
                    recovery="restore the cache from the eval artifact that committed it",
                )
            return
        self._verify_payload(record, number)
        self._located[key] = _Located(offset, length, record_hash)

    def _verify_payload(self, record: dict[str, Any], number: int) -> None:
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise self._invalid(number, "has no payload object")
        trace_key = record["trace_key"]
        if record["record_type"] == "request":
            request = self._request(payload.get("request"), trace_key)
            self._requests[trace_key] = request
            return
        request = self._requests.get(trace_key)
        if request is None:
            raise self._invalid(number, "completes a request the cache never recorded")
        episode = self._episode(payload.get("episode"), trace_key)
        if not request.accepts(episode):
            raise self._invalid(number, "completes the request with mismatched episode evidence")

    def _request(self, document: Any, trace_key: str) -> ToolTraceRequest:
        if not isinstance(document, dict):
            raise self._invalid(self._records_seen, "has no request document")
        claimed = document.get("trace_key")
        payload = {key: value for key, value in document.items() if key != "trace_key"}
        try:
            request = ToolTraceRequest.model_validate(payload)
        except Exception as exc:
            raise self._invalid(
                self._records_seen,
                "contains a request that violates the tool-trace contract",
            ) from exc
        if claimed != request.trace_key or trace_key != request.trace_key:
            raise self._invalid(self._records_seen, "carries an invalid request trace_key")
        return request

    def _episode(self, document: Any, trace_key: str) -> ExecutableEpisode:
        if not isinstance(document, dict):
            raise self._invalid(self._records_seen, "has no executable episode document")
        claimed = document.get("episode_hash")
        derived = {"episode_hash", "released_tool_results", "released_user_turns"}
        payload = {key: value for key, value in document.items() if key not in derived}
        try:
            episode = ExecutableEpisode.model_validate(payload)
        except Exception as exc:
            raise self._invalid(
                self._records_seen,
                "contains evidence that violates the executable episode contract",
            ) from exc
        if claimed != episode.episode_hash:
            raise self._invalid(self._records_seen, "carries an invalid episode_hash")
        request = self._requests.get(trace_key)
        if request is not None and not request.accepts(episode):
            raise self._invalid(self._records_seen, "episode identity does not match its request")
        return episode

    def _read(self, located: _Located) -> dict[str, Any]:
        with self.path.open("rb") as handle:
            handle.seek(located.offset)
            raw = handle.read(located.length)
        return json.loads(raw.decode("utf-8"))

    def _invalid(self, number: int, problem: str) -> ToolTraceCacheError:
        return ToolTraceCacheError(
            f"{self.path.name}:{number}",
            problem,
            expected="a hash-verified append-only tool-trace record",
            recovery="restore the cache from its committed eval artifact",
        )


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


__all__ = ["PublishedEpisode", "PublishedTurn", "ToolTraceCache"]
