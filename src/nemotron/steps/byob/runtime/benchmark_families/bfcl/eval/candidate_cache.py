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

"""Append-only, hash-verified cache for candidate HTTP observations."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_contract import (
    CANDIDATE_CLIENT_CONTRACT_VERSION,
    CandidateAttempt,
    CandidateCallOutcome,
    CandidateRequest,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_errors import (
    CandidateCacheConflictError,
    CandidateCacheError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

try:  # pragma: no cover - BFCL production targets are POSIX.
    import fcntl
except ImportError:  # pragma: no cover - Windows has no advisory file locking.
    fcntl = None  # type: ignore[assignment]

_THREAD_LOCK = threading.RLock()
_RECORD_TYPES = frozenset({"request", "attempt", "completion"})

RecordKey = tuple[str, str, int | None]


def _sha256_json(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


class _Located(NamedTuple):
    """Where a verified record lives, so its bytes are read only on demand."""

    offset: int
    length: int
    record_hash: str


class CandidateIOCache:
    """Persist request, attempt, and completion records without replacing any.

    A completion is the replay boundary. Request and attempt records without one
    are retained as crash evidence but are never returned as a cache hit.

    An eval run holds one of these for millions of records, so the cache keeps
    only offsets and hashes in memory and verifies each record exactly once, when
    it first appears. Reads of the response bytes go back to the file.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock_path = path.with_suffix(f"{path.suffix}.lock")
        self._located: dict[RecordKey, _Located] = {}
        self._requests: set[str] = set()
        self._attempt_summaries: dict[str, dict[int, str]] = {}
        self._offset = 0
        self._records_seen = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _THREAD_LOCK, _file_lock(self._lock_path):
            self._sync()

    def get(self, request_hash: str) -> CandidateCallOutcome | None:
        with _THREAD_LOCK, _file_lock(self._lock_path):
            self._sync()
            located = self._located.get(("completion", request_hash, None))
            if located is None:
                if request_hash in self._requests:
                    raise CandidateCacheError(
                        f"candidate_io_cache[{request_hash}]",
                        "holds an unfinished request without a completion marker",
                        expected="either no observation or a completed logical candidate call",
                        recovery=(
                            "let an in-flight call finish, or keep this cache as crash evidence "
                            "and resume into a new eval output directory"
                        ),
                    )
                return None
            record = self._read(located)
        try:
            outcome = CandidateCallOutcome.model_validate(record["payload"]["outcome"])
        except Exception as exc:
            raise CandidateCacheError(
                f"candidate_io_cache[{request_hash}]",
                "contains a completion that does not satisfy the candidate I/O contract",
                expected="a valid CandidateCallOutcome",
                recovery="restore the cache from its committed artifact or start a new eval output directory",
            ) from exc
        return outcome.model_copy(update={"replayed": True})

    def put_request(self, request: CandidateRequest) -> bool:
        """Claim a logical request; False means another writer claimed it first."""
        return self._append(
            "request",
            request.request_hash,
            {"request": request.as_document()},
        )

    def put_attempt(self, request_hash: str, attempt: CandidateAttempt) -> None:
        self._append(
            "attempt",
            request_hash,
            {"attempt": attempt.as_document()},
            attempt_index=attempt.attempt_index,
        )

    def put_completion(self, outcome: CandidateCallOutcome) -> None:
        if outcome.status != "completed" or outcome.response is None:
            raise CandidateCacheError(
                f"candidate_io_cache[{outcome.request_hash}]",
                "cannot cache an outcome without a semantically completed response",
                actual=outcome.status,
                expected="status=completed with a parsed candidate response",
                recovery="leave transient/provider failures uncached so a later run can retry",
            )
        self._append(
            "completion",
            outcome.request_hash,
            {"outcome": outcome.model_copy(update={"replayed": False}).as_document()},
        )

    def _durable_requests(self) -> tuple[set[str], set[str]]:
        with _THREAD_LOCK, _file_lock(self._lock_path):
            self._sync()
            completed = {
                request_hash
                for record_type, request_hash, attempt_index in self._located
                if record_type == "completion" and attempt_index is None
            }
            return set(self._requests), completed

    def validate_complete(self) -> None:
        """Prove no claimed request was left without a durable completion.

        This is all that can be proved when the published run kept no episode
        evidence to compare the cache against.
        """
        requested, completed = self._durable_requests()
        if requested != completed:
            raise CandidateCacheError(
                self.path.name,
                "holds a claimed candidate request without a durable completion",
                actual={"requests": len(requested), "completions": len(completed)},
                expected="one completion for every claimed candidate request",
                recovery=(
                    "keep this cache as crash evidence and resume into a new eval "
                    "output directory"
                ),
            )

    def validate_for_publication(
        self,
        expected: dict[str, tuple[str, str | None]],
    ) -> None:
        """Prove the cache is complete and matches every published episode turn."""
        requested, completed = self._durable_requests()
        if requested != set(expected) or completed != set(expected):
            raise CandidateCacheError(
                self.path.name,
                "does not exactly cover the candidate turns in the published episodes",
                actual={
                    "requests": len(requested),
                    "completions": len(completed),
                    "expected": len(expected),
                },
                expected="one completed cache request for every and only published candidate turn",
                recovery="restore the candidate cache produced by these executable episodes",
            )
        for request_hash, (status, response_hash) in expected.items():
            outcome = self.get(request_hash)
            if outcome is None:  # pragma: no cover - exact completion set proved above.
                raise AssertionError("completed candidate cache request disappeared")
            actual_response_hash = (
                outcome.response.response_hash if outcome.response is not None else None
            )
            if outcome.status != status or actual_response_hash != response_hash:
                raise CandidateCacheError(
                    f"candidate_io_cache[{request_hash}]",
                    "does not match the status or response retained by the executable episode",
                    actual={
                        "status": outcome.status,
                        "response_hash": actual_response_hash,
                    },
                    expected=f"status={status}, response_hash={response_hash}",
                    recovery="restore both caches from the same completed eval run",
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
        request_hash: str,
        payload: dict[str, Any],
        *,
        attempt_index: int | None = None,
    ) -> bool:
        body = {
            "schema_version": CANDIDATE_CLIENT_CONTRACT_VERSION,
            "record_type": record_type,
            "request_hash": request_hash,
            "attempt_index": attempt_index,
            "payload": payload,
        }
        record = {**body, "record_hash": _sha256_json(body)}
        encoded = (
            json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        ).encode("utf-8")
        with _THREAD_LOCK, _file_lock(self._lock_path):
            # Another process may have appended since this instance last looked.
            self._sync()
            existing = self._located.get((record_type, request_hash, attempt_index))
            if existing is not None:
                if existing.record_hash != record["record_hash"]:
                    raise CandidateCacheConflictError(
                        f"candidate_io_cache[{request_hash}]",
                        f"already contains a different {record_type} record",
                        actual=existing.record_hash,
                        expected=record["record_hash"],
                        recovery="use a new eval output directory; immutable observations may not be replaced",
                    )
                return False
            # Verify against the durable neighbours before writing, so a record
            # that violates the contract never reaches the file.
            self._ingest(encoded[:-1].decode("utf-8"), offset=self._offset, length=len(encoded))
            descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            self._offset += len(encoded)
            return True

    def _sync(self) -> None:
        """Verify and index only the bytes appended since the last look."""
        if not self.path.exists():
            return
        size = self.path.stat().st_size
        if size == self._offset:
            return
        if size < self._offset:
            raise CandidateCacheError(
                self.path.name,
                "was truncated or replaced while this run was appending to it",
                expected="an append-only cache that only ever grows",
                recovery="keep the surviving file as evidence and resume into a new eval output directory",
            )
        with self.path.open("rb") as handle:
            handle.seek(self._offset)
            tail = handle.read()
        try:
            text = tail.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CandidateCacheError(
                self.path.name,
                "contains bytes that are not valid UTF-8",
                expected="UTF-8 JSONL records",
                recovery="restore the cache from the eval artifact that committed it",
            ) from exc
        if not text.endswith("\n"):
            raise CandidateCacheError(
                self.path.name,
                "ends without a complete JSONL record terminator",
                expected="every durable record to end in a newline",
                recovery="preserve the cache as crash evidence and resume into a new eval output directory",
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
            raise CandidateCacheError(
                f"{self.path.name}:{number}",
                "contains an empty record",
                expected="one complete JSON object per line",
                recovery="restore the committed cache; do not silently skip a partial observation",
            )
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CandidateCacheError(
                f"{self.path.name}:{number}",
                "is truncated or is not JSON",
                expected="one complete JSON object per line",
                recovery="restore the committed cache; a partial observation is not replayable",
            ) from exc
        if not isinstance(record, dict):
            raise self._invalid(number, "is not a JSON object")
        body = {key: value for key, value in record.items() if key != "record_hash"}
        record_hash = _sha256_json(body)
        if record.get("record_hash") != record_hash:
            raise self._invalid(number, "has an invalid record_hash")
        if record.get("schema_version") != CANDIDATE_CLIENT_CONTRACT_VERSION:
            raise self._invalid(number, "does not declare the candidate cache schema version")
        record_type = record.get("record_type")
        request_hash = record.get("request_hash")
        attempt_index = record.get("attempt_index")
        if record_type not in _RECORD_TYPES or not isinstance(request_hash, str):
            raise self._invalid(number, "has no valid record type or request hash")
        if record_type == "attempt":
            if type(attempt_index) is not int or attempt_index < 0:
                raise self._invalid(number, "has an invalid attempt index")
        elif attempt_index is not None:
            raise self._invalid(number, "carries an attempt index it cannot own")
        key: RecordKey = (record_type, request_hash, attempt_index)
        located = self._located.get(key)
        if located is not None:
            if located.record_hash != record_hash:
                raise CandidateCacheConflictError(
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
        request_hash = record["request_hash"]
        record_type = record["record_type"]
        if record_type == "request":
            self._verify_request(payload, request_hash, number)
            self._requests.add(request_hash)
            return
        if record_type == "attempt":
            summary = self._verify_attempt(payload, record["attempt_index"], number)
            self._attempt_summaries.setdefault(request_hash, {})[record["attempt_index"]] = summary
            return
        self._verify_completion(payload, request_hash, number)

    def _verify_request(self, payload: dict[str, Any], request_hash: str, number: int) -> None:
        try:
            parsed = CandidateRequest.model_validate(payload["request"])
        except Exception as exc:
            raise self._invalid(number, "has a payload that violates its record contract") from exc
        if parsed.request_hash != request_hash:
            raise self._invalid(number, "carries a request that belongs to another request hash")

    def _verify_attempt(self, payload: dict[str, Any], attempt_index: int, number: int) -> str:
        try:
            parsed = CandidateAttempt.model_validate(payload["attempt"])
        except Exception as exc:
            raise self._invalid(number, "has a payload that violates its record contract") from exc
        if parsed.attempt_index != attempt_index:
            raise self._invalid(number, "carries an attempt that reports another attempt index")
        return _sha256_json(parsed.as_summary())

    def _verify_completion(self, payload: dict[str, Any], request_hash: str, number: int) -> None:
        try:
            outcome = CandidateCallOutcome.model_validate(payload["outcome"])
        except Exception as exc:
            raise self._invalid(number, "has a payload that violates its record contract") from exc
        if outcome.request_hash != request_hash:
            raise self._invalid(number, "carries a completion that belongs to another request hash")
        if outcome.status != "completed" or outcome.response is None:
            raise self._invalid(
                number,
                "caches an outcome without a semantically completed response",
            )
        if request_hash not in self._requests:
            raise CandidateCacheError(
                f"{self.path.name}:{number}",
                "completes a request the cache never recorded",
                expected="one request, its contiguous attempts, then a completion",
                recovery="restore the cache from the eval artifact that committed it",
            )
        recorded = self._attempt_summaries.get(request_hash, {})
        cited = {attempt.attempt_index: _sha256_json(attempt.as_summary()) for attempt in outcome.attempts}
        if set(recorded) != set(cited):
            raise CandidateCacheError(
                f"{self.path.name}:{number}",
                "cites attempts the cache never recorded, or omits ones it did",
                expected="the completion to cover the exact contiguous attempt set",
                recovery="restore the cache from the eval artifact that committed it",
            )
        if divergent := sorted(index for index, digest in cited.items() if recorded[index] != digest):
            raise CandidateCacheError(
                f"{self.path.name}:{number}",
                f"restates attempt(s) {divergent} differently from their immutable records",
                expected="the completion to cite every immutable attempt exactly",
                recovery="restore the cache from the eval artifact that committed it",
            )

    def _read(self, located: _Located) -> dict[str, Any]:
        with self.path.open("rb") as handle:
            handle.seek(located.offset)
            raw = handle.read(located.length)
        return json.loads(raw.decode("utf-8"))

    def _invalid(self, number: int, problem: str) -> CandidateCacheError:
        return CandidateCacheError(
            f"{self.path.name}:{number}",
            problem,
            expected="a hash-verified candidate cache record",
            recovery="restore the cache from the eval artifact that committed it",
        )


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Serialize writers across processes on POSIX; the thread lock covers tests."""
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


__all__ = ["CandidateIOCache"]
