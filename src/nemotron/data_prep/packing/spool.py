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

"""SequenceSpool intermediate format for low-memory ChatSFT processing.

This module defines an append-only intermediate representation for tokenized
sequences (input_ids + loss_mask) that is efficient to write in streaming
fashion and efficient to read for a later "central pack" finalizer step.

Spool layout (per shard):
- tokens.bin   : flat int32 token ids (concatenated)
- masks.bin    : flat uint8 loss masks (concatenated)
- offsets.bin  : uint64 token offsets (start index per sequence)
- lengths.bin  : uint32 sequence lengths (tokens per sequence)
- manifest.json: metadata and validation info written on finalize

Notes:
- This format is designed to minimize Python object overhead (no list-of-lists).
- Random-access reads require a seekable file object (most local/Lustre paths).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from fsspec import AbstractFileSystem

from nemotron.data_prep.filesystem import ensure_dir, read_json


def _join_path(root_dir: str, filename: str) -> str:
    root = root_dir.rstrip("/")
    return f"{root}/{filename}"


def _rename(fs: AbstractFileSystem, src: str, dst: str) -> None:
    try:
        fs.rename(src, dst)
        return
    except Exception:
        pass
    try:
        fs.mv(src, dst)
        return
    except Exception as e:
        raise RuntimeError(f"Failed to rename/move '{src}' -> '{dst}'") from e


def _rm_if_exists(fs: AbstractFileSystem, path: str) -> None:
    try:
        if fs.exists(path):
            try:
                fs.rm(path)
            except Exception:
                fs.delete(path)
    except Exception:
        # Best-effort cleanup only
        pass


def _write_json_atomic(fs: AbstractFileSystem, path: str, payload: dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    parent = str(path).rsplit("/", 1)[0] if "/" in str(path) else ""
    if parent:
        ensure_dir(fs, parent)

    with fs.open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    _rename(fs, tmp_path, path)


@dataclass(frozen=True)
class SequenceSpoolPaths:
    """Concrete paths for a SequenceSpool instance."""

    root_dir: str
    tokens_path: str
    masks_path: str
    offsets_path: str
    lengths_path: str
    manifest_path: str

    @classmethod
    def for_root(cls, root_dir: str) -> "SequenceSpoolPaths":
        root = root_dir.rstrip("/")
        return cls(
            root_dir=root,
            tokens_path=_join_path(root, "tokens.bin"),
            masks_path=_join_path(root, "masks.bin"),
            offsets_path=_join_path(root, "offsets.bin"),
            lengths_path=_join_path(root, "lengths.bin"),
            manifest_path=_join_path(root, "manifest.json"),
        )

    def with_suffix(self, suffix: str) -> "SequenceSpoolPaths":
        return SequenceSpoolPaths(
            root_dir=self.root_dir,
            tokens_path=f"{self.tokens_path}{suffix}",
            masks_path=f"{self.masks_path}{suffix}",
            offsets_path=f"{self.offsets_path}{suffix}",
            lengths_path=f"{self.lengths_path}{suffix}",
            manifest_path=f"{self.manifest_path}{suffix}",
        )

    def tmp(self) -> "SequenceSpoolPaths":
        return self.with_suffix(".tmp")


class SequenceSpoolWriter:
    """Append-only writer for SequenceSpool.

    Usage:
        paths = SequenceSpoolPaths.for_root("/path/to/spool/shard_000000")
        w = SequenceSpoolWriter(fs=output_fs, paths=paths)
        w.append(input_ids, loss_mask)
        ...
        manifest = w.finalize(extra_manifest={"pack_size": 4096})
    """

    def __init__(
        self,
        *,
        fs: AbstractFileSystem,
        paths: SequenceSpoolPaths,
        tokens_dtype: np.dtype = np.dtype("int32"),
        masks_dtype: np.dtype = np.dtype("uint8"),
        overwrite_tmp: bool = True,
    ) -> None:
        self._fs = fs
        self._final_paths = paths
        self._tmp_paths = paths.tmp()
        self._tokens_dtype = np.dtype(tokens_dtype)
        self._masks_dtype = np.dtype(masks_dtype)

        if self._tokens_dtype != np.dtype("int32"):
            raise ValueError(f"tokens_dtype must be int32, got {self._tokens_dtype}")
        if self._masks_dtype != np.dtype("uint8"):
            raise ValueError(f"masks_dtype must be uint8, got {self._masks_dtype}")

        ensure_dir(self._fs, self._final_paths.root_dir)

        if overwrite_tmp:
            _rm_if_exists(self._fs, self._tmp_paths.tokens_path)
            _rm_if_exists(self._fs, self._tmp_paths.masks_path)
            _rm_if_exists(self._fs, self._tmp_paths.offsets_path)
            _rm_if_exists(self._fs, self._tmp_paths.lengths_path)
            _rm_if_exists(self._fs, self._tmp_paths.manifest_path)

        # Open append handles once (faster than per-append open/close).
        self._tokens_f = self._fs.open(self._tmp_paths.tokens_path, "ab")
        self._masks_f = self._fs.open(self._tmp_paths.masks_path, "ab")
        self._offsets_f = self._fs.open(self._tmp_paths.offsets_path, "ab")
        self._lengths_f = self._fs.open(self._tmp_paths.lengths_path, "ab")

        self._num_sequences = 0
        self._total_tokens = 0
        self._closed = False

    @property
    def num_sequences(self) -> int:
        return self._num_sequences

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def append(self, input_ids: np.ndarray | list[int], loss_mask: np.ndarray | list[int] | None) -> None:
        if self._closed:
            raise RuntimeError("SequenceSpoolWriter is closed")

        input_arr = np.asarray(input_ids, dtype=self._tokens_dtype)
        if input_arr.ndim != 1:
            raise ValueError(f"input_ids must be 1D, got shape={input_arr.shape}")

        if input_arr.size == 0:
            return

        if loss_mask is None:
            mask_arr = np.ones((input_arr.size,), dtype=self._masks_dtype)
        else:
            mask_arr = np.asarray(loss_mask, dtype=self._masks_dtype)

        if mask_arr.ndim != 1:
            raise ValueError(f"loss_mask must be 1D, got shape={mask_arr.shape}")
        if mask_arr.size != input_arr.size:
            raise ValueError(
                f"loss_mask length mismatch: loss_mask={mask_arr.size}, input_ids={input_arr.size}"
            )

        offset = np.array([self._total_tokens], dtype=np.uint64)
        length = np.array([input_arr.size], dtype=np.uint32)

        # Write offsets/lengths first (small), then bulk token/mask bytes.
        self._offsets_f.write(offset.tobytes(order="C"))
        self._lengths_f.write(length.tobytes(order="C"))
        self._tokens_f.write(input_arr.tobytes(order="C"))
        self._masks_f.write(mask_arr.tobytes(order="C"))

        self._num_sequences += 1
        self._total_tokens += int(input_arr.size)

    def finalize(self, *, extra_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("SequenceSpoolWriter is already finalized/closed")

        self._closed = True

        # Close all handles before rename.
        try:
            self._tokens_f.close()
        finally:
            try:
                self._masks_f.close()
            finally:
                try:
                    self._offsets_f.close()
                finally:
                    self._lengths_f.close()

        # Promote tmp files to final names.
        _rename(self._fs, self._tmp_paths.tokens_path, self._final_paths.tokens_path)
        _rename(self._fs, self._tmp_paths.masks_path, self._final_paths.masks_path)
        _rename(self._fs, self._tmp_paths.offsets_path, self._final_paths.offsets_path)
        _rename(self._fs, self._tmp_paths.lengths_path, self._final_paths.lengths_path)

        manifest: dict[str, Any] = {
            "version": "spool_v1",
            "num_sequences": int(self._num_sequences),
            "total_tokens": int(self._total_tokens),
            "tokens_dtype": str(np.dtype(self._tokens_dtype)),
            "mask_dtype": str(np.dtype(self._masks_dtype)),
            "offsets_dtype": "uint64",
            "lengths_dtype": "uint32",
        }
        if extra_manifest:
            manifest.update(extra_manifest)

        _write_json_atomic(self._fs, self._final_paths.manifest_path, manifest)
        return manifest


class SequenceSpoolReader:
    """Reader for SequenceSpool.

    This reader supports loading offsets/lengths and reading arbitrary sequences
    by index using byte-range seeks into tokens.bin and masks.bin.
    """

    def __init__(
        self,
        *,
        fs: AbstractFileSystem,
        paths: SequenceSpoolPaths,
        tokens_dtype: np.dtype = np.dtype("int32"),
        masks_dtype: np.dtype = np.dtype("uint8"),
    ) -> None:
        self._fs = fs
        self._paths = paths
        self._tokens_dtype = np.dtype(tokens_dtype)
        self._masks_dtype = np.dtype(masks_dtype)

        self._offsets: np.ndarray | None = None
        self._lengths: np.ndarray | None = None
        self._tokens_f = None
        self._masks_f = None

    def read_manifest(self) -> dict[str, Any] | None:
        try:
            if not self._fs.exists(self._paths.manifest_path):
                return None
            return read_json(self._fs, self._paths.manifest_path)
        except Exception:
            return None

    def load_offsets_and_lengths(self) -> tuple[np.ndarray, np.ndarray]:
        if self._offsets is not None and self._lengths is not None:
            return self._offsets, self._lengths

        with self._fs.open(self._paths.offsets_path, "rb") as f:
            offsets_bytes = f.read()
        with self._fs.open(self._paths.lengths_path, "rb") as f:
            lengths_bytes = f.read()

        offsets = np.frombuffer(offsets_bytes, dtype=np.uint64)
        lengths = np.frombuffer(lengths_bytes, dtype=np.uint32)

        if offsets.shape[0] != lengths.shape[0]:
            raise ValueError(
                f"Spool offsets/lengths mismatch: offsets={offsets.shape[0]}, lengths={lengths.shape[0]}"
            )

        self._offsets = offsets
        self._lengths = lengths
        return offsets, lengths

    @property
    def num_sequences(self) -> int:
        _, lengths = self.load_offsets_and_lengths()
        return int(lengths.shape[0])

    @property
    def total_tokens(self) -> int:
        _, lengths = self.load_offsets_and_lengths()
        return int(lengths.sum())

    def _ensure_open(self) -> None:
        if self._tokens_f is None:
            self._tokens_f = self._fs.open(self._paths.tokens_path, "rb")
        if self._masks_f is None:
            self._masks_f = self._fs.open(self._paths.masks_path, "rb")

        # Validate seek support (required for random access reads).
        if not hasattr(self._tokens_f, "seek") or not hasattr(self._masks_f, "seek"):
            raise RuntimeError("SequenceSpoolReader requires seekable file objects")

    def close(self) -> None:
        if self._tokens_f is not None:
            try:
                self._tokens_f.close()
            finally:
                self._tokens_f = None
        if self._masks_f is not None:
            try:
                self._masks_f.close()
            finally:
                self._masks_f = None

    def read_sequence(self, seq_index: int) -> tuple[np.ndarray, np.ndarray]:
        offsets, lengths = self.load_offsets_and_lengths()

        if seq_index < 0 or seq_index >= int(lengths.shape[0]):
            raise IndexError(f"seq_index out of range: {seq_index}")

        self._ensure_open()

        offset_tokens = int(offsets[seq_index])
        length_tokens = int(lengths[seq_index])

        tok_byte_offset = offset_tokens * self._tokens_dtype.itemsize
        tok_byte_len = length_tokens * self._tokens_dtype.itemsize

        mask_byte_offset = offset_tokens * self._masks_dtype.itemsize
        mask_byte_len = length_tokens * self._masks_dtype.itemsize

        self._tokens_f.seek(tok_byte_offset)
        tok_bytes = self._tokens_f.read(tok_byte_len)

        self._masks_f.seek(mask_byte_offset)
        mask_bytes = self._masks_f.read(mask_byte_len)

        input_ids = np.frombuffer(tok_bytes, dtype=self._tokens_dtype)
        loss_mask = np.frombuffer(mask_bytes, dtype=self._masks_dtype)

        # Defensive validation (helps catch corrupt/incomplete spools early).
        if input_ids.shape[0] != length_tokens:
            raise RuntimeError(
                f"Failed to read tokens for seq_index={seq_index}: got={input_ids.shape[0]}, expected={length_tokens}"
            )
        if loss_mask.shape[0] != length_tokens:
            raise RuntimeError(
                f"Failed to read masks for seq_index={seq_index}: got={loss_mask.shape[0]}, expected={length_tokens}"
            )

        return input_ids, loss_mask


__all__ = [
    "SequenceSpoolPaths",
    "SequenceSpoolWriter",
    "SequenceSpoolReader",
]