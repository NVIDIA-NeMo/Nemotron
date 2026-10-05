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

"""Turn a raw corpus into what the rest of the category can read.

Two things stood between "here is my data" and running the flow, and both were
being pushed onto the user:

**Format.** Corpora arrive as parquet at least as often as JSONL, and the reading
here is thin on purpose — Curator's ``ParquetReaderStage.read_data`` would serve
if it streamed, and calling it does *not* require a cluster, whatever an earlier
version of this docstring claimed. Two narrow differences are why it is not
called. It reads a whole file (``pd.read_parquet(path)`` per path, then
``concat``) where this streams row-group by row-group, and a corpus shard is
routinely larger than the memory of the machine someone first tries this on. And
``pd.read_json(lines=True)`` raises on the first malformed line, taking the file
down; here an unparsable line is counted and reported, because a corpus with
forty bad lines in ten million should be describable, not fatal.

**Identity.** ``subset`` and ``decontamination`` are statements about *sets of
document ids*, and most web corpora carry none. This is the part Curator really
does not cover. Its readers can generate ids, but ``_generate_ids_func`` assigns
``np.arange(min_id, min_id + num_rows)`` from a Ray actor: positional, so
resharding renames every document and any claim made about the old ids silently
becomes false — and cluster-bound, so ids cannot be minted before one exists. So
an id is minted from content instead: reshard, reorder, re-split, and it does not
move.

That choice has one consequence this module refuses to hide. Two documents with
the same configured identity fields mint the *same* id. The recipe is required
to include text, so this includes byte-identical text; real corpora contain it —
328 of 20,000 in one Hindi corpus measured here, the largest group 293 copies.
What to do about it is a decision with three defensible answers. It is the
caller's, not this module's: see :data:`ON_DUPLICATE`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

#: Bumped when a change would alter the id a given document receives.
SCHEMA_VERSION = 2

#: How an id is derived when the corpus carries none. Recorded in the ingest
#: report so a later run can reproduce it, and so an id can be traced back to the
#: fields it came from rather than being an opaque string.
ID_RECIPE = "sha256(canonical JSON array of field values), prefixed"

#: What to do when two documents mint or provide the same id. There is no safe
#: default: dropping changes the corpus, suffixing makes ids no longer purely
#: content-derived, and refusing stops a run over something many corpora simply
#: contain. So the caller chooses, and the choice is recorded.
ON_DUPLICATE = ("refuse", "drop", "suffix")

FORMATS = ("jsonl", "parquet")


class IngestError(ValueError):
    """The corpus cannot be ingested as specified."""


def detect_format(paths: list[str]) -> str:
    """Infer the corpus format from its file extensions.

    Refuses a mixed set rather than guessing: two formats under one glob is
    usually a stray file, and silently reading half a corpus is worse than
    stopping.
    """
    suffixes = {Path(p).suffix.lower() for p in paths}
    parquet = {".parquet", ".pq"}
    jsonl = {".jsonl", ".json", ".ndjson"}

    if suffixes <= parquet and suffixes:
        return "parquet"
    if suffixes <= jsonl and suffixes:
        return "jsonl"
    raise IngestError(
        f"cannot infer one format from extensions {sorted(suffixes)}. Set ingest.format "
        "explicitly, or narrow the glob — reading only the files that happen to match "
        "would describe a corpus you did not ask for."
    )


def iter_jsonl(path: str) -> Iterator[dict[str, Any] | None]:
    """Yield mappings and use ``None`` to expose valid non-object JSON values."""
    with Path(path).open("rb") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except (json.JSONDecodeError, UnicodeDecodeError):
                yield {"__unparsable__": True}
                continue
            if isinstance(record, dict):
                yield record
            else:
                # A scalar or list is valid JSON but not a corpus record. The
                # caller must count it; silently yielding nothing loses data
                # without putting the loss in the ingest report.
                yield None


def iter_parquet(path: str, batch_size: int = 8192) -> Iterator[dict[str, Any]]:
    """Stream a parquet file row-group by row-group.

    Streamed rather than loaded whole: a corpus shard is routinely larger than
    memory, and an ingest step that only works on small inputs is not one.
    """
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def mint_id(record: dict[str, Any], fields: list[str], prefix: str) -> str:
    """A content-derived identifier that survives resharding.

    ``fields`` is part of the recipe, not an implementation detail: an id built
    from text alone and one built from url+text are different ids, and a corpus
    re-ingested under a different recipe is a corpus whose ids mean something
    else. The recipe is recorded alongside the output.
    """
    # A delimiter is not framing: ["a\nb", "c"] and ["a", "b\nc"] produce
    # the same newline-joined bytes. JSON preserves field boundaries and value
    # types, while sort_keys makes nested mapping values deterministic.
    payload = json.dumps(
        [record.get(field) for field in fields],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    # The corpus can hold billions of documents. A 64-bit truncated digest has
    # a meaningful birthday-collision probability at that scale and would make
    # unrelated documents look like duplicates, so retain the full SHA-256.
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{prefix}{digest}" if prefix else digest


def normalise(
    record: dict[str, Any],
    *,
    text_field: str,
    id_field: str | None,
    source_field: str | None,
    source_value: str | None,
    keep: list[str],
    id_fields: list[str],
    id_prefix: str,
) -> dict[str, Any] | None:
    """One raw record as the rest of the category expects it.

    Returns ``None`` for a record with no usable text — counted by the caller
    rather than silently skipped, because a corpus that loses a third of itself
    at ingestion should say so before anything measures the remainder.
    """
    text = record.get(text_field)
    if not isinstance(text, str) or not text:
        return None

    out: dict[str, Any] = {"text": text}
    if id_field:
        raw = record.get(id_field)
        if raw is None or str(raw).strip() == "":
            return None
        out["id"] = str(raw)
    else:
        out["id"] = mint_id(record, id_fields, id_prefix)

    if source_field:
        out["source"] = str(record.get(source_field) or "unknown")
    elif source_value:
        out["source"] = source_value

    for field in keep:
        if field in record and field not in out:
            value = record[field]
            # Parquet carries real datetimes; JSON does not.
            out[field] = value.isoformat() if hasattr(value, "isoformat") else value
    return out
