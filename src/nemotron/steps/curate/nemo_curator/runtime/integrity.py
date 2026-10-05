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

"""Integrity measurement of a curated corpus, read from the files on disk.

Three properties this module is careful about, because getting them wrong makes
the resulting report worse than no report:

*Readable is not complete.* A shard can parse perfectly and still be missing
rows. Nothing here claims completeness; the caller may only do that against a
producer-emitted manifest.

*A row delta is not an error.* Filtering removes records on purpose. A delta is
reported as an observation, and only a producer's own declaration can turn it
into a finding.

*Digest independence has to be named.* :func:`corpus_digest` is independent of
the order files are enumerated in, and deliberately not independent of their
names — losing that would mean the report could no longer say which shard
changed. Independence from how rows are distributed across shards is a
different, more expensive guarantee, provided by :func:`containment`.
"""

from __future__ import annotations

import glob
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

READ_CHUNK = 1 << 20


class ContainmentConfigError(ValueError):
    """Raised when a containment comparison is requested without a field choice."""


class ReleaseLayoutError(ValueError):
    """Raised when the corpus on disk does not match the layout the caller described."""


class UnreadableCorpusError(ValueError):
    """Raised when a fingerprint was asked for over data this reader cannot read."""


@dataclass
class ShardReport:
    """What a single shard turned out to be."""

    path: str
    size_bytes: int
    row_count: int = 0
    digest: str | None = None
    readable: bool = True
    error: str | None = None
    error_byte_offset: int | None = None
    bad_record_count: int = 0
    also_truncated: bool = False
    per_source: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "row_count": self.row_count,
            "readable": self.readable,
        }
        if self.digest is not None:
            out["digest"] = self.digest
        if self.error is not None:
            out["error"] = self.error
            out["error_byte_offset"] = self.error_byte_offset
            out["bad_record_count"] = self.bad_record_count
            if self.also_truncated:
                out["also_truncated"] = True
        if self.per_source:
            out["per_source"] = dict(sorted(self.per_source.items()))
        return out


def iter_records(
    paths: Iterable[str | Path], damage: dict[str, int] | None = None
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(shard_path, record)`` for every parsable mapping.

    An unparsable line or a JSON value that is not an object is COUNTED into
    ``damage``, never silently skipped. That is a property of the category
    rather than of one step: ``curate/audit`` calls exactly this damage a
    finding, so a reader that stayed quiet about it would describe a corpus its
    own sibling considers broken.

    ``damage`` is optional because ``curate/profile`` reads the corpus twice and
    only the counting pass has somewhere to put the tally.
    """
    for path in paths:
        text = str(path)
        with Path(path).open("rb") as handle:
            for raw in handle:
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    if damage is not None:
                        damage[text] = damage.get(text, 0) + 1
                    continue
                if not isinstance(record, dict):
                    if damage is not None:
                        damage[text] = damage.get(text, 0) + 1
                    continue
                yield text, record


def scan_shard(
    path: str | Path,
    *,
    source_field: str | None = None,
    want_digest: bool = True,
) -> ShardReport:
    """Read one JSONL shard, reporting what is there rather than raising.

    A single pass produces the content digest, the row count, and — when the
    file is damaged — the byte offset where parsing stopped. The offset is the
    part that makes the finding actionable: "shard is corrupt" sends someone
    hunting, "corrupt at byte 4,177,920" points at the write that was cut off.

    A final line with no trailing newline that fails to parse is reported as
    truncated, which is what a writer killed mid-record leaves behind. The same
    failure in the middle of a file is reported as corrupt instead, because a
    complete line following it means the file was not merely cut short.
    """
    p = Path(path)
    report = ShardReport(path=str(p), size_bytes=p.stat().st_size if p.is_file() else 0)

    if not p.is_file():
        report.readable = False
        report.error = "not a file"
        return report

    hasher = hashlib.sha256() if want_digest else None
    offset = 0
    counts: Counter[str] = Counter()

    try:
        with p.open("rb") as handle:
            for raw in handle:
                if hasher is not None:
                    hasher.update(raw)
                line_offset = offset
                offset += len(raw)

                stripped = raw.strip()
                if not stripped:
                    continue

                try:
                    record = json.loads(stripped)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    truncated = not raw.endswith(b"\n")
                    report.readable = False
                    report.bad_record_count += 1
                    # Keep the FIRST fault, not the last. A file corrupted in the
                    # middle and then also truncated at the tail would otherwise
                    # be reported as a clean truncation, sending someone to look
                    # at the end of the file when the damage starts earlier.
                    if report.error is None:
                        report.error = "truncated final record" if truncated else "unparsable record"
                        report.error_byte_offset = line_offset
                    elif truncated:
                        report.also_truncated = True
                    # Keep scanning: the rows already counted are real, and a
                    # caller comparing counts needs them.
                    continue

                if not isinstance(record, dict):
                    report.readable = False
                    report.bad_record_count += 1
                    if report.error is None:
                        report.error = "non-object record"
                        report.error_byte_offset = line_offset
                    continue

                report.row_count += 1
                if source_field is not None:
                    value = record.get(source_field)
                    key = str(value) if value is not None else "__missing__"
                    counts[key] += 1
    except OSError as exc:
        report.readable = False
        report.error = f"unreadable: {exc}"
        return report

    if hasher is not None:
        report.digest = f"sha256:{hasher.hexdigest()}"
    report.per_source = dict(counts)
    return report


def scan_corpus(
    paths: Iterable[str | Path],
    *,
    source_field: str | None = None,
    want_digest: bool = True,
) -> list[ShardReport]:
    """Scan every shard, in sorted path order so two runs agree."""
    return [scan_shard(p, source_field=source_field, want_digest=want_digest) for p in sorted(str(x) for x in paths)]


def corpus_digest(reports: Sequence[ShardReport], *, root: str | Path | None = None) -> str:
    """A digest of the corpus that does not depend on enumeration order.

    Per-shard lines are sorted before hashing, so a filesystem that returns
    files in a different order yields the same value. Shard names are part of
    the input on purpose: a digest blind to names could not tell a reader which
    shard moved.
    """
    base = Path(root).resolve() if root is not None else None
    lines = []
    for report in reports:
        name = report.path
        if base is not None:
            try:
                name = str(Path(report.path).resolve().relative_to(base))
            except ValueError as exc:
                # Falling back to the absolute path would silently mix two naming
                # schemes, and the digest would then change when the corpus moved
                # — reported to a user as "the corpus changed".
                raise ReleaseLayoutError(
                    f"digest_root {base} is not a parent of {report.path}; "
                    "either point it at the corpus root or leave it unset"
                ) from exc
        lines.append(f"{name}\t{report.digest or 'unreadable'}\t{report.size_bytes}")

    joined = "\n".join(sorted(lines))
    return f"sha256:{hashlib.sha256(joined.encode('utf-8')).hexdigest()}"


class RowDigest:
    """Order-independent digest of a corpus's rows, in constant memory.

    ``corpus_digest`` fingerprints *shards*, so it changes when a corpus is
    resharded even though the documents did not. This fingerprints the documents
    themselves, which is what ties a filtering policy to the data its thresholds
    were derived from: reshard the corpus and the answer is unchanged; change one
    document and it is not.

    Keys are **summed** rather than XOR-ed. Under XOR a duplicated pair cancels,
    so a corpus containing a document twice would fingerprint identically to one
    containing it zero times — the exact confusion a duplicate check exists to
    surface.
    """

    __slots__ = ("_total", "_count")

    _MODULUS = 1 << 256

    def __init__(self) -> None:
        self._total = 0
        self._count = 0

    def add(self, key: str) -> None:
        value = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest(), "big")
        self._total = (self._total + value) % self._MODULUS
        self._count += 1

    @property
    def rows(self) -> int:
        return self._count

    def hexdigest(self) -> str:
        # The count is folded in so two corpora cannot collide merely by having
        # key sums that happen to agree modulo 2**256.
        payload = f"{self._count}:{self._total:064x}"
        return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


#: Extensions taken to be corpus when a directory is RECURSED INTO. That happens
#: only when the caller names a bare directory: a glob or an explicit file says
#: what it wants, and second-guessing it would silently drop files the user
#: pointed at. Without the filter a stray ``README.md`` beside the shards would
#: reach :func:`ingest.detect_format` and stop the run.
CORPUS_EXTENSIONS = ("jsonl", "json", "ndjson", "parquet", "pq")

#: What a JSONL reader parses. ``.ndjson`` is included because this category
#: already treats it as JSONL -- :func:`ingest.detect_format` groups it with
#: ``.jsonl`` and ``.json``.
JSONL_READER_EXTENSIONS = (".jsonl", ".json", ".ndjson")

#: Corpus formats that resolve but need ``curate/ingest`` to normalise them first.
INGEST_ONLY_EXTENSIONS = (".parquet", ".pq")


def partition_by_reader(paths: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split resolved paths into (readable as JSONL, needs ingest first).

    Lives here rather than in the filter step so that preflight and the runtime
    apply ONE rule, not merely one resolver. They shared ``expand_inputs`` and
    still disagreed: preflight counted a parquet shard as corpus, and the step
    then dropped it (when JSONL sat beside it) or refused (when it did not). That
    is "validate one corpus, process another" with an extra step, which is the
    thing the shared resolver exists to stop.

    ``run_flow`` must import this without importing the filter step, which pulls
    in Curator at module scope -- preflight has to answer "can this run start?"
    before any cluster exists.

    Anything in neither list is a companion a broad glob swept up -- a README, a
    ``_SUCCESS`` marker -- and is not corpus in either direction.
    """
    readable, ingest_only = [], []
    for path in paths:
        suffix = Path(path).suffix.casefold()
        if suffix in JSONL_READER_EXTENSIONS:
            readable.append(str(path))
        elif suffix in INGEST_ONLY_EXTENSIONS:
            ingest_only.append(str(path))
    return readable, ingest_only


# Reports and accounting artifacts are JSON, but they are not corpus shards.
# A bare directory means "discover the corpus under here", so including one of
# these files turns a successful run's own metadata into input records.
CORPUS_SIDECAR_NAMES = frozenset(
    {
        "audit_report.json",
        "curation_ledger.json",
        "decontamination_report.json",
        "flow_plan.json",
        "flow_report.json",
        "ingest_report.json",
        "plan.json",
        "profile_report.json",
        "run_manifest.json",
        "sample_manifest.json",
        "subset_report.json",
        "token_counts.json",
    }
)


def explain_no_match(pattern: str | list[str]) -> str:
    """Why a pattern found nothing, in terms of where it was actually looked for.

    Relative paths in a config resolve against the process working directory,
    not against the directory the config sits in, so the same file can name two
    different places depending on where the command was run from. That is a real
    trap: a run that wrote its output next to the config on Monday writes it
    somewhere else on Tuesday, and the only symptom is a corpus that suddenly
    matches nothing.

    Until the resolution rule itself is settled, the least this can do is show
    both halves of the ambiguity rather than leaving the reader to guess which
    one applied.
    """
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    cwd = Path.cwd()
    lines = [f"working directory: {cwd}"]
    for item in patterns:
        candidate = Path(item)
        if candidate.is_absolute():
            lines.append(f"  {item} (absolute)")
        else:
            lines.append(f"  {item} -> {(cwd / item)}")
        # Walk up to the first parent that exists, so the message names the point
        # where the path stops being real rather than only its unreachable end.
        probe = (cwd / candidate).parent if not candidate.is_absolute() else candidate.parent
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        lines.append(f"    deepest existing directory: {probe}")
    lines.append(
        "Relative paths resolve against the working directory above, not against the "
        "directory holding the config. Run from the config's directory, or make the path "
        "absolute."
    )
    return "\n".join(lines)


def _is_corpus_file(path: Path, denied: frozenset[str]) -> bool:
    """Whether discovery under a directory should take this file as corpus."""
    return path.is_file() and path.suffix.lstrip(".").casefold() in CORPUS_EXTENSIONS and path.name not in denied


def expand_inputs(pattern: str | list[str] | None, *, allow_sidecars: bool = False) -> list[str]:
    """Resolve a corpus reference the way every curate step already does.

    ``allow_sidecars`` is for the one caller whose reference legitimately names
    an accounting artifact rather than a corpus: ``curate/audit``'s
    ``ledger_glob`` points *at* ``curation_ledger.json`` on purpose. It is an
    explicit opt-in rather than a looser default because the denylist is what
    stops a previous run's own metadata being read back as corpus rows.

    A glob, a bare directory, or a list of either. One implementation because
    the alternative was demonstrated: ``run_flow`` grew its own bare
    ``glob.glob``, so a directory — a spelling every other step accepts —
    resolved to no files at all, and the caller reported a fingerprint over an
    empty corpus as though it had read one.

    Curator has ``utils.file_utils.get_all_file_paths_under``, which does the
    traversal better — fsspec, so globs, directories and remote URLs all work.
    It is deliberately *not* used. This module is imported by ``run_flow``'s
    preflight and by ``run_ingest``, both of which must answer "can this run
    start?" before any cluster exists; importing Curator here would make
    resolving a path require the dependency that resolving a path is supposed to
    precede. Fifteen lines of stdlib is the cheaper side of that trade, and the
    only thing lost is remote-URL support, which no step accepts anyway.
    """
    if not pattern:
        return []
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    denied = frozenset() if allow_sidecars else CORPUS_SIDECAR_NAMES
    found: list[str] = []
    for item in patterns:
        item = str(item)
        if any(ch in item for ch in "*?["):
            # A glob says what it wants, so CORPUS_EXTENSIONS does not narrow the
            # FILES it matches. The sidecar denylist is a different rule and does
            # apply: the flow writes run_manifest.json and curation_ledger.json
            # *into* the corpus directory, and the documented reuse pattern points
            # a later run at that directory, so `filtered_jsonl/*.json*` would
            # otherwise read a previous run's own accounting back in as corpus
            # rows -- and, because the manifest is pretty-printed, charge it as
            # unparsable damage.
            # A glob match that is a DIRECTORY is not recursed into. Doing so was
            # tried and reverted: it changed what every pre-existing `<root>/*`
            # config resolves to across all six resolvers, and turned a working
            # `input: ./raw/*` over a parquet corpus with a converted/ subdirectory
            # into an IngestError ("cannot infer one format"). A partitioned corpus
            # is named by its bare directory -- `input: ./root` -- which has always
            # recursed and still does.
            found.extend(p for p in glob.glob(item, recursive=True) if Path(p).name not in denied)
            continue
        path = Path(item)
        if path.is_dir():
            # Every corpus extension, not just .jsonl: ingest reads parquet too,
            # and matching only .jsonl made `input: ./raw/` over a parquet corpus
            # resolve to nothing and report an empty corpus as though read.
            found.extend(str(p) for p in path.rglob("*") if _is_corpus_file(p, denied))
        else:
            found.append(item)
    return sorted({f for f in found if Path(f).is_file()})


def corpus_fingerprint(pattern: str | list[str] | None, text_field: str, id_field: str | None = None) -> str:
    """Order-independent fingerprint of a corpus's **contents**.

    Ties a filtering policy to the data its thresholds were measured on. The key
    covers the document text, not merely its identifier: hashing ids alone makes
    two corpora that share an id scheme — ``doc-0``, ``doc-1``, … — fingerprint
    identically, so an approval granted against one would verify cleanly against
    the other while the report claimed the corpus had been checked.

    The id is folded in as well when there is one, so re-identifying the same
    text is also a change.

    Reshard-stable by construction: :class:`RowDigest` is order-independent and
    nothing here depends on which file a row arrived in.

    Reads JSONL, and says so by refusing anything else. ``expand_inputs``
    deliberately resolves parquet as well, because ingest accepts it — but a
    parquet file read as lines yields no records, and returning the digest of no
    records would hand back one constant for every such corpus, equal to the
    digest of no input at all. That is the substitution this function exists to
    catch, so an unreadable corpus is refused rather than fingerprinted.
    """
    digest = RowDigest()
    resolved = expand_inputs(pattern)
    documents = 0
    for path in resolved:
        with Path(path).open("rb") as handle:
            for raw in handle:
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                # A line that parses is not necessarily a document: binary input
                # can yield a bare scalar, and calling .get on it raised.
                if not isinstance(record, dict):
                    continue
                text = record.get(text_field)
                if not isinstance(text, str):
                    continue
                documents += 1
                content = hashlib.sha256(text.encode("utf-8")).hexdigest()
                doc_id = record.get(id_field) if id_field else None
                digest.add(f"{doc_id}\0{content}" if doc_id is not None else content)
    if not resolved:
        # The case the guard below was written for, minus its files. A corpus
        # that resolves to nothing digests to one constant shared by every other
        # empty corpus — and the approve gate compares exactly this value to
        # decide whether a policy describes the data in front of it. Returning it
        # let an approval be verified against a corpus that did not exist yet.
        raise UnreadableCorpusError(
            f"{pattern} matched no files. A fingerprint over no input is the same value for "
            "every such corpus, so it cannot stand for this one — and the approve gate would "
            "compare it as though it did.\n" + explain_no_match(pattern)
        )
    if not documents:
        raise UnreadableCorpusError(
            f"{pattern} resolved to {len(resolved)} file(s) but no JSONL document with a "
            f"{text_field!r} string field. This reader is JSONL-only; a fingerprint over "
            "zero documents is the same value for every corpus, so it cannot stand for this "
            "one. Fingerprint the JSONL a reader produced from it instead."
        )
    return digest.hexdigest()


def summarize(reports: Sequence[ShardReport]) -> dict[str, Any]:
    """Roll per-shard facts up to corpus level."""
    per_source: Counter[str] = Counter()
    for report in reports:
        per_source.update(report.per_source)

    unreadable = [r for r in reports if not r.readable]
    return {
        "file_count": len(reports),
        "row_count": sum(r.row_count for r in reports),
        "bytes": sum(r.size_bytes for r in reports),
        "unreadable_count": len(unreadable),
        "bad_record_count": sum(r.bad_record_count for r in reports),
        "unreadable": [
            {
                "path": r.path,
                "error": r.error,
                "error_byte_offset": r.error_byte_offset,
                "bad_record_count": r.bad_record_count,
            }
            for r in unreadable
        ],
        **({"per_source": dict(sorted(per_source.items()))} if per_source else {}),
    }


#: Cap on per-record complaints carried into a report. A corpus missing the
#: comparison field on every row would otherwise write one line per document.
MAX_REPORTED_EXAMPLES = 5


@dataclass
class KeyedRows:
    """Records reduced to comparable keys, plus what could not be reduced."""

    keys: Counter[str] = field(default_factory=Counter)
    rows_seen: int = 0
    rows_keyed: int = 0
    rows_missing_field: int = 0
    rows_unparsable: int = 0
    duplicate_keys: int = 0
    examples: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Whether every record present was successfully keyed."""
        return self.rows_seen == self.rows_keyed

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows_seen": self.rows_seen,
            "rows_keyed": self.rows_keyed,
            "rows_missing_field": self.rows_missing_field,
            "rows_unparsable": self.rows_unparsable,
            "duplicate_keys": self.duplicate_keys,
            "examples": list(self.examples),
        }


def row_keys(
    paths: Iterable[str | Path],
    comparison_fields: Sequence[str],
    *,
    duplicate_ids: str = "reject",
) -> KeyedRows:
    """Reduce every record to one comparable key, counting what could not be.

    ``comparison_fields`` has no default here by design. The pipeline adds
    columns of its own — ``language`` and ``domain`` among them — so hashing
    whatever the two corpora happen to share would report differences that are
    simply the pipeline doing its job.

    Records that cannot be keyed are counted rather than dropped. A containment
    result computed over an unknown fraction of a corpus is not a containment
    result, so the caller needs to know how much was actually compared.
    """
    if not comparison_fields:
        raise ContainmentConfigError(
            "comparison_fields must name the fields to compare. There is no "
            "'all common fields' default: the pipeline adds columns such as "
            "language and domain, so an implicit choice reports false differences. "
            "Use [id] when the corpus carries a stable identifier."
        )

    out = KeyedRows()

    for path in sorted(str(p) for p in paths):
        p = Path(path)
        if not p.is_file():
            continue
        with p.open("rb") as handle:
            for raw in handle:
                stripped = raw.strip()
                if not stripped:
                    continue
                out.rows_seen += 1
                try:
                    record = json.loads(stripped)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    out.rows_unparsable += 1
                    if len(out.examples) < MAX_REPORTED_EXAMPLES:
                        out.examples.append(f"{path}: unparsable record")
                    continue
                missing = [f for f in comparison_fields if f not in record]
                if missing:
                    out.rows_missing_field += 1
                    if len(out.examples) < MAX_REPORTED_EXAMPLES:
                        out.examples.append(f"{path}: record missing {missing}")
                    continue
                payload = json.dumps(
                    [record[f] for f in comparison_fields],
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                out.keys[hashlib.sha256(payload.encode("utf-8")).hexdigest()] += 1
                out.rows_keyed += 1

    out.duplicate_keys = sum(1 for n in out.keys.values() if n > 1)
    return out


def containment(
    subset_paths: Iterable[str | Path],
    superset_paths: Iterable[str | Path],
    comparison_fields: Sequence[str],
    *,
    duplicate_ids: str = "reject",
) -> dict[str, Any]:
    """Whether every record of one corpus is present in another.

    Multiset containment, so a record appearing twice in the subset must appear
    at least twice in the superset.

    ``contained`` is only ever true when both sides were fully keyed. Comparing
    zero rows and calling the result containment would be a false all-clear from
    a check whose entire purpose is to catch a problem.
    """
    sub = row_keys(subset_paths, comparison_fields, duplicate_ids=duplicate_ids)
    sup = row_keys(superset_paths, comparison_fields, duplicate_ids=duplicate_ids)

    missing = {k: n - sup.keys.get(k, 0) for k, n in sub.keys.items() if n > sup.keys.get(k, 0)}
    verifiable = sub.complete and sup.complete and sub.rows_keyed > 0

    reasons: list[str] = []
    if sub.rows_keyed == 0:
        reasons.append("no target record could be keyed")
    if not sub.complete:
        reasons.append(f"{sub.rows_seen - sub.rows_keyed} of {sub.rows_seen} target rows could not be keyed")
    if not sup.complete:
        reasons.append(f"{sup.rows_seen - sup.rows_keyed} of {sup.rows_seen} reference rows could not be keyed")
    if duplicate_ids == "reject":
        for side, keyed in (("target", sub), ("reference", sup)):
            if keyed.duplicate_keys:
                reasons.append(f"{keyed.duplicate_keys} {side} key(s) repeat while duplicate_ids='reject'")

    return {
        "comparison_fields": list(comparison_fields),
        "duplicate_ids": duplicate_ids,
        "subset_rows": sub.rows_keyed,
        "superset_rows": sup.rows_keyed,
        "verifiable": verifiable,
        "contained": verifiable and not missing,
        "missing_key_count": len(missing),
        "missing_row_count": sum(missing.values()),
        "target": sub.as_dict(),
        "reference": sup.as_dict(),
        "problems": reasons + sub.examples + sup.examples,
    }
