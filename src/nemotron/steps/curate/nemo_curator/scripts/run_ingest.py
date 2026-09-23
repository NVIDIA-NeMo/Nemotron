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

"""CLI for ``steps/curate/ingest`` — raw corpus in, curatable corpus out.

Without this step the promise "provide a config and your data" was not true: a
parquet corpus had to be converted by hand, and a corpus with no document id had
to have one minted by hand, before the flow would read it. Both are mechanical,
both are easy to get subtly wrong, and neither is the user's job.

What it does **not** do is rewrite text. Normalisation that changes content
belongs to a filtering decision somebody approves, not to reading a file: this
step selects fields, mints an identifier, and writes JSONL.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from nemotron.steps.curate.nemo_curator.runtime import ingest as ingest_module
from nemotron.steps.curate.nemo_curator.runtime import integrity
from nemotron.steps.curate.nemo_curator.runtime import manifest as run_manifest

logger = logging.getLogger("curate.ingest")

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "ingest" / "config" / "default.yaml"

#: Documents per output shard. Small enough that a failed write loses little,
#: large enough that a big corpus does not become a directory of tiny files.
SHARD_SIZE = 50_000


def _reset_output_artifacts(output_dir: Path) -> None:
    """Remove only artifacts owned by this step, including interrupted writes."""
    for pattern in ("part_*.jsonl", ".part_*.jsonl.tmp"):
        for path in output_dir.glob(pattern):
            path.unlink(missing_ok=True)
    for name in ("ingest_report.json", ".ingest_report.json.tmp"):
        (output_dir / name).unlink(missing_ok=True)


def run(cfg: dict, started_at: str | None = None) -> dict[str, Any]:
    """Read the raw corpus, mint ids where needed, write JSONL.

    The uniform seam every curate step exposes. Does not raise ``SystemExit``.
    """
    started_at = started_at or run_manifest.utc_now()

    output_dir = Path(cfg.get("output_dir") or "./output/ingested")

    paths = integrity.expand_inputs(cfg.get("input"))
    # A step must never read its own output. When output_dir sits inside the
    # input directory — the obvious layout, and what the README's own example
    # produces — the previous run's shards and its ingest_report.json are both
    # under the input glob. Reading them back counted the report's lines as
    # corrupt documents, and the shards are deleted below while `paths` still
    # names them, which then raises FileNotFoundError from inside the reader.
    resolved_out = output_dir.resolve()
    own_output = [p for p in paths if resolved_out in Path(p).resolve().parents]
    if own_output:
        paths = [p for p in paths if p not in set(own_output)]
        skipped_own_output = len(own_output)
    else:
        skipped_own_output = 0

    if not paths:
        raise ingest_module.IngestError(f"input matched no files: {cfg.get('input')!r}")

    fmt = cfg.get("format") or "auto"
    if fmt == "auto":
        fmt = ingest_module.detect_format(paths)
    elif fmt not in ingest_module.FORMATS:
        raise ingest_module.IngestError(f"format must be one of {ingest_module.FORMATS} or 'auto', got {fmt!r}")

    on_duplicate = cfg.get("on_duplicate", "refuse")
    if on_duplicate not in ingest_module.ON_DUPLICATE:
        raise ingest_module.IngestError(
            f"on_duplicate must be one of {ingest_module.ON_DUPLICATE}, got {on_duplicate!r}"
        )

    text_field = cfg.get("text_field") or "text"
    id_field = cfg.get("id_from")
    raw_id_fields = cfg.get("id_fields")
    if raw_id_fields is not None and (
        not isinstance(raw_id_fields, list) or not all(isinstance(field, str) and field for field in raw_id_fields)
    ):
        raise ingest_module.IngestError("id_fields must be a list of non-empty column names")
    id_fields = list(raw_id_fields or [field for field in (cfg.get("url_from"), text_field) if field])
    if not id_field and text_field not in id_fields:
        raise ingest_module.IngestError(
            f"id_fields must include text_field {text_field!r}. Otherwise the advertised "
            "content-derived identity could stay unchanged while document content changes."
        )
    id_prefix = cfg.get("id_prefix") or ""
    raw_keep = cfg.get("keep_fields") or []
    if not isinstance(raw_keep, list) or not all(isinstance(field, str) and field for field in raw_keep):
        raise ingest_module.IngestError("keep_fields must be a list of non-empty column names")
    keep = list(raw_keep)

    output_dir.mkdir(parents=True, exist_ok=True)
    _reset_output_artifacts(output_dir)

    reader = ingest_module.iter_jsonl if fmt == "jsonl" else ingest_module.iter_parquet
    seen: dict[str, int] = {}
    stats = {
        "records_read": 0,
        "unparsable_lines": 0,
        "skipped_non_mapping": 0,
        "skipped_missing_text": 0,
        "skipped_missing_id": 0,
        "duplicate_ids": 0,
        "written": 0,
    }
    duplicate_examples: list[str] = []
    duplicate_groups: set[str] = set()
    columns_seen: set[str] = set()

    shard_index = -1
    in_shard = 0
    handle = None
    temporary_shards: list[Path] = []
    try:
        try:
            for path in paths:
                for raw in reader(path):
                    if raw is None:
                        stats["records_read"] += 1
                        stats["skipped_non_mapping"] += 1
                        continue
                    if raw.get("__unparsable__"):
                        stats["unparsable_lines"] += 1
                        continue
                    stats["records_read"] += 1
                    columns_seen.update(raw)

                    record = ingest_module.normalise(
                        raw,
                        text_field=text_field,
                        id_field=id_field,
                        source_field=cfg.get("source_from"),
                        source_value=cfg.get("source"),
                        keep=keep,
                        id_fields=id_fields,
                        id_prefix=id_prefix,
                    )
                    if record is None:
                        if isinstance(raw.get(text_field), str) and raw[text_field]:
                            stats["skipped_missing_id"] += 1
                        else:
                            stats["skipped_missing_text"] += 1
                        continue

                    doc_id = record["id"]
                    if doc_id in seen:
                        stats["duplicate_ids"] += 1
                        duplicate_groups.add(doc_id)
                        # Distinct ids, not the first N occurrences: the largest
                        # duplicate group in one real corpus held 293 copies, so
                        # "the first three" would print one id three times and say
                        # nothing about how widespread the problem is.
                        if (
                            doc_id not in duplicate_examples
                            and len(duplicate_examples) < integrity.MAX_REPORTED_EXAMPLES
                        ):
                            duplicate_examples.append(doc_id)
                        if on_duplicate == "drop" or on_duplicate == "refuse":
                            continue
                        if on_duplicate == "suffix":
                            suffix = seen[doc_id] + 1
                            candidate = f"{doc_id}-{suffix}"
                            # A corpus-provided id can already occupy the suffix
                            # namespace. Allocate until the emitted id is unique,
                            # and reserve it so a later raw id cannot collide.
                            while candidate in seen:
                                suffix += 1
                                candidate = f"{doc_id}-{suffix}"
                            seen[doc_id] = suffix
                            record["id"] = candidate
                            seen[candidate] = 0
                    else:
                        seen[doc_id] = 0

                    if handle is None or in_shard >= SHARD_SIZE:
                        if handle is not None:
                            handle.close()
                        shard_index += 1
                        in_shard = 0
                        shard_path = output_dir / f".part_{shard_index}.jsonl.tmp"
                        temporary_shards.append(shard_path)
                        handle = shard_path.open("w", encoding="utf-8")

                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats["written"] += 1
                    in_shard += 1
        finally:
            if handle is not None:
                handle.close()

        if stats["duplicate_ids"] and on_duplicate == "refuse":
            if id_field:
                reason = (
                    f"{stats['duplicate_ids']} duplicate {id_field!r} occurrence(s), in "
                    f"{len(duplicate_groups)} group(s) (e.g. {', '.join(duplicate_examples)}). "
                    "The corpus-provided id does not uniquely identify a document."
                )
            else:
                reason = (
                    f"{stats['duplicate_ids']} document(s) repeat an earlier document's "
                    f"configured identity fields, in {len(duplicate_groups)} group(s) "
                    f"(e.g. {', '.join(duplicate_examples)}). A derived id cannot tell them apart."
                )
            raise ingest_module.IngestError(
                f"{reason} Subset and decontamination both refuse a non-unique id. Choose: "
                "on_duplicate: drop to keep the first of each, or suffix to keep every copy "
                "under a distinguishable id. Both change what the corpus is, which is why "
                "neither happens by default."
            )

        if not stats["written"]:
            raise ingest_module.IngestError(
                f"no usable documents: {stats['records_read']} read, "
                f"{stats['skipped_missing_text']} without {text_field!r}, "
                f"{stats['skipped_missing_id']} without {id_field!r}, "
                f"{stats['skipped_non_mapping']} non-object JSON value(s), "
                f"{stats['unparsable_lines']} unparsable"
            )

        report = {
            "schema_version": ingest_module.SCHEMA_VERSION,
            "step_id": "curate/ingest",
            "started_at": started_at,
            "completed_at": run_manifest.utc_now(),
            "input": {
                "glob": cfg.get("input"),
                "format": fmt,
                "files": len(paths),
                # Named rather than silently dropped: "3 files" when the user expected
                # 4 should be traceable to the reason without re-deriving it.
                "skipped_own_output": skipped_own_output,
            },
            "output_dir": str(output_dir),
            "identity": {
                "source": "corpus" if id_field else "minted",
                "field": id_field,
                "recipe": None if id_field else ingest_module.ID_RECIPE,
                "fields": None if id_field else id_fields,
                "prefix": None if id_field else (id_prefix or None),
                "note": (
                    "Ids come from the corpus."
                    if id_field
                    else "Ids are derived from configured identity fields, so they survive "
                    "resharding — unlike a positional id, which renames every document when "
                    "the corpus is re-split."
                ),
            },
            "on_duplicate": on_duplicate,
            "columns_available": sorted(columns_seen),
            "counts": stats,
            "warnings": [],
        }
        report["counts"]["duplicate_groups"] = len(duplicate_groups)
        if stats["duplicate_ids"]:
            kind = "duplicate corpus id" if id_field else "repeated derived identity"
            report["warnings"].append(
                f"{stats['duplicate_ids']} {kind} occurrence(s) in {len(duplicate_groups)} "
                f"group(s), handled with on_duplicate: {on_duplicate}"
            )
        if stats["unparsable_lines"]:
            report["warnings"].append(f"{stats['unparsable_lines']} unparsable line(s) skipped")
        if stats["skipped_non_mapping"]:
            report["warnings"].append(f"{stats['skipped_non_mapping']} valid non-object JSON value(s) skipped")
        if stats["skipped_missing_text"]:
            report["warnings"].append(
                f"{stats['skipped_missing_text']} record(s) had no {text_field!r} and were dropped"
            )
        if stats["skipped_missing_id"]:
            report["warnings"].append(f"{stats['skipped_missing_id']} record(s) had no {id_field!r} and were dropped")

        report_tmp = output_dir / ".ingest_report.json.tmp"
        report_tmp.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        for shard in temporary_shards:
            shard.replace(output_dir / shard.name.removeprefix(".").removesuffix(".tmp"))
        # The report is the completion marker and is committed last.
        report_tmp.replace(output_dir / "ingest_report.json")
    except BaseException:
        _reset_output_artifacts(output_dir)
        raise

    for warning in report["warnings"]:
        logger.warning(warning)
    print(f"curate/ingest: {stats['written']:,} document(s) from {len(paths)} {fmt} file(s) -> {output_dir}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a raw corpus for curation")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    try:
        run(cfg)
    except ingest_module.IngestError as exc:
        print(f"curate/ingest: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
