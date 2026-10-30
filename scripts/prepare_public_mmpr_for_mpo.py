#!/usr/bin/env python3
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

"""Prepare public Hugging Face MMPR (full) data for MPO training.

This script converts a raw `hf download` snapshot of `OpenGVLab/MMPR`
into the cache layout expected by the MPO trainer-side loader. It is
the MMPR-public counterpart to ``prepare_mmpr_tiny_for_vision_rl.py``;
the two share extraction shape and validation conventions but operate
on different upstream layouts (MMPR-Tiny ships a single parquet of
preference rows; MMPR-Public ships a directory of per-subset JSONL
annotations with absolute petrelfs paths in ``meta.json``).

Input directory requirements (typical layout from
``huggingface_hub.snapshot_download(repo_id="OpenGVLab/MMPR")``):
- ``images.zip``         (~14 GB, expands to ``MMPR/images/<subset>/...``)
- ``annotations.zip``    (~1.5 GB, expands to per-subset .jsonl files)
- ``meta.json``          (manifest of 90 subsets keyed by name)

Output directory layout:
- ``<output_dir>/MMPR/images/<subset>/...``
- ``<output_dir>/MMPR/annotations/<subset>.jsonl``  (one per entry in meta)
- ``<output_dir>/<meta_name>``                       (rewritten meta)
- ``<output_dir>/mmpr_public_preview.jsonl``         (sanity preview)
- ``<output_dir>/prepare_public_mmpr_for_mpo_summary.json``
- ``<output_dir>/.mmpr_ready``

The ``meta.json`` rewrite swaps absolute ``/mnt/petrelfs/...`` paths in
each entry's ``annotation`` field for paths *relative to ``output_dir``*
(``MMPR/annotations/<basename>``). The ``root`` field is preserved as-is
because upstream already uses the ``MMPR/images/...`` relative form.
This matches the convention in ``prepare_mmpr_tiny_for_vision_rl.py``
where image references resolve under the prep cache root.

If the trainer-side loader prefers absolute paths, swap
``_relpath_for_meta`` for an absolute-path variant — the rest of the
script doesn't depend on the choice.

Note: a future cleanup will vendor this logic into
``nemotron.data_prep.stages.vlm_preference_prep`` alongside
``_run_tiny`` so the Xenna stage doesn't need a subprocess shell-out.
Until then, mpo.yaml's ``builder_command`` invokes this script.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from tqdm.auto import tqdm


# Required keys per ``meta.json`` entry. The dataset-card schema has been
# stable across MMPR-v1.0/1.1/1.2 — if an upstream bump introduces new
# required fields, surface them here so we fail fast on prep rather than
# at training time.
REQUIRED_META_ENTRY_KEYS = ("root", "annotation", "length")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw MMPR-Public download into a local MPO-ready cache.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing raw MMPR files from `hf download`.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Target directory to write the processed MMPR cache into.",
    )
    parser.add_argument(
        "--meta-name",
        type=str,
        default="meta_public.json",
        help="Filename for the rewritten meta JSON inside --output-dir.",
    )
    parser.add_argument(
        "--preview-name",
        type=str,
        default="mmpr_public_preview.jsonl",
        help="Name of the generated inspection JSONL inside --output-dir.",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="prepare_public_mmpr_for_mpo_summary.json",
        help="Name of the generated summary JSON file inside --output-dir.",
    )
    parser.add_argument(
        "--preview-rows-per-subset",
        type=int,
        default=2,
        help=(
            "How many rows from each subset to write into the preview JSONL. "
            "Set to 0 to skip preview generation entirely."
        ),
    )
    return parser.parse_args()


def ensure_required_inputs(input_dir: Path) -> dict[str, Path]:
    paths = {
        "images_zip": input_dir / "images.zip",
        "annotations_zip": input_dir / "annotations.zip",
        "meta": input_dir / "meta.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required MMPR-Public inputs: {missing}")
    return paths


# =============================================================================
# Zip extraction
# =============================================================================
#
# MMPR-Public's images.zip is large enough (~14 GB / millions of files)
# that we keep one progress bar at the file granularity rather than
# loading everything into memory. The annotations zip is small enough
# that the same code shape works either way.


def _find_extracted_subdir(temp_dir: Path, *, target_name: str) -> Path:
    """Locate ``<target_name>/`` under a freshly-extracted zip tree.

    Mirrors the tiny script's approach: zips can vary in nesting depth,
    so we look at known direct candidates first and fall back to a
    shallowest-match glob.
    """
    direct_candidates = [
        temp_dir / target_name,
        temp_dir / "MMPR" / target_name,
    ]
    for candidate in direct_candidates:
        if candidate.is_dir():
            return candidate

    candidates = sorted(
        (path for path in temp_dir.rglob(target_name) if path.is_dir()),
        key=lambda path: (len(path.parts), str(path)),
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not locate extracted `{target_name}/` directory in zip."
        )
    return candidates[0]


def _extract_zip_to_named_subdir(
    *,
    zip_path: Path,
    output_dir: Path,
    relative_target: PurePosixPath,
    subdir_name_in_zip: str,
    desc: str,
) -> Path:
    """Extract ``zip_path`` and place the contents at ``output_dir / relative_target``.

    Idempotent: if the target already exists, returns it unchanged.

    Args:
        zip_path: Source zip file (e.g. images.zip).
        output_dir: Cache root.
        relative_target: Where to land the extracted dir, relative to
            ``output_dir`` (e.g. ``MMPR/images``).
        subdir_name_in_zip: The directory name to find inside the zip
            after extraction (e.g. ``images`` for images.zip).
        desc: tqdm description.
    """
    final_dir = output_dir / relative_target
    if final_dir.exists():
        print(f"Already extracted at {final_dir}")
        return final_dir

    temp_dir = output_dir / f"_mmpr_extract_tmp_{subdir_name_in_zip}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {zip_path} to {final_dir}...")
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        with tqdm(members, total=len(members), desc=desc, unit="file") as progress:
            for member in progress:
                zf.extract(member, temp_dir)

    extracted = _find_extracted_subdir(temp_dir, target_name=subdir_name_in_zip)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(extracted), str(final_dir))
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"Finished extracting {subdir_name_in_zip}.")
    return final_dir


# =============================================================================
# Meta rewriting
# =============================================================================
#
# Upstream meta.json entries reference annotation files via absolute
# petrelfs paths that don't exist on the prep host:
#
#     "annotation": "/mnt/petrelfs/.../MMPR/annotations/<file>.jsonl"
#
# We rewrite to *relative* paths under the cache root so the meta is
# portable and the consuming loader doesn't need to know the absolute
# layout:
#
#     "annotation": "MMPR/annotations/<file>.jsonl"
#
# The ``root`` field is already in relative-from-cache-root form upstream
# (``MMPR/images/<subset>``) so we leave it alone.


def _relpath_for_meta(annotation_path: str) -> str:
    """Return the post-prep annotation path for a meta.json entry."""
    basename = PurePosixPath(annotation_path.replace("\\", "/")).name
    return PurePosixPath("MMPR/annotations") / basename


def _validate_entry(name: str, entry: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_META_ENTRY_KEYS if k not in entry]
    if missing:
        raise ValueError(
            f"meta.json entry {name!r} is missing required keys: {missing}. "
            "If upstream changed the schema, update REQUIRED_META_ENTRY_KEYS."
        )


def rewrite_meta(
    input_meta_path: Path,
    annotations_dir: Path,
    output_meta_path: Path,
) -> dict[str, Any]:
    """Read ``meta.json``, rewrite annotation paths, and verify each file exists.

    Returns the rewritten meta dict so callers can compute summary stats.
    """
    raw = json.loads(input_meta_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected meta.json to be a dict of subset -> spec, got {type(raw).__name__}"
        )

    rewritten: dict[str, Any] = {}
    missing_annotations: list[str] = []
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"meta.json entry {name!r} is not a dict (got {type(entry).__name__})"
            )
        _validate_entry(name, entry)

        new_entry = dict(entry)
        new_relpath = _relpath_for_meta(entry["annotation"])
        new_entry["annotation"] = str(new_relpath)
        rewritten[name] = new_entry

        # Verify the annotation file actually landed where we expect.
        # The annotation directory is ``output_dir / MMPR / annotations``
        # which `_relpath_for_meta` returns rooted at the cache.
        candidate = annotations_dir / new_relpath.name
        if not candidate.exists():
            missing_annotations.append(f"{name}: {candidate}")

    if missing_annotations:
        # Surface these as a hard error so we don't write a meta the
        # trainer can't consume.
        head = "\n  ".join(missing_annotations[:10])
        more = (
            f"\n  ... and {len(missing_annotations) - 10} more"
            if len(missing_annotations) > 10
            else ""
        )
        raise FileNotFoundError(
            "Annotations missing for meta.json entries (zip extraction may "
            "have produced a different layout than expected):\n  "
            f"{head}{more}"
        )

    output_meta_path.write_text(
        json.dumps(rewritten, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return rewritten


# =============================================================================
# Preview JSONL (best-effort sanity)
# =============================================================================
#
# Unlike MMPR-Tiny (single parquet, well-known schema), MMPR-Public's
# subsets each ship JSONL with potentially-different shapes. We don't
# attempt to extract a normalized question/answer; we just emit the
# first-N raw rows per subset so operators can spot-check the layout
# made it through extraction. If the trainer-side loader exposes a
# row-validation hook we can plug in later, replace ``_emit_raw_rows``
# with that.


def _emit_raw_rows(
    annotations_dir: Path,
    rewritten_meta: dict[str, Any],
    preview_path: Path,
    rows_per_subset: int,
) -> dict[str, int]:
    stats: Counter[str] = Counter()
    if rows_per_subset <= 0:
        preview_path.write_text("", encoding="utf-8")
        return dict(stats)

    with preview_path.open("w", encoding="utf-8") as fout:
        for subset_name, entry in tqdm(
            rewritten_meta.items(),
            total=len(rewritten_meta),
            desc="Sampling MMPR-Public subsets",
            unit="subset",
        ):
            jsonl_path = annotations_dir / PurePosixPath(entry["annotation"]).name
            try:
                with jsonl_path.open("r", encoding="utf-8") as fin:
                    for i, line in enumerate(fin):
                        if i >= rows_per_subset:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            stats["unparseable_rows"] += 1
                            continue
                        preview_record = {
                            "subset": subset_name,
                            "row_index": i,
                            "row": row,
                        }
                        fout.write(json.dumps(preview_record, ensure_ascii=False) + "\n")
                        stats["preview_rows"] += 1
            except FileNotFoundError:
                stats["missing_subset_files"] += 1
            except OSError:
                stats["unreadable_subset_files"] += 1

    return dict(stats)


def write_ready_marker(output_dir: Path) -> Path:
    ready_marker = output_dir / ".mmpr_ready"
    ready_marker.write_text("ready\n", encoding="utf-8")
    return ready_marker


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    paths = ensure_required_inputs(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = _extract_zip_to_named_subdir(
        zip_path=paths["images_zip"],
        output_dir=output_dir,
        relative_target=PurePosixPath("MMPR/images"),
        subdir_name_in_zip="images",
        desc="Extracting MMPR-Public images",
    )
    annotations_dir = _extract_zip_to_named_subdir(
        zip_path=paths["annotations_zip"],
        output_dir=output_dir,
        relative_target=PurePosixPath("MMPR/annotations"),
        subdir_name_in_zip="annotations",
        desc="Extracting MMPR-Public annotations",
    )

    output_meta_path = output_dir / args.meta_name
    rewritten = rewrite_meta(paths["meta"], annotations_dir, output_meta_path)

    preview_path = output_dir / args.preview_name
    stats = _emit_raw_rows(
        annotations_dir,
        rewritten,
        preview_path,
        rows_per_subset=args.preview_rows_per_subset,
    )

    ready_marker = write_ready_marker(output_dir)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "meta_path": str(output_meta_path),
        "images_dir": str(images_dir),
        "annotations_dir": str(annotations_dir),
        "preview_path": str(preview_path),
        "ready_marker": str(ready_marker),
        "subset_count": len(rewritten),
        "total_examples_advertised": sum(
            int(e.get("length", 0)) for e in rewritten.values()
        ),
        "preview_stats": stats,
    }
    summary_path = output_dir / args.summary_name
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote rewritten meta to {output_meta_path}")
    print(f"Wrote preview JSONL to {preview_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote ready marker to {ready_marker}")
    print(
        f"Prepared {summary['subset_count']} subsets, "
        f"{summary['total_examples_advertised']:,} advertised examples."
    )

    flagged = {k: v for k, v in stats.items() if v}
    if flagged:
        print("Flagged during preview generation:")
        for reason, count in sorted(flagged.items()):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
