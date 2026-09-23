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

"""CLI for ``steps/curate/audit``.

Turns the measurements in :mod:`nemotron.steps.curate.nemo_curator.runtime.integrity` into a
report, and decides which of them are findings. The decision rule is the whole
point of the step: a row count that differs from the input is an observation,
because filtering is supposed to remove rows; the same difference contradicting
a manifest the producer wrote is a finding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from nemotron.steps.curate.nemo_curator.runtime import integrity, ledger
from nemotron.steps.curate.nemo_curator.runtime import manifest as run_manifest

MODES = ("integrity", "digest", "containment", "all")

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "audit" / "config" / "default.yaml"


class ConfigError(ValueError):
    """The audit cannot run as configured — a problem the user can fix.

    Named to match every sibling runner: main() turns this into exit 2 with a
    message, while anything else stays a traceback because it is a bug. Pointing
    a step at a corpus that does not exist used to raise FileNotFoundError here,
    which no main() catches, so one user mistake produced a clean refusal in
    three steps and a raw traceback in this one.
    """


def expand(pattern: str | list[str] | None, *, allow_sidecars: bool = False) -> list[str]:
    """Expand a glob, a directory, or a literal path.

    Delegates rather than repeating the logic: this was a second copy that
    matched only ``*.jsonl`` inside a directory, so the same reference resolved
    to different corpora depending on which step read it.
    """
    return integrity.expand_inputs(pattern, allow_sidecars=allow_sidecars)


def _nested(document: Any, *keys: str) -> Any:
    """Walk a mapping defensively; a malformed manifest must not crash the audit."""
    current = document
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def audit(cfg: dict[str, Any]) -> dict[str, Any]:
    """Measure the target corpus and decide which measurements are findings."""
    mode = cfg.get("mode", "integrity")
    if mode not in MODES:
        raise ConfigError(f"mode must be one of {MODES}, got {mode!r}")

    target_paths = expand(cfg.get("target_glob"))
    if not target_paths:
        raise ConfigError(f"target_glob matched no files: {cfg.get('target_glob')!r}")

    declared_path = cfg.get("declared_manifest")
    declared = None
    unreadable: str | None = None
    if declared_path:
        try:
            declared = run_manifest.read_manifest(declared_path)
        except run_manifest.ManifestUnreadableError as exc:
            # The artifact the audit was pointed at is missing or truncated. That
            # is a finding about the run, not a failure of the auditor, and
            # reporting it is the whole job.
            unreadable = str(exc)
    if declared_path and not isinstance(declared, dict):
        # A manifest that is valid JSON but not an object would otherwise fail
        # with an AttributeError deep in the comparison, which reads as a crash
        # rather than as the finding it is.
        declared = {"__malformed__": unreadable or "declared_manifest is not a JSON object"}
    source_field = _nested(declared, "output", "source_field") or cfg.get("source_field")

    want_digest = mode in ("digest", "all")
    reports = integrity.scan_corpus(target_paths, source_field=source_field, want_digest=want_digest)

    report: dict[str, Any] = {
        "step_id": "curate/audit",
        "mode": mode,
        "target": {"glob": cfg.get("target_glob"), **integrity.summarize(reports)},
        "findings": [],
        "observations": [],
    }
    findings: list[dict[str, Any]] = report["findings"]

    # -- readability ---------------------------------------------------------
    for shard in reports:
        if not shard.readable:
            findings.append(
                {
                    "name": "unreadable_shard",
                    "path": shard.path,
                    "error": shard.error,
                    "byte_offset": shard.error_byte_offset,
                    "message": (
                        f"{shard.path}: {shard.error}"
                        + (f" at byte {shard.error_byte_offset}" if shard.error_byte_offset is not None else "")
                    ),
                }
            )

    # An empty shard parses perfectly, so it is not a readability fault — a
    # strict filter can legitimately empty one. It is still the signature of a
    # writer that was killed before it wrote anything, so it is surfaced rather
    # than left for someone to notice in a directory listing.
    empty = [s for s in reports if s.readable and s.row_count == 0]
    if empty:
        report["observations"].append(
            {
                "name": "zero_row_shard",
                "count": len(empty),
                "paths": [s.path for s in empty[:10]],
                "note": (
                    "these shards parse but hold no records. A strict filter can produce that "
                    "legitimately; a writer killed before its first record produces it too. "
                    "Compare against the producer's manifest to tell them apart."
                ),
            }
        )

    # -- digest --------------------------------------------------------------
    if want_digest:
        report["target"]["digest"] = integrity.corpus_digest(reports, root=cfg.get("digest_root"))
        report["target"]["shards"] = [s.as_dict() for s in reports]

    # -- manifest comparison -------------------------------------------------
    if declared is None:
        report["completeness"] = {
            "claimed": False,
            "reason": (
                "no declared_manifest: row counts are informational. A corpus can "
                "parse cleanly and still be missing rows, so completeness is only "
                "claimable against a manifest the producing step wrote."
            ),
        }
    else:
        problems = run_manifest.validate_manifest(declared)
        if problems:
            findings.append(
                {"name": "manifest_mismatch", "message": "declared manifest is not conformant", "problems": problems}
            )

        if not run_manifest.is_complete(declared):
            findings.append(
                {
                    "name": "manifest_incomplete",
                    "message": (
                        "declared manifest has no completed_at: the producing run did "
                        "not reach its write barrier, so its counts describe a partial run."
                    ),
                }
            )

        expected = _nested(declared, "output") or {}
        actual_rows = report["target"]["row_count"]
        actual_files = report["target"]["file_count"]
        report["completeness"] = {
            "claimed": True,
            "declared_rows": expected.get("row_count"),
            "actual_rows": actual_rows,
            "declared_files": expected.get("file_count"),
            "actual_files": actual_files,
        }
        if expected.get("row_count") != actual_rows or expected.get("file_count") != actual_files:
            findings.append(
                {
                    "name": "manifest_mismatch",
                    "message": (
                        f"producer declared {expected.get('file_count')} files / "
                        f"{expected.get('row_count')} rows; found {actual_files} / {actual_rows}"
                    ),
                }
            )

    # -- reference delta (informational) -------------------------------------
    reference_paths = expand(cfg.get("reference_glob"))
    reference_reports: list[integrity.ShardReport] = []
    if reference_paths:
        reference_reports = integrity.scan_corpus(reference_paths, source_field=source_field, want_digest=False)
        ref = integrity.summarize(reference_reports)
        report["observations"].append(
            {
                "name": "reference_delta",
                "reference_rows": ref["row_count"],
                "target_rows": report["target"]["row_count"],
                "delta": ref["row_count"] - report["target"]["row_count"],
                "note": "filters remove rows on purpose; a delta is only an error against a producer declaration",
            }
        )
        # A damaged reference makes the target look like it gained rows. Say so
        # before any containment result is read as a fault in the target.
        if ref["unreadable_count"]:
            findings.append(
                {
                    "name": "unreadable_reference_shard",
                    "message": (
                        f"{ref['unreadable_count']} reference shard(s) are damaged; comparisons "
                        "against this corpus describe the reference, not the target"
                    ),
                    "shards": ref["unreadable"],
                }
            )

    # -- containment ---------------------------------------------------------
    if mode in ("containment", "all"):
        if not reference_paths:
            raise ConfigError("containment mode requires reference_glob")
        duplicate_ids = _nested(declared, "canonicalization", "duplicate_ids") or "reject"
        result = integrity.containment(
            target_paths, reference_paths, cfg.get("comparison_fields") or [], duplicate_ids=duplicate_ids
        )
        report["containment"] = result

        if not result["verifiable"]:
            # The failure this branch exists for: comparing nothing and calling
            # it containment is a false all-clear from a check whose whole
            # purpose is to catch a problem.
            findings.append(
                {
                    "name": "containment_unverifiable",
                    "message": (
                        "containment could not be established: "
                        + "; ".join(result["problems"][:3])
                        + f" (keyed {result['subset_rows']} of {result['target']['rows_seen']} target rows)"
                    ),
                }
            )
        elif not result["contained"]:
            findings.append(
                {
                    "name": "containment_violation",
                    "message": (
                        f"{result['missing_row_count']} target row(s) are not present in the reference corpus"
                    ),
                }
            )

        if duplicate_ids == "reject":
            repeated = result["target"]["duplicate_keys"] + result["reference"]["duplicate_keys"]
            if repeated:
                findings.append(
                    {
                        "name": "duplicate_ids",
                        "message": (
                            f"{repeated} key(s) repeat while the manifest declares "
                            "duplicate_ids='reject'; the identifier does not identify a document"
                        ),
                    }
                )

    # -- attribution (v2) ----------------------------------------------------
    #
    # Everything above detects. This is the only part that attributes, and it
    # can only do so because the producer wrote down what it did at the time.
    # Reading the output afterwards cannot distinguish a record removed by a
    # language filter from one lost with a shard.
    # The only reference in the category that names an accounting artifact on
    # purpose: ledger_glob points AT curation_ledger.json, which every other
    # caller must exclude. Without the opt-in a glob spelling resolves to
    # nothing and the audit reports "no ledger_glob" -- it would still pass,
    # having quietly lost the attribution it was asked for.
    ledger_paths = expand(cfg.get("ledger_glob"), allow_sidecars=True)
    if not ledger_paths:
        report["attribution"] = {
            "available": False,
            "reason": (
                "no ledger_glob: this audit detects loss but cannot attribute it. "
                "A record missing from the output is indistinguishable from one "
                "removed on purpose unless the producing stage recorded which."
            ),
        }
    else:
        ledgers = []
        for path in ledger_paths:
            try:
                ledgers.append(ledger.load_ledger(path))
            except ledger.LedgerInvalidError as exc:
                findings.append({"name": "ledger_unreadable", "message": str(exc)})

        if ledgers:
            declared_input = _nested(declared, "input", "row_count") if declared else None
            if declared_input is not None:
                observed = declared_input - report["target"]["row_count"]
            elif reference_paths:
                observed = ref["row_count"] - report["target"]["row_count"]
            else:
                observed = None

            attribution = ledger.attribute(ledgers, observed or 0)
            attribution["available"] = True
            attribution["ledgers"] = len(ledgers)
            attribution["stages"] = sorted({led.stage for led in ledgers})
            if observed is None:
                attribution["observed_delta"] = None
                attribution["unexplained"] = None
                attribution["note"] = (
                    "no input row count to compare against: set declared_manifest or "
                    "reference_glob so the ledger's declarations can be checked against "
                    "an observed delta rather than merely reported."
                )
            # The producer writes the SAME breakdown into two artifacts, from two
            # independent reconciliations against its own disk counts. Having both
            # in hand and not comparing them is how they were allowed to disagree:
            # the manifest declared attribution unavailable while the ledger beside
            # it carried a full per-gate breakdown, and this audit passed anyway.
            declared_block = (declared or {}).get("declared") or {}
            declared_gates = declared_block.get("filtered")
            observed_gates = attribution.get("filtered_by_reason")
            if declared_gates is not None and observed_gates is not None and declared_gates != observed_gates:
                findings.append(
                    {
                        "name": "attribution_disagreement",
                        "message": (
                            "the run manifest and the curation ledger attribute the same "
                            "removals differently. They are written by one step from one "
                            "run, so a difference means one of the two reconciliations is "
                            "wrong and neither figure can be quoted."
                        ),
                        "manifest": declared_gates,
                        "ledger": observed_gates,
                    }
                )
            elif observed_gates and declared_gates is None and declared is not None:
                attribution["manifest_declares_attribution"] = False

            report["attribution"] = attribution

            unbalanced = [led for led in ledgers if not led.balanced]
            if unbalanced:
                findings.append(
                    {
                        "name": "ledger_imbalanced",
                        "message": (
                            f"{len(unbalanced)} ledger(s) do not reconcile: a stage reported "
                            "success while records were unaccounted for"
                        ),
                        "stages": sorted({led.stage for led in unbalanced}),
                        "detail": [led.summary() for led in unbalanced[:3]],
                    }
                )

            lost = ledger.lost_unit_report(ledgers, "curate/audit", where=str(cfg.get("ledger_glob")))
            if lost:
                # Units, not records: a shard too damaged to open reports zero
                # rows, so a record-count gate cannot see this loss at all.
                findings.append({"name": "lost_units", "message": lost})

            gap = attribution.get("unexplained")
            accounted = (
                f"The ledgers account for {attribution['declared_filtered']} filtered, "
                f"{attribution['declared_failed']} failed and "
                f"{attribution['declared_quarantined']} quarantined."
            )
            if gap and gap > 0:
                findings.append(
                    {
                        "name": "unexplained_loss",
                        "message": (f"{gap} record(s) left the pipeline for a reason no stage recorded. {accounted}"),
                    }
                )
            elif gap and gap < 0:
                # The opposite direction, and it needs its own name. Reporting it
                # as "-838 records left the pipeline" states the reverse of what
                # happened and leaves an operator with no route to the real
                # diagnosis, which is that rows appeared: a retried write, a
                # duplicated shard, an output directory that was not empty.
                findings.append(
                    {
                        "name": "unaccounted_gain",
                        "message": (
                            f"the output holds {-gap} more record(s) than the input minus what "
                            f"the ledgers declare removed. {accounted} Rows appearing is not "
                            "something filtering can cause — check for a retried or duplicated "
                            "write, or an output directory that was not empty before the run."
                        ),
                    }
                )

    report["passed"] = not findings
    return report


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Audit the corpus, write the report, and return it.

    The uniform entry point every curate step exposes: config dict in, report
    dict out, artifacts on disk as a side effect. It deliberately does **not**
    raise ``SystemExit`` — a caller running several steps needs to decide what a
    failing audit means for the rest of the run, and a step that exits the
    process takes that decision away from it. ``main`` maps the report to an
    exit code instead.
    """
    output_dir = Path(cfg.get("output_dir") or ".")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "audit_report.json"
    temporary = output_dir / ".audit_report.json.tmp"
    destination.unlink(missing_ok=True)
    temporary.unlink(missing_ok=True)

    try:
        report = audit(cfg)
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except BaseException:
        destination.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        raise
    report["artifacts"] = {"audit_report": str(destination)}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a curated corpus for integrity")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    try:
        report = run(cfg)
    except (ConfigError, integrity.ContainmentConfigError) as exc:
        print(f"curate/audit: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    for finding in report["findings"]:
        print(f"curate/audit: {finding['name']}: {finding['message']}", file=sys.stderr)
    print(f"curate/audit: report written to {report['artifacts']['audit_report']}")

    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
