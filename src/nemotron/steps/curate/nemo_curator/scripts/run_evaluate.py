#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Measure a policy against labelled documents: false rejection and noise removal.

Invoked as a module rather than through ``nemotron steps run``, like the flow: it
evaluates a policy the six registered steps produce rather than being one of
them, and it deliberately runs without NeMo Curator so it works in CI and on a
laptop before any cluster exists.

    uv run --extra curate python -m nemotron.steps.curate.nemo_curator.scripts.run_evaluate \\
        --policy ./output/vi/policy/approved_policy.yaml \\
        --labelled ./eval/vi.jsonl \\
        --langpack-dir ./langpacks

The flow writes the approved policy to ``<output_root>/policy/``, not under
``profile/`` -- ``profile/`` holds the *candidates* a person chooses from. And the
policy records which pack calibrated it but not where that pack lives, so
``--langpack-dir`` is required unless the policy was hand-edited to add one.

Two rates come back, and neither is meaningful alone. A gate that drops nothing
has a perfect false-rejection rate; a gate that drops everything has perfect
noise removal. The per-phenomenon table is the part that finds the real defect:
an aggregate over a mostly-clean set reports a good number while rejecting every
OCR-noised document in it.

Exit codes follow the category: 0 measured, 2 a problem the user can fix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from nemotron.steps.curate.nemo_curator.runtime import evaluation, langpack
from nemotron.steps.curate.nemo_curator.runtime import policy as policy_module


class ConfigError(ValueError):
    """A user-fixable problem with the arguments or the files they name."""


def _thresholds_from_policy(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    """Read an approved policy's thresholds and the pack identity it was made with.

    Reading the policy rather than taking bounds on the command line is the
    point: the thing evaluated has to be the thing that will run, or the number
    describes a policy nobody executes.
    """
    raw = Path(path).read_bytes()
    document = yaml.safe_load(raw.decode("utf-8")) or {}

    # The same gate the filter step applies. Evaluating a policy the filter would
    # refuse reports a rate for something that will never run -- and an
    # unapproved or schema-invalid policy is exactly what a person is most likely
    # to point this at while iterating.
    problems = policy_module.validate_approved_policy(document)
    if problems:
        raise ConfigError(
            f"{path}: this policy would be refused by curate/nemo_curator, so a rate for it "
            f"describes a run that cannot happen: {problems}"
        )

    thresholds = document.get("thresholds") or []
    if not thresholds:
        raise ConfigError(f"{path}: the policy declares no thresholds, so there is nothing to evaluate")
    return list(thresholds), dict(document.get("langpack") or {}), hashlib.sha256(raw).hexdigest()


def evaluate_policy(
    policy: str | Path,
    labelled: list[str] | list[Path],
    *,
    language: str | None = None,
    langpack_dir: str | Path | None = None,
) -> dict[str, Any]:
    thresholds, declared, policy_digest = _thresholds_from_policy(policy)

    tag = language or declared.get("language_tag")
    root = langpack_dir or declared.get("langpack_dir")
    if not tag or not root:
        raise ConfigError(
            "the pack is required: the thresholds are calibrated against one, and scoring without it "
            "would measure a different thing. Pass --language and --langpack-dir, or use a policy "
            "whose langpack block names them."
        )
    try:
        pack = langpack.load(str(tag), root)
    except (langpack.LanguagePackNotFoundError, langpack.LanguagePackInvalidError) as exc:
        raise ConfigError(str(exc)) from exc

    documents = evaluation.read_labelled(labelled)
    report = evaluation.evaluate(documents, thresholds, pack=pack)

    out = report.as_dict()
    # Without these a report cannot be reproduced: two runs naming the same file
    # paths may have read different bytes.
    out["policy"] = str(policy)
    out["policy_sha256"] = policy_digest
    out["labelled_sets"] = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in labelled}
    out["langpack"] = pack.describe()
    return out


def _format(report: dict[str, Any]) -> str:
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{100 * value:.1f}%"

    lines = [
        f"documents            {report['documents']}"
        f"  (labelled keep {report['labelled_keep']}, drop {report['labelled_drop']})",
        f"false rejection      {pct(report['false_rejection_rate'])}   good documents the policy dropped",
        f"noise removal        {pct(report['noise_removal_rate'])}   noise the policy caught",
        "",
        f"{'phenomenon':<16}{'docs':>6}{'false rej':>12}{'noise rem':>12}",
    ]
    caveats = []
    for name, row in report["by_phenomenon"].items():
        mark = " *" if row.get("not_exercised") else ""
        lines.append(
            f"{name:<16}{row['documents']:>6}{pct(row['false_rejection_rate']):>12}"
            f"{pct(row['noise_removal_rate']):>12}{mark}"
        )
        if row.get("not_exercised"):
            caveats.append(f"  * {name}: {row['not_exercised']}")
    if caveats:
        lines += [""] + caveats
    if report["by_signal"]:
        lines += ["", f"{'signal':<28}{'rejected keep':>14}{'rejected drop':>14}"]
        for name, row in report["by_signal"].items():
            lines.append(f"{name:<28}{row['rejected_keep']:>14}{row['rejected_drop']:>14}")
    if report["unscored_signals"]:
        lines += [
            "",
            "not scored (they wrap a NeMo Curator filter, which this harness does not load): "
            + ", ".join(report["unscored_signals"]),
        ]
    lines += ["", "A rate is worth what its labelled set is worth. See runtime/evaluation.py."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", required=True, help="approved policy whose thresholds are evaluated")
    parser.add_argument("--labelled", required=True, nargs="+", help="JSONL file(s) of labelled documents")
    parser.add_argument("--language", help="BCP-47 tag; defaults to the policy's langpack block")
    parser.add_argument("--langpack-dir", help="pack root; defaults to the policy's langpack block")
    parser.add_argument("--report", help="write the full report as JSON here")
    args = parser.parse_args(argv)

    try:
        report = evaluate_policy(
            args.policy,
            args.labelled,
            language=args.language,
            langpack_dir=args.langpack_dir,
        )
    except (ConfigError, evaluation.EvaluationDataError) as exc:
        print(f"curate/evaluate: {exc}", file=sys.stderr)
        return 2

    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(_format(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
