# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Does a threshold keep the right documents? Measured against labelled data.

A retention curve says how much a gate removes. It cannot say whether removing
it was right, and the two questions have different answers in opposite
directions: a threshold that keeps 93% of a corpus may be discarding exactly the
7% that was good, and one that keeps 99% may be leaving the noise in. Retention
was the only quality evidence this category produced.

Two rates, and both are needed. Reporting either alone is reporting nothing:

*False rejection* -- of the documents a human labelled ``keep``, how many did the
thresholds drop. Good content lost. A gate that removes nothing scores 0 here and
is still useless.

*Noise removal* -- of the documents a human labelled ``drop``, how many did the
thresholds drop. Junk caught. A gate that removes everything scores 1 here and
has destroyed the corpus.

**What this measures and what it does not.** Every score comes from the signals
implemented in ``runtime/signals.py`` -- the pack-backed ones plus
``unicode_alpha_numeric``. Those are the language-sensitive measurements, which
is what a multilingual evaluation is about, and they run without NeMo Curator so
this is executable in CI. Signals that wrap a Curator filter are NOT evaluated
here; the report names them rather than scoring them silently.

**A number from this harness is a regression signal, not a quality claim.** It is
worth exactly as much as the labelled set behind it. A set constructed to exercise
known failure modes -- which is what ships -- proves the pipeline still handles
those modes. It does not establish a rate for your corpus, and the report says so
in its own text rather than leaving a reader to assume otherwise.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemotron.steps.curate.nemo_curator.runtime import registry as signal_registry

#: What a human can say about a document.
LABELS = ("keep", "drop")

#: The failure modes a multilingual evaluation has to exercise. Named here rather
#: than left to whoever writes the data, because a set that happens to contain no
#: OCR noise reports a false-rejection rate that says nothing about OCR noise.
PHENOMENA = (
    "clean",
    "nfd",
    "local_digits",
    "ocr_noise",
    "code_mixing",
    "langid_error",
)

#: Modes this harness labels but does not exercise. The rate reported for one is
#: real -- it is what the policy's signals did to those documents -- but it is
#: not evidence about the mode itself, and a table row with a percentage in it
#: reads like evidence. ``langid_error`` needs the FastText model actually run
#: against the document; the harness deliberately loads no models, so the label
#: describes the document rather than testing the pipeline's handling of it.
NOT_EXERCISED = {
    "langid_error": (
        "the FastText language gate is not run by this harness, so this row shows how the "
        "policy's own signals treat these documents, not whether language-ID errors are handled"
    )
}


class EvaluationDataError(ValueError):
    """The labelled set does not meet the contract."""


@dataclass(frozen=True)
class Judgement:
    """One document: what a person said, and what the thresholds did."""

    doc_id: str
    label: str
    phenomenon: str
    kept: bool
    #: Signals that rejected it, in registry order. Empty when kept.
    rejected_by: tuple[str, ...] = ()

    @property
    def false_rejection(self) -> bool:
        return self.label == "keep" and not self.kept

    @property
    def noise_removed(self) -> bool:
        return self.label == "drop" and not self.kept


def _rate(numerator: int, denominator: int) -> float | None:
    """None, not zero, when there is nothing to divide by.

    Zero would read as a measured rate of nothing going wrong, which is the
    opposite of "this set contains no documents of that kind".
    """
    return None if denominator == 0 else numerator / denominator


@dataclass(frozen=True)
class Report:
    judgements: tuple[Judgement, ...]
    #: Signals named by the thresholds that this harness cannot score.
    unscored_signals: tuple[str, ...] = ()
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def kept_labels(self) -> int:
        return sum(1 for j in self.judgements if j.label == "keep")

    @property
    def drop_labels(self) -> int:
        return sum(1 for j in self.judgements if j.label == "drop")

    @property
    def false_rejection_rate(self) -> float | None:
        return _rate(sum(1 for j in self.judgements if j.false_rejection), self.kept_labels)

    @property
    def noise_removal_rate(self) -> float | None:
        return _rate(sum(1 for j in self.judgements if j.noise_removed), self.drop_labels)

    def by_phenomenon(self) -> dict[str, dict[str, Any]]:
        """Per failure mode, because an aggregate hides the one that is broken.

        A set that is 90% clean text reports a good overall rate while rejecting
        every OCR-noised document in it.
        """
        out: dict[str, dict[str, Any]] = {}
        for phenomenon in sorted({j.phenomenon for j in self.judgements}):
            rows = [j for j in self.judgements if j.phenomenon == phenomenon]
            keeps = [j for j in rows if j.label == "keep"]
            drops = [j for j in rows if j.label == "drop"]
            out[phenomenon] = {
                "documents": len(rows),
                "labelled_keep": len(keeps),
                "labelled_drop": len(drops),
                "false_rejection_rate": _rate(sum(1 for j in keeps if not j.kept), len(keeps)),
                "noise_removal_rate": _rate(sum(1 for j in drops if not j.kept), len(drops)),
            }
            if phenomenon in NOT_EXERCISED:
                out[phenomenon]["not_exercised"] = NOT_EXERCISED[phenomenon]
        return out

    def by_signal(self) -> dict[str, dict[str, int]]:
        """Which signal did the rejecting. Attribution, not just a total.

        "The policy rejects 12% of good Hindi" sends someone hunting; "script_ratio
        rejects 12% of good Hindi" names the threshold to move.
        """
        out: dict[str, dict[str, int]] = {}
        for judgement in self.judgements:
            for name in judgement.rejected_by:
                entry = out.setdefault(name, {"rejected_keep": 0, "rejected_drop": 0})
                entry["rejected_keep" if judgement.label == "keep" else "rejected_drop"] += 1
        return dict(sorted(out.items()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "documents": len(self.judgements),
            "labelled_keep": self.kept_labels,
            "labelled_drop": self.drop_labels,
            "false_rejection_rate": self.false_rejection_rate,
            "noise_removal_rate": self.noise_removal_rate,
            "by_phenomenon": self.by_phenomenon(),
            "by_signal": self.by_signal(),
            "unscored_signals": list(self.unscored_signals),
            "notes": dict(self.notes),
        }


def read_labelled(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Read a labelled set, refusing anything that would make the rates a lie.

    Every defect here is one that produces a plausible number rather than an
    error: a missing label silently shrinks the denominator, an unknown
    phenomenon hides a whole failure mode from the per-mode table, and a
    duplicate id double-counts one document's verdict.
    """
    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationDataError(f"{path}:{lineno}: not JSON: {exc}") from exc
            for key in ("id", "text", "label", "phenomenon"):
                if not record.get(key):
                    raise EvaluationDataError(f"{path}:{lineno}: '{key}' is required")
            if record["label"] not in LABELS:
                raise EvaluationDataError(f"{path}:{lineno}: label must be one of {list(LABELS)}")
            if record["phenomenon"] not in PHENOMENA:
                raise EvaluationDataError(
                    f"{path}:{lineno}: phenomenon {record['phenomenon']!r} is not one of {list(PHENOMENA)}. "
                    "A mode with no name does not appear in the per-mode table."
                )
            if record["id"] in seen:
                raise EvaluationDataError(f"{path}:{lineno}: duplicate id {record['id']!r}")
            seen.add(record["id"])
            documents.append(record)
    if not documents:
        raise EvaluationDataError("the labelled set is empty; there is nothing to measure")
    return documents


def evaluate(
    documents: Sequence[dict[str, Any]],
    thresholds: Sequence[dict[str, Any]],
    *,
    pack: Any,
    text_field: str = "text",
) -> Report:
    """Apply a policy's thresholds to labelled documents and report both rates.

    ``thresholds`` is the shape an approved policy carries -- ``{"signal": name,
    "min"/"max": bound}`` -- so a policy can be evaluated exactly as it will run
    rather than through a re-implementation that might disagree with it.
    """
    scorers: list[tuple[str, Any]] = []
    unscored: list[str] = []
    for entry in thresholds:
        name = entry.get("signal")
        if name not in signal_registry.SIGNALS:
            raise EvaluationDataError(f"unknown signal {name!r}")
        signal = signal_registry.SIGNALS[name]
        # The step's own mapping, not a second one. Zipping min/max positionally
        # onto threshold_params maps a `max:` bound onto a `min_*` parameter and
        # inverts the gate, so the evaluator would report rates for a policy that
        # gates the opposite way from the one that will run.
        bounds = signal_registry.threshold_bounds(signal, entry)
        extra = {"pack": pack} if name in signal_registry.PACK_SIGNALS else {}
        try:
            scorers.append((name, signal.build(*bounds, **extra)))
        except (ImportError, ModuleNotFoundError):
            # The one build failure that is not a defect: a Curator-backed signal
            # on a machine without Curator. Named, never silent -- a policy only
            # half applied does not produce a rate for the policy.
            unscored.append(name)

    if not scorers:
        raise EvaluationDataError(
            "no threshold in this policy could be scored here, so there is nothing to measure. "
            f"Unscorable: {sorted(set(unscored))}."
        )

    # The `language` field was read into the report and otherwise ignored, so a
    # Hindi set could be scored with the Vietnamese pack and every rate came back
    # looking ordinary. The pack decides what the scores mean; a document from a
    # different language is measured against the wrong word lists and charset.
    tag = str(getattr(pack, "language_tag", "") or "")
    wrong = sorted({str(d["language"]) for d in documents if d.get("language") and str(d["language"]) != tag})
    if wrong:
        raise EvaluationDataError(
            f"the labelled set contains documents in {wrong} but the pack is {tag!r}. "
            "Scoring them against this pack measures the wrong word lists and character set. "
            "Evaluate one language at a time."
        )

    judgements = []
    for record in documents:
        text = record.get(text_field) or ""
        rejected = []
        for name, scorer in scorers:
            # Deliberately not guarded. A scorer that raises leaves the verdict
            # unknown, and counting the document as kept was the failure this
            # harness exists to detect: noise a scorer choked on read as noise the
            # policy let through, and the rate looked better for it.
            if not scorer.keep_document(scorer.score_document(text)):
                rejected.append(name)
        judgements.append(
            Judgement(
                doc_id=str(record["id"]),
                label=record["label"],
                phenomenon=record["phenomenon"],
                kept=not rejected,
                rejected_by=tuple(rejected),
            )
        )

    notes = {
        "completeness": (
            "every threshold in the policy was scored"
            if not unscored
            else f"INCOMPLETE: {sorted(set(unscored))} could not be built here, so these rates "
            "describe a policy with those thresholds removed, not the policy as written"
        ),
        "scope": (
            "Scores come from the locally implemented signals only. Signals wrapping a NeMo Curator "
            "filter are listed under unscored_signals rather than scored, because this harness runs "
            "without Curator."
        ),
        "interpretation": (
            "A rate here is worth what the labelled set behind it is worth. A set built to exercise "
            "known failure modes shows the pipeline still handles those modes; it does not establish "
            "a rate for any particular corpus."
        ),
    }
    return Report(tuple(judgements), tuple(sorted(set(unscored))), notes)
