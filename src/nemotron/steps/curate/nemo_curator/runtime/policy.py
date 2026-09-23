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

"""Filtering policies and the line between proposing one and approving one.

Distribution analysis can say what a threshold removes. It cannot say whether
what it removes is worth removing — that needs a training run, or at minimum a
human looking at the documents. So the output of profiling must not be
executable.

Three artifacts keep that separation honest:

``profile_report.json``     measurements. Not a policy.
``candidate_policies.yaml`` proposals, ``approved: false``. Not executable.
``approved_policy.yaml``    promoted by a person or an ablation. Executable.

This module can write the first two. It has no function that writes an approved
policy, and :func:`write_candidate_policies` refuses a document claiming to be
one. Promotion is a separate, deliberate, recorded act.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1

#: Fields an approved policy must carry. A policy that cannot say which corpus,
#: which language pack, and which profile it came from cannot be audited later,
#: and a filtering decision nobody can trace is one nobody can revisit.
APPROVED_REQUIRED = (
    "schema_version",
    "approved",
    "corpus",
    "signals_impl_version",
    "profile_digest",
    "thresholds",
)

APPROVAL_METHODS = ("manual", "ablation")


class PolicyNotApprovedError(ValueError):
    """A policy was used for filtering while still marked unapproved."""


def _sha256_digest(value: Any) -> bool:
    payload = value.removeprefix("sha256:") if isinstance(value, str) else ""
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(payload) == hashlib.sha256().digest_size * 2
        and all(character in "0123456789abcdef" for character in payload.casefold())
    )


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def digest(document: Mapping[str, Any]) -> str:
    """Content hash of a report or policy, for provenance links between them."""
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def build_candidate_policies(
    *,
    candidates: Sequence[Mapping[str, Any]],
    profile_digest: str,
    signals_impl_version: str,
    corpus: Mapping[str, Any],
    langpack: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble proposed threshold sets.

    ``approved`` is written as ``False`` and is not a parameter. A caller cannot
    ask this function for an approved policy, which is the point.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "approved": False,
        "note": (
            "Candidate thresholds derived from distribution analysis. These describe what each "
            "threshold would remove; they do not establish that what it removes is low quality. "
            "Promote one only after validating it — see approval.method in the approved policy "
            "schema. curate/nemo_curator refuses an unapproved policy unless explicitly overridden."
        ),
        "corpus": dict(corpus),
        **({"langpack": dict(langpack)} if langpack else {}),
        "signals_impl_version": signals_impl_version,
        "profile_digest": profile_digest,
        "candidates": [dict(c) for c in candidates],
    }


def write_candidate_policies(path: str | Path, document: Mapping[str, Any]) -> Path:
    """Write candidates as YAML, refusing anything claiming approval."""
    if document.get("approved"):
        raise ValueError(
            "refusing to write a candidate policy marked approved. Approval is a separate, "
            "recorded act; profiling cannot perform it."
        )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.tmp")
    tmp.write_text(
        yaml.safe_dump(dict(document), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    tmp.replace(destination)
    return destination


def _swept_values(signal: Any) -> list[float]:
    """The threshold values the profile actually measured for this signal.

    Empty when the signal has no one-dimensional grid — an interval signal is
    swept as a surface, and "was this point measured" is a different question
    there, so silence is better than a warning that means something else.
    """
    grid = getattr(signal, "grid", None)
    values = getattr(grid, "values", None)
    if not callable(values):
        return []
    try:
        return [float(v) for v in values()]
    except Exception:  # noqa: BLE001 - a warning helper must never end a promotion
        return []


class PolicyNotPromotableError(ValueError):
    """A candidate cannot be turned into an approved policy as specified."""


def promote(
    candidate: Mapping[str, Any],
    *,
    thresholds: Sequence[Mapping[str, Any]],
    approval: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Turn a candidate policy into an approved one, carrying provenance forward.

    Between ``curate/profile`` and ``curate/nemo_curator`` there was no path: the
    profile emits ``candidates`` with ``bands``, the filter needs ``thresholds``
    with ``min``/``max``, and the corpus fingerprint the consumer requires was
    never produced. Anyone promoting a policy had to hand-write it and discover
    the schema by being rejected.

    This closes that gap without weakening the gate. Most of what the gate rests
    on is machine-checkable — the corpus fingerprint, the profile digest, the
    scorer version and the direction of every bound — and those checks refuse a
    wrong run in a way a signature never could.

    ``approver`` and ``evidence`` are required all the same, and the reason is
    not that a name proves something to a machine. It is that ``approved: true``
    is an artifact claiming a person decided. Letting it be written with nobody
    named and nothing cited makes the claim untrue, and every document that
    carries it downstream — the manifest's ``approved`` state, the flow report —
    inherits the untruth. A run with no one to name is not unapprovable: leave
    the policy unapproved, or set ``allow_unvalidated_policy``, which proceeds
    and records ``override_unvalidated`` instead of laundering it as approval.
    The thresholds are the caller's choice too — a band is a range, and picking a
    number inside it is the judgement the profile explicitly declines to make.

    Returns ``(document, warnings)``. Warnings name choices that are legal but
    worth seeing: most importantly a threshold outside every band that was
    actually measured, which is a number nobody has evidence for.
    """
    if not isinstance(candidate, dict):
        raise PolicyNotPromotableError(f"candidate must be a mapping, got {type(candidate).__name__}")
    if candidate.get("approved") is True:
        raise PolicyNotPromotableError(
            "this document is already approved. Promoting an approved policy again would "
            "replace one approval record with another and lose the first."
        )
    if not thresholds:
        raise PolicyNotPromotableError("no thresholds chosen; an approved policy that gates nothing is not one")

    missing = [field for field in ("approver", "evidence") if not str(approval.get(field) or "").strip()]
    if missing:
        raise PolicyNotPromotableError(
            f"approval is missing {missing}. `approved: true` is an artifact claiming a person "
            "decided; writing it with nobody named and nothing cited makes that claim untrue, and "
            "the manifest and flow report inherit it. Name an approver and cite the evidence you "
            "looked at, or leave the policy unapproved and set allow_unvalidated_policy, which "
            "records the override instead of laundering it as an approval."
        )

    from nemotron.steps.curate.nemo_curator.runtime import registry as signal_registry

    profiled: dict[str, dict[str, Any]] = {}
    for profiled_candidate in candidate.get("candidates") or []:
        if not isinstance(profiled_candidate, dict):
            continue
        profiled_name = profiled_candidate.get("signal")
        if isinstance(profiled_name, str) and profiled_name:
            profiled[profiled_name] = profiled_candidate
    warnings: list[str] = []

    for entry in thresholds:
        name = entry.get("signal")
        if not isinstance(name, str) or not name:
            raise PolicyNotPromotableError("each threshold must name a non-empty signal")
        if name not in profiled:
            raise PolicyNotPromotableError(
                f"{name!r} was not profiled on this corpus (profiled: {sorted(profiled)}). "
                "Approving a threshold for a signal nobody measured here is a number without evidence."
            )
        signal = signal_registry.SIGNALS.get(name)
        bands = profiled[name].get("bands") or []
        chosen = [entry[k] for k in ("min", "max") if k in entry]
        invalid = [value for value in chosen if not _finite_number(value)]
        if invalid:
            raise PolicyNotPromotableError(f"{name!r} has a non-finite or non-numeric threshold {invalid[0]!r}")
        if "min" in entry and "max" in entry and entry["min"] > entry["max"]:
            raise PolicyNotPromotableError(f"{name!r} has min {entry['min']!r} greater than max {entry['max']!r}")
        for value in chosen:
            if bands and not any(
                b.get("threshold_low", float("-inf")) <= value <= b.get("threshold_high", float("inf"))
                for b in bands
                if isinstance(b, dict)
            ):
                warnings.append(
                    f"{name}={value} falls outside every retention-stable band measured on this "
                    f"corpus ({[(b.get('threshold_low'), b.get('threshold_high')) for b in bands]}); "
                    "the retention at that threshold was not measured."
                )
        # Retention is measured at grid points. A threshold *between* them has no
        # measured retention of its own, so quoting the nearest swept value as if
        # it were this threshold's is how an approval record ends up citing a
        # number for a threshold nobody evaluated. Only said when it is true.
        swept = _swept_values(signal)
        offgrid = [v for v in chosen if swept and not any(abs(v - g) < 1e-9 for g in swept)]
        if offgrid:
            warnings.append(
                f"{name}={offgrid[0]} is not one of the thresholds the profile swept; its "
                "retention was not measured directly. Quote the retention of a swept value, "
                "or re-profile at this threshold."
            )

    document = {
        "schema_version": SCHEMA_VERSION,
        "approved": True,
        "corpus": dict(candidate.get("corpus") or {}),
        "signals_impl_version": candidate.get("signals_impl_version"),
        "profile_digest": candidate.get("profile_digest"),
        "approval": dict(approval),
        "thresholds": [dict(t) for t in thresholds],
    }
    if candidate.get("langpack"):
        document["langpack"] = dict(candidate["langpack"])

    problems = validate_approved_policy(document)
    if problems:
        raise PolicyNotPromotableError(
            "the promoted policy would not be executable: "
            + "; ".join(problems)
            + ". Returning it anyway would move the failure to whoever runs the pipeline."
        )
    return document, warnings


def validate_approved_policy(document: Any) -> list[str]:
    """Contract violations in an approved policy; empty means usable.

    Defined here rather than in the filtering step so the producer of a policy
    and its consumer check the same schema.
    """
    problems: list[str] = []

    if not isinstance(document, dict):
        return [f"policy must be a mapping, got {type(document).__name__}"]

    for key in APPROVED_REQUIRED:
        if key not in document:
            problems.append(f"{key} is required")

    if document.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}")

    if document.get("approved") is not True:
        problems.append("approved must be true for a policy to be executable")

    signals_impl_version = document.get("signals_impl_version")
    if not isinstance(signals_impl_version, str) or not signals_impl_version.strip():
        problems.append("signals_impl_version must be a non-empty string")

    profile_digest = document.get("profile_digest")
    if not _sha256_digest(profile_digest):
        problems.append("profile_digest must be a sha256 digest")

    # approver and evidence are required when the document claims approval, and
    # checked here as well as in promote() because a policy can be hand-written.
    # Not because a signature proves anything to a machine -- the fingerprint,
    # the profile digest, the scorer version and the direction of every bound do
    # that work, and they refuse a wrong run in a way a name never could. The
    # reason is narrower: `approved: true` asserts that a person decided, the
    # manifest and the flow report repeat that assertion downstream, and an
    # assertion nobody stands behind should not be writable. `date` stays
    # optional -- it is provenance, not accountability.
    approval = document.get("approval")
    if document.get("approved") is True:
        block = approval if isinstance(approval, dict) else {}
        for field in ("approver", "evidence"):
            if not str(block.get(field) or "").strip():
                problems.append(
                    f"approval.{field} is required when approved is true: an approval nobody is "
                    "named on is a claim the artifacts repeat and no one made. Leave the policy "
                    "unapproved, or use allow_unvalidated_policy, which records the override."
                )
    if approval is not None and not isinstance(approval, dict):
        problems.append("approval must be a mapping when present")
    elif isinstance(approval, dict) and approval.get("method") is not None:
        if approval.get("method") not in APPROVAL_METHODS:
            problems.append(f"approval.method must be one of {list(APPROVAL_METHODS)}")

    thresholds = document.get("thresholds")
    if not isinstance(thresholds, list) or not thresholds:
        problems.append("thresholds must be a non-empty list")
    else:
        # Imported here so this module still loads without the registry's
        # dependencies; the registry itself imports nemo_curator only lazily.
        from nemotron.steps.curate.nemo_curator.runtime import registry as signal_registry

        for i, entry in enumerate(thresholds):
            if not isinstance(entry, dict) or "signal" not in entry:
                problems.append(f"thresholds[{i}] must name a signal")
                continue

            name = entry.get("signal")
            if not isinstance(name, str) or not name:
                problems.append(f"thresholds[{i}].signal must be a non-empty string")
                continue
            signal = signal_registry.SIGNALS.get(name)
            if signal is None:
                # Checked here and not only at execution: this function exists so
                # the producer and consumer of a policy check the same schema, and
                # a name that validates cleanly then fails at pipeline
                # construction makes that claim false.
                problems.append(
                    f"thresholds[{i}] names unknown signal {name!r}; allowed: {sorted(signal_registry.SIGNALS)}"
                )
                continue

            present = [bound for bound in ("min", "max") if bound in entry]
            if not present:
                problems.append(f"thresholds[{i}] ({name}) sets neither min nor max")
                continue
            for bound in present:
                if not _finite_number(entry[bound]):
                    problems.append(f"thresholds[{i}] ({name}) {bound} must be a finite number, got {entry[bound]!r}")
            if (
                "min" in entry
                and "max" in entry
                and _finite_number(entry["min"])
                and _finite_number(entry["max"])
                and entry["min"] > entry["max"]
            ):
                problems.append(f"thresholds[{i}] ({name}) min must not be greater than max")

            # Direction, not merely presence. A `max` bound on a min-direction
            # signal is a valid-looking document that inverts the gate, keeping
            # exactly the documents the policy meant to drop.
            expected = {"min": ["min"], "max": ["max"], "interval": ["min", "max"]}.get(signal.direction, [])
            wrong = [b for b in present if b not in expected]
            if wrong:
                problems.append(
                    f"thresholds[{i}] ({name}) is a {signal.direction}-direction signal and "
                    f"takes {' and '.join(expected)}, but sets {wrong[0]!r}; applying it "
                    "would invert the gate"
                )
            elif len(present) != len(expected):
                missing = [b for b in expected if b not in present]
                problems.append(
                    f"thresholds[{i}] ({name}) gates from both sides and needs "
                    f"{' and '.join(expected)}, but omits {missing[0]!r}"
                )

    corpus = document.get("corpus")
    if not isinstance(corpus, dict):
        problems.append("corpus must be a mapping")
    elif not _sha256_digest(corpus.get("fingerprint")):
        problems.append(
            "corpus.fingerprint must be a sha256 digest so the policy can be tied to the data it was derived from"
        )

    return problems


def require_approved(document: Mapping[str, Any], *, allow_unvalidated: bool = False) -> list[str]:
    """Gate a policy before it is executed.

    Returns warnings when an override is in force. Raises when it is not.
    """
    problems = validate_approved_policy(document)
    if not problems:
        return []

    if allow_unvalidated:
        return [
            "allow_unvalidated_policy is set: filtering with a policy that does not meet the "
            "approval contract. Unmet: " + "; ".join(problems)
        ]

    raise PolicyNotApprovedError(
        "policy is not approved for execution: "
        + "; ".join(problems)
        + ". Promote it deliberately, or set allow_unvalidated_policy to override."
    )
