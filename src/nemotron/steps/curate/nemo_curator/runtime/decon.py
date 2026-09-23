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

"""Document near-duplicate overlap between a training corpus and a holdout.

**Threat model, fixed and narrow.** This detects *whole-document
near-duplicates*: a training document largely the same as a holdout document.

It does **not** detect a short benchmark question embedded in a long training
document. Whole-document Jaccard cannot: a 30-token question inside a 4,000-token
page moves the similarity by well under any usable threshold. That is substring
contamination, it needs a containment-oriented algorithm, and it is a different
design. Every phrase this module produces says "near-duplicate overlap detected
and removed" and never "holdout verified clean", because the second claim is one
this method cannot support.

MinHash and LSH are Curator's (``MinHashStage``, ``LSHStage``,
``BucketsToEdgesStage``) and are GPU-backed. They produce *candidate* pairs — LSH
trades recall for speed and its buckets contain false positives by construction.
What lives here is what Curator does not provide:

* the normalization applied before shingling, declared as data so a report can
  state what was compared rather than leaving it implicit
* **exact** Jaccard verification of candidate pairs, so a removal decision rests
  on a computed similarity and not on a bucket collision
* the direction rule: only the training split ever shrinks

That last one needs stating precisely, because Curator has half of it. On the
*semantic* path ``RankingStrategy.metadata_based(cols, ascending)`` orders each
cluster by arbitrary metadata and ``PairwiseCosineSimilarityStage`` scores each
document only against earlier-ranked ones, so the top-ranked member always
survives — ranking a union on a ``split`` column with ``"holdout" < "train"``
expresses exactly the direction rule, and does it better than anything here.

The fuzzy/exact path this module uses does not: ``_get_removal_ids`` selects with
``duplicated(keep="first")``, an arbitrary member of each group, and the removal
frame carries no provenance to say which side a document came from. So the rule
is enforced here because *this* path cannot express it — not because Curator
cannot. Whether the semantic path is the better substrate is an open question
this module does not settle; see the note in ``curate/decontamination``'s README.

The word "decontamination" itself appears nowhere in Curator's text stages (only
in audio ASR hallucination filtering), but that is a labelling difference, not a
missing capability.

Recall is not asserted anywhere. LSH recall depends on band/row structure and
the corpus, so a number stated here would be a guess; :func:`candidate_recall`
measures it against brute force on a sample instead.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from nemotron.steps.curate.nemo_curator.runtime.signals import is_alphanumeric

#: Bumped when a change would alter which pairs are considered duplicates.
SCHEMA_VERSION = 1

#: Shingling must match whatever generated the candidates. Curator's
#: ``FuzzyDeduplicationWorkflow`` defaults to **character** 24-grams, so these
#: defaults do too: verifying at word 5-grams what was proposed at char 24-grams
#: computes a similarity over a different set than the one MinHash approximated,
#: and the threshold then means nothing. Both are recorded in every report.
DEFAULT_SHINGLE_KIND = "char"
DEFAULT_SHINGLE_SIZE = 24

SHINGLE_KINDS = ("char", "word")

_WHITESPACE = re.compile(r"\s+")


class DeconError(ValueError):
    """A decontamination run cannot proceed as specified."""


class HoldoutModifiedError(DeconError):
    """Something tried to change a held-out split. It is never allowed."""


@dataclass(frozen=True)
class Normalization:
    """What was done to text before shingling, as declared data.

    Carried into the report so a similarity figure can be reproduced. Two runs
    that normalized differently did not measure the same thing, and without this
    there is nothing in the output that would show it.
    """

    casefold: bool = True
    nfc: bool = True
    collapse_whitespace: bool = True
    strip_punctuation: bool = True

    def apply(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        out = text
        if self.nfc:
            out = unicodedata.normalize("NFC", out)
        if self.casefold:
            out = out.casefold()
        if self.strip_punctuation:
            # Not a regex. Python's ``\w`` covers letters and digits but **not**
            # category M, so ``[^\w\s]`` strips Devanagari matras and turns
            # भाषा into भ ष — deleting the vowels and changing every shingle.
            # The shared definition accepts L, N, all of M, and the two joiners
            # that are obligatory inside Indic conjuncts.
            out = "".join(c if (is_alphanumeric(c) or c.isspace()) else " " for c in out)
        if self.collapse_whitespace:
            out = _WHITESPACE.sub(" ", out).strip()
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "casefold": self.casefold,
            "nfc": self.nfc,
            "collapse_whitespace": self.collapse_whitespace,
            "strip_punctuation": self.strip_punctuation,
        }


def shingles(
    text: str,
    size: int = DEFAULT_SHINGLE_SIZE,
    norm: Normalization | None = None,
    kind: str = DEFAULT_SHINGLE_KIND,
) -> set[str]:
    """The n-gram set MinHash approximates.

    ``kind`` must match the candidate generator. Curator shingles on characters;
    word shingles are offered because they are what a hand-rolled generator
    usually produces, and comparing across the two is the error this parameter
    exists to make impossible to commit silently.

    Returns an empty set for text too short to form one, which callers must
    treat as *unverifiable* rather than as similarity zero.
    """
    if size < 1:
        raise DeconError(f"shingle size must be at least 1, got {size}")
    if kind not in SHINGLE_KINDS:
        raise DeconError(f"shingle kind must be one of {SHINGLE_KINDS}, got {kind!r}")

    normalized = (norm or Normalization()).apply(text)
    if kind == "char":
        if len(normalized) < size:
            return set()
        return {normalized[i : i + size] for i in range(len(normalized) - size + 1)}

    words = normalized.split()
    if len(words) < size:
        return set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    """Exact Jaccard similarity. ``0.0`` when either side is empty."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


@dataclass(frozen=True)
class Pair:
    """A candidate pair and the exact similarity computed for it."""

    train_id: str
    holdout_id: str
    similarity: float
    verifiable: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_id": self.train_id,
            "holdout_id": self.holdout_id,
            "similarity": round(self.similarity, 6),
            "verifiable": self.verifiable,
            "reason": self.reason,
        }


def verify_pairs(
    candidates: Iterable[tuple[str, str]],
    train_text: dict[str, str],
    holdout_text: dict[str, str],
    *,
    threshold: float,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
    shingle_kind: str = DEFAULT_SHINGLE_KIND,
    norm: Normalization | None = None,
) -> list[Pair]:
    """Compute exact Jaccard for each candidate pair LSH proposed.

    LSH buckets contain false positives by construction — that is the trade it
    makes for speed. Removing a document because it shared a bucket, without
    computing the similarity, removes training data on the strength of a hash
    collision.

    A pair whose text is too short to shingle is returned with
    ``verifiable=False`` and is **not** counted as a duplicate. Treating it as
    similarity zero would report a clean result for a comparison that never
    happened.
    """
    if not 0.0 < threshold <= 1.0:
        raise DeconError(f"threshold must be in (0, 1], got {threshold}")

    norm = norm or Normalization()
    cache: dict[tuple[str, str], set[str]] = {}

    def shingle_for(side: str, key: str, text: str) -> set[str]:
        entry = (side, key)
        if entry not in cache:
            cache[entry] = shingles(text, shingle_size, norm, shingle_kind)
        return cache[entry]

    verified: list[Pair] = []
    for train_id, holdout_id in candidates:
        left_text = train_text.get(train_id)
        right_text = holdout_text.get(holdout_id)
        if left_text is None or right_text is None:
            verified.append(Pair(train_id, holdout_id, 0.0, False, "a document in the pair was not found"))
            continue

        left = shingle_for("train", train_id, left_text)
        right = shingle_for("holdout", holdout_id, right_text)
        if not left or not right:
            verified.append(
                Pair(
                    train_id,
                    holdout_id,
                    0.0,
                    False,
                    f"fewer than {shingle_size} {shingle_kind}s after normalization; "
                    "too short for a document-level similarity to mean anything",
                )
            )
            continue

        verified.append(Pair(train_id, holdout_id, jaccard(left, right), True))

    return verified


def removals(pairs: Sequence[Pair], threshold: float) -> dict[str, Pair]:
    """Training ids to remove, each with the pair that justified it.

    Keyed by training id and keeping the strongest match, so a document matching
    several holdout documents is removed once and reported with its best
    evidence.
    """
    chosen: dict[str, Pair] = {}
    for pair in pairs:
        if not pair.verifiable or pair.similarity < threshold:
            continue
        current = chosen.get(pair.train_id)
        if current is None or pair.similarity > current.similarity:
            chosen[pair.train_id] = pair
    return chosen


def assert_holdout_untouched(before: Sequence[str], after: Sequence[str]) -> None:
    """Only the training split shrinks. Ever.

    Removing a leaked document from the holdout would make the benchmark agree
    with the training data by changing the benchmark, which is the one repair
    that invalidates every number computed against it — including numbers
    already published from earlier runs.
    """
    lost = sorted(set(before) - set(after))
    if lost:
        raise HoldoutModifiedError(
            f"{len(lost)} document(s) were removed from the holdout split, e.g. "
            f"{', '.join(repr(d) for d in lost[:3])}. Only the training split may shrink: "
            "changing a benchmark to agree with the training data invalidates every "
            "result measured against it."
        )


def candidate_recall(
    candidates: Iterable[tuple[str, str]],
    train_text: dict[str, str],
    holdout_text: dict[str, str],
    *,
    threshold: float,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
    shingle_kind: str = DEFAULT_SHINGLE_KIND,
    norm: Normalization | None = None,
) -> dict[str, Any]:
    """Measure what the candidate generator missed, by brute force.

    Recall is a property of the LSH band/row structure *and* the corpus, so a
    number asserted in documentation would be a guess. This computes it on a
    sample small enough to compare every pair, which is a benchmark rather than
    a claim.

    Quadratic on purpose. Never call it on a full corpus.
    """
    norm = norm or Normalization()
    train_shingles = {k: shingles(v, shingle_size, norm, shingle_kind) for k, v in train_text.items()}
    holdout_shingles = {k: shingles(v, shingle_size, norm, shingle_kind) for k, v in holdout_text.items()}

    truth = {
        (t, h)
        for t, ts in train_shingles.items()
        for h, hs in holdout_shingles.items()
        if ts and hs and jaccard(ts, hs) >= threshold
    }
    proposed = set(candidates)
    found = truth & proposed

    return {
        "threshold": threshold,
        "shingle_size": shingle_size,
        "shingle_kind": shingle_kind,
        "true_pairs": len(truth),
        "candidate_pairs": len(proposed),
        "recalled": len(found),
        "missed": sorted(truth - proposed),
        "recall": len(found) / len(truth) if truth else None,
        "false_positive_rate": ((len(proposed) - len(proposed & truth)) / len(proposed) if proposed else None),
        "note": (
            "Measured by brute force on this sample. Recall depends on the LSH band and "
            "row structure and on the corpus, so it does not transfer to another run."
        ),
    }


def corpus_fingerprint(ids: Iterable[str]) -> str:
    """Order-independent digest of a split's membership.

    Lets a report state which corpus a decontamination decision was made against
    without embedding the whole id list.
    """
    digest = hashlib.sha256()
    for value in sorted(set(ids)):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"
