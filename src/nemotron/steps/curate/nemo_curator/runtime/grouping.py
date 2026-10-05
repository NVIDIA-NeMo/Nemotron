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

"""Source-document identity, so related records never straddle a split.

Splitting a corpus on a hash of the document text guarantees that *byte
identical* copies land together, and nothing more. That is weaker than it
sounds:

* two crawls of the same page differing only in a nav bar or a timestamp hash
  differently and can land on opposite sides of a train/holdout boundary;
* a corpus that joins ``title`` and ``content`` into ``text`` gives the same
  article republished under a slightly different title a different hash;
* the same source record re-emitted by two crawls keeps one stable ``id`` that a
  content hash ignores entirely.

Each of those is a leak: near-identical text on both sides of a split, which
inflates held-out scores without anything failing. So the group key has to be
the most stable identity available, not the content hash.

Precedence, first non-empty winning:

1. ``url`` — canonicalised, so ``http://a.example/x?utm_source=ads`` and
   ``https://www.A.example/x/`` group together
2. ``id`` — namespaced, because ids are only unique within a corpus; a
   cross-split comparison shares one namespace so a document present in both
   splits keeps a single key
3. normalised-text hash — NFC, casefolded, whitespace-collapsed, which still
   groups text a raw hash would separate
4. raw content hash, then a positional identity as an absolute last resort

The key that was used is recorded alongside the key itself, so a grouping
decision can be inspected after the fact rather than inferred.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

GROUP_KEY_FIELD = "__group_key"
GROUP_KEY_SOURCE_FIELD = "__group_key_field"

#: The name group_key reports when it fell through to a positional identity.
#: Per-split by construction, so it never establishes comparability.
POSITIONAL_FIELD = "_rowid"
NORM_HASH_FIELD = "__norm_text_hash"

#: Default id namespace for a cross-split comparison. Both sides share it because
#: a train split and a holdout split are normally one corpus divided two ways, so
#: the same id means the same document. Namespacing by side instead would give one
#: document two keys and find no overlap at all.
SHARED_CORPUS = "corpus"

_WHITESPACE = re.compile(r"\s+")
_SCHEME = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.IGNORECASE)

#: Corpora do not agree on what the URL column is called. Reading only a field
#: literally named ``url`` silently demotes corpora that use another name to
#: content-hash grouping — which is exactly the case same-page near-duplicates
#: leak through. Checked in this order.
URL_FIELD_ALIASES: tuple[str, ...] = (
    "url",
    "warc-target-uri",
    "warc_target_uri",
    "source_url",
    "uri",
    "link",
)


#: Query parameters that identify a *campaign*, not a page. Dropped during
#: canonicalisation; everything else is kept, because on a query-driven site the
#: parameter is the page. Deliberately a small, well-known list — a permissive
#: one would start merging documents again.
TRACKING_PARAMS: frozenset[str] = frozenset(
    {
        # Google Analytics / UTM. The utm_ namespace is larger than the classic
        # five: GA4 emits the last three officially, and omitting one splits a
        # page from its own untagged copy — an under-merge, which on the
        # skip_similarity path is a leak nothing else recovers.
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "utm_source_platform",
        "utm_creative_format",
        "utm_marketing_tactic",
        "utm_referrer",
        # Click identifiers. gbraid/wbraid replaced gclid for iOS traffic, and
        # ttclid is high-volume on the Vietnamese web this pack targets.
        "gclid",
        "gclsrc",
        "dclid",
        "gbraid",
        "wbraid",
        "fbclid",
        "msclkid",
        "ttclid",
        "twclid",
        "yclid",
        "igshid",
        "srsltid",
        "epik",
        # Mail and analytics linkers.
        "mc_cid",
        "mc_eid",
        "mkt_tok",
        "_ga",
        "_gl",
        "_openstat",
        # Referral attribution. ``ref`` is deliberately NOT here: on every
        # Git-hosting site it selects a branch or commit, so dropping it would
        # collapse distinct file revisions into one group — the same over-merge
        # that stripping the whole query caused. ``ref_src``/``ref_url`` are
        # Twitter-specific and unambiguous.
        "ref_src",
        "ref_url",
        "spm",
    }
)


def resolve_url_field(fields: Iterable[str]) -> str | None:
    """First URL-ish field present, in precedence order, or ``None``.

    Some corpora genuinely have no URL. Callers must fall through to the next
    key rather than assuming one exists.
    """
    present = set(fields)
    for name in URL_FIELD_ALIASES:
        if name in present:
            return name
    return None


def canonical_url(url: Any) -> str:
    """Strip scheme, ``www.``, fragment, tracking parameters and trailing slash.

    Two crawls of one page routinely differ only by ``http`` vs ``https`` or a
    tracking parameter. Without canonicalisation they are different groups, and
    the same page can end up on both sides of a split.

    **Tracking parameters are dropped; every other parameter is kept.** Dropping
    the whole query string is the tempting shortcut and it silently merges
    distinct pages on any query-driven site: measured on Vietnamese C4,
    ``forums.voz.vn/showthread.php?t=<thread>`` collapsed every thread on that
    path into one group, and a single shared path then pulled 29 unrelated
    documents out of a training split whose pairwise Jaccard was 0.00.

    Remaining parameters are sorted, so two orderings of the same query are one
    group rather than two.
    """
    if not isinstance(url, str):
        return ""
    value = url.strip()
    if not value:
        return ""
    value = _SCHEME.sub("", value)
    value = value.split("#", 1)[0]

    head, sep, query = value.partition("?")
    head = head.rstrip("/")
    # Repeated, not once: a rewrite rule that prepends ``www.`` to a host that
    # already has it produces ``www.www.example.com``, and stripping a single
    # prefix would leave that in a different group from the page it duplicates.
    while head.lower().startswith("www."):
        head = head[4:]
    head = head.lower()

    if not sep or not query:
        return head

    kept = sorted(
        part for part in query.split("&") if part and part.split("=", 1)[0].strip().lower() not in TRACKING_PARAMS
    )
    return f"{head}?{'&'.join(kept)}" if kept else head


def normalize_text(text: Any) -> str:
    """NFC, casefold, collapse whitespace. Empty string for anything unusable."""
    if not isinstance(text, str):
        return ""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def normalized_text_hash(text: Any) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass
class GroupKeyConfig:
    """Which fields identify a source document, in precedence order.

    ``id_field`` names the record's own identifier. It is both the last resort
    for keying and the value reported back for every match, so a corpus whose id
    lives anywhere other than ``id`` needs it set: leaving the default reads a
    field that is not there, keys every record off the positional fallback —
    which is per-side and so can never match across two splits — and reports
    every matched id as the empty string.
    """

    fields: list[str] | None = None
    text_field: str = "text"
    hash_field: str = "__text_hash"
    use_normalized_text: bool = True
    id_field: str = "id"
    #: Put ``id_field`` ahead of the URL aliases instead of behind them.
    #:
    #: The default order is right when the id is synthetic and the URL is the
    #: real identity of a page. It is wrong when the URL column is coarse: a
    #: ``source_url`` holding a site homepage groups every document from that
    #: site together, so one holdout page marks the whole site's training
    #: documents as overlapping and they are removed without being duplicates
    #: of anything. Set this when the corpus id is the authoritative one.
    prefer_id_field: bool = False

    def __post_init__(self) -> None:
        if self.fields is None:
            self.fields = (
                [self.id_field, *URL_FIELD_ALIASES] if self.prefer_id_field else [*URL_FIELD_ALIASES, self.id_field]
            )


def group_key(
    record: dict[str, Any],
    source: str,
    cfg: GroupKeyConfig | None = None,
    *,
    id_namespace: str | None = None,
) -> tuple[str, str]:
    """Return ``(key, which_field_produced_it)`` for one record.

    Never returns an empty key. The fallbacks terminate at a synthetic identity
    so every record receives exactly one group — a record dropped for lacking a
    key is a record silently missing from the split it belonged to.
    """
    cfg = cfg or GroupKeyConfig()

    for name in cfg.fields:
        if name not in record:
            continue
        if name != cfg.id_field and name in URL_FIELD_ALIASES:
            # Classified by role, not by name. A corpus whose identifier column
            # happens to be called `uri` or `link` had its ids run through
            # canonical_url -- lowercased, scheme-stripped, query-trimmed -- so
            # two distinct ids could collapse to one key, and the configured
            # id_field was the one thing that should never be reinterpreted.
            #
            # Every alias produces the same ``url:`` namespace, so a corpus that
            # renames the field still groups with one that does not.
            canonical = canonical_url(record[name])
            if canonical:
                return f"url:{canonical}", name
        else:
            value = str(record[name]).strip() if record[name] is not None else ""
            if value:
                # Ids are only unique inside their own corpus, so they are
                # namespaced. ``id_namespace`` exists because the namespace must
                # be the *corpus*, not whatever label the caller gave this batch:
                # comparing a train split against a holdout split of one corpus
                # under the labels "train" and "holdout" would give the same
                # document two different keys and find no overlap at all.
                namespace = source if id_namespace is None else id_namespace
                return f"{name}@{namespace}:{value}", name

    if cfg.use_normalized_text:
        existing = record.get(NORM_HASH_FIELD)
        digest = str(existing) if existing else normalized_text_hash(record.get(cfg.text_field))
        if digest:
            return f"norm:{digest}", NORM_HASH_FIELD

    raw = record.get(cfg.hash_field)
    if raw:
        return f"hash:{raw}", cfg.hash_field

    return "", ""


def assign_group_keys(
    records: Iterable[dict[str, Any]],
    source: str,
    cfg: GroupKeyConfig | None = None,
    *,
    id_namespace: str | None = None,
) -> list[dict[str, Any]]:
    """Annotate every record with its group key and the field that produced it."""
    out = []
    for index, record in enumerate(records):
        key, which = group_key(record, source, cfg, id_namespace=id_namespace)
        if not key:
            # Absolute last resort: a positional identity. This one keeps
            # ``source`` and is deliberately NOT shared across splits — row 0 of
            # the training split is not the same document as row 0 of the
            # holdout, and giving them one key would invent an overlap.
            key, which = f"row:{source}:{index}", "_rowid"
        out.append({**record, GROUP_KEY_FIELD: key, GROUP_KEY_SOURCE_FIELD: which})
    return out


def cross_split_groups(
    left: Iterable[dict[str, Any]],
    right: Iterable[dict[str, Any]],
    *,
    left_source: str = "train",
    right_source: str = "holdout",
    cfg: GroupKeyConfig | None = None,
    id_namespace: str | None = SHARED_CORPUS,
) -> dict[str, Any]:
    """Group keys appearing on both sides of a split.

    This is a **cheap exact check that runs before any similarity work** and
    catches a class of leak that near-duplicate detection can miss entirely: the
    same page, rewritten enough that its shingles no longer overlap, is still
    the same source document and still must not span a split.

    ``id_namespace`` is shared by both sides by default, and that default is the
    whole reason document ids work here at all. A train split and a holdout split
    are normally one corpus divided two ways, so a document appearing in both
    carries the *same* id — namespacing it by the side it was read from would
    give one document two keys and report no overlap. A corpus without a URL
    field, which falls straight through to the id, would then be checked by a
    comparison that structurally cannot match.

    Pass ``id_namespace=None`` when the two sides genuinely come from different
    corpora, where equal ids do not imply the same document. The positional
    fallback stays per-side regardless: row 0 of one split is not row 0 of the
    other.

    Reported rather than acted on. Which side loses a record is a policy
    decision, and for a decontamination run the answer is always the training
    split — but that belongs to the caller, not here.
    """
    # Normalised once so the reported id and the keying rules come from the same
    # object. Callers may pass None, and the defaults have to agree.
    cfg = cfg or GroupKeyConfig()

    right_keys: dict[str, list[str]] = {}
    right_fields: Counter[str] = Counter()
    for record in assign_group_keys(right, right_source, cfg, id_namespace=id_namespace):
        right_keys.setdefault(record[GROUP_KEY_FIELD], []).append(str(record.get(cfg.id_field, "")))
        right_fields[record[GROUP_KEY_SOURCE_FIELD]] += 1

    shared: dict[str, dict[str, Any]] = {}
    left_fields: Counter[str] = Counter()
    left_total = 0
    for record in assign_group_keys(left, left_source, cfg, id_namespace=id_namespace):
        left_total += 1
        left_fields[record[GROUP_KEY_SOURCE_FIELD]] += 1
        key = record[GROUP_KEY_FIELD]
        if key in right_keys:
            entry = shared.setdefault(
                key,
                {
                    "key": key,
                    "key_field": record[GROUP_KEY_SOURCE_FIELD],
                    "left_ids": [],
                    "right_ids": right_keys[key],
                },
            )
            entry["left_ids"].append(str(record.get(cfg.id_field, "")))

    # Each key is namespaced by the field that produced it, so a record keyed off
    # a field the other side never uses sits in a key space nothing can match. It
    # contributes a guaranteed zero, and a count alone cannot distinguish that
    # from a genuine absence of overlap.
    #
    # Counting RECORDS rather than intersecting field NAMES is the whole point.
    # A holdout that keys half its records off url and half off id shares the url
    # field with a url-keyed corpus, so an intersection test calls the comparison
    # sound while the id-keyed half can never match -- and it is the half nobody
    # is told about. _rowid is excluded on both sides because it is positional
    # and deliberately per-split: it can never match by construction, so counting
    # it as a shared field would satisfy the check on its own.
    left_usable = {f for f in left_fields if f != POSITIONAL_FIELD}
    right_usable = {f for f in right_fields if f != POSITIONAL_FIELD}
    unmatchable_left = sum(n for f, n in left_fields.items() if f not in right_usable)
    unmatchable_right = sum(n for f, n in right_fields.items() if f not in left_usable)
    comparable = unmatchable_left == 0 and unmatchable_right == 0

    return {
        "left_records": left_total,
        "right_records": sum(len(v) for v in right_keys.values()),
        "shared_group_count": len(shared),
        "left_records_affected": sum(len(v["left_ids"]) for v in shared.values()),
        "shared_groups": [shared[k] for k in sorted(shared)],
        "left_key_fields": dict(sorted(left_fields.items())),
        "right_key_fields": dict(sorted(right_fields.items())),
        "unmatchable_left": unmatchable_left,
        "unmatchable_right": unmatchable_right,
        "comparable": comparable,
    }
