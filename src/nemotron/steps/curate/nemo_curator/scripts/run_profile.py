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

"""CLI for ``steps/curate/profile``.

Two passes over the corpus. The first counts documents per source, because a
sample budget cannot be split proportionally without knowing the proportions.
The second draws the allocated number per source and scores only those.

Signals are Curator's own ``DocumentFilter`` classes, called through
:mod:`nemotron.steps.curate.nemo_curator.runtime.registry`. They are invoked directly rather
than through Curator's ``Score`` stage: ``Score.process`` is
``df[col] = df[text].apply(fn)``, so on a bounded sample a Ray cluster would add
startup cost without changing the arithmetic. The signals are Curator's either
way; only the loop around them is local. Bypassing the stage means bypassing its
``setup``, so :func:`load_models` replicates the part of it a filter needs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from nemotron.steps.curate.nemo_curator.runtime import determinism, integrity, langpack, policy, profiling
from nemotron.steps.curate.nemo_curator.runtime import manifest as manifest_mod
from nemotron.steps.curate.nemo_curator.runtime import registry as signal_registry

logger = logging.getLogger("curate.profile")

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "profile" / "config" / "default.yaml"

SHARD_SOURCE_NOTE = (
    "source_field was not present in the records, so each shard path is treated as its own "
    "source. Per-source figures below therefore describe shards, not corpora."
)

#: Below this share of successfully scored documents a distribution describes so
#: little of the sample that reporting quantiles for it would mislead.
MIN_SCORED_FRACTION = 0.5


class SignalUnavailableError(RuntimeError):
    """A registered signal could not be constructed for this run."""


class ConfigError(ValueError):
    """The config asks for something that cannot be reported honestly."""


def resolve_tokenizer(cfg: dict[str, Any]) -> tuple[str, str] | None:
    """The tokenizer for ``token_count``, as a ``(name, revision)`` pair.

    A revision is required for the same reason ``curate/subset`` requires one:
    two tokenizer revisions do not produce comparable counts, and a threshold
    promoted from a profile that cannot name its revision cannot be checked
    against the corpus it was calibrated on.
    """
    block = (cfg.get("models") or {}).get("tokenizer")
    if block in (None, False, ""):
        return None
    if not isinstance(block, dict) or not block.get("name"):
        raise ConfigError(
            "models.tokenizer must be a mapping with a name and a revision, or absent to "
            "skip the token_count signal. A bare model name cannot be pinned."
        )
    if not block.get("revision"):
        raise ConfigError(
            "models.tokenizer.revision is required. Token counts from two revisions are not "
            "comparable, so a token_count threshold promoted from this profile could not be "
            "verified later. Remove models.tokenizer to skip the signal instead."
        )
    return str(block["name"]), str(block["revision"])


def expand(pattern: str | list[str] | None) -> list[str]:
    """Expand a glob, a directory, or a literal path.

    Delegates rather than repeating the logic. This was the sixth copy of the
    same resolver, and the one left behind when the others were consolidated:
    it matched only ``*.jsonl`` inside a directory, so the same reference
    resolved to a different corpus depending on which step read it.
    """
    return integrity.expand_inputs(pattern)


def iter_records(paths: list[str], damage: dict[str, int] | None = None) -> Iterator[tuple[str, dict[str, Any]]]:
    """Shared with every other step. See :func:`integrity.iter_records`."""
    return integrity.iter_records(paths, damage)


def source_of(path: str, record: dict[str, Any], source_field: str | None) -> str:
    if source_field and record.get(source_field) is not None:
        return str(record[source_field])
    return path


def count_sources(
    paths: list[str], source_field: str | None, text_field: str, id_field: str | None = None
) -> tuple[dict[str, int], dict[str, int]]:
    """First pass: how many profileable documents each source holds.

    Only records the second pass can actually sample are counted. Counting a
    record here that pass two skips would make every per-source weight describe
    a population the sample was never drawn from, which silently biases the
    micro view.

    The corpus fingerprint is accumulated here rather than in a pass of its own:
    this pass already reads every record, and a policy cannot be promoted without
    a fingerprint tying its thresholds to the data they were measured on.
    """
    fingerprint = integrity.RowDigest()
    populations: dict[str, int] = {}
    stats: dict[str, Any] = {
        "records": 0,
        "with_source_field": 0,
        "without_text": 0,
        "unparsable_lines": 0,
    }
    damage: dict[str, int] = {}

    for path, record in iter_records(paths, damage):
        stats["records"] += 1
        if not isinstance(record.get(text_field), str):
            stats["without_text"] += 1
            continue
        if source_field and record.get(source_field) is not None:
            stats["with_source_field"] += 1
        key = source_of(path, record, source_field)
        populations[key] = populations.get(key, 0) + 1
        # id + content, never id alone: two corpora sharing an id scheme would
        # otherwise fingerprint identically, and the policy promoted from this
        # profile would verify cleanly against the wrong data.
        content = hashlib.sha256(record[text_field].encode("utf-8")).hexdigest()
        doc_id = record.get(id_field) if id_field else None
        fingerprint.add(f"{doc_id}\0{content}" if doc_id is not None else content)

    stats["unparsable_lines"] = sum(damage.values())
    stats["damaged_shards"] = len(damage)
    stats["fingerprint"] = fingerprint.hexdigest()
    return populations, stats


def draw_sample(
    paths: list[str],
    allocations: dict[str, determinism.SourceAllocation],
    *,
    source_field: str | None,
    text_field: str,
    id_field: str | None,
    seed: int,
) -> dict[str, list[tuple[str, str]]]:
    """Second pass: the allocated number of documents from each source."""
    records: list[tuple[str, str, str]] = []
    for index, (path, record) in enumerate(iter_records(paths)):
        text = record.get(text_field)
        if not isinstance(text, str):
            continue

        def fallback(i: int = index) -> str:
            return f"row:{i}"

        key = determinism.make_key(record, id_field, text_field, fallback)
        records.append((source_of(path, record, source_field), key, text))

    return determinism.sample_by_source(records, allocations, seed)


def load_models(document_filter: Any) -> None:
    """Run the setup a Curator filter expects before ``score_document``.

    ``ScoreFilter.setup`` calls ``load_model`` then ``load_tokenizer`` on any
    filter that defines them; construction only records the model name. Scoring
    without that step is not a degraded measurement, it is no measurement:
    ``token_count`` leaves its tokenizer at None and raises on every document.

    ``model_check_or_download`` is deliberately not called. It snapshots the
    entire repo, which for a tokenizer named after a 30B checkpoint means
    downloading the weights to count tokens. ``load_tokenizer`` reads the cache
    that ``curate/subset`` already populates.
    """
    for hook in ("load_model", "load_tokenizer"):
        method = getattr(document_filter, hook, None)
        if callable(method):
            method()


def score_sample(
    signals: list[signal_registry.Signal],
    sample: dict[str, list[tuple[str, str]]],
    extra_kwargs: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, profiling.SignalScores], dict[str, dict[str, Any]]]:
    """Apply each signal to every sampled document, counting what failed.

    A document that makes a scorer raise becomes NaN so one bad record cannot end
    a profile, but the failures are counted. A signal that fails on everything
    would otherwise report a clean-looking distribution over nothing.
    """
    scored: dict[str, profiling.SignalScores] = {}
    health: dict[str, dict[str, Any]] = {}
    extra_kwargs = extra_kwargs or {}

    for signal in signals:
        thresholds = _placeholder_thresholds(signal)
        try:
            document_filter = signal.build(*thresholds, **extra_kwargs.get(signal.name, {}))
            load_models(document_filter)
        except Exception as exc:  # noqa: BLE001 - surfaced as a finding, not a stack trace
            raise SignalUnavailableError(f"{signal.name} could not be constructed: {exc}") from exc

        failures = 0
        non_finite = 0
        by_source: dict[str, list[float]] = {}
        for source, items in sample.items():
            values = []
            for _key, text in items:
                try:
                    value = float(document_filter.score_document(text))
                except Exception:  # noqa: BLE001 - one bad document must not end the profile
                    values.append(float("nan"))
                    failures += 1
                    continue
                if not math.isfinite(value):
                    values.append(float("nan"))
                    failures += 1
                    non_finite += 1
                else:
                    values.append(value)
            by_source[source] = values

        scores = profiling.SignalScores(name=signal.name, by_source=by_source)
        flat = scores.flat()
        finite = sum(1 for value in flat if math.isfinite(value))
        scored[signal.name] = scores
        health[signal.name] = {
            "documents_attempted": len(flat),
            "documents_scored": finite,
            "scoring_failures": failures,
            "non_finite_scores": non_finite,
        }

    return scored, health


def language_composition(sample: dict[str, list[tuple[str, str]]], model_path: str | None) -> dict[str, Any] | None:
    """What languages the corpus is actually in, and how sure the model is.

    Every other figure here describes the *shape* of the text — character
    ratios, punctuation, script. None of them answer the question a person asks
    first, which is what language this corpus is in. script_ratio and
    latin_ratio come closest and cannot help: languages that share a script are
    indistinguishable to a script ratio.

    That gap is not academic. A config naming the wrong language in
    steps.filter.language_codes removes almost everything and reports success,
    because the row counts still reconcile. Measured here, the mistake is
    visible before the filter runs rather than after.

    Returns None when no model is configured. The absence is recorded rather
    than filled in, because guessing the language is the thing this step exists
    not to do.

    A model that IS configured but cannot be read is a different thing entirely,
    and refusing is the only honest answer. Returning None there would produce a
    report identical in every byte to one from a run that asked for no model, so
    a typo in the path would be indistinguishable from a deliberate choice --
    and the reader would conclude the corpus had no language measured rather
    than that theirs failed to run.
    """
    if not model_path:
        return None
    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"models.fasttext_langid points at {model_path}, which is not a file. Language "
            "composition cannot be measured without it. Fix the path, or remove the key to "
            "profile without a language breakdown -- but do not leave a wrong path, because "
            "the report it produces is indistinguishable from one nobody asked for."
        )

    import fasttext

    model = fasttext.load_model(model_path)
    labels: dict[str, int] = {}
    scores: dict[str, list[float]] = {}
    total = 0
    for rows in sample.values():
        for _key, text in rows:
            # FastText reads one line; a newline mid-document truncates it.
            flat = " ".join(text.split())
            if not flat:
                continue
            # A list, not a bare string. fasttext 0.9.3's single-string branch
            # calls np.array(probs, copy=False), which numpy>=2 raises on; the
            # batch branch does not. Curator's own FastTextLangId passes a list
            # for the same reason.
            label, score = model.predict([flat], k=1)
            code = label[0][0].removeprefix("__label__")
            labels[code] = labels.get(code, 0) + 1
            scores.setdefault(code, []).append(float(score[0][0]))
            total += 1

    if not total:
        return None

    rows = []
    for code, count in sorted(labels.items(), key=lambda kv: -kv[1]):
        confidences = sorted(scores[code])
        rows.append(
            {
                "language": code,
                "documents": count,
                "share": count / total,
                "confidence_p50": confidences[len(confidences) // 2],
                "confidence_p10": confidences[max(0, int(0.10 * len(confidences)) - 1)],
            }
        )

    all_scores = sorted(v for vs in scores.values() for v in vs)

    def q(p: float) -> float:
        return all_scores[max(0, min(len(all_scores) - 1, int(p * len(all_scores))))]

    return {
        "scored": total,
        "model": model_path,
        "languages": rows,
        # The confidence gate is a separate decision from the code gate, so its
        # distribution is reported separately. min_langid_score drops documents
        # the model was unsure about whatever language it guessed.
        "confidence": {"p1": q(0.01), "p5": q(0.05), "p25": q(0.25), "p50": q(0.50), "p75": q(0.75)},
        "note": (
            "Measured with the same FastText model the filter uses, on the same sample as the "
            "signals above. Descriptive: it says what the corpus is, not which languages to keep."
        ),
    }


def _placeholder_thresholds(signal: signal_registry.Signal) -> tuple[float, ...]:
    """Any valid construction: ``score_document`` does not depend on the threshold."""
    if isinstance(signal.grid, signal_registry.IntervalGrid):
        return (signal.grid.lo_grid.values()[0], signal.grid.hi_grid.values()[-1])
    return (signal.grid.values()[-1],)


def build_report(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Measure the corpus. Returns (report, candidate policies, sample manifest)."""
    started_at = manifest_mod.utc_now()
    paths = expand(cfg.get("input_glob"))
    if not paths:
        # ConfigError, not FileNotFoundError: this is a config the user can fix,
        # and main() turns that into exit 2 with a message. FileNotFoundError is
        # caught by no main() in the category, so the same mistake produced a
        # raw traceback here and a clean refusal in every sibling step.
        raise ConfigError(f"input_glob matched no files: {cfg.get('input_glob')!r}")

    source_field = cfg.get("source_field") or None
    text_field = cfg.get("text_field") or "text"
    id_field = cfg.get("id_field") or None
    seed = int(cfg.get("seed") or 0)
    max_total_docs = int(cfg.get("max_total_docs") or 0)

    populations, scan_stats = count_sources(paths, source_field, text_field, cfg.get("id_field"))
    if not populations:
        raise ConfigError(
            f"corpus contains no records with a string {text_field!r} field ({scan_stats['records']} record(s) read)"
        )

    notes: list[str] = []
    if not source_field:
        notes.append(SHARD_SOURCE_NOTE)
    elif scan_stats["with_source_field"] == 0:
        notes.append(SHARD_SOURCE_NOTE)
    elif scan_stats["with_source_field"] < sum(populations.values()):
        # Partial presence is worse than absence: the report would mix real
        # source names with shard paths and look like a corpus with more sources
        # than it has.
        notes.append(
            f"{sum(populations.values()) - scan_stats['with_source_field']} of "
            f"{sum(populations.values())} records carry no {source_field!r}; those are grouped "
            "under their shard path, so some 'sources' below are shards."
        )
    if scan_stats.get("unparsable_lines"):
        notes.append(
            f"{scan_stats['unparsable_lines']} line(s) across {scan_stats['damaged_shards']} shard(s) "
            "would not parse and are excluded from every figure. curate/audit reports the same "
            "damage as a finding; profile the corpus only after it is repaired."
        )
    if scan_stats["without_text"]:
        notes.append(
            f"{scan_stats['without_text']} record(s) had no string {text_field!r} field and are "
            "excluded from every figure, including document_count."
        )

    allocations = determinism.allocate(populations, max_total_docs)
    sample = draw_sample(
        paths,
        allocations,
        source_field=source_field,
        text_field=text_field,
        id_field=id_field,
        seed=seed,
    )

    # The pack decides which signals can run at all. There is deliberately no
    # default language: a wrong default silently produces wrong numbers for a
    # corpus, which is worse than refusing to start.
    language = cfg.get("language")
    if not isinstance(language, str) or not language:
        raise ConfigError("language must be a non-empty BCP-47 tag")
    pack = langpack.load(language, cfg.get("langpack_dir"))

    tokenizer = resolve_tokenizer(cfg)
    capabilities = set(pack.capabilities)
    if tokenizer:
        capabilities.add("tokenizer")

    extra_kwargs: dict[str, dict[str, Any]] = {name: {"pack": pack} for name in signal_registry.PACK_SIGNALS}
    if tokenizer:
        name, revision = tokenizer
        extra_kwargs["token_count"] = {
            "hf_model_name": name,
            "transformers_init_kwargs": {"revision": revision},
        }

    signals, warnings = signal_registry.resolve(cfg.get("signals") or None, capabilities)
    notes.extend(warnings)

    scored, health = score_sample(signals, sample, extra_kwargs)

    # Weights come from what was actually drawn, not what was requested. A source
    # that yielded fewer documents than its allocation would otherwise have each
    # of them standing for too few of its peers.
    drawn = {source: len(items) for source, items in sample.items()}
    weights = {
        source: (allocation.population / drawn[source]) if drawn.get(source) else 0.0
        for source, allocation in allocations.items()
    }
    per_signal: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for signal in signals:
        scores = scored[signal.name]
        flat = scores.flat()
        profiling.verify_direction(signal, flat, build_kwargs=extra_kwargs.get(signal.name))

        entry: dict[str, Any] = {
            **profiling.distribution(scores, weights),
            "direction": signal.direction,
            "units": signal.units,
            "impl_version": signal.impl_version,
            "health": health[signal.name],
        }
        if signal.notes:
            entry["notes"] = signal.notes

        attempted = health[signal.name]["documents_attempted"] or 1
        usable = health[signal.name]["documents_scored"] / attempted
        if usable < MIN_SCORED_FRACTION:
            # Retention over a handful of survivors is not retention for the
            # corpus, and a curve drawn from it reads as though it were.
            entry["retention"] = None
            entry["retention_suppressed"] = (
                f"only {health[signal.name]['documents_scored']} of {attempted} sampled documents "
                f"produced a usable score ({health[signal.name]['scoring_failures']} scorer "
                "failures); retention figures would not describe this corpus"
            )
            # Also at report level: a reader scanning `notes` should not have to
            # open every signal entry to learn one of them measured nothing.
            notes.append(f"{signal.name}: {entry['retention_suppressed']}")
            per_signal.append(entry)
            continue

        micro_weights = scores.micro_weights(weights)

        if isinstance(signal.grid, signal_registry.IntervalGrid):
            entry["retention"] = profiling.retention_surface(flat, signal.grid, micro_weights)
            entry["retention_view"] = profiling.MICRO
            candidates.append(
                {
                    "signal": signal.name,
                    "kind": "interval",
                    "surface_axes": {
                        "min": entry["retention"]["min_axis"],
                        "max": entry["retention"]["max_axis"],
                    },
                    "note": "two-sided gate; choose a (min, max) pair from the surface",
                }
            )
        else:
            curve = profiling.retention_curve(flat, signal.direction, signal.grid, micro_weights)
            entry["retention"] = {"kind": "curve", "points": curve}
            entry["retention_view"] = profiling.MICRO
            entry["retention_stable_bands"] = profiling.retention_stable_bands(
                curve,
                float((cfg.get("band_search") or {}).get("min_keep_rate", 0.80)),
                float((cfg.get("band_search") or {}).get("max_keep_rate", 0.995)),
            )
            candidates.append(
                {
                    "signal": signal.name,
                    # Without this a reader cannot pair the band's fields: for a
                    # `max` signal retention rises with the threshold, for a
                    # `min` signal it falls, and the file is the only thing they
                    # have when choosing a value to approve.
                    "direction": signal.direction,
                    "units": signal.units,
                    "bands": entry["retention_stable_bands"],
                    "note": (
                        "retention-stable range; not a recommendation. Each band gives the "
                        "retention AT each of its own thresholds — read "
                        "retained_at_threshold_low with threshold_low, never as a min/max pair."
                    ),
                }
            )

        if signal.curator_default:
            entry["curator_default"] = {
                "thresholds": list(signal.curator_default),
                "retained": _retention_at(flat, signal, signal.curator_default, micro_weights),
                "note": (
                    "what Curator's default bound would keep on this corpus. A "
                    "reference point, not a description of Curator's behaviour: its "
                    "non-English cascade omits the ASCII-assuming filters entirely."
                ),
            }

        per_signal.append(entry)

    # What the proposed policy costs as a whole. The thresholds are the ones the
    # approve block offers, so the figure answers the question a person asks
    # right before pasting it: applying all of these at once, what is left? Read
    # off the per-signal tables it cannot be answered at all -- those are each
    # gate alone, and gates overlap.
    proposed: dict[str, tuple[float, ...]] = {}
    for entry in per_signal:
        if entry.get("direction") not in ("min", "max"):
            continue
        table = profiling.gate_table(entry)
        if table:
            proposed[str(entry["signal"])] = (table[0][0],)
    # Same order the approve block prints, so the simulation's incremental column
    # and the block a reader is copying from line up. Passed explicitly rather
    # than left to the default: agreeing by coincidence is not agreeing.
    simulation = profiling.policy_simulation(
        {name: sc.flat() for name, sc in scored.items()}, signals, proposed, order=sorted(proposed)
    )
    if simulation is None:
        notes.append(
            "policy simulation not computed: fewer than two signals produced a gate "
            "table, so there is no policy to combine. This is 'the union was not "
            "measured', not 'the gates do not overlap'."
        )

    # Co-occurrence is only defined AT a threshold per signal, so it can only be
    # computed for signals that carry a reference operating point. Most do not:
    # a pack-parameterised signal has no Curator default to borrow.
    operating_points = {s.name: s.curator_default for s in signals if s.curator_default}
    masks = profiling.operating_point_masks(
        {name: sc.flat() for name, sc in scored.items()}, signals, operating_points
    )

    # An empty co-occurrence list reads exactly like "these gates do not overlap".
    # When it is empty because it could not be computed, say so — the whole point
    # of this step is that a degraded result must not look like a clean one.
    if len(masks) < 2:  # noqa: PLR2004 - overlap needs two sets
        without = sorted(s.name for s in signals if not s.curator_default)
        notes.append(
            f"cooccurrence not computed: it is only defined at a named operating point per "
            f"signal, and {len(masks)} of the {len(signals)} profiled signals carry one. "
            f"Signals without a reference threshold: {without}. This is 'overlap was not "
            f"measured', not 'the gates do not overlap'."
        )

    report = {
        "step_id": "curate/profile",
        "schema_version": 1,
        "signals_impl_version": signal_registry.IMPL_VERSION,
        "langpack": pack.describe(),
        "corpus": {
            "glob": cfg.get("input_glob"),
            "file_count": len(paths),
            "document_count": sum(populations.values()),
            "records_read": scan_stats["records"],
            "records_without_text": scan_stats["without_text"],
            "unparsable_lines": scan_stats.get("unparsable_lines", 0),
            "damaged_shards": scan_stats.get("damaged_shards", 0),
            "source_count": len(populations),
            "source_field": source_field,
        },
        "sampling": {
            "seed": seed,
            "max_total_docs": max_total_docs,
            "sampled": sum(len(v) for v in sample.values()),
            "method": "hash-bottom-k per source, proportional allocation",
        },
        # Recorded even when absent: a token_count figure is only meaningful
        # against a named revision, and its absence explains why the signal
        # is missing from the report.
        "tokenizer": ({"name": tokenizer[0], "revision": tokenizer[1]} if tokenizer else None),
        "views": {
            profiling.MACRO: "each source weighted equally",
            profiling.MICRO: "each sampled document weighted by the documents it stands for",
        },
        "notes": notes,
        "signals": per_signal,
        # What language the corpus is in, which no signal above can answer:
        # languages that share a script are indistinguishable to a script ratio.
        "language_composition": language_composition(sample, (cfg.get("models") or {}).get("fasttext_langid")),
        "cooccurrence": profiling.cooccurrence(masks),
        # The union of the proposed gates, which no per-signal table and no
        # pairwise overlap can give.
        "policy_simulation": simulation,
        "interpretation": (
            "Descriptive only. These figures say what a threshold removes, not whether what it "
            "removes is low quality. Selecting among candidates requires downstream validation."
        ),
    }

    # Before the digest, not after: the digest has to describe the document that
    # actually lands on disk, and a signal that scored nothing contributes NaN
    # quantiles that no strict JSON parser would read back.
    report = manifest_mod.json_safe(report)

    # profile_digest is taken over the measurements only. Timestamps live in
    # producer and are excluded on purpose: a policy carrying the digest of an
    # earlier profile should look stale when the measurements changed, not
    # every time the corpus is re-profiled with the same config.
    report["profile_digest"] = policy.digest(report)
    report["producer"] = {
        "step_id": "curate/profile",
        "tool_revision": manifest_mod.tool_revision(),
        "config_hash": manifest_mod.config_hash(cfg),
        "started_at": started_at,
        "completed_at": manifest_mod.utc_now(),
        "digest_covers": "every key except producer and profile_digest",
    }

    manifest = {
        "seed": seed,
        "max_total_docs": max_total_docs,
        "allocations": [
            {**a.as_dict(), "drawn": drawn.get(a.source, 0), "effective_weight": weights.get(a.source, 0.0)}
            for a in allocations.values()
        ],
        "sampled_keys": {
            source: [determinism.stable_uint64(key, seed) for key, _ in items]
            for source, items in sorted(sample.items())
        },
    }

    policies = policy.build_candidate_policies(
        candidates=candidates,
        profile_digest=report["profile_digest"],
        signals_impl_version=signal_registry.IMPL_VERSION,
        corpus={
            "glob": cfg.get("input_glob"),
            "document_count": sum(populations.values()),
            # Required by policy.validate_approved_policy: a policy that cannot
            # name the corpus its thresholds came from cannot be audited later.
            "fingerprint": scan_stats.get("fingerprint"),
        },
        langpack={
            "id": pack.pack_id,
            "language_tag": pack.language_tag,
            "version": pack.version,
            "content_hash": pack.content_hash,
        },
    )

    return report, policies, manifest


def _retention_at(
    scores: list[float],
    signal: signal_registry.Signal,
    thresholds: tuple[float, ...],
    weights: list[float],
) -> float:
    import numpy as np

    arr = np.asarray([s if s is not None else np.nan for s in scores], dtype=float)
    finite = np.isfinite(arr)
    w = np.asarray(weights, dtype=float)
    total = float(w[finite].sum()) or 1.0
    mask = profiling.keep_mask(arr, signal.direction, *thresholds) & finite
    return float(w[mask].sum() / total)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Profile the corpus, write the three artifacts, and return the report.

    The uniform entry point every curate step exposes. Does not raise
    ``SystemExit``; ``main`` translates the exceptions into exit codes so a
    caller running several steps keeps control of what a failure means.
    """
    output_dir = Path(cfg.get("output_dir") or ".")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_names = (
        "profile_report.json",
        "sample_manifest.json",
        "candidate_policies.yaml",
        "profile_summary.md",
    )
    for name in artifact_names:
        (output_dir / name).unlink(missing_ok=True)
        (output_dir / f".{name}.tmp").unlink(missing_ok=True)

    try:
        report, policies, manifest = build_report(cfg)

        # allow_nan=False asserts the sanitising pass in build_report covered
        # everything: a leak fails here rather than writing a file that a strict
        # parser cannot read.
        report_tmp = output_dir / ".profile_report.json.tmp"
        sample_tmp = output_dir / ".sample_manifest.json.tmp"
        summary_tmp = output_dir / ".profile_summary.md.tmp"
        report_tmp.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        sample_tmp.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        # The half a person reads. profile_report.json carries 64 retention
        # points and a 32-bin histogram per signal — the machine's copy, and
        # unreadable by eye.
        summary_tmp.write_text(profiling.summarise(report), encoding="utf-8")
        policy.write_candidate_policies(output_dir / "candidate_policies.yaml", policies)
        sample_tmp.replace(output_dir / "sample_manifest.json")
        summary_tmp.replace(output_dir / "profile_summary.md")
        # The report is the completion marker and is committed last.
        report_tmp.replace(output_dir / "profile_report.json")
    except BaseException:
        for name in artifact_names:
            (output_dir / name).unlink(missing_ok=True)
            (output_dir / f".{name}.tmp").unlink(missing_ok=True)
        raise

    for note in report["notes"]:
        logger.warning(note)
    for entry in report["signals"]:
        default = entry.get("curator_default")
        if default and default["retained"] < 0.5:
            logger.warning(
                "%s: Curator's default %s would retain %.1f%% of this corpus",
                entry["signal"],
                default["thresholds"],
                default["retained"] * 100,
            )

    report["artifacts"] = {
        "profile_report": str(output_dir / "profile_report.json"),
        "candidate_policies": str(output_dir / "candidate_policies.yaml"),
        "sample_manifest": str(output_dir / "sample_manifest.json"),
        "profile_summary": str(output_dir / "profile_summary.md"),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile what candidate thresholds do to a corpus")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    try:
        run(cfg)
    except (langpack.LanguagePackNotFoundError, langpack.LanguagePackInvalidError) as exc:
        print(f"curate/profile: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except (signal_registry.UnknownSignalError, signal_registry.SignalRequirementsUnmetError) as exc:
        print(f"curate/profile: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except ConfigError as exc:
        print(f"curate/profile: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except profiling.DirectionMismatchError as exc:
        print(f"curate/profile: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc

    output_dir = Path(cfg.get("output_dir") or ".")
    print(
        f"curate/profile: read profile_summary.md first. Wrote it plus "
        f"profile_report.json, candidate_policies.yaml, "
        f"sample_manifest.json to {output_dir}"
    )


if __name__ == "__main__":
    main()
