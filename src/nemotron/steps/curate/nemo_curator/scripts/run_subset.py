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

"""CLI for ``steps/curate/subset``.

Four phases, in this order for a reason:

``scan``
    Read every document once, count its tokens with the configured tokenizer,
    and record its stratum inputs. Token counts are cached keyed by
    ``(id, content digest, tokenizer, revision)`` because re-tokenizing a
    corpus to change a budget is the slowest possible way to answer a cheap
    question.

``plan``
    Compute every tier's per-stratum quota **before writing anything**, and
    write ``plan.json``. A plan that can be read and rejected before a corpus is
    materialized is worth more than one inferred from the output afterwards.

``materialize``
    Write one output directory per tier.

``verify``
    Check ``ids(N1) ⊆ ids(N2)`` for every pair on the ids actually written, and
    fail the run if it does not hold. Verifying the plan rather than the output
    would check the arithmetic and not the artifact.

Tokenization goes through Curator's ``TokenCountFilter``. It has no ``revision``
parameter, but forwards ``transformers_init_kwargs`` verbatim to
``AutoTokenizer.from_pretrained``, which is where the pin is applied. The
resolved name and revision go into every tier's manifest: two subsets counted
under different revisions are not comparable and must not be presented as an
ablation pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import yaml

from nemotron.steps.curate.nemo_curator.runtime import integrity, subset
from nemotron.steps.curate.nemo_curator.runtime import manifest as run_manifest

logger = logging.getLogger("curate.subset")

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "subset" / "config" / "default.yaml"

#: Cache schema. Bumped when a change would alter a cached count's meaning.
CACHE_VERSION = 2


class ConfigError(ValueError):
    """The run cannot start as configured."""


# -- input --------------------------------------------------------------------


def resolve_inputs(input_glob: str | list[str]) -> list[str]:
    """One resolver, shared with every other curate step.

    Previously a bare ``glob.glob``, which accepted only globs: a directory —
    the spelling ``curate/audit`` and ``curate/ingest`` both take — resolved to
    nothing and the run died claiming the corpus was empty.
    """
    resolved = integrity.expand_inputs(input_glob)
    if not resolved:
        raise ConfigError(f"input_glob matched no files: {input_glob!r}")
    return resolved


def iter_records(paths: list[str], damage: dict[str, int] | None = None) -> Iterator[tuple[str, dict[str, Any]]]:
    """Shared with every other step. See :func:`integrity.iter_records`."""
    return integrity.iter_records(paths, damage)


# -- tokenization -------------------------------------------------------------


class TokenCounter:
    """Curator's ``TokenCountFilter``, pinned to a revision and cached on disk.

    The cache is keyed by ``(tokenizer, revision, document id, content hash)``.
    The content hash matters because filtering or annotation can rewrite text
    without changing an id; reusing the old count would silently change tier
    membership.
    """

    def __init__(self, name: str, revision: str, cache_path: Path | None) -> None:
        self.name = name
        self.revision = revision
        self.cache_path = cache_path
        self._counts: dict[str, dict[str, Any]] = {}
        self._filter: Any = None
        self.hits = 0
        self.misses = 0

        if cache_path and cache_path.is_file():
            try:
                loaded = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("token cache at %s is unreadable and is ignored: %s", cache_path, exc)
                loaded = None
            if not isinstance(loaded, dict):
                if loaded is not None:
                    logger.warning("token cache at %s is not a JSON object and is ignored", cache_path)
            elif (
                loaded.get("cache_version") == CACHE_VERSION
                and loaded.get("tokenizer") == name
                and loaded.get("revision") == revision
            ):
                raw_counts = loaded.get("counts")
                if isinstance(raw_counts, dict):
                    self._counts = {
                        str(doc_id): dict(entry) for doc_id, entry in raw_counts.items() if isinstance(entry, dict)
                    }
            else:
                logger.warning(
                    "token cache at %s was written for a different tokenizer or schema and is ignored",
                    cache_path,
                )

    def _load(self) -> Any:
        if self._filter is None:
            from nemo_curator.stages.text.filters.token.token_count import TokenCountFilter

            # hf_model_name, not tokenizer: the latter takes a *loaded*
            # AutoTokenizer. Passing a model-name string there is accepted by the
            # constructor and then fails per document as
            # ``str.encode(encoding=<the document>)`` -> LookupError, and, worse,
            # leaves transformers_init_kwargs unread, so the revision pin — the
            # whole point of recording one — would be silently inert.
            self._filter = TokenCountFilter(
                hf_model_name=self.name,
                transformers_init_kwargs={"revision": self.revision},
            )
            if hasattr(self._filter, "load_tokenizer"):
                self._filter.load_tokenizer()
        return self._filter

    def count(self, doc_id: str, text: str) -> int:
        content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cached = self._counts.get(doc_id)
        if (
            isinstance(cached, dict)
            and cached.get("content_sha256") == content_sha256
            and isinstance(cached.get("tokens"), int)
            and not isinstance(cached.get("tokens"), bool)
        ):
            self.hits += 1
            return int(cached["tokens"])
        self.misses += 1
        value = int(self._load().score_document(text))
        self._counts[doc_id] = {
            "content_sha256": content_sha256,
            "tokens": value,
        }
        return value

    def save(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_name(f".{self.cache_path.name}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "cache_version": CACHE_VERSION,
                    "tokenizer": self.name,
                    "revision": self.revision,
                    "counts": self._counts,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.cache_path)


class WordCounter:
    """Whitespace words, for runs that cannot load a tokenizer.

    Available so the step is testable and so a smoke run does not need model
    weights. Every artifact records ``unit: words``, because a budget in words
    is not a budget in tokens and two subsets measured in different units are
    not an ablation pair.
    """

    name = "whitespace"
    revision = "n/a"
    hits = 0
    misses = 0

    def count(self, doc_id: str, text: str) -> int:  # noqa: ARG002
        return len(text.split())

    def save(self) -> None:
        return


def build_counter(cfg: dict) -> tuple[Any, str]:
    block = cfg.get("tokenizer")
    if block in (None, False):
        return WordCounter(), "words"
    if not isinstance(block, dict) or not isinstance(block.get("name"), str) or not block["name"].strip():
        raise ConfigError("tokenizer must be a mapping with a name, or null to count words")
    if not isinstance(block.get("revision"), str) or not block["revision"].strip():
        raise ConfigError(
            "tokenizer.revision is required. Token counts from two revisions are not "
            "comparable, and a subset that cannot name its revision cannot be reused as "
            "an ablation baseline. Set it to null to count whitespace words instead."
        )
    cache = cfg.get("token_cache")
    return (
        TokenCounter(block["name"], block["revision"], Path(cache) if cache else None),
        "tokens",
    )


# -- phases -------------------------------------------------------------------


def scan(cfg: dict, paths: list[str], counter: Any) -> tuple[list[subset.ScanRow], dict[str, Any]]:
    id_field = cfg.get("id_field")
    if not id_field:
        raise ConfigError(
            "id_field is required. A subset is a set of document ids; without one, nesting "
            "cannot be stated. Curator's AddId is positional and does not survive resharding."
        )
    text_field = cfg.get("text_field") or "text"
    source_field = cfg.get("source_field")
    score_field = cfg.get("quality_score_field")

    damage: dict[str, int] = {}
    rows: list[subset.ScanRow] = []
    skipped_no_text = 0
    skipped_no_id = 0

    for path, record in iter_records(paths, damage):
        text = record.get(text_field)
        if not isinstance(text, str) or not text:
            skipped_no_text += 1
            continue
        raw_id = record.get(id_field)
        if raw_id is None or str(raw_id).strip() == "":
            skipped_no_id += 1
            continue
        doc_id = str(raw_id)
        raw_score = record.get(score_field) if score_field else None
        try:
            score = float(raw_score) if raw_score is not None else None
        except (TypeError, ValueError) as exc:
            raise subset.SubsetError(
                f"document {doc_id!r} has non-numeric quality score {raw_score!r} in {score_field!r}"
            ) from exc
        tokens = counter.count(doc_id, text)
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise subset.SubsetError(
                f"document {doc_id!r} has non-positive token count {tokens!r}. "
                "A token budget requires every selected document to have a positive cost."
            )
        rows.append(
            subset.ScanRow(
                doc_id=doc_id,
                source=str(record.get(source_field) or path) if source_field else path,
                tokens=tokens,
                score=score,
            )
        )

    stats = {
        "files": len(paths),
        "documents": len(rows),
        "skipped_missing_text": skipped_no_text,
        "skipped_missing_id": skipped_no_id,
        "unparsable_lines": sum(damage.values()),
        "unparsable_by_shard": dict(sorted(damage.items())),
        "source_field_used": bool(source_field),
    }
    if not rows:
        raise ConfigError(
            f"no usable documents in {len(paths)} file(s): "
            f"{skipped_no_text} without {text_field!r}, {skipped_no_id} without {id_field!r}, "
            f"{sum(damage.values())} unparsable line(s)"
        )
    return rows, stats


def _tier_dir_names(cfg: dict) -> list[str]:
    """The tier directories this run will write, named from the config alone.

    The unit is decided by whether a tokenizer is configured, so the names are
    known before anything is read -- which is what lets the cleanup below be
    scoped to this run instead of to the glob.
    """
    unit = "words" if cfg.get("tokenizer") in (None, False) else "tokens"
    return [f"budget_{budget}_{unit}" for budget in cfg.get("token_budgets") or []]


def _reset_output_artifacts(output_dir: Path, tier_names: Sequence[str]) -> None:
    """Remove the artifacts THIS run owns, and only those.

    Globbing ``budget_*`` swept away tiers belonging to other runs: a directory
    holding budget_500000000_tokens from last week is not stale merely because
    today's config asks for budget_2000000_tokens. The names come from the
    config, so a tier this run will not write is a tier it must not delete.
    """
    for name in ("plan.json", "subset_report.json", ".plan.json.tmp", ".subset_report.json.tmp"):
        (output_dir / name).unlink(missing_ok=True)
    for name in tier_names:
        path = output_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build nested stratified token-budget subsets")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    started_at = run_manifest.utc_now()

    try:
        run(cfg, started_at)
    except ConfigError as exc:
        print(f"curate/subset: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except subset.NestingViolationError as exc:
        print(f"curate/subset: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc
    except subset.SubsetError as exc:
        print(f"curate/subset: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def run(cfg: dict, started_at: str | None = None) -> dict[str, Any]:
    """Run the subset lifecycle, removing all owned artifacts on failure.

    Nothing is deleted until the configuration has been validated. The previous
    order cleared ``budget_*`` before checking token_budgets, so a config error
    destroyed the previous run's tiers on the way to reporting itself — and with
    output_dir defaulting to the working directory, it did that wherever the
    caller happened to be standing.
    """
    started_at = started_at or run_manifest.utc_now()
    raw_output_dir = cfg.get("output_dir")
    if not raw_output_dir or not str(raw_output_dir).strip():
        raise ConfigError(
            "output_dir is required and has no default. This step deletes the budget_* "
            "directories it owns before writing, and a defaulted path decides where that "
            "happens on the caller's behalf."
        )
    prepared = _validate_config(cfg)

    # Stripped, because the emptiness check above already strips: validating one
    # string and then using another lets "  /data/out" through as a *relative*
    # path whose first component is two spaces, created under the caller's
    # working directory while /data/out is never touched.
    output_dir = Path(str(raw_output_dir).strip())
    output_dir.mkdir(parents=True, exist_ok=True)
    tier_names = _tier_dir_names(cfg)
    _reset_output_artifacts(output_dir, tier_names)
    try:
        return _run(cfg, started_at, output_dir, prepared)
    except BaseException:
        _reset_output_artifacts(output_dir, tier_names)
        raise


def _validate_config(cfg: dict) -> tuple[list[str], Any, str]:
    """Every check that can be made before anything is removed.

    Returns what the checks had to build anyway -- the resolved input paths and
    the counter -- so ``_run`` does not resolve them a second time.

    The narrow version of this function checked token_budgets and length_bands
    only, and everything else still raised from inside _run, after the delete. A
    mistyped input_glob destroyed the previous run's tiers on its way to
    reporting itself. Any check that needs no output directory belongs here.
    """
    budgets = cfg.get("token_budgets") or []
    if (
        not isinstance(budgets, list)
        or not budgets
        or any(isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0 for budget in budgets)
    ):
        raise ConfigError("token_budgets must be a non-empty list of positive integers")
    raw_length_bands = cfg.get("length_bands")
    length_bands = subset.DEFAULT_LENGTH_BANDS if raw_length_bands is None else raw_length_bands
    if (
        not isinstance(length_bands, (list, tuple))
        or not length_bands
        or any(isinstance(edge, bool) or not isinstance(edge, int) or edge <= 0 for edge in length_bands)
        or list(length_bands) != sorted(set(length_bands))
    ):
        raise ConfigError("length_bands must be a non-empty, strictly increasing list of positive integers")

    if not cfg.get("id_field"):
        raise ConfigError("id_field is required: a subset that cannot name its documents cannot be nested")
    try:
        int(cfg.get("seed") or 0)
    except (TypeError, ValueError):
        raise ConfigError(f"seed must be an integer, got {cfg.get('seed')!r}") from None
    if "input_glob" not in cfg:
        raise ConfigError("input_glob is required")
    # build_counter validates the whole tokenizer block; resolve_inputs refuses a
    # pattern matching nothing. Both read configuration, neither writes.
    counter, unit = build_counter(cfg)
    paths = resolve_inputs(cfg["input_glob"])
    return paths, counter, unit


def _run(cfg: dict, started_at: str, output_dir: Path, prepared: tuple[list[str], Any, str]) -> dict[str, Any]:
    budgets = cfg["token_budgets"]
    raw_length_bands = cfg.get("length_bands")
    length_bands = subset.DEFAULT_LENGTH_BANDS if raw_length_bands is None else raw_length_bands

    paths, counter, unit = prepared

    rows, scan_stats = scan(cfg, paths, counter)
    counter.save()

    plan = subset.build_plan(
        rows,
        budgets,
        seed=int(cfg.get("seed") or 0),
        score_field=cfg.get("quality_score_field"),
        length_bands=tuple(length_bands),
    )
    plan_doc = plan.to_dict()
    plan_doc["unit"] = unit
    plan_doc["tokenizer"] = {"name": counter.name, "revision": counter.revision}
    plan_tmp = output_dir / ".plan.json.tmp"
    plan_tmp.write_text(json.dumps(plan_doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    plan_tmp.replace(output_dir / "plan.json")
    for warning in plan.warnings:
        logger.warning(warning)

    results = subset.materialize(plan, rows)

    # Verify on the ids that were selected, before writing, then again on what
    # each tier actually contains. The first catches a planning defect, the
    # second catches a writing one, and they are different failures.
    problems = subset.verify_nesting(results)
    if problems:
        raise subset.NestingViolationError(
            "the tiers do not nest, so they cannot support the ablation they exist for: " + "; ".join(problems)
        )

    id_field = cfg["id_field"]
    text_field = cfg.get("text_field") or "text"
    tiers: list[dict[str, Any]] = []
    written_ids: dict[int, set[str]] = {}

    for budget in plan.budgets:
        result = results[budget]
        wanted = set(result.doc_ids)
        tier_dir = output_dir / f"budget_{budget}_{unit}"
        tier_dir.mkdir(parents=True, exist_ok=True)
        damage: dict[str, int] = {}
        # Collected from the write itself, not copied from the plan. The second
        # verification below exists to catch a WRITING defect; handing it
        # `wanted` made it re-check the arithmetic the first verification had
        # already checked, so a tier that silently failed to write a document
        # still reported nesting_verified: true.
        actually_written: set[str] = set()
        tier_tmp = tier_dir / ".subset.jsonl.tmp"
        with tier_tmp.open("w", encoding="utf-8") as handle:
            for _, record in iter_records(paths, damage):
                raw_id = record.get(id_field)
                if raw_id is None:
                    continue
                if str(raw_id) in wanted and isinstance(record.get(text_field), str):
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    actually_written.add(str(raw_id))
        tier_tmp.replace(tier_dir / "subset.jsonl")
        written = len(actually_written)
        written_ids[budget] = actually_written

        entry = result.to_dict()
        entry["unit"] = unit
        entry["documents_written"] = written
        entry["output"] = str(tier_dir)
        if written != len(wanted):
            raise subset.SubsetError(
                f"tier {budget}: planned {len(wanted)} document(s) but wrote {written}. "
                "The corpus changed between planning and materialization, or a selected id "
                "could not be addressed; publishing a partial tier would make the plan false."
            )
        tiers.append(entry)

    written_problems = subset.verify_nesting(
        {b: subset.TierResult(b, sorted(ids), 0, 0, {}, 0, []) for b, ids in written_ids.items()}
    )
    if written_problems:
        raise subset.NestingViolationError("written tiers do not nest: " + "; ".join(written_problems))

    report = {
        "schema_version": subset.SCHEMA_VERSION,
        "step_id": "curate/subset",
        "started_at": started_at,
        "completed_at": run_manifest.utc_now(),
        "config_hash": run_manifest.config_hash(cfg),
        "unit": unit,
        "tokenizer": {"name": counter.name, "revision": counter.revision},
        "token_cache": {"hits": counter.hits, "misses": counter.misses},
        "seed": plan.seed,
        "corpus": {"total_tokens": plan.total_tokens, **scan_stats},
        "stratification": {
            "score_field": plan.score_field,
            "length_bands": list(plan.length_bands),
            "strata": len(plan.strata),
        },
        "tiers": tiers,
        "nesting_verified": True,
        # Tiers this run neither wrote nor removed. Narrowing the cleanup to
        # this run's own tiers was right -- deleting another run's output is
        # not this step's business -- but it leaves a directory holding tiers
        # that do not nest with each other, described by a report whose
        # interpretation says they nest by construction. That claim is about
        # `tiers` above, so name what it does not cover.
        "foreign_tiers": sorted(
            entry.name
            for entry in output_dir.glob("budget_*")
            if entry.name not in {Path(t["output"]).name for t in tiers}
        ),
        "warnings": list(plan.warnings),
        "interpretation": (
            "Tiers nest by construction: each is a prefix of one fixed per-stratum ordering. "
            "Tokens delivered are at most the budget; any difference is reported as "
            "token_shortfall and is never made up from another stratum. The claim covers "
            "`tiers` only -- any directory listed under `foreign_tiers` was written by a "
            "different run and nests with nothing here."
        ),
    }
    report_tmp = output_dir / ".subset_report.json.tmp"
    report_tmp.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    report_tmp.replace(output_dir / "subset_report.json")

    print(f"curate/subset: wrote {len(tiers)} tier(s), plan.json and subset_report.json to {output_dir}")
    return report


if __name__ == "__main__":
    main()
