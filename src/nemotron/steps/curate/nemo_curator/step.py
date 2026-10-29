#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/curate/nemo_curator"
#
# [tool.runspec.run]
# launch = "python"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
# format = "yaml"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 0
# ///
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Lightweight JSONL curation via NeMo Curator."""

from __future__ import annotations

import argparse
import hashlib
import os
from ast import literal_eval
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from huggingface_hub import snapshot_download

# Ray otherwise replays the outer ``uv run`` command for every worker. Its
# packaged working directory excludes ``.venv``, so ``uv --no-sync`` creates an
# empty worker environment and then fails to import Ray itself.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter

from nemotron.steps.curate.nemo_curator.runtime import integrity as _integrity
from nemotron.steps.curate.nemo_curator.runtime import manifest as run_manifest
from nemotron.steps.curate.nemo_curator.runtime import registry as _registry
from nemotron.steps.curate.nemo_curator.runtime.language import validate_language_codes

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"


def keep_language(value: str, allowed: set[str]) -> bool:
    """Whether a FastText language label is one the config asked for.

    ``FastTextLangId.score_document`` returns ``str([score, lang_code])`` with the
    label exactly as the model emitted it — ``lid.176`` emits lowercase ISO codes.
    The comparison is case-folded because the config's codes are upper-cased on
    load, and an exact match would then reject every document while looking like
    a filter that simply found nothing.

    A label may also carry a script suffix (``zh_Hans``); matching the part before
    the underscore lets ``ZH`` select those without enumerating every variant.
    """
    score, lang_code = literal_eval(value)
    if score < 0.0:
        return False
    label = str(lang_code).casefold()
    wanted = {code.casefold() for code in allowed}
    return label in wanted or label.split("_", 1)[0] in wanted


def reader_fields(cfg: dict) -> list[str]:
    """Columns to project at the reader.

    ``JsonlReader`` treats ``fields`` as a projection, so anything left out here
    is gone for the rest of the pipeline. Listing ``metadata_fields`` is the only
    way a document's own ``id`` — the one identifier that survives resharding —
    reaches the output. Absent key reproduces the historical text-only read.
    """
    declared = list(cfg.get("metadata_fields") or [])

    # A field named for the manifest but absent from the projection is read away
    # at the first stage, so the manifest would come back with no per-source
    # counts and nothing would say why. Carry them rather than fail: the user
    # asked for the column by naming it.
    for key in ("id_field", "source_field"):
        value = cfg.get(key)
        if value and value not in declared:
            declared.append(value)

    fields = [cfg["text_field"], *declared]
    seen: set[str] = set()
    unique_fields = []
    for field in fields:
        if field not in seen:
            seen.add(field)
            unique_fields.append(field)
    return unique_fields


#: What ``JsonlReader`` can parse, and what needs ``curate/ingest`` first.
#:
#: Re-exported, not redefined. Preflight applies the same rule from the same
#: place (``run_flow`` imports ``integrity``, never this module -- importing it
#: would pull in Curator before preflight can answer "can this run start?").
#: Owning a private copy here is what let preflight count a parquet shard as
#: corpus while this step dropped or refused it: one resolver, two rules.
READER_EXTENSIONS = _integrity.JSONL_READER_EXTENSIONS
INGEST_ONLY_EXTENSIONS = _integrity.INGEST_ONLY_EXTENSIONS

# Pandas otherwise infers a column-wide numeric or datetime dtype. That changes
# identifiers such as "001" to 1 before the first pipeline stage sees them.
JSONL_READ_KWARGS = {"dtype": False, "convert_dates": False}


def resolve_inputs(input_glob: str | list[str]) -> list[str]:
    """Expand the corpus reference. One resolver for the whole category.

    This was a sixth, private copy of the resolver with its own rules: a
    narrower extension list (no ``ndjson``, no ``parquet``) and no sidecar
    denylist at all, plus a Curator ``get_all_file_paths_under`` branch that
    applied a third contract again whenever Curator happened to be importable.
    Preflight resolves with :func:`integrity.expand_inputs`, so this step
    validated one corpus and filtered another: point a run at a previous run's
    output directory -- the documented reuse pattern -- and preflight saw the
    two shards while this saw the shards *plus* ``run_manifest.json`` and
    ``curation_ledger.json``, then charged the pretty-printed manifest to the
    corpus as unparsable rows. The shipped fixture ``audit/data/tiny/`` is
    already enough to show the divergence.

    ``integrity.expand_inputs`` is now the single contract; its docstring states
    the inclusion rules. What this step can *read* is a separate question, and
    :data:`READER_EXTENSIONS` answers it at the point of use rather than by
    quietly dropping files during resolution.
    """
    return _integrity.expand_inputs(input_glob)


def readable_inputs(input_glob: str | list[str]) -> list[str]:
    """The resolved corpus, narrowed to what this step's reader can parse.

    One helper rather than three call sites, because they disagreed: ``run``
    narrowed while ``emit_manifest`` and ``emit_ledger`` re-resolved wide, so the
    same config produced two different corpora depending on which function asked,
    and the audit consumed the wider count.

    Non-corpus companions a broad glob sweeps up -- ``README.md``, ``_SUCCESS``
    -- are dropped here, which is what this step did before the resolver was
    shared. Corpus formats the reader cannot parse are NOT dropped; they are left
    in the caller's hands so it can refuse by name.
    """
    return [p for p in resolve_inputs(input_glob) if Path(p).suffix.casefold() in READER_EXTENSIONS]


#: Minimum FastText confidence for a language prediction to be trusted. Matches
#: Curator's own ``FastTextLangId`` default so the two agree; ``language_codes``
#: decides WHICH language, this decides how sure the model has to be.
DEFAULT_LANGID_SCORE = 0.3

#: Passed to MultilingualDomainClassifier instead of accepting its default.
#:
#: Curator's tokenizer truncates IN PLACE — ``df[text] = df[text].str.slice(0,
#: max_chars)`` at stages/text/models/tokenizer.py:160 — on the DataFrame that
#: continues to the writer. So the classifier's reading window silently becomes
#: the delivered corpus. Its default of 2000 rewrote 53.4% of one Vietnamese run,
#: destroying 29.7 million characters, while the ledger, manifest and audit all
#: reported success: every one of them counts rows, none reads content.
#:
#: None disables the slice entirely. The model still bounds what it reads, at the
#: token level, inside its own tokenizer — where truncation belongs, because it
#: does not touch the corpus.
DOMAIN_MAX_CHARS = None

MODES = ("filter", "annotate", "both")


@dataclass(frozen=True)
class PolicyResolution:
    thresholds: list[dict[str, Any]]
    langpack_spec: dict[str, Any]
    warnings: list[str]
    identity: dict[str, Any]
    #: True when allow_unvalidated_policy carried a policy past the approval
    #: contract it did not meet. The thresholds still ran, so the corpus was
    #: gated -- but recording that as "approved" launders the exact check that
    #: was skipped, and the manifest is what a reader has once the log is gone.
    overridden: bool = False


def _resolve_policy(cfg: dict[str, Any], input_files: list[str] | None = None) -> PolicyResolution:
    block = cfg.get("heuristic_filters") or {}
    path = block.get("approved_policy")
    if not path:
        return PolicyResolution([], {}, [], {})

    from nemotron.steps.curate.nemo_curator.runtime import policy as policy_module

    policy_bytes = Path(path).read_bytes()
    document = yaml.safe_load(policy_bytes) or {}
    try:
        warnings = list(
            policy_module.require_approved(document, allow_unvalidated=bool(block.get("allow_unvalidated_policy")))
        )
    except policy_module.PolicyNotApprovedError as exc:
        # require_approved reports which fields are unmet but never sees a path.
        # A run that names several policies needs to know which one was refused.
        raise policy_module.PolicyNotApprovedError(f"{path}: {exc}") from exc
    warnings = [f"{path}: {w}" for w in warnings]
    # The flag alone is not an override. A policy that meets the contract is
    # approved whether or not someone left the escape hatch open.
    overridden = bool(block.get("allow_unvalidated_policy")) and bool(policy_module.validate_approved_policy(document))

    # The scorers themselves are versioned. Thresholds were calibrated by one
    # implementation; running them under another silently measures a different
    # quantity, and the profile records the version precisely so this can be
    # checked rather than assumed.
    from nemotron.steps.curate.nemo_curator.runtime import registry as signal_registry

    declared_impl = document.get("signals_impl_version")
    if declared_impl != signal_registry.IMPL_VERSION:
        raise ValueError(
            f"{path}: thresholds were calibrated by signals implementation {declared_impl}, "
            f"but this run has {signal_registry.IMPL_VERSION}. A scorer change moves the "
            "numbers the thresholds refer to; re-profile against the current implementation."
        )

    from nemotron.steps.curate.nemo_curator.runtime import integrity

    declared_fingerprint = (document.get("corpus") or {}).get("fingerprint")
    actual_fingerprint = declared_fingerprint
    if input_files is not None or "input_glob" in cfg:
        # readable_inputs: the policy fingerprint must cover the files the
        # reader actually reads, not the companions a glob swept up.
        resolved_inputs = input_files if input_files is not None else readable_inputs(cfg["input_glob"])
        actual_fingerprint = integrity.corpus_fingerprint(
            resolved_inputs,
            cfg["text_field"],
            cfg.get("id_field"),
        )
        if declared_fingerprint != actual_fingerprint:
            raise ValueError(
                f"{path}: policy was approved for corpus {declared_fingerprint}, but the "
                f"configured input fingerprints to {actual_fingerprint}. Re-profile and "
                "approve against this input instead of applying thresholds calibrated on "
                "different data."
            )

    declared_pack = dict(document.get("langpack") or {})
    declared = declared_pack.get("content_hash")
    expected = block.get("langpack_content_hash")
    if declared and expected and declared != expected:
        raise ValueError(
            f"{path}: policy was derived from language pack {declared}, but this run loads "
            f"{expected}. Thresholds do not transfer between packs; re-profile or pin the pack."
        )

    # Config supplies what only the run knows: where packs live, and which
    # tokenizer to count with. Both travel with the pack spec so policy_stages
    # has one place to look.
    for key in ("langpack_dir", "tokenizer"):
        if block.get(key) is not None:
            declared_pack[key] = block[key]

    thresholds = [dict(entry) for entry in document.get("thresholds") or []]
    approval_block = document.get("approval")
    identity = {
        "policy_digest": f"sha256:{hashlib.sha256(policy_bytes).hexdigest()}",
        "profile_digest": document.get("profile_digest"),
        "signals_impl_version": declared_impl,
        "corpus_fingerprint": actual_fingerprint,
        # Carried so the manifest can say whether anyone claimed to approve this,
        # separately from whether it met the schema. `approval` is optional in the
        # contract -- deliberately, since a name in YAML proves nothing a machine
        # can check -- so "approved" alone over-claims when the block is absent.
        "approval_method": (approval_block or {}).get("method") if isinstance(approval_block, dict) else None,
        "thresholds": thresholds,
        **({"langpack_content_hash": declared_pack["content_hash"]} if declared_pack.get("content_hash") else {}),
    }
    return PolicyResolution(thresholds, declared_pack, warnings, identity, overridden=overridden)


def resolve_policy(cfg: dict) -> tuple[list[dict], dict, list[str]]:
    """Load and validate the approved policy against the configured input."""
    resolved = _resolve_policy(cfg)
    return resolved.thresholds, resolved.langpack_spec, resolved.warnings


def manifest_config(cfg: dict[str, Any], policy_resolution: PolicyResolution | None = None) -> dict[str, Any]:
    """Resolved run identity used by the manifest's configuration hash."""
    document = deepcopy(cfg)
    if policy_resolution is not None and policy_resolution.identity:
        document["resolved_policy"] = deepcopy(policy_resolution.identity)
    return document


#: Requirements a signal may declare that are satisfied by loading a language
#: pack. Every one of these is also a pack *capability* name, which is why the
#: pack itself is the thing that supplies them.
PACK_REQUIREMENTS: frozenset[str] = frozenset(
    {
        "boilerplate_hits",
        "diacritic_ratio",
        "script_ratio",
        "sentence_end_ratio",
        "stopword_ratio",
        "stopword_ratio_folded",
    }
)

#: Capabilities that assert something about the WRITING SYSTEM rather than name
#: data in the pack. Deliberately NOT in PACK_REQUIREMENTS: that set drives pack
#: loading and the ``content_hash`` check, and a hash exists because thresholds
#: are calibrated against a pack's word lists and character set. A ``word_count``
#: bound is calibrated against no such thing, so demanding a hash for it would
#: refuse a policy for a reason that does not apply to it.
#:
#: They are enforced where they actually bite: ``registry.resolve`` refuses them
#: at profile time, which always has a pack, and ``quality_filters`` -- the other
#: door, which bypasses the registry entirely -- is checked against the pack by
#: :func:`_refuse_word_filter_without_segmentation` and by the flow's preflight.
ORTHOGRAPHY_REQUIREMENTS: frozenset[str] = frozenset(
    {"word_segmentation", "ascii_digits", "ascii_punctuation", "ascii_alphabet"}
)

#: Everything this step knows how to supply. A signal declaring anything else is
#: refused by name rather than reaching ``Signal.build`` and failing on a missing
#: keyword argument — the failure mode that let ``token_count`` through.
KNOWN_REQUIREMENTS: frozenset[str] = PACK_REQUIREMENTS | ORTHOGRAPHY_REQUIREMENTS | {"tokenizer"}


def _refuse_word_filter_without_segmentation(cfg: dict) -> None:
    """Refuse ``min_words``/``max_words`` for a language that does not use spaces.

    Refuse rather than silently disable. A filter quietly dropped is a corpus the
    user did not ask for and cannot reconstruct -- the same reasoning that made
    ``curator_default`` fail loudly instead of substituting 0.25. The message
    names the pack, the capability and the two ways forward.

    Only fires when the run declares a pack. Without one there is no language to
    be wrong about, and a config that predates this keeps its behaviour.
    """
    from nemotron.steps.curate.nemo_curator.runtime import langpack as langpack_module

    block = cfg.get("heuristic_filters") or {}
    tag = block.get("language_tag") or block.get("language")
    root = block.get("langpack_dir")
    if not tag or not root:
        return
    try:
        pack = langpack_module.load(tag, root)
    except (langpack_module.LanguagePackNotFoundError, langpack_module.LanguagePackInvalidError):
        # Resolving the pack is not this check's job; whoever needs it will fail
        # with a better message than one about word counts.
        return
    if pack.supports("word_segmentation"):
        return
    raise ValueError(
        f"quality_filters sets min_words/max_words, but the {pack.language_tag!r} pack does not "
        "declare word_segmentation: WordCountFilter splits on whitespace this script does not use, "
        "so it would measure one word per document. Remove min_words and max_words, or declare "
        "word_segmentation in the pack if the language really is space-delimited."
    )


def load_policy_pack(langpack_spec: dict, needed_by: list[str], *, require_hash: bool = True) -> Any:
    """Load the language pack a policy was derived from.

    Half the registry's signals are parameterised by a pack — their word lists,
    character set and patterns arrive as data — so a policy naming one cannot be
    executed without it. ``curate/profile`` emits candidates for exactly these
    signals, so a policy that could be produced but not run would be a dead end
    between the two steps.

    The declared hash is checked against the pack actually loaded, not merely
    against another string in the config. Thresholds are calibrated against a
    pack's contents; running them against a different revision of that pack
    silently measures something else.
    """
    from nemotron.steps.curate.nemo_curator.runtime import langpack

    tag = langpack_spec.get("language_tag") or langpack_spec.get("id")
    if not isinstance(tag, str) or not tag:
        raise ValueError(
            f"policy uses language-pack signals {sorted(needed_by)} but declares no "
            "langpack.language_tag. There is no default language: a wrong one produces "
            "plausible numbers for the wrong corpus."
        )

    pack = langpack.load(tag, langpack_spec.get("langpack_dir"))
    if not require_hash:
        # The pack was loaded to answer "does this language work that way?", not
        # to supply word lists. A word_count bound is calibrated against no pack
        # content, so demanding the hash would refuse a policy for a reason that
        # does not apply to it.
        return pack
    declared = langpack_spec.get("content_hash")
    if not declared:
        # Required, not merely compared when present. A policy that does not say
        # which pack revision calibrated its thresholds cannot be checked at all,
        # and an unenforced guarantee reads exactly like an enforced one.
        raise ValueError(
            f"policy uses language-pack signals {sorted(needed_by)} but declares no "
            "langpack.content_hash. Thresholds are calibrated against a pack's word lists "
            "and character set; without the hash there is nothing to verify them against. "
            "Re-run curate/profile, which records it."
        )
    if pack.content_hash != declared:
        raise ValueError(
            f"policy was derived from language pack {tag} at {declared}, but the pack on this "
            f"machine hashes to {pack.content_hash}. Thresholds are calibrated against a pack's "
            "word lists and character set and do not transfer across revisions; re-profile "
            "against the current pack or pin the pack version."
        )
    return pack


#: Re-exported from the registry, not redefined. The evaluator applies the
#: same policy bounds without importing this module, which pulls in Curator at
#: module scope -- and two mappings of min/max onto threshold parameters is two
#: chances to invert a gate.
BOUND_KEYS = _registry.BOUND_KEYS
threshold_bounds = _registry.threshold_bounds


def policy_stages(thresholds: list[dict], text_field: str, mode: str, langpack_spec: dict | None = None) -> list[Any]:
    """Turn approved thresholds into Curator stages.

    Signal names resolve through the closed allowlist, never an import path from
    config: a policy file is a document people paste between machines.
    """
    if not thresholds:
        # No policy configured. Import nothing, add nothing: a run that predates
        # F2 must build exactly the pipeline it built before.
        return []

    from nemotron.steps.curate.nemo_curator.runtime import registry as signal_registry

    named: list[str] = []
    for index, entry in enumerate(thresholds):
        name = entry.get("signal")
        if not isinstance(name, str) or not name:
            raise ValueError(f"threshold {index} must name a non-empty signal")
        named.append(name)

    unknown = [name for name in named if name not in signal_registry.SIGNALS]
    if unknown:
        raise ValueError(
            f"unknown signal {unknown[0]!r} in policy. Allowed: {sorted(signal_registry.SIGNALS)}. "
            "Signals are a closed allowlist; a policy cannot name an import path."
        )

    # Driven by each signal's declared `requires`, not by a hand-maintained set of
    # pack-backed names: a set has to be remembered, and the one thing already
    # forgotten was token_count, which needs a tokenizer and is proposed by
    # curate/profile whenever models.tokenizer is set.
    spec = dict(langpack_spec or {})
    requirements = {req for n in named for req in signal_registry.SIGNALS[n].requires}
    unsupported = requirements - KNOWN_REQUIREMENTS
    if unsupported:
        raise ValueError(
            f"policy names signal(s) requiring {sorted(unsupported)}, which this step cannot "
            f"supply. Known requirements: {sorted(KNOWN_REQUIREMENTS)}. Either the registry "
            "gained a requirement the filter step was not taught to satisfy, or the policy "
            "names a signal that only curate/profile can run."
        )

    # Load the pack once, and only if something actually needs it, so a policy of
    # pack-free signals still runs on a machine with no packs installed.
    pack_backed = sorted({n for n in named if signal_registry.SIGNALS[n].requires})
    # Every requirement a pack answers, not only the data-backed ones. Loading
    # only for PACK_REQUIREMENTS left the orthographic capabilities unchecked on
    # this path: registry.resolve refuses `numbers_ratio` for a Hindi pack at
    # profile time, and policy_stages built it anyway, because ascii_digits is
    # not a data requirement and so never loaded a pack to ask.
    pack_capabilities = requirements & (PACK_REQUIREMENTS | ORTHOGRAPHY_REQUIREMENTS)
    pack = (
        load_policy_pack(spec, pack_backed, require_hash=bool(requirements & PACK_REQUIREMENTS))
        if pack_capabilities
        else None
    )
    if pack is not None:
        unmet = sorted(
            f"{name} needs {req}"
            for name in named
            for req in signal_registry.SIGNALS[name].requires
            if req in pack_capabilities and not pack.supports(req)
        )
        if unmet:
            raise ValueError(
                f"policy names signal(s) the {pack.language_tag!r} pack does not support: "
                f"{unmet}. curate/profile refuses these for this language; applying them here "
                "would measure something the language does not do -- a word count on a script "
                "without spaces, or an ASCII digit ratio on one that writes numbers otherwise. "
                "Remove the threshold, or declare the capability in the pack if it really applies."
            )

    tokenizer = spec.get("tokenizer")
    if "tokenizer" in requirements and not tokenizer:
        needs = sorted(n for n in named if "tokenizer" in signal_registry.SIGNALS[n].requires)
        raise ValueError(
            f"policy names {needs}, which counts tokens and therefore needs a tokenizer, but "
            "heuristic_filters.tokenizer is not set. A token budget counted by a different "
            "tokenizer is a different budget, so there is no safe default."
        )

    _, ScoreFilter = text_filter_stages()  # noqa: N806 -- these are Curator stage classes, not variables
    Score = score_stage() if mode == "annotate" else None  # noqa: N806 -- these are Curator stage classes, not variables

    stages = []
    for entry in thresholds:
        name = entry["signal"]
        signal = signal_registry.SIGNALS[name]
        bounds = threshold_bounds(signal, entry)
        extra: dict[str, Any] = {}
        if PACK_REQUIREMENTS & set(signal.requires):
            extra["pack"] = pack
        if "tokenizer" in signal.requires:
            extra["hf_model_name"] = tokenizer
        document_filter = signal.build(*bounds, **extra)

        if mode == "annotate":
            # Score never filters, so every row survives carrying its score.
            assert Score is not None
            stages.append(Score(document_filter, score_field=f"__{name}", text_field=text_field))
        elif mode == "both":
            # ScoreFilter writes score_field before applying keep_document, so one
            # stage already does score-then-filter. Score + Filter would be a
            # two-pass spelling of the same result.
            stages.append(ScoreFilter(document_filter, text_field=text_field, score_field=f"__{name}"))
        else:
            # score_field defaults to None, which discards the score after use —
            # the same shape as the WordCountFilter gate above it.
            stages.append(ScoreFilter(document_filter, text_field=text_field))
    return stages


def visible_gpu_count() -> int:
    """How many GPUs this process was actually given.

    ``CUDA_VISIBLE_DEVICES`` is what the scheduler hands a job, so it is the
    honest answer — not what the machine physically holds, which a shared node
    would over-report.
    """
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices is None:
        return 0
    return len([d for d in devices.split(",") if d.strip() not in ("", "-1")])


def ray_client_kwargs(cfg: dict) -> dict:
    kwargs = dict(cfg.get("ray") or {})
    if "num_cpus" not in kwargs and os.environ.get("NEMOTRON_CURATOR_RAY_NUM_CPUS"):
        kwargs["num_cpus"] = int(os.environ["NEMOTRON_CURATOR_RAY_NUM_CPUS"])
    if "num_gpus" not in kwargs:
        # RayClient defaults num_gpus to None, which starts Ray declaring ZERO
        # GPUs however many the job was given. A pipeline holding a classifier
        # then dies with "requires 1.0 but only 0 are available" — after the
        # reader has already run. Declaring what the scheduler allocated is the
        # correct default; ray.num_gpus in config still overrides it.
        override = os.environ.get("NEMOTRON_CURATOR_RAY_NUM_GPUS")
        count = int(override) if override else visible_gpu_count()
        if count:
            kwargs["num_gpus"] = count
    return kwargs


def text_filter_stages() -> tuple[type[Any], type[Any]]:
    """Return Filter/ScoreFilter across supported NeMo Curator releases."""
    try:
        from nemo_curator.stages.text.modules import Filter, ScoreFilter
    except ImportError:
        from nemo_curator.stages.text.filters import Filter, ScoreFilter
    return Filter, ScoreFilter


def score_stage() -> type[Any]:
    """Return Score across supported NeMo Curator releases."""
    try:
        from nemo_curator.stages.text.modules import Score
    except ImportError:
        from nemo_curator.stages.text.filters import Score
    return cast(type[Any], Score)


def build_pipeline(
    cfg: dict,
    *,
    input_files: list[str] | None = None,
    policy_resolution: PolicyResolution | None = None,
) -> tuple[Any, list[str]]:
    """Construct the Curator pipeline for this config, without running it.

    Split out from :func:`run` so the pipeline's *shape* can be inspected and
    tested without a Ray cluster, and so a caller orchestrating several steps can
    fail on a bad config before any of them starts work.
    """
    mode = cfg.get("mode", "filter")
    allowed_languages = {code.upper() for code in cfg.get("language_codes") or []}
    models = cfg.get("models") or {}
    quality_filters = cfg.get("quality_filters") or {}

    pipeline = Pipeline(name="curate_nemo_curator")
    pipeline.add_stage(
        JsonlReader(
            file_paths=input_files if input_files is not None else cfg["input_glob"],
            fields=reader_fields(cfg),
            read_kwargs=dict(JSONL_READ_KWARGS),
        )
    )
    if allowed_languages:
        Filter, ScoreFilter = text_filter_stages()  # noqa: N806 -- these are Curator stage classes, not variables
        from nemo_curator.stages.text.filters.fasttext import FastTextLangId

        pipeline.add_stage(
            ScoreFilter(
                FastTextLangId(
                    model_path=models["fasttext_langid"],
                    # Curator's own default, not 0.0. A fallback of 0.0 disables
                    # the confidence gate entirely while the config says nothing,
                    # so a run that named language_codes would keep every document
                    # the model was unsure about — and the shipped default.yaml
                    # said 0.3, so the flow and the standalone step disagreed.
                    min_langid_score=quality_filters.get("min_langid_score", DEFAULT_LANGID_SCORE),
                ),
                text_field=cfg["text_field"],
                score_field="language",
            )
        )

        def language_code(value: str) -> bool:
            return keep_language(value, allowed_languages)

        # Curator's Filter hardcodes its stage name to "filter_fn"
        # (score_filter.py:24, and __post_init__ overwrites whatever the
        # constructor is given). The ledger's per-gate attribution is keyed on
        # stage names, so that produced an entry reading "filter_fn: 5134" — a
        # count with no way to tell what removed those documents. The two
        # language stages are the confidence gate and the code gate; the ledger
        # has to say which is which, so the name is set after construction.
        language_stage = Filter(filter_fn=language_code, filter_field="language")
        language_stage.name = "language_code"
        pipeline.add_stage(language_stage)

    has_word_filter = any(key in quality_filters for key in ("min_words", "max_words"))
    if has_word_filter:
        if not all(key in quality_filters for key in ("min_words", "max_words")):
            raise ValueError("quality_filters must set both min_words and max_words to enable WordCountFilter")
        # quality_filters does not go through the signal registry, so the
        # capability gate that stops `word_count` running on an unsegmented
        # script does not reach WordCountFilter. Same measurement, same hazard,
        # different door. Checked only when this run has a pack to ask: a config
        # with no language declared keeps its historical behaviour exactly.
        _refuse_word_filter_without_segmentation(cfg)
        _, ScoreFilter = text_filter_stages()  # noqa: N806 -- these are Curator stage classes, not variables
        from nemo_curator.stages.text.filters.heuristic import WordCountFilter

        pipeline.add_stage(
            ScoreFilter(
                WordCountFilter(
                    min_words=quality_filters["min_words"],
                    max_words=quality_filters["max_words"],
                ),
                text_field=cfg["text_field"],
            )
        )
    # Two ways to reach the classifier, because "which domains do I keep" is a
    # decision that needs evidence, and the evidence is the labels themselves.
    # Filtering on a first run drops most of a corpus with nothing to justify it;
    # annotating writes the label and class-probability vector and drops nothing,
    # so the next run can gate on a distribution somebody looked at.
    if cfg.get("domains") or cfg.get("annotate_domains"):
        from nemo_curator.stages.text.classifiers import MultilingualDomainClassifier

        domain_kwargs: dict[str, Any] = {
            "text_field": cfg["text_field"],
            # None means label everything and drop nothing.
            "filter_by": cfg.get("domains") or None,
            "cache_dir": models.get("hf_cache_dir"),
            # Never the default. See DOMAIN_MAX_CHARS.
            "max_chars": cfg.get("domain_max_chars", DOMAIN_MAX_CHARS),
        }
        # Curator writes the complete class-probability vector into this field,
        # in model label order. Passing it unconditionally, even as None, changes
        # the constructed call for every config that already used `domains`.
        if cfg.get("domain_score_field"):
            domain_kwargs["score_field"] = cfg["domain_score_field"]

        pipeline.add_stage(MultilingualDomainClassifier(**domain_kwargs))
    # F2: approved-policy filtering. Absent config adds nothing, so a run that
    # predates it builds exactly the pipeline it built before.
    resolved = policy_resolution or _resolve_policy(cfg, input_files)
    thresholds = resolved.thresholds
    langpack_spec = resolved.langpack_spec
    policy_warnings = resolved.warnings
    for stage in policy_stages(thresholds, cfg["text_field"], mode, langpack_spec):
        pipeline.add_stage(stage)

    pipeline.add_stage(JsonlWriter(path=cfg["output_dir"]))
    return pipeline, policy_warnings


def _reset_artifact_paths(*paths: str | None) -> None:
    destinations = [Path(raw_path) for raw_path in paths if raw_path]
    resolved = [path.resolve() for path in destinations]
    if len(resolved) != len(set(resolved)):
        raise ValueError("emit_manifest and emit_ledger must use different paths")

    for path in destinations:
        if path.exists() and not path.is_file():
            raise ValueError(f"artifact path is not a file: {path}")
        path.unlink(missing_ok=True)


def refuse_empty_language_output(cfg: dict, input_files: list[str]) -> None:
    """Make a total language-gate drop a failed run, never a silent success."""
    if not cfg.get("language_codes"):
        return
    input_rows = run_manifest.count_jsonl(input_files)["row_count"]
    output_files = sorted(str(path) for path in Path(cfg["output_dir"]).rglob("*.jsonl"))
    output_rows = run_manifest.count_jsonl(output_files)["row_count"]
    if input_rows > 0 and output_rows == 0:
        raise ValueError(
            f"the language-filtered run kept 0 of {input_rows} documents. Check language_codes, "
            "the FastText model, and min_langid_score; set language_codes to [] to disable the gate"
        )


def run(cfg: dict) -> dict[str, Any]:
    """Curate the corpus and return a report describing what the run did."""
    mode = cfg.get("mode", "filter")
    manifest_path = cfg.get("emit_manifest")
    ledger_path = cfg.get("emit_ledger")
    started_at = run_manifest.utc_now()
    input_files: list[str] | None = None
    policy_resolution: PolicyResolution | None = None

    _reset_artifact_paths(manifest_path, ledger_path)
    try:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

        if cfg.get("dataset"):
            snapshot_download(**cfg["dataset"])

        resolved = resolve_inputs(cfg["input_glob"])
        if not resolved:
            raise ValueError(f"input_glob {cfg['input_glob']!r} matched no corpus files")

        # NARROW BEFORE REFUSING. The failure handler below writes a manifest and
        # a ledger from `input_files`, so a raise taken while it still held the
        # wide list charged the corpus with unparsable rows from a file the step
        # never read -- the exact damage this narrowing exists to prevent.
        input_files = [p for p in resolved if Path(p).suffix.casefold() in READER_EXTENSIONS]

        # A corpus format the reader cannot parse is a user error worth stopping
        # for -- but only when it is the WHOLE corpus. A stray parquet beside the
        # JSONL shards is the shape a Hugging Face snapshot has, and refusing it
        # aborted runs that worked before the resolver was shared.
        if not input_files:
            wrong_format = sorted(p for p in resolved if Path(p).suffix.casefold() in INGEST_ONLY_EXTENSIONS)
            if wrong_format:
                raise ValueError(
                    f"input_glob {cfg['input_glob']!r} resolved {len(wrong_format)} file(s) this step "
                    f"cannot read: {', '.join(wrong_format[:5])}. curate/nemo_curator reads "
                    f"{' or '.join(READER_EXTENSIONS)}. Run curate/ingest first to normalise the "
                    "corpus to JSONL, or narrow input_glob."
                )
            raise ValueError(
                f"input_glob {cfg['input_glob']!r} matched {len(resolved)} file(s) but none this "
                f"step can read. curate/nemo_curator reads {' or '.join(READER_EXTENSIONS)}."
            )

        # Against the model, before Ray starts. Comparing two user-supplied
        # strings cannot catch a label the model never emits.
        validate_language_codes(cfg)

        # Curator's writer names shards by content hash, so a second run into the
        # same directory adds to it instead of replacing the previous corpus.
        existing = sorted(Path(cfg["output_dir"]).rglob("*.jsonl")) if cfg.get("output_dir") else []
        if existing:
            raise ValueError(
                f"{cfg['output_dir']} already holds {len(existing)} .jsonl file(s) from a "
                f"previous run, e.g. {existing[0].name}. Remove the directory, or point "
                "output_dir somewhere new."
            )

        policy_resolution = _resolve_policy(cfg, input_files)
        pipeline, policy_warnings = build_pipeline(
            cfg,
            input_files=input_files,
            policy_resolution=policy_resolution,
        )
        for warning in policy_warnings:
            print(f"curate/nemo_curator: WARNING {warning}")

        ray_client = RayClient(**ray_client_kwargs(cfg))
        ray_started = False
        try:
            ray_client.start()
            ray_started = True
            tasks = pipeline.run()
        finally:
            if ray_started:
                ray_client.stop()

        refuse_empty_language_output(cfg, input_files)

        stage_counts = collect_stage_counts(tasks or [])
        stage_names = [getattr(stage, "name", "") for stage in getattr(pipeline, "stages", [])]
        identity_config = manifest_config(cfg, policy_resolution)

        if manifest_path:
            emit_manifest(
                cfg,
                manifest_path,
                started_at,
                completed=True,
                stage_names=stage_names,
                stage_counts=stage_counts,
                input_files=input_files,
                identity_config=identity_config,
                policy_resolution=policy_resolution,
            )
        if ledger_path:
            emit_ledger(
                cfg,
                ledger_path,
                completed=True,
                stage_names=stage_names,
                stage_counts=stage_counts,
                input_files=input_files,
            )

        return {
            "step_id": "curate/nemo_curator",
            "started_at": started_at,
            "completed_at": run_manifest.utc_now(),
            "mode": mode,
            "output_dir": cfg["output_dir"],
            "warnings": list(policy_warnings),
            "artifacts": {
                key: value
                for key, value in (
                    ("run_manifest", manifest_path),
                    ("curation_ledger", ledger_path),
                )
                if value
            },
        }
    except BaseException:
        identity_config = manifest_config(cfg, policy_resolution)
        if manifest_path:
            try:
                emit_manifest(
                    cfg,
                    manifest_path,
                    started_at,
                    completed=False,
                    input_files=input_files,
                    identity_config=identity_config,
                    policy_resolution=policy_resolution,
                )
            except Exception as exc:  # noqa: BLE001 - preserve the original pipeline failure
                print(f"curate/nemo_curator: WARNING could not write failed manifest: {exc}")
        if ledger_path:
            try:
                emit_ledger(
                    cfg,
                    ledger_path,
                    completed=False,
                    input_files=input_files,
                )
            except Exception as exc:  # noqa: BLE001 - preserve the original pipeline failure
                print(f"curate/nemo_curator: WARNING could not write failed ledger: {exc}")
        raise


#: What ``filtered`` is attributed to when the per-gate breakdown could not be
#: reconciled. Not a placeholder: it is the honest name for what this step then
#: knows — how many documents entered and left, and nothing in between.
UNATTRIBUTED = "unattributed"

ATTRIBUTION_NOTE = (
    "Per-gate removal counts come from Curator's own per-stage item counters "
    "(StagePerfStats.num_items_processed, collected via TaskPerfUtils.collect_stage_metrics) and "
    "are published only when they sum to the removal measured independently by counting the input "
    "and output files on disk. When the two disagree the breakdown is discarded and everything is "
    "recorded as 'unattributed', because a plausible-looking attribution that does not reconcile "
    "is worse than declaring the breakdown unavailable."
)

UNRECONCILED_NOTE = (
    "per-stage counters were collected but did not sum to the {observed} documents the corpus "
    "actually lost; the breakdown was discarded rather than published"
)


def collect_stage_counts(tasks: list) -> dict[str, int]:
    """How many documents each Curator stage processed, by stage name.

    Curator tracks this already — every ``ProcessingStage`` accumulates a
    ``StagePerfStats`` on the tasks flowing through it — so nothing is
    instrumented here. ``collect_stage_metrics`` handles the awkward part: a
    batch that fans out into several tasks carries the same stats object on each,
    and it deduplicates by identity so the fan-out is not counted twice.

    Returns ``{}`` rather than raising when the executor reports nothing. That is
    a reason to fall back to an unattributed ledger, not to fail a run whose
    output is already written.
    """
    if not tasks:
        return {}
    try:
        from nemo_curator.tasks.utils import TaskPerfUtils

        collected = TaskPerfUtils.collect_stage_metrics(tasks)
    except Exception:  # noqa: BLE001 - telemetry must never fail a completed run
        return {}

    counts: dict[str, int] = {}
    for stage_name, metrics in (collected or {}).items():
        values = metrics.get("num_items_processed")
        if values is None:
            continue
        counts[str(stage_name)] = int(sum(float(v) for v in values))
    return counts


def attribute_removals(stage_names: list[str], counts: dict[str, int], observed_removed: int) -> dict[str, int] | None:
    """Turn per-stage item counts into per-gate removal counts.

    A filter chain is a sequence of stages each fed by the previous one, so what
    a stage removed is what it received minus what the next stage received. The
    counts are matched to declared stage names by prefix because a
    ``CompositeStage`` decomposes into differently-named parts at execution.

    Returns ``None`` — meaning "record this as unattributed" — whenever the
    result cannot be trusted: fewer than two stages reported, or the removals do
    not sum to ``observed_removed``, which was measured by counting the files on
    disk and is therefore independent of anything the pipeline reported about
    itself. That reconciliation is the whole safeguard. Without it a stage
    silently dropping rows would produce a breakdown that looks complete.
    """
    ordered: list[tuple[str, int]] = []
    used_metrics: set[str] = set()
    for declared in stage_names:
        exact = declared if declared in counts and declared not in used_metrics else None
        candidates = [
            name
            for name in counts
            if name not in used_metrics
            and name.startswith(declared)
            and not any(
                other != declared and len(other) > len(declared) and name.startswith(other) for other in stage_names
            )
        ]
        matched = exact or min(candidates, key=lambda name: (len(name), name), default=None)
        if matched is not None:
            ordered.append((declared, counts[matched]))
            used_metrics.add(matched)

    if len(ordered) < 2:  # noqa: PLR2004 - a diff needs two points
        return None

    removals = {
        name: ordered[i][1] - ordered[i + 1][1]
        for i, (name, _) in enumerate(ordered[:-1])
        if ordered[i][1] - ordered[i + 1][1] > 0
    }
    if sum(removals.values()) != observed_removed:
        return None
    return removals


def emit_ledger(
    cfg: dict,
    path: str,
    *,
    completed: bool,
    stage_names: list[str] | None = None,
    stage_counts: dict[str, int] | None = None,
    input_files: list[str] | None = None,
) -> None:
    """Write the run's record accounting.

    The producer half of ``curation_ledger``. ``curate/audit`` consumes it to tell
    a record removed on purpose from one lost to a swallowed exception — a
    distinction nothing in the input and output alone can supply.

    What it can and cannot say is stated in the ledger itself rather than left to
    be inferred: see :data:`ATTRIBUTION_NOTE`.
    """
    from nemotron.steps.curate.nemo_curator.runtime import ledger as ledger_module

    source_field = cfg.get("source_field")
    # readable_inputs, not resolve_inputs: run() narrows to what the reader
    # parses, and an emitter that re-resolved wide described a different
    # corpus from the one filtered -- a number the audit consumes.
    resolved_inputs = input_files if input_files is not None else readable_inputs(cfg["input_glob"])
    input_counts = run_manifest.count_jsonl(resolved_inputs, source_field)
    output_files = sorted(str(p) for p in Path(cfg["output_dir"]).rglob("*.jsonl"))
    output_counts = run_manifest.count_jsonl(output_files, source_field)

    entry = ledger_module.StageLedger(stage="curate/nemo_curator")
    entry.add_input(input_counts["row_count"])
    entry.add_success(output_counts["row_count"])
    removed = input_counts["row_count"] - output_counts["row_count"]
    if not completed:
        entry.add_failed(
            cfg["output_dir"],
            "pipeline did not complete; absent rows cannot be classified as filtered",
            max(removed, 0),
        )
    elif removed > 0:
        per_gate = attribute_removals(list(stage_names or []), dict(stage_counts or {}), removed)
        if per_gate:
            for gate, count in per_gate.items():
                entry.add_filtered(gate, count)
        else:
            entry.add_filtered(UNATTRIBUTED, removed)
            # Distinguish "no counters reached us" from "counters reached us and
            # disagreed with the files on disk". The second is a finding.
            if stage_counts:
                entry.notes["unreconciled"] = UNRECONCILED_NOTE.format(observed=removed)
    if completed:
        entry.notes["attribution"] = ATTRIBUTION_NOTE
    entry.notes["completed"] = completed
    entry.notes["mode"] = cfg.get("mode", "filter")

    if removed < 0:
        # More rows out than in. Balancing this by inventing a negative filtered
        # count would hide it; recording it as an unprocessable unit makes the
        # audit say so out loud.
        entry.add_quarantined(
            cfg["output_dir"],
            f"output has {-removed} more rows than input; the run did not only remove documents",
            0,
        )

    # A failed run must preserve its terminal-state evidence even when the
    # available counts cannot balance.
    entry.write(path, require_balanced=completed and removed >= 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate JSONL text with NeMo Curator")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    run(cfg)


def emit_manifest(
    cfg: dict,
    path: str,
    started_at: str,
    *,
    completed: bool,
    stage_names: list[str] | None = None,
    stage_counts: dict[str, int] | None = None,
    input_files: list[str] | None = None,
    identity_config: dict[str, Any] | None = None,
    policy_resolution: PolicyResolution | None = None,
) -> None:
    """Write the run manifest after the pipeline's own write barrier.

    Row counting happens here rather than inside the pipeline because Curator's
    writers return file paths without row counts — ``BaseWriter.process`` logs
    ``task.num_items`` and discards it — so the wrapper has no per-stage figure
    to carry forward from the writer.

    The per-gate breakdown does reach here, from Curator's per-stage counters. It
    is reconciled independently of the ledger, against this function's own disk
    counts: if the two artifacts ever disagreed, that disagreement would itself
    be the finding, and sharing one computation between them would hide it.
    """
    source_field = cfg.get("source_field")
    output_files = sorted(str(p) for p in Path(cfg["output_dir"]).rglob("*.jsonl"))

    # readable_inputs, not resolve_inputs: run() narrows to what the reader
    # parses, and an emitter that re-resolved wide described a different
    # corpus from the one filtered -- a number the audit consumes.
    resolved_inputs = input_files if input_files is not None else readable_inputs(cfg["input_glob"])
    input_counts = run_manifest.count_jsonl(resolved_inputs, source_field)
    output_counts = run_manifest.count_jsonl(output_files, source_field)

    declared = None
    removed = input_counts.get("row_count", 0) - output_counts.get("row_count", 0)
    if not completed:
        declared = {
            "attribution": run_manifest.ATTRIBUTION_DECLARED,
            "rows_absent_from_output": max(removed, 0),
            "filtered": None,
            "failed": max(removed, 0),
            "quarantined": None,
        }
    elif removed > 0:
        per_gate = attribute_removals(list(stage_names or []), dict(stage_counts or {}), removed)
        if per_gate:
            declared = {
                "attribution": run_manifest.ATTRIBUTION_DECLARED,
                "rows_absent_from_output": removed,
                "filtered": per_gate,
                "failed": None,
                "quarantined": None,
            }

    document = run_manifest.build_manifest(
        step_id="curate/nemo_curator",
        config=identity_config if identity_config is not None else cfg,
        started_at=started_at,
        input_glob=cfg["input_glob"],
        input_counts=input_counts,
        output_counts=output_counts,
        id_field=cfg.get("id_field"),
        source_field=source_field,
        completed_at=run_manifest.utc_now() if completed else None,
        declared=declared,
    )
    # Say whether a person approved the thresholds that produced this corpus.
    # The output directory is called filtered_jsonl whether or not a policy was
    # applied, so on a first run — where there is nothing to approve yet, by
    # design — the artifact looks exactly like a released one. Naming the state
    # is cheaper than renaming the directory and does not disturb the paths the
    # flow derives for the steps downstream.
    # Reuse the resolution the run already made. Re-resolving from disk read the
    # policy a second time and re-globbed the corpus without the input_files it
    # was handed, so a run whose input_glob covered its own output_dir raised
    # here and was reported as failed after succeeding. And when the run failed
    # *because* the policy was refused, re-resolving raised the same error inside
    # the failure handler -- so the one failure that most needs a manifest got
    # none at all.
    resolved = policy_resolution
    unresolvable: str | None = None
    if resolved is None:
        try:
            resolved = _resolve_policy(cfg, input_files)
        except Exception as exc:  # noqa: BLE001 - the manifest records this, it does not raise it
            resolved = PolicyResolution([], {}, [], {})
            unresolvable = str(exc)
    thresholds = resolved.thresholds
    document["policy"] = {
        # The override is tested before the count. A policy carried past the
        # contract with nothing to apply -- the documented escape hatch points
        # approved_policy at candidate_policies.yaml, which has `candidates` and
        # no `thresholds` -- was recorded as plainly "unapproved", which reads as
        # "no policy was configured" rather than "one was, it was overridden, and
        # it turned out to gate nothing".
        "status": (
            "unresolved"
            if unresolvable
            else "override_unvalidated"
            if resolved.overridden
            else "approved"
            if thresholds
            else "unapproved"
        ),
        "thresholds_applied": len(thresholds),
        # What the artifact can actually support. The gate rests on the corpus
        # fingerprint and the scorer version, not on a name: `approval` is
        # optional in the contract, so "approved" means the policy met the
        # schema, never that a person signed it. Recording the declared method
        # separates the two for a reader who has only this file.
        "approval_declared": (resolved.identity or {}).get("approval_method"),
        "note": (
            f"The configured policy could not be resolved, so none was applied: {unresolvable}"
            if unresolvable
            else (
                f"{len(thresholds)} threshold(s) applied under allow_unvalidated_policy: the "
                "policy did not meet the approval contract and ran anyway. The run log names "
                "which fields were unmet. Treat this corpus as an experiment, not a release."
                if resolved.overridden
                else "Thresholds from an approved policy, checked against a fingerprint of the "
                "corpus they were measured on."
            )
            if thresholds or resolved.overridden
            else "No approved policy was applied. Any filtering came from configuration "
            "written by hand, which carries no record of the corpus it was measured on. "
            "Treat this corpus as a measurement, not as a release."
        ),
    }
    run_manifest.write_manifest(path, document)


if __name__ == "__main__":
    main()
