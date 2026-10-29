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

"""Language profiles for tokenizer extension.

Adding a new target language should be a data change here, not a code change
anywhere else. A profile answers the two questions the extend step asks that
are genuinely language-dependent:

  normalizer   how to normalize corpus text before BPE training. NFKC is always
               applied; this names any *additional* script-specific pass.
  script       which key in replace_bpe.SCRIPT_UNICODE_RANGES identifies the
               tokens that method=replace should prune. For a language with its
               own block that is the block; for a Latin-script language it is
               the set of codepoints only that language uses.

Both are overridable per-config (`script_normalizer:` / `remove_script:`), so a
one-off experiment never needs an entry here. `language:` just spares you from
restating the same two values in every config.

    language: vietnamese      # -> normalizer=none, remove_script=vietnamese
    language: hindi           # -> normalizer=devanagari, remove_script=devanagari
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageProfile:
    normalizer: str  # key into NORMALIZERS ("none" = NFKC only)
    script: str  # key into SCRIPT_UNICODE_RANGES
    fasttext: str  # fastText cc.<code>.300 code, for the FOCUS init
    encoder: str  # encoder covering this language, for the bert init


# MuRIL covers 17 Indian languages and nothing else, so it is the right
# auxiliary encoder for the Indic entries and the wrong one everywhere else.
# XLM-R is the general fallback: massively multilingual, same structural role.
_MURIL = "google/muril-base-cased"
_XLMR = "FacebookAI/xlm-roberta-base"

LANGUAGES: dict[str, LanguageProfile] = {
    # Devanagari-block languages share both the normalizer and the prune set.
    "hindi": LanguageProfile("devanagari", "devanagari", "hi", _MURIL),
    "marathi": LanguageProfile("devanagari", "devanagari", "mr", _MURIL),
    "nepali": LanguageProfile("devanagari", "devanagari", "ne", _MURIL),
    "sanskrit": LanguageProfile("devanagari", "devanagari", "sa", _MURIL),
    "bengali": LanguageProfile("none", "bengali", "bn", _MURIL),
    "punjabi": LanguageProfile("none", "gurmukhi", "pa", _MURIL),
    "gujarati": LanguageProfile("none", "gujarati", "gu", _MURIL),
    "odia": LanguageProfile("none", "oriya", "or", _MURIL),
    "tamil": LanguageProfile("none", "tamil", "ta", _MURIL),
    "telugu": LanguageProfile("none", "telugu", "te", _MURIL),
    "kannada": LanguageProfile("none", "kannada", "kn", _MURIL),
    "malayalam": LanguageProfile("none", "malayalam", "ml", _MURIL),
    "urdu": LanguageProfile("none", "arabic", "ur", _MURIL),
    # Latin-script: no extra normalizer, and "its own tokens" are the ones
    # carrying codepoints only this language uses. MuRIL has no Vietnamese.
    "vietnamese": LanguageProfile("none", "vietnamese", "vi", _XLMR),
}

FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{code}.300.bin.gz"


def profile(name: str) -> LanguageProfile:
    key = (name or "").strip().lower()
    if key not in LANGUAGES:
        raise ValueError(
            f"Unknown language {name!r}. Known: {sorted(LANGUAGES)}. Add a LanguageProfile in languages.py."
        )
    return LANGUAGES[key]


def fasttext_url(name: str) -> str:
    return FASTTEXT_URL.format(code=profile(name).fasttext)


def script_ranges(name: str) -> list[tuple[int, int]]:
    """Unicode ranges identifying tokens that belong to this language.

    Used to find the base model's existing target-language rows, which anchor
    both the norm correction and the FOCUS candidate pool.
    """
    from script_ranges import SCRIPT_UNICODE_RANGES

    return SCRIPT_UNICODE_RANGES[profile(name).script]


def get_normalizer(name: str):
    """Return a normalizer object exposing .normalize(str), or None for NFKC-only."""
    key = (name or "none").strip().lower()
    if key in ("none", "nfkc", ""):
        return None
    if key == "devanagari":
        try:
            from indicnlp.normalize.indic_normalize import DevanagariNormalizer
        except ImportError as exc:
            # Do NOT silently fall back to NFKC-only: that trains a *different*
            # tokenizer from the same config, with no signal in the artifacts.
            raise ImportError(
                "script_normalizer='devanagari' needs indic-nlp-library. "
                "Install it with:  pip install 'nemotron[tokenizer-extension]'  "
                "(or set script_normalizer: none to accept NFKC-only "
                "normalization, which produces a different tokenizer)."
            ) from exc
        return DevanagariNormalizer()
    raise ValueError(f"Unknown script_normalizer {name!r}. Known: none, nfkc, devanagari.")


def resolve(cfg: dict) -> tuple[str, str]:
    """Resolve (script_normalizer, remove_script) from a config.

    Precedence: explicit key > language profile > legacy Devanagari default.
    The legacy default keeps every pre-existing Hindi config working untouched.
    """
    lang = cfg.get("language")
    if lang is not None:
        key = str(lang).strip().lower()
        if key not in LANGUAGES:
            raise ValueError(
                f"Unknown language {lang!r}. Known: {sorted(LANGUAGES)}. "
                "Add a LanguageProfile in languages.py, or set script_normalizer "
                "and remove_script explicitly."
            )
        prof = LANGUAGES[key]
        norm, script = prof.normalizer, prof.script
    else:
        norm, script = "devanagari", "devanagari"

    # A key present but null means "defer to the profile", exactly as if it were
    # absent. Without this, `remove_script: null` became the string "None" and
    # failed downstream with `Unknown script 'None'`.
    def _override(key: str, fallback: str) -> str:
        val = cfg.get(key)
        return fallback if val is None else str(val)

    return (_override("script_normalizer", norm).lower(), _override("remove_script", script))


def describe(cfg: dict) -> dict:
    """The resolved language settings, and where each value came from.

    ``resolve`` returns the two values a run acts on but not their provenance,
    and a summary recording only model, method and budgets cannot say which
    language a tokenizer was built for, nor whether an override was in effect.
    Two tokenizers built from one corpus under different scripts are otherwise
    indistinguishable after the fact.
    """
    raw = cfg.get("language")
    key = str(raw).strip().lower() if raw is not None else None
    normalizer, script = resolve(cfg)
    profile = LANGUAGES.get(key) if key else None
    return {
        "language": key,
        "language_source": "config" if key else "legacy-devanagari-default",
        "script_normalizer": normalizer,
        "remove_script": script,
        "overrides": {
            "script_normalizer": cfg.get("script_normalizer") is not None,
            "remove_script": cfg.get("remove_script") is not None,
        },
        "profile_defaults": (
            {
                "script_normalizer": profile.normalizer,
                "remove_script": profile.script,
                "fasttext": profile.fasttext,
                "encoder": profile.encoder,
            }
            if profile is not None
            else None
        ),
    }
