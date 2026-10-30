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

"""Quality signals Curator does not provide, parameterised by a language pack.

Every class here is a ``DocumentFilter`` subclass, which is how Curator's own
CLIMB tutorial adds a scorer without modifying the library
(``tutorials/text/nemotron-climb-data-curation/3_prune.py:50``). None of them
contains a language name: the character sets, word lists, patterns and fold maps
all arrive from a :class:`~nemotron.steps.curate.nemo_curator.runtime.langpack.LanguagePack`.

The base class is resolved lazily so this module imports on a host without
``nemo_curator``. Subclassing the real base matters at runtime: Curator's
``Score`` stage checks ``isinstance(score_fn, DocumentFilter)`` and would
otherwise treat one of these as a plain callable.
"""

from __future__ import annotations

import unicodedata
from typing import Any, cast

from nemotron.steps.curate.nemo_curator.runtime.langpack import LanguagePack
from nemotron.steps.curate.nemo_curator.runtime.registry import IMPL_VERSION

#: Bumped when a signal's numbers change, so a policy or profile measured under
#: the previous version is refused rather than silently compared. 0.2.0 scores on
#: NFC: every signal that reads letters or diacritics returns a different value
#: for non-NFC text than 0.1.0 did, and 11.69% of a Vietnamese and 28.82% of a
#: Hindi C4 sample are affected.


def _document_filter_base() -> type:
    """Curator's DocumentFilter when it is installed, a stand-in when it is not."""
    try:
        from nemo_curator.stages.text.filters.doc_filter import DocumentFilter

        return cast(type, DocumentFilter)
    except Exception:  # noqa: BLE001 - importability on a plain CI host is the point

        class _StandaloneDocumentFilter:
            """Same contract, so tests exercise the real scoring code."""

            def __init__(self) -> None:
                self._name = type(self).__name__

            def score_document(self, text: str) -> float:  # pragma: no cover - overridden
                raise NotImplementedError

            def keep_document(self, score: float) -> bool:  # pragma: no cover - overridden
                raise NotImplementedError

        return _StandaloneDocumentFilter


_RAW_BASE = _document_filter_base()


class _Base(_RAW_BASE):  # type: ignore[valid-type,misc]
    """Every signal in this module, with one property enforced for all of them.

    Scoring is done on NFC. Vietnamese and Devanagari both have two encodings of
    the same correct text, and none of the signals distinguished them: in NFD a
    combining mark is its own codepoint, so ``unicodedata.category(ch)[0] == "L"``
    excludes it and the base letter folds to itself. Measured on one correct
    Vietnamese sentence, NFC vs NFD vs the same text with its diacritics actually
    stripped:

    ::

                            NFC       NFD    stripped
        diacritic_ratio  0.3231    0.0615      0.0615
        stopword_ratio   0.1000    0.0000      0.0000

    NFD scored identically to genuinely degraded text, on the two signals whose
    whole purpose is to detect that degradation. It is not rare: 11.69% of a
    20,000-document Vietnamese C4 sample and 28.82% of the Hindi one are not NFC,
    and a corpus from macOS or OCR is routinely majority-NFD.

    The normalisation is applied to the string handed to ``score_document`` and
    nowhere else. The delivered corpus keeps the bytes it arrived with -- a
    measurement stage that rewrites the text it measures is exactly the defect
    that let a classifier truncate 53.4% of a corpus while every count reconciled.

    Enforced in ``__init_subclass__`` rather than by convention, because a signal
    added later would otherwise reintroduce this silently and its scores would
    look entirely reasonable.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        scorer = cls.__dict__.get("score_document")
        if scorer is None or getattr(scorer, "_nfc_wrapped", False):
            return

        def score_document(self: Any, text: str, _inner: Any = scorer) -> float:
            if text and not unicodedata.is_normalized("NFC", text):
                text = unicodedata.normalize("NFC", text)
            return cast(float, _inner(self, text))

        score_document._nfc_wrapped = True  # type: ignore[attr-defined]
        score_document.__doc__ = scorer.__doc__
        score_document.__name__ = scorer.__name__
        cls.score_document = score_document  # type: ignore[method-assign]


# -- Unicode helpers ----------------------------------------------------------


#: Zero-width joiner and non-joiner. Unicode category ``Cf``, so no category
#: rule reaches them, but they are obligatory inside Devanagari and other Indic
#: conjuncts. Measured: ordinary joiner use moves a correct Hindi paragraph from
#: 0.218 to 0.265 — across the 0.25 default — so leaving them out rejects text
#: for being spelled correctly.
LINGUISTIC_FORMAT_CHARS = frozenset({"‌", "‍"})


def is_alphanumeric(ch: str) -> bool:
    """Whether a character carries linguistic content.

    Accepts Unicode categories L, N and **all** of M — not just ``Mn``.
    Devanagari matras split across ``Mn`` (nonspacing) and ``Mc`` (spacing
    combining) with ``Mc`` in the majority, while Vietnamese NFD produces only
    ``Mn``. A fix written as ``\\p{Mn}`` therefore passes every Vietnamese test
    and still rejects every Indic and South-East Asian script.

    Also accepts the two joiners in :data:`LINGUISTIC_FORMAT_CHARS`, which no
    category rule covers.
    """
    return unicodedata.category(ch)[0] in ("L", "N", "M") or ch in LINGUISTIC_FORMAT_CHARS


def _tokens(text: str) -> list[str]:
    """Whitespace tokens, stripped of surrounding punctuation."""
    return [t for t in (w.strip("".join(_PUNCT)) for w in text.split()) if t]


_PUNCT = tuple(chr(c) for c in range(0x21, 0x40) if not chr(c).isalnum())


# -- the Unicode-correct replacement -----------------------------------------


class UnicodeAwareNonAlphaNumericFilter(_Base):
    """Non-alphanumeric character ratio that works outside Latin-ASCII.

    Curator's ``NonAlphaNumericFilter`` uses ``[a-zA-Z0-9\\n?!,.]``. Measured
    against the canonical sample set, at its 0.25 default:

    ::

                            ascii        LN      LN_Mn      LN_M
            en             0.193 P    0.193 P   0.193 P   0.193 P
            vi-NFC         0.465 F    0.221 P   0.221 P   0.221 P
            vi-NFD         0.427 F    0.391 F   0.173 P   0.173 P
            vi-stripped    0.221 P    0.221 P   0.221 P   0.221 P
            hi-NFC         1.000 F    0.495 F   0.369 F   0.216 P
            hi-NFD         1.000 F    0.495 F   0.369 F   0.216 P

    The signal is inverted for Vietnamese — diacritic-stripped text scores
    *better* than correct text — and Hindi is rejected outright.

    This is an *addition*, not a bug report. Curator knows: its shipped
    ``heuristic_filter_non_english_pipeline.yaml`` omits ``NonAlphaNumericFilter``,
    ``CommonEnglishWordsFilter`` and ``WordsWithoutAlphabetsFilter`` — exactly the
    three that assume ASCII — where the English pipeline includes all three. So
    the numbers above are what those filters *would* do if reached for, not what
    Curator does to a non-English corpus.

    The gap that leaves is the reason this class exists: dropping the filter
    means a non-English corpus gets no character-composition gate at all, and
    junk that is 80% punctuation passes as readily as prose. Scoring by Unicode
    category rather than by ASCII range restores the gate for any script.

    Uses ``unicodedata.category`` rather than the ``regex`` module. Curator's
    ``constants.py`` does import ``regex`` and uses POSIX classes on the two
    lines above the ASCII one, so the dependency exists; stdlib simply avoids
    adding another.
    """

    def __init__(self, max_non_alpha_numeric_to_text_ratio: float = 0.25, keep: str = "\n?!,.") -> None:
        super().__init__()
        self._cutoff = max_non_alpha_numeric_to_text_ratio
        self._keep = frozenset(keep)
        self._name = "unicode_alpha_numeric"

    def score_document(self, text: str) -> float:
        if not text:
            return 1.0
        kept = sum(1 for ch in text if is_alphanumeric(ch) or ch in self._keep)
        return (len(text) - kept) / len(text)

    def keep_document(self, score: float) -> bool:
        return score <= self._cutoff


# -- pack-parameterised signals ----------------------------------------------


class _PackFilter(_Base):
    """Base for signals whose behaviour comes entirely from a language pack."""

    capability = ""

    #: The registry key this class implements. Distinct from ``capability``,
    #: which several signals share: ScriptRatio, LatinRatio and
    #: ForeignScriptRatio all need the pack's ``script_ratio`` data, and naming
    #: the stage after the capability gave all three the same name. Two of them
    #: in one policy then wrote the same ``__script_ratio`` column, each
    #: overwriting the other, and collapsed into a single entry in the ledger's
    #: per-gate attribution — which is the one figure that says which gate
    #: removed what.
    signal_name = ""

    def __init__(self, pack: LanguagePack) -> None:
        super().__init__()
        if self.capability and not pack.supports(self.capability):
            raise ValueError(
                f"{type(self).__name__} needs the {self.capability!r} capability, which the "
                f"{pack.language_tag!r} pack does not declare"
            )
        self._pack = pack
        self._name = self.signal_name or self.capability or type(self).__name__


class ScriptRatio(_PackFilter):
    """Share of letters that belong to this language's own script.

    Continuous, unlike Curator's ``HistogramFilter``, which thresholds
    internally and returns 0 or 1 — a binary value cannot be swept, so it cannot
    tell you what a threshold would cost.

    **This separates scripts, not languages.** A Latin-script pack necessarily
    includes the unmarked Latin letters, so English scores as high as Vietnamese
    against the Vietnamese pack — correctly, since both are written in Latin.
    Use :class:`DiacriticRatio` or :class:`StopwordRatio` to tell apart languages
    that share a script; use this one to notice a corpus in the wrong script
    entirely.
    """

    capability = "script_ratio"
    signal_name = "script_ratio"

    def __init__(self, pack: LanguagePack, min_script_ratio: float = 0.0) -> None:
        super().__init__(pack)
        self._cutoff = min_script_ratio

    def score_document(self, text: str) -> float:
        letters = [ch for ch in text if unicodedata.category(ch)[0] in ("L", "M")]
        if not letters:
            return 0.0
        return sum(1 for ch in letters if ch in self._pack.charset) / len(letters)

    def keep_document(self, score: float) -> bool:
        return score >= self._cutoff


class DiacriticRatio(_PackFilter):
    """Share of letters carrying a mark this language treats as removable.

    Only meaningful where marks are removable at all. Vietnamese tone marks
    strip to degraded but readable text; Devanagari matras are obligatory
    vowels, so the Hindi pack does not declare this capability and the signal is
    absent from its report rather than computed on a false premise.
    """

    capability = "diacritic_ratio"
    signal_name = "diacritic_ratio"

    def __init__(self, pack: LanguagePack, min_diacritic_ratio: float = 0.0) -> None:
        super().__init__(pack)
        self._cutoff = min_diacritic_ratio

    def score_document(self, text: str) -> float:
        letters = [ch for ch in text if unicodedata.category(ch)[0] == "L"]
        if not letters:
            return 0.0
        marked = sum(1 for ch in letters if self._pack.fold(ch) != ch)
        return marked / len(letters)

    def keep_document(self, score: float) -> bool:
        return score >= self._cutoff


class LatinRatio(_PackFilter):
    """Share of letters that are plain unmarked Latin.

    High values on a non-Latin corpus mean untranslated boilerplate, code, or
    the wrong language entirely.
    """

    capability = "script_ratio"
    signal_name = "latin_ratio"

    def __init__(self, pack: LanguagePack, max_latin_ratio: float = 1.0) -> None:
        super().__init__(pack)
        self._cutoff = max_latin_ratio

    def score_document(self, text: str) -> float:
        letters = [ch for ch in text if unicodedata.category(ch)[0] == "L"]
        if not letters:
            return 0.0
        plain = sum(1 for ch in letters if ch.isascii() and ch.isalpha())
        return plain / len(letters)

    def keep_document(self, score: float) -> bool:
        return score <= self._cutoff


class ForeignScriptRatio(_PackFilter):
    """Share of letters from neither this language's script nor base Latin."""

    capability = "script_ratio"
    signal_name = "foreign_script_ratio"

    def __init__(self, pack: LanguagePack, max_foreign_ratio: float = 1.0) -> None:
        super().__init__(pack)
        self._cutoff = max_foreign_ratio

    def score_document(self, text: str) -> float:
        letters = [ch for ch in text if unicodedata.category(ch)[0] == "L"]
        if not letters:
            return 0.0
        foreign = sum(1 for ch in letters if ch not in self._pack.charset and not (ch.isascii() and ch.isalpha()))
        return foreign / len(letters)

    def keep_document(self, score: float) -> bool:
        return score <= self._cutoff


class StopwordRatio(_PackFilter):
    """Share of tokens that are function words.

    Prose has a characteristic function-word density; keyword lists, navigation
    furniture and machine output do not.
    """

    capability = "stopword_ratio"
    signal_name = "stopword_ratio"

    def __init__(self, pack: LanguagePack, min_stopword_ratio: float = 0.0) -> None:
        super().__init__(pack)
        self._cutoff = min_stopword_ratio

    def score_document(self, text: str) -> float:
        tokens = _tokens(text.casefold())
        if not tokens:
            return 0.0
        return sum(1 for t in tokens if t in self._pack.stopwords) / len(tokens)

    def keep_document(self, score: float) -> bool:
        return score >= self._cutoff


class FoldedStopwordRatio(_PackFilter):
    """Stopword density measured after removing marks from both sides.

    Recovers text written without diacritics, which a mark-sensitive gate would
    delete outright. Folding collapses distinct words together, so this signal
    is noisier than :class:`StopwordRatio` by construction — the collision count
    is a property of the language and belongs in the report, not a defect.
    """

    capability = "stopword_ratio_folded"
    signal_name = "stopword_ratio_folded"

    def __init__(self, pack: LanguagePack, min_stopword_ratio_folded: float = 0.0) -> None:
        super().__init__(pack)
        self._cutoff = min_stopword_ratio_folded
        self._folded = pack.folded_stopwords()

    def score_document(self, text: str) -> float:
        tokens = _tokens(self._pack.fold(text).casefold())
        if not tokens:
            return 0.0
        return sum(1 for t in tokens if t in self._folded) / len(tokens)

    def keep_document(self, score: float) -> bool:
        return score >= self._cutoff

    def collision_count(self) -> int:
        """How many stopwords folding merges away, for the report."""
        return len(self._pack.stopwords) - len(self._folded)


class BoilerplateHits(_PackFilter):
    """How many of this language's boilerplate patterns a document matches.

    Curator's ``BoilerPlateStringFilter`` hardcodes English cookie and privacy
    phrases, so on any other language it matches nothing and reports nothing.
    """

    capability = "boilerplate_hits"
    signal_name = "boilerplate_hits"

    def __init__(self, pack: LanguagePack, max_boilerplate_hits: int = 1_000_000) -> None:
        super().__init__(pack)
        self._cutoff = max_boilerplate_hits

    def score_document(self, text: str) -> float:
        return float(sum(1 for pattern in self._pack.boilerplate if pattern.search(text)))

    def keep_document(self, score: float) -> bool:
        return score <= self._cutoff


class SentenceEndRatio(_PackFilter):
    """Share of sentence-like spans ending in a terminator this language uses.

    Curator's ``PunctuationFilter`` looks for ``.``, ``!`` and ``?``. Hindi ends
    sentences with the danda ``।`` (U+0964, category ``Po``), so a correct Hindi
    paragraph scores zero against it. Terminators come from the pack.
    """

    capability = "sentence_end_ratio"
    signal_name = "sentence_end_ratio"

    def __init__(self, pack: LanguagePack, min_sentence_end_ratio: float = 0.0) -> None:
        super().__init__(pack)
        self._cutoff = min_sentence_end_ratio
        self._terminators = frozenset(pack.sentence_terminators)

    def score_document(self, text: str) -> float:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return 0.0
        return sum(1 for line in lines if line[-1] in self._terminators) / len(lines)

    def keep_document(self, score: float) -> bool:
        return score >= self._cutoff


#: Which class serves each capability, for the registry to wire up.
BY_CAPABILITY: dict[str, tuple[type, ...]] = {
    "script_ratio": (ScriptRatio, LatinRatio, ForeignScriptRatio),
    "diacritic_ratio": (DiacriticRatio,),
    "stopword_ratio": (StopwordRatio,),
    "stopword_ratio_folded": (FoldedStopwordRatio,),
    "boilerplate_hits": (BoilerplateHits,),
    "sentence_end_ratio": (SentenceEndRatio,),
}


def describe() -> dict[str, Any]:
    """What this module provides, for the report's provenance block."""
    return {
        "impl_version": IMPL_VERSION,
        "language_neutral": True,
        "classes": sorted(
            {cls.__name__ for group in BY_CAPABILITY.values() for cls in group}
            | {UnicodeAwareNonAlphaNumericFilter.__name__}
        ),
    }
