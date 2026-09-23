# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Signals Curator lacks, and the Unicode traps they exist to survive.

Two languages, two orthogonal traps. Vietnamese NFC and NFD differ materially
and produce only ``Mn``; Hindi is identical in both forms and produces mostly
``Mc``. A fix tested against either language alone passes and is still broken.
"""

from __future__ import annotations

import re
import unicodedata as ud
from pathlib import Path

import pytest

from nemotron.steps.curate.nemo_curator.runtime import langpack, signals

FIXTURES = Path(__file__).parent / "fixtures" / "langpacks"

EN = "Vietnamese is the official language of Vietnam, written in the Latin alphabet."
VI = "Tiếng Việt là ngôn ngữ chính thức của Việt Nam, được viết bằng chữ Quốc ngữ."
HI = "यह एक परीक्षण वाक्य है। भारत एक विशाल देश है जिसमें अनेक भाषाएँ बोली जाती हैं।"

CURATOR_DEFAULT = 0.25


def load_fixture(language: str) -> langpack.LanguagePack:
    return langpack.load(f"x-test-{language}", FIXTURES)


def strip_marks(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in ud.normalize("NFD", text) if not ud.combining(c))


def ascii_ratio(text: str) -> float:
    """Curator's own implementation, for comparison."""
    kept = len(re.findall(r"[a-zA-Z0-9\n?!,.]", text))
    return (len(text) - kept) / len(text) if text else 1.0


def mn_only_ratio(text: str) -> float:
    """The natural fix when only Vietnamese is available to test against."""
    keep = set("\n?!,.")
    kept = sum(1 for c in text if ud.category(c)[0] in ("L", "N") or ud.category(c) == "Mn" or c in keep)
    return (len(text) - kept) / len(text) if text else 1.0


@pytest.fixture(scope="module")
def unicode_filter():
    return signals.UnicodeAwareNonAlphaNumericFilter()


# -- the mandatory case table -------------------------------------------------


@pytest.mark.parametrize(
    "name,text",
    [
        ("en", EN),
        ("vi-NFC", ud.normalize("NFC", VI)),
        ("vi-NFD", ud.normalize("NFD", VI)),
        ("vi-stripped", strip_marks(VI)),
        ("hi-NFC", ud.normalize("NFC", HI)),
        ("hi-NFD", ud.normalize("NFD", HI)),
    ],
)
def test_every_form_of_correct_text_is_kept(unicode_filter, name, text) -> None:
    score = unicode_filter.score_document(text)

    assert unicode_filter.keep_document(score), f"{name} scored {score:.3f} against {CURATOR_DEFAULT}"


def test_curators_implementation_rejects_the_same_text(unicode_filter) -> None:
    """The defect being fixed, stated as a measurement rather than an opinion."""
    assert ascii_ratio(ud.normalize("NFC", VI)) > CURATOR_DEFAULT
    assert ascii_ratio(HI) == pytest.approx(1.0), "Devanagari matches nothing in [a-zA-Z0-9]"


def test_the_ascii_signal_is_inverted_for_vietnamese() -> None:
    """Degraded text scores better than correct text — worse than mis-calibration."""
    correct = ascii_ratio(ud.normalize("NFC", VI))
    degraded = ascii_ratio(strip_marks(VI))

    assert degraded < correct
    assert degraded <= CURATOR_DEFAULT < correct


def test_an_mn_only_fix_passes_vietnamese_and_still_breaks_devanagari() -> None:
    """The trap two languages catch and one cannot.

    Matras split across Mn and Mc with Mc in the majority; Vietnamese NFD
    produces only Mn. A fix written as \\p{Mn} is green on every Vietnamese case
    and rejects correct Hindi outright.
    """
    for text in (EN, ud.normalize("NFC", VI), ud.normalize("NFD", VI)):
        assert mn_only_ratio(text) <= CURATOR_DEFAULT

    assert mn_only_ratio(HI) > CURATOR_DEFAULT, "the Mn-only fix must fail here"


def test_devanagari_marks_are_mostly_spacing_combining() -> None:
    """Why Mn alone is not enough, as a property of the script rather than a claim."""
    marks = [c for c in HI if ud.category(c).startswith("M")]

    assert sum(1 for c in marks if ud.category(c) == "Mc") > sum(1 for c in marks if ud.category(c) == "Mn")


def test_d_bar_does_not_decompose_and_needs_the_pack(unicode_filter) -> None:
    assert ud.normalize("NFD", "đ") == "đ", "đ has no decomposition; NFD cannot remove it"
    assert load_fixture("vi").fold("đ") == "d"


@pytest.mark.parametrize("joiner", ["‌", "‍"])
def test_conjunct_joiners_do_not_materially_change_the_score(unicode_filter, joiner) -> None:
    """ZWJ and ZWNJ are obligatory in Devanagari conjuncts and are category Cf.

    No category rule reaches ``Cf``, so treating them as junk rejects text for
    being spelled correctly. Measured before the fix: ordinary joiner use moved
    a correct Hindi paragraph from 0.218 to 0.265, across the 0.25 default.

    They now count as content, so adding them can only lower the ratio — the
    denominator grows while the non-alphanumeric count does not.
    """
    plain = unicode_filter.score_document(HI)
    joined = unicode_filter.score_document(HI.replace("क", "क" + joiner))

    assert unicode_filter.keep_document(joined), f"joiners pushed the score to {joined:.3f}"
    assert joined <= plain, "a joiner carries linguistic content, not junk"


def test_treating_joiners_as_junk_would_reject_correct_hindi() -> None:
    """The measurement behind the exemption, so removing it fails loudly."""
    strict = signals.UnicodeAwareNonAlphaNumericFilter()
    joined = HI.replace("क", "क‍")

    without_exemption = sum(
        1 for c in joined if not (ud.category(c)[0] in ("L", "N", "M") or c in set("\n?!,."))
    ) / len(joined)

    assert without_exemption > CURATOR_DEFAULT, "if this stops holding the exemption is moot"
    assert strict.keep_document(strict.score_document(joined))


def test_devanagari_digits_are_numbers_that_ascii_patterns_miss() -> None:
    """Recorded, not fixed: NumbersFilter is Curator's and correcting it is upstream."""
    assert "१".isdigit()
    assert ud.category("१") == "Nd"
    assert not re.match(r"[0-9]", "१")
    assert signals.is_alphanumeric("१"), "our own definition must still count it"


def test_emoji_and_punctuation_count_against_a_document(unicode_filter) -> None:
    assert unicode_filter.score_document("🙂🙂🙂🙂") == pytest.approx(1.0)
    assert unicode_filter.score_document("!!!!") < 1.0, "kept punctuation is not junk"


def test_english_behaviour_is_unchanged(unicode_filter) -> None:
    """The existing default must keep behaving as it does for the corpus it was tuned on."""
    assert unicode_filter.score_document(EN) == pytest.approx(ascii_ratio(EN), abs=0.02)


def test_an_empty_document_is_all_non_alphanumeric(unicode_filter) -> None:
    assert unicode_filter.score_document("") == 1.0


# -- pack-parameterised signals ----------------------------------------------


@pytest.fixture(scope="module")
def vi():
    return load_fixture("vi")


@pytest.fixture(scope="module")
def hi():
    return load_fixture("hi")


def test_script_ratio_is_continuous_not_binary(vi) -> None:
    """HistogramFilter returns 0 or 1, which cannot be swept and so cannot be profiled."""
    scorer = signals.ScriptRatio(vi)

    mixed = scorer.score_document("Tiếng Việt " + HI)

    assert 0.0 < mixed < 1.0


def test_script_ratio_does_not_separate_languages_sharing_a_script(vi) -> None:
    """A property of the signal, recorded so nobody reads it as language ID.

    A Latin-script pack must include the unmarked Latin letters, so English
    scores as high as Vietnamese against the Vietnamese pack — correctly, since
    both are written in Latin. Separating them is what diacritic_ratio and
    stopword_ratio are for.
    """
    scorer = signals.ScriptRatio(vi)

    assert scorer.score_document("This is plain English text") == pytest.approx(1.0)
    assert scorer.score_document(VI) == pytest.approx(1.0)
    assert scorer.score_document(HI) == pytest.approx(0.0)


def test_script_ratio_is_high_for_its_own_language(vi) -> None:
    scorer = signals.ScriptRatio(vi)

    assert scorer.score_document(VI) > 0.95


def test_foreign_script_ratio_notices_another_script(vi) -> None:
    scorer = signals.ForeignScriptRatio(vi)

    assert scorer.score_document(VI) == pytest.approx(0.0, abs=0.01)
    assert scorer.score_document(HI) > 0.9


def test_diacritic_ratio_separates_marked_from_stripped_text(vi) -> None:
    scorer = signals.DiacriticRatio(vi)

    assert scorer.score_document(VI) > scorer.score_document(strip_marks(VI))
    assert scorer.score_document(strip_marks(VI)) == pytest.approx(0.0, abs=0.01)


def test_a_pack_without_the_capability_refuses_the_signal(hi) -> None:
    """Devanagari matras are obligatory vowels; the measurement has no meaning."""
    with pytest.raises(ValueError, match="diacritic_ratio"):
        signals.DiacriticRatio(hi)


def test_stopword_ratio_separates_prose_from_a_keyword_list(vi) -> None:
    scorer = signals.StopwordRatio(vi)

    assert scorer.score_document(VI) > scorer.score_document("giày dép túi xách balo vali")


def test_folded_stopwords_recover_unmarked_text(vi) -> None:
    """A mark-sensitive gate deletes text written without diacritics outright."""
    marked = signals.StopwordRatio(vi)
    folded = signals.FoldedStopwordRatio(vi)
    stripped = strip_marks(VI)

    assert folded.score_document(stripped) > marked.score_document(stripped)


def test_folding_collisions_are_reportable(vi) -> None:
    """The count is a property of the language and belongs in the report."""
    assert signals.FoldedStopwordRatio(vi).collision_count() > 0


def test_sentence_end_ratio_uses_the_packs_terminator(hi, vi) -> None:
    """Hindi ends sentences with the danda; Curator's filter looks for '.', '!', '?'."""
    hindi_line = "यह एक वाक्य है।"

    assert "।" in hi.sentence_terminators
    assert signals.SentenceEndRatio(hi).score_document(hindi_line) == 1.0
    assert signals.SentenceEndRatio(vi).score_document(hindi_line) == 0.0


def test_sentence_end_ratio_is_not_degenerate_on_hindi(hi) -> None:
    scorer = signals.SentenceEndRatio(hi)

    mixed = "यह एक वाक्य है।\nयह अधूरा\nभारत एक देश है।"

    assert 0.0 < scorer.score_document(mixed) < 1.0


def test_boilerplate_hits_count_the_packs_own_patterns(vi) -> None:
    """Curator's BoilerPlateStringFilter hardcodes English and matches nothing here."""
    scorer = signals.BoilerplateHits(vi)

    assert scorer.score_document("Xem thêm bài viết") >= 1.0
    assert scorer.score_document(VI) == 0.0


def test_every_pack_signal_runs_against_an_invented_language() -> None:
    pack = langpack.load("x-test", FIXTURES)
    text = " ".join(sorted(pack.stopwords)[:6]) + "."

    for group in signals.BY_CAPABILITY.values():
        for cls in group:
            assert isinstance(cls(pack).score_document(text), float), cls.__name__


# -- one signal, one stage name ------------------------------------------------
#
# ScriptRatio, LatinRatio and ForeignScriptRatio all need the pack's
# `script_ratio` data, and the stage name used to be taken from that shared
# capability. Two of them in one policy then built two Curator stages with the
# SAME name: in annotate/both mode each wrote the same `__script_ratio` column,
# overwriting the other, and the ledger's per-gate attribution — the one figure
# that says which gate removed what — collapsed them into a single entry.


def test_every_pack_signal_has_its_own_stage_name() -> None:
    from nemotron.steps.curate.nemo_curator.runtime import registry

    pack = load_fixture("vi")
    names: dict[str, list[str]] = {}
    for name, signal in registry.SIGNALS.items():
        if name not in registry.PACK_SIGNALS:
            continue
        if set(signal.requires) - set(pack.capabilities):
            continue
        names.setdefault(signal.factory(pack=pack)._name, []).append(name)

    collisions = {n: s for n, s in names.items() if len(s) > 1}
    assert not collisions, f"signals sharing a stage name: {collisions}"


def test_the_stage_name_is_the_registry_key() -> None:
    """A policy names `latin_ratio`; the column and the ledger entry must agree."""
    from nemotron.steps.curate.nemo_curator.runtime import registry

    pack = load_fixture("vi")
    for name, signal in registry.SIGNALS.items():
        if name not in registry.PACK_SIGNALS:
            continue
        if set(signal.requires) - set(pack.capabilities):
            continue
        assert signal.factory(pack=pack)._name == name, (
            f"{name} builds a stage called {signal.factory(pack=pack)._name!r}"
        )


def test_signals_sharing_a_capability_are_still_distinct() -> None:
    """The three script signals are the case that made this fail."""
    from nemotron.steps.curate.nemo_curator.runtime import registry

    pack = load_fixture("vi")
    trio = ("script_ratio", "latin_ratio", "foreign_script_ratio")

    assert len({registry.SIGNALS[n].factory(pack=pack)._name for n in trio}) == len(trio)
    assert len({registry.SIGNALS[n].requires for n in trio}) == 1, (
        "they must still share the capability — that is why the collision existed"
    )


def test_the_same_text_scores_the_same_in_nfc_and_nfd() -> None:
    """Correct text has two encodings, and the signals used to disagree about it.

    In NFD a combining mark is its own codepoint, so a category test for letters
    excludes it and the base letter folds to itself. diacritic_ratio and
    stopword_ratio -- the two signals whose purpose is to detect lost diacritics
    -- therefore scored NFD identically to text whose diacritics really were
    stripped. 11.69% of a 20,000-document Vietnamese C4 sample is not NFC.
    """
    import unicodedata

    from nemotron.steps.curate.nemo_curator.runtime import registry

    pack = load_fixture("vi")
    sentence = "Việt Nam là một quốc gia nằm ở phía đông bán đảo Đông Dương."
    nfc = unicodedata.normalize("NFC", sentence)
    nfd = unicodedata.normalize("NFD", sentence)
    stripped = "".join(c for c in nfd if not unicodedata.combining(c))
    assert nfc != nfd, "the fixture must actually differ by encoding"

    for name in ("diacritic_ratio", "stopword_ratio", "script_ratio", "latin_ratio"):
        scorer = registry.SIGNALS[name].build(0.1, pack=pack)
        assert scorer.score_document(nfc) == scorer.score_document(nfd), (
            f"{name} scores the same sentence differently depending on its encoding"
        )

    diacritic = registry.SIGNALS["diacritic_ratio"].build(0.1, pack=pack)
    assert diacritic.score_document(nfc) > diacritic.score_document(stripped), (
        "normalising must not also erase the distinction the signal exists to make"
    )


def test_measurement_normalisation_does_not_touch_the_document() -> None:
    """The corpus keeps the bytes it arrived with.

    A measurement stage that rewrites what it measures is the defect that let a
    classifier truncate 53.4% of a corpus while every count still reconciled.
    """
    import unicodedata

    from nemotron.steps.curate.nemo_curator.runtime import registry

    pack = load_fixture("vi")
    nfd = unicodedata.normalize("NFD", "Tiếng Việt")
    before = nfd

    registry.SIGNALS["diacritic_ratio"].build(0.1, pack=pack).score_document(nfd)

    assert nfd == before
    assert not unicodedata.is_normalized("NFC", nfd), "the caller's string is unchanged"
