# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Scripts the shipped packs do not cover, proved without shipping a pack for them.

The Japanese and Thai validation fixtures were removed: the Japanese charset
alone was 21,298 lines, and three packs ship (en, vi, hi). The evidence those
fixtures carried must not go with them, because it is the evidence that the
runtime is script-agnostic rather than tuned for the languages it happens to
bundle. Every case below is built from a few dozen inline characters and proves
exactly what the 21k-line pack proved:

* an unsegmented script (CJK, Thai) has no whitespace, so a word-based signal
  measures one "word" for a whole sentence -- which is why such a pack must
  decline `stopword_ratio` rather than report a distribution over nothing;
* a correct sentence in a non-Latin script scores as its own script;
* declining a capability is recorded, not silent.

The Devanagari Mn/Mc trap and the Vietnamese NFC/NFD pair are the other half of
this evidence and live in test_signals.py, which never depended on the removed
fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nemotron.steps.curate.nemo_curator.runtime import langpack, signals

#: A correct sentence in each unsegmented script, and the characters that spell
#: it. That is the whole pack: a charset is a set of characters, so the fixture
#: for "is this text in this script" needs only the characters under test.
UNSEGMENTED = {
    "ja": "今日は良い天気です。",
    "th": "วันนี้อากาศดีมาก",
}


def _inline_pack(tmp_path: Path, tag: str, text: str, *, supports: tuple[str, ...], note: str = "") -> Path:
    """A minimal pack whose charset is exactly the characters of `text`."""
    pack = tmp_path / tag
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "charset.txt").write_text("\n".join(sorted(set(text) - {" ", "\n"})) + "\n", encoding="utf-8")
    notes = f'\n[notes]\nstopword_ratio_not_declared = "{note}"\n' if note else ""
    # A file present in the directory is NOT read unless [sources] declares it,
    # with file/origin/license all set. Omitting the table leaves charset empty
    # and the pack is refused for "capabilities declared without the data behind
    # them" -- which names the capability but not the missing declaration.
    (pack / "pack.toml").write_text(
        f'[pack]\npack_id = "{tag}"\nlanguage_tag = "{tag}"\nversion = "1"\nschema = 1\n\n'
        "[sources]\n"
        f'charset = {{ file = "charset.txt", origin = "inline test fixture", license = "CC0-1.0" }}\n\n'
        f"[capabilities]\nsupports = {list(supports)!r}\n".replace("'", '"')
        + notes,
        encoding="utf-8",
    )
    return pack


@pytest.mark.parametrize("tag", sorted(UNSEGMENTED))
def test_an_unsegmented_script_has_no_word_boundaries_to_count(tag) -> None:
    """Why a word-based signal cannot work here, stated as a property of the text.

    This is the measurement behind the pack decision: whitespace tokenisation
    sees one token for an entire sentence, so a stopword ratio over it is a
    clean-looking distribution over nothing.
    """
    text = UNSEGMENTED[tag]

    assert " " not in text, f"{tag} sample must be genuinely unsegmented"
    assert len(text.split()) == 1, f"{tag}: whitespace tokenisation found {len(text.split())} words"
    assert len(text) > 8, "a whole sentence, not a single word"


@pytest.mark.parametrize("tag", sorted(UNSEGMENTED))
def test_a_correct_sentence_scores_as_its_own_script(tmp_path, tag) -> None:
    """ScriptRatio is charset-driven, so a pack for the script accepts it fully."""
    text = UNSEGMENTED[tag]
    pack = langpack.load_pack(_inline_pack(tmp_path, tag, text, supports=("script_ratio",)))

    assert signals.ScriptRatio(pack).score_document(text) == 1.0


@pytest.mark.parametrize("tag", sorted(UNSEGMENTED))
def test_such_a_pack_declines_stopword_ratio_and_records_why(tmp_path, tag) -> None:
    """The decision the removed packs encoded, and the record that made it legible.

    A capability absent with no note reads as an oversight; the point is that it
    is a measurement someone made.
    """
    pack = langpack.load_pack(
        _inline_pack(
            tmp_path,
            tag,
            UNSEGMENTED[tag],
            supports=("script_ratio",),
            note="whitespace tokenisation finds one token per sentence in this script",
        )
    )

    assert not pack.supports("stopword_ratio")
    assert not pack.supports("stopword_ratio_folded")
    assert "stopword_ratio_not_declared" in pack.notes
    assert pack.describe()["notes"]["stopword_ratio_not_declared"], "the note must reach the report"


@pytest.mark.parametrize("tag", sorted(UNSEGMENTED))
def test_such_a_pack_declines_diacritic_ratio(tmp_path, tag) -> None:
    """Japanese dakuten and Thai tone marks are not removable orthography.

    Measuring their density would be the Devanagari trap in another script.
    """
    pack = langpack.load_pack(_inline_pack(tmp_path, tag, UNSEGMENTED[tag], supports=("script_ratio",)))

    assert "diacritic_ratio" not in pack.capabilities


def test_a_shipped_pack_for_a_segmented_script_still_declares_stopword_ratio() -> None:
    """The contrast that makes the rule a rule rather than a blanket exclusion."""
    packs = Path(langpack.__file__).parent.parent / "data" / "langpacks"

    for tag in ("en", "vi", "hi"):
        assert langpack.load(tag, packs).supports("stopword_ratio"), tag
