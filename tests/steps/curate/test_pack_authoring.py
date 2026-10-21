# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The pack-authoring documentation, pinned against the loader it describes.

SPEC.md told authors to record a declined-capability measurement "in a key
beside `supports`". The loader reads `capabilities.supports` and nothing else
from that table, so the measurement went nowhere: a documented practice that
silently did nothing. It also described `charset.txt` as "one character per
line" without saying that the file is parsed as the union of every character on
every line, so an author writing `a-z` got three characters and no warning.

These tests pin both facts against the real loader, and pin that the onboarding
document covers the steps the reviewer asked for.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from nemotron.steps.curate.nemo_curator.runtime import langpack

FIXTURES = Path(__file__).parent / "fixtures" / "langpacks"
RUNTIME = Path(langpack.__file__).parent
PACKS = RUNTIME.parent / "data" / "langpacks"
ONBOARDING = PACKS / "ADDING_A_LANGUAGE.md"
SPEC = PACKS / "SPEC.md"


def _pack_manifests():
    for root in (PACKS, FIXTURES):
        yield from sorted(root.glob("*/pack.toml"))


# -- the loader's real contract ------------------------------------------------


def test_a_charset_range_is_not_expanded(tmp_path) -> None:
    """`a-z` is three characters, not twenty-six. SPEC.md now says so."""
    import shutil

    pack = tmp_path / "x-test-en"
    shutil.copytree(FIXTURES / "x-test-en", pack)
    (pack / "charset.txt").write_text("a-z\n", encoding="utf-8")

    assert langpack.load_pack(pack).charset == {"a", "-", "z"}


def test_several_characters_on_one_line_all_load(tmp_path) -> None:
    """One per line is the convention, not the rule -- so the doc must not imply it is."""
    import shutil

    pack = tmp_path / "x-test-en"
    shutil.copytree(FIXTURES / "x-test-en", pack)
    (pack / "charset.txt").write_text("AB\nC\n", encoding="utf-8")

    assert langpack.load_pack(pack).charset == {"A", "B", "C"}


def test_no_pack_records_a_note_beside_supports() -> None:
    """The practice SPEC.md used to recommend, which the loader ignores.

    `capabilities` is read for `supports` alone. A sibling key there is dropped
    without a word, so a pack that followed the old wording would carry a
    measurement no report ever shows.
    """
    offenders = []
    for manifest in _pack_manifests():
        with manifest.open("rb") as fh:
            raw = tomllib.load(fh)
        extra = set(raw.get("capabilities", {})) - {"supports"}
        if extra:
            offenders.append(f"{manifest.parent.name}: {sorted(extra)}")

    assert not offenders, f"keys under [capabilities] are silently ignored; use [notes]: {offenders}"


def test_a_note_beside_supports_reaches_no_report(tmp_path) -> None:
    """Proves the ignoring, so the SPEC correction cannot be quietly reverted."""
    import shutil

    pack = tmp_path / "x-test-en"
    shutil.copytree(FIXTURES / "x-test-en", pack)
    manifest = pack / "pack.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            "[capabilities]", '[capabilities]\nstopword_ratio_not_declared = "measured"'
        ),
        encoding="utf-8",
    )

    described = langpack.load_pack(pack).describe()

    assert not any("measured" in str(v) for v in described.values())


def test_the_notes_table_is_what_actually_reaches_the_report(tmp_path) -> None:
    """The correct home, so the doc points somewhere that works."""
    import shutil

    pack = tmp_path / "x-test-en"
    shutil.copytree(FIXTURES / "x-test-en", pack)
    manifest = pack / "pack.toml"
    # The fixture already declares [notes]; a second table is a TOML error, so
    # the key goes into the existing one.
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace("[notes]", '[notes]\nstopword_ratio_not_declared = "measured"'),
        encoding="utf-8",
    )

    described = langpack.load_pack(pack).describe()

    assert described["notes"]["stopword_ratio_not_declared"] == "measured"


# -- the onboarding document ---------------------------------------------------


def test_the_onboarding_document_exists() -> None:
    assert ONBOARDING.is_file(), "ADDING_A_LANGUAGE.md is the reviewer's named deliverable"


@pytest.mark.parametrize(
    "topic,needle",
    [
        ("pack layout", "## 1."),
        ("required vs optional files", "## 2."),
        ("character format", "ranges are not expanded"),
        ("fasttext model setup", "lid.176.bin"),
        ("validation", "langpack.load("),
        ("profiling", "run_profile"),
        ("threshold approval", "## 7."),
        ("evaluation", "false rejection"),
        ("activation", "## 9."),
    ],
)
def test_the_onboarding_document_covers_every_named_topic(topic, needle) -> None:
    """The nine topics the reviewer enumerated. A missing one is a missing step."""
    assert needle.lower() in ONBOARDING.read_text(encoding="utf-8").lower(), f"{topic} is not covered"


def test_the_fasttext_model_is_named_with_a_reachable_source() -> None:
    """The prerequisite that blocked a first-time user in the first five minutes."""
    text = ONBOARDING.read_text(encoding="utf-8")

    assert "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin" in text
    assert "176" in text, "the language-coverage caveat must survive"


def test_the_two_language_settings_are_distinguished() -> None:
    """corpus.language selects a pack; language_codes decides what is kept.

    Conflating them produces a corpus scored by one language's rules and
    filtered to another's, with every row count reconciling.
    """
    text = ONBOARDING.read_text(encoding="utf-8")

    assert "corpus.language" in text
    assert "language_codes" in text


def test_the_spec_no_longer_recommends_the_ignored_key() -> None:
    text = SPEC.read_text(encoding="utf-8")

    assert "in a key beside `supports`" not in text, "the ignored practice is back"
    assert "[notes]" in text
    assert "Ranges are not expanded" in text


def test_the_readmes_route_a_reader_to_the_procedure() -> None:
    """SPEC.md had three inbound links and none from the entry-point README."""
    for readme in (
        RUNTIME.parent.parent / "README.md",
        RUNTIME.parent / "README.md",
        RUNTIME.parent / "profile" / "README.md",
    ):
        assert "ADDING_A_LANGUAGE.md" in readme.read_text(encoding="utf-8"), readme


def test_the_onboarding_document_no_longer_claims_nothing_is_automated() -> None:
    """The sentence the reviewer quoted. Step 8 is a command now, so the claim
    that onboarding is entirely manual is false as well as discouraging."""
    text = ONBOARDING.read_text(encoding="utf-8")

    assert "Nothing here is automated" not in text
    assert "run_evaluate" in text, "step 8 must name the command that computes the rates"
    assert "false rejection" in text
    assert "noise removal" in text
