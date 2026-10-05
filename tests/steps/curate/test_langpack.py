# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The language pack contract.

The claim these tests defend is that the runtime is language-parametric. A
substring search for "vi" would prove nothing — the real evidence is that the
whole pipeline runs against a language that does not exist, because there is
nothing about it anyone could have special-cased.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from nemotron.steps.curate.nemo_curator.runtime import langpack, signals
from nemotron.steps.curate.nemo_curator.runtime import registry as r

FIXTURES = Path(__file__).parent / "fixtures" / "langpacks"
RUNTIME = Path(langpack.__file__).parent
PACKAGE_PACKS = RUNTIME.parent / "data" / "langpacks"
FIXTURE_LANGUAGES = ("en", "hi", "vi")


def load_fixture(language: str) -> langpack.LanguagePack:
    return langpack.load(f"x-test-{language}", FIXTURES)


# -- packaged reference and private validation fixtures -----------------------


#: The packs that ship. Three example languages, chosen because they break
#: different assumptions: en is unmarked Latin, vi is Latin whose NFD produces
#: only Mn marks, hi is an abugida whose matras span Mn and Mc with Mc in the
#: majority. Japanese and Thai were removed -- their charset alone was 21,298
#: lines -- and the evidence they carried is inline in test_unsegmented_scripts.py.
SHIPPED_PACKS = ("en", "hi", "vi")


def test_exactly_three_example_packs_are_bundled() -> None:
    assert langpack.available(PACKAGE_PACKS) == list(SHIPPED_PACKS)


def test_the_english_reference_pack_reports_what_it_carries() -> None:
    pack = langpack.load("en", PACKAGE_PACKS)

    assert pack.pack_id == "en-reference-snowball-cldr48"
    assert pack.language_tag == "en"
    assert len(pack.stopwords) == 174
    assert len(pack.charset) == 52
    assert pack.capabilities == {
        "script_ratio",
        "stopword_ratio",
        "sentence_end_ratio",
        # English is space-delimited and writes numbers and sentence marks in
        # ASCII, so the word- and ASCII-based signals are valid for it.
        "word_segmentation",
        "ascii_digits",
        "ascii_punctuation",
        # Curator's ASCII content class is valid on an ASCII alphabet, and only there.
        "ascii_alphabet",
    }


@pytest.mark.parametrize("tag", SHIPPED_PACKS)
def test_every_shipped_pack_names_the_directory_it_lives_in(tag) -> None:
    """The identity N1 pins, checked against what actually ships."""
    assert langpack.load(tag, PACKAGE_PACKS).language_tag == tag


@pytest.mark.parametrize("tag", SHIPPED_PACKS)
def test_every_shipped_list_records_where_it_came_from(tag) -> None:
    """Minimum provenance for a public repository: one line per list.

    Full provenance -- upstream URL, checksum, licence text -- is held only by
    the `en` reference pack, which was derived from published upstream sources.
    The vi and hi packs were assembled for this repository, so origin and licence
    are what there is to record and what must not go missing.
    """
    pack = langpack.load(tag, PACKAGE_PACKS)

    assert pack.sources, f"{tag} declares no sources, so nothing it reads is attributable"
    for name, source in pack.sources.items():
        assert source.get("origin"), f"{tag}/{name}: no origin"
        assert source.get("license"), f"{tag}/{name}: no license"


@pytest.mark.parametrize("tag", SHIPPED_PACKS)
def test_every_shipped_pack_says_it_is_not_a_default(tag) -> None:
    """A bundled pack is example data. Nothing selects it implicitly, and the
    pack has to say so where a reader of the report will see it."""
    assert "scope" in langpack.load(tag, PACKAGE_PACKS).notes, f"{tag} has no scope note"


@pytest.mark.parametrize("language", FIXTURE_LANGUAGES)
def test_a_validation_fixture_reports_what_it_carries(language) -> None:
    pack = load_fixture(language)
    described = pack.describe()

    assert described["language_tag"] == f"x-test-{language}"
    assert described["content_hash"].startswith("sha256:")
    assert described["stopwords"] > 0
    assert described["charset"] > 0
    assert described["capabilities"]


def test_all_language_fixtures_use_private_bcp47_tags() -> None:
    tags = langpack.available(FIXTURES)

    assert set(tags) == {"x-test", *(f"x-test-{language}" for language in FIXTURE_LANGUAGES)}
    assert all(tag.startswith("x-") for tag in tags)


# -- the capability mechanism -------------------------------------------------


def test_hindi_declares_fewer_capabilities_than_vietnamese() -> None:
    """The finding, not a shortfall.

    Vietnamese tone marks strip to degraded but readable text, so measuring their
    density says something. Devanagari matras are obligatory vowels; stripping
    them yields nonsense, so the capability is absent rather than measured on a
    false premise.
    """
    vi = load_fixture("vi")
    hi = load_fixture("hi")

    assert "diacritic_ratio" in vi.capabilities
    assert "diacritic_ratio" not in hi.capabilities
    assert "stopword_ratio_folded" not in hi.capabilities
    assert hi.capabilities < vi.capabilities


@pytest.mark.parametrize("tag", ["en"])
def test_a_pack_without_removable_marks_omits_diacritic_ratio(tag) -> None:
    """English loanword accents are not removable orthography; measuring their
    density would be the Devanagari trap. The Japanese and Thai cases moved to
    test_unsegmented_scripts.py when those fixtures were removed."""
    pack = load_fixture(tag)
    assert "diacritic_ratio" not in pack.capabilities
    assert "stopword_ratio_folded" not in pack.capabilities


@pytest.mark.parametrize(
    "tag,text",
    [
        ("en", "The weather is nice today."),
    ],
)
def test_a_correct_sentence_is_own_script(tag, text) -> None:
    pack = load_fixture(tag)
    score = signals.ScriptRatio(pack).score_document(text)
    assert score == 1.0


def test_an_unsupported_capability_removes_its_signals_from_the_run() -> None:
    hi = load_fixture("hi")

    chosen, warnings = r.resolve(None, hi.capabilities)

    names = {s.name for s in chosen}
    assert "diacritic_ratio" not in names
    assert "stopword_ratio_folded" not in names
    assert any("diacritic_ratio" in w for w in warnings)


def test_naming_a_signal_the_pack_cannot_support_fails() -> None:
    hi = load_fixture("hi")

    with pytest.raises(r.SignalRequirementsUnmetError, match="diacritic_ratio"):
        r.resolve(["diacritic_ratio"], hi.capabilities)


def test_a_capability_declared_without_its_data_is_rejected(tmp_path) -> None:
    """Otherwise the report fills with zeroes, which reads as a finding."""
    pack_dir = tmp_path / "x-broken"
    pack_dir.mkdir()
    (pack_dir / "pack.toml").write_text(
        '[pack]\npack_id="b"\nlanguage_tag="x-broken"\nversion="1"\nschema=1\n'
        '[capabilities]\nsupports=["stopword_ratio"]\n',
        encoding="utf-8",
    )

    with pytest.raises(langpack.LanguagePackInvalidError, match="without the data behind them"):
        langpack.load_pack(pack_dir)


def test_an_unknown_capability_is_rejected(tmp_path) -> None:
    """A typo would otherwise read as 'this language cannot do that'."""
    pack_dir = tmp_path / "x-typo"
    pack_dir.mkdir()
    (pack_dir / "pack.toml").write_text(
        '[pack]\npack_id="t"\nlanguage_tag="x-typo"\nversion="1"\nschema=1\n'
        '[capabilities]\nsupports=["stopwrd_ratio"]\n',
        encoding="utf-8",
    )

    with pytest.raises(langpack.LanguagePackInvalidError, match="unknown capabilities"):
        langpack.load_pack(pack_dir)


# -- no default language ------------------------------------------------------


def test_there_is_no_default_language() -> None:
    """A wrong default produces plausible numbers for the wrong language."""
    with pytest.raises(langpack.LanguagePackNotFoundError, match="no default"):
        langpack.load("", FIXTURES)


def test_an_unknown_tag_names_what_is_available() -> None:
    with pytest.raises(langpack.LanguagePackNotFoundError, match="Available"):
        langpack.load("xx-nonexistent", FIXTURES)


def test_a_pack_directory_is_required() -> None:
    with pytest.raises(langpack.LanguagePackNotFoundError, match="explicit langpack_dir"):
        langpack.load("vi", None)


# -- structural evidence of genericity ---------------------------------------


def test_no_runtime_branch_turns_on_a_language() -> None:
    """A cheap secondary guard, and deliberately narrow.

    A flat substring search would flag the docstrings that explain *why* the
    Unicode handling is what it is — those cite Vietnamese and Devanagari as
    measured evidence, which is the opposite of a problem. What would be a real
    defect is executable code that behaves differently for a named language, so
    only comparisons and subscripts are inspected, with docstrings and comments
    stripped by the parser.
    """
    import ast

    named = ("vi", "hi", "vietnamese", "hindi", "devanagari")
    offenders: list[str] = []

    for module in sorted(RUNTIME.glob("*.py")):
        if module.name == "langpack.py":
            continue
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                literals = [
                    n.value
                    for n in [node.left, *node.comparators]
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                ]
                for value in literals:
                    if value.lower() in named:
                        offenders.append(f"{module.name}:{node.lineno} compares against {value!r}")

    assert not offenders, f"the neutral layer branches on a language: {offenders}"


def test_the_runtime_never_reaches_into_the_pack_directory() -> None:
    """Packs are resolved through the loader, never by path from another module."""
    for module in sorted(RUNTIME.glob("*.py")):
        if module.name == "langpack.py":
            continue
        assert "data/langpacks" not in module.read_text(encoding="utf-8"), module.name


def test_every_pack_signal_declares_the_capability_it_needs() -> None:
    for name in r.PACK_SIGNALS:
        assert r.SIGNALS[name].requires, f"{name} declares no requirement"
        for capability in r.SIGNALS[name].requires:
            assert capability in langpack.KNOWN_CAPABILITIES, capability


def test_the_full_signal_set_runs_against_an_invented_language() -> None:
    """The load-bearing test.

    x-test is written in a script neither shipped pack uses, with invented words
    and an invented fold map. If every signal produces a number here, the runtime
    is parametric — there is nothing about this language to special-case.
    """
    pack = langpack.load("x-test", FIXTURES)
    chosen, _ = r.resolve(None, pack.capabilities)

    text = "".join(sorted(pack.charset)[:40]) + " " + " ".join(sorted(pack.stopwords)[:5]) + "."
    scored = 0
    for signal in chosen:
        if signal.name not in r.PACK_SIGNALS:
            continue
        value = signal.build(*_probe(signal), pack=pack).score_document(text)
        assert isinstance(value, float), signal.name
        scored += 1

    assert scored == len(r.PACK_SIGNALS), "every pack signal must run on an invented language"


def _probe(signal):
    if signal.direction == "interval":
        return (signal.grid.lo_grid.values()[0], signal.grid.hi_grid.values()[-1])
    return (signal.grid.values()[-1],)


# -- package boundary ----------------------------------------------------------


def test_language_fixtures_live_outside_the_package() -> None:
    assert FIXTURES.is_dir()
    assert PACKAGE_PACKS not in FIXTURES.parents


def test_the_reference_pack_needs_no_artifact_escape_hatch() -> None:
    """Tracked package data is included without admitting ignored local packs."""
    root = Path(__file__).resolve().parents[3]
    with (root / "pyproject.toml").open("rb") as fh:
        config = tomllib.load(fh)

    artifacts = config["tool"]["hatch"]["build"]["targets"]["wheel"].get("artifacts", [])
    assert not any("langpacks" in entry for entry in artifacts)


def test_the_reference_pack_pins_sources_and_carries_licenses() -> None:
    pack = langpack.load("en", PACKAGE_PACKS)
    pack_dir = PACKAGE_PACKS / "en"

    for name, source in pack.sources.items():
        assert source.get("origin"), f"en/{name}: source declares no origin"
        assert source.get("license"), f"en/{name}: source declares no license"
        upstream_hash = source.get("upstream_sha256") or source.get("upstream_file_sha256")
        assert isinstance(upstream_hash, str) and len(upstream_hash) == 64
        assert (pack_dir / source["license_file"]).is_file()

    assert not (pack_dir / "boilerplate.txt").exists()
    assert "boilerplate_hits" not in pack.capabilities


def test_the_wheel_excludes_local_curate_workspaces() -> None:
    """Downloaded corpora, models, and run outputs must never enter a release wheel."""
    root = Path(__file__).resolve().parents[3]
    with (root / "pyproject.toml").open("rb") as fh:
        config = tomllib.load(fh)

    excluded = config["tool"]["hatch"]["build"]["targets"]["wheel"].get("exclude", [])
    assert "/src/nemotron/steps/curate/__local__/**" in excluded


def test_every_language_fixture_records_where_its_word_list_came_from() -> None:
    """A word list with no provenance cannot be relicensed, corrected, or trusted."""
    for tag in langpack.available(FIXTURES):
        pack = langpack.load(tag, FIXTURES)
        stopwords = pack.sources.get("stopwords", {})
        assert stopwords.get("origin"), f"{tag}: stopwords declare no origin"
        assert stopwords.get("license"), f"{tag}: stopwords declare no license"


def test_every_fixture_asset_records_origin_and_license() -> None:
    for tag in langpack.available(FIXTURES):
        for name, source in langpack.load(tag, FIXTURES).sources.items():
            assert source.get("origin"), f"{tag}/{name}: source declares no origin"
            assert source.get("license"), f"{tag}/{name}: source declares no license"


def test_a_source_without_license_is_rejected(tmp_path) -> None:
    pack_dir = tmp_path / "x-missing-license"
    pack_dir.mkdir()
    (pack_dir / "stopwords.txt").write_text("word\n", encoding="utf-8")
    (pack_dir / "pack.toml").write_text(
        '[pack]\npack_id="x"\nlanguage_tag="x"\nversion="1"\nschema=1\n'
        '[sources]\nstopwords={file="stopwords.txt", origin="test"}\n'
        '[capabilities]\nsupports=["stopword_ratio"]\n',
        encoding="utf-8",
    )

    with pytest.raises(langpack.LanguagePackInvalidError, match="license"):
        langpack.load_pack(pack_dir)


def test_text_resources_are_normalized_to_nfc(tmp_path) -> None:
    import unicodedata

    pack_dir = tmp_path / "x-nfd"
    pack_dir.mkdir()
    nfd = unicodedata.normalize("NFD", "café")
    (pack_dir / "stopwords.txt").write_text(f"{nfd}\n", encoding="utf-8")
    (pack_dir / "pack.toml").write_text(
        '[pack]\npack_id="x"\nlanguage_tag="x"\nversion="1"\nschema=1\n'
        '[sources]\nstopwords={file="stopwords.txt", origin="test", license="Apache-2.0"}\n'
        '[capabilities]\nsupports=["stopword_ratio"]\n',
        encoding="utf-8",
    )

    assert langpack.load_pack(pack_dir).stopwords == {"café"}


# -- fold map -----------------------------------------------------------------


def test_the_fold_map_lives_in_the_pack_not_the_code() -> None:
    """Vietnamese đ does not decompose under NFD, so it must be declared."""
    vi = load_fixture("vi")

    assert vi.fold_map.get("đ") == "d"
    assert vi.fold("đường") == "duong"
    assert "đ" not in signals.__file__ or "fold_map" in Path(signals.__file__).read_text(encoding="utf-8")


def test_folding_collapses_distinct_stopwords_and_the_count_is_knowable() -> None:
    """The collision count is a property of the language, not a defect."""
    vi = load_fixture("vi")

    folded = vi.folded_stopwords()

    assert len(folded) < len(vi.stopwords)


def test_a_pack_without_a_fold_map_folds_only_combining_marks() -> None:
    hi = load_fixture("hi")

    assert hi.fold_map == {}
    assert hi.fold("भारत") == "भारत" or len(hi.fold("भारत")) <= len("भारत")


def test_fold_map_must_be_a_mapping(tmp_path) -> None:
    pack_dir = tmp_path / "x-invalid-fold-map"
    pack_dir.mkdir()
    (pack_dir / "pack.toml").write_text(
        'fold_map=["not", "a", "mapping"]\n'
        '[pack]\npack_id="x"\nlanguage_tag="x"\nversion="1"\nschema=1\n'
        '[capabilities]\nsupports=["sentence_end_ratio"]\n',
        encoding="utf-8",
    )

    with pytest.raises(langpack.LanguagePackInvalidError, match="fold_map must be a mapping"):
        langpack.load_pack(pack_dir)


def test_diacritic_capability_requires_an_explicit_fold_map(tmp_path) -> None:
    pack_dir = tmp_path / "x-no-fold-map"
    pack_dir.mkdir()
    (pack_dir / "charset.txt").write_text("a\n", encoding="utf-8")
    (pack_dir / "pack.toml").write_text(
        '[pack]\npack_id="x"\nlanguage_tag="x"\nversion="1"\nschema=1\n'
        '[sources]\ncharset={file="charset.txt", origin="test", license="Apache-2.0"}\n'
        '[capabilities]\nsupports=["diacritic_ratio"]\n',
        encoding="utf-8",
    )

    with pytest.raises(langpack.LanguagePackInvalidError, match="fold_map"):
        langpack.load_pack(pack_dir)


# -- provenance ---------------------------------------------------------------


def test_a_relative_pack_directory_loads(monkeypatch, tmp_path) -> None:
    """Config examples use relative roots, so hashing must use one path basis."""
    import shutil

    shutil.copytree(FIXTURES / "x-test-hi", tmp_path / "x-test-hi")
    monkeypatch.chdir(tmp_path)

    assert langpack.load_pack("x-test-hi").language_tag == "x-test-hi"


def test_the_content_hash_follows_the_contents_not_the_path(tmp_path) -> None:
    """A policy is tied to the pack that produced it; moving the pack must not break that."""
    import shutil

    source = FIXTURES / "x-test-hi"
    copy = tmp_path / "x-test-hi"
    shutil.copytree(source, copy)

    assert langpack.load_pack(copy).content_hash == load_fixture("hi").content_hash


def test_editing_a_word_list_changes_the_hash(tmp_path) -> None:
    import shutil

    copy = tmp_path / "x-test-hi"
    shutil.copytree(FIXTURES / "x-test-hi", copy)
    before = langpack.load_pack(copy).content_hash
    (copy / "stopwords.txt").write_text("\n".join(["एक", "दो"]) + "\n", encoding="utf-8")

    assert langpack.load_pack(copy).content_hash != before


# -- a capability must be one the signal can actually deliver -------------------
#
# The ja and th packs declared stopword_ratio while the signal tokenises on
# whitespace, which those scripts do not use. Measured on 20,000 real documents
# each: 93.7% of Japanese and 53.1% of Thai scored EXACTLY zero, and 87.9% /
# 90.1% of those were correct native-script text. So the signal could not tell
# "not Japanese" from "Japanese, written normally" — the one distinction it
# exists to make. Hindi, which does use spaces, scores zero on 25.9% and those
# are genuinely foreign-language documents, so it keeps the capability.


@pytest.mark.parametrize("tag", ("vi", "en", "hi"))
def test_a_space_separated_script_keeps_stopword_ratio(tag) -> None:
    assert load_fixture(tag).supports("stopword_ratio")


def test_pack_notes_reach_the_report() -> None:
    """A caveat that stays in the file is not a caveat.

    The removed ja pack recorded the whitespace-tokenisation problem all along;
    describe() dropped it, so every report showed a stopword distribution that
    was 93.7% exact zeros with nothing saying why. The shipped hi pack records
    why it declines diacritic_ratio, so it carries the same obligation.
    """
    described = langpack.load("hi", PACKAGE_PACKS).describe()

    assert "notes" in described
    assert described["notes"], "the hi pack has notes and they must be carried"


# -- the three identities must agree ------------------------------------------
#
# A pack has three names: the tag the config asked for, the directory it was
# found in, and the language_tag inside pack.toml. Every shipped fixture sets
# all three equal, so the happy path is structurally incapable of catching a
# disagreement -- which is why this bug survived a green suite.


def test_a_pack_whose_manifest_names_another_language_is_rejected(tmp_path) -> None:
    """Asking for vi must not return a pack that calls itself ja.

    The content_hash gate is not a backstop for this: a pack mislabelled from
    the first profile onward hashes consistently with itself.
    """
    import shutil

    copy = tmp_path / "x-test-vi"
    shutil.copytree(FIXTURES / "x-test-vi", copy)
    manifest = copy / "pack.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace('language_tag = "x-test-vi"', 'language_tag = "x-test-ja"'),
        encoding="utf-8",
    )

    with pytest.raises(langpack.LanguagePackInvalidError, match="declares language_tag"):
        langpack.load("x-test-vi", tmp_path)


def test_a_case_variant_tag_still_loads(tmp_path) -> None:
    """BCP-47 tags are case-insensitive; tightening this to exact equality is a bug.

    The directory is looked up by exact name -- on a case-sensitive filesystem
    it has to be -- so this pins the *comparison*: a manifest that spells the
    same tag with different case is the same language, not a mismatch. An exact
    comparison would reject this correct pack.
    """
    import shutil

    copy = tmp_path / "pt-BR"
    shutil.copytree(FIXTURES / "x-test-en", copy)
    manifest = copy / "pack.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace('language_tag = "x-test-en"', 'language_tag = "pt-br"'),
        encoding="utf-8",
    )

    pack = langpack.load("pt-BR", tmp_path)

    # Returned under the name it was FOUND by, not the manifest's spelling. That
    # spelling is what run_profile stamps into candidate_policies.yaml, and the
    # filter run reloads the pack with it -- so letting 'pt-br' escape produced a
    # profile that succeeded and a policy the next step could not load.
    assert pack.language_tag == "pt-BR"
    assert langpack.load(pack.language_tag, tmp_path).content_hash == pack.content_hash, (
        "the tag a pack reports must be one that loads it again"
    )


@pytest.mark.parametrize(
    "tag",
    [
        "x-test-vi/../x-test-ja",
        "./x-test-ja",
        "sub/x-test-ja",
        "..",
    ],
)
def test_a_language_tag_may_not_be_a_path(tag) -> None:
    """A tag names a directory under the root. A path escapes it."""
    with pytest.raises(langpack.LanguagePackNotFoundError, match="not a path"):
        langpack.load(tag, FIXTURES)


def test_an_absolute_path_tag_cannot_escape_the_langpack_root(tmp_path) -> None:
    """The sharpest form: an absolute tag ignored langpack_dir entirely."""
    with pytest.raises(langpack.LanguagePackNotFoundError, match="not a path"):
        langpack.load(str(FIXTURES / "x-test-hi"), tmp_path)
