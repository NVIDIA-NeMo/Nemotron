# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""G7: near-duplicate decontamination, and the contamination it cannot see.

The most likely overclaim in this area is "holdout verified clean". These tests
pin the boundary as a measurement: whole-document Jaccard is shown *failing* to
notice a benchmark question embedded in a long training document, so nobody can
later read the step as offering that guarantee.
"""

from __future__ import annotations

import pytest

from nemotron.steps.curate.nemo_curator.runtime import decon, grouping

QUESTION = "What is the capital city of the Socialist Republic of Vietnam"
FILLER = " ".join(f"unrelated sentence number {i} about other matters" for i in range(120))


# -- shingling and similarity -------------------------------------------------


def test_identical_text_is_similarity_one() -> None:
    a = decon.shingles("the quick brown fox jumps over the lazy dog today", 5, kind="word")

    assert decon.jaccard(a, a) == 1.0


def test_disjoint_text_is_similarity_zero() -> None:
    a = decon.shingles("alpha beta gamma delta epsilon zeta", 5, kind="word")
    b = decon.shingles("one two three four five six", 5, kind="word")

    assert decon.jaccard(a, b) == 0.0


def test_a_document_shorter_than_a_shingle_yields_nothing() -> None:
    """Not similarity zero — an unanswerable comparison, which callers must see."""
    assert decon.shingles("only four words here", size=5, kind="word") == set()
    assert decon.shingles("short", size=24, kind="char") == set()


def test_the_default_shingling_matches_curators_candidate_generator() -> None:
    """Verifying at a different granularity than LSH proposed at means nothing.

    Curator's FuzzyDeduplicationWorkflow defaults to character 24-grams. An
    exact Jaccard computed over word 5-grams is a similarity over a different
    set than the one MinHash approximated, so the threshold no longer refers to
    the quantity the candidates were selected by.
    """
    assert decon.DEFAULT_SHINGLE_KIND == "char"
    assert decon.DEFAULT_SHINGLE_SIZE == 24


def test_the_two_granularities_do_not_agree() -> None:
    """If they always agreed, mixing them would be harmless.

    Checked over a family of edits rather than one pair — a single example can
    coincide, and a test that relied on a coincidence would be evidence of
    nothing.
    """
    base = " ".join(f"sentence {i} of a document about various unrelated topics" for i in range(12))
    edits = [
        base.replace("sentence 3", "paragraph three"),
        base.replace("various", "many"),
        base + " and one appended trailing clause at the very end here",
        base[: len(base) // 2],
    ]

    disagreements = [
        abs(
            decon.jaccard(decon.shingles(base, 24, kind="char"), decon.shingles(e, 24, kind="char"))
            - decon.jaccard(decon.shingles(base, 5, kind="word"), decon.shingles(e, 5, kind="word"))
        )
        for e in edits
    ]

    assert max(disagreements) > 0.05, "char and word shingling must be able to disagree, or the parameter is pointless"


def test_an_unknown_shingle_kind_is_refused() -> None:
    with pytest.raises(decon.DeconError, match="shingle kind"):
        decon.shingles("text", kind="token")


def test_normalization_is_applied_before_shingling() -> None:
    plain = decon.shingles("The Quick Brown Fox Jumps", 5, kind="word")
    punctuated = decon.shingles("the, quick! brown? fox: jumps.", 5, kind="word")

    assert plain == punctuated


def test_normalization_is_unicode_aware() -> None:
    """An ASCII punctuation class would delete non-Latin text outright."""
    norm = decon.Normalization()

    assert "tiếng" in norm.apply("Tiếng, Việt!")
    assert "भाषा" in norm.apply("भाषा।")


def test_the_normalization_is_reported_as_data() -> None:
    """Two runs that normalized differently did not measure the same thing."""
    assert set(decon.Normalization().to_dict()) == {
        "casefold",
        "nfc",
        "collapse_whitespace",
        "strip_punctuation",
    }


# -- the threat model boundary ------------------------------------------------


def test_a_near_duplicate_document_is_detected() -> None:
    """Model (A): what this method does support."""
    original = " ".join(f"paragraph {i} of the original article text" for i in range(40))
    edited = original.replace("paragraph 3 ", "paragraph three ")

    similarity = decon.jaccard(decon.shingles(original), decon.shingles(edited))  # char-24, Curator's default

    assert similarity > 0.8


def test_an_embedded_benchmark_question_is_not_detected() -> None:
    """Model (B): what it does not, stated as a measurement rather than a caveat.

    A short question inside a long page moves whole-document Jaccard far below
    any usable threshold. If this ever starts passing, the step's scope claim
    needs revisiting — it does not mean the step got better.
    """
    training_doc = f"{FILLER} {QUESTION} {FILLER}"

    similarity = decon.jaccard(decon.shingles(training_doc), decon.shingles(QUESTION))

    assert similarity < 0.05, "whole-document Jaccard cannot see substring contamination"


def test_the_module_never_claims_the_holdout_is_clean() -> None:
    """The overclaim most likely to be made, blocked in the source itself."""
    import inspect

    source = inspect.getsource(decon).lower()

    assert "verified clean" not in source.replace('never "holdout verified clean"', "")


# -- verification of candidates -----------------------------------------------


def test_a_candidate_pair_is_verified_by_exact_jaccard() -> None:
    """LSH proposes; only a computed similarity removes anything."""
    train = {"t1": "alpha beta gamma delta epsilon zeta eta theta"}
    holdout = {"h1": "alpha beta gamma delta epsilon zeta eta theta"}

    pairs = decon.verify_pairs([("t1", "h1")], train, holdout, threshold=0.8)

    assert pairs[0].similarity == 1.0
    assert pairs[0].verifiable


def test_a_false_positive_candidate_survives_verification() -> None:
    """An LSH bucket collision is not evidence; removing on it loses good data."""
    train = {"t1": "completely different words appear in this training document here"}
    holdout = {"h1": "nothing at all like the other one not even close"}

    pairs = decon.verify_pairs([("t1", "h1")], train, holdout, threshold=0.8)

    assert pairs[0].similarity < 0.8
    assert decon.removals(pairs, 0.8) == {}


def test_a_pair_too_short_to_shingle_is_unverifiable_not_clean(caplog) -> None:
    train = {"t1": "three short words"}
    holdout = {"h1": "three short words"}

    pair = decon.verify_pairs([("t1", "h1")], train, holdout, threshold=0.8)[0]

    assert not pair.verifiable
    assert "too short" in pair.reason
    assert decon.removals([pair], 0.8) == {}, "an unverified pair must not remove data"


def test_a_missing_document_is_unverifiable_not_clean() -> None:
    pair = decon.verify_pairs([("t1", "h1")], {}, {}, threshold=0.8)[0]

    assert not pair.verifiable
    assert "not found" in pair.reason


def test_a_document_matching_several_is_removed_once_with_its_best_evidence() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    train = {"t1": text}
    holdout = {"h1": text, "h2": text[:40] + " something else entirely different here"}

    pairs = decon.verify_pairs([("t1", "h1"), ("t1", "h2")], train, holdout, threshold=0.5)
    chosen = decon.removals(pairs, 0.5)

    assert list(chosen) == ["t1"]
    assert chosen["t1"].holdout_id == "h1"


def test_the_threshold_is_configurable_not_hardcoded() -> None:
    train = {"t1": "alpha beta gamma delta epsilon zeta eta theta"}
    holdout = {"h1": "alpha beta gamma delta epsilon zeta eta different"}
    pairs = decon.verify_pairs([("t1", "h1")], train, holdout, threshold=0.1)

    assert decon.removals(pairs, 0.1)
    assert not decon.removals(pairs, 0.99)


def test_an_impossible_threshold_is_refused() -> None:
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(decon.DeconError, match="threshold"):
            decon.verify_pairs([], {}, {}, threshold=bad)


# -- direction ----------------------------------------------------------------


def test_removing_from_the_holdout_is_refused() -> None:
    """Changing a benchmark to agree with training data invalidates every result."""
    with pytest.raises(decon.HoldoutModifiedError) as excinfo:
        decon.assert_holdout_untouched(["h1", "h2", "h3"], ["h1", "h3"])

    assert "h2" in str(excinfo.value)
    assert "Only the training split may shrink" in str(excinfo.value)


def test_an_unchanged_holdout_passes() -> None:
    decon.assert_holdout_untouched(["h1", "h2"], ["h2", "h1"])


# -- recall, measured rather than asserted ------------------------------------


def test_recall_is_measured_against_brute_force() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    train = {"t1": text, "t2": "totally unrelated content with no overlap whatsoever here"}
    holdout = {"h1": text}

    result = decon.candidate_recall([("t1", "h1")], train, holdout, threshold=0.8)

    assert result["true_pairs"] == 1
    assert result["recall"] == 1.0
    assert result["missed"] == []


def test_a_missed_pair_is_named() -> None:
    """A recall number without the misses cannot be acted on."""
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    train = {"t1": text}
    holdout = {"h1": text}

    result = decon.candidate_recall([], train, holdout, threshold=0.8)

    assert result["recall"] == 0.0
    assert result["missed"] == [("t1", "h1")]


def test_recall_says_it_does_not_transfer() -> None:
    result = decon.candidate_recall([], {}, {}, threshold=0.8)

    assert "does not transfer" in result["note"]
    assert result["recall"] is None, "no true pairs means recall is undefined, not 1.0"


# -- fingerprint --------------------------------------------------------------


def test_the_fingerprint_is_order_independent() -> None:
    assert decon.corpus_fingerprint(["a", "b", "c"]) == decon.corpus_fingerprint(["c", "a", "b"])


def test_the_fingerprint_changes_with_membership() -> None:
    assert decon.corpus_fingerprint(["a", "b"]) != decon.corpus_fingerprint(["a", "b", "c"])


# -- the cheap exact check that runs first ------------------------------------


def test_the_same_url_across_a_split_is_caught_regardless_of_similarity() -> None:
    """The leak near-duplicate detection can miss entirely.

    A page rewritten enough that its shingles no longer overlap is still the
    same source document, and must still not span a split.
    """
    train = [{"id": "t1", "url": "https://www.Example.com/article?utm_source=x", "text": FILLER}]
    holdout = [{"id": "h1", "url": "http://example.com/article/", "text": QUESTION}]

    result = grouping.cross_split_groups(train, holdout)

    assert result["shared_group_count"] == 1
    assert result["shared_groups"][0]["key_field"] == "url"


def test_similarity_alone_would_miss_that_pair() -> None:
    """Which is why the group check is not redundant with the MinHash pass."""
    similarity = decon.jaccard(decon.shingles(FILLER), decon.shingles(QUESTION))

    assert similarity < 0.05


def test_url_canonicalization_groups_the_forms_that_differ_trivially() -> None:
    canonical = grouping.canonical_url("https://www.Example.com/a/b/?utm_source=x#frag")

    assert canonical == "example.com/a/b"
    assert grouping.canonical_url("http://example.com/a/b") == canonical


def test_a_meaningful_query_parameter_is_not_a_tracking_parameter() -> None:
    """Dropping the whole query string merges distinct pages on query-driven sites.

    Measured on Vietnamese C4: forum threads differing only by ``?t=<id>``
    collapsed into one group, and a single shared path then pulled 29 unrelated
    training documents — pairwise Jaccard 0.00 — out of the split.
    """
    a = grouping.canonical_url("https://forums.example.vn/showthread.php?t=12345")
    b = grouping.canonical_url("https://forums.example.vn/showthread.php?t=67890")

    assert a != b
    assert a == "forums.example.vn/showthread.php?t=12345"


def test_tracking_parameters_are_dropped_but_the_page_parameter_survives() -> None:
    with_tracking = grouping.canonical_url("https://forums.example.vn/showthread.php?t=123&utm_source=fb&fbclid=abc")

    assert with_tracking == "forums.example.vn/showthread.php?t=123"


def test_query_parameter_order_does_not_split_a_group() -> None:
    a = grouping.canonical_url("https://example.vn/p?b=2&a=1")
    b = grouping.canonical_url("https://example.vn/p?a=1&b=2")

    assert a == b


def test_a_query_of_only_tracking_parameters_leaves_a_bare_path() -> None:
    assert grouping.canonical_url("https://example.vn/p?utm_source=x&fbclid=y") == "example.vn/p"


def test_a_repeated_www_prefix_still_groups_with_the_page() -> None:
    """A rewrite rule that prepends www. to a host that already has it.

    Stripping a single prefix leaves www.example.com in a different group from
    example.com — the two would sit on opposite sides of a split while being
    the same page.
    """
    assert grouping.canonical_url("http://www.www.example.com/a") == "example.com/a"
    assert grouping.canonical_url("https://example.com/a") == "example.com/a"


def test_a_corpus_naming_its_url_column_differently_still_groups_by_page() -> None:
    """Reading only a field named 'url' demotes such a corpus to content hashing."""
    train = [{"id": "t1", "warc-target-uri": "https://example.com/p", "text": FILLER}]
    holdout = [{"id": "h1", "url": "https://example.com/p", "text": FILLER}]

    assert grouping.cross_split_groups(train, holdout)["shared_group_count"] == 1


def test_ids_are_namespaced_by_source() -> None:
    """Otherwise two corpora that both number from 1 collide on every document."""
    a, _ = grouping.group_key({"id": "1", "text": "x"}, "web")
    b, _ = grouping.group_key({"id": "1", "text": "x"}, "wiki")

    assert a != b


def test_grouping_falls_back_to_normalized_text_when_there_is_no_identity() -> None:
    key, which = grouping.group_key({"text": "The  Quick   Brown Fox"}, "s")
    other, _ = grouping.group_key({"text": "the quick brown fox"}, "s")

    assert which == grouping.NORM_HASH_FIELD
    assert key == other, "a raw content hash would separate these"


def test_every_record_receives_exactly_one_group() -> None:
    """A record dropped for lacking a key is one silently missing from its split."""
    records = [{"text": ""}, {"id": None}, {}]

    assigned = grouping.assign_group_keys(records, "s")

    assert all(r[grouping.GROUP_KEY_FIELD] for r in assigned)
    assert len({r[grouping.GROUP_KEY_FIELD] for r in assigned}) == 3


def test_the_field_that_produced_the_key_is_recorded() -> None:
    """So a grouping decision can be inspected rather than inferred."""
    assigned = grouping.assign_group_keys([{"id": "1", "url": "https://example.com/a", "text": "x"}], "s")

    assert assigned[0][grouping.GROUP_KEY_SOURCE_FIELD] == "url"


def test_modern_click_identifiers_are_treated_as_tracking() -> None:
    """A missing campaign parameter is an under-merge, and an under-merge is a leak.

    gbraid/wbraid replaced gclid for iOS traffic and ttclid is high-volume on the
    Vietnamese web; omitting any of them splits a page from its own untagged copy.
    On the skip_similarity path the identity pass is the only check, so nothing
    downstream recovers it.
    """
    bare = grouping.canonical_url("https://vnexpress.net/bai-123.html")

    for param in ("gbraid", "wbraid", "ttclid", "srsltid", "_gl", "utm_source_platform"):
        tagged = grouping.canonical_url(f"https://vnexpress.net/bai-123.html?{param}=abc")
        assert tagged == bare, f"{param} was not recognised as a tracking parameter"


def test_ref_is_not_treated_as_tracking() -> None:
    """On every Git-hosting site ``ref`` selects a branch or commit.

    Dropping it would collapse distinct file revisions into one group — the same
    over-merge that stripping the whole query string caused.
    """
    main = grouping.canonical_url("https://code.example/r/blob/x.py?ref=main")
    tag = grouping.canonical_url("https://code.example/r/blob/x.py?ref=v2.0")

    assert main != tag
    assert "ref" not in grouping.TRACKING_PARAMS


def test_twitter_specific_referral_params_are_tracking() -> None:
    """``ref_src``/``ref_url`` are unambiguous where bare ``ref`` is not."""
    bare = grouping.canonical_url("https://example.vn/a")

    assert grouping.canonical_url("https://example.vn/a?ref_src=twsrc") == bare
    assert grouping.canonical_url("https://example.vn/a?ref_url=https%3A%2F%2Fx") == bare


# -- identity across a split, for corpora with no URL --------------------------
#
# Vietnamese C4 has a url field, so every cross-split check there resolved on the
# URL and the id path was never exercised. Sangraha carries doc_id and no URL,
# which is what surfaced this.


def test_the_same_document_id_on_both_sides_is_one_group() -> None:
    """A corpus without a URL falls straight through to the id.

    Namespacing that id by the side it was read from gives one document two keys,
    so the check that is supposed to catch the leak cannot match anything at all.
    """
    train = [{"id": "doc-abc", "text": "a hindi paragraph about some topic"}]
    holdout = [{"id": "doc-abc", "text": "a hindi paragraph about some topic"}]

    result = grouping.cross_split_groups(train, holdout)

    assert result["shared_group_count"] == 1
    assert result["shared_groups"][0]["key_field"] == "id"


def test_two_different_corpora_may_reuse_an_id_without_being_merged() -> None:
    """Two corpora that both number from 1 are not describing the same documents."""
    train = [{"id": "1", "text": "one corpus"}]
    holdout = [{"id": "1", "text": "a different corpus entirely"}]

    result = grouping.cross_split_groups(train, holdout, id_namespace=None)

    assert result["shared_group_count"] == 0


def test_the_positional_fallback_is_never_shared_across_sides() -> None:
    """Row 0 of the training split is not row 0 of the holdout.

    The id namespace is shared on purpose; the last-resort positional key must
    not be, or two documents with no identity at all would be reported as one.
    """
    train = [{"text": ""}, {"text": ""}]
    holdout = [{"text": ""}, {"text": ""}]

    assert grouping.cross_split_groups(train, holdout)["shared_group_count"] == 0


def test_a_shared_id_still_groups_when_only_one_side_has_a_url() -> None:
    """Mixed provenance: the id is the only identity both sides share."""
    train = [{"id": "d1", "url": "https://example.vn/a", "text": "x y z"}]
    holdout = [{"id": "d1", "text": "x y z"}]

    result = grouping.cross_split_groups(train, holdout)

    # The train side resolves on url, the holdout side on id, so the keys differ
    # and the normalised-text hash is what has to catch it. Recorded rather than
    # asserted as a win: identity precedence is per-record, not per-corpus.
    assert result["shared_group_count"] == 0, (
        "documented behaviour: precedence is evaluated per record, so a url on one "
        "side and only an id on the other do not meet"
    )


def test_splits_keyed_by_different_fields_are_refused_not_reported_clean() -> None:
    """The failure this reproduces was found on real data, not constructed.

    A Vietnamese Wikipedia run planted 300 verbatim copies of corpus documents
    into the holdout. Decontamination removed 0 of 300 and the step reported ok.
    The corpus carried url, the holdout did not, so group_key took url on one
    side and id on the other. Keys are namespaced by the field that produced
    them, so the two key spaces were disjoint and no pair could ever match --
    the zero was structural, and identical text made no difference to it.
    """
    from nemotron.steps.curate.nemo_curator.runtime import grouping

    text = "Tư bản - Phê phán khoa kinh tế chính trị " * 40
    train = [{"id": "minted-a1b2", "url": "https://vi.wikipedia.org/wiki/Tư_bản", "text": text}]
    holdout = [{"id": "87737", "text": text}]

    result = grouping.cross_split_groups(
        train,
        holdout,
        left_source="train",
        right_source="holdout",
        cfg=grouping.GroupKeyConfig(text_field="text"),
    )

    assert result["shared_group_count"] == 0, "identical text still shares no key -- that is the bug"
    assert result["left_key_fields"] == {"url": 1}
    assert result["right_key_fields"] == {"id": 1}
    assert result["unmatchable_left"] == 1
    assert result["unmatchable_right"] == 1
    assert result["comparable"] is False, (
        "disjoint key spaces must be reported as incomparable; a bare count of 0 reads as "
        "'measured, found nothing' when nothing could have been found"
    )


def test_splits_sharing_a_key_field_stay_comparable() -> None:
    """The refusal must not fire on the ordinary case it is meant to protect."""
    from nemotron.steps.curate.nemo_curator.runtime import grouping

    text = "Tư bản - Phê phán khoa kinh tế chính trị " * 40
    url = "https://vi.wikipedia.org/wiki/Tư_bản"
    result = grouping.cross_split_groups(
        [{"id": "a", "url": url, "text": text}],
        [{"id": "b", "url": url, "text": text}],
        left_source="train",
        right_source="holdout",
        cfg=grouping.GroupKeyConfig(text_field="text"),
    )

    assert result["comparable"] is True
    assert result["shared_group_count"] == 1
    assert result["left_records_affected"] == 1


def test_a_holdout_that_mixes_key_fields_is_refused_too() -> None:
    """The first version of this guard intersected field NAMES, and passed here.

    A holdout keying half its records off url and half off id shares the url
    field with a url-keyed corpus, so the name intersection is non-empty and the
    comparison was declared sound -- while the id-keyed half sat in a namespace
    the corpus never reaches and contributed a guaranteed zero. The count looked
    like a measurement. Counting unmatchable RECORDS is what distinguishes
    'searched and found 50' from 'searched half of it'.
    """
    from nemotron.steps.curate.nemo_curator.runtime import grouping

    text = "Tư bản - Phê phán khoa kinh tế chính trị " * 40
    url = "https://vi.wikipedia.org/wiki/Tu_ban_"
    train = [{"id": f"t{i}", "url": f"{url}{i}", "text": text + str(i)} for i in range(100)]
    holdout = [{"id": f"h{i}", "url": f"{url}{i}", "text": text + str(i)} for i in range(50)] + [
        {"id": f"h{i}", "text": text + str(i)} for i in range(50, 100)
    ]

    result = grouping.cross_split_groups(
        train,
        holdout,
        left_source="train",
        right_source="holdout",
        cfg=grouping.GroupKeyConfig(text_field="text"),
    )

    assert result["shared_group_count"] == 50, "half the duplicates are structurally invisible"
    assert result["unmatchable_right"] == 50
    assert result["comparable"] is False, "a shared field name is not comparability; the id-keyed half can never match"


def test_a_positional_fallback_key_never_establishes_comparability() -> None:
    """_rowid is per-split by construction, so two sides both falling through to
    it share a field name and nothing else. Counting it would let the guard be
    satisfied by the one key that can never match."""
    from nemotron.steps.curate.nemo_curator.runtime import grouping

    train = [{"text": "one"}, {"text": "two"}]
    holdout = [{"text": "one"}, {"text": "two"}]

    result = grouping.cross_split_groups(
        train,
        holdout,
        left_source="train",
        right_source="holdout",
        cfg=grouping.GroupKeyConfig(text_field="text", use_normalized_text=False),
    )

    assert result["left_key_fields"] == {grouping.POSITIONAL_FIELD: 2}
    assert result["comparable"] is False


def test_the_configured_id_field_is_what_identity_matching_reads() -> None:
    """A corpus whose id is not called `id` had its exact pass silently disabled.

    The exact-identity pass exists to catch the case similarity cannot: the same
    source document, rewritten enough that no shingle survives. It found it by
    keying on the record id. That id was read as a hardcoded `record["id"]`, so a
    corpus using `doc_id` fell through to the normalised-text-hash fallback —
    which by construction cannot match a document whose text changed. The run
    then reported zero overlap, and the report does not distinguish "measured and
    found none" from "keyed off a field that is not there".
    """
    train = [{"doc_id": "a1", "text": "The original wording of this document."}]
    holdout = [{"doc_id": "a1", "text": "This document, but reworded."}]

    unset = grouping.cross_split_groups(train, holdout, cfg=grouping.GroupKeyConfig(text_field="text"))
    assert unset["shared_group_count"] == 0, "the bug: same document, no overlap reported"
    assert list(unset["left_key_fields"]) == ["__norm_text_hash"]

    named = grouping.cross_split_groups(
        train, holdout, cfg=grouping.GroupKeyConfig(text_field="text", id_field="doc_id")
    )
    assert named["shared_group_count"] == 1
    assert list(named["left_key_fields"]) == ["doc_id"]
    assert named["shared_groups"][0]["left_ids"] == ["a1"], (
        "ids were reported as empty strings, so removal downstream had nothing to remove"
    )


def test_a_coarse_url_column_can_be_overruled_by_the_configured_id() -> None:
    """The URL aliases outranked id_field with no way to say otherwise.

    That order is right when the URL is a page's real identity and the id is
    synthetic. It is wrong when the column is coarse: a `source_url` holding a
    site homepage puts every document from that site in one key, so a single
    holdout page from that site marks all of its training documents as
    overlapping — and they are removed without duplicating anything.
    """
    train = [
        {"doc_id": "a1", "source_url": "https://site.example/", "text": "one article"},
        {"doc_id": "a2", "source_url": "https://site.example/", "text": "a different article"},
    ]
    holdout = [{"doc_id": "b9", "source_url": "https://site.example/", "text": "unrelated third article"}]

    default = grouping.cross_split_groups(
        train, holdout, cfg=grouping.GroupKeyConfig(text_field="text", id_field="doc_id")
    )
    assert list(default["left_key_fields"]) == ["source_url"]
    assert default["left_records_affected"] == 2, "both training documents flagged, neither a duplicate"

    preferred = grouping.cross_split_groups(
        train,
        holdout,
        cfg=grouping.GroupKeyConfig(text_field="text", id_field="doc_id", prefer_id_field=True),
    )
    assert list(preferred["left_key_fields"]) == ["doc_id"]
    assert preferred["shared_group_count"] == 0
    assert preferred["left_records_affected"] == 0


def test_the_shared_id_space_can_be_turned_off() -> None:
    """The docstring documented an argument the config could not reach.

    Two splits of one corpus share an id space, which is why SHARED_CORPUS is the
    default. Two separately built corpora that both number from zero do not, and
    there was no way to say so from a config file.
    """
    train = [{"doc_id": "1", "text": "corpus A document one"}]
    holdout = [{"doc_id": "1", "text": "corpus B document one, unrelated"}]
    cfg = grouping.GroupKeyConfig(text_field="text", id_field="doc_id")

    shared = grouping.cross_split_groups(train, holdout, cfg=cfg, id_namespace=grouping.SHARED_CORPUS)
    assert shared["shared_group_count"] == 1, "equal ids read as the same document"

    per_side = grouping.cross_split_groups(train, holdout, cfg=cfg, id_namespace=None)
    assert per_side["shared_group_count"] == 0, "equal ids read as a collision"
