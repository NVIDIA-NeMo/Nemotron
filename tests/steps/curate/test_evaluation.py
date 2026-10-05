# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Filter quality measured against labelled data, not inferred from retention.

Retention was the only quality evidence this category produced, and it cannot
distinguish a gate that keeps 93% of a corpus from one that discards exactly the
7% that was good. These tests pin the harness that answers the other question,
and the failure modes it has to exercise: Unicode normalisation, local digits,
OCR noise, code-mixing and language-ID errors.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest

from nemotron.steps.curate.nemo_curator.runtime import evaluation, langpack
from nemotron.steps.curate.nemo_curator.runtime import registry as r_registry
from nemotron.steps.curate.nemo_curator.scripts import run_evaluate

PACKS = Path(langpack.__file__).parent.parent / "data" / "langpacks"
LABELLED = Path(langpack.__file__).parent.parent / "data" / "evaluation"


def _pack(tag: str = "vi"):
    return langpack.load(tag, PACKS)


def _raises(**_kwargs):
    """Stands in for a Curator-backed signal on a machine without Curator."""
    raise ModuleNotFoundError("No module named 'nemo_curator'")


def _doc(doc_id, label, phenomenon, text, language="vi"):
    return {"id": doc_id, "language": language, "label": label, "phenomenon": phenomenon, "text": text}


# -- the two rates -------------------------------------------------------------


def test_a_gate_that_drops_nothing_has_no_false_rejections_and_removes_no_noise() -> None:
    """Both rates are needed. Either alone is satisfied by a useless filter."""
    docs = [_doc("a", "keep", "clean", "x"), _doc("b", "drop", "ocr_noise", "y")]

    report = evaluation.evaluate(docs, [{"signal": "script_ratio", "min": 0.0}], pack=_pack())

    assert report.false_rejection_rate == 0.0
    assert report.noise_removal_rate == 0.0


def test_a_gate_that_drops_everything_removes_all_noise_and_all_content() -> None:
    docs = [_doc("a", "keep", "clean", "x"), _doc("b", "drop", "ocr_noise", "y")]

    report = evaluation.evaluate(docs, [{"signal": "script_ratio", "min": 1.1}], pack=_pack())

    assert report.false_rejection_rate == 1.0
    assert report.noise_removal_rate == 1.0


def test_a_rate_over_no_documents_is_none_not_zero() -> None:
    """Zero would read as a measured rate of nothing going wrong."""
    report = evaluation.evaluate(
        [_doc("a", "keep", "clean", "x")], [{"signal": "script_ratio", "min": 0.0}], pack=_pack()
    )

    assert report.noise_removal_rate is None, "no drop-labelled documents means no rate, not a perfect one"
    assert report.as_dict()["noise_removal_rate"] is None


def test_the_report_attributes_a_rejection_to_the_signal_that_made_it() -> None:
    """'The policy rejects 12% of good Hindi' sends someone hunting; naming the
    signal names the threshold to move."""
    docs = [_doc("a", "keep", "clean", "xxxx")]

    report = evaluation.evaluate(docs, [{"signal": "script_ratio", "min": 1.1}], pack=_pack())

    assert report.by_signal()["script_ratio"]["rejected_keep"] == 1


def test_an_aggregate_cannot_hide_a_broken_phenomenon() -> None:
    """The reason the per-mode table exists: nine clean documents and one broken
    mode report a 10% aggregate and a 100% failure on the mode that matters."""
    docs = [_doc(f"clean-{i}", "keep", "clean", "aaaa") for i in range(9)]
    docs.append(_doc("ocr-1", "keep", "ocr_noise", "!!!!"))

    report = evaluation.evaluate(docs, [{"signal": "script_ratio", "min": 0.5}], pack=_pack())
    per_mode = report.by_phenomenon()

    assert per_mode["ocr_noise"]["false_rejection_rate"] == 1.0
    assert report.false_rejection_rate < 0.5, "the aggregate looks fine, which is the point"


# -- the labelled set has to be trustworthy ------------------------------------


@pytest.mark.parametrize(
    "record,reason",
    [
        ({"id": "a", "text": "x", "phenomenon": "clean"}, "label"),
        ({"id": "a", "text": "x", "label": "keep"}, "phenomenon"),
        ({"id": "a", "text": "x", "label": "maybe", "phenomenon": "clean"}, "label must be"),
        ({"id": "a", "text": "x", "label": "keep", "phenomenon": "smudged"}, "not one of"),
    ],
)
def test_a_defect_that_would_produce_a_plausible_number_is_refused(tmp_path, record, reason) -> None:
    """Every one of these shrinks a denominator or hides a mode rather than raising."""
    path = tmp_path / "set.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(evaluation.EvaluationDataError, match=reason):
        evaluation.read_labelled([path])


def test_a_duplicate_id_is_refused(tmp_path) -> None:
    """It double-counts one document's verdict in both numerator and denominator."""
    path = tmp_path / "set.jsonl"
    row = json.dumps(_doc("a", "keep", "clean", "x"))
    path.write_text(row + "\n" + row + "\n", encoding="utf-8")

    with pytest.raises(evaluation.EvaluationDataError, match="duplicate id"):
        evaluation.read_labelled([path])


def test_an_empty_set_is_refused(tmp_path) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text("# only a comment\n", encoding="utf-8")

    with pytest.raises(evaluation.EvaluationDataError, match="nothing to measure"):
        evaluation.read_labelled([path])


# -- the phenomena the reviewer named ------------------------------------------


def test_the_shipped_sets_cover_every_named_phenomenon() -> None:
    """A set containing no OCR noise reports a rate that says nothing about it."""
    documents = evaluation.read_labelled(sorted(LABELLED.glob("*.jsonl")))
    covered = {d["phenomenon"] for d in documents}

    assert covered == set(evaluation.PHENOMENA), f"missing: {sorted(set(evaluation.PHENOMENA) - covered)}"


def test_the_shipped_sets_label_both_directions() -> None:
    """Only keep-labelled documents makes noise removal unmeasurable."""
    documents = evaluation.read_labelled(sorted(LABELLED.glob("*.jsonl")))

    assert {d["label"] for d in documents} == set(evaluation.LABELS)


@pytest.mark.parametrize("tag", ["vi", "hi"])
def test_nfc_and_nfd_of_the_same_document_get_the_same_verdict(tag) -> None:
    """The invariant, restated as an evaluation: a normalisation form is not a
    quality difference, and a filter that disagrees across them is broken."""
    text = "Hôm nay trời đẹp." if tag == "vi" else "आज मौसम अच्छा है।"
    thresholds = [{"signal": "unicode_alpha_numeric", "max": 0.25}, {"signal": "script_ratio", "min": 0.3}]

    docs = [
        _doc("nfc", "keep", "clean", unicodedata.normalize("NFC", text), tag),
        _doc("nfd", "keep", "nfd", unicodedata.normalize("NFD", text), tag),
    ]
    report = evaluation.evaluate(docs, thresholds, pack=_pack(tag))

    verdicts = {j.doc_id: j.kept for j in report.judgements}
    assert verdicts["nfc"] == verdicts["nfd"], report.by_signal()


def test_ocr_noise_is_removed_and_clean_text_is_not() -> None:
    """The pair that makes the number mean something: removing noise is only
    evidence of quality if the clean document beside it survives."""
    documents = evaluation.read_labelled([LABELLED / "vi.jsonl"])
    thresholds = [{"signal": "unicode_alpha_numeric", "max": 0.3333333333333333}]

    report = evaluation.evaluate(documents, thresholds, pack=_pack("vi"))
    per_mode = report.by_phenomenon()

    assert per_mode["ocr_noise"]["noise_removal_rate"] == 1.0
    assert per_mode["clean"]["false_rejection_rate"] == 0.0


def test_a_code_mixed_document_is_not_treated_as_noise() -> None:
    """Latin technical terms inside Vietnamese prose are ordinary writing."""
    documents = evaluation.read_labelled([LABELLED / "vi.jsonl"])
    thresholds = [{"signal": "unicode_alpha_numeric", "max": 0.3333333333333333}]

    report = evaluation.evaluate(documents, thresholds, pack=_pack("vi"))

    assert report.by_phenomenon()["code_mixing"]["false_rejection_rate"] == 0.0


def test_local_digits_do_not_make_a_document_look_foreign() -> None:
    """Devanagari digits are category Nd; a filter reading them as non-content
    would reject correct Hindi."""
    documents = evaluation.read_labelled([LABELLED / "hi.jsonl"])
    thresholds = [{"signal": "unicode_alpha_numeric", "max": 0.3333333333333333}]

    report = evaluation.evaluate(documents, thresholds, pack=_pack("hi"))

    assert report.by_phenomenon()["local_digits"]["false_rejection_rate"] == 0.0


# -- what the harness will not claim -------------------------------------------


def test_a_signal_that_cannot_be_built_is_named_not_silently_dropped(monkeypatch) -> None:
    """Skipping it quietly would report a rate for a policy only half applied.

    The factory is made to fail rather than relying on Curator being absent: the
    previous version asserted that `word_count` lands in unscored_signals, which
    is true only on a machine without NeMo Curator. A test whose verdict depends
    on what is installed measures the environment, not the code.
    """
    import dataclasses

    signals_copy = dict(r_registry.SIGNALS)
    signals_copy["word_count"] = dataclasses.replace(signals_copy["word_count"], factory=_raises)
    monkeypatch.setattr(r_registry, "SIGNALS", signals_copy)

    report = evaluation.evaluate(
        [_doc("a", "keep", "clean", "x")],
        [{"signal": "word_count", "min": 1, "max": 100000}, {"signal": "script_ratio", "min": 0.0}],
        pack=_pack(),
    )

    assert "word_count" in report.unscored_signals
    assert "INCOMPLETE" in report.notes["completeness"], (
        "a policy only half applied must not report a rate as though it were the whole policy"
    )


def test_the_report_says_what_its_numbers_are_worth() -> None:
    """A rate over a constructed set is a regression signal, not a quality claim,
    and the artifact has to say so rather than leave a reader to assume."""
    report = evaluation.evaluate(
        [_doc("a", "keep", "clean", "x")], [{"signal": "script_ratio", "min": 0.0}], pack=_pack()
    )

    assert "interpretation" in report.notes
    assert "labelled set" in report.notes["interpretation"]


# -- the CLI -------------------------------------------------------------------


def _policy(tmp_path, thresholds, tag="vi", **overrides):
    """A policy the FILTER step would accept.

    The helper used to write `{approved, thresholds, langpack}` and nothing else
    -- a document curate/nemo_curator refuses outright. Evaluating it reported a
    rate for a run that could never happen, which is the defect these tests now
    guard.
    """
    import yaml

    from nemotron.steps.curate.nemo_curator.runtime import policy as policy_module
    from nemotron.steps.curate.nemo_curator.runtime import registry as reg

    document = {
        "schema_version": policy_module.SCHEMA_VERSION,
        "approved": True,
        "corpus": {"fingerprint": "sha256:" + "a" * 64},
        "signals_impl_version": reg.IMPL_VERSION,
        "profile_digest": "sha256:" + "b" * 64,
        "approval": {
            "method": "manual",
            "approver": "reviewer@example.test",
            "date": "2026-09-09",
            "evidence": "labelled set reviewed",
        },
        "thresholds": thresholds,
        "langpack": {"language_tag": tag, "langpack_dir": str(PACKS)},
    }
    document.update(overrides)
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_the_cli_reports_both_rates_and_writes_json(tmp_path, capsys) -> None:
    policy = _policy(tmp_path, [{"signal": "unicode_alpha_numeric", "max": 0.3333333333333333}])
    out = tmp_path / "report.json"

    code = run_evaluate.main(["--policy", str(policy), "--labelled", str(LABELLED / "vi.jsonl"), "--report", str(out)])

    assert code == 0
    printed = capsys.readouterr().out
    assert "false rejection" in printed
    assert "noise removal" in printed
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["by_phenomenon"]["ocr_noise"]["noise_removal_rate"] == 1.0


def test_a_policy_with_no_thresholds_exits_two(tmp_path, capsys) -> None:
    policy = _policy(tmp_path, [])

    code = run_evaluate.main(["--policy", str(policy), "--labelled", str(LABELLED / "vi.jsonl")])

    assert code == 2
    # The shared policy validation now catches this before the evaluator does,
    # which is the point: one gate, applied by both.
    assert "thresholds" in capsys.readouterr().err


def test_a_missing_pack_exits_two_rather_than_scoring_without_one(tmp_path, capsys) -> None:
    """Scoring without the pack the thresholds were calibrated against would
    measure a different thing and report it as the same one."""
    policy = _policy(tmp_path, [{"signal": "script_ratio", "min": 0.3}], tag="zz-nope")

    code = run_evaluate.main(["--policy", str(policy), "--labelled", str(LABELLED / "vi.jsonl")])

    assert code == 2
    assert "no pack" in capsys.readouterr().err


# -- the command has to be findable ---------------------------------------------


def test_the_command_is_documented_where_a_reader_would_look() -> None:
    """It was documented in exactly one place: ADDING_A_LANGUAGE.md, which is
    about adding a LANGUAGE. Someone who already has one and wants to check their
    thresholds had no path to it -- the same defect SPEC.md had, three inbound
    links and none from an entry point.
    """
    curate = Path(langpack.__file__).parents[2]

    for doc, why in (
        (curate / "README.md", "the category entry point"),
        (curate / "nemo_curator" / "profile" / "README.md", "where thresholds come from"),
        (curate / "nemo_curator" / "data" / "langpacks" / "ADDING_A_LANGUAGE.md", "the onboarding procedure"),
    ):
        assert "run_evaluate" in doc.read_text(encoding="utf-8"), f"{doc} ({why}) does not name the command"


def test_the_developer_journey_ends_with_checking_the_policy() -> None:
    """The journey used to stop at 'the run succeeded', which is the claim this
    whole category exists to distrust."""
    text = (Path(langpack.__file__).parents[2] / "README.md").read_text(encoding="utf-8")

    assert "Checking A Policy Against Labelled Documents" in text
    assert "false rejection" in text
    assert "noise removal" in text


def test_a_scorer_that_raises_is_not_counted_as_keeping_the_document(monkeypatch) -> None:
    """The fail-open that made every metric a lie.

    A scorer raising left the verdict unknown and the document was recorded as
    kept, so noise the scorer choked on read as noise the policy let through --
    and the rate looked better for it. Unknown is not "kept".
    """
    import dataclasses

    class _Explodes:
        def score_document(self, text):
            raise ValueError("cannot score this")

        def keep_document(self, score):  # pragma: no cover - never reached
            return True

    signals_copy = dict(r_registry.SIGNALS)
    signals_copy["script_ratio"] = dataclasses.replace(signals_copy["script_ratio"], factory=lambda **kw: _Explodes())
    monkeypatch.setattr(r_registry, "SIGNALS", signals_copy)

    with pytest.raises(ValueError, match="cannot score this"):
        evaluation.evaluate(
            [_doc("a", "drop", "ocr_noise", "x")], [{"signal": "script_ratio", "min": 0.5}], pack=_pack()
        )


def test_a_policy_the_filter_would_refuse_is_not_evaluated(tmp_path, capsys) -> None:
    """Reporting a rate for a policy that cannot run describes a run that will
    never happen -- and an unapproved or malformed policy is exactly what someone
    points this at while iterating."""
    policy = _policy(tmp_path, [{"signal": "script_ratio", "min": 0.3}], approved=False)

    code = run_evaluate.main(["--policy", str(policy), "--labelled", str(LABELLED / "vi.jsonl")])

    assert code == 2
    assert "would be refused" in capsys.readouterr().err


def test_a_wrong_direction_bound_is_refused(tmp_path, capsys) -> None:
    """script_ratio is a min-direction signal. Accepting `max:` here would gate
    the opposite way from the policy that runs, and report rates for it."""
    policy = _policy(tmp_path, [{"signal": "script_ratio", "max": 0.3}])

    code = run_evaluate.main(["--policy", str(policy), "--labelled", str(LABELLED / "vi.jsonl")])

    assert code == 2


def test_a_set_in_another_language_is_refused(tmp_path, capsys) -> None:
    """The `language` field was read into the report and otherwise ignored, so a
    Hindi set scored against the Vietnamese pack produced ordinary-looking rates."""
    policy = _policy(tmp_path, [{"signal": "script_ratio", "min": 0.3}], tag="vi")

    code = run_evaluate.main(["--policy", str(policy), "--labelled", str(LABELLED / "hi.jsonl")])

    assert code == 2
    assert "pack is 'vi'" in capsys.readouterr().err


def test_the_report_records_what_it_was_computed_from(tmp_path) -> None:
    """Two runs naming the same paths may have read different bytes."""
    policy = _policy(tmp_path, [{"signal": "script_ratio", "min": 0.3}])
    out = tmp_path / "report.json"

    run_evaluate.main(["--policy", str(policy), "--labelled", str(LABELLED / "vi.jsonl"), "--report", str(out)])
    written = json.loads(out.read_text(encoding="utf-8"))

    assert len(written["policy_sha256"]) == 64
    assert all(len(d) == 64 for d in written["labelled_sets"].values())


def test_a_mode_the_harness_does_not_exercise_says_so_in_the_report() -> None:
    """A table row with a percentage in it reads like evidence.

    The harness loads no models, so `langid_error` labels the kind of document
    rather than testing how language-ID errors are handled -- and the summary
    that claimed end-to-end coverage of it was wrong. The rate is real; what it
    is evidence OF has to be stated where the rate appears.
    """
    documents = evaluation.read_labelled([LABELLED / "vi.jsonl"])

    report = evaluation.evaluate(documents, [{"signal": "script_ratio", "min": 0.3}], pack=_pack("vi"))
    row = report.by_phenomenon()["langid_error"]

    assert "not_exercised" in row
    assert "FastText" in row["not_exercised"]


def test_a_mode_the_harness_does_exercise_carries_no_such_caveat() -> None:
    documents = evaluation.read_labelled([LABELLED / "vi.jsonl"])

    report = evaluation.evaluate(documents, [{"signal": "script_ratio", "min": 0.3}], pack=_pack("vi"))

    assert "not_exercised" not in report.by_phenomenon()["ocr_noise"]
