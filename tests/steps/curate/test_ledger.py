# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""G8: record accounting, and the gate that a record count cannot implement.

The failure this module exists for is a stage that catches every worker
exception, logs it, and exits 0. Detection after the fact cannot tell such a
loss from a filter doing its job, so the producer has to say which at the time.
"""

from __future__ import annotations

import json

import pytest

from nemotron.steps.curate.nemo_curator.runtime import ledger


def balanced(stage: str = "s", source: str = "") -> ledger.StageLedger:
    led = ledger.StageLedger(stage=stage, source=source)
    led.add_input(100)
    led.add_success(70)
    led.add_filtered("language_id", 20)
    led.add_filtered("word_count", 10)
    return led


# -- the contract -------------------------------------------------------------


def test_a_balanced_ledger_reconciles() -> None:
    led = balanced()

    assert led.balanced
    assert led.n_accounted == led.n_input
    led.assert_balanced()


def test_an_unbalanced_ledger_names_the_gap() -> None:
    led = ledger.StageLedger(stage="enrich")
    led.add_input(100)
    led.add_success(70)

    with pytest.raises(ledger.LedgerImbalanceError) as excinfo:
        led.assert_balanced()

    message = str(excinfo.value)
    assert "enrich" in message
    assert "+30" in message, "30 records are unaccounted for and the message must say so"


def test_writing_an_unbalanced_ledger_is_refused(tmp_path) -> None:
    """A ledger nobody could reconcile is worse than none: it looks like evidence."""
    led = ledger.StageLedger(stage="s")
    led.add_input(10)

    with pytest.raises(ledger.LedgerImbalanceError):
        led.write(tmp_path / "ledger.json")

    assert not (tmp_path / "ledger.json").exists()


def test_an_unbalanced_ledger_can_be_written_deliberately(tmp_path) -> None:
    """So a crashing stage can still record what it knew before it died."""
    led = ledger.StageLedger(stage="s")
    led.add_input(10)

    path = led.write(tmp_path / "ledger.json", require_balanced=False)

    assert json.loads(path.read_text())["balanced"] is False


def test_filtered_records_carry_the_reason_that_removed_them() -> None:
    """'5,187,587 filtered' and '...filtered by language_id' are different facts."""
    led = balanced()

    assert led.as_dict()["filtered_by_reason"] == {"language_id": 20, "word_count": 10}


# -- the gate a record count cannot implement ---------------------------------


def test_a_truncated_shard_reports_zero_rows_and_still_counts_as_a_loss() -> None:
    """The failure mode the whole design turns on.

    A shard too damaged to open reports 0 rows, so ``n_failed`` is 0 and any
    gate written as ``n_failed > 0`` sees nothing. Counting units sees it.
    """
    led = ledger.StageLedger(stage="classify")
    led.add_input(0)
    led.add_failed("shard_0042.jsonl", "truncated: unexpected EOF", n_records=0)

    assert led.n_failed == 0, "the record count genuinely cannot see this"
    assert led.balanced, "and the reconciliation is satisfied too"

    with pytest.raises(ledger.LedgerImbalanceError, match="1 unit"):
        ledger.assert_no_lost_units([led], "classify")


def test_the_lost_unit_report_says_its_record_count_is_a_floor() -> None:
    led = ledger.StageLedger(stage="s")
    led.add_quarantined("shard_1.jsonl", "bad schema", n_records=0)

    report = ledger.lost_unit_report([led], "s")

    assert "FLOOR" in report


def test_a_clean_run_reports_no_lost_units() -> None:
    assert ledger.lost_unit_report([balanced()], "s") is None
    ledger.assert_no_lost_units([balanced()], "s")


def test_many_lost_units_are_summarised(tmp_path) -> None:
    led = ledger.StageLedger(stage="s")
    for i in range(25):
        led.add_failed(f"shard_{i}.jsonl", "worker died", n_records=0)

    report = ledger.lost_unit_report([led], "s")

    assert "and 15 more" in report
    assert report.count("shard_") == ledger.MAX_REPORTED_UNITS


# -- merging ------------------------------------------------------------------


def test_merging_sums_the_counts() -> None:
    merged = ledger.merge_ledgers("s", [balanced("s", "a"), balanced("s", "b")])

    assert merged.n_input == 200
    assert merged.n_success == 140
    assert merged.filtered["language_id"] == 40
    assert merged.balanced


def test_numeric_notes_sum_rather_than_taking_the_first() -> None:
    """A merge that kept the first value would report one source as the total.

    That is not hypothetical — it is the shape of an undercount that reads as a
    correct-looking number, which is the hardest kind to notice.
    """
    a, b = balanced("s", "a"), balanced("s", "b")
    a.notes["written_train"] = 45_913_678
    b.notes["written_train"] = 72_700_894

    merged = ledger.merge_ledgers("s", [a, b])

    assert merged.notes["written_train"] == 118_614_572


def test_disagreeing_non_numeric_notes_are_recorded_as_disagreeing() -> None:
    a, b = balanced("s", "a"), balanced("s", "b")
    a.notes["tokenizer"] = "model-x"
    b.notes["tokenizer"] = "model-y"

    merged = ledger.merge_ledgers("s", [a, b])

    assert "varies" in merged.notes["tokenizer"]


def test_booleans_are_not_treated_as_numbers() -> None:
    """``True + True == 2`` would turn two agreeing flags into a count."""
    a, b = balanced("s", "a"), balanced("s", "b")
    a.notes["gpu"] = True
    b.notes["gpu"] = True

    assert ledger.merge_ledgers("s", [a, b]).notes["gpu"] is True


def test_merged_units_are_all_retained() -> None:
    a, b = balanced("s", "a"), balanced("s", "b")
    a.add_failed("shard_1", "x", 0)
    b.add_failed("shard_2", "y", 0)

    assert len(ledger.merge_ledgers("s", [a, b]).lost_units) == 2


# -- round trip ---------------------------------------------------------------


def test_a_ledger_survives_a_round_trip(tmp_path) -> None:
    led = balanced("s", "web")
    led.add_quarantined("shard_9", "unreadable", 5)
    led.add_input(5)
    led.notes["region"] = "vi"

    path = led.write(tmp_path / "ledger.json")
    loaded = ledger.load_ledger(path)

    assert loaded.as_dict() == led.as_dict()


def test_the_write_is_atomic(tmp_path) -> None:
    """A job killed mid-write must leave the old ledger, not a truncated one."""
    path = tmp_path / "ledger.json"
    balanced().write(path)

    assert not list(tmp_path.glob("*.tmp")), "the temporary file must not survive"


def test_a_non_ledger_document_is_refused(tmp_path) -> None:
    path = tmp_path / "not-a-ledger.json"
    path.write_text('{"hello": "world"}', encoding="utf-8")

    with pytest.raises(ledger.LedgerInvalidError, match="no 'stage' field"):
        ledger.load_ledger(path)


def test_invalid_json_is_refused(tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ledger.LedgerInvalidError, match="not valid JSON"):
        ledger.load_ledger(path)


def test_an_unsupported_schema_version_is_refused(tmp_path) -> None:
    path = balanced().write(tmp_path / "ledger.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["schema_version"] = 999
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ledger.LedgerInvalidError, match="schema_version"):
        ledger.load_ledger(path)


def test_malformed_terminal_state_data_is_refused(tmp_path) -> None:
    path = balanced().write(tmp_path / "ledger.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["failed_units"] = [{"unit": "part-0", "reason": "failed", "records": -1}]
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ledger.LedgerInvalidError, match="records"):
        ledger.load_ledger(path)


def test_serialized_totals_must_match_terminal_state_records(tmp_path) -> None:
    path = balanced().write(tmp_path / "ledger.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["n_filtered"] += 1
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ledger.LedgerInvalidError, match="n_filtered"):
        ledger.load_ledger(path)


# -- attribution --------------------------------------------------------------


def test_attribution_explains_a_delta_the_ledgers_account_for() -> None:
    result = ledger.attribute([balanced()], observed_delta=30)

    assert result["unexplained"] == 0
    assert result["declared_filtered"] == 30
    assert result["filtered_by_reason"] == {"language_id": 20, "word_count": 10}


def test_attribution_surfaces_a_delta_nobody_recorded() -> None:
    """The number that matters: records gone for a reason no stage wrote down."""
    result = ledger.attribute([balanced()], observed_delta=5_187_587)

    assert result["unexplained"] == 5_187_557


def test_attribution_flags_that_counts_are_a_floor_when_units_were_lost() -> None:
    led = balanced()
    led.add_failed("shard_7", "truncated", 0)
    led.add_input(0)

    result = ledger.attribute([led], observed_delta=30)

    assert result["record_counts_are_a_floor"] is True
    assert result["lost_units"] == 1


def test_attribution_reports_whether_every_ledger_reconciled() -> None:
    broken = ledger.StageLedger(stage="s")
    broken.add_input(10)

    assert ledger.attribute([balanced()], 30)["all_ledgers_balanced"] is True
    assert ledger.attribute([balanced(), broken], 30)["all_ledgers_balanced"] is False
