"""Contract tests for BFCL Stage 12 exports."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    BENCHMARK_ROW_FIELDS,
    BFCL_JSON_SCHEMA_VERSION,
    BFCL_UPSTREAM_SCHEMA_VERSION,
    EXPORT_CONTRACT_VERSION,
    EXPORT_FORMAT_SCHEMA_VERSIONS,
    EXPORT_FORMATS,
    EXPORT_SCORING_METRICS,
    EXPORT_TRUTH_FIELDS,
    NEMO_EVALUATOR_SCHEMA_VERSION,
    BfclJsonRecord,
    CanonicalExportRow,
    ExportFormatReport,
    ExportRowFailure,
    ExportValidationReport,
    NemoEvaluatorBundle,
    NemoEvaluatorRecord,
    NemoEvaluatorScoring,
    NemoEvaluatorSource,
    decode_canonical_json,
    decode_lossless_arguments,
    export_manifest_section,
    json_equal,
    json_type_tag,
    validate_export_equivalence,
    validate_json_value,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_projection import conversation_plan
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import (
    benchmark_schema,
    canonical_json,
    encode_arguments,
)

HASH = "sha256:" + "0" * 64
OTHER_HASH = "sha256:" + "1" * 64

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_balance",
            "description": "Return one account balance.",
            "parameters": {
                "type": "object",
                "properties": {"account_id": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["account_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {"name": "list_cards", "description": "List cards.", "parameters": None},
    },
]
ARGUMENTS = {"account_id": "1", "limit": 1, "verbose": True, "note": None}


def _row(**overrides: Any) -> dict[str, Any]:
    """Build one benchmark row exactly as Stage 12 writes it to parquet."""
    row = {
        "task_id": "pack__tpl__abcdef",
        "template_id": "tpl",
        "variant_index": 0,
        "messages": [
            {"role": "system", "content": "You use tools.", "tool_calls": None, "tool_call_id": None},
            {"role": "user", "content": "Balance of 1?", "tool_calls": None, "tool_call_id": None},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "get_balance", "arguments": canonical_json(ARGUMENTS)},
                    }
                ],
                "tool_call_id": None,
            },
            {
                "role": "tool",
                "content": canonical_json({"balance": 10}),
                "tool_calls": None,
                "tool_call_id": "call_0",
            },
            {"role": "assistant", "content": "It is 10.", "tool_calls": None, "tool_call_id": None},
        ],
        "tools": canonical_json(TOOLS),
        "expected_tool_calls": [
            {
                "turn_index": 0,
                "call_group": 0,
                "position_in_group": 0,
                "function_name": "get_balance",
                "arguments": encode_arguments(ARGUMENTS),
            }
        ],
        "success_assertions": ["assert_balance_reported"],
        "fixture_refs": ['["accounts","1"]'],
        "intent": "check_balance",
        "category": "accounts",
        "difficulty": "easy",
        "required_tools": ["get_balance"],
        "required_tools_fingerprint": canonical_json(["get_balance"]),
        "tools_present": ["get_balance", "list_cards"],
        "turn_policy": "single_turn",
        "is_multi_turn": False,
        "num_tool_calls": 1,
        "call_order": "strict",
        "call_order_prefix": None,
        "system_prompt_id": "sp-1",
        "tier": "gold",
        "gold_eligible": True,
        "validated_by": ["schema", "replay", "assertions"],
        "pack_id": "pack",
        "pack_version": "1.0.0",
        "seed": 7,
        "paraphrase_model": None,
        "paraphrase_model_canonical": None,
        "held_out_hit": False,
        "src": "pack:tpl",
        "metadata": canonical_json(
            {
                "language": "en",
                "expt_name": "expt",
                "base_task_id": None,
                "surface_source": "template",
                "profile_hash": None,
            }
        ),
    }
    row.update(overrides)
    return row


def _canonical(**overrides: Any) -> CanonicalExportRow:
    return CanonicalExportRow.from_benchmark_row(_row(**overrides))


def _format_report(**overrides: Any) -> ExportFormatReport:
    payload: dict[str, Any] = {
        "format": "bfcl_json",
        "format_schema_version": BFCL_JSON_SCHEMA_VERSION,
        "path": "exports/bfcl.json",
        "rows": 1,
        "content_hash": HASH,
        "equivalent": True,
    }
    payload.update(overrides)
    return ExportFormatReport.model_validate(payload)


def _bundle(**overrides: Any) -> NemoEvaluatorBundle:
    payload: dict[str, Any] = {
        "task_name": "bfcl_pack",
        "dataset_file": "dataset.jsonl",
        "dataset_schema_file": "dataset.schema.json",
        "metadata_file": "metadata.json",
        "evaluator_config_file": "evaluator.yaml",
        "system_prompt_file": "system_prompts.json",
        "record_count": 1,
        "dataset_content_hash": HASH,
        "scoring": NemoEvaluatorScoring(
            metrics=EXPORT_SCORING_METRICS,
            call_order_policies=("strict",),
        ),
        "source": NemoEvaluatorSource(
            benchmark_file="benchmark.parquet",
            benchmark_content_hash=OTHER_HASH,
            pack_id="pack",
            pack_version="1.0.0",
            expt_name="expt",
        ),
    }
    payload.update(overrides)
    return NemoEvaluatorBundle.model_validate(payload)


# --- registries -------------------------------------------------------------


def test_the_contract_row_fields_track_the_published_parquet_schema() -> None:
    assert list(BENCHMARK_ROW_FIELDS) == list(benchmark_schema().names)
    assert set(EXPORT_TRUTH_FIELDS) <= set(BENCHMARK_ROW_FIELDS)
    assert set(EXPORT_FORMAT_SCHEMA_VERSIONS) == set(EXPORT_FORMATS)


def test_every_export_format_declares_a_schema_version() -> None:
    assert EXPORT_FORMAT_SCHEMA_VERSIONS["bfcl_json"] == BFCL_JSON_SCHEMA_VERSION
    assert EXPORT_FORMAT_SCHEMA_VERSIONS["nemo_evaluator_bundle"] == NEMO_EVALUATOR_SCHEMA_VERSION
    assert CanonicalExportRow.model_fields["schema_version"].default == EXPORT_CONTRACT_VERSION
    assert BfclJsonRecord.model_fields["schema_version"].default == BFCL_JSON_SCHEMA_VERSION
    assert NemoEvaluatorRecord.model_fields["schema_version"].default == NEMO_EVALUATOR_SCHEMA_VERSION
    assert NemoEvaluatorSource.model_fields["schema_version"].default == NEMO_EVALUATOR_SCHEMA_VERSION
    assert ExportValidationReport.model_fields["schema_version"].default == EXPORT_CONTRACT_VERSION


def test_the_config_export_keys_match_the_contract_registry() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import _EXPORT_KEYS

    assert set(EXPORT_FORMATS) == set(_EXPORT_KEYS)


# --- JSON strictness --------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("1", 1),
        (1, True),
        (0, False),
        (1, 1.0),
        ([1], [1.0]),
        ({"a": 1}, {"a": True}),
        (None, False),
        (None, ""),
        ({"a": 1}, {"a": 1, "b": 2}),
        ([1, 2], [2, 1]),
    ],
)
def test_json_equality_distinguishes_types_python_equality_conflates(left: Any, right: Any) -> None:
    assert not json_equal(left, right)


def test_json_equality_treats_a_tuple_and_a_list_as_one_json_array() -> None:
    # Pydantic stores sequences as tuples, so a projection must still compare
    # equal to the list a JSON reader produces.
    assert json_equal((1, {"a": [None, True]}), [1, {"a": [None, True]}])


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), {1: "x"}, {"a": object()}, b"bytes", (object(),)],
)
def test_a_value_the_canonical_codec_cannot_round_trip_is_rejected(value: Any) -> None:
    with pytest.raises(ValueError):
        validate_json_value(value, label="value")


def test_an_unsupported_value_never_matches_a_json_value_of_the_same_shape() -> None:
    assert json_type_tag(object()).startswith("unsupported")
    assert not json_equal(object(), object())


def test_arguments_decode_from_the_arrow_map_without_changing_a_type() -> None:
    pairs = [
        ("account_id", '"1"'),
        ("limit", "1"),
        ("verbose", "true"),
        ("note", "null"),
        ("tags", '["ưu tiên",null,2]'),
        ("options", '{"nested":{"enabled":false}}'),
    ]

    decoded = decode_lossless_arguments(pairs, label="arguments")

    assert decoded == {
        "account_id": "1",
        "limit": 1,
        "verbose": True,
        "note": None,
        "tags": ["ưu tiên", None, 2],
        "options": {"nested": {"enabled": False}},
    }
    assert json_type_tag(decoded["account_id"]) == "str"
    assert json_type_tag(decoded["limit"]) == "int"
    assert json_type_tag(decoded["verbose"]) == "bool"


def test_a_repeated_argument_name_in_the_arrow_map_is_refused() -> None:
    # Arrow maps can hold duplicate keys that no JSON object can express, so the
    # decoded object would depend on iteration order.
    with pytest.raises(ValueError, match="repeats argument name"):
        decode_lossless_arguments([("limit", "1"), ("limit", "2")], label="arguments")


@pytest.mark.parametrize("stored", ["1.50", "{'a':1}", "NaN", '{ "a" : 1 }'])
def test_a_stored_argument_that_is_not_canonical_json_is_refused(stored: str) -> None:
    with pytest.raises(ValueError):
        decode_lossless_arguments([("limit", stored)], label="arguments")


def test_a_column_an_export_would_reformat_is_refused() -> None:
    with pytest.raises(ValueError, match="canonical JSON"):
        decode_canonical_json('{"b": 1, "a": 2}', label="tools")


# --- canonical projection ---------------------------------------------------


def test_the_canonical_projection_decodes_a_published_row_losslessly() -> None:
    row = _canonical()

    assert row.schema_version == EXPORT_CONTRACT_VERSION
    assert row.task_id == "pack__tpl__abcdef"
    assert json_equal(row.tools, TOOLS)
    assert row.expected_tool_calls[0].arguments == ARGUMENTS
    assert json_type_tag(row.expected_tool_calls[0].arguments["account_id"]) == "str"
    assert json_type_tag(row.expected_tool_calls[0].arguments["limit"]) == "int"
    assert row.success_assertions == ("assert_balance_reported",)
    assert row.metadata["language"] == "en"


def test_canonical_json_cannot_be_mutated_after_validation() -> None:
    row = _canonical()

    with pytest.raises(TypeError, match="immutable"):
        row.expected_tool_calls[0].arguments["limit"] = "changed"
    with pytest.raises(TypeError, match="immutable"):
        row.tools[0]["function"]["name"] = "changed"
    with pytest.raises(TypeError, match="immutable"):
        row.metadata["language"] = "changed"
    with pytest.raises(TypeError):
        row.tools[0]["function"]["parameters"]["required"] += ("another",)


def test_a_row_that_is_not_the_published_schema_is_refused() -> None:
    row = _row()
    row.pop("success_assertions")
    row["extra"] = 1

    with pytest.raises(ValueError, match="missing=\\['success_assertions'\\], extra=\\['extra'\\]"):
        CanonicalExportRow.from_benchmark_row(row)


def test_a_row_whose_call_count_disagrees_with_its_calls_is_refused() -> None:
    with pytest.raises(ValidationError, match="reports 2 tool calls but exports 1"):
        _canonical(num_tool_calls=2)


def test_message_tool_calls_must_match_expected_calls() -> None:
    messages = _row()["messages"]
    messages[2]["tool_calls"][0]["function"]["arguments"] = canonical_json({**ARGUMENTS, "limit": "1"})

    with pytest.raises(ValidationError, match="do not match expected_tool_calls"):
        _canonical(messages=messages)


def test_a_row_expecting_a_tool_it_does_not_expose_is_refused() -> None:
    # An unanswerable task would score as a model failure on every run.
    with pytest.raises(ValidationError, match="the exported tools do not expose"):
        _canonical(tools=canonical_json([TOOLS[1]]))


def test_a_row_whose_expected_calls_left_trace_order_is_refused() -> None:
    calls = [
        {
            "turn_index": 0,
            "call_group": 1,
            "position_in_group": 0,
            "function_name": "get_balance",
            "arguments": encode_arguments({"account_id": "1"}),
        },
        {
            "turn_index": 0,
            "call_group": 0,
            "position_in_group": 0,
            "function_name": "list_cards",
            "arguments": encode_arguments({}),
        },
    ]

    with pytest.raises(ValidationError, match="trace order"):
        _canonical(expected_tool_calls=calls, num_tool_calls=2)


def test_a_row_repeating_one_trace_position_is_refused() -> None:
    call = {
        "turn_index": 0,
        "call_group": 0,
        "position_in_group": 0,
        "function_name": "get_balance",
        "arguments": encode_arguments({"account_id": "1"}),
    }

    with pytest.raises(ValidationError, match="repeat a trace position"):
        _canonical(expected_tool_calls=[call, dict(call)], num_tool_calls=2)


@pytest.mark.parametrize(
    ("call_order", "prefix", "complaint"),
    [
        ("prefix", None, "without call_order_prefix"),
        ("strict", 1, "without call_order: prefix"),
        ("prefix", 2, "exceeds its 1 required tools"),
    ],
)
def test_an_inconsistent_call_order_policy_is_refused(call_order: str, prefix: int | None, complaint: str) -> None:
    with pytest.raises(ValidationError, match=complaint):
        _canonical(call_order=call_order, call_order_prefix=prefix)


def test_a_row_whose_required_tools_fingerprint_drifted_is_refused() -> None:
    with pytest.raises(ValidationError, match="required_tools_fingerprint"):
        _canonical(required_tools_fingerprint=canonical_json(["list_cards"]))


def test_a_row_carrying_an_unversioned_surface_metadata_key_is_refused() -> None:
    # A new surface field must revise the contract rather than ride into an export.
    with pytest.raises(ValidationError, match="canonical keys"):
        _canonical(
            metadata=canonical_json(
                {
                    "language": "en",
                    "expt_name": "expt",
                    "base_task_id": None,
                    "surface_source": "template",
                    "profile_hash": None,
                    "reviewer": "someone",
                }
            )
        )


def test_a_row_exposing_one_tool_name_twice_is_refused() -> None:
    with pytest.raises(ValidationError, match="duplicate function name"):
        _canonical(tools=canonical_json([TOOLS[0], TOOLS[0]]))


@pytest.mark.parametrize(
    ("message", "complaint"),
    [
        ({"role": "user", "content": None, "tool_calls": None, "tool_call_id": None}, "requires content"),
        (
            {"role": "tool", "content": "{}", "tool_calls": None, "tool_call_id": None},
            "requires the tool_call_id",
        ),
        (
            {"role": "assistant", "content": None, "tool_calls": None, "tool_call_id": None},
            "either content or tool calls",
        ),
        (
            {"role": "assistant", "content": "text", "tool_calls": None, "tool_call_id": "call_0"},
            "cannot answer a tool call",
        ),
    ],
)
def test_a_malformed_message_is_refused(message: dict[str, Any], complaint: str) -> None:
    with pytest.raises(ValidationError, match=complaint):
        _canonical(messages=[message])


def test_an_assistant_message_cannot_carry_both_prose_and_tool_calls() -> None:
    messages = _row()["messages"]
    messages[2] = {**messages[2], "content": "Let me check."}

    with pytest.raises(ValidationError, match="cannot also carry content"):
        _canonical(messages=messages)


def test_a_row_that_asks_the_model_nothing_is_refused() -> None:
    with pytest.raises(ValidationError, match="exports no messages"):
        _canonical(messages=[], expected_tool_calls=[], num_tool_calls=0)


def test_the_canonical_row_refuses_a_coerced_scalar() -> None:
    with pytest.raises(ValidationError):
        _canonical(variant_index="0")


# --- compatibility formats --------------------------------------------------


def test_both_formats_preserve_every_truth_field_of_the_canonical_row() -> None:
    canonical = [_canonical()]
    bfcl = [BfclJsonRecord.from_canonical(row) for row in canonical]
    nemo = [NemoEvaluatorRecord.from_canonical(row) for row in canonical]

    assert validate_export_equivalence(canonical, bfcl) == []
    assert validate_export_equivalence(canonical, nemo) == []
    assert bfcl[0].truth_payload() == canonical[0].truth_payload()
    assert nemo[0].truth_payload() == canonical[0].truth_payload()


def test_the_bfcl_record_keys_on_id_and_keeps_provenance_out_of_the_truth_fields() -> None:
    record = BfclJsonRecord.from_canonical(_canonical())

    assert record.id == "pack__tpl__abcdef"
    assert record.truth_payload()["task_id"] == record.id
    assert record.metadata.pack_version == "1.0.0"
    assert record.metadata.surface["language"] == "en"
    assert "metadata" not in record.truth_payload()


def test_the_bfcl_record_emits_the_pinned_upstream_v4_pair() -> None:
    record = BfclJsonRecord.from_canonical(_canonical())

    assert record.upstream_schema_version == BFCL_UPSTREAM_SCHEMA_VERSION
    question = record.question_record(conversation_plan(_canonical()).call_group_payload)
    assert question["id"] == "pack__tpl__abcdef"
    assert question["question"] == [
        [
            {"role": "system", "content": "You use tools."},
            {"role": "user", "content": "Balance of 1?"},
        ]
    ]
    assert question["function"] == [tool["function"] for tool in TOOLS]
    assert question["x-nemotron"]["tools"] == TOOLS
    assert question["x-nemotron"]["success_assertions"] == ["assert_balance_reported"]
    assert question["x-nemotron"]["expected_tool_calls"][0]["call_group"] == 0
    assert record.ground_truth_record(conversation_plan(_canonical()).calls_by_user_turn) == {
        "id": "pack__tpl__abcdef",
        "ground_truth": [
            ["get_balance(account_id='1', limit=1, verbose=True, note=None)"],
        ],
    }


def test_the_bfcl_record_refuses_a_turn_grouping_that_is_not_its_own() -> None:
    record = BfclJsonRecord.from_canonical(_canonical())
    calls = conversation_plan(_canonical()).calls_by_user_turn

    with pytest.raises(ValueError, match="1 user turn"):
        record.ground_truth_record([*calls, ()])
    with pytest.raises(ValueError, match="does not account for exactly its expected calls"):
        record.ground_truth_record([()])


def test_the_bfcl_record_refuses_a_call_grouping_that_is_not_its_own() -> None:
    record = BfclJsonRecord.from_canonical(_canonical())
    (group,) = conversation_plan(_canonical()).call_group_payload

    with pytest.raises(ValueError, match="must partition its 1 expected call"):
        record.question_record([])
    with pytest.raises(ValueError, match="parallel exactly when"):
        record.question_record([{**group, "is_parallel": True}])
    with pytest.raises(ValueError, match="unexpected keys"):
        record.question_record([{**group, "extra": 1}])
    with pytest.raises(ValueError, match="issues nothing"):
        record.question_record([{**group, "calls": []}])


def test_an_export_that_changed_an_argument_type_is_reported_per_field() -> None:
    canonical = [_canonical()]
    record = BfclJsonRecord.from_canonical(canonical[0])
    drifted = record.model_copy(
        update={
            "expected_tool_calls": (
                record.expected_tool_calls[0].model_copy(update={"arguments": {**ARGUMENTS, "limit": "1"}}),
            )
        }
    )

    failures = validate_export_equivalence(canonical, [drifted])

    assert failures == [
        ExportRowFailure(
            task_id="pack__tpl__abcdef",
            reason="truth_field_changed",
            field="expected_tool_calls",
        )
    ]


def test_an_export_that_dropped_an_assertion_is_reported() -> None:
    canonical = [_canonical()]
    record = BfclJsonRecord.from_canonical(canonical[0]).model_copy(update={"success_assertions": ()})

    failures = validate_export_equivalence(canonical, [record])

    assert [failure.field for failure in failures] == ["success_assertions"]


def test_a_missing_or_extra_row_is_reported_instead_of_a_field_diff() -> None:
    canonical = [_canonical(), _canonical(task_id="pack__tpl__other")]
    exported = [
        BfclJsonRecord.from_canonical(canonical[0]),
        BfclJsonRecord.from_canonical(canonical[0]).model_copy(update={"id": "pack__tpl__ghost"}),
    ]

    failures = validate_export_equivalence(canonical, exported)

    assert [(failure.task_id, failure.reason) for failure in failures] == [
        ("pack__tpl__ghost", "unexpected_row"),
        ("pack__tpl__other", "missing_row"),
    ]


def test_an_export_that_reordered_the_publication_set_is_reported() -> None:
    canonical = [_canonical(), _canonical(task_id="pack__tpl__other")]
    exported = [BfclJsonRecord.from_canonical(row) for row in reversed(canonical)]

    failures = validate_export_equivalence(canonical, exported)

    assert {failure.reason for failure in failures} == {"row_order_changed"}
    assert {failure.task_id for failure in failures} == {"pack__tpl__abcdef", "pack__tpl__other"}


def test_a_repeated_row_in_an_export_is_reported() -> None:
    canonical = [_canonical()]
    record = BfclJsonRecord.from_canonical(canonical[0])

    failures = validate_export_equivalence(canonical, [record, record])

    assert [failure.reason for failure in failures] == ["duplicate_row"]


# --- evaluator bundle -------------------------------------------------------


def test_the_bundle_declares_the_scoring_dimensions_rather_than_leaving_them_implied() -> None:
    bundle = _bundle()

    assert bundle.schema_version == NEMO_EVALUATOR_SCHEMA_VERSION
    assert bundle.source.schema_version == NEMO_EVALUATOR_SCHEMA_VERSION
    assert bundle.dataset_schema_file == "dataset.schema.json"
    assert bundle.metadata_file == "metadata.json"
    assert bundle.evaluator_config_file == "evaluator.yaml"
    assert bundle.system_prompt_file == "system_prompts.json"
    assert bundle.scoring.metrics == EXPORT_SCORING_METRICS
    assert bundle.scoring.argument_match == "canonical_json_exact"
    assert bundle.scoring.call_order_policies == ("strict",)


def test_a_bundle_scoring_no_dimension_is_refused() -> None:
    with pytest.raises(ValidationError, match="at least one scoring metric"):
        NemoEvaluatorScoring(metrics=(), call_order_policies=("strict",))


def test_a_bundle_scoring_without_an_order_policy_is_refused() -> None:
    with pytest.raises(ValidationError, match="at least one call_order policy"):
        NemoEvaluatorScoring(metrics=("arguments",), call_order_policies=())


def test_bundle_scoring_metrics_are_normalized_to_the_registry_order() -> None:
    scoring = NemoEvaluatorScoring(
        metrics=("task_success", "tool_selection"),
        call_order_policies=("prefix", "any"),
    )

    assert scoring.metrics == ("tool_selection", "task_success")
    assert scoring.call_order_policies == ("any", "prefix")


@pytest.mark.parametrize("task_name", ["BFCL_pack", "bfcl pack", "_pack", "bfcl/pack"])
def test_a_bundle_task_name_a_launcher_cannot_carry_is_refused(task_name: str) -> None:
    with pytest.raises(ValidationError, match="task_name"):
        _bundle(task_name=task_name)


@pytest.mark.parametrize(
    "field",
    [
        "dataset_file",
        "dataset_schema_file",
        "metadata_file",
        "evaluator_config_file",
        "system_prompt_file",
    ],
)
@pytest.mark.parametrize("path", ["/abs/file", "../file", "sub\\file", "C:/file"])
def test_a_bundle_path_that_escapes_the_export_directory_is_refused(field: str, path: str) -> None:
    with pytest.raises(ValidationError, match=field):
        _bundle(**{field: path})


def test_a_bundle_hash_that_is_not_a_sha256_digest_is_refused() -> None:
    with pytest.raises(ValidationError):
        _bundle(dataset_content_hash="deadbeef")


def test_a_bundle_requires_distinct_nonempty_artifacts() -> None:
    with pytest.raises(ValidationError):
        _bundle(record_count=0)
    with pytest.raises(ValidationError, match="distinct paths"):
        _bundle(metadata_file="dataset.jsonl")


# --- export report ----------------------------------------------------------


def test_a_report_passes_only_when_every_format_matched() -> None:
    report = ExportValidationReport(
        benchmark_rows=1,
        benchmark_content_hash=HASH,
        formats=(_format_report(),),
        status="passed",
    )

    assert report.schema_version == EXPORT_CONTRACT_VERSION
    assert report.status == "passed"


def test_a_passing_format_must_contain_every_benchmark_row() -> None:
    with pytest.raises(ValidationError, match="row counts disagree"):
        ExportValidationReport(
            benchmark_rows=200,
            benchmark_content_hash=HASH,
            formats=(_format_report(rows=1),),
            status="passed",
        )


def test_a_report_claiming_success_for_a_failed_format_is_refused() -> None:
    failed = _format_report(
        equivalent=False,
        failures=(ExportRowFailure(task_id="t", reason="missing_row"),),
    )

    with pytest.raises(ValidationError, match="must be 'failed'"):
        ExportValidationReport(
            benchmark_rows=1,
            benchmark_content_hash=HASH,
            formats=(failed,),
            status="passed",
        )


def test_a_format_that_recorded_failures_cannot_call_itself_equivalent() -> None:
    with pytest.raises(ValidationError, match="partial match is a failed export"):
        _format_report(failures=(ExportRowFailure(task_id="t", reason="missing_row"),))


def test_a_format_report_at_the_wrong_schema_version_is_refused() -> None:
    with pytest.raises(ValidationError, match="schema"):
        _format_report(format_schema_version="0.9")


def test_a_report_listing_one_format_twice_is_refused() -> None:
    with pytest.raises(ValidationError, match="at most one entry per format"):
        ExportValidationReport(
            benchmark_rows=1,
            benchmark_content_hash=HASH,
            formats=(_format_report(), _format_report()),
            status="passed",
        )


def test_a_field_level_failure_must_name_a_truth_field() -> None:
    with pytest.raises(ValidationError, match="truth_field_changed"):
        ExportRowFailure(task_id="t", reason="truth_field_changed", field="tier")
    with pytest.raises(ValidationError, match="concerns the whole row"):
        ExportRowFailure(task_id="t", reason="missing_row", field="tools")


# --- manifest lineage -------------------------------------------------------


def test_the_manifest_reports_a_disabled_format_as_unwritten_not_as_passing() -> None:
    section = export_manifest_section(enabled={name: False for name in EXPORT_FORMATS}, report=None)

    assert section == {
        "schema_version": EXPORT_CONTRACT_VERSION,
        "evaluated": False,
        "status": None,
        "benchmark_rows": 0,
        "benchmark_content_hash": None,
        "validation_report": None,
        "formats": {name: {"enabled": False} for name in EXPORT_FORMATS},
    }


def test_the_manifest_pins_the_hash_and_schema_of_every_written_format() -> None:
    report = ExportValidationReport(
        benchmark_rows=2,
        benchmark_content_hash=OTHER_HASH,
        formats=(_format_report(rows=2),),
        status="passed",
    )

    section = export_manifest_section(
        enabled={"bfcl_json": True, "nemo_evaluator_bundle": False},
        report=report,
        validation_report_path="export_validation_report.json",
        validation_report_content_hash=HASH,
    )

    assert section["evaluated"] is True
    assert section["status"] == "passed"
    assert section["benchmark_rows"] == 2
    assert section["benchmark_content_hash"] == OTHER_HASH
    assert section["validation_report"] == {
        "path": "export_validation_report.json",
        "content_hash": HASH,
    }
    assert section["formats"]["bfcl_json"] == {
        "enabled": True,
        "schema_version": BFCL_JSON_SCHEMA_VERSION,
        "path": "exports/bfcl.json",
        "rows": 2,
        "content_hash": HASH,
        "equivalent": True,
    }
    assert section["formats"]["nemo_evaluator_bundle"] == {"enabled": False}


def test_an_enabled_format_without_a_report_stops_the_manifest() -> None:
    with pytest.raises(ValueError, match="enabled but carry no export report"):
        export_manifest_section(enabled={"bfcl_json": True}, report=None)


def test_a_format_written_without_being_enabled_stops_the_manifest() -> None:
    report = ExportValidationReport(
        benchmark_rows=1,
        benchmark_content_hash=HASH,
        formats=(_format_report(),),
        status="passed",
    )

    with pytest.raises(ValueError, match="written without being enabled"):
        export_manifest_section(
            enabled={"bfcl_json": False},
            report=report,
            validation_report_path="export_validation_report.json",
            validation_report_content_hash=HASH,
        )


def test_an_unknown_export_format_stops_the_manifest() -> None:
    with pytest.raises(ValueError, match="unknown export format"):
        export_manifest_section(enabled={"parquet_v2": True}, report=None)


def test_a_written_report_without_complete_lineage_stops_the_manifest() -> None:
    report = ExportValidationReport(
        benchmark_rows=1,
        benchmark_content_hash=HASH,
        formats=(_format_report(),),
        status="passed",
    )

    with pytest.raises(ValueError, match="requires its path and content hash"):
        export_manifest_section(enabled={"bfcl_json": True}, report=report)


def test_a_report_cannot_make_disabled_exports_look_evaluated() -> None:
    report = ExportValidationReport(
        benchmark_rows=0,
        benchmark_content_hash=HASH,
        formats=(),
        status="passed",
    )

    with pytest.raises(ValueError, match="every export format is disabled"):
        export_manifest_section(
            enabled={name: False for name in EXPORT_FORMATS},
            report=report,
            validation_report_path="export_validation_report.json",
            validation_report_content_hash=HASH,
        )


def test_non_boolean_manifest_enablement_is_refused() -> None:
    with pytest.raises(ValueError, match="must use booleans"):
        export_manifest_section(enabled={"bfcl_json": 1}, report=None)  # type: ignore[dict-item]
