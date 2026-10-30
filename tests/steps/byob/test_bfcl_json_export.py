"""Contract tests for the bfcl_json writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from nemotron.steps.byob.runtime.benchmark_families.bfcl.bfcl_json_export import (
    BFCL_JSON_ANSWER_FILE,
    BFCL_JSON_QUESTION_FILE,
    BFCL_JSON_ROOT,
    BfclJsonArtifact,
    BfclJsonWriteError,
    bfcl_json_record_pair,
    read_bfcl_json,
    write_bfcl_json,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    BFCL_UPSTREAM_SCHEMA_VERSION,
    BfclJsonRecord,
    export_content_hash,
    export_tree_hash,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_projection import (
    CanonicalExportProjection,
    ProjectionSource,
    conversation_plan,
    project_benchmark_rows,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import (
    canonical_json,
    encode_arguments,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "chuyen_khoan",
            "description": "Chuyển khoản nội bộ.",
            "parameters": {
                "type": "object",
                "properties": {"tai_khoan": {"type": "string"}, "so_tien": {"type": "integer"}},
                "required": ["tai_khoan"],
            },
        },
    },
    {
        "type": "function",
        "function": {"name": "liet_ke_the", "description": "Liệt kê thẻ.", "parameters": None},
    },
]


def _system(content: str = "Bạn dùng công cụ.") -> dict[str, Any]:
    return {"role": "system", "content": content, "tool_calls": None, "tool_call_id": None}


def _user(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content, "tool_calls": None, "tool_call_id": None}


def _assistant_text(content: str) -> dict[str, Any]:
    return {"role": "assistant", "content": content, "tool_calls": None, "tool_call_id": None}


def _assistant_calls(calls: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {"name": name, "arguments": canonical_json(arguments)},
            }
            for index, (name, arguments) in enumerate(calls)
        ],
        "tool_call_id": None,
    }


def _tool_result(call_id: str, payload: Any) -> dict[str, Any]:
    return {"role": "tool", "content": canonical_json(payload), "tool_calls": None, "tool_call_id": call_id}


def _expected(name: str, arguments: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    call = {
        "turn_index": 0,
        "call_group": 0,
        "position_in_group": 0,
        "function_name": name,
        "arguments": encode_arguments(arguments),
    }
    call.update(overrides)
    return call


NESTED_ARGUMENTS: dict[str, Any] = {
    "tai_khoan": "1",
    "so_tien": 1,
    "ghi_chu": None,
    "xac_nhan": True,
    "chi_tiet": {"muc": ["a", "b"], "so": 1.5},
}


def _row(task_id: str = "pack__tpl__abcdef", **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "task_id": task_id,
        "template_id": "tpl",
        "variant_index": 0,
        "messages": [
            _system(),
            _user("Chuyển giúp tôi 1 đồng."),
            _assistant_calls([("chuyen_khoan", NESTED_ARGUMENTS)]),
            _tool_result("call_0", {"trang_thai": "đã chuyển"}),
            _assistant_text("Đã chuyển xong."),
        ],
        "tools": canonical_json(TOOLS),
        "expected_tool_calls": [_expected("chuyen_khoan", NESTED_ARGUMENTS)],
        "success_assertions": ["assert_transfer_committed"],
        "fixture_refs": ['["accounts","1"]'],
        "intent": "chuyen_khoan",
        "category": "payments",
        "difficulty": "easy",
        "required_tools": ["chuyen_khoan"],
        "required_tools_fingerprint": canonical_json(["chuyen_khoan"]),
        "tools_present": ["chuyen_khoan", "liet_ke_the"],
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
                "language": "vi",
                "expt_name": "expt",
                "base_task_id": None,
                "surface_source": "template",
                "profile_hash": None,
            }
        ),
    }
    row.update(overrides)
    return row


def _parallel_row(task_id: str = "pack__tpl__parallel") -> dict[str, Any]:
    transfer = {"tai_khoan": "1"}
    cards: dict[str, Any] = {}
    return _row(
        task_id,
        turn_policy="multi_tool",
        num_tool_calls=2,
        required_tools=["chuyen_khoan", "liet_ke_the"],
        required_tools_fingerprint=canonical_json(["chuyen_khoan", "liet_ke_the"]),
        messages=[
            _system(),
            _user("Chuyển tiền và liệt kê thẻ."),
            _assistant_calls([("chuyen_khoan", transfer), ("liet_ke_the", cards)]),
            _tool_result("call_0", {"trang_thai": "ok"}),
            _tool_result("call_1", {"the": []}),
            _assistant_text("Xong."),
        ],
        expected_tool_calls=[
            _expected("chuyen_khoan", transfer, position_in_group=0),
            _expected("liet_ke_the", cards, position_in_group=1),
        ],
    )


def _multi_turn_row(task_id: str = "pack__tpl__multiturn") -> dict[str, Any]:
    transfer = {"tai_khoan": "1"}
    return _row(
        task_id,
        turn_policy="missing_slot",
        is_multi_turn=True,
        messages=[
            _system(),
            _user("Chuyển tiền giúp tôi."),
            _assistant_text("Chuyển tới tài khoản nào?"),
            _user("Tài khoản 1."),
            _assistant_calls([("chuyen_khoan", transfer)]),
            _tool_result("call_0", {"trang_thai": "ok"}),
            _assistant_text("Đã chuyển."),
        ],
        expected_tool_calls=[_expected("chuyen_khoan", transfer, turn_index=1)],
    )


def _callless_row(task_id: str = "pack__tpl__irrelevant") -> dict[str, Any]:
    return _row(
        task_id,
        turn_policy="irrelevant",
        num_tool_calls=0,
        call_order="any",
        required_tools=[],
        required_tools_fingerprint=canonical_json([]),
        success_assertions=[],
        messages=[
            _system(),
            _user("Hôm nay trời thế nào?"),
            _assistant_text("Tôi chỉ hỗ trợ nghiệp vụ ngân hàng."),
        ],
        expected_tool_calls=[],
        validated_by=["schema", "replay"],
    )


def _projection(rows: list[dict[str, Any]] | None = None) -> CanonicalExportProjection:
    rows = rows if rows is not None else [_row()]
    return project_benchmark_rows(
        rows,
        source=ProjectionSource(file="benchmark.parquet", content_hash="sha256:" + "0" * 64, rows=len(rows)),
    )


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_the_writer_produces_two_jsonl_files_joined_by_id(tmp_path: Path) -> None:
    artifact = write_bfcl_json(_projection([_row("t1"), _parallel_row("t2")]), tmp_path)

    assert artifact.format == "bfcl_json"
    assert artifact.upstream_schema_version == BFCL_UPSTREAM_SCHEMA_VERSION
    assert artifact.root == BFCL_JSON_ROOT
    assert artifact.rows == 2
    questions = _lines(tmp_path / BFCL_JSON_QUESTION_FILE)
    answers = _lines(tmp_path / BFCL_JSON_ANSWER_FILE)
    assert [record["id"] for record in questions] == ["t1", "t2"]
    assert [record["id"] for record in answers] == ["t1", "t2"]


def test_the_question_file_never_carries_the_answers(tmp_path: Path) -> None:
    write_bfcl_json(_projection(), tmp_path)

    (question,) = _lines(tmp_path / BFCL_JSON_QUESTION_FILE)
    assert "ground_truth" not in question
    # Upstream fields show only what the model is shown: prompt turns and tools.
    assert set(question) == {"id", "question", "function", "x-nemotron"}
    (turn,) = question["question"]
    assert [message["role"] for message in turn] == ["system", "user"]


def test_oracle_results_stay_out_of_the_prompt_and_the_answer(tmp_path: Path) -> None:
    write_bfcl_json(_projection(), tmp_path)
    (question,) = _lines(tmp_path / BFCL_JSON_QUESTION_FILE)
    (answer,) = _lines(tmp_path / BFCL_JSON_ANSWER_FILE)

    prompt_text = json.dumps(question["question"], ensure_ascii=False)
    assert "đã chuyển" not in prompt_text
    assert "đã chuyển" not in json.dumps(answer, ensure_ascii=False)
    # The recorded result is retained as provenance, so a run stays auditable.
    tool_messages = [message for message in question["x-nemotron"]["messages"] if message["role"] == "tool"]
    assert json.loads(tool_messages[0]["content"]) == {"trang_thai": "đã chuyển"}


def test_provenance_is_carried_beside_the_prompt_not_inside_it(tmp_path: Path) -> None:
    write_bfcl_json(_projection(), tmp_path)

    (question,) = _lines(tmp_path / BFCL_JSON_QUESTION_FILE)
    metadata = question["x-nemotron"]["metadata"]
    assert metadata["pack_id"] == "pack"
    assert metadata["pack_version"] == "1.0.0"
    assert metadata["seed"] == 7
    assert metadata["category"] == "payments"
    assert metadata["difficulty"] == "easy"
    assert metadata["turn_policy"] == "single_turn"
    rendered = json.dumps(question["question"], ensure_ascii=False)
    for leaked in ("pack__tpl__abcdef", "1.0.0", "sp-1", "single_turn"):
        assert leaked not in rendered


def test_the_question_file_carries_openai_compatible_tool_schemas(tmp_path: Path) -> None:
    write_bfcl_json(_projection(), tmp_path)

    (question,) = _lines(tmp_path / BFCL_JSON_QUESTION_FILE)
    assert question["function"] == [tool["function"] for tool in TOOLS]
    assert question["x-nemotron"]["tools"] == TOOLS


def test_a_single_call_is_written_with_its_exact_argument_types(tmp_path: Path) -> None:
    write_bfcl_json(_projection(), tmp_path)

    (answer,) = _lines(tmp_path / BFCL_JSON_ANSWER_FILE)
    expected = (
        "chuyen_khoan(tai_khoan='1', so_tien=1, ghi_chu=None, xac_nhan=True, chi_tiet={'muc': ['a', 'b'], 'so': 1.5})"
    )
    assert answer["ground_truth"] == [[expected]]
    (question,) = _lines(tmp_path / BFCL_JSON_QUESTION_FILE)
    assert question["x-nemotron"]["expected_tool_calls"][0]["arguments"] == NESTED_ARGUMENTS


@pytest.mark.parametrize("argument_name", ["user-id", "class"])
def test_argument_names_that_bfcl_call_strings_cannot_represent_are_refused(
    tmp_path: Path,
    argument_name: str,
) -> None:
    arguments = {argument_name: "1"}
    row = _row(
        messages=[
            _system(),
            _user("Lookup."),
            _assistant_calls([("chuyen_khoan", arguments)]),
            _tool_result("call_0", {"ok": True}),
        ],
        expected_tool_calls=[_expected("chuyen_khoan", arguments)],
    )

    with pytest.raises(BfclJsonWriteError, match="argument name.*cannot represent"):
        write_bfcl_json(_projection([row]), tmp_path)


def test_function_names_that_bfcl_call_strings_cannot_represent_are_refused(tmp_path: Path) -> None:
    arguments = {"value": 1}
    tools = [
        {
            "type": "function",
            "function": {
                "name": "my-tool",
                "description": "Not a Python call target.",
                "parameters": {"type": "object"},
            },
        }
    ]
    row = _row(
        tools=canonical_json(tools),
        tools_present=["my-tool"],
        required_tools=["my-tool"],
        required_tools_fingerprint=canonical_json(["my-tool"]),
        messages=[
            _system(),
            _user("Call it."),
            _assistant_calls([("my-tool", arguments)]),
            _tool_result("call_0", {"ok": True}),
        ],
        expected_tool_calls=[_expected("my-tool", arguments)],
    )

    with pytest.raises(BfclJsonWriteError, match="function 'my-tool'.*cannot be represented"):
        write_bfcl_json(_projection([row]), tmp_path)


def test_parallel_calls_are_written_as_one_group(tmp_path: Path) -> None:
    write_bfcl_json(_projection([_parallel_row()]), tmp_path)

    (question,) = _lines(tmp_path / BFCL_JSON_QUESTION_FILE)
    (group,) = question["x-nemotron"]["call_groups"]
    assert group == {
        "turn_index": 0,
        "call_group": 0,
        "user_turn_index": 0,
        "is_parallel": True,
        "calls": [0, 1],
    }
    (answer,) = _lines(tmp_path / BFCL_JSON_ANSWER_FILE)
    # Upstream's per-turn list is flat, so both calls land in the same turn.
    assert len(answer["ground_truth"]) == 1
    assert len(answer["ground_truth"][0]) == 2


def test_multi_turn_answers_align_with_the_user_turns(tmp_path: Path) -> None:
    write_bfcl_json(_projection([_multi_turn_row()]), tmp_path)

    (question,) = _lines(tmp_path / BFCL_JSON_QUESTION_FILE)
    (answer,) = _lines(tmp_path / BFCL_JSON_ANSWER_FILE)
    assert len(question["question"]) == 2
    assert answer["ground_truth"] == [[], ["chuyen_khoan(tai_khoan='1')"]]
    (group,) = question["x-nemotron"]["call_groups"]
    assert group["turn_index"] == 1
    assert group["user_turn_index"] == 1
    assert not group["is_parallel"]


def test_a_task_that_expects_no_call_still_exports_its_turn(tmp_path: Path) -> None:
    write_bfcl_json(_projection([_callless_row()]), tmp_path)

    (question,) = _lines(tmp_path / BFCL_JSON_QUESTION_FILE)
    (answer,) = _lines(tmp_path / BFCL_JSON_ANSWER_FILE)
    assert question["x-nemotron"]["call_groups"] == []
    assert question["x-nemotron"]["success_assertions"] == []
    assert answer["ground_truth"] == [[]]


def test_every_ordering_policy_survives_the_writer(tmp_path: Path) -> None:
    rows = [
        _row("strict", call_order="strict"),
        _row("any", call_order="any"),
        _row(
            "prefix",
            call_order="prefix",
            call_order_prefix=1,
        ),
    ]

    write_bfcl_json(_projection(rows), tmp_path)

    questions = {record["id"]: record["x-nemotron"] for record in _lines(tmp_path / BFCL_JSON_QUESTION_FILE)}
    assert questions["strict"]["call_order"] == "strict"
    assert questions["strict"]["call_order_prefix"] is None
    assert questions["any"]["call_order"] == "any"
    assert questions["prefix"]["call_order"] == "prefix"
    assert questions["prefix"]["call_order_prefix"] == 1


def test_vietnamese_text_is_written_readable_rather_than_escaped(tmp_path: Path) -> None:
    write_bfcl_json(_projection(), tmp_path)

    text = (tmp_path / BFCL_JSON_QUESTION_FILE).read_text(encoding="utf-8")
    assert "Chuyển giúp tôi 1 đồng." in text
    assert "\\u" not in text


def test_two_writes_of_one_projection_produce_the_same_bytes(tmp_path: Path) -> None:
    rows = [_row("t1"), _parallel_row("t2"), _multi_turn_row("t3")]
    first = write_bfcl_json(_projection(rows), tmp_path / "first")
    second = write_bfcl_json(_projection(rows), tmp_path / "second")

    assert first.content_hash == second.content_hash
    for name in first.files:
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    assert (tmp_path / "first" / BFCL_JSON_QUESTION_FILE).read_bytes().endswith(b"\n")


def test_the_tree_hash_notices_a_changed_file(tmp_path: Path) -> None:
    artifact = write_bfcl_json(_projection(), tmp_path)
    path = tmp_path / BFCL_JSON_ANSWER_FILE
    path.write_text(path.read_text(encoding="utf-8").replace("chuyen_khoan", "liet_ke_the"), encoding="utf-8")

    assert export_tree_hash(tmp_path, artifact.files) != artifact.content_hash


def test_the_tree_hash_notices_two_files_swapping_places(tmp_path: Path) -> None:
    artifact = write_bfcl_json(_projection(), tmp_path)
    question = tmp_path / BFCL_JSON_QUESTION_FILE
    answer = tmp_path / BFCL_JSON_ANSWER_FILE
    question_bytes = question.read_bytes()
    question.write_bytes(answer.read_bytes())
    answer.write_bytes(question_bytes)

    assert export_tree_hash(tmp_path, artifact.files) != artifact.content_hash


def test_a_tree_hash_needs_at_least_one_distinct_file(tmp_path: Path) -> None:
    write_bfcl_json(_projection(), tmp_path)

    with pytest.raises(ValueError, match="wrote no file"):
        export_tree_hash(tmp_path, [])
    with pytest.raises(ValueError, match="distinct files"):
        export_tree_hash(tmp_path, [BFCL_JSON_QUESTION_FILE, BFCL_JSON_QUESTION_FILE])


def test_content_hash_normalizes_names_before_indexing_bytes() -> None:
    assert export_content_hash({" dataset.jsonl ": b"{}\n"}) == export_content_hash({"dataset.jsonl": b"{}\n"})
    with pytest.raises(ValueError, match="after path normalization"):
        export_content_hash({"dataset.jsonl": b"one", " dataset.jsonl ": b"two"})
    with pytest.raises(TypeError, match="content must be bytes"):
        export_content_hash({"dataset.jsonl": "{}\n"})  # type: ignore[dict-item]


def test_a_record_cannot_be_written_with_another_task_s_plan() -> None:
    record = BfclJsonRecord.from_canonical(_projection().rows[0])
    other = conversation_plan(_projection([_parallel_row()]).rows[0])

    with pytest.raises(BfclJsonWriteError, match="was given the conversation plan of"):
        bfcl_json_record_pair(record, other)


def test_the_export_reads_back_as_it_was_written(tmp_path: Path) -> None:
    rows = [_row("t1"), _multi_turn_row("t2")]
    artifact = write_bfcl_json(_projection(rows), tmp_path)

    questions, answers = read_bfcl_json(tmp_path, artifact)

    assert [record["id"] for record in questions] == ["t1", "t2"]
    assert [record["id"] for record in answers] == ["t1", "t2"]


def test_a_blank_or_malformed_line_is_refused(tmp_path: Path) -> None:
    artifact = write_bfcl_json(_projection(), tmp_path)
    path = tmp_path / BFCL_JSON_QUESTION_FILE
    original = path.read_text(encoding="utf-8")

    path.write_text(original + "\n", encoding="utf-8")
    with pytest.raises(BfclJsonWriteError, match="is blank"):
        read_bfcl_json(tmp_path, artifact)

    path.write_text(original + "{not json\n", encoding="utf-8")
    with pytest.raises(BfclJsonWriteError, match="not valid JSON"):
        read_bfcl_json(tmp_path, artifact)

    path.write_text(original + "[1]\n", encoding="utf-8")
    with pytest.raises(BfclJsonWriteError, match="not a JSON object"):
        read_bfcl_json(tmp_path, artifact)


def test_a_missing_export_file_is_refused(tmp_path: Path) -> None:
    artifact = write_bfcl_json(_projection(), tmp_path)
    (tmp_path / BFCL_JSON_ANSWER_FILE).unlink()

    with pytest.raises(BfclJsonWriteError, match="is missing"):
        read_bfcl_json(tmp_path, artifact)


def test_a_task_written_to_only_one_file_is_refused(tmp_path: Path) -> None:
    artifact = write_bfcl_json(_projection([_row("t1"), _row("t2")]), tmp_path)
    path = tmp_path / BFCL_JSON_ANSWER_FILE
    path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")

    with pytest.raises(BfclJsonWriteError, match="every task is written to both files"):
        read_bfcl_json(tmp_path, artifact)


def test_an_artifact_must_describe_the_benchmark_it_projected() -> None:
    source = ProjectionSource(file="benchmark.parquet", content_hash="sha256:" + "0" * 64, rows=2)

    with pytest.raises(ValidationError, match="every row of the benchmark"):
        BfclJsonArtifact(rows=1, content_hash="sha256:" + "1" * 64, source=source)


def test_an_artifact_may_not_write_answers_into_the_question_file() -> None:
    source = ProjectionSource(file="benchmark.parquet", content_hash="sha256:" + "0" * 64, rows=1)

    with pytest.raises(ValidationError, match="separate files"):
        BfclJsonArtifact(
            rows=1,
            content_hash="sha256:" + "1" * 64,
            source=source,
            question_file=BFCL_JSON_QUESTION_FILE,
            answer_file=BFCL_JSON_QUESTION_FILE,
        )


def test_an_artifact_keeps_its_files_under_its_own_root() -> None:
    source = ProjectionSource(file="benchmark.parquet", content_hash="sha256:" + "0" * 64, rows=1)

    with pytest.raises(ValidationError, match="must live under"):
        BfclJsonArtifact(
            rows=1,
            content_hash="sha256:" + "1" * 64,
            source=source,
            answer_file="exports/elsewhere/answers.jsonl",
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("root", "../bfcl_json"),
        ("question_file", "exports/bfcl_json/../../question.jsonl"),
        ("answer_file", "/tmp/answer.jsonl"),
        ("answer_file", r"exports\bfcl_json\answer.jsonl"),
    ],
)
def test_an_artifact_refuses_paths_that_can_escape_its_root(
    field: str,
    value: str,
) -> None:
    source = ProjectionSource(file="benchmark.parquet", content_hash="sha256:" + "0" * 64, rows=1)

    with pytest.raises(ValidationError, match="relative|POSIX path|inside"):
        BfclJsonArtifact(
            rows=1,
            content_hash="sha256:" + "1" * 64,
            source=source,
            **{field: value},
        )
