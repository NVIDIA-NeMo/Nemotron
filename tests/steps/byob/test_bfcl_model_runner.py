from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nemotron.steps.byob.runtime.benchmark_families.bfcl.model_runner import (
    read_structured_responses,
)


def _write_dataset(path: Path, response: pa.Array) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "request_id": pa.array(["mcp_assertion_specs"]),
                # A structured column arrives beside the seed columns it was rendered
                # from, and those are large; reading has to be able to skip them.
                "evidence": pa.array(["{}"]),
                "response": response,
            }
        ),
        path / "batch_00000.parquet",
    )
    return path


def test_a_struct_field_null_in_every_row_still_reads_back(tmp_path: Path) -> None:
    # What the assertion-spec draft looks like once every predicate is over the call
    # trace: none names an `argument`, and a bundle whose probes observed everything
    # leaves each `blocked_on` empty. Both fields are then null for the whole column, and
    # pandas' Arrow backend sizes such a child to one element rather than to the length of
    # the struct it belongs to, so the frame Data Designer returns cannot be iterated.
    assertions = pa.array(
        [
            [
                {"assertion_id": "create_dispute_called", "argument": None, "blocked_on": []},
                {"assertion_id": "get_balance_called", "argument": None, "blocked_on": []},
            ]
        ],
        type=pa.list_(
            pa.struct(
                [
                    ("assertion_id", pa.string()),
                    ("argument", pa.null()),
                    ("blocked_on", pa.list_(pa.null())),
                ]
            )
        ),
    )
    dataset = _write_dataset(
        tmp_path / "parquet-files",
        pa.array([{"assertions": assertions[0].as_py()}], type=pa.struct([("assertions", assertions.type)])),
    )

    records = read_structured_responses(dataset)

    assert len(records) == 1
    assert records[0]["request_id"] == "mcp_assertion_specs"
    drafted = records[0]["response"]["assertions"]
    assert [entry["assertion_id"] for entry in drafted] == [
        "create_dispute_called",
        "get_balance_called",
    ]
    # The null field has to survive as a stated absence, because a predicate over the
    # trace names no argument and the schema it is validated against expects that.
    assert all(entry["argument"] is None for entry in drafted)
    assert all(entry["blocked_on"] == [] for entry in drafted)
    # Seed columns are skipped rather than paid for a second time.
    assert set(records[0]) == {"request_id", "response"}
