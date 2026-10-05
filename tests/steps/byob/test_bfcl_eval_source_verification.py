"""Verify that evaluation reads the committed benchmark publication.

Every fixture here builds a real publication tree — two parquets written with the
benchmark schema, a manifest that declares them, and a resolvable oracle pack —
because the whole point of this stage is that it reads bytes back from disk. A
test that mocked the read would prove nothing about drift.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import OraclePackRef
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval import (
    EVAL_CONFIG_SCHEMA_VERSION,
    SOURCE_VERIFICATION_FAILURE_FILE,
    SOURCE_VERIFICATION_REPORT_FILE,
    BenchmarkHashMismatchError,
    BenchmarkSchemaMismatchError,
    BfclEvalConfig,
    OraclePackDriftError,
    OracleResourceMismatchError,
    PublicationSemanticsError,
    SourceChangedDuringEvalError,
    SourceManifestDriftError,
    SourceManifestSchemaError,
    SourceTaskIndexError,
    SourceVerificationError,
    TranslationLineageError,
    assert_source_unchanged,
    describe_source_verification_error,
    load_eval_config,
    source_verification_report,
    verify_eval_source,
    write_source_failure_diagnostic,
    write_source_verification_report,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.source_contract import (
    BFCL_TRANSLATION_CONTRACT_VERSION,
    TRANSLATION_PRESERVED_FIELDS,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    CanonicalExportRow,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    pack_file_hashes,
    pack_fingerprint,
    resolve_declared_pack_paths,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.publication_contract import (
    PUBLICATION_BENCHMARK_TABLE,
    PUBLICATION_CONTRACT_VERSION,
    RAW_BENCHMARK_TABLE,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import (
    benchmark_schema,
    canonical_json,
    encode_arguments,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.translation import (
    protected_translation_field,
)

IMMUTABLE_REVISION = "9f2c1b7d4e6a8c0b2d4f6a8c0e2b4d6f8a0c2e4b"
ORACLE_CLOCK = "2026-01-01T00:00:00+00:00"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_balance",
            "description": "Return one account balance.",
            "parameters": {
                "type": "object",
                "properties": {"account_id": {"type": "string"}},
                "required": ["account_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {"name": "list_cards", "description": "List cards.", "parameters": None},
    },
]

BACKEND_SOURCE = '''\
"""A minimal oracle backend, complete enough to be driven by a worker."""

STATE = {"calls": 0}


def list_tools():
    return ["get_balance", "list_cards"]


def reset(*, ctx, fixtures=None):
    STATE["calls"] = 0
    return None


def call_tool(name, arguments, *, ctx):
    STATE["calls"] += 1
    return {"ok": True}


def get_state():
    return dict(STATE)
'''


def _hash(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _file_hash(path: Path) -> str:
    return _hash(path.read_bytes())


def _row(task_id: str, **overrides: Any) -> dict[str, Any]:
    """One published benchmark row, exactly as Stage 12 writes it to parquet."""
    arguments = {"account_id": "1"}
    row: dict[str, Any] = {
        "task_id": task_id,
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
                        "function": {"name": "get_balance", "arguments": canonical_json(arguments)},
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
                "arguments": encode_arguments(arguments),
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
        "pack_id": "test_pack",
        "pack_version": "1.0.0",
        "seed": 7,
        "paraphrase_model": None,
        "paraphrase_model_canonical": None,
        # No held-out policy ran for this fixture, so the column records no
        # verdict; a ``false`` here would claim a scan that never happened.
        "held_out_hit": None,
        "src": "test_pack:tpl",
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


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows, schema=benchmark_schema()), path)


def _write_pack(pack_dir: Path, *, endpoint: bool = False, extra_backend: bool = False) -> Path:
    """Write an oracle pack whose manifest names everything it declares."""
    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "tools.json").write_text(canonical_json(TOOLS) + "\n", encoding="utf-8")
    (pack_dir / "task_templates.yaml").write_text(yaml.safe_dump([{"id": "tpl"}]), encoding="utf-8")
    (pack_dir / "assertions.py").write_text(
        "def assert_balance_reported(*, state, trace, task, ctx):\n    return None\n", encoding="utf-8"
    )
    (pack_dir / "validation_cases.yaml").write_text(yaml.safe_dump([{"id": "case"}]), encoding="utf-8")
    if extra_backend:
        (pack_dir / "other_backend.py").write_text(BACKEND_SOURCE, encoding="utf-8")
    if endpoint:
        (pack_dir / "ca.pem").write_text("-----BEGIN CERTIFICATE-----\n", encoding="utf-8")
        (pack_dir / "endpoint_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "protocol_version": "bfcl-oracle-http-v1",
                    "base_url": "https://oracle.example.com/bfcl",
                    "auth": {"bearer_token_env": "BFCL_ORACLE_TOKEN"},
                    "expected": {
                        "oracle_id": "test_pack",
                        "oracle_version": "1.0.0",
                        "content_digest": _hash(b"oracle-content"),
                    },
                    "tls": {"ca_bundle_path": "ca.pem"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        resource = pack_dir / "endpoint_config.yaml"
        paths_block = {"endpoint": "endpoint_config.yaml"}
    else:
        (pack_dir / "backend.py").write_text(BACKEND_SOURCE, encoding="utf-8")
        resource = pack_dir / "backend.py"
        paths_block = {"backend": "backend.py"}
    (pack_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {"pack_id": "test_pack", "version": "1.0.0", "paths": paths_block},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return resource


def _disabled_roles() -> dict[str, Any]:
    """The ``models`` block Stage 12 writes when no lineage role is enabled."""
    return {
        role: {
            "alias": None,
            "provider": None,
            "model_identity": None,
            "canonical_id": None,
            "config_hash": None,
            "enabled": False,
        }
        for role in ("profile", "paraphrase", "surface_judge")
    }


def _enabled_role(
    canonical_id: str,
    *,
    provider: str = "nvidia",
    model: str = "org/surface-model",
    source: str | None = None,
    revision: str | None = None,
    weights_digest: str | None = None,
) -> dict[str, Any]:
    """One enabled role, shaped exactly as ``_model_role`` records it."""
    return {
        "alias": "surface",
        "provider": provider,
        "model_identity": {
            "source": source,
            "model": model,
            "revision": revision,
            "weights_digest": weights_digest,
        },
        "canonical_id": canonical_id,
        "config_hash": _hash(b"role-config"),
        "enabled": True,
    }


@dataclass
class Publication:
    """A publication tree plus the pack it was generated from."""

    run_dir: Path
    pack_dir: Path
    resource: Path
    endpoint: bool

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "run_manifest.json"

    @property
    def benchmark_path(self) -> Path:
        return self.run_dir / PUBLICATION_BENCHMARK_TABLE

    @property
    def raw_path(self) -> Path:
        return self.run_dir / RAW_BENCHMARK_TABLE

    def manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def rewrite_manifest(self, mutate: Any) -> None:
        document = self.manifest()
        mutate(document)
        self.manifest_path.write_text(
            json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def _publish(
    tmp_path: Path,
    *,
    name: str = "published",
    run_id: str = "expt-20260819T090000000000Z-abc-1",
    published_rows: list[dict[str, Any]] | None = None,
    raw_rows: list[dict[str, Any]] | None = None,
    endpoint: bool = False,
    extra_backend: bool = False,
    gold_eligible: bool = True,
    held_out_evaluated: bool = False,
    ordering: str = "raw_order",
    manifest_overrides: dict[str, Any] | None = None,
    published_hash: str | None = None,
    models: dict[str, Any] | None = None,
) -> Publication:
    """Write a publication tree that verification is expected to accept."""
    root = tmp_path / name
    run_dir = root / "expt"
    run_dir.mkdir(parents=True)
    pack_dir = root / "oracle_pack"
    resource = _write_pack(pack_dir, endpoint=endpoint, extra_backend=extra_backend)
    pack_paths = resolve_declared_pack_paths(
        OraclePackRef(manifest_path=pack_dir / "manifest.yaml"), (pack_dir,)
    )
    fingerprint = pack_fingerprint(pack_paths)
    file_hashes = pack_file_hashes(pack_paths)

    published = published_rows if published_rows is not None else [_row("test_pack__tpl__aaaaaaaaaaaaaaaa")]
    raw = raw_rows if raw_rows is not None else [*published, _row("test_pack__tpl__bbbbbbbbbbbbbbbb")]
    publication = Publication(run_dir=run_dir, pack_dir=pack_dir, resource=resource, endpoint=endpoint)
    _write_parquet(publication.benchmark_path, published)
    _write_parquet(publication.raw_path, raw)

    benchmark_hash = published_hash or _file_hash(publication.benchmark_path)
    manifest: dict[str, Any] = {
        "schema_version": "1.1",
        "run_id": run_id,
        "created_at": "2026-08-19T09:00:00+00:00",
        "oracle_clock": ORACLE_CLOCK,
        "generation_config_hash": _hash(b"generation-config"),
        "resolved_config_hash": _hash(b"resolved-config"),
        "lineage_policy": "strict_separation",
        "tier": "gold" if gold_eligible else "silver",
        "gold_eligible": gold_eligible,
        "pack": {
            "pack_id": "test_pack",
            "version": "1.0.0",
            "content_hash": f"sha256:{fingerprint}",
            "files": dict(file_hashes),
        },
        "oracle": {
            "kind": "endpoint" if endpoint else "python",
            "endpoint_metadata": (
                {
                    "protocol_version": "bfcl-oracle-http-v1",
                    "oracle_id": "test_pack",
                    "oracle_version": "1.0.0",
                    "content_digest": _hash(b"oracle-content"),
                }
                if endpoint
                else None
            ),
        },
        "models": models if models is not None else _disabled_roles(),
        "held_out": {"contract_version": "1.0", "source": None, "evaluated": held_out_evaluated},
        "publication": {
            "schema_version": PUBLICATION_CONTRACT_VERSION,
            "raw": {
                "file": RAW_BENCHMARK_TABLE,
                "rows": len(raw),
                "content_hash": _file_hash(publication.raw_path),
                "contains": "schema_valid_and_replay_valid_rows",
            },
            "published": {
                "file": PUBLICATION_BENCHMARK_TABLE,
                "rows": len(published),
                "content_hash": benchmark_hash,
                "surface_gate": "surface_quality",
                "dedup_balancing_applied": ordering == "selection_rank",
                "held_out_evaluated": held_out_evaluated,
                "ordering": ordering,
            },
            "restated_fields": [],
            "verified": True,
        },
        "artifacts": {
            "benchmark_parquet": {"content_hash": benchmark_hash},
            "benchmark_raw_parquet": {"content_hash": _file_hash(publication.raw_path)},
        },
    }
    manifest.update(manifest_overrides or {})
    publication.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return publication


def _config_data(
    publication: Publication,
    output_dir: Path,
    *,
    modes: list[str] | None = None,
    with_oracle: bool = True,
    resource: Path | None = None,
    translation_manifest: Path | None = None,
) -> dict[str, Any]:
    contract = output_dir.parent / "contracts" / "eval_spec.md"
    contract.parent.mkdir(parents=True, exist_ok=True)
    contract.write_text("argument match: schema then canonical\n", encoding="utf-8")
    return {
        "schema_version": EVAL_CONFIG_SCHEMA_VERSION,
        "config_status": "resolved",
        "source_run_manifest": str(publication.manifest_path),
        "source_oracle": (
            {
                "kind": "endpoint" if publication.endpoint else "python",
                "pack_manifest": str(publication.pack_dir / "manifest.yaml"),
                "resource": str(resource or publication.resource),
            }
            if with_oracle
            else None
        ),
        "translation_manifest": str(translation_manifest) if translation_manifest else None,
        "eval": {"mode": modes or ["trace"]},
        "scoring": {
            "contract": str(contract),
            "argument_matching": "schema_then_canonical",
            "insert_declared_defaults": True,
            "respect_call_order": True,
            "respect_call_group": True,
            "allow_llm_repair": False,
            "task_success": "all_applicable_gates",
        },
        "limits": {
            "max_turns": 5,
            "tool_timeout_s": 30.0,
            "candidate_timeout_s": 60.0,
            "episode_timeout_s": 120.0,
            "max_parallel_tasks": 1,
            "max_retries": 2,
        },
        "candidates": [
            {
                "alias": "candidate_a",
                "model": "nemotron-route-a",
                "provider": "nvidia",
                "provider_api_version": "v1",
                "api": {"base_url": "https://integrate.example.com/v1", "api_key_env": "NVIDIA_API_KEY"},
                "model_identity": {
                    "source": "huggingface",
                    "model": "org/model-a",
                    "revision": IMMUTABLE_REVISION,
                    "weights_digest": None,
                },
                "inference": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 1024,
                    "seed": 42,
                    "tool_choice": "auto",
                },
            }
        ],
        "contamination": {"enforce": True, "on_violation": "fail_run", "comparison_set": "common_intersection"},
        "publication": {"requested": True, "require_same_task_ids": True},
        "outputs": {
            "output_dir": str(output_dir),
            "write_task_results": True,
            "write_eval_manifest": True,
            "cache_candidate_responses": True,
            "cache_tool_results": True,
        },
    }


def _load(tmp_path: Path, data: dict[str, Any], *, name: str = "eval") -> BfclEvalConfig:
    config_dir = tmp_path / name
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "eval_config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_eval_config(path)


def _resolved(tmp_path: Path, **publish: Any) -> tuple[Publication, BfclEvalConfig]:
    """A publication plus a trace-only eval config resolved against it."""
    modes = publish.pop("modes", None)
    with_oracle = publish.pop("with_oracle", True)
    resource = publish.pop("resource", None)
    publication = _publish(tmp_path, **publish)
    data = _config_data(
        publication,
        tmp_path / "eval_out",
        modes=modes,
        with_oracle=with_oracle,
        resource=resource,
    )
    return publication, _load(tmp_path, data)


# --- the accepted source -----------------------------------------------------


def test_a_committed_publication_verifies_trace_only(tmp_path: Path) -> None:
    publication, config = _resolved(tmp_path)

    source = verify_eval_source(config)

    assert source.source_run_id == "expt-20260819T090000000000Z-abc-1"
    assert source.claim_scope == "trace_only"
    assert source.benchmark.file == PUBLICATION_BENCHMARK_TABLE
    assert source.benchmark.content_hash == _file_hash(publication.benchmark_path)
    assert source.benchmark.rows == 1
    assert source.raw_benchmark.rows == 2
    assert source.publication.published_rows == 1
    assert source.task_ids == ("test_pack__tpl__aaaaaaaaaaaaaaaa",)
    assert source.task_index.gold_task_ids == source.task_ids
    assert source.task_index.turn_policy_counts == {"single_turn": 1}
    assert source.verification_identity.startswith("sha256:")
    assert source.exposures == ()
    assert {check.name for check in source.checks} == {
        "commit_marker",
        "published_bytes",
        "publication_semantics",
        "task_index",
        "oracle_pack",
        "model_exposure",
    }
    assert all(check.status == "passed" for check in source.checks)


def test_a_verified_source_is_frozen(tmp_path: Path) -> None:
    _, config = _resolved(tmp_path)

    source = verify_eval_source(config)

    with pytest.raises(Exception):
        source.benchmark = None  # type: ignore[misc]


def test_a_trace_only_run_does_not_need_the_oracle_at_all(tmp_path: Path) -> None:
    publication, config = _resolved(tmp_path, with_oracle=False)
    # Delete the pack outright: a trace-only claim rests on the gold trace in the
    # published table, not on the ability to replay it.
    shutil.rmtree(publication.pack_dir)

    source = verify_eval_source(config)

    assert source.oracle is None
    assert source.claim_scope == "trace_only"


def test_a_source_that_moved_intact_keeps_its_verification_identity(tmp_path: Path) -> None:
    publication, config = _resolved(tmp_path)
    original = verify_eval_source(config)

    shutil.copytree(publication.run_dir.parent, tmp_path / "relocated")
    moved = Publication(
        run_dir=tmp_path / "relocated" / "expt",
        pack_dir=tmp_path / "relocated" / "oracle_pack",
        resource=tmp_path / "relocated" / "oracle_pack" / publication.resource.name,
        endpoint=False,
    )
    relocated = verify_eval_source(
        _load(tmp_path, _config_data(moved, tmp_path / "eval_out_moved"), name="eval_moved")
    )

    assert relocated.verification_identity == original.verification_identity
    assert relocated.publication_dir != original.publication_dir


# --- the commit marker -------------------------------------------------------


def test_a_manifest_that_disappeared_is_reported_as_drift(tmp_path: Path) -> None:
    publication, config = _resolved(tmp_path)
    publication.manifest_path.unlink()

    with pytest.raises(SourceManifestDriftError, match="no longer exists"):
        verify_eval_source(config)


def test_a_manifest_edited_after_the_config_resolved_is_refused(tmp_path: Path) -> None:
    publication, config = _resolved(tmp_path)
    publication.rewrite_manifest(lambda document: document.update(tier="silver"))

    with pytest.raises(SourceManifestDriftError, match="changed after the eval config resolved"):
        verify_eval_source(config)


def test_a_manifest_that_is_not_a_json_object_is_refused(tmp_path: Path) -> None:
    publication, config = _resolved(tmp_path)
    publication.manifest_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(SourceManifestSchemaError, match="not a JSON object"):
        verify_eval_source(config)


def test_a_manifest_missing_lineage_fields_is_not_a_commit_marker(tmp_path: Path) -> None:
    publication, config = _resolved(tmp_path)
    document = publication.manifest()
    del document["held_out"]
    del document["oracle_clock"]
    publication.manifest_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(SourceManifestSchemaError) as error:
        verify_eval_source(config)

    assert "held_out" in str(error.value)
    assert "oracle_clock" in str(error.value)


def test_a_manifest_under_another_name_is_not_a_commit_marker(tmp_path: Path) -> None:
    publication = _publish(tmp_path)
    renamed = publication.run_dir / "manifest_copy.json"
    renamed.write_text(publication.manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
    data = _config_data(publication, tmp_path / "eval_out")
    data["source_run_manifest"] = str(renamed)
    config = _load(tmp_path, data)

    with pytest.raises(SourceManifestSchemaError, match="run_manifest.json"):
        verify_eval_source(config)


# --- benchmark integrity -----------------------------------------------------


def test_a_benchmark_whose_bytes_do_not_match_the_manifest_is_refused(tmp_path: Path) -> None:
    _, config = _resolved(tmp_path, published_hash=_hash(b"a hash of nothing on disk"))

    with pytest.raises(BenchmarkHashMismatchError, match="does not match the bytes on disk"):
        verify_eval_source(config)


def test_two_manifest_hash_entries_that_disagree_are_refused(tmp_path: Path) -> None:
    publication = _publish(tmp_path)
    document = publication.manifest()
    document["publication"]["published"]["content_hash"] = _hash(b"another table")
    publication.manifest_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config = _load(tmp_path, _config_data(publication, tmp_path / "eval_out"))

    with pytest.raises(BenchmarkHashMismatchError) as error:
        verify_eval_source(config)

    assert "publication.published.content_hash" in str(error.value)


def test_a_benchmark_removed_after_resolution_is_refused(tmp_path: Path) -> None:
    publication, config = _resolved(tmp_path)
    publication.benchmark_path.unlink()

    with pytest.raises(BenchmarkHashMismatchError, match="not present in the publication tree"):
        verify_eval_source(config)


def test_a_benchmark_replaced_by_a_symlink_is_refused(tmp_path: Path) -> None:
    publication, config = _resolved(tmp_path)
    elsewhere = tmp_path / "elsewhere.parquet"
    elsewhere.write_bytes(publication.benchmark_path.read_bytes())
    publication.benchmark_path.unlink()
    publication.benchmark_path.symlink_to(elsewhere)

    with pytest.raises(BenchmarkHashMismatchError, match="symbolic link"):
        verify_eval_source(config)


def test_a_table_written_with_another_schema_is_refused(tmp_path: Path) -> None:
    publication, config = _resolved(tmp_path)
    pq.write_table(
        pa.table({"task_id": ["test_pack__tpl__aaaaaaaaaaaaaaaa"]}),
        publication.benchmark_path,
    )
    publication.rewrite_manifest(
        lambda document: _restate_published_hash(document, _file_hash(publication.benchmark_path))
    )
    config = _load(tmp_path, _config_data(publication, tmp_path / "eval_out2"), name="eval2")

    with pytest.raises(BenchmarkSchemaMismatchError, match="not written with the benchmark schema"):
        verify_eval_source(config)


def test_a_row_whose_canonical_json_is_corrupt_is_refused(tmp_path: Path) -> None:
    corrupt = _row("test_pack__tpl__aaaaaaaaaaaaaaaa", tools="not json at all")
    _, config = _resolved(tmp_path, published_rows=[corrupt], raw_rows=[corrupt])

    with pytest.raises(BenchmarkSchemaMismatchError):
        verify_eval_source(config)


def test_an_empty_benchmark_is_refused(tmp_path: Path) -> None:
    _, config = _resolved(tmp_path, published_rows=[], raw_rows=[_row("test_pack__tpl__aaaaaaaaaaaaaaaa")])

    with pytest.raises(PublicationSemanticsError):
        verify_eval_source(config)


def test_a_duplicated_task_id_is_refused(tmp_path: Path) -> None:
    duplicated = [_row("test_pack__tpl__aaaaaaaaaaaaaaaa"), _row("test_pack__tpl__aaaaaaaaaaaaaaaa")]
    _, config = _resolved(tmp_path, published_rows=duplicated, raw_rows=duplicated)

    with pytest.raises(PublicationSemanticsError):
        verify_eval_source(config)


@pytest.mark.parametrize(
    ("task_id", "problem"),
    [
        ("test_pack__tpl__a/b", "path character"),
        ("../escape", "starts with a dash or a dot"),
        ("test_pack__tpl__a b", "whitespace"),
        ("test_pack__tpl__a\x00b", "control or format character"),
    ],
)
def test_a_task_id_that_cannot_be_addressed_is_refused(tmp_path: Path, task_id: str, problem: str) -> None:
    unsafe = _row(task_id)
    _, config = _resolved(tmp_path, published_rows=[unsafe], raw_rows=[unsafe])

    with pytest.raises(SourceTaskIndexError, match=problem):
        verify_eval_source(config)


def test_a_pack_authored_in_another_language_keeps_its_task_ids(tmp_path: Path) -> None:
    localized = _row("ngân_hàng__tpl__aaaaaaaaaaaaaaaa", pack_id="test_pack")

    _, config = _resolved(tmp_path, published_rows=[localized], raw_rows=[localized])

    assert verify_eval_source(config).task_ids == ("ngân_hàng__tpl__aaaaaaaaaaaaaaaa",)


def test_published_rows_out_of_raw_order_are_refused(tmp_path: Path) -> None:
    first = _row("test_pack__tpl__aaaaaaaaaaaaaaaa")
    second = _row("test_pack__tpl__bbbbbbbbbbbbbbbb")
    _, config = _resolved(tmp_path, published_rows=[second, first], raw_rows=[first, second])

    with pytest.raises(PublicationSemanticsError, match="order"):
        verify_eval_source(config)


def test_a_published_row_that_differs_from_its_raw_row_is_refused(tmp_path: Path) -> None:
    published = _row("test_pack__tpl__aaaaaaaaaaaaaaaa")
    rewritten = _row("test_pack__tpl__aaaaaaaaaaaaaaaa", success_assertions=[])
    _, config = _resolved(tmp_path, published_rows=[published], raw_rows=[rewritten])

    with pytest.raises(PublicationSemanticsError, match="restates"):
        verify_eval_source(config)


def test_a_published_held_out_row_is_refused(tmp_path: Path) -> None:
    leaked = _row("test_pack__tpl__aaaaaaaaaaaaaaaa", held_out_hit=True)
    _, config = _resolved(
        tmp_path,
        published_rows=[leaked],
        raw_rows=[leaked],
        held_out_evaluated=True,
    )

    with pytest.raises(PublicationSemanticsError, match="held-out"):
        verify_eval_source(config)


def test_a_manifest_that_contradicts_itself_about_held_out_is_refused(tmp_path: Path) -> None:
    publication = _publish(tmp_path)
    document = publication.manifest()
    document["held_out"]["evaluated"] = True
    publication.manifest_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config = _load(tmp_path, _config_data(publication, tmp_path / "eval_out"))

    with pytest.raises(PublicationSemanticsError, match="disagrees"):
        verify_eval_source(config)


def test_a_declared_row_count_that_the_table_does_not_carry_is_refused(tmp_path: Path) -> None:
    publication = _publish(tmp_path)
    document = publication.manifest()
    document["publication"]["published"]["rows"] = 9
    publication.manifest_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config = _load(tmp_path, _config_data(publication, tmp_path / "eval_out"))

    with pytest.raises(PublicationSemanticsError, match="rows"):
        verify_eval_source(config)


def test_an_unverified_publication_is_refused(tmp_path: Path) -> None:
    publication = _publish(tmp_path)
    document = publication.manifest()
    document["publication"]["verified"] = False
    publication.manifest_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config = _load(tmp_path, _config_data(publication, tmp_path / "eval_out"))

    with pytest.raises(PublicationSemanticsError, match="did not verify"):
        verify_eval_source(config)


def test_a_publication_contract_this_build_does_not_verify_is_refused(tmp_path: Path) -> None:
    publication = _publish(tmp_path)
    document = publication.manifest()
    document["publication"]["schema_version"] = "9.9"
    publication.manifest_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config = _load(tmp_path, _config_data(publication, tmp_path / "eval_out"))

    with pytest.raises(PublicationSemanticsError, match="publication contract"):
        verify_eval_source(config)


# --- oracle pack and resource ------------------------------------------------


def test_executable_mode_verifies_the_python_oracle_it_will_replay(tmp_path: Path) -> None:
    publication, config = _resolved(tmp_path, modes=["trace", "executable"])

    source = verify_eval_source(config)

    assert source.claim_scope == "trace_and_executable"
    assert source.executable is True
    assert source.oracle is not None
    assert source.oracle.kind == "python"
    assert source.oracle.interface_probed is True
    assert set(source.oracle.backend_interface) >= {"call_tool", "get_state", "list_tools", "reset"}
    assert source.oracle.actual_pack_content_hash == source.oracle.expected_pack_content_hash
    assert source.oracle.resource_path == publication.resource


def test_a_pack_file_changed_after_generation_fails_before_inference(tmp_path: Path) -> None:
    publication = _publish(tmp_path)
    (publication.pack_dir / "tools.json").write_text("[]\n", encoding="utf-8")
    config = _load(tmp_path, _config_data(publication, tmp_path / "eval_out", modes=["trace", "executable"]))

    with pytest.raises(OraclePackDriftError, match="no longer fingerprints"):
        verify_eval_source(config)


def test_pack_drift_names_the_file_that_changed(tmp_path: Path) -> None:
    """The aggregate proves the pack moved; the report has to say which file did."""
    publication = _publish(tmp_path)
    (publication.pack_dir / "tools.json").write_text("[]\n", encoding="utf-8")
    config = _load(tmp_path, _config_data(publication, tmp_path / "eval_out", modes=["trace", "executable"]))

    with pytest.raises(OraclePackDriftError) as raised:
        verify_eval_source(config)

    assert "changed tree/tools.json" in str(raised.value)


def test_pack_drift_separates_a_documentation_edit_from_an_oracle_change(tmp_path: Path) -> None:
    """A README the manifest never declared still fails the run, and is still reported as itself.

    Nothing stops a backend from reading it, so the run must not proceed. What
    the operator needs is the difference between this and an edited backend,
    which is the difference between restoring a doc and distrusting a score.
    """
    publication = _publish(tmp_path)
    (publication.pack_dir / "README.md").write_text("notes about the pack\n", encoding="utf-8")
    config = _load(tmp_path, _config_data(publication, tmp_path / "eval_out", modes=["trace", "executable"]))

    with pytest.raises(OraclePackDriftError) as raised:
        verify_eval_source(config)

    message = str(raised.value)
    assert "added tree/README.md [not a declared oracle input]" in message
    assert "every declared oracle input is unchanged" in message


def test_pack_drift_admits_when_generation_recorded_no_file_map(tmp_path: Path) -> None:
    """Releases published before per-file recording cannot name the drifted file."""
    publication = _publish(tmp_path)
    document = publication.manifest()
    document["pack"].pop("files")
    publication.manifest_path.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (publication.pack_dir / "tools.json").write_text("[]\n", encoding="utf-8")
    config = _load(tmp_path, _config_data(publication, tmp_path / "eval_out", modes=["trace", "executable"]))

    with pytest.raises(OraclePackDriftError) as raised:
        verify_eval_source(config)

    assert "recorded no per-file hashes" in str(raised.value)


def test_a_backend_edited_after_the_config_resolved_is_refused(tmp_path: Path) -> None:
    publication, config = _resolved(tmp_path, modes=["trace", "executable"])
    publication.resource.write_text(BACKEND_SOURCE + "\n# a late edit\n", encoding="utf-8")

    with pytest.raises(OraclePackDriftError, match="changed after the eval config resolved"):
        verify_eval_source(config)


def test_a_backend_the_pack_manifest_does_not_select_is_refused(tmp_path: Path) -> None:
    publication = _publish(tmp_path, extra_backend=True)
    config = _load(
        tmp_path,
        _config_data(
            publication,
            tmp_path / "eval_out",
            modes=["trace", "executable"],
            resource=publication.pack_dir / "other_backend.py",
        ),
    )

    with pytest.raises(OracleResourceMismatchError, match="not the backend the pack manifest selects"):
        verify_eval_source(config)


def test_a_backend_without_the_runner_interface_is_refused(tmp_path: Path) -> None:
    publication, config = _resolved(tmp_path, modes=["trace", "executable"])
    publication.resource.write_text("def reset(*, ctx, fixtures=None):\n    return None\n", encoding="utf-8")
    # Re-resolve so the resource hash matches; the pack fingerprint is restated
    # too, because this test is about the interface and not about drift.
    _restate_pack(publication)
    config = _load(
        tmp_path,
        _config_data(publication, tmp_path / "eval_out", modes=["trace", "executable"]),
        name="eval_interface",
    )

    with pytest.raises(OracleResourceMismatchError, match="does not expose the backend interface"):
        verify_eval_source(config)


def test_executable_probe_loads_fixtures_without_a_held_out_policy(
    tmp_path: Path,
) -> None:
    publication = _publish(tmp_path)
    fixtures = {"accounts": [{"account_id": "1", "balance": 10}]}
    (publication.pack_dir / "fixtures.json").write_text(
        json.dumps(fixtures),
        encoding="utf-8",
    )
    manifest = yaml.safe_load(
        (publication.pack_dir / "manifest.yaml").read_text(encoding="utf-8")
    )
    manifest["paths"]["fixtures"] = "fixtures.json"
    (publication.pack_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    publication.resource.write_text(
        BACKEND_SOURCE.replace(
            'STATE["calls"] = 0\n    return None',
            'assert fixtures == {"accounts": [{"account_id": "1", "balance": 10}]}\n'
            '    STATE["calls"] = 0\n'
            '    return None',
        ),
        encoding="utf-8",
    )
    _restate_pack(publication)
    config = _load(
        tmp_path,
        _config_data(
            publication,
            tmp_path / "eval_out",
            modes=["trace", "executable"],
        ),
    )

    source = verify_eval_source(config)

    assert source.oracle is not None
    assert source.oracle.interface_probed is True


def test_a_backend_that_cannot_be_imported_is_refused(tmp_path: Path) -> None:
    publication = _publish(tmp_path)
    publication.resource.write_text("import a_module_that_does_not_exist\n", encoding="utf-8")
    _restate_pack(publication)
    config = _load(tmp_path, _config_data(publication, tmp_path / "eval_out", modes=["trace", "executable"]))

    with pytest.raises(OracleResourceMismatchError, match="cannot be imported"):
        verify_eval_source(config)


def test_an_executable_claim_cannot_skip_the_interface_probe(tmp_path: Path) -> None:
    _, config = _resolved(tmp_path, modes=["trace", "executable"])

    with pytest.raises(OracleResourceMismatchError, match="was not interface-probed"):
        verify_eval_source(config, probe_oracle=False)


def test_an_endpoint_oracle_is_verified_without_contacting_it(tmp_path: Path) -> None:
    publication, config = _resolved(tmp_path, endpoint=True, modes=["trace", "executable"])

    source = verify_eval_source(config)

    assert source.oracle is not None
    assert source.oracle.kind == "endpoint"
    assert source.oracle.endpoint is not None
    assert source.oracle.endpoint.oracle_id == "test_pack"
    assert source.oracle.endpoint.base_url.startswith("https://")
    assert source.oracle.interface_probed is False
    assert publication.endpoint is True


def test_an_endpoint_pinning_another_oracle_revision_is_refused(tmp_path: Path) -> None:
    publication = _publish(tmp_path, endpoint=True)
    document = publication.manifest()
    document["oracle"]["endpoint_metadata"]["content_digest"] = _hash(b"another oracle build")
    publication.manifest_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config = _load(tmp_path, _config_data(publication, tmp_path / "eval_out", modes=["trace", "executable"]))

    with pytest.raises(OracleResourceMismatchError, match="oracle identity"):
        verify_eval_source(config)


def test_an_endpoint_whose_ca_bundle_is_missing_is_refused(tmp_path: Path) -> None:
    publication, config = _resolved(tmp_path, endpoint=True, modes=["trace", "executable"])
    (publication.pack_dir / "ca.pem").unlink()

    with pytest.raises(OracleResourceMismatchError, match="does not resolve into a complete oracle pack"):
        verify_eval_source(config)


def test_an_oracle_kind_that_does_not_match_the_source_run_is_refused(tmp_path: Path) -> None:
    publication = _publish(tmp_path)
    document = publication.manifest()
    document["oracle"]["kind"] = "endpoint"
    publication.manifest_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    data = _config_data(publication, tmp_path / "eval_out")
    data["source_oracle"]["kind"] = "python"
    config_path = tmp_path / "eval" / "eval_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    # The config contract already refuses a kind the source run did not use, so
    # this pair never reaches verification.
    with pytest.raises(Exception, match="oracle"):
        load_eval_config(config_path)


# --- translation lineage -----------------------------------------------------


def _translation(
    tmp_path: Path,
    publication: Publication,
    *,
    rows: list[dict[str, Any]] | None = None,
    document_overrides: dict[str, Any] | None = None,
    declared_hash: str | None = None,
) -> Path:
    """Write a translated benchmark plus the manifest that claims it."""
    directory = tmp_path / "translated"
    directory.mkdir(parents=True, exist_ok=True)
    translated_rows = rows if rows is not None else [_translate_row(_row("test_pack__tpl__aaaaaaaaaaaaaaaa"))]
    table_path = directory / "benchmark.vi.parquet"
    _write_parquet(table_path, translated_rows)
    source_rows = pq.read_table(publication.benchmark_path).to_pylist()
    source_ids = [row["task_id"] for row in source_rows]
    translated_by_id = {row["task_id"]: row for row in translated_rows}
    source_manifest = publication.manifest()
    source_manifest_hash = _file_hash(publication.manifest_path)
    stage_cache = directory / "stage_cache"
    stage_cache.mkdir(exist_ok=True)
    evidence_paths = {
        "translation_units": stage_cache / "translation_units.parquet",
        "backtranslation_units": stage_cache / "backtranslation_units.parquet",
        "quality_metrics": stage_cache / "quality_metrics.parquet",
    }
    field_paths: list[str] = []
    original_texts: list[str] = []
    source_texts: list[str] = []
    translated_texts: list[str] = []
    for source_row in source_rows:
        task_id = source_row["task_id"]
        translated_row = translated_by_id.get(task_id, source_row)
        source_model = CanonicalExportRow.from_benchmark_row(source_row)

        def add_field(path: str, source_text: str, translated_text: str) -> None:
            field_paths.append(path)
            original_texts.append(source_text)
            source_texts.append(
                protected_translation_field(
                    source_model,
                    source_text,
                    path=path,
                )
            )
            translated_texts.append(
                protected_translation_field(
                    source_model,
                    translated_text,
                    path=path,
                )
            )

        for index, message in enumerate(source_row["messages"]):
            if message["content"] and (
                message["role"] in {"system", "user"} or (message["role"] == "assistant" and not message["tool_calls"])
            ):
                path = f"tasks/{task_id}/messages/{index}/content"
                add_field(
                    path,
                    str(message["content"]),
                    str(translated_row["messages"][index]["content"]),
                )
        if source_row["intent"]:
            path = f"tasks/{task_id}/intent"
            add_field(
                path,
                str(source_row["intent"]),
                str(translated_row["intent"]),
            )
        source_tools = json.loads(source_row["tools"])
        translated_tools = json.loads(translated_row["tools"])
        for index, tool in enumerate(source_tools):
            description = tool["function"].get("description")
            if isinstance(description, str) and description:
                path = f"tasks/{task_id}/tools/{index}/function/description"
                add_field(
                    path,
                    description,
                    str(translated_tools[index]["function"]["description"]),
                )
    translation_ids = [
        _hash(
            canonical_json(
                {
                    "contract": "bfcl-translation-fields/1.0",
                    "source_manifest": source_manifest_hash,
                    "path": path,
                    "text": original_texts[index],
                }
            ).encode("utf-8")
        )
        for index, path in enumerate(field_paths)
    ]
    forward_rows = [
        {
            "translation_id": identifier,
            "field_path": path,
            "text": source_text,
            "source_language_code": "en",
            "target_language_code": "vi",
            "translation": translated_text,
        }
        for identifier, path, source_text, translated_text in zip(
            translation_ids,
            field_paths,
            source_texts,
            translated_texts,
            strict=True,
        )
    ]
    backward_rows = [
        {
            "translation_id": identifier,
            "field_path": path,
            "text": translated_text,
            "source_language_code": "vi",
            "target_language_code": "en",
            "translation": source_text,
        }
        for identifier, path, source_text, translated_text in zip(
            translation_ids,
            field_paths,
            source_texts,
            translated_texts,
            strict=True,
        )
    ]
    quality_rows = [
        {
            "translation_id": identifier,
            "field_path": path,
            "source_text": source_text,
            "translation": translated_text,
            "backtranslation": source_text,
            "score_chrf": 100.0,
            "score_chrf_passed": True,
            "is_quality_metric_passed": True,
        }
        for identifier, path, source_text, translated_text in zip(
            translation_ids,
            field_paths,
            source_texts,
            translated_texts,
            strict=True,
        )
    ]
    for name, evidence_rows in (
        ("translation_units", forward_rows),
        ("backtranslation_units", backward_rows),
        ("quality_metrics", quality_rows),
    ):
        pq.write_table(
            pa.Table.from_pylist(evidence_rows),
            evidence_paths[name],
        )
    source_pack = source_manifest["pack"]
    document: dict[str, Any] = {
        "schema_version": "1.0",
        "translation_contract": BFCL_TRANSLATION_CONTRACT_VERSION,
        "source_run_id": source_manifest["run_id"],
        "source_run_manifest_content_hash": source_manifest_hash,
        "source_benchmark_content_hash": _file_hash(publication.benchmark_path),
        "source_oracle_pack": {
            "pack_id": source_pack["pack_id"],
            "version": source_pack["version"],
            "content_hash": source_pack["content_hash"],
        },
        "source_language": "en",
        "language": "vi",
        "benchmark": {
            "file": table_path.name,
            "rows": len(translated_rows),
            "content_hash": declared_hash or _file_hash(table_path),
        },
        "task_ids_hash": _hash(canonical_json(source_ids).encode("utf-8")),
        "field_policy": {
            "flattening_contract": "bfcl-translation-fields/1.0",
            "preserved_fields": list(TRANSLATION_PRESERVED_FIELDS),
            "localized_fields": [
                "messages.system.content",
                "messages.user.content",
                "messages.assistant_without_tool_calls.content",
                "intent",
                "metadata.language",
                "tools.function.description",
            ],
            "field_paths_hash": _hash(canonical_json(field_paths).encode("utf-8")),
            "unit_count": len(field_paths),
        },
        "protected_tokens": {
            "contract": "bfcl-protected-tokens/1.0",
            "occurrences": sum(text.count("__BFCL_PROTECTED_") for text in source_texts),
            "fields_with_tokens": sum("__BFCL_PROTECTED_" in text for text in source_texts),
        },
        "model": {
            "provider": "test",
            "model": "translator-v1",
            "canonical_id": "translator-v1",
            "source": "test-registry",
            "revision": IMMUTABLE_REVISION,
            "weights_digest": None,
            "config_hash": _hash(b"translator config"),
        },
        "contamination": {
            "role": "translator",
            "scope": "all_translated_rows",
            "task_ids_hash": _hash(canonical_json(source_ids).encode("utf-8")),
            "task_count": len(source_ids),
            "model_canonical_id": "translator-v1",
        },
        "quality": {
            "backtranslation": True,
            "metrics": [{"type": "chrf", "threshold": 0}],
            "row_filtering": False,
        },
        "localization_validation": {
            "contract": "bfcl-localization-validation/1.0",
            "model_role": "translator",
            "deterministic_fixes": {
                "line_endings": "lf",
                "trailing_whitespace": "removed",
                "unicode": "NFC",
            },
            "minimum_changed_fraction": 0.01,
            "changed_fraction": sum(
                source != translated for source, translated in zip(source_texts, translated_texts, strict=True)
            )
            / len(source_texts),
            "forbidden_patterns": [],
            "required_script": None,
            "executable_replay": "truth_projection_and_parquet_readback",
        },
        "artifacts": {
            name: {
                "file": f"stage_cache/{path.name}",
                "rows": len(field_paths),
                "content_hash": _file_hash(path),
            }
            for name, path in evidence_paths.items()
        },
    }
    document.update(document_overrides or {})
    document["translation_id"] = _hash(canonical_json(document).encode("utf-8"))
    manifest_path = directory / "translation_manifest.json"
    manifest_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _translate_row(row: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Translate only the surface a translation is allowed to change."""
    translated = dict(row)
    messages = [dict(message) for message in row["messages"]]
    messages[1] = {**messages[1], "content": "Số dư của 1?"}
    translated["messages"] = messages
    translated["intent"] = "kiểm_tra_số_dư"
    metadata = json.loads(row["metadata"])
    metadata["language"] = "vi"
    translated["metadata"] = canonical_json(metadata)
    translated.update(overrides)
    return translated


def test_a_translation_of_this_source_verifies(tmp_path: Path) -> None:
    publication = _publish(tmp_path)
    manifest_path = _translation(tmp_path, publication)
    config = _load(
        tmp_path,
        _config_data(publication, tmp_path / "eval_out", translation_manifest=manifest_path),
    )

    source = verify_eval_source(config)

    assert source.translation is not None
    assert source.translation.language == "vi"
    assert source.translation.task_ids_hash == source.task_index.task_ids_hash
    assert source.evaluation_benchmark.file == "benchmark.vi.parquet"
    assert source.benchmark.file == PUBLICATION_BENCHMARK_TABLE


def test_a_translation_without_the_bfcl_content_addressed_contract_is_refused(
    tmp_path: Path,
) -> None:
    publication = _publish(tmp_path)
    manifest_path = _translation(tmp_path, publication)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    del document["translation_contract"]
    document.pop("translation_id")
    document["translation_id"] = _hash(canonical_json(document).encode("utf-8"))
    manifest_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = _load(
        tmp_path,
        _config_data(
            publication,
            tmp_path / "eval_out",
            translation_manifest=manifest_path,
        ),
    )

    with pytest.raises(TranslationLineageError, match="required BFCL translation contract"):
        verify_eval_source(config)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("model", {}),
        ("field_policy", {}),
        ("protected_tokens", {}),
        ("artifacts", {}),
    ],
)
def test_translation_manifest_requires_complete_provenance_blocks(
    tmp_path: Path,
    field: str,
    replacement: dict[str, Any],
) -> None:
    publication = _publish(tmp_path)
    manifest_path = _translation(tmp_path, publication)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document[field] = replacement
    document.pop("translation_id")
    document["translation_id"] = _hash(canonical_json(document).encode("utf-8"))
    manifest_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = _load(
        tmp_path,
        _config_data(
            publication,
            tmp_path / "eval_out",
            translation_manifest=manifest_path,
        ),
    )

    with pytest.raises(TranslationLineageError, match=field):
        verify_eval_source(config)


def test_translation_manifest_rejects_unknown_quality_metric(tmp_path: Path) -> None:
    publication = _publish(tmp_path)
    manifest_path = _translation(tmp_path, publication)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["quality"]["metrics"] = [{"type": "invented", "threshold": -1}]
    document.pop("translation_id")
    document["translation_id"] = _hash(canonical_json(document).encode("utf-8"))
    manifest_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = _load(
        tmp_path,
        _config_data(
            publication,
            tmp_path / "eval_out",
            translation_manifest=manifest_path,
        ),
    )

    with pytest.raises(TranslationLineageError, match="quality"):
        verify_eval_source(config)


def test_translation_manifest_recomputes_quality_verdicts(tmp_path: Path) -> None:
    publication = _publish(tmp_path)
    manifest_path = _translation(tmp_path, publication)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    quality_path = manifest_path.parent / document["artifacts"]["quality_metrics"]["file"]
    rows = pq.read_table(quality_path).to_pylist()
    rows[0]["score_chrf_passed"] = False
    rows[0]["is_quality_metric_passed"] = False
    pq.write_table(pa.Table.from_pylist(rows), quality_path)
    document["artifacts"]["quality_metrics"]["content_hash"] = _file_hash(quality_path)
    document.pop("translation_id")
    document["translation_id"] = _hash(canonical_json(document).encode("utf-8"))
    manifest_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = _load(
        tmp_path,
        _config_data(
            publication,
            tmp_path / "eval_out",
            translation_manifest=manifest_path,
        ),
    )

    with pytest.raises(TranslationLineageError, match="threshold verdict"):
        verify_eval_source(config)


def test_translation_manifest_binds_declared_languages_to_rows(tmp_path: Path) -> None:
    publication = _publish(tmp_path)
    manifest_path = _translation(tmp_path, publication)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["source_language"] = "fr"
    document.pop("translation_id")
    document["translation_id"] = _hash(canonical_json(document).encode("utf-8"))
    manifest_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = _load(
        tmp_path,
        _config_data(
            publication,
            tmp_path / "eval_out",
            translation_manifest=manifest_path,
        ),
    )

    with pytest.raises(TranslationLineageError, match="language"):
        verify_eval_source(config)


def test_translation_manifest_verifies_evidence_field_identity(tmp_path: Path) -> None:
    publication = _publish(tmp_path)
    manifest_path = _translation(tmp_path, publication)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    quality_path = manifest_path.parent / document["artifacts"]["quality_metrics"]["file"]
    rows = pq.read_table(quality_path).to_pylist()
    rows[0]["field_path"] = "tasks/another-task/intent"
    pq.write_table(pa.Table.from_pylist(rows), quality_path)
    document["artifacts"]["quality_metrics"]["content_hash"] = _file_hash(quality_path)
    document.pop("translation_id")
    document["translation_id"] = _hash(canonical_json(document).encode("utf-8"))
    manifest_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = _load(
        tmp_path,
        _config_data(
            publication,
            tmp_path / "eval_out",
            translation_manifest=manifest_path,
        ),
    )

    with pytest.raises(TranslationLineageError, match="stable path"):
        verify_eval_source(config)


@pytest.mark.parametrize(
    ("mutation", "changed_field"),
    [
        ("call_ids", "messages"),
        ("profile_hash", "metadata"),
        ("system_prompt_id", "system_prompt_id"),
    ],
)
def test_translation_preserves_message_and_lineage_structure(
    tmp_path: Path,
    mutation: str,
    changed_field: str,
) -> None:
    publication = _publish(tmp_path)
    translated = _translate_row(_row("test_pack__tpl__aaaaaaaaaaaaaaaa"))
    if mutation == "call_ids":
        messages = [dict(message) for message in translated["messages"]]
        messages[2] = {
            **messages[2],
            "tool_calls": [
                {
                    **messages[2]["tool_calls"][0],
                    "id": "localized-call-id",
                }
            ],
        }
        messages[3] = {
            **messages[3],
            "tool_call_id": "localized-call-id",
        }
        translated["messages"] = messages
    elif mutation == "profile_hash":
        metadata = json.loads(translated["metadata"])
        metadata["profile_hash"] = "sha256:" + "0" * 64
        translated["metadata"] = canonical_json(metadata)
    else:
        translated["system_prompt_id"] = "localized-system-prompt"
    manifest_path = _translation(tmp_path, publication, rows=[translated])
    config = _load(
        tmp_path,
        _config_data(
            publication,
            tmp_path / "eval_out",
            translation_manifest=manifest_path,
        ),
    )

    with pytest.raises(TranslationLineageError, match=changed_field):
        verify_eval_source(config)


def test_a_translation_may_localize_only_the_tool_function_description(
    tmp_path: Path,
) -> None:
    publication = _publish(tmp_path)
    translated = _translate_row(_row("test_pack__tpl__aaaaaaaaaaaaaaaa"))
    tools = json.loads(translated["tools"])
    tools[0]["function"]["description"] = "Tra cứu số dư tài khoản"
    translated["tools"] = canonical_json(tools)
    manifest_path = _translation(tmp_path, publication, rows=[translated])
    config = _load(
        tmp_path,
        _config_data(
            publication,
            tmp_path / "eval_out",
            translation_manifest=manifest_path,
        ),
    )

    source = verify_eval_source(config)

    assert source.translation is not None


def test_a_translation_may_not_localize_a_tool_parameter_schema(
    tmp_path: Path,
) -> None:
    publication = _publish(tmp_path)
    translated = _translate_row(_row("test_pack__tpl__aaaaaaaaaaaaaaaa"))
    tools = json.loads(translated["tools"])
    tools[0]["function"]["parameters"]["properties"]["account_id"]["type"] = "integer"
    translated["tools"] = canonical_json(tools)
    manifest_path = _translation(tmp_path, publication, rows=[translated])
    config = _load(
        tmp_path,
        _config_data(
            publication,
            tmp_path / "eval_out",
            translation_manifest=manifest_path,
        ),
    )

    with pytest.raises(TranslationLineageError, match="tools"):
        verify_eval_source(config)


def test_a_translation_of_another_run_is_refused(tmp_path: Path) -> None:
    publication = _publish(tmp_path)
    other = _publish(tmp_path, name="other", run_id="expt-20260819T100000000000Z-def-2")
    # The correct manifest hash is kept, so the config contract accepts this
    # translation; verification refuses it because it names another run.
    manifest_path = _translation(
        tmp_path,
        publication,
        document_overrides={"source_run_id": other.manifest()["run_id"]},
    )
    config = _load(
        tmp_path,
        _config_data(publication, tmp_path / "eval_out", translation_manifest=manifest_path),
    )

    with pytest.raises(TranslationLineageError, match="does not derive"):
        verify_eval_source(config)


def test_a_translation_with_a_conflicting_source_manifest_hash_is_refused(tmp_path: Path) -> None:
    publication = _publish(tmp_path)
    # The run id is correct, so config resolution succeeds. Source verification must still
    # reject the second lineage reference instead of accepting contradictory
    # statements about which publication was translated.
    manifest_path = _translation(
        tmp_path,
        publication,
        document_overrides={"source_run_manifest_content_hash": _hash(b"another source manifest")},
    )
    config = _load(
        tmp_path,
        _config_data(publication, tmp_path / "eval_out", translation_manifest=manifest_path),
    )

    with pytest.raises(TranslationLineageError, match="does not derive"):
        verify_eval_source(config)


def test_an_unknown_translation_contract_version_is_refused(tmp_path: Path) -> None:
    publication = _publish(tmp_path)
    manifest_path = _translation(
        tmp_path,
        publication,
        document_overrides={"schema_version": "999"},
    )
    config = _load(
        tmp_path,
        _config_data(publication, tmp_path / "eval_out", translation_manifest=manifest_path),
    )

    with pytest.raises(TranslationLineageError, match="contract this build does not verify"):
        verify_eval_source(config)


def test_a_translation_manifest_row_count_must_match_its_table(tmp_path: Path) -> None:
    publication = _publish(tmp_path)
    manifest_path = _translation(tmp_path, publication)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["benchmark"]["rows"] = 999
    document.pop("translation_id")
    document["translation_id"] = _hash(canonical_json(document).encode("utf-8"))
    manifest_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config = _load(
        tmp_path,
        _config_data(publication, tmp_path / "eval_out", translation_manifest=manifest_path),
    )

    with pytest.raises(TranslationLineageError, match="does not match the translated table"):
        verify_eval_source(config)


def test_a_translated_table_that_does_not_match_its_hash_is_refused(tmp_path: Path) -> None:
    publication = _publish(tmp_path)
    manifest_path = _translation(tmp_path, publication, declared_hash=_hash(b"another translation"))
    config = _load(
        tmp_path,
        _config_data(publication, tmp_path / "eval_out", translation_manifest=manifest_path),
    )

    with pytest.raises(TranslationLineageError, match="does not match the translated table"):
        verify_eval_source(config)


def test_a_translation_that_drops_a_row_is_refused(tmp_path: Path) -> None:
    publication = _publish(
        tmp_path,
        published_rows=[_row("test_pack__tpl__aaaaaaaaaaaaaaaa"), _row("test_pack__tpl__bbbbbbbbbbbbbbbb")],
        raw_rows=[_row("test_pack__tpl__aaaaaaaaaaaaaaaa"), _row("test_pack__tpl__bbbbbbbbbbbbbbbb")],
    )
    manifest_path = _translation(
        tmp_path,
        publication,
        rows=[_translate_row(_row("test_pack__tpl__aaaaaaaaaaaaaaaa"))],
    )
    config = _load(
        tmp_path,
        _config_data(publication, tmp_path / "eval_out", translation_manifest=manifest_path),
    )

    with pytest.raises(TranslationLineageError, match="source task ids"):
        verify_eval_source(config)


def test_a_translation_that_rewrites_the_gold_call_is_refused(tmp_path: Path) -> None:
    publication = _publish(tmp_path)
    rewritten = _translate_row(_row("test_pack__tpl__aaaaaaaaaaaaaaaa"))
    rewritten["expected_tool_calls"] = [
        {
            "turn_index": 0,
            "call_group": 0,
            "position_in_group": 0,
            "function_name": "get_balance",
            "arguments": encode_arguments({"account_id": "2"}),
        }
    ]
    messages = [dict(message) for message in rewritten["messages"]]
    messages[2] = {
        **messages[2],
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "get_balance", "arguments": canonical_json({"account_id": "2"})},
            }
        ],
    }
    rewritten["messages"] = messages
    manifest_path = _translation(tmp_path, publication, rows=[rewritten])
    config = _load(
        tmp_path,
        _config_data(publication, tmp_path / "eval_out", translation_manifest=manifest_path),
    )

    with pytest.raises(TranslationLineageError, match="expected_tool_calls"):
        verify_eval_source(config)


def test_a_translation_manifest_without_the_contract_fails_loudly(tmp_path: Path) -> None:
    publication = _publish(tmp_path)
    manifest_path = _translation(tmp_path, publication)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    del document["task_ids_hash"]
    document.pop("translation_id")
    document["translation_id"] = _hash(canonical_json(document).encode("utf-8"))
    manifest_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config = _load(
        tmp_path,
        _config_data(publication, tmp_path / "eval_out", translation_manifest=manifest_path),
    )

    with pytest.raises(TranslationLineageError, match="task_ids_hash"):
        verify_eval_source(config)


# --- the report --------------------------------------------------------------


def test_the_report_is_written_into_the_eval_output_tree(tmp_path: Path) -> None:
    publication, config = _resolved(tmp_path)
    source = verify_eval_source(config)

    path, content_hash = write_source_verification_report(config, source)

    assert path == config.outputs.output_dir / SOURCE_VERIFICATION_REPORT_FILE
    assert content_hash == _file_hash(path)
    assert not list(publication.run_dir.glob("source_verification*"))
    assert not list(path.parent.glob("*.tmp"))
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["status"] == "passed"
    assert document["verification_identity"] == source.verification_identity
    assert document["claim_scope"] == "trace_only"
    assert document["benchmark"]["content_hash"] == source.benchmark.content_hash
    assert document["publication"]["verified"] is True
    assert document["task_index"]["task_ids"] == list(source.task_ids)
    assert [check["status"] for check in document["checks"]] == ["passed"] * len(source.checks)


def test_the_report_payload_is_deterministic_apart_from_its_timestamp(tmp_path: Path) -> None:
    _, config = _resolved(tmp_path)
    source = verify_eval_source(config)

    first = source_verification_report(source).as_document()
    second = source_verification_report(source).as_document()

    assert first.pop("verified_at") is not None
    assert second.pop("verified_at") is not None
    assert first == second


def test_a_failure_is_recorded_under_a_name_that_cannot_pass_for_success(tmp_path: Path) -> None:
    publication, config = _resolved(tmp_path)
    publication.benchmark_path.unlink()
    with pytest.raises(BenchmarkHashMismatchError) as error:
        verify_eval_source(config)

    path, content_hash = write_source_failure_diagnostic(config, error.value)

    assert path.name == SOURCE_VERIFICATION_FAILURE_FILE
    assert not (config.outputs.output_dir / SOURCE_VERIFICATION_REPORT_FILE).exists()
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["status"] == "failed"
    assert document["error"]["code"] == "eval_source_benchmark_hash_mismatch"
    assert content_hash == _file_hash(path)


def test_a_new_failure_removes_a_stale_passing_report(tmp_path: Path) -> None:
    publication, config = _resolved(tmp_path)
    source = verify_eval_source(config)
    write_source_verification_report(config, source)
    publication.benchmark_path.unlink()
    with pytest.raises(BenchmarkHashMismatchError) as error:
        verify_eval_source(config)

    write_source_failure_diagnostic(config, error.value)

    assert not (config.outputs.output_dir / SOURCE_VERIFICATION_REPORT_FILE).exists()
    assert (config.outputs.output_dir / SOURCE_VERIFICATION_FAILURE_FILE).is_file()


def test_a_new_pass_removes_a_stale_failure_diagnostic(tmp_path: Path) -> None:
    _, config = _resolved(tmp_path)
    source = verify_eval_source(config)
    write_source_failure_diagnostic(
        config,
        SourceVerificationError(
            "source",
            "temporary failure",
            expected="an intact source",
            recovery="verify again",
        ),
    )

    write_source_verification_report(config, source)

    assert (config.outputs.output_dir / SOURCE_VERIFICATION_REPORT_FILE).is_file()
    assert not (config.outputs.output_dir / SOURCE_VERIFICATION_FAILURE_FILE).exists()


def test_a_failure_summary_names_the_code_and_the_artifact(tmp_path: Path) -> None:
    publication, config = _resolved(tmp_path)
    publication.benchmark_path.unlink()

    with pytest.raises(BenchmarkHashMismatchError) as error:
        verify_eval_source(config)

    summary = describe_source_verification_error(error.value)
    assert summary.startswith("[eval_source_benchmark_hash_mismatch]")
    assert PUBLICATION_BENCHMARK_TABLE in summary


def test_a_drift_report_states_both_hashes(tmp_path: Path) -> None:
    publication, config = _resolved(tmp_path)
    publication.rewrite_manifest(lambda document: document.update(tier="silver"))

    with pytest.raises(SourceManifestDriftError) as error:
        verify_eval_source(config)

    report = error.value.as_report()
    assert report["expected"] == config.source.run_manifest.content_hash
    assert report["actual"] == _file_hash(publication.manifest_path)
    assert report["actual"] != report["expected"]


# --- time of check to time of use -------------------------------------------


def test_an_unchanged_source_passes_the_second_pin(tmp_path: Path) -> None:
    _, config = _resolved(tmp_path, modes=["trace", "executable"])
    source = verify_eval_source(config)

    assert assert_source_unchanged(source) is None


def test_a_benchmark_replaced_after_verification_is_detected(tmp_path: Path) -> None:
    publication, config = _resolved(tmp_path)
    source = verify_eval_source(config)
    _write_parquet(publication.benchmark_path, [_row("test_pack__tpl__cccccccccccccccc")])

    with pytest.raises(SourceChangedDuringEvalError, match="changed after the source was verified"):
        assert_source_unchanged(source)


def test_a_manifest_replaced_after_verification_is_detected(tmp_path: Path) -> None:
    publication, config = _resolved(tmp_path)
    source = verify_eval_source(config)
    publication.manifest_path.unlink()

    with pytest.raises(SourceChangedDuringEvalError, match="no longer present"):
        assert_source_unchanged(source)


def test_a_pack_edited_after_verification_is_detected(tmp_path: Path) -> None:
    publication, config = _resolved(tmp_path, modes=["trace", "executable"])
    source = verify_eval_source(config)
    (publication.pack_dir / "tools.json").write_text("[]\n", encoding="utf-8")

    with pytest.raises(SourceChangedDuringEvalError, match="changed after the source was verified"):
        assert_source_unchanged(source)


def test_a_pack_manifest_removed_after_verification_uses_the_toctou_error(tmp_path: Path) -> None:
    publication, config = _resolved(tmp_path, modes=["trace", "executable"])
    source = verify_eval_source(config)
    (publication.pack_dir / "manifest.yaml").unlink()

    with pytest.raises(SourceChangedDuringEvalError, match="can no longer be resolved"):
        assert_source_unchanged(source)


def _restate_published_hash(document: dict[str, Any], content_hash: str) -> None:
    document["publication"]["published"]["content_hash"] = content_hash
    document["artifacts"]["benchmark_parquet"]["content_hash"] = content_hash


def _restate_pack(publication: Publication) -> None:
    """Re-record the pack fingerprint after a test edits a pack file on purpose."""
    paths = resolve_declared_pack_paths(
        OraclePackRef(manifest_path=publication.pack_dir / "manifest.yaml"),
        (publication.pack_dir,),
    )
    document = publication.manifest()
    document["pack"]["content_hash"] = f"sha256:{pack_fingerprint(paths)}"
    document["pack"]["files"] = pack_file_hashes(paths)
    publication.manifest_path.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
