"""What candidate pack assembly binds together, and what it refuses to bind.

Assembly is the one step that turns proposals into a loadable pack, so its value is in the
refusals: a pack must never contain a source file the certification did not fingerprint, an
assertion nobody compiled, or a tool the source never published. Intake here is real but
runs at A0, because these bindings do not depend on the probe tier.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BYOB_ROOT
from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json
from nemotron.steps.byob.runtime.pack_authoring.compile_assertions import (
    compile_assertions,
)
from nemotron.steps.byob.runtime.pack_authoring.drafts import AssertionSpecPlan
from nemotron.steps.byob.runtime.pack_authoring.pack_assembly import (
    PackAssemblyError,
    assemble_candidate_pack,
)
from nemotron.steps.byob.runtime.source_adapters.certification import (
    AdapterTier,
    CertificationAuthority,
)
from nemotron.steps.byob.runtime.source_adapters.evidence import PackIdentity
from nemotron.steps.byob.runtime.source_adapters.held_out import (
    build_not_applicable_decision,
)
from nemotron.steps.byob.runtime.source_adapters.intake import run_conventional_intake

TINY = BYOB_ROOT / "data" / "tiny_oracle_pack"
PACK_ID = "tiny_library"
PACK_VERSION = "0.1.0"

_ASSERTION_SPECS: dict[str, Any] = {
    "assertions": [
        {
            "assertion_id": "status_checked",
            "subject": "trace",
            "predicate": "tool_called",
            "target": "get_book_status",
            "argument": None,
            "tool": None,
            "rationale": "A status question is only answered by consulting the catalogue.",
            "blocked_on": [],
        },
        {
            "assertion_id": "checkout_committed",
            "subject": "trace",
            "predicate": "tool_called",
            "target": "checkout_book",
            "argument": None,
            "tool": None,
            "rationale": "A borrow request has to reach the checkout tool.",
            "blocked_on": [],
        },
        {
            "assertion_id": "no_status_checked",
            "subject": "trace",
            "predicate": "tool_not_called",
            "target": "get_book_status",
            "argument": None,
            "tool": None,
            "rationale": "A request the library cannot serve must not query the catalogue.",
            "blocked_on": [],
        },
        {
            "assertion_id": "no_checkout_attempted",
            "subject": "trace",
            "predicate": "tool_not_called",
            "target": "checkout_book",
            "argument": None,
            "tool": None,
            "rationale": "A request the library cannot serve must not lend anything.",
            "blocked_on": [],
        },
    ]
}

_TEMPLATE_ASSERTIONS = {
    "lib_status_single": ["assert_status_checked"],
    "lib_checkout_confirm": ["assert_checkout_committed"],
    "lib_status_parallel": ["assert_status_checked"],
    "lib_irrelevant_renew": [
        "assert_no_status_checked",
        "assert_no_checkout_attempted",
    ],
}


class Session:
    """One certified source plus one drafting output, reused by every case below."""

    def __init__(self, root: Path, evidence_digest: str, package: Path) -> None:
        self.root = root
        self.evidence_digest = evidence_digest
        self.package = package

    @property
    def evidence(self) -> Path:
        return self.root / "intake" / "evidence_bundle.json"

    @property
    def drafts(self) -> Path:
        return self.root / "drafting" / "drafts"


def _source_package(root: Path) -> Path:
    package = root / "library-source"
    package.mkdir(parents=True)
    for name in ("backend.py", "tools.json", "fixtures.json"):
        shutil.copyfile(TINY / name, package / name)
    (package / "dependency-lock.json").write_text(
        json.dumps(
            {"schema_version": "bfcl-python-dependency-lock-v1", "dependencies": []}
        ),
        encoding="utf-8",
    )
    return package


def _write_drafting_output(root: Path, *, evidence_digest: str) -> Path:
    """Stand in for a drafting run: the assembler reads only these two artifacts."""
    drafts = root / "drafts"
    drafts.mkdir(parents=True)
    (drafts / "assertions.py").write_text(
        compile_assertions(AssertionSpecPlan.model_validate(_ASSERTION_SPECS)),
        encoding="utf-8",
    )
    (root / "draft_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": "bfcl-authoring-draft-provenance-v1",
                "evidence": {"bundle_digest": evidence_digest},
                "blocked_on": [],
                "assertions_compiled": True,
            }
        ),
        encoding="utf-8",
    )
    return drafts


@pytest.fixture(scope="module")
def session(tmp_path_factory: pytest.TempPathFactory) -> Session:
    root = tmp_path_factory.mktemp("assembly")
    package = _source_package(root)
    brief = root / "domain-brief.txt"
    brief.write_text(
        "Benchmark deterministic library circulation: status lookup and confirmed loan.",
        encoding="utf-8",
    )
    result = run_conventional_intake(
        {
            "declaration_version": "bfcl-source-declaration-v1",
            "local_python": {"path": package.name},
        },
        root / "intake",
        source_base_dir=root,
        allowed_roots=(root,),
        pack=PackIdentity(pack_id=PACK_ID, version=PACK_VERSION),
        domain_brief_path=brief,
        certification_authority=CertificationAuthority(
            key_id="bfcl-assembly",
            private_key=Ed25519PrivateKey.generate(),
        ),
        held_out_decision=build_not_applicable_decision(
            "The catalogue is public reference data.",
            reviewed_by="reviewer@example.test",
        ),
        required_tier=AdapterTier.A0,
    )
    digest = result.finalized.evidence.bundle_digest
    _write_drafting_output(root / "drafting", evidence_digest=digest)
    return Session(root, digest, package)


def _supplement_document() -> dict[str, Any]:
    templates = yaml.safe_load(
        (TINY / "task_templates.yaml").read_text(encoding="utf-8")
    )
    for template in templates:
        template["success_assertions"] = _TEMPLATE_ASSERTIONS[template["template_id"]]
    manifest = yaml.safe_load((TINY / "manifest.yaml").read_text(encoding="utf-8"))
    return {
        "schema_version": "bfcl-candidate-pack-supplement-v1",
        "languages": manifest["languages"],
        "clock": manifest["clock"],
        "absent_ids": manifest["absent_ids"],
        "assistant_turn_templates": manifest["assistant_turn_templates"],
        "task_templates": templates,
        "validation_cases": yaml.safe_load(
            (TINY / "validation_cases.yaml").read_text(encoding="utf-8")
        ),
    }


def _supplement(path: Path, **overrides: Any) -> Path:
    document = {**_supplement_document(), **overrides}
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _assemble(
    session: Session,
    tmp_path: Path,
    *,
    source_root: Path | None = None,
    draft_root: Path | None = None,
    supplement: Path | None = None,
    output: Path | None = None,
) -> Any:
    return assemble_candidate_pack(
        evidence_path=session.evidence,
        source_root=source_root or session.package,
        draft_root=draft_root or session.drafts,
        supplement_path=supplement or _supplement(tmp_path / "supplement.yaml"),
        output_root=output or tmp_path / "candidate",
    )


def test_assembly_binds_the_pack_to_the_certified_source_and_compiled_drafts(
    session: Session,
    tmp_path: Path,
) -> None:
    assembled = _assemble(session, tmp_path)

    manifest = yaml.safe_load(assembled.manifest_path.read_text(encoding="utf-8"))
    assert manifest["pack_id"] == PACK_ID
    assert manifest["version"] == PACK_VERSION
    assert manifest["paths"]["assertions"] == "assertions.py"
    assert manifest["paths"]["fixtures"] == "fixtures.json"

    # The oracle files are the certified source's own bytes, not a human's retyped copy.
    for name in ("backend.py", "tools.json", "fixtures.json"):
        assert (assembled.pack_root / name).read_bytes() == (
            session.package / name
        ).read_bytes()
    assert (assembled.pack_root / "assertions.py").read_bytes() == (
        session.drafts / "assertions.py"
    ).read_bytes()

    record = assembled.record
    assert record["evidence_digest"] == session.evidence_digest
    # The record names the drafting run it was assembled from, so a pack can be traced
    # back to the proposals a reviewer signed off on rather than only to the evidence.
    assert record["draft_provenance_digest"] == "sha256:" + hashlib.sha256(
        (session.drafts.parent / "draft_provenance.json").read_bytes()
    ).hexdigest()
    assert record["compiled_assertions"] == [
        "assert_checkout_committed",
        "assert_no_checkout_attempted",
        "assert_no_status_checked",
        "assert_status_checked",
    ]
    assert set(record["pack_files"]) == {
        "backend.py",
        "tools.json",
        "fixtures.json",
        "assertions.py",
        "task_templates.yaml",
        "validation_cases.yaml",
        "manifest.yaml",
    }
    unsigned = {
        key: value for key, value in record.items() if key != "record_digest"
    }
    assert record["record_digest"] == sha256_json(unsigned)


def test_assembly_carries_a_reviewed_system_prompt_into_the_pack(
    session: Session,
    tmp_path: Path,
) -> None:
    prompt = "Bạn là trợ lý ngân hàng sử dụng công cụ."

    assembled = _assemble(
        session,
        tmp_path,
        supplement=_supplement(tmp_path / "prompted.yaml", system_prompt=prompt),
    )

    manifest = yaml.safe_load(assembled.manifest_path.read_text(encoding="utf-8"))
    # The prompt has to be readable from the manifest alone, because assembly copies no
    # file the certified source did not publish, so a path here would name nothing.
    assert manifest["system_prompt"] == prompt
    assert "system_prompt_path" not in manifest
    assert set(assembled.record["pack_files"]) == {
        "backend.py",
        "tools.json",
        "fixtures.json",
        "assertions.py",
        "task_templates.yaml",
        "validation_cases.yaml",
        "manifest.yaml",
    }


def test_assembly_refuses_a_source_tree_that_changed_after_certification(
    session: Session,
    tmp_path: Path,
) -> None:
    tampered = tmp_path / "tampered-source"
    shutil.copytree(session.package, tampered)
    (tampered / "backend.py").write_text(
        (tampered / "backend.py").read_text(encoding="utf-8") + "\n# late edit\n",
        encoding="utf-8",
    )

    with pytest.raises(PackAssemblyError) as refusal:
        _assemble(session, tmp_path, source_root=tampered)

    assert refusal.value.code == "source_identity_mismatch"
    assert not (tmp_path / "candidate").exists()


def test_assembly_refuses_drafts_from_another_evidence_revision(
    session: Session,
    tmp_path: Path,
) -> None:
    stale = _write_drafting_output(
        tmp_path / "stale-drafting",
        evidence_digest="sha256:" + "b" * 64,
    )

    with pytest.raises(PackAssemblyError) as refusal:
        _assemble(session, tmp_path, draft_root=stale)

    assert refusal.value.code == "draft_evidence_mismatch"


def test_assembly_refuses_drafts_that_are_still_blocked(
    session: Session,
    tmp_path: Path,
) -> None:
    blocked = _write_drafting_output(
        tmp_path / "blocked-drafting",
        evidence_digest=session.evidence_digest,
    )
    provenance = blocked.parent / "draft_provenance.json"
    document = json.loads(provenance.read_text(encoding="utf-8"))
    document["blocked_on"] = ["checkout_book return shape is unobserved"]
    provenance.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(PackAssemblyError) as refusal:
        _assemble(session, tmp_path, draft_root=blocked)

    assert refusal.value.code == "draft_blocked"
    assert "checkout_book" in refusal.value.detail


def test_assembly_refuses_a_drafting_run_that_compiled_no_assertions(
    session: Session,
    tmp_path: Path,
) -> None:
    uncompiled = _write_drafting_output(
        tmp_path / "uncompiled-drafting",
        evidence_digest=session.evidence_digest,
    )
    provenance = uncompiled.parent / "draft_provenance.json"
    document = json.loads(provenance.read_text(encoding="utf-8"))
    document["assertions_compiled"] = False
    provenance.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(PackAssemblyError) as refusal:
        _assemble(session, tmp_path, draft_root=uncompiled)

    assert refusal.value.code == "compiled_assertions_missing"


def test_assembly_refuses_a_supplement_naming_an_unpublished_tool(
    session: Session,
    tmp_path: Path,
) -> None:
    templates = _supplement_document()["task_templates"]
    templates[0]["required_tools"] = ["get_book_status", "shred_book"]

    with pytest.raises(PackAssemblyError) as refusal:
        _assemble(
            session,
            tmp_path,
            supplement=_supplement(
                tmp_path / "unknown-tool.yaml",
                task_templates=templates,
            ),
        )

    assert refusal.value.code == "supplement_tool_unknown"
    assert "shred_book" in refusal.value.detail


def test_assembly_refuses_a_supplement_naming_an_uncompiled_assertion(
    session: Session,
    tmp_path: Path,
) -> None:
    templates = _supplement_document()["task_templates"]
    templates[0]["success_assertions"] = ["assert_patron_was_charged"]

    with pytest.raises(PackAssemblyError) as refusal:
        _assemble(
            session,
            tmp_path,
            supplement=_supplement(
                tmp_path / "unknown-assertion.yaml",
                task_templates=templates,
            ),
        )

    assert refusal.value.code == "supplement_assertion_unknown"
    assert "assert_patron_was_charged" in refusal.value.detail


def test_assembly_refuses_to_overwrite_an_earlier_pack(
    session: Session,
    tmp_path: Path,
) -> None:
    output = tmp_path / "candidate"
    _assemble(session, tmp_path, output=output)
    earlier = (output / "pack" / "manifest.yaml").read_bytes()

    with pytest.raises(PackAssemblyError) as refusal:
        _assemble(session, tmp_path, output=output)

    assert refusal.value.code == "pack_output_exists"
    assert (output / "pack" / "manifest.yaml").read_bytes() == earlier


def test_a_refused_assembly_leaves_no_partial_pack_behind(
    session: Session,
    tmp_path: Path,
) -> None:
    output = tmp_path / "nested" / "candidate"
    templates = _supplement_document()["task_templates"]
    templates[0]["success_assertions"] = ["assert_patron_was_charged"]

    with pytest.raises(PackAssemblyError):
        _assemble(
            session,
            tmp_path,
            supplement=_supplement(
                tmp_path / "refused.yaml",
                task_templates=templates,
            ),
            output=output,
        )

    assert not output.exists()
    # A half-written pack under a staging name is as dangerous as one under the real name.
    assert list(tmp_path.rglob(".candidate.staging-*")) == []
