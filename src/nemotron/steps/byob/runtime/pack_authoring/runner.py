# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One drafting run: verify, approve, draft, compile what can be compiled, record.

The gates come first and the writes come last, so a run that is going to be refused leaves
nothing behind that looks like output. Drafts are written as YAML next to the pack rather
than into it, because they are still proposals: `task_templates.yaml` inside a pack directory
would be loaded by the pipeline as though a human had written it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from nemotron.steps.byob.runtime.authoring_workflow.quota import (
    DEFAULT_AUTHORING_QUOTA,
    RunQuota,
    RunQuotaLimits,
    RunQuotaSnapshot,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.model_io_cache import (
    ImmutableModelIOCache,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_text,
    write_text_atomic,
)
from nemotron.steps.byob.runtime.pack_authoring.authorization import (
    ExposureAuthorization,
    build_exposure_subject,
    load_exposure_authorization,
    verify_exposure_authorization,
)
from nemotron.steps.byob.runtime.pack_authoring.bundle import (
    Approval,
    BundleError,
    EvidenceView,
    load_approval,
    load_evidence_bundle,
)
from nemotron.steps.byob.runtime.pack_authoring.compile_assertions import (
    CompilationError,
    compile_assertions,
)
from nemotron.steps.byob.runtime.pack_authoring.drafts import (
    DraftBundle,
    DraftingContext,
    draft_all,
)
from nemotron.steps.byob.runtime.pack_authoring.model_client import (
    AuthoringModel,
    StructuredCaller,
)
from nemotron.steps.byob.runtime.pack_authoring.provenance import (
    DraftProvenance,
    build_draft_provenance,
    write_draft_provenance,
)
from nemotron.steps.byob.runtime.pack_authoring.questions import (
    verify_answered_revision,
)

DRAFT_DIRECTORY_NAME = "drafts"
CACHE_FILE_NAME = "authoring_io_cache.jsonl"
PROVENANCE_FILE_NAME = "draft_provenance.json"
ASSERTIONS_FILE_NAME = "assertions.py"


@dataclass(frozen=True)
class DraftingResult:
    evidence: EvidenceView
    approval: Approval
    drafts: DraftBundle
    provenance: DraftProvenance
    output_root: Path
    assertions_path: Path | None
    compilation_refusals: tuple[str, ...]
    quota_usage: RunQuotaSnapshot

    @property
    def draft_root(self) -> Path:
        return self.output_root / DRAFT_DIRECTORY_NAME


def _dump_yaml(document: object) -> str:
    return str(
        yaml.safe_dump(
            document,
            sort_keys=True,
            default_flow_style=False,
            allow_unicode=True,
            width=100,
        )
    )


def run_drafting(
    bundle_path: Path,
    approval_path: Path,
    output_root: Path,
    model: AuthoringModel,
    *,
    caller: StructuredCaller | None = None,
    certification_report_path: Path | None = None,
    trusted_certification_keys: Mapping[str, Ed25519PublicKey] | None = None,
    domain_brief_source_path: Path | None = None,
    domain_brief_report_path: Path | None = None,
    held_out_redaction_report_path: Path | None = None,
    held_out_policy_path: Path | None = None,
    held_out_content_path: Path | None = None,
    source_bundle_path: Path | None = None,
    migration_record_path: Path | None = None,
    source_observations_path: Path | None = None,
    parent_evidence_path: Path | None = None,
    open_questions_path: Path | None = None,
    answer_set_path: Path | None = None,
    exposure_authorization_path: Path | None = None,
    organizational_policy_digest: str | None = None,
    allow_legacy_v1_model_exposure: bool = False,
    quota_limits: RunQuotaLimits = DEFAULT_AUTHORING_QUOTA,
    resolved_authoring_config_digest: str | None = None,
) -> DraftingResult:
    """Draft the pack artifacts an approved evidence bundle can support."""
    evidence = load_evidence_bundle(
        bundle_path,
        certification_report_path=certification_report_path,
        trusted_certification_keys=trusted_certification_keys,
        domain_brief_source_path=domain_brief_source_path,
        domain_brief_report_path=domain_brief_report_path,
        held_out_redaction_report_path=held_out_redaction_report_path,
        held_out_policy_path=held_out_policy_path,
        held_out_content_path=held_out_content_path,
        source_bundle_path=source_bundle_path,
        migration_record_path=migration_record_path,
        source_observations_path=source_observations_path,
    )
    exposure_authorization: ExposureAuthorization | None = None
    if evidence.source_evidence is not None:
        verify_answered_revision(
            evidence.source_evidence,
            parent_evidence_path=parent_evidence_path,
            open_questions_path=open_questions_path,
            answer_set_path=answer_set_path,
            expected_root_digest=evidence.source_digest,
            expected_normalized_origin_digest=(
                evidence.migration.normalized_digest if evidence.migration is not None else None
            ),
        )
        if exposure_authorization_path is None:
            raise BundleError("v2 drafting requires an explicit model exposure authorization")
        if evidence.domain_brief_report is None or evidence.held_out_redaction_report is None:
            raise BundleError("v2 evidence reports are incomplete")
        exposure_authorization = load_exposure_authorization(exposure_authorization_path)
        verify_exposure_authorization(
            exposure_authorization,
            expected_subject=build_exposure_subject(
                evidence.source_evidence,
                domain_brief_report=evidence.domain_brief_report,
                held_out_redaction_report=evidence.held_out_redaction_report,
                resolved_authoring_config_digest=resolved_authoring_config_digest,
            ),
            expected_organizational_policy_digest=organizational_policy_digest,
        )
    else:
        if not allow_legacy_v1_model_exposure:
            raise BundleError("legacy v1 evidence cannot reach a model implicitly; normalize to v2")
        if exposure_authorization_path is not None or organizational_policy_digest is not None:
            raise BundleError("model exposure authorization is valid only for v2 evidence")
    approval = load_approval(approval_path, evidence)

    root = output_root.resolve()
    quota = RunQuota(quota_limits)
    context = DraftingContext(
        evidence=evidence,
        model=model,
        cache=ImmutableModelIOCache(root / CACHE_FILE_NAME),
        run_dir=root / "model_runs",
        caller=caller,
        quota=quota,
    )
    drafts = draft_all(context)

    source: str | None = None
    refusals: tuple[str, ...] = ()
    try:
        source = compile_assertions(drafts.assertions)
        compile(source, ASSERTIONS_FILE_NAME, "exec")
    except SyntaxError as exc:
        refusals = (f"generated assertions are not valid Python: line {exc.lineno}: {exc.msg}",)
        source = None
    except CompilationError as exc:
        refusals = exc.reasons

    draft_root = root / DRAFT_DIRECTORY_NAME
    for name, document in drafts.as_documents().items():
        write_text_atomic(_dump_yaml(document), draft_root / f"{name}.yaml")

    assertions_path: Path | None = None
    if source is not None:
        assertions_path = write_text_atomic(source, draft_root / ASSERTIONS_FILE_NAME)

    provenance = build_draft_provenance(
        evidence,
        approval,
        model,
        drafts,
        assertions_compiled=source is not None,
        compilation_refusals=refusals,
        exposure_authorization=exposure_authorization,
        quota_snapshot=quota.snapshot(),
        resolved_authoring_config_digest=resolved_authoring_config_digest,
    )
    write_draft_provenance(provenance, root / PROVENANCE_FILE_NAME)
    if assertions_path is not None and source is not None:
        # Prove the file on disk is the source that was digested into provenance.
        written = assertions_path.read_text(encoding="utf-8")
        if sha256_text(written) != sha256_text(source):
            raise CompilationError(["compiled assertions.py on disk does not match the compiled source"])
    return DraftingResult(
        evidence=evidence,
        approval=approval,
        drafts=drafts,
        provenance=provenance,
        output_root=root,
        assertions_path=assertions_path,
        compilation_refusals=refusals,
        quota_usage=quota.snapshot(),
    )
