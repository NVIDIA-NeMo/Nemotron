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

"""Adapter-neutral review, freeze, and publication release kernel."""

from nemotron.steps.byob.runtime.authoring_release.assembly import (
    AssembledReview,
    ReviewContext,
    VerifiedReleaseAdapter,
    assemble_review,
    release_adapter_for_packet,
)
from nemotron.steps.byob.runtime.authoring_release.contracts import (
    AdapterReviewContribution,
    FreezeHookContext,
    PublicationAdapter,
    ReleaseAdapter,
)
from nemotron.steps.byob.runtime.authoring_release.freeze import (
    AuthoringFreezeError,
    FreezeInputsV2,
    FrozenReleaseV2,
    freeze_canonical_pack,
    load_frozen_release,
)
from nemotron.steps.byob.runtime.authoring_release.handoff import (
    AuthoringHandoffError,
    PublicationHandoffV2,
    handoff_frozen_release,
)
from nemotron.steps.byob.runtime.authoring_release.publication import (
    BfclPublicationAdapter,
    publication_adapter_for_release,
)
from nemotron.steps.byob.runtime.authoring_release.review import (
    REQUIRED_CHECKLIST_V2,
    AuthoringReviewError,
    ReviewApprovalV2,
    ReviewApprovalV3,
    ReviewPacketV2,
    build_review_approval,
    build_review_packet,
    load_review_approval,
    load_review_packet,
    write_review_approval,
    write_review_packet,
)
from nemotron.steps.byob.runtime.authoring_release.revocation import (
    ReleaseRevocationError,
    ReleaseRevocationRecord,
    ReleaseRevocationRegistry,
    RevocationAuthority,
    RevocationRegistryVerifier,
    RevocationTarget,
    RevocationVerdict,
    build_revocation_record,
    build_revocation_registry,
    load_revocation_registry,
    verify_release_revocation,
    write_revocation_registry,
)
from nemotron.steps.byob.runtime.release_seal import (
    ReleaseSealAuthority,
    load_release_seal_authority,
    load_trusted_release_seal_key,
)

__all__ = [
    "REQUIRED_CHECKLIST_V2",
    "AdapterReviewContribution",
    "AssembledReview",
    "AuthoringFreezeError",
    "AuthoringHandoffError",
    "AuthoringReviewError",
    "BfclPublicationAdapter",
    "FreezeHookContext",
    "FreezeInputsV2",
    "FrozenReleaseV2",
    "PublicationAdapter",
    "PublicationHandoffV2",
    "ReleaseAdapter",
    "ReleaseRevocationError",
    "ReleaseRevocationRecord",
    "ReleaseRevocationRegistry",
    "ReleaseSealAuthority",
    "RevocationAuthority",
    "RevocationRegistryVerifier",
    "RevocationTarget",
    "RevocationVerdict",
    "ReviewContext",
    "ReviewApprovalV2",
    "ReviewApprovalV3",
    "ReviewPacketV2",
    "VerifiedReleaseAdapter",
    "assemble_review",
    "build_review_approval",
    "build_review_packet",
    "build_revocation_record",
    "build_revocation_registry",
    "freeze_canonical_pack",
    "handoff_frozen_release",
    "load_frozen_release",
    "load_review_approval",
    "load_review_packet",
    "load_release_seal_authority",
    "load_revocation_registry",
    "load_trusted_release_seal_key",
    "publication_adapter_for_release",
    "release_adapter_for_packet",
    "write_review_approval",
    "write_review_packet",
    "verify_release_revocation",
    "write_revocation_registry",
]
