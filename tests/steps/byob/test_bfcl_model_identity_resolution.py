"""Resolving a candidate's weight identity instead of asking someone to type it.

The rules under test are the ones that decide whether a later score means
anything: a moving reference is resolved to the commit it names *now*, weights on
disk are named by their bytes, and a route nothing pins is reported as unpinned
rather than turned into a value that looks like a pin.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.identity import (
    ModelIdentityClaim,
    compare_model_identity,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.model_identity_resolution import (
    ModelIdentityResolutionError,
    digest_weight_directory,
    identity_document,
    provider_managed_identity,
    resolve_local_identity,
    resolve_registry_identity,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.schemas import (
    WEIGHT_MANIFEST_DIGEST_SCHEME,
)

RESOLVED_COMMIT = "3f9a1c5e7b2d4068a1c3e5b7d9f0a2c4e6b8d0f2"
SCRIPT = "nemotron.steps.byob.scripts.resolve_bfcl_model_identity"


def _weights(root: Path, *, payload: bytes = b"weight-bytes") -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_bytes(b'{"architectures": ["Test"]}')
    (root / "model.safetensors").write_bytes(payload)
    return root


def test_a_moving_reference_is_resolved_to_the_commit_it_currently_names() -> None:
    """``main`` is what an operator has; the commit is what the config must keep."""
    seen: list[tuple[str, str, str | None]] = []

    def resolver(source: str, model: str, revision: str | None) -> str:
        seen.append((source, model, revision))
        return RESOLVED_COMMIT

    identity = resolve_registry_identity(
        source="huggingface",
        model="org/model",
        revision="main",
        resolver=resolver,
    )

    assert seen == [("huggingface", "org/model", "main")]
    assert identity.revision == RESOLVED_COMMIT
    assert identity.assurance == "weights_pinned"


def test_a_registry_that_answers_with_a_moving_reference_is_refused() -> None:
    """A resolver is not trusted to have resolved anything: the contract still rules."""
    with pytest.raises(ModelIdentityResolutionError, match="refuses"):
        resolve_registry_identity(
            source="huggingface",
            model="org/model",
            revision="main",
            resolver=lambda *_args: "main",
        )


def test_local_weights_are_named_by_their_bytes(tmp_path: Path) -> None:
    identity = resolve_local_identity(
        source="local",
        model="org/model",
        weights_dir=_weights(tmp_path / "weights"),
    )

    assert identity.weights_digest is not None
    assert identity.weights_digest.startswith(f"{WEIGHT_MANIFEST_DIGEST_SCHEME}:")
    assert identity.assurance == "weights_pinned"


def test_a_manifest_digest_names_its_scheme_so_it_is_never_read_as_raw_weight_bytes(
    tmp_path: Path,
) -> None:
    """Two schemes disagree for identical weights, and the gate must not call that a difference."""
    manifest_digest = digest_weight_directory(_weights(tmp_path / "weights"))
    body = manifest_digest.split(":", 1)[1]
    raw_looking = ModelIdentityClaim(weight_model="org/model", weights_digest=f"sha256:{body[:-1]}0")
    scoped = ModelIdentityClaim(weight_model="org/model", weights_digest=manifest_digest)

    assert not manifest_digest.startswith("sha256:")
    assert compare_model_identity(scoped, raw_looking) == "unknown"


def test_two_equal_manifest_digests_settle_the_comparison_between_themselves(
    tmp_path: Path,
) -> None:
    """Nothing outside the weights can make two directories agree, so equality still decides."""
    same = digest_weight_directory(_weights(tmp_path / "a"))
    claim = ModelIdentityClaim(weight_model="org/model", weights_digest=same)

    assert compare_model_identity(claim, ModelIdentityClaim(weight_model="org/model", weights_digest=same)) == "match"


def test_a_note_left_beside_untouched_weights_does_not_clear_the_candidate(
    tmp_path: Path,
) -> None:
    """The manifest covers the whole directory, so its inequality is not about the weights.

    Reading it as a difference is what would clear a candidate that answered from
    the very weights that wrote the rows, on the strength of a README.
    """
    served = _weights(tmp_path / "served")
    documented = _weights(tmp_path / "documented")
    (documented / "README.md").write_bytes(b"# served internally\n")

    bare = digest_weight_directory(served)
    annotated = digest_weight_directory(documented)

    assert bare != annotated
    assert (
        compare_model_identity(
            ModelIdentityClaim(weight_model="org/model", weights_digest=annotated),
            ModelIdentityClaim(weight_model="org/model", weights_digest=bare),
        )
        == "unknown"
    )


def test_genuinely_different_weights_are_still_not_cleared_by_a_manifest_digest(
    tmp_path: Path,
) -> None:
    """A manifest cannot tell the two cases apart, so it declines both rather than guessing."""
    mine = digest_weight_directory(_weights(tmp_path / "a"))
    theirs = digest_weight_directory(_weights(tmp_path / "b", payload=b"other-weight-bytes"))

    assert (
        compare_model_identity(
            ModelIdentityClaim(weight_model="org/model", weights_digest=mine),
            ModelIdentityClaim(weight_model="org/model", weights_digest=theirs),
        )
        == "unknown"
    )


def test_raw_weight_digests_still_tell_two_sets_of_weights_apart() -> None:
    """The scope-widening rule is about the manifest scheme, not about digests generally."""
    mine = ModelIdentityClaim(weight_model="org/model", weights_digest="sha256:" + "a" * 64)
    theirs = ModelIdentityClaim(weight_model="org/model", weights_digest="sha256:" + "b" * 64)

    assert compare_model_identity(mine, theirs) == "different"


def test_a_source_this_build_has_no_registry_client_for_is_refused() -> None:
    """Resolving one registry's name against another's API would record a false pin."""
    with pytest.raises(ModelIdentityResolutionError, match="no registry client for source 'modelscope'"):
        resolve_registry_identity(source="modelscope", model="org/model", revision="main")


def test_a_changed_weight_file_is_a_different_identity(tmp_path: Path) -> None:
    first = digest_weight_directory(_weights(tmp_path / "a"))
    second = digest_weight_directory(_weights(tmp_path / "b", payload=b"other-weight-bytes"))

    assert first != second
    assert digest_weight_directory(_weights(tmp_path / "c")) == first


def test_a_renamed_weight_file_is_a_different_identity(tmp_path: Path) -> None:
    """The digest covers a manifest, so which file held the bytes is part of it."""
    original = _weights(tmp_path / "original")
    renamed = _weights(tmp_path / "renamed")
    (renamed / "model.safetensors").rename(renamed / "model-00001.safetensors")

    assert digest_weight_directory(original) != digest_weight_directory(renamed)


def test_a_symlinked_weight_file_is_refused_rather_than_followed(tmp_path: Path) -> None:
    root = _weights(tmp_path / "weights")
    outside = tmp_path / "elsewhere.safetensors"
    outside.write_bytes(b"weights-somewhere-else")
    (root / "shard-2.safetensors").symlink_to(outside)

    with pytest.raises(ModelIdentityResolutionError, match="symlink"):
        digest_weight_directory(root)


def test_an_empty_directory_names_no_weights(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(ModelIdentityResolutionError, match="no files to digest"):
        digest_weight_directory(empty)


def test_a_route_nobody_pins_resolves_to_provider_managed_not_to_a_digest() -> None:
    identity = provider_managed_identity(source="openai", model="frontier-x")

    document = identity_document(identity)

    assert identity.weights_digest is None
    assert identity.revision is None
    assert document["assurance"] == "provider_managed"
    assert document["canonical_id"] == "openai:frontier-x@provider_managed"
    assert document["identity_publication_gate"] == "blocked"


def test_the_cli_writes_a_paste_ready_fragment_and_reports_what_it_is_worth(
    tmp_path: Path,
) -> None:
    fragment = tmp_path / "identity.yaml"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            SCRIPT,
            "local",
            "--model",
            "org/model",
            "--weights-dir",
            str(_weights(tmp_path / "weights")),
            "--output",
            str(fragment),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    reported = json.loads(completed.stdout)
    assert reported["status"] == "resolved"
    assert reported["assurance"] == "weights_pinned"
    assert reported["identity_publication_gate"] == "satisfied"
    written = yaml.safe_load(fragment.read_text(encoding="utf-8"))
    assert written == {"model_identity": reported["model_identity"]}


def test_the_cli_refuses_to_overwrite_a_fragment_and_fails_with_a_reason(
    tmp_path: Path,
) -> None:
    fragment = tmp_path / "identity.yaml"
    fragment.write_text("model_identity: {}\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            SCRIPT,
            "provider-managed",
            "--source",
            "openai",
            "--model",
            "frontier-x",
            "--output",
            str(fragment),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert json.loads(completed.stderr)["status"] == "fail"
    assert fragment.read_text(encoding="utf-8") == "model_identity: {}\n"
