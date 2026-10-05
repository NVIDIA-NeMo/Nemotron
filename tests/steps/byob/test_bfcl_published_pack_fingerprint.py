"""Refuse a pack edit that would stop every eval of an already published release.

An eval scores a benchmark only against the pack revision that certified its
gold traces, and the fingerprint covers every file in the pack directory. So an
edit anywhere in a published pack — a helper module, a fixture, or only the
README — makes each downstream eval fail preflight with
``eval_source_oracle_pack_drift``, hours later and on someone else's machine.

`banking-vn-gold-v1-1392` was published from `banking_vn` 0.1.0 and is still
scored, so that directory is frozen. This test is where that fact is enforced,
at the commit that breaks it rather than at the next eval.
"""

from __future__ import annotations

from pathlib import Path

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import OraclePackRef
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    ResolvedPackPaths,
    declared_pack_inputs,
    pack_file_hashes,
    pack_fingerprint,
    resolve_declared_pack_paths,
)

BANKING_PACK_ROOT = (
    Path(__file__).resolve().parents[3] / "src" / "nemotron" / "steps" / "byob" / "data" / "banking_vn_oracle_pack"
)

# What generation recorded in the release's run_manifest.json, under
# `pack.content_hash`. Changing this constant to match an edited pack does not
# make the edited pack scoreable: the hash lives in the published manifest too.
PUBLISHED_RELEASE = "banking-vn-gold-v1-1392"
PUBLISHED_PACK_FINGERPRINT = "sha256:c5bb5c39033c28dd4c6e5d7197e88db2801e85887458457088a16cccf7e740eb"

# The aggregate is one rolling digest and cannot say which file moved. That
# release predates per-file recording, so the map is pinned here instead, which
# is what lets this test name the file a commit changed.
PUBLISHED_PACK_FILES = {
    "tree/README.md": "sha256:a5acf71838942ac8d7bffe51f801a949b095d90dd53233ee8b043ff8581ac966",
    "tree/assertions.py": "sha256:fad583adf0f75c4621b9cb886e67e4816cfd1760f491b49f6fd006cf8f306a5f",
    "tree/backend.py": "sha256:47b424e6842efd7baec371b0bdce03388434609b78f4038a26d411d6623066d6",
    "tree/fixtures.json": "sha256:464f073d1f19fbec54af33ffbec844b116efec6cf3e8492b2ec81a7db492611e",
    "tree/manifest.yaml": "sha256:79e10ec1f61dd53832350c3a141c6bb16254435d443d29fb57965e8cd18b4db7",
    "tree/task_templates.yaml": "sha256:e3126b935f28397cd075f2f9ab3f9d5e8be176b7c0088f4e922030613604a421",
    "tree/tools.json": "sha256:b61f0a89a91c7eb58c42efa3804f8d8be3026a06b87261dc72b843da591352d8",
    "tree/validation_cases.yaml": "sha256:a5786a2500edaa9fe50f50886a9fef3ce471c7d2bce96ca822bc8862c29d9402",
}


def _paths() -> ResolvedPackPaths:
    return resolve_declared_pack_paths(
        OraclePackRef(manifest_path=BANKING_PACK_ROOT / "manifest.yaml"),
        (BANKING_PACK_ROOT,),
    )


def test_banking_vn_pack_still_matches_the_release_it_published() -> None:
    paths = _paths()
    observed = pack_file_hashes(paths)
    declared = declared_pack_inputs(paths)

    def describe(name: str) -> str:
        return name if name in declared else f"{name} (documentation or other untracked-by-manifest file)"

    changed = sorted(
        describe(name)
        for name, digest in observed.items()
        if name in PUBLISHED_PACK_FILES and PUBLISHED_PACK_FILES[name] != digest
    )
    added = sorted(describe(name) for name in observed if name not in PUBLISHED_PACK_FILES)
    removed = sorted(describe(name) for name in PUBLISHED_PACK_FILES if name not in observed)
    assert not (changed or added or removed), (
        f"the {PUBLISHED_RELEASE} pack changed, so every eval of that release will fail preflight.\n"
        f"  changed: {changed or 'none'}\n"
        f"  added  : {added or 'none'}\n"
        f"  removed: {removed or 'none'}\n"
        "Notes about a published pack belong outside its directory — see "
        "references/bfcl-banking-vn-pack-operations.md. If the oracle genuinely has to change, "
        "publish a new release and score against that instead of editing this one."
    )
    assert f"sha256:{pack_fingerprint(paths)}" == PUBLISHED_PACK_FINGERPRINT


def test_the_pinned_file_map_is_the_set_the_fingerprint_hashes() -> None:
    """Keep the pin honest: a file the fingerprint covers but the map omits is a blind spot."""
    assert set(pack_file_hashes(_paths())) == set(PUBLISHED_PACK_FILES)


def test_documentation_is_not_exempt_from_the_pack_fingerprint() -> None:
    """State the rule this test exists for, so a later reader does not assume docs are free.

    Nothing stops a backend from reading its own README, so the fingerprint
    covers it. That is why the release record and the runnable commands live in
    references/ and not in the pack.
    """
    paths = _paths()
    assert "tree/README.md" in pack_file_hashes(paths)
    assert "tree/README.md" not in declared_pack_inputs(paths)
