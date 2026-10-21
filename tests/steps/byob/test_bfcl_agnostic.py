"""Guards that keep the BFCL runtime independent of checked-in example domains."""

from __future__ import annotations

from pathlib import Path

BYOB_ROOT = (
    Path(__file__).resolve().parents[3] / "src" / "nemotron" / "steps" / "byob"
)
BFCL_CORE = BYOB_ROOT / "runtime" / "benchmark_families" / "bfcl"


def test_bfcl_core_contains_no_example_pack_vocabulary() -> None:
    forbidden = {
        "banking_vn",
        "tiny_library",
        "beneficiary_id",
        "transfer_id",
        "checkout_book",
        "book_id",
        "vnd",
    }
    matches: dict[str, list[str]] = {}
    for path in BFCL_CORE.rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        found = sorted(token for token in forbidden if token in text)
        if found:
            matches[str(path.relative_to(BFCL_CORE))] = found
    assert matches == {}


def test_tiny_pack_is_selected_only_by_config() -> None:
    config = (BYOB_ROOT / "bfcl" / "config" / "tiny.yaml").read_text(encoding="utf-8")
    assert "data/tiny_oracle_pack/manifest.yaml" in config
    core_text = "\n".join(
        path.read_text(encoding="utf-8") for path in BFCL_CORE.rglob("*.py")
    )
    assert "tiny_oracle_pack" not in core_text
