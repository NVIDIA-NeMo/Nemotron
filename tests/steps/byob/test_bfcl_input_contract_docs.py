from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.assertion_capabilities import (
    ASSERTION_CATEGORIES,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    TURN_POLICIES,
)
from nemotron.steps.byob.scripts.scaffold_oracle_pack import (
    OPERATOR_REFERENCE_SLUGS,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE = REPO_ROOT / "docs" / "build-benchmarks" / "function-calling" / "reference"
HOW_TO = REPO_ROOT / "docs" / "build-benchmarks" / "function-calling" / "how-to"
EXPLANATION = REPO_ROOT / "docs" / "build-benchmarks" / "function-calling" / "explanation"


def _read(name: str) -> str:
    return (REFERENCE / name).read_text(encoding="utf-8")


def test_task_template_reference_tracks_runtime_turn_policies() -> None:
    text = _read("task-templates.md")

    for policy in TURN_POLICIES:
        assert f"`{policy}`" in text


def test_task_template_yaml_examples_parse() -> None:
    blocks = re.findall(r"```yaml\n(.*?)\n```", _read("task-templates.md"), re.DOTALL)

    assert blocks
    for block in blocks:
        assert yaml.safe_load(block) is not None


def test_python_backend_reference_names_the_complete_interface() -> None:
    text = _read("python-backend.md")

    for signature in (
        "def list_tools() -> list[str]",
        "def reset(*, ctx, fixtures=None)",
        "def call_tool(name: str, arguments: dict, *, ctx)",
        "def get_state() -> dict",
    ):
        assert signature in text


def test_oracle_pack_input_reference_links_the_field_guides() -> None:
    text = _read("oracle-pack-inputs.md")

    assert "{doc}`python-backend`" in text
    assert "{doc}`task-templates`" in text
    for slug in OPERATOR_REFERENCE_SLUGS:
        if slug != "oracle-pack-inputs":
            assert f"{{doc}}`{slug}`" in text
    for pack_input in (
        "manifest.yaml",
        "tools.json",
        "backend.py",
        "endpoint_config.yaml",
        "fixtures.json",
        "task_templates.yaml",
        "assertions.py",
        "validation_cases.yaml",
        "held_out.yaml",
    ):
        assert pack_input in text

    for supported_command in (
        "scaffold_oracle_pack",
        "scaffold_source_package",
        "check_source_package",
        "validate_oracle_pack",
    ):
        assert supported_command in text


def test_model_assisted_backend_guidance_preserves_oracle_boundary() -> None:
    text = _read("python-backend.md")
    normalized = " ".join(text.split())

    assert "--draft-with-model" in text
    assert re.search(r"does not accept arbitrary .*Python", normalized)
    assert re.search(r"optional .*not the preferred source of oracle semantics", normalized)
    assert all(term in normalized for term in ("domain fidelity", "executable checks"))


def test_authoring_docs_do_not_require_distinct_reviewer_identities() -> None:
    flow = (EXPLANATION / "authoring-flows.md").read_text(encoding="utf-8")
    assisted = (HOW_TO / "assisted-authoring.md").read_text(encoding="utf-8")

    normalized = " ".join(f"{flow} {assisted}".split()).casefold()
    assert all(
        term in normalized
        for term in (
            "not a requirement for two different reviewers",
            "does not compare the identities",
            "does not require two people",
        )
    )


def test_artifact_references_follow_the_standard_operator_structure() -> None:
    for slug in OPERATOR_REFERENCE_SLUGS:
        if slug == "oracle-pack-inputs":
            continue
        page = f"{slug}.md"
        text = _read(page)
        assert re.search(r"^## Create\b", text, re.MULTILINE), page
        assert re.search(r"^## Validate\b", text, re.MULTILINE), page
        assert "## Common Failures" in text, page
        assert "Example" in text, page


def test_new_artifact_reference_examples_are_parseable() -> None:
    for page in (
        "manifest.md",
        "validation-cases.md",
        "endpoint-config.md",
        "held-out-policy.md",
    ):
        blocks = re.findall(r"```yaml\n(.*?)\n```", _read(page), re.DOTALL)
        assert blocks, page
        for block in blocks:
            assert yaml.safe_load(block) is not None, page

    json_blocks = re.findall(r"```json\n(.*?)\n```", _read("tools-and-fixtures.md"), re.DOTALL)
    assert json_blocks
    for block in json_blocks:
        assert json.loads(block) is not None


def test_assertion_reference_tracks_runtime_categories() -> None:
    text = _read("assertions.md")

    for category in ASSERTION_CATEGORIES:
        assert f"`{category}`" in text


def test_reference_navigation_lists_every_pack_artifact_page() -> None:
    text = _read("index.md")

    for doc_name in OPERATOR_REFERENCE_SLUGS:
        assert f"<{doc_name}>" in text


def test_endpoint_route_is_consistent_with_normative_contract() -> None:
    operator_text = _read("endpoint-config.md")
    normative_text = (
        REPO_ROOT / "src" / "nemotron" / "steps" / "byob" / "references" / "bfcl-oracle-pack.md"
    ).read_text(encoding="utf-8")

    for text in (operator_text, normative_text):
        assert "GET /v1/conformance" in text
        assert "bfcl-endpoint-conformance-v1" in text
