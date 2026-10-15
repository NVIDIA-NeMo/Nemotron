# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The notebook runs top to bottom in replay mode, and nothing in the folder leaks secrets or local paths."""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "visual_investigation_agent.ipynb"
README = ROOT / "README.md"
ENV_EXAMPLE = ROOT / ".env.example"
SKIP = {".venv", ".world", "runs", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
GA_BASE_URL = "https://integrate.api.nvidia.com/v1"
GA_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
SELF_HOST_NOTE = "download the checkpoint from Hugging Face and serve it from your own OpenAI-compatible vLLM endpoint"


def _files():
    return [p for p in ROOT.rglob("*") if p.is_file() and not SKIP & set(p.relative_to(ROOT).parts)]


def test_replay_data_folder_is_not_media_ambiguous():
    assert (ROOT / "replay" / "model-switch-visual" / "manifest.json").is_file()
    assert not (ROOT / "recorded").exists()


@pytest.mark.slow
def test_notebook_runs_in_replay_mode(tmp_path):
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(NOTEBOOK, as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert errors == []


def test_no_secrets_or_local_paths():
    home = [re.escape("/" + d + "/") + "[a-z]" for d in ("Users", "home")]  # written in pieces so this file passes its own scan
    patterns = [re.compile(p) for p in home + [r"\bnvapi-[A-Za-z0-9_-]{16,}", r"\bsk-[A-Za-z0-9]{20,}",
                                                r"Bearer [A-Za-z0-9._~-]{16,}", r"127\.0\.0\.1:\d{4,5}"]]
    for path in _files():
        if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            continue
        text = path.read_text(errors="ignore")
        for pattern in patterns:
            assert not pattern.search(text), f"{path.relative_to(ROOT)} matches {pattern.pattern}"


def test_license_headers():
    model_facing = {"playbook", "policy"}  # any edit, even a header, is a new agent version; the folder's license covers them
    for path in _files():
        if model_facing & set(path.relative_to(ROOT).parts):
            continue
        if path.suffix in (".py", ".md", ".txt", ".ini", ".example") or path.name in (".gitignore",):
            assert "SPDX-License-Identifier: Apache-2.0" in path.read_text()[:400], path.relative_to(ROOT)
    first_code = next(c for c in json.loads(NOTEBOOK.read_text())["cells"] if c["cell_type"] == "code")
    assert "SPDX-License-Identifier" in "".join(first_code["source"])


def test_main_narrative_uses_reader_facing_language():
    notebook = json.loads(NOTEBOOK.read_text())
    markdown = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
    normalized = " ".join(markdown.split())
    for phrase in ("private reasoning", "The agent never clicks", "None of them names this month's answer",
                   "independent check on the agent's SQL"):
        assert phrase not in markdown
    assert "—" not in markdown
    assert "build.nvidia.com" in markdown
    assert "projected" in markdown.lower()
    assert "synthetic yet realistic" in markdown.lower()
    assert "recording of a live super vl run" in markdown.lower()
    assert "The quotation above is Super VL's first note" in markdown
    assert "> “The dashboard shows AI spend rose 30%" in markdown
    assert "Our explanation:" in markdown and "formed a hypothesis" in markdown
    assert "change-log query then found the Sep 8–10 Vesk 4 → Osric rollout" in markdown
    assert "Our takeaway from the trace:" in markdown
    assert "As its instructions require, Super VL takes no exact numbers from chart pixels" in markdown
    assert "over 100 times more reasoning tokens on a technical ticket" in markdown
    assert "turn the diagnosis into a proposal" in markdown
    assert ("The next card follows the decision in order: what it has, what it does next, what it says, and what happens "
            "next") in normalized
    assert "The exact route table, projected savings, math, evidence, and report checks are folded below" in normalized
    assert "Our check beneath those steps" not in normalized
    assert "the quality rule, Super VL's recommendation, what we verified, and who decides" not in normalized
    assert "**Our takeaway from the trace:** Super VL turned a screenshot, the help-desk data" in normalized
    assert "the cheapest model that meets the team's quality bar" in normalized
    assert "the projected saving holds: about $103,000 a month" in normalized
    assert "two small slips (see *Check the report against the data*)" in normalized
    assert "calculated in code before writing the report" in normalized
    assert "Super VL traced the increase to the three rollout entries" not in markdown
    assert "In the recording, Super VL made two good calls here" not in markdown
    assert "About a third of the new model's tokens are output" not in markdown
    assert "Notice what Super VL does with those numbers" not in markdown
    assert "selected recorded demonstration" not in markdown.lower()
    assert "same agent, different causes" not in markdown.lower()
    assert "trust and limitations" not in markdown.lower()
    assert "related nemotron resources" in markdown.lower()
    assert "nvidia build" in markdown.lower()
    assert "the included synthetic run" not in NOTEBOOK.read_text().lower()


def test_live_instructions_use_only_the_public_ga_path():
    notebook = json.loads(NOTEBOOK.read_text())
    markdown = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
    readme, env = README.read_text(), ENV_EXAMPLE.read_text()
    for text in (readme, markdown, env):
        assert GA_BASE_URL in text and GA_MODEL in text
        assert "nvidia/nvidia/" not in text
    for text in (readme, markdown):
        assert text.count(SELF_HOST_NOTE) == 1
        assert "MODEL_API_KEY" not in text
    assert f"MODEL_BASE_URL={GA_BASE_URL}" in env
    assert f"MODEL_NAME={GA_MODEL}" in env
    assert "NVIDIA_API_KEY=" in env and "MODEL_API_KEY" not in env
    assert "an NVIDIA Build API key" in markdown
    setup = next(cell for cell in notebook["cells"] if "".join(cell["source"]).startswith("## Run this notebook"))
    setup = "".join(setup["source"])
    code = setup.split("<b>Get the code</b> and open this notebook from the cloned folder, not on its own</summary>\n\n", 1)
    assert len(code) == 2 and "git clone https://github.com/NVIDIA-NeMo/Nemotron.git" in code[1].split("</details>")[0]
    folded = setup.split("<b>To run live</b> with your own API key</summary>\n\n", 1)
    assert "**To replay**, run all cells" in folded[0] and len(folded) == 2
    for text in (GA_BASE_URL, GA_MODEL, SELF_HOST_NOTE, "No two runs are identical"):
        assert text in folded[1].split("</details>")[0]
    forbidden = ("vllm serve", "localhost:8000", "--trust-remote-code", "--tool-call-parser", "--reasoning-parser",
                 "endpoint that serves Super VL", "hosted/self-hosted")
    for text in (readme, markdown, env):
        assert not any(term in text for term in forbidden)


def test_notebook_images_are_embedded():
    """Colab opens the notebook without this folder, so markdown images must be data URIs, not paths."""
    notebook = json.loads(NOTEBOOK.read_text())
    markdown = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
    images = re.findall(r"!\[[^\]]*\]\(([^)]{0,40})", markdown)
    assert len(images) == 2 and all(src.startswith("data:image/png;base64,") for src in images), images
