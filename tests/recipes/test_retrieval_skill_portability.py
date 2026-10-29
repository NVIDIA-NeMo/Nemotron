# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that a copied skill's documentation dependencies stay self-contained."""

import re
import shutil
from pathlib import Path
from urllib.parse import unquote, urlsplit


def test_copied_skill_references_are_complete_and_external_sources_are_pinned(tmp_path):
    source = Path(__file__).resolve().parents[2] / "skills/nemotron-retrieval-recipes"
    skill = tmp_path / "installed-skill"
    shutil.copytree(source, skill)
    pending = [skill / "SKILL.md", skill / "skill-card.md"]
    visited = set()
    while pending:
        page = pending.pop()
        if page in visited:
            continue
        visited.add(page)
        for destination in re.findall(r"\[[^\]]+\]\(([^)]+)\)", page.read_text()):
            url = urlsplit(destination)
            if url.scheme:
                # The disclosure portal is a live reporting service, not source documentation.
                if destination == "https://app.intigriti.com/programs/nvidia/nvidiavdp/detail":
                    continue
                assert url.scheme == "https" and url.netloc == "github.com", destination
                assert re.match(r"/[^/]+/[^/]+/blob/[0-9a-f]{40}/", url.path), destination
                continue
            target = (page.parent / unquote(url.path)).resolve() if url.path else page
            assert target.is_relative_to(skill), f"{page.name} escapes the copied skill: {destination}"
            assert target.is_file(), f"{page.name} has a missing bundled reference: {destination}"
            if target.suffix == ".md":
                pending.append(target)
    assert set((skill / "references").glob("*.md")).issubset(visited), "Unreachable bundled guidance"
