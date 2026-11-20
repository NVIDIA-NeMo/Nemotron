"""Synthetic pages must never silently switch their font layout engine."""

import pytest
from PIL import ImageFont, features

from nemotron.recipes.parse.dental import data, render


def test_unavailable_raqm_rejects_pages_before_writing_images(tmp_path, monkeypatch):
    monkeypatch.setattr(render.features, "check_feature", lambda feature: False)
    output = tmp_path / "candidates"
    with pytest.raises(RuntimeError, match="libfribidi0"):
        data.save_pages([{}], output, "rl_candidates")
    assert not output.exists()
    with pytest.raises(RuntimeError, match="BASIC layout changes"):
        render.render_page({})


@pytest.mark.parametrize("serif", [False, True])
def test_bundled_fonts_use_raqm(serif):
    if not features.check_feature("raqm"):
        pytest.skip("Requires the documented native Raqm rendering dependencies")
    assert render.font(18, serif=serif).layout_engine == ImageFont.Layout.RAQM
