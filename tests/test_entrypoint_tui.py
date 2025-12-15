from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch


def test_console_entrypoint_launches_recipe_tui_for_bare_pretrain() -> None:
    from nemotron.__main__ import main

    with patch("nemotron.kit.cli.recipe.run_recipe_tui", return_value=None) as m:
        with patch.object(sys, "argv", ["nemotron", "nano3", "pretrain"]):
            try:
                main()
            except SystemExit as e:
                assert e.code == 0

    assert m.call_count == 1


def test_console_entrypoint_falls_back_to_app_run_when_not_intercepted() -> None:
    from nemotron.__main__ import main

    dummy_app = MagicMock()
    nano3_mod = ModuleType("nemotron.recipes.nano3")
    nano3_mod.app = dummy_app

    with patch.dict(sys.modules, {"nemotron.recipes.nano3": nano3_mod}):
        with patch.object(sys, "argv", ["nemotron", "nano3", "data", "prep", "pretrain"]):
            main()

        dummy_app.run.assert_called_once()
