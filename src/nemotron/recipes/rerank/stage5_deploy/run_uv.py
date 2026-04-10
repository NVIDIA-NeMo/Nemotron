#!/usr/bin/env python3
"""UV dependency wrapper — delegates to shared nemotron.kit.run_uv."""
import sys
from pathlib import Path

# Ensure nemotron package is importable (for container execution)
_src = str(Path(__file__).resolve().parents[4])  # -> .../src
if _src not in sys.path:
    sys.path.insert(0, _src)

from nemotron.kit.run_uv import main

main(stage_dir=Path(__file__).parent)
