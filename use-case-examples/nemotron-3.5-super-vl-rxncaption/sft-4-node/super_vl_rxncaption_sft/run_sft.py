#!/usr/bin/env python3
"""Chemical reaction parsing with NVIDIA Nemotron 3.5 Super VL on four nodes."""

import sys
from pathlib import Path

# Local tutorial support code; this is not a PyPI dependency.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "super_vl_rxncaption"))
from pipeline import main

if __name__ == "__main__":
    main("sft")
