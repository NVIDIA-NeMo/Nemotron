#!/bin/bash
# Wrapper script to execute train.py with UV and proper flags
# This script is packaged and executed remotely via nemo-run

# Use --system to reuse container's existing packages (especially torch)
# Use --with to add Nemotron library to the environment
# UV auto-detects PEP 723 metadata from main.py

exec uv run --system --with libraries/Nemotron main.py "$@"
