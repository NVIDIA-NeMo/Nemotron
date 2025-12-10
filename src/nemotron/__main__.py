#!/usr/bin/env python3
"""Nemotron CLI entry point.

Usage:
    nemotron nano3 data prep pretrain --help
    nemotron nano3 data prep sft --help
    nemotron nano3 data prep rl --help
    nemotron nano3 pretrain --help
    nemotron nano3 sft --help
    nemotron nano3 rl --help
"""

from __future__ import annotations

import sys


def main() -> None:
    """Main CLI entry point."""
    args = sys.argv[1:]

    if not args:
        print("Usage: nemotron <recipe> <command> [options]")
        print("\nRecipes:")
        print("  nano3    Nano3 training recipe")
        print("\nRun 'nemotron <recipe> --help' for more information.")
        sys.exit(1)

    recipe = args[0]

    if recipe == "nano3":
        # Pass remaining args directly to nano3 app
        sys.argv = [sys.argv[0]] + args[1:]

        from nemotron.recipes.nano3 import app

        app.run()
    elif recipe in ("--help", "-h"):
        print("Usage: nemotron <recipe> <command> [options]")
        print("\nRecipes:")
        print("  nano3    Nano3 training recipe")
        print("\nRun 'nemotron <recipe> --help' for more information.")
    else:
        print(f"Unknown recipe: {recipe}")
        print("Available recipes: nano3")
        sys.exit(1)


if __name__ == "__main__":
    main()
