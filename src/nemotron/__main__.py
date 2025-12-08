#!/usr/bin/env python3
"""Nemotron CLI entry point.

Usage:
    nemotron nano3 data prep --help
    nemotron nano3 pretrain --help
    nemotron nano3 sft --help
    nemotron nano3 rl --help
    nemotron nano3 eval --help
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
        # Multi-word subcommands: join "data prep" -> "data prep" as single arg
        remaining = args[1:]
        multi_word_commands = ["data prep", "data curate"]

        # Check if first two args form a multi-word command
        if len(remaining) >= 2:
            candidate = f"{remaining[0]} {remaining[1]}"
            if candidate in multi_word_commands:
                remaining = [candidate] + remaining[2:]

        sys.argv = [sys.argv[0]] + remaining

        import tyro

        from nemotron.recipes.nano3 import cli as nano3_cli

        result = tyro.cli(nano3_cli)
        if isinstance(result, int):
            sys.exit(result)
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
