#!/usr/bin/env python3
"""Nemotron CLI entry point.

Usage:
    nemotron nano3 pretrain -c test                       # local execution
    nemotron nano3 pretrain --config test --run dlw       # nemo-run attached
    nemotron nano3 pretrain -c test -r dlw train.train_iters=5000
    nemotron nano3 pretrain -c test --dry-run             # preview config

Legacy usage (still supported):
    nemotron-legacy nano3 pretrain --help
"""

from __future__ import annotations


def main() -> None:
    """Main CLI entry point."""
    from nemotron.cli.bin.nemotron import main as typer_main

    typer_main()


def main_legacy() -> None:
    """Legacy CLI entry point using tyro-based App."""
    import sys

    args = sys.argv[1:]

    if not args:
        print("Usage: nemotron-legacy <recipe> <command> [options]")
        print("\nRecipes:")
        print("  nano3    Nano3 training recipe")
        print("\nRun 'nemotron-legacy <recipe> --help' for more information.")
        sys.exit(1)

    recipe = args[0]

    if recipe == "nano3":
        # Pass remaining args directly to nano3 app
        sys.argv = [sys.argv[0]] + args[1:]

        from nemotron.recipes.nano3 import app

        app.run()
    elif recipe in ("--help", "-h"):
        print("Usage: nemotron-legacy <recipe> <command> [options]")
        print("\nRecipes:")
        print("  nano3    Nano3 training recipe")
        print("\nRun 'nemotron-legacy <recipe> --help' for more information.")
    else:
        print(f"Unknown recipe: {recipe}")
        print("Available recipes: nano3")
        sys.exit(1)


if __name__ == "__main__":
    main()
