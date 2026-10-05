# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Validate keep-list codes against the labels the model actually emits."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def normalise_code(value: object) -> str:
    """Strip ``__label__`` and any script suffix, casefolded."""
    code = str(value).strip().casefold()
    if code.startswith("__label__"):
        code = code.removeprefix("__label__")
    return code.split("_", 1)[0]


@lru_cache(maxsize=4)
def _model_labels(model_path: str) -> frozenset[str]:
    try:
        import fasttext

        labels = fasttext.load_model(str(model_path)).get_labels()
    except Exception as exc:  # noqa: BLE001 - make backend failures actionable configuration errors
        raise ValueError(f"could not read language labels from {model_path!r}: {exc}") from exc

    return frozenset(normalise_code(label) for label in labels)


def validate_language_codes(cfg: dict) -> None:
    """Refuse keep-list entries the configured FastText model cannot emit."""
    requested = list(cfg.get("language_codes") or [])
    if not requested:
        return

    model_path = (cfg.get("models") or {}).get("fasttext_langid")
    if not model_path:
        raise ValueError(
            "language_codes is non-empty but models.fasttext_langid is not set; "
            "provide a FastText language model or set language_codes to []"
        )
    if not Path(model_path).is_file():
        raise ValueError(f"models.fasttext_langid points at {model_path!r}, which is not a file")

    supported = _model_labels(str(model_path))
    unknown = [str(code) for code in requested if normalise_code(code) not in supported]
    if unknown:
        examples = ", ".join(sorted(supported)[:12])
        raise ValueError(
            f"language_codes contains {unknown}, but {model_path!r} emits none of those labels. "
            f"The model exposes {len(supported)} normalised label(s)"
            f"{f' (for example: {examples})' if examples else ''}. "
            "Use labels from this model, or set language_codes to [] to disable the language gate."
        )
