# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Small compatibility guard for saving resized Hugging Face checkpoints."""

from __future__ import annotations

from packaging.version import Version


def _shares_storage(left, right) -> bool:
    if left is right:
        return True
    try:
        return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            "Cannot verify whether the input and output embeddings share storage; "
            "refusing to rewrite tied-weight metadata."
        ) from exc


def normalize_untied_weight_metadata(model, transformers_version: str | None = None) -> bool:
    """Adapt legacy list metadata for Transformers 5 when embeddings are untied.

    Nemotron's remote model code can expose the Transformers 4-era list form of
    ``_tied_weights_keys``.  Transformers 5 expects a target-to-source mapping.
    An empty mapping is correct only when the config and the tensors both prove
    that the input and output embeddings are not tied.
    """
    if transformers_version is None:
        from transformers import __version__ as transformers_version

    if Version(transformers_version).major < 5:
        return False
    tied_keys = getattr(model, "_tied_weights_keys", None)
    if not isinstance(tied_keys, list):
        return False
    if getattr(model.config, "tie_word_embeddings", None) is not False:
        raise RuntimeError(
            "Transformers 5 requires mapping-form _tied_weights_keys, but the model "
            "declares tied word embeddings; refusing to discard that metadata."
        )

    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    if _shares_storage(input_weight, output_weight):
        raise RuntimeError(
            "Transformers 5 requires mapping-form _tied_weights_keys, but the input "
            "and output embeddings share storage; refusing to mark them untied."
        )

    model._tied_weights_keys = {}
    return True


def save_pretrained(model, output_dir) -> None:
    """Save after applying the narrowly scoped metadata compatibility guard."""
    normalize_untied_weight_metadata(model)
    model.save_pretrained(output_dir)
