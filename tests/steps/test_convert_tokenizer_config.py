"""`convert/megatron_to_hf` must emit a tokenizer any consumer can load.

Megatron-Bridge writes its own wrapper name into the export's
`tokenizer_config.json`. vLLM serves such a directory fine, so the defect stays
hidden until something tokenizes client-side — log-likelihood evals, for
instance — and then it breaks for every consumer of that export.
"""

from __future__ import annotations

import json

import pytest

from nemotron.steps._runners.convert import normalize_exported_tokenizer_config


def _export(tmp_path, config: dict, *, with_tokenizer_json: bool = True):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
    if with_tokenizer_json:
        (tmp_path / "tokenizer.json").write_text("{}")
    return tmp_path


def _read(tmp_path) -> dict:
    return json.loads((tmp_path / "tokenizer_config.json").read_text())


def test_internal_wrapper_name_is_replaced(tmp_path):
    """The exact failure: `ValueError: Tokenizer class TokenizersBackend does
    not exist or is not currently imported`."""
    _export(tmp_path, {"tokenizer_class": "TokenizersBackend", "backend": "tokenizers", "is_local": False})
    assert normalize_exported_tokenizer_config(tmp_path) is True
    config = _read(tmp_path)
    assert config["tokenizer_class"] == "PreTrainedTokenizerFast"
    # Keys transformers does not understand.
    assert "backend" not in config
    assert "is_local" not in config


@pytest.mark.parametrize("name", ["TokenizersBackend", "HuggingFaceTokenizer", "MegatronTokenizer"])
def test_all_known_internal_names(tmp_path, name):
    _export(tmp_path, {"tokenizer_class": name})
    assert normalize_exported_tokenizer_config(tmp_path) is True
    assert _read(tmp_path)["tokenizer_class"] == "PreTrainedTokenizerFast"


def test_a_real_transformers_class_is_left_alone(tmp_path):
    """Exports that already name a real class must not be rewritten."""
    original = {"tokenizer_class": "LlamaTokenizerFast", "model_max_length": 8192}
    _export(tmp_path, original)
    assert normalize_exported_tokenizer_config(tmp_path) is False
    assert _read(tmp_path) == original


def test_other_settings_are_preserved(tmp_path):
    _export(tmp_path, {"tokenizer_class": "TokenizersBackend", "model_max_length": 262144, "bos_token": "<s>"})
    normalize_exported_tokenizer_config(tmp_path)
    config = _read(tmp_path)
    assert config["model_max_length"] == 262144
    assert config["bos_token"] == "<s>"


def test_without_tokenizer_json_it_warns_instead_of_guessing(tmp_path, capsys):
    """`PreTrainedTokenizerFast` is only correct when there is a tokenizer.json
    to load; otherwise writing it would just fail differently."""
    _export(tmp_path, {"tokenizer_class": "TokenizersBackend"}, with_tokenizer_json=False)
    assert normalize_exported_tokenizer_config(tmp_path) is False
    assert _read(tmp_path)["tokenizer_class"] == "TokenizersBackend"
    assert "AutoTokenizer will fail" in capsys.readouterr().out


def test_a_missing_or_unreadable_config_does_not_fail_the_conversion(tmp_path, capsys):
    """A conversion that produced weights must not die over tokenizer metadata."""
    assert normalize_exported_tokenizer_config(tmp_path) is False  # no file at all
    (tmp_path / "tokenizer_config.json").write_text("{ not json")
    assert normalize_exported_tokenizer_config(tmp_path) is False
    assert "could not read" in capsys.readouterr().out
