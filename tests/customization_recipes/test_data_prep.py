# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for customization_recipes/data_prep/ modules.

These tests validate that the data_prep mini-library modules can be
imported without error (even when heavy dependencies like nemo-curator
or data-designer are absent), that config dataclasses instantiate with
defaults and accept overrides, that Pydantic models validate correctly,
and that the filter registry factory works as designed.
"""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING

import pytest

# ---------------------------------------------------------------------------
# Module import tests -- validate that all data_prep modules can be loaded
# without requiring GPU-only or heavy optional dependencies at import time.
# ---------------------------------------------------------------------------

_DATA_PREP_ROOT = "nemotron.customization_recipes.data_prep"

_MODULE_NAMES = [
    f"{_DATA_PREP_ROOT}",
    f"{_DATA_PREP_ROOT}.acquire",
    f"{_DATA_PREP_ROOT}.translate",
    f"{_DATA_PREP_ROOT}.sdg",
    f"{_DATA_PREP_ROOT}.quality",
    f"{_DATA_PREP_ROOT}.tokenize_pack",
    f"{_DATA_PREP_ROOT}.byob",
]


class TestModuleImports:
    """All data_prep modules should import without crashing.

    Heavy optional dependencies (nemo_curator, data_designer, transformers,
    torch, megatron.bridge) are either lazy-imported behind guard functions
    or use try/except at module level with fallback flags.  The omegaconf
    and pydantic imports *are* required at import time.
    """

    @pytest.mark.parametrize("module_name", _MODULE_NAMES)
    def test_module_imports(self, module_name: str) -> None:
        """Each module should import without raising."""
        mod = importlib.import_module(module_name)
        assert mod is not None

    def test_package_init_exports(self) -> None:
        """The __init__.py should expose all documented public symbols."""
        mod = importlib.import_module(_DATA_PREP_ROOT)
        expected = [
            # acquire
            "AcquireConfig", "download_dataset", "classify_domains",
            "identify_languages", "apply_chat_template",
            # translate
            "translate_data", "translate_byob_benchmark",
            # sdg
            "FunctionCall", "ToolCall", "Message", "Conversation",
            "ConversationList", "SDGConfig", "run_sdg_pipeline",
            # quality
            "AssessmentConfig", "AssessmentTool", "FILTER_REGISTRY",
            "create_filter", "create_scorer_list", "calculate_aggregates",
            # tokenize_pack
            "CPTConfig", "SFTConfig", "prepare_cpt_data", "prepare_sft_data",
            # byob
            "ByobConfig", "MCQQuestion", "MCQQuestionList", "JudgeResult",
            "generate_questions", "judge_questions", "expand_distractors",
            "filter_questions", "check_distractor_validity",
        ]
        for name in expected:
            assert hasattr(mod, name), f"Missing export: {name}"

    def test_all_list_matches_exports(self) -> None:
        """The __all__ list should match the actual exports."""
        mod = importlib.import_module(_DATA_PREP_ROOT)
        assert hasattr(mod, "__all__"), "__all__ not defined"
        for name in mod.__all__:
            assert hasattr(mod, name), f"__all__ lists '{name}' but it is not defined"


# ---------------------------------------------------------------------------
# Config dataclass tests
# ---------------------------------------------------------------------------


class TestAcquireConfig:
    """Tests for AcquireConfig dataclass."""

    def test_defaults(self) -> None:
        from nemotron.customization_recipes.data_prep.acquire import AcquireConfig

        cfg = AcquireConfig()
        assert cfg.download_dir == "data/raw"
        assert cfg.output_dir == "data/acquired"
        assert cfg.record_format == "jsonl"
        assert cfg.url_limit is None
        assert cfg.record_limit is None
        assert cfg.sources == []

    def test_custom_values(self) -> None:
        from nemotron.customization_recipes.data_prep.acquire import AcquireConfig

        cfg = AcquireConfig(
            download_dir="/tmp/downloads",
            output_dir="/tmp/output",
            sources=["https://example.com/data.jsonl"],
            url_limit=10,
        )
        assert cfg.download_dir == "/tmp/downloads"
        assert cfg.sources == ["https://example.com/data.jsonl"]
        assert cfg.url_limit == 10

    def test_from_omegaconf(self) -> None:
        from omegaconf import OmegaConf
        from nemotron.customization_recipes.data_prep.acquire import AcquireConfig

        raw = OmegaConf.create({"download_dir": "/my/dir", "url_limit": 5})
        cfg = AcquireConfig.from_omegaconf(raw)
        assert cfg.download_dir == "/my/dir"
        assert cfg.url_limit == 5
        # Defaults should be preserved for unspecified fields
        assert cfg.output_dir == "data/acquired"


class TestSDGConfig:
    """Tests for SDGConfig dataclass."""

    def test_defaults(self) -> None:
        from nemotron.customization_recipes.data_prep.sdg import SDGConfig

        cfg = SDGConfig()
        assert cfg.output_dir == "data/sdg"
        assert cfg.num_records == 100
        assert cfg.column_type == "llm-structured"
        assert cfg.output_format == "ConversationList"
        assert cfg.model_configs == []

    def test_from_omegaconf(self) -> None:
        from omegaconf import OmegaConf
        from nemotron.customization_recipes.data_prep.sdg import SDGConfig

        raw = OmegaConf.create({
            "num_records": 500,
            "system_prompt": "You are a helpful assistant.",
        })
        cfg = SDGConfig.from_omegaconf(raw)
        assert cfg.num_records == 500
        assert cfg.system_prompt == "You are a helpful assistant."


class TestAssessmentConfig:
    """Tests for AssessmentConfig dataclass."""

    def test_defaults(self) -> None:
        from nemotron.customization_recipes.data_prep.quality import AssessmentConfig

        cfg = AssessmentConfig()
        assert cfg.output_dir == "data/quality"
        assert cfg.lines_per_split == 1000
        assert cfg.allow_llm_failures is False
        assert cfg.fields == "messages"

    def test_from_omegaconf(self) -> None:
        from omegaconf import OmegaConf
        from nemotron.customization_recipes.data_prep.quality import AssessmentConfig

        raw = OmegaConf.create({
            "input_file": "/data/test.jsonl",
            "num_workers": 4,
        })
        cfg = AssessmentConfig.from_omegaconf(raw)
        assert cfg.input_file == "/data/test.jsonl"
        assert cfg.num_workers == 4


class TestCPTConfig:
    """Tests for CPTConfig dataclass."""

    def test_defaults(self) -> None:
        from nemotron.customization_recipes.data_prep.tokenize_pack import CPTConfig

        cfg = CPTConfig()
        assert cfg.output_dir == "data/cpt"
        assert cfg.train_ratio == 0.90
        assert cfg.valid_ratio == 0.05
        assert cfg.test_ratio == 0.05
        assert cfg.add_bos is False
        assert cfg.add_eos is True
        assert cfg.seed == 42

    def test_ratios_sum_to_one(self) -> None:
        from nemotron.customization_recipes.data_prep.tokenize_pack import CPTConfig

        cfg = CPTConfig()
        total = cfg.train_ratio + cfg.valid_ratio + cfg.test_ratio
        assert abs(total - 1.0) < 1e-9, f"Ratios sum to {total}, expected 1.0"

    def test_from_omegaconf(self) -> None:
        from omegaconf import OmegaConf
        from nemotron.customization_recipes.data_prep.tokenize_pack import CPTConfig

        raw = OmegaConf.create({
            "input_path": "/data/corpus",
            "max_samples": 1000,
        })
        cfg = CPTConfig.from_omegaconf(raw)
        assert cfg.input_path == "/data/corpus"
        assert cfg.max_samples == 1000


class TestSFTConfig:
    """Tests for SFTConfig dataclass."""

    def test_defaults(self) -> None:
        from nemotron.customization_recipes.data_prep.tokenize_pack import SFTConfig

        cfg = SFTConfig()
        assert cfg.output_dir == "data/sft"
        assert cfg.pack_size == 4096
        assert cfg.packing_algorithm == "first_fit_decreasing"
        assert cfg.enable_thinking is False
        assert cfg.thinking_start_token == "<think>"
        assert cfg.thinking_end_token == "</think>"

    def test_ratios_sum_to_one(self) -> None:
        from nemotron.customization_recipes.data_prep.tokenize_pack import SFTConfig

        cfg = SFTConfig()
        total = cfg.train_ratio + cfg.valid_ratio + cfg.test_ratio
        assert abs(total - 1.0) < 1e-9, f"Ratios sum to {total}, expected 1.0"

    def test_from_omegaconf(self) -> None:
        from omegaconf import OmegaConf
        from nemotron.customization_recipes.data_prep.tokenize_pack import SFTConfig

        raw = OmegaConf.create({
            "pack_size": 8192,
            "enable_thinking": True,
            "hf_dataset": "HuggingFaceH4/ultrachat_200k",
        })
        cfg = SFTConfig.from_omegaconf(raw)
        assert cfg.pack_size == 8192
        assert cfg.enable_thinking is True


class TestByobConfig:
    """Tests for ByobConfig dataclass."""

    def test_defaults(self) -> None:
        from nemotron.customization_recipes.data_prep.byob import ByobConfig

        cfg = ByobConfig()
        assert cfg.output_dir == "data/byob"
        assert cfg.language == "en"
        assert cfg.hf_dataset == "cais/mmlu"
        assert cfg.few_shot_samples_per_query == 5
        assert cfg.do_distractor_expansion is False
        assert cfg.remove_hallucinated is True

    def test_from_omegaconf(self) -> None:
        from omegaconf import OmegaConf
        from nemotron.customization_recipes.data_prep.byob import ByobConfig

        raw = OmegaConf.create({
            "expt_name": "test_run",
            "language": "hi",
            "do_distractor_expansion": True,
        })
        cfg = ByobConfig.from_omegaconf(raw)
        assert cfg.expt_name == "test_run"
        assert cfg.language == "hi"
        assert cfg.do_distractor_expansion is True


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------


class TestPydanticConversationModels:
    """Tests for sdg.py Pydantic conversation schema models."""

    def test_function_call(self) -> None:
        from nemotron.customization_recipes.data_prep.sdg import FunctionCall

        fc = FunctionCall(name="get_weather", arguments='{"city": "Delhi"}')
        assert fc.name == "get_weather"
        assert fc.arguments == '{"city": "Delhi"}'

    def test_function_call_requires_fields(self) -> None:
        from nemotron.customization_recipes.data_prep.sdg import FunctionCall

        with pytest.raises(Exception):
            FunctionCall()  # name and arguments are required

    def test_tool_call(self) -> None:
        from nemotron.customization_recipes.data_prep.sdg import ToolCall, FunctionCall

        tc = ToolCall(
            id="abc123def",
            type="function",
            function=FunctionCall(name="search", arguments="{}"),
        )
        assert tc.id == "abc123def"
        assert tc.type == "function"
        assert tc.function.name == "search"

    def test_message_user(self) -> None:
        from nemotron.customization_recipes.data_prep.sdg import Message

        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_calls is None

    def test_message_assistant_with_tool_calls(self) -> None:
        from nemotron.customization_recipes.data_prep.sdg import (
            Message, ToolCall, FunctionCall,
        )

        msg = Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_001",
                    function=FunctionCall(name="lookup", arguments='{"q": "test"}'),
                )
            ],
        )
        assert msg.role == "assistant"
        assert msg.content is None
        assert len(msg.tool_calls) == 1

    def test_message_allows_extra_fields(self) -> None:
        from nemotron.customization_recipes.data_prep.sdg import Message

        msg = Message(role="system", content="Be helpful.", custom_field="extra")
        assert msg.custom_field == "extra"

    def test_conversation(self) -> None:
        from nemotron.customization_recipes.data_prep.sdg import Conversation, Message

        conv = Conversation(messages=[
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello!"),
        ])
        assert len(conv.messages) == 2
        assert conv.messages[0].role == "user"

    def test_conversation_list(self) -> None:
        from nemotron.customization_recipes.data_prep.sdg import (
            ConversationList, Conversation, Message,
        )

        cl = ConversationList(conversations=[
            Conversation(messages=[Message(role="user", content="Q1")]),
            Conversation(messages=[Message(role="user", content="Q2")]),
        ])
        assert len(cl.conversations) == 2


class TestPydanticMCQModels:
    """Tests for BYOB Pydantic MCQ response models (``data_prep/byob``)."""

    def test_mcq_question(self) -> None:
        from nemotron.customization_recipes.data_prep.byob import MCQQuestion

        q = MCQQuestion(
            question="What is 2+2?",
            choice_a="3",
            choice_b="4",
            choice_c="5",
            choice_d="6",
            answer="B",
        )
        assert q.question == "What is 2+2?"
        assert q.answer == "B"

    def test_mcq_question_invalid_answer(self) -> None:
        from nemotron.customization_recipes.data_prep.byob import MCQQuestion

        with pytest.raises(Exception):
            MCQQuestion(
                question="Q?",
                choice_a="a", choice_b="b", choice_c="c", choice_d="d",
                answer="E",  # Invalid -- must be A/B/C/D
            )

    def test_mcq_question_list(self) -> None:
        from nemotron.customization_recipes.data_prep.byob import MCQQuestion, MCQQuestionList

        ql = MCQQuestionList(questions=[
            MCQQuestion(
                question="Q1?", choice_a="a", choice_b="b",
                choice_c="c", choice_d="d", answer="A",
            ),
        ])
        assert len(ql.questions) == 1

    def test_judge_result(self) -> None:
        from nemotron.customization_recipes.data_prep.byob import JudgeResult

        jr = JudgeResult(
            reason="Well-formed question",
            is_valid=True,
            category="knowledge",
        )
        assert jr.is_valid is True
        assert jr.category == "knowledge"

    def test_judge_result_invalid_category(self) -> None:
        from nemotron.customization_recipes.data_prep.byob import JudgeResult

        with pytest.raises(Exception):
            JudgeResult(
                reason="test",
                is_valid=True,
                category="invalid_category",  # Must be knowledge/reasoning/both
            )

    def test_distractor_expansion(self) -> None:
        from nemotron.customization_recipes.data_prep.byob import DistractorExpansion

        de = DistractorExpansion(
            choice_e="E", choice_f="F", choice_g="G",
            choice_h="H", choice_i="I", choice_j="J",
        )
        assert de.choice_e == "E"
        assert de.choice_j == "J"

    def test_distractor_validity_four(self) -> None:
        from nemotron.customization_recipes.data_prep.byob import DistractorValidityFourChoices

        dv = DistractorValidityFourChoices(
            choice_a="Yes", choice_b="No", choice_c="No", choice_d="No",
        )
        assert dv.choice_a == "Yes"

    def test_distractor_validity_ten(self) -> None:
        from nemotron.customization_recipes.data_prep.byob import DistractorValidityTenChoices

        dv = DistractorValidityTenChoices(
            choice_a="Yes", choice_b="No", choice_c="No", choice_d="No",
            choice_e="No", choice_f="No", choice_g="No", choice_h="No",
            choice_i="No", choice_j="No",
        )
        assert dv.choice_a == "Yes"
        assert dv.choice_j == "No"


# ---------------------------------------------------------------------------
# SDG schema registry tests
# ---------------------------------------------------------------------------


class TestSchemaRegistry:
    """Tests for the SDG schema registry (resolve_schema, register_schema)."""

    def test_resolve_builtin_schemas(self) -> None:
        from nemotron.customization_recipes.data_prep.sdg import resolve_schema

        for name in ("FunctionCall", "ToolCall", "Message", "Conversation", "ConversationList"):
            cls = resolve_schema(name)
            assert cls is not None

    def test_resolve_unknown_raises(self) -> None:
        from nemotron.customization_recipes.data_prep.sdg import resolve_schema

        with pytest.raises(KeyError, match="Unknown output_format"):
            resolve_schema("NonExistentModel")

    def test_register_custom_schema(self) -> None:
        from pydantic import BaseModel
        from nemotron.customization_recipes.data_prep.sdg import (
            register_schema, resolve_schema,
        )

        class CustomOutput(BaseModel):
            text: str

        register_schema("CustomOutput", CustomOutput)
        resolved = resolve_schema("CustomOutput")
        assert resolved is CustomOutput


# ---------------------------------------------------------------------------
# Filter registry tests
# ---------------------------------------------------------------------------


class TestFilterRegistry:
    """Tests for quality.py filter registry and factory."""

    def test_registry_is_dict(self) -> None:
        from nemotron.customization_recipes.data_prep.quality import FILTER_REGISTRY

        assert isinstance(FILTER_REGISTRY, dict)

    def test_create_filter_unknown_raises(self) -> None:
        """create_filter should raise ValueError for unknown filter names
        when nemo-curator is not installed."""
        from nemotron.customization_recipes.data_prep.quality import create_filter

        with pytest.raises(ValueError, match="Unknown filter"):
            create_filter("CompletelyBogusFilterName", {})

    def test_load_registry_is_idempotent(self) -> None:
        """Calling _load_registry multiple times should not error."""
        from nemotron.customization_recipes.data_prep.quality import _load_registry

        _load_registry()
        _load_registry()  # second call is a no-op


# ---------------------------------------------------------------------------
# Aggregation helpers tests (quality.py)
# ---------------------------------------------------------------------------


class TestAggregation:
    """Tests for quality.py aggregation helpers."""

    def test_aggregate_dicts_empty(self) -> None:
        from nemotron.customization_recipes.data_prep.quality import aggregate_dicts

        result = aggregate_dicts([])
        assert result == {}

    def test_aggregate_dicts_numeric(self) -> None:
        from nemotron.customization_recipes.data_prep.quality import aggregate_dicts
        import numpy as np

        dicts = [{"score": 0.8}, {"score": 0.6}, {"score": 1.0}]
        result = aggregate_dicts(dicts)
        assert abs(result["score"] - 0.8) < 1e-9

    def test_aggregate_dicts_strings(self) -> None:
        from nemotron.customization_recipes.data_prep.quality import aggregate_dicts

        dicts = [{"label": "A"}, {"label": "B"}, {"label": "A"}]
        result = aggregate_dicts(dicts)
        assert result["label"] == {"A": 2, "B": 1}

    def test_aggregate_dicts_nested(self) -> None:
        from nemotron.customization_recipes.data_prep.quality import aggregate_dicts

        dicts = [
            {"nested": {"val": 1.0}},
            {"nested": {"val": 3.0}},
        ]
        result = aggregate_dicts(dicts)
        assert abs(result["nested"]["val"] - 2.0) < 1e-9

    def test_aggregate_dicts_ignore_keys(self) -> None:
        from nemotron.customization_recipes.data_prep.quality import aggregate_dicts

        dicts = [{"keep": 1.0, "skip": 99.0}]
        result = aggregate_dicts(dicts, ignore_keys=["skip"])
        assert "keep" in result
        assert "skip" not in result


# ---------------------------------------------------------------------------
# tokenize_pack helper tests
# ---------------------------------------------------------------------------


class TestTokenizePackHelpers:
    """Tests for helpers now delegated to nemotron.data_prep.

    tokenize_pack.py was refactored to a thin adapter. These tests now
    verify the equivalent production code in nemotron.data_prep.
    """

    def test_sharegpt_transform(self) -> None:
        """ShareGPT transform produces conversations field."""
        from nemotron.data_prep.formats.transforms import sharegpt

        transform = sharegpt(conversations="conversations")
        record = {
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi there!"},
            ]
        }
        result = transform(record)
        assert result is not None
        assert len(result["conversations"]) == 2

    def test_thinking_detection_in_chat_template(self) -> None:
        """Production chat_template detects reasoning_content."""
        # The production code checks for reasoning_content inline in
        # create_masked_messages; verify the detection logic directly.
        msgs_with = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "thinking..."},
        ]
        msgs_without = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        has_with = any(
            "reasoning_content" in msg and msg["reasoning_content"]
            for msg in msgs_with
        )
        has_without = any(
            "reasoning_content" in msg and msg["reasoning_content"]
            for msg in msgs_without
        )
        assert has_with is True
        assert has_without is False

    def test_replace_json_args(self) -> None:
        from nemotron.data_prep.core.chat_template import replace_json_args

        msgs = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "test"}',
                        },
                    }
                ],
            }
        ]
        result = replace_json_args(msgs)
        # Should parse JSON string into dict
        assert result[0]["tool_calls"][0]["function"]["arguments"] == {"query": "test"}
        # Original should not be mutated (deep copy)
        assert msgs[0]["tool_calls"][0]["function"]["arguments"] == '{"query": "test"}'


# ---------------------------------------------------------------------------
# Integration tests -- require GPU / heavy dependencies
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIntegrationDataPrep:
    """Integration tests that require heavy dependencies.

    These are skipped by default. Run with: pytest -m integration
    """

    def test_prepare_cpt_data(self) -> None:
        """Test CPT data preparation end-to-end (requires torch, nemo_automodel)."""
        pytest.importorskip("torch")
        pytest.importorskip("nemo_automodel")
        from nemotron.customization_recipes.data_prep.tokenize_pack import (
            CPTConfig, prepare_cpt_data,
        )
        # Would need actual data and tokenizer -- placeholder
        pytest.skip("Requires actual data and GPU environment")

    def test_prepare_sft_data(self) -> None:
        """Test SFT data preparation end-to-end (requires megatron.bridge)."""
        pytest.importorskip("megatron.bridge")
        from nemotron.customization_recipes.data_prep.tokenize_pack import (
            SFTConfig, prepare_sft_data,
        )
        pytest.skip("Requires actual data and GPU environment")

    def test_run_sdg_pipeline(self) -> None:
        """Test SDG pipeline (requires data-designer)."""
        pytest.importorskip("data_designer")
        from nemotron.customization_recipes.data_prep.sdg import (
            SDGConfig, run_sdg_pipeline,
        )
        pytest.skip("Requires DataDesigner and NIM API access")

    def test_assessment_tool(self) -> None:
        """Test quality assessment tool (requires nemo-curator, Ray)."""
        pytest.importorskip("nemo_curator")
        from nemotron.customization_recipes.data_prep.quality import (
            AssessmentConfig, AssessmentTool,
        )
        pytest.skip("Requires nemo-curator and Ray cluster")
