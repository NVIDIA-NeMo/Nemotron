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

import json

import pytest

from long_context_chat_sdg.core.persona_loader import (
    attach_personas_to_seeds,
    normalize_persona_record,
    parse_hf_dataset_id,
    persona_source_is_external,
    sample_personas,
)


def test_parse_hf_dataset_id_from_url_and_repo():
    assert parse_hf_dataset_id("nvidia/Nemotron-Personas-Vietnam") == "nvidia/Nemotron-Personas-Vietnam"
    assert parse_hf_dataset_id(
        "https://huggingface.co/datasets/nvidia/Nemotron-Personas-Vietnam"
    ) == "nvidia/Nemotron-Personas-Vietnam"
    assert parse_hf_dataset_id(
        "https://huggingface.co/datasets/nvidia/Nemotron-Personas-Vietnam/tree/main"
    ) == "nvidia/Nemotron-Personas-Vietnam"


def test_parse_hf_dataset_id_rejects_bad_input():
    with pytest.raises(ValueError):
        parse_hf_dataset_id("not-a-repo")
    with pytest.raises(ValueError):
        parse_hf_dataset_id("")


def test_normalize_maps_region_to_state():
    rec = normalize_persona_record({
        "persona": "A curious student.",
        "region": "Thành Phố Đà Nẵng",
        "country": "Việt Nam",
        "age": "34",
        "uuid": "ignored",
    })
    assert rec["state"] == "Thành Phố Đà Nẵng"
    assert rec["age"] == 34
    assert "uuid" not in rec


def test_sample_and_attach_personas(tmp_path):
    pool = [{"persona": f"p{i}", "region": "HN", "country": "VN"} for i in range(5)]
    picks = sample_personas(pool, 3, seed=7)
    assert len(picks) == 3
    assert all("state" in p or "region" in p for p in picks)

    seeds = [{"query": "q1"}, {"query": "q2"}]
    # attach via local jsonl without hitting HF
    path = tmp_path / "personas.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for p in pool:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    out = attach_personas_to_seeds(seeds, {"local_path": str(path), "seed": 1})
    assert len(out) == 2
    for row in out:
        persona = json.loads(row["persona"])
        assert "persona" in persona


def test_persona_source_is_external():
    assert persona_source_is_external({"hf_dataset": "nvidia/Nemotron-Personas-Vietnam"})
    assert persona_source_is_external({"local_path": "./x.parquet"})
    assert not persona_source_is_external({"enabled": True, "locale": "en_IN"})
