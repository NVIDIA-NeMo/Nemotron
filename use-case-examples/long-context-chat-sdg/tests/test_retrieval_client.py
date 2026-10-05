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

import random

import pytest

from long_context_chat_sdg.retrieval.client import HttpRetrievalClient


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _make_post(recorder):
    def post(endpoint, json=None, headers=None, timeout=None):
        recorder.append(json)
        n = json["top_k"]
        return _FakeResp({"chunks": [{"id": f"d{i}", "text": f"chunk {i}", "score": 1.0 - i * 0.01}
                                     for i in range(n)]})
    return post


def test_oversample_and_subsample_to_k():
    calls = []
    client = HttpRetrievalClient("http://x", oversample_factor=2, post_fn=_make_post(calls))
    got = client.retrieve("q", k=4, rng=random.Random(1))
    assert calls[0]["top_k"] == 8            # requested k * oversample_factor
    assert calls[0]["query"] == "q"
    assert len(got) == 4                       # subsampled back down to k


def test_no_dedup_across_calls():
    calls = []
    client = HttpRetrievalClient("http://x", oversample_factor=2, post_fn=_make_post(calls))
    first = client.retrieve("q", k=4, rng=random.Random(1))
    second = client.retrieve("q", k=4, rng=random.Random(1))
    # no cross-call dedup: identical rng + identical pool => identical selection
    assert [c.id for c in first] == [c.id for c in second]


def test_deterministic_given_rng():
    calls = []
    client = HttpRetrievalClient("http://x", oversample_factor=2, post_fn=_make_post(calls))
    a = client.retrieve("q", k=3, rng=random.Random(42))
    b = client.retrieve("q", k=3, rng=random.Random(42))
    assert [c.id for c in a] == [c.id for c in b]


def test_field_map_adapts_schema():
    def post(endpoint, json=None, headers=None, timeout=None):
        # the retrieval service returns a bare list with different field names
        return _FakeResp([{"passage_id": "p1", "body": "hello", "relevance": 0.9}])
    client = HttpRetrievalClient("http://x", oversample_factor=1, post_fn=post, field_map={
        "results_path": "", "id_field": "passage_id", "text_field": "body", "score_field": "relevance"})
    got = client.retrieve("q", k=1, rng=random.Random(0))
    assert got[0].id == "p1" and got[0].text == "hello"


def test_content_hash_id_when_service_gives_none():
    def post(endpoint, json=None, headers=None, timeout=None):
        return _FakeResp({"chunks": [{"rank": 1, "text": "some passage"}]})
    client = HttpRetrievalClient("http://x", oversample_factor=1, post_fn=post)
    got = client.retrieve("q", k=1, rng=random.Random(0))
    assert got[0].id.startswith("h") and got[0].text == "some passage"


def test_persistent_transport_failure_is_not_silently_empty():
    calls = []

    def post(*_args, **_kwargs):
        calls.append(1)
        raise ConnectionError("closed port")

    client = HttpRetrievalClient("http://127.0.0.1:9/search", max_retries=2,
                                 backoff=0, post_fn=post)
    with pytest.raises(RuntimeError, match=r"failed after 3 attempt\(s\).+closed port"):
        client.retrieve("q", k=2, rng=random.Random(0))
    assert len(calls) == 3


@pytest.mark.parametrize("payload, message", [
    ({"wrong_key": []}, "results_path"),
    ({"chunks": [{"id": "x", "text": ""}]}, "no non-empty"),
])
def test_invalid_response_mapping_fails_loudly(payload, message):
    client = HttpRetrievalClient(
        "http://x", max_retries=0,
        post_fn=lambda *_args, **_kwargs: _FakeResp(payload),
    )
    with pytest.raises(ValueError, match=message):
        client.retrieve("q", k=1, rng=random.Random(0))
