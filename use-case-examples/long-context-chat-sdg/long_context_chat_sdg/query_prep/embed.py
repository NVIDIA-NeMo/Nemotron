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

"""Lightweight query embedding — MiniLM by default. Shared by dedup + cluster."""

from __future__ import annotations

from typing import List

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def embed_texts(texts: List[str], model_name: str = EMBED_MODEL):
    """One L2-normalized vector per text (numpy float32)."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name or EMBED_MODEL)
    emb = model.encode([t or " " for t in texts], batch_size=64, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype("float32")
