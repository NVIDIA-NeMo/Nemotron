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

"""A minimal OpenAI-compatible caller for the offline stages.

Returns ``caller(system, user) -> str``. Used where we do NOT need tool-calling
or the DD runtime (currently the decoupled judge in ``evaluate.py``).
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional


def make_openai_caller(model: str, endpoint: str, api_key_env: str = "NVIDIA_API_KEY",
                       params: Optional[Dict] = None) -> Callable[[str, str], str]:
    from openai import OpenAI
    # open endpoints need no key, but the OpenAI client requires a non-empty string
    client = OpenAI(base_url=endpoint, api_key=os.environ.get(api_key_env, "") or "EMPTY",
                    max_retries=4, timeout=120)
    params = params or {}

    def call(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **params)
        return resp.choices[0].message.content or ""

    return call
