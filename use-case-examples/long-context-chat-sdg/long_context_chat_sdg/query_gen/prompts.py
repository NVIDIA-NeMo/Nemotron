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

"""Prompts for query generation — language- and domain-agnostic.

The model writes a realistic END-USER question answerable ONLY from the supplied
source passage(s). It must write in the SAME language as the passages and must not
lift their wording (real users don't phrase like the source). Kind directives shape
the question type; multi-chunk kinds force a question that spans several passages.
"""

from __future__ import annotations

QUERY_GEN_SYSTEM = """You write realistic questions that a real person would ask an AI assistant which can \
search a knowledge base. You are given one or more SOURCE PASSAGES from that knowledge base; your question must \
be answerable FROM THOSE PASSAGES.

<RULES>
- Write the question in the SAME LANGUAGE as the source passages.
- Ground it in the passages' actual content, but phrase it in a natural end-user voice — do NOT copy the
  passages' wording.
- NEVER refer to the source material: do not say "the passage", "the document", "Passage 1/2", "the text
  above", ids, or otherwise reveal that any text was given to you. Ask as if you already have the question in mind.
- Ask something a curious non-expert would genuinely ask; it must be answerable from the passages' content.
- Do NOT invent specifics that are not supported by the passages.
- Output STRICTLY a JSON object and nothing else: {{"query": "<the question>"}}
</RULES>"""

# per-kind directive injected into the turn prompt
KIND_DIRECTIVES = {
    "factual": "Ask a specific, concrete factual question answerable from the passage.",
    "multi_hop": "The passages below are on a RELATED theme. Ask ONE natural, non-trivial question whose full "
                 "answer requires DRAWING ON SEVERAL of them together — no single passage should be enough on its "
                 "own. If they genuinely don't connect, ask a single natural question from the most substantive one.",
    "comparative": "The passages cover related situations. Ask a natural question that COMPARES or contrasts what "
                   "they say (e.g. how the treatment or outcome differs across them). Do not force a comparison "
                   "that isn't supported; if they don't compare, ask one question from the most substantive passage.",
    "exploratory": "The passages touch a shared topic from different angles. Ask an open-ended question inviting a "
                   "broad explanation that PULLS TOGETHER what the several passages cover.",
    "ambiguous": "Ask a genuinely VAGUE, open question about the passage's broad topic — the kind a real person "
                 "asks before they know the specifics. Leave out the particular names, sections, dates, or exact "
                 "sub-point, so an assistant would naturally need to ask what you mean. Keep it short and natural.",
}

QUERY_GEN_TURN = """Source passage(s) from the knowledge base:
<PASSAGES>
{passages}
</PASSAGES>

Task: {directive}

Return only the JSON object with your question."""
