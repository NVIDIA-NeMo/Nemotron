<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

(sdg-output-projections)=
# Output Projections

The `output_projection` block in a config maps raw Data Designer records into the schema expected by downstream training steps. Each projection type extracts specific columns and writes one JSON object per line.

## OpenAI Messages

Produces single-turn OpenAI chat-format records. Use for SFT chat data that feeds `data_prep/sft_packing` or AutoModel SFT.

**YAML**:

```yaml
output_projection:
  type: openai_messages
  user_field: user_query        # column containing the user turn
  assistant_field: assistant_response  # column containing the assistant turn
  metadata_fields: [persona, topic]    # additional columns to include at top level
```

**Output** (one JSON object per line):

```json
{
  "messages": [
    {"role": "user", "content": "How do I calibrate the sensor threshold?"},
    {"role": "assistant", "content": "Set the threshold in the device settings under Calibration → Sensor Range. A value of 0.85 works well for most environments."}
  ],
  "persona": "engineer",
  "topic": "industrial sensor calibration"
}
```

Fields:

| Field | Required | Description |
|---|---|---|
| `type` | yes | `"openai_messages"` |
| `user_field` | yes | Column name for the user message content |
| `assistant_field` | yes | Column name for the assistant message content |
| `metadata_fields` | no | List of additional column names to include at the top level |

## DPO Preference

Produces preference triples for DPO training. Use with `rl_pref.yaml` and the `llm_judge` column pattern. Output feeds `data_prep/rl_prep`.

**YAML**:

```yaml
output_projection:
  type: dpo_preference
  prompt_field: prompt          # column containing the input prompt
  response_a_field: response_a  # column containing the first candidate response
  response_b_field: response_b  # column containing the second candidate response
  judge_field: judge            # column containing the judge's structured output
  winner_field: winner          # key inside the judge output that holds "A" or "B"
```

**Output** (one JSON object per line):

```json
{
  "prompt": "Explain why retrieval-augmented generation can reduce hallucinations in enterprise assistants.",
  "chosen": "RAG grounds the model in retrieved passages, so factual claims are tied to source documents rather than purely to learned weights.",
  "rejected": "RAG is better because it uses the internet and knows more things than a regular model."
}
```

Fields:

| Field | Required | Description |
|---|---|---|
| `type` | yes | `"dpo_preference"` |
| `prompt_field` | yes | Column name for the input prompt |
| `response_a_field` | yes | Column name for candidate A |
| `response_b_field` | yes | Column name for candidate B |
| `judge_field` | yes | Column name for the judge's structured output |
| `winner_field` | yes | Key within the judge output JSON that holds `"A"` or `"B"` |

The projection raises `ValueError` if `winner` is not `"A"` or `"B"`. The `llm_judge` column must be configured to return exactly this structure.

## Structured Messages

Produces multi-turn records with `messages` and an optional `tools` array.
The shipped `customer_support_tools.yaml` config generates this shape with an `llm_text` column that returns a JSON object; `llm_structured` columns can also feed this projection when their output is a mapping with the same fields.
Output feeds `data_prep/sft_packing` or AutoModel SFT.

**YAML**:

```yaml
output_projection:
  type: structured_messages
  source_field: conversation    # column containing the structured JSON object
  messages_field: messages      # key inside the structured object for the messages array
  tools_field: tools            # key inside the structured object for the tools array
  metadata_fields: [customer_name, issue, urgency, channel]
```

**Output** (one JSON object per line):

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful ecommerce support agent."},
    {"role": "user", "content": "I haven't received my order yet."},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_001", "type": "function", "function": {"name": "lookup_order", "arguments": "{\"order_id\":\"ORD-10492\"}"}}]},
    {"role": "tool", "tool_call_id": "call_001", "name": "lookup_order", "content": "{\"status\":\"delayed\",\"eta\":\"tomorrow\"}"},
    {"role": "assistant", "content": "Your order is delayed and will arrive tomorrow. I can arrange an expedited replacement if needed."}
  ],
  "tools": [{"type": "function", "function": {"name": "lookup_order", "description": "Look up order status by ID.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}}],
  "customer_name": "Priya",
  "issue": "late delivery",
  "urgency": "frustrated",
  "channel": "web_chat"
}
```

Fields:

| Field | Required | Description |
|---|---|---|
| `type` | yes | `"structured_messages"` |
| `source_field` | yes | Column containing the structured JSON conversation object |
| `messages_field` | no | Key in `source_field` for the messages array. Default: `"messages"` |
| `tools_field` | no | Key in `source_field` for the tools array. Omitted from output if not present in the record |
| `metadata_fields` | no | List of additional column names to include at the top level |

The `source_field` column value may be a JSON string or a dict; both are handled.

## Related Information

- {doc}`config-schema` — Full YAML config field reference.
- {doc}`../how-to/tool-call-data` — Using `structured_messages` with `customer_support_tools.yaml`.
- {doc}`../how-to/preference-data` — Using `dpo_preference` with `rl_pref.yaml`.
