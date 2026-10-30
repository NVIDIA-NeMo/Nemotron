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

(sdg-tool-call-data)=
# Generate Tool-Calling Data for SFT

Use this guide when you need multi-turn chat JSONL where the assistant issues OpenAI-style `tool_calls` and a `tool` role returns structured results, suitable for supervised fine-tuning (SFT) with a `tools` definition array.

You will use the sample config `customer_support_tools.yaml`, which produces ecommerce-style support threads. Each output row includes a `messages` array (with tool turns) and a `tools` array, ready for packing and training.

## Outcomes

- Understand how the shipped config asks one `llm_text` column to emit a full JSON multi-turn trace in a single model call.
- Preview, generate, and validate records before training.
- Know how to retarget seeds, prompts, and schema for your own domain.

## How It Works

Compared with single-turn configs such as `default.yaml`, this setup drives the whole conversation from one `llm_text` column.
The prompt tells the model to return a JSON object with `tools` and `messages` keys.
The `structured_messages` output projection parses that JSON object, extracts `messages` and `tools`, adds metadata, and serializes nested tool payload objects into OpenAI-compatible string fields.

```{literalinclude} ../../../src/nemotron/steps/sdg/data_designer/config/customer_support_tools.yaml
:language: yaml
:lines: 15-
:class: scrollable
```

Each seed row supplies five anchor fields the prompt interpolates: `customer_name`, `issue`, `order_id`, `product`, and `policy_hint`. Two extra category columns (`urgency`, `channel`) add variety without multiplying seed rows for every combination.

## Prerequisites

- Nemotron CLI available and working; if this is your first SDG run, complete {doc}`../getting-started`.
- `NVIDIA_API_KEY` set in the environment.
- The bundled seed file `data/customer_support_tool_seeds.jsonl` (shipped with the step). Add rows, or point the config at your own JSONL.

## Procedure

1. Preview two records so structured output matches the schema:

   ```console
   $ nemotron steps run sdg/data_designer -c customer_support_tools preview=true num_records=2
   ```

   In the preview, confirm:

   - Exactly one assistant message with `tool_calls`.
   - Exactly one `tool` message whose `tool_call_id` matches the call.
   - `function.arguments` and tool-message `content` are JSON strings after projection.
   - The assistant’s closing turn references the tool result (not a generic reply).
   - No markdown in message `content` if your trainer expects plain text.

2. Generate the dataset:

   ```console
   $ nemotron steps run sdg/data_designer -c customer_support_tools num_records=200
   ```

   Output path: `./output/sdg/customer_support_tool_sft.jsonl`.
   Spot-check a few lines. Each record exposes top-level `messages` and `tools` plus metadata, like the following example:

   ```text
   {
     "messages": [
       {"role": "system", "content": "You are a helpful ecommerce support agent..."},
       {"role": "user", "content": "Hi, I haven't received my headphones yet..."},
       {"role": "assistant", "content": "I'd be happy to help. Could you share your order number?"},
       {"role": "user", "content": "It's ORD-10492."},
       {"role": "assistant", "content": "", "tool_calls": [{"id": "call_001", "type": "function", "function": {"name": "lookup_order", "arguments": "{\"order_id\":\"ORD-10492\"}"}}]},
       {"role": "tool", "tool_call_id": "call_001", "name": "lookup_order", "content": "{\"status\":\"delayed\",\"eta\":\"tomorrow\"}"},
       {"role": "assistant", "content": "Your order is delayed and should arrive tomorrow. Per our policy, I can arrange an expedited replacement if you prefer."}
     ],
     "tools": [{"type": "function", "function": {"name": "lookup_order", "description": "...", "parameters": {...}}}],
     "customer_name": "Priya", "issue": "late delivery", "urgency": "frustrated", "channel": "web_chat"
   }
   ```

## Adapt to Your Domain

1. Replace or extend the seed file so rows cover your entities. You may rename the five anchor fields as long as the prompt and YAML refer to the same names.
2. Update `seed_dataset.fields` in the YAML to match those names.
3. Rewrite the `prompt` for your scenario and tool surface.
4. Adjust the JSON schema described in the prompt if the message layout changes, for example multiple tool calls per conversation.

Keep `output_projection` as `structured_messages` so the step extracts `messages` and `tools` from the structured column and merges category metadata onto each record.

## Validation Checklist

Before training, sample at least 50 records and verify:

- [ ] Every `tool_calls` block has a matching `tool` message with the same `tool_call_id`.
- [ ] `function.arguments` and tool-message `content` values are JSON strings in the projected JSONL.
- [ ] The assistant’s final reply uses the tool result (not a canned answer that ignores it).
- [ ] No unexpected markdown in `content` if the trainer assumes plain text.
- [ ] `tools` is present and non-empty on every record.

## Downstream Use

```text
customer_support_tool_sft.jsonl  →  data_prep/sft_packing  →  SFT training
```

The `structured_messages` projection writes `messages` and `tools` at the top level, matching formats common to AutoModel-style SFT and Megatron-Bridge-style workflows. Run `data_prep/sft_packing` in dry-run mode before a large training job to confirm the packer accepts your file.

## Next Steps

- Output projection reference: {doc}`../reference/output-projections` to learn the `structured_messages` schema.
- Config schema: {doc}`../reference/config-schema` for column types and output projections.
