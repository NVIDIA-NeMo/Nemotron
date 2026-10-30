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

(sdg-using-skills)=
# Use the SDG Skill With Confidence

This page is for newcomers to model training and new to *synthetic data generation (SDG)*.
The main goal is to help you run a productive, efficient session with a coding agent: less back-and-forth, fewer clarifying questions, and clearer handoffs between what you decide and what the agent edits in the repository.

This page aligns with the `nemotron steps run sdg/data_designer` command.
Use an agent to translate your intent into the right YAML, seed files, and `nemotron` commands.

## Keeping an Agent Session Productive

Provide a short brief you write yourself, not something the agent drafts for you:

```{div} sd-font-italic sd-font-weight-lighter
- "We need data for a model that can answer short questions about our company’s travel and expense policy."
- "I need multi-turn  conversations for a retail support bot that can call tools such as order lookup and return eligibility. The tone must be friendly and concise."
```

Ask the agent to start from shipped configs and the {doc}`getting-started` flow unless there is a strong reason to invent a new layout.

If you want a reusable shape, you can copy the following block into the chat and fill in the bracketed lines.

```text
Context: [product or domain in one sentence]
Goal for this session: [one outcome, for example ten seed ideas or a preview command that works]
“Good” means: [two bullets]
Hard limits: [language, tone, privacy, or “do not touch cluster dispatch yet”]
Please: [one request]. Use Nemotron SDG defaults from the repo unless something blocks that.
```

## What Success Looks Like on Day One

A reasonable first success is a small preview run that writes plausible rows to `output_path`, plus a short list of seed ideas you believe are on-brand for your domain.
If you have that, you are already operating SDG: iterate small, then scale record counts.

The hands-on path is {doc}`getting-started`.
When you are ready to attach your own domain, follow {doc}`how-to/create-domain-dataset`.

## Where Domain-Specific Ideas Come From

Seed data can be short anchors that tell the generator which slice of the world each row should reflect.
A newcomer can build a first seed list the same way a product owner scopes a feature.

Runbooks, internal FAQs, and training decks can inspire situations when policy allows.
If you cannot paste source text, a neutral rewrite still carries domain truth, for example “partial refund after a split shipment” instead of a ticket ID.

Standards, regulator explainers, textbooks, and course outlines supply topics and jargon.
Your operator value is the twist that matches your product, not the generic paragraph anyone could find online.

## Ask the Agent to Propose, Then You Curate

Paste a product brief or policy summary and then ask for candidate seed lines.
The opening brief in the section above keeps this step short: one propose-and-curate round per session is usually enough before you run a preview again.

## Staying Grounded on Policy and Quality

Check licensing and confidentiality before you drop internal documents into an agent or into a seed file.
Keep evaluation benchmarks separate from training seeds so synthetic items do not leak into the set you use to claim quality.
Skim `src/nemotron/steps/sdg/SKILL.md` for the short list of pattern links on blending, versioning, and benchmarks when you move past experiments.

## How SKILL.md Fits Your Session

`src/nemotron/steps/sdg/SKILL.md` is written for assistants that route work into the right shipped YAML profile and guardrails.
You do not need to memorize it.

Skim the decision table once so you know which bundled config matches which need, then let the agent open that file when you change output format or scale record counts.
You can also say in the chat, “follow `src/nemotron/steps/sdg/SKILL.md` for SDG,” so guardrails land in the thread without a long lecture.

## Next Steps

- Run the tutorial: {doc}`getting-started`.
- Adapt seeds and YAML to your domain: {doc}`how-to/create-domain-dataset`.
- Look up flags and fields when the agent names them: {doc}`reference/cli-reference` and {doc}`reference/config-schema`.
