<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Task Template Fields

`task_templates.yaml` is a YAML list. Each item describes one reusable conversation
shape; `expand` binds its slots into concrete task instances, and later stages turn its
milestones into rendered turns and expected tool calls.

This page covers the fields a new pack normally fills. The complete normative contract,
including correction, dependent calls, edge signatures, and surface-generation guards,
is `src/nemotron/steps/byob/references/bfcl-oracle-pack.md`.

Examples use either the bundled English library pack or the neutral `get_record`
starter emitted by the scaffolder. Their names and business behavior illustrate the
contract; they are not framework defaults.

## Create A Template Skeleton

The manual pack scaffolder writes runnable `single_turn`, structured-error, and
`irrelevant` starter templates:

```bash
python -m nemotron.steps.byob.scripts.scaffold_oracle_pack \
  --domain my_domain \
  --target /srv/bfcl/packs/my_domain \
  --transport python \
  --language en \
  --version 0.1.0
```

Edit `task_templates.yaml` in the generated directory. Prefer copying a conversation
shape from a bundled pack over starting from an empty mapping:

- `tiny_oracle_pack` has compact `single_turn`, `confirmation`, `multi_tool`, and
  `irrelevant` examples.
- `banking_vn_oracle_pack` covers every supported turn policy in a larger localized
  domain.

In assisted authoring, `bfcl_author draft` may propose template plans from certified
evidence. Drafts are written outside the pack and are not authoritative. A human
reviews slot bindings, policy, language, and validation semantics in the candidate
supplement before assembly.

## Validate Templates

There is no standalone template-validation CLI because most checks require sibling
files: tools come from `tools.json`, slots bind fixtures, text uses manifest languages,
and assertion names resolve in `assertions.py`.

Run whole-pack preparation during each authoring iteration:

```bash
python -m nemotron.steps.byob.scripts.validate_oracle_pack \
  --config /srv/bfcl/packs/my_domain/validate.yaml \
  --output-dir /tmp/bfcl-my-domain-validation
```

This checks tool references and slot sources, then the representative-generation
contract expands, plans, renders, schema-validates, replays, and asserts one
deterministic instance from every template. In assisted authoring, assembly first
checks that supplement templates name only certified tools and compiled assertions;
fresh whole-pack validation still runs before review and publication.

## Core Fields

| Field | Required | Meaning |
| --- | --- | --- |
| `template_id` | Yes | Unique stable identifier within the pack. |
| `intent` | Recommended; required by intent-based analysis | Semantic capability this task exercises. It is carried into published metadata. |
| `category` | Optional | Publication-budget group. When omitted, expansion uses `template_id` as the category, giving that template its own budget. |
| `difficulty` | Optional; required when balancing targets it | Pack-defined difficulty label carried into reporting and optional balancing. |
| `turn_policy` | Yes | Claim about the required conversation shape. See [Turn policies](#turn-policies). |
| `mutates` | For state-changing tasks | Declares that the conversation is expected to change oracle state. |
| `call_order` | Optional | `strict`, `any`, or `prefix`; defaults to `strict`. |
| `required_tools` | Yes in practice | Tool names the gold trajectory calls. Use an empty list for no-call policies. |
| `tools_present` | Optional | Tool names shown to the candidate. Omit it to expose the complete `tools.json` catalog. Every required tool must be present. |
| `slots` | Yes | Named values bound during expansion. An empty mapping is valid for a task with no bound value. |
| `success_assertions` | Yes for Gold | Assertion names that define success. A no-call task still needs an assertion such as `assert_no_tool_called`. |
| `user_turn_templates` | Yes | First user message per language. Visible slot placeholders must appear in this text. |
| `assistant_milestones` | Yes | Ordered assistant text and tool-call steps. |
| `user_simulator_turns` | When the assistant expects another user turn | Deterministic user replies after clarification, confirmation, or correction. |
| `assistant_turn_templates` | When overriding pack-wide text | Per-template wording for text milestones. Normally declared once in `manifest.yaml`. |
| `paraphrase` | Optional | Template-level surface-generation guards and variant limit. |

## Slot Fields

Each entry under `slots` declares:

| Field | Required | Meaning |
| --- | --- | --- |
| `source` | Yes | Where expansion obtains values, such as `fixture:books.book_id`, `literal:[...]`, or an absent-id source. |
| `filter` | Optional | Predicate that narrows eligible fixture rows before selecting the field. |
| `visible_in_first_turn` | Yes | `true` means the opening user message must contain the value; `false` means it must be withheld and collected before a call uses it. |
| `label.<lang>` | When `{slot_name}` should read naturally | Human label used by `ask_for_slot`, for example `book id` instead of `book_id`. |

Do not omit `visible_in_first_turn`. Rendering needs to place every slot in exactly one
of two guard sets: values the first user turn must preserve, and values it must omit.

## Assistant Milestones

Every item under `assistant_milestones` has a `type`:

| Type | Additional fields | Result |
| --- | --- | --- |
| `tool_call` | `tool`; optional `args`, `id`, `call_group` | One expected function call. Required same-named tool parameters may be filled from task slots. |
| `ask_for_slot` | `slot` when more than one slot is hidden; optional `id` | Assistant text asking for missing information. |
| `ask_confirm` | Optional `id` | Assistant text requesting confirmation before the following authorized call batch. |
| `decline` | Optional `id` | Assistant text declining an unsupported request. |
| `final_answer` | Optional `id` | Terminal assistant text after calls finish. |

Consecutive tool calls with the same `call_group` form one parallel assistant call
turn. Milestone ids must be unique and are required when a simulator reply or dependent
argument would otherwise refer to an ambiguous repeated milestone.

### Argument Binding

A tool-call milestone may omit a required argument when a task slot has the same name:

```yaml
slots:
  book_id:
    source: "fixture:books.book_id"
    visible_in_first_turn: true
assistant_milestones:
  - {type: tool_call, tool: get_book_status}
```

`expected_trace` fills `book_id` from the slot because it is required by the tool
schema. Optional arguments are never injected this way; declare them explicitly under
`args`.

An argument that is exactly a slot placeholder preserves the slot's JSON type. A
placeholder embedded in a larger string produces a string.

## User Simulator Turns

A simulator entry releases a later user message only after its `after` milestone:

```yaml
user_simulator_turns:
  - after: ask_confirm
    content_template:
      en: "Yes, please confirm."
```

If the same milestone type occurs more than once, assign ids and reference the intended
id. A closing `ask_for_slot` in `clarify_only` has no simulator reply because the
conversation ends there.

## Turn Policies

A policy is not merely a label; Stage 5 validates the planned shape:

| Policy | Required shape |
| --- | --- |
| `single_turn` | Exactly one user turn and at least one tool call. |
| `multi_tool` | At least two tool calls. |
| `missing_slot` | Every hidden slot is asked for and receives a simulator reply before the first call. |
| `confirmation` | An `ask_confirm` and simulator reply authorize the following call batch. |
| `correction` | A later user turn supplies `slot_updates`. |
| `dependent_call` | Strict call order and an argument derived from an earlier call result. |
| `negative_path` | At least one assertion states the expected rejected behavior. |
| `clarify_only` | No tool call; conversation ends with `ask_for_slot`. |
| `irrelevant` | No tool call; conversation ends with `decline`. |

## Worked Shapes

These examples are reduced from the bundled packs. Fields not relevant to the shape
remain in the source files.

### Single Turn

From `tiny_oracle_pack/task_templates.yaml`:

```yaml
- template_id: lib_status_single
  intent: check_book_status
  category: circulation
  difficulty: easy
  turn_policy: single_turn
  required_tools: [get_book_status]
  slots:
    book_id:
      source: "fixture:books.book_id"
      filter: "status == 'available'"
      visible_in_first_turn: true
  success_assertions: [assert_book_status_reported]
  user_turn_templates:
    en: "Can you check whether book {book_id} is available?"
  assistant_milestones:
    - {type: tool_call, tool: get_book_status}
    - {type: final_answer}
```

`book_id` appears in the opening user turn, the call is made without another user
message, and the final answer uses the pack-wide assistant template.

### Confirmation

From `tiny_oracle_pack/task_templates.yaml`:

```yaml
- template_id: lib_checkout_confirm
  intent: checkout_book
  category: circulation
  difficulty: medium
  turn_policy: confirmation
  mutates: true
  required_tools: [checkout_book]
  slots:
    book_id:
      source: "fixture:books.book_id"
      visible_in_first_turn: true
    patron_id:
      source: "fixture:patrons.patron_id"
      visible_in_first_turn: true
  success_assertions:
    - assert_checkout_awaiting_then_committed
    - assert_book_now_on_loan
  user_turn_templates:
    en: "I want to borrow book {book_id} for patron {patron_id}."
  assistant_milestones:
    - {type: ask_confirm}
    - {type: tool_call, tool: checkout_book, args: {confirm: true}}
    - {type: final_answer}
  user_simulator_turns:
    - after: ask_confirm
      content_template:
        en: "Yes, please confirm."
```

Removing the simulator reply leaves the confirmation unanswered; moving the tool call
before `ask_confirm` produces an unauthorized confirmed mutation.

### Missing Slot

This English example extends the neutral `get_record` scaffold with a withheld id:

```yaml
- template_id: record_lookup_missing_id
  intent: look_up_record
  category: records
  difficulty: medium
  turn_policy: missing_slot
  required_tools: [get_record]
  slots:
    record_id:
      source: "fixture:records.record_id"
      visible_in_first_turn: false
      label: {en: "record id"}
  success_assertions: [assert_record_reported]
  user_turn_templates:
    en: "Show me a record."
  assistant_milestones:
    - {type: ask_for_slot, slot: record_id}
    - {type: tool_call, tool: get_record}
    - {type: final_answer}
  user_simulator_turns:
    - after: ask_for_slot
      content_template:
        en: "The record id is {record_id}."
```

The opening user turn omits `record_id`. Stage 5 requires the assistant to ask for it
and receive the simulator reply before the call. The required same-named tool parameter
is then bound from the slot. The localized banking example in
{doc}`../explanation/pipeline-worked-example` demonstrates the same policy through all
generation stages.

### Irrelevant Request

From `tiny_oracle_pack/task_templates.yaml`:

```yaml
- template_id: lib_irrelevant_renew
  intent: renew_card_online
  category: circulation
  difficulty: easy
  turn_policy: irrelevant
  required_tools: []
  slots: {}
  success_assertions: [assert_no_tool_called]
  user_turn_templates:
    en: "Can you renew my library card online for me?"
  assistant_milestones:
    - {type: decline}
```

An irrelevant task calls no tool and still has a success assertion. Adding a tool-call
milestone or ending with anything other than `decline` violates the policy shape.

## Common Failures

| Failure | Field to correct |
| --- | --- |
| `template_without_success_assertion` | Add a meaningful `success_assertions` entry. |
| `slot_missing_visibility_flag` | Add boolean `visible_in_first_turn` to every slot. |
| `unknown_turn_policy` | Use one supported policy from this page. |
| `ask_for_slot_without_a_named_slot` | Add `slot:` to the milestone when several slots are hidden. |
| `missing_assistant_turn_templates` | Add wording in the pack manifest or template for each text milestone used. |
| Required tool not exposed | Add the name to `tools_present`, or omit `tools_present` to expose the full catalog. |

See {doc}`troubleshooting` for the complete failure taxonomy.

## Related Information

- {doc}`oracle-pack-inputs` for the complete file map and manifest fields.
- {doc}`manifest` for languages and shared assistant text.
- {doc}`tools-and-fixtures` for tool schemas and slot-source records.
- {doc}`python-backend` for the tool implementation contract.
- {doc}`assertions` for `success_assertions`.
- {doc}`validation-cases` for direct oracle probes.
- {doc}`../explanation/pipeline-worked-example` for how one template becomes a row.
- {doc}`../how-to/author-a-pack` for validation and smoke-run commands.
