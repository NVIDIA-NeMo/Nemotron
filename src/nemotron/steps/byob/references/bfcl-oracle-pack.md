# BFCL Oracle Pack Contract

The `byob/bfcl` family generates function-calling benchmarks from an executable
oracle pack. All domain content lives in the pack; `runtime/benchmark_families/bfcl`
stays domain-agnostic.

## Pack Layout

A pack is a directory under an `oracle_runtime.allowed_roots` entry (default
`byob/data`). Paths come from `manifest.yaml.paths`, from `oracle_pack.*` config
overrides, or from these defaults:

Relative paths in the BFCL config resolve from the checked-in `steps/byob` root,
not from the YAML file's directory or the shell working directory. External packs
should use absolute `manifest_path` / `allowed_roots` values. Paths declared inside
`manifest.yaml.paths` remain relative to that pack's own directory.

| File | Required | Purpose |
| --- | --- | --- |
| `manifest.yaml` | yes | `pack_id`, `version`, optional `paths` overrides. |
| `tools.json` | yes | OpenAI-style tool array plus pack-local `x-*` keys. |
| `backend.py` | one oracle required | Local executable oracle. Mutually exclusive with `endpoint_config.yaml`. |
| `endpoint_config.yaml` | one oracle required | BFCL Oracle HTTP v1 endpoint, expected identity, TLS, and secret environment references. Mutually exclusive with `backend.py`. |
| `held_out.yaml` | optional | Versioned fixture primary IDs and template IDs excluded from normal generation; referenced by `manifest.yaml` as `held_out: held_out.yaml`. |
| `fixtures.json` | no | Seed state passed to `reset`. |
| `task_templates.yaml` | yes | Task templates with `slots` and `paraphrase`. |
| `validation_cases.yaml` | yes | Declared probes proving tool behavior. |
| `assertions.py` | yes | Success assertions referenced by templates. |

For operator-facing creation paths, field examples, validation commands, and common
failures, start with the
[Oracle Pack input map](../../../../../docs/build-benchmarks/function-calling/reference/oracle-pack-inputs.md).
Its standardized artifact references cover the
[manifest](../../../../../docs/build-benchmarks/function-calling/reference/manifest.md),
[tool catalog and fixtures](../../../../../docs/build-benchmarks/function-calling/reference/tools-and-fixtures.md),
[Python backend](../../../../../docs/build-benchmarks/function-calling/reference/python-backend.md),
[task templates](../../../../../docs/build-benchmarks/function-calling/reference/task-templates.md),
[assertions](../../../../../docs/build-benchmarks/function-calling/reference/assertions.md),
[validation cases](../../../../../docs/build-benchmarks/function-calling/reference/validation-cases.md),
[endpoint configuration](../../../../../docs/build-benchmarks/function-calling/reference/endpoint-config.md),
and [held-out policy](../../../../../docs/build-benchmarks/function-calling/reference/held-out-policy.md).

`backend.py` and `assertions.py` may import helper modules from the pack: the worker
puts the module's own directory and the pack root on `sys.path` before importing, and
the fingerprint already covers every file in the pack tree. Those two directories take
precedence during pack execution, so a helper must not be named after a module the pack
itself imports (`yaml.py`, `json.py`). Anything a pack imports from outside its own tree
is invisible to the fingerprint, so a pack that depends on such a module can be replayed
against a different version of it without the run noticing.

Covering the whole tree means documentation is covered too, and there is no read
sandbox that would make a `.md` file provably inert — a backend can open its own
README. So publishing a benchmark from a pack freezes that directory: any later
edit, including to a comment or a README, makes every eval of that benchmark fail
preflight with `eval_source_oracle_pack_drift` until the bytes are restored. Keep
notes about a pack outside it, and publish a new release rather than editing a
pack that is still scored. Generation records `pack.files`, a hash per hashed
file, beside `pack.content_hash`; the aggregate is one rolling digest and proves
only that something moved, while the map is what lets a drift report name the
file and say whether any declared oracle input was involved.

Every model-facing entry is validated as an OpenAI function tool: `type` is
`function`, `function` is a mapping, `description` is text when present, and
`strict` is boolean when present. `tools.json` keys read only by the pack pipeline
are `x-mutates` and `x-requires-confirmation`.
A tool marked `x-requires-confirmation` must expose the confirmation parameter and
must not mutate state when that parameter is `false`. `x-mutates` declares that a
tool changes state; validation compares the claim against what the probes actually
did and rejects a tool that changed state without declaring it.
### Manifest Keys

| Key | Required | Purpose |
| --- | --- | --- |
| `pack_id`, `version` | yes | Row provenance and cache identity. |
| `paths` | no | Override any default file location. |
| `primary_keys.<collection>` | when ambiguous | Field that identifies a fixture row. Convention covers `<singular>_id` and `id`; a collection carrying its own id plus a foreign key must declare which is which. |
| `absent_ids.<collection>` | when used | Ids that deliberately do not exist, for `absent:` slots. |
| `languages`, `default_language` | no | Surface languages the pack offers and the one to use when config names none. |
| `system_prompt` / `system_prompt_path` | no | Inline or file system prompt; the frozen default is used otherwise. |
| `assistant_turn_templates.<type>.<lang>` | per milestone type used | Pack-wide assistant text, overridable per template. |
| `confirmation.parameter` / `.status_field` / `.pending_status` | no | The pack's own confirmation vocabulary; defaults to `confirm`, `status`, `awaiting_confirmation`. |
| `surface_guards.tool_names_exempt` | no | Tool names that are ordinary words of the domain language and must not count as tool-name leakage. |

## Backend Contract

`backend.py` exposes four callables; the pipeline owns reset, clock, seed, and
timeouts, so the backend must not read wall-clock time or the environment:

```python
def list_tools() -> list[str]: ...
def reset(*, ctx, fixtures=None) -> None: ...
def call_tool(name: str, arguments: dict, *, ctx) -> dict: ...
def get_state() -> dict: ...
```

`ctx` is a `RunContext` carrying `clock`, `seed`, `timeout_s`, `task_id`, and
`turn_index`. Every `call_tool` return value must be JSON-serializable. Errors are
returned as data, never raised:

```python
{"error": {"code": "not_found", "entity": "books", "id": "BK-9", "field": "book_id", "message": "..."}}
```

Only `code` is required — it is what scoring compares. `entity`, `id`, `field`, and
`message` are optional detail for a reviewer, so a domain whose failures are not
about one entity is not forced to pad the envelope with nulls.

### HTTPS Endpoint Contract

An endpoint-backed pack exposes the same oracle behavior through BFCL Oracle
HTTP v1. `endpoint_config.yaml` declares:

- `protocol_version: bfcl-oracle-http-v1`
- an HTTPS `base_url`
- expected `oracle_id`, `oracle_version`, and `sha256:` `content_digest`
- optional bearer-token and custom-header environment-variable names
- an optional pinned `bfcl-endpoint-conformance-v1` attestation, required for Gold
- an optional allowlisted CA bundle plus request- and response-size limits

The fixed routes are `GET /v1/metadata`, `GET /v1/tools`, `GET /v1/conformance`,
`POST /v1/sessions`, `POST /v1/sessions/{id}/calls`,
`GET /v1/sessions/{id}/state`, and `DELETE /v1/sessions/{id}`.
Creating a session is the reset operation: the request carries the frozen
`RunContext` fields and fixtures, and the response returns a unique session id
plus the endpoint identity. Every replay gets a new session.

`GET /v1/metadata` returns the identity object, and `GET /v1/tools` returns
`{"tools": ["tool_name", ...]}`. Session creation uses:

```json
{
  "context": {
    "clock": "2026-03-02T09:00:00+07:00",
    "seed": 7,
    "timeout_s": 5.0,
    "task_id": "pack__template__0001"
  },
  "fixtures": {}
}
```

and returns:

```json
{
  "session_id": "opaque-session-id",
  "oracle": {
    "protocol_version": "bfcl-oracle-http-v1",
    "oracle_id": "example",
    "oracle_version": "1.0.0",
    "content_digest": "sha256:..."
  }
}
```

A call request is
`{"name": "tool_name", "arguments": {}, "turn_index": 0}` and its response is
the tool result itself. The state route returns a JSON object. Session ids are
opaque and URL-escaped by the client. A successful session DELETE may return
either an empty `204 No Content` response or a JSON object.

Remote identity must match the config during prepare, session creation,
generation, and publication. A changed version or digest aborts the run.
Requests are not redirected and mutating requests are never retried. Secret
values are resolved from the environment at runtime and are not included in
fingerprints, reports, logs, or manifests.

`assertions.py` functions must accept exactly the keyword arguments
`(*, state, trace, task, ctx)`. Return `None` on success, raise `AssertionError`
on failure, or return `{"status": "not_applicable", "detail": "..."}` when the
declared predicate does not apply. Any other return is an infrastructure error.
Export assertions through an `ASSERTIONS` dict or name them `assert_*`.
Optional literal `ASSERTION_CAPABILITIES` entries declare boolean `trace` and
`executable` support plus category `state`, `path`, `result`, `final_answer`, or
`unclassified`; omitted entries default to executable-only and unclassified.
Validation and executable projection both read the mapping as a literal
assignment rather than importing it, so a computed mapping, an unknown assertion
name, or a malformed entry fails validation with
`invalid_assertion_capability`. That reason is separate from `invalid_signature`,
which is only about the assertion's keyword arguments.
Assertions run inside the
same process worker as the local backend or endpoint client. For endpoints,
`state` is fetched from the active remote session; `trace` holds the calls the
episode just made: `[{"tool": ..., "arguments": {...}, "result": {...}}]`.

## Template Surface Requirements

Generation renders every assistant and user turn from the pack, never from a model:

- `user_turn_templates.<lang>` supplies the first user turn; `{slot}` placeholders
  are substituted with bound values.
- Placeholders follow one rule everywhere, surface text and call arguments alike:
  `{name}` is replaced in a single pass, so a bound value that itself contains braces
  is inserted as data rather than rescanned, and a name nothing bound fails the run.
  A milestone argument that is exactly `{name}` keeps the slot's own type; a
  placeholder inside a longer argument string is substituted in place, so a call and
  the turn describing it always show the same value. Required tool parameters may be
  filled from same-named slots as a shorthand. Optional parameters are never injected:
  declare them under milestone `args` when the gold call should send them.
- `user_simulator_turns[].content_template.<lang>` supplies later user turns. Each
  entry's `after` must resolve to exactly one milestone — use a milestone `id`
  whenever the same milestone type repeats. An `ask_confirm` needs such an entry;
  a closing `ask_for_slot` does not, because the conversation ends there.
- Every template declares one supported `turn_policy`, and every declared milestone
  `id` is a unique non-empty string. Missing policies and duplicate ids are rejected
  before references can silently bind to the wrong conversation edge.
- A template may declare `edge_signatures` as a unique list of non-empty,
  pack-defined labels for rare behavior not already distinguished by
  `turn_policy`. Expansion carries these labels into Stage 11, where
  deduplication and balancing preserve at least one selected row for every
  complete language/policy/edge bucket.
- A policy is a claim about the conversation's shape, because it is what consumers
  slice the published rows by, so each one is held to that shape:

| `turn_policy` | the shape it must plan |
| --- | --- |
| `single_turn` | exactly one user turn, and at least one tool call |
| `multi_tool` | at least two tool calls |
| `missing_slot` | every slot marked `visible_in_first_turn: false` is named by an `ask_for_slot` before the first call and receives a simulator reply |
| `confirmation` | an `ask_confirm` answered by a `user_simulator_turns` entry before the call batch it authorizes |
| `correction` | a user turn carrying `slot_updates` |
| `dependent_call` | `call_order: strict`, and a call that reads an earlier call's result |
| `negative_path` | at least one `success_assertions` entry, which is what states the refusal the call must produce |
| `clarify_only` | no tool call, ending in `ask_for_slot` |
| `irrelevant` | no tool call, ending in `decline` |
- `assistant_turn_templates.<milestone_type>.<lang>` supplies text for
  `ask_for_slot`, `ask_confirm`, `decline`, and `final_answer`. Declare it on the
  manifest (pack-wide) or override it per template. A milestone with no template
  fails the run rather than inventing text.
- The render language is chosen from the languages `user_turn_templates` offers —
  that is the one block every task renders — narrowed by config `language` or
  `manifest.default_language` / `languages` when the pack offers several. Every
  other block the plan renders must state that same language: `render` lists all
  the gaps at once before any task is rendered, so a pack that declares assistant
  text in one language and simulator turns in another hears about both.
- `{slot_name}` in assistant text resolves to the slot that milestone asks about:
  the milestone's own `slot:` when declared, otherwise the template's single withheld
  slot. A template that withholds two slots and asks about one must name it, because
  a question that lists both is not the question the conversation needs. The slot may
  carry `label.<lang>` so the question reads as prose rather than quoting the slot key.
- A visible slot must appear verbatim in the rendered user turn. `must_preserve`,
  `must_omit`, and `must_not_mention` are re-checked in Python after rendering and
  a violating row is dropped, so a template whose prose paraphrases a slot value
  (`rail: napas` rendered as "qua Napas") is rejected — spell the value out or
  make the slot hidden.
- `must_not_mention` entries are forbidden phrases, matched case-insensitively over
  every user turn. The reserved entry `tool_names` means "no tool name may appear";
  it is also checked whenever `surface_generation.prevent_tool_name_leakage` is on,
  so declaring a phrase adds a rule instead of replacing the leakage guard. A tool
  name is matched as a whole word, and a name that is an ordinary word of the domain
  language can be exempted through `manifest.surface_guards.tool_names_exempt`.
- Guard matching reads a value as a whole token, so a value buried in a longer one was
  never stated: an amount of `4` is not stated by `400000`, and a rail of `us` is not
  stated by `status`. The class excluded follows the value's own edges, so a number may
  carry a unit written against it (`200000` is stated by `200000đ`) while a word must
  not sit inside a longer word.
- Every slot must declare `visible_in_first_turn`. The flag is what puts the slot in
  the preserve set or the omit set, so omitting it would leave it in neither. A
  visible slot's initial value must occur in the **first** user turn; values introduced
  by corrections are preserved across the full conversation.
- `must_omit` covers the **first** turn only. That is what makes a withheld slot
  workable: the value stays out of the opening request and arrives in the
  `user_simulator_turns` reply that answers the assistant's `ask_for_slot`.
- A call may carry `confirm: true` only when its own assistant turn is covered by a
  confirmation: the user answered an earlier `ask_confirm`, no later turn corrected
  a value, and no earlier call batch already consumed that confirmation. One reply
  authorizes the next call batch only; a separate later action needs another
  `ask_confirm`. The check reads positions in the plan, not the `turn_policy` label.
- The system prompt comes from `manifest.system_prompt`, `manifest.system_prompt_path`,
  or the frozen `bfcl/prompts/default_system.txt`. The row stores its
  `system_prompt_id` and `messages[0]` is the resolved text. A declared prompt
  path must remain under `oracle_runtime.allowed_roots` and participates in the
  pack fingerprint. The frozen prompt is English, so a non-English render must declare
  its own prompt rather than publish a mixed-language gold row; under
  `lineage.policy: smoke_no_publication` the mismatch is only a warning, because that
  run publishes nothing.

## Validation Cases

Each entry probes one tool call. Keys:

- `id`: unique probe id.
- `tool`, `arguments`: the call to run.
- `reset_before`: `true` starts a fresh episode; `false` chains onto the previous
  case inside the same worker process, which lets a probe observe state produced
  by the case before it. The first case in the file cannot use `false`.
- `expect.result_class`: `success`, `structured_error`, or `awaiting_confirmation`.
- `expect.error_code`: expected `error.code`, or `null` for success.
- `expect.state_unchanged`: assert `get_state()` is identical before and after.
- Any other `expect.<field>`: compared against `result[<field>]`, which is how
  business outcomes such as `status: rejected_insufficient_funds` are pinned.
- `coverage: negative`: mark a probe as the negative case when the tool signals a
  miss through a normal result instead of `error` or `awaiting_confirmation`
  (for example a collection query returning an empty list).
- `notes`: free text for reviewers.

Every tool in `tools.json` needs at least one success probe and one negative
probe. A negative probe is any case whose result is a `structured_error` or
`awaiting_confirmation`, or one explicitly marked `coverage: negative`. A negative
probe may deliberately violate the schema to test backend error handling. A success
probe must satisfy the model-facing parameter schema; backend-only arguments cannot
prove success coverage.

## Validation Report And Tiers

`stage=prepare` normalizes the pack into `output_dir/expt_name/stage_cache/` and
writes `oracle_validation_report.json` containing `tier`, `gold_eligible`,
`pack_fingerprint`, per-check failures, and pack stats. Checks:

1. `template_tool_names` — templates only reference declared tools.
2. `template_slot_sources` — every slot source resolves and its filter matches rows.
3. `backend_schema_alignment` — `list_tools()` and `tools.json` agree, and every
   parameter schema is one this pipeline enforces. A constraint that no argument can
   satisfy is refused rather than certified: an unsupported keyword, a malformed or
   inconsistent bound (`minimum` above `maximum`), an empty `enum`, a repeated `enum`
   value, `required` name or type-union member, and an `enum` or `const` value the
   declared `type` rejects. Local `$ref` through `$defs` or `definitions` and
   `allOf` conjunctions are supported and enforced recursively; external, missing,
   and cyclic references are refused. Because defaults participate in call
   comparison, every declared `default` must itself satisfy the schema under which
   it would be inserted.
4. `assertions_importable` — every template declares at least one `success_assertions`
   entry, and each referenced assertion exists with a valid signature. A template
   naming none has no statement of success, so replay could only confirm that its
   trace ran, not that it was right.
5. `declared_validation_cases` — every probe matches its `expect`, with success and
   negative coverage per tool.
6. `confirmation_policy` — `confirm: false` yields `awaiting_confirmation` and
   leaves state untouched.
7. `representative_generation_contract` — the run-wide inputs hold and every template
   can produce a publishable row. First the category budget must keep at least one
   instance per template and the render contract must resolve (language, every text
   block in that language, system prompt, `tool_names_exempt`). Then the first
   deterministic instance of every template expands, binds an expected trace
   (including `from_result` dependencies), passes its tool schemas, renders without
   breaking a surface guard, replays twice deterministically, and passes its declared
   assertions. Gold means the pack generates, so a typoed argument, an unbound
   placeholder, a missing text block, a starved budget, a broken assertion, or a
   paraphrase that always drops its own rows is reported here instead of aborting a
   run or silently publishing fewer rows than the pack declares. Only the budget is
   settled over the templates generation may bind; a reserved template is still
   compiled, because a private held-out run compiles it from this same pack. When
   reserved fixtures remain in backend state, that private representative may use
   them. Under `fixtures_in_backend_state: false`, only template blocking is disabled
   and fixture reservations stay active, so replay uses exactly the state the oracle
   may observe.

Extra checks: `M1` mutation declaration (observed changes require `x-mutates`, and a
tool declaring it must change state in at least one successful probe), `D1`
determinism (repeat one observed success probe per tool with an identical
`RunContext`, comparing both result and final state), `D2` error shape (every
observed structured error carries a `code`),
`T1` timeout enforcement (a tool that never returns is killed on the same path pack
tools run through), `I1` isolation (`oracle_runtime.worker: process`). D1 and D2
replay the full `reset_before: false` prefix of a chained case rather than executing
that case out of context.

A check whose preconditions failed is recorded as `skipped`, never as `pass`:
check 5 and `M1` skip when check 3 failed, because probing a backend that disagrees
with `tools.json` would only report noise.

`gold` requires every check to pass plus a backend, templates, and assertions.
A `skipped` check is not a pass, so it keeps the pack below gold.
`worker: thread` is a debug aid that runs pack code in-process and can never
reach gold: the pack sees the caller's environment, and a tool that hangs cannot
always be stopped. Only `worker: process` sanitizes the environment and enforces
hard deadlines, which is why a gold claim requires it. `stage=generate` refuses a
pack that is not `gold_eligible`, and it derives that verdict from the individual
checks rather than the summary flag. The report on disk is a human-readable artifact,
not a signed attestation, so `stage=generate` never trusts one written by an earlier
run: it reuses a verdict only when the same process produced it for the same pack and
config fingerprints, which is why `stage=all` validates once rather than twice.
The fingerprint is checked before and after validation and again before final output.
Pack code and concurrent writers must not modify files in the pack while a run is
active; drift aborts publication rather than stamping a report and benchmark from
different content.
`expt_name` is one directory name, not a path: every artifact hash and cache path is
stated relative to the run directory, so a separator or a `.` / `..` segment would move
the run somewhere the config does not name. `output_dir/expt_name` must also remain
outside the oracle pack root: otherwise generated reports and parquets would become pack
inputs and change the fingerprint they claim to describe.

## Generation Stages And Output

`stage=generate` consumes a gold-eligible report and runs, in order: `expand`
(bind slots into locked task instances), `state_machine` (order milestones into
turns and batch `call_group`s), `render` (verbatim surface plus guard checks),
`expected_trace` (derive `expected_tool_calls`), `schema_validation` (check calls
against the tool schemas), `executable_replay` (reset and replay each task twice,
then run its assertions), and `final_output` (assemble `messages`, write the
parquets and `run_manifest.json`).

Reusing an `expt_name` starts a new invocation: completed parquets and the old manifest
are invalidated before validation, parquet files are written through temporary files,
and the manifest is atomically installed last. A failed rerun therefore leaves no
completed manifest that could make stale output look current.

Each stage leaves one artifact under `stage_cache/`, keyed by `task_id` with one
row per task, so a run can be read stage by stage instead of guessed at:

| Stage | Artifact | Holds |
| --- | --- | --- |
| `expand` | `task_instances.parquet` | bound slots, the slot timeline, seed, required tools, assertions |
| `state_machine` | `conversation_plans.parquet` | ordered steps, turn counts, confirmation and correction flags |
| `render` | `rendered_conversations.parquet` | rendered turns, language, guard verdict and violations |
| `expected_trace` | `expected_traces.parquet` | derived `expected_tool_calls` per task |
| `schema_validation` | `schema_validated_traces.parquet` | per-task verdict, reject reason, failure detail |
| `executable_replay` | `replay_validated_tasks.parquet` | verdict, tool results, assertion outcomes |

Every table carries the same `task_id` set, so joining them shows exactly which
stage dropped a task. A task that fails to bind at `expected_trace` keeps a row
there with `derived: false` and appears in the later tables as a skip
(`reject_reason` / `reason: trace_not_derived`) rather than vanishing. Nested
content (slot bindings, plan steps, rendered turns, derived calls, failure lists)
is canonical JSON text, for the reason given under the two benchmark columns below.
`run_manifest.json` records a content hash per artifact — including the
validation report and prepare-normalized files — so a published row is
traceable to the intermediates behind it.

`task_generation.tasks_per_category` budgets a whole **category**, not one
template. Each template in the category first contributes one instance, then a
second, and so on until the budget is spent, so adding a template spreads the
budget instead of growing the set. A budget smaller than the number of templates
in a category is rejected rather than silently dropping the last templates. Slot
candidates are enumerated in fixture order and each template's product is capped
as it is built, so the same pack and config always produce the same instances —
but the cap keeps the earliest candidates, so raising it varies the last slot
first. Use narrower filters when you need spread across a specific slot.

Set `task_generation.candidate_tasks_per_category` above the publication budget
when Stage 11 needs inventory from which to satisfy difficulty or turn-mix
quotas. Stage 4 expands to that candidate ceiling; Stage 11 still enforces
`tasks_per_category`. The candidate ceiling must never be smaller than the
publication ceiling.
Declare `task_generation.target_published_tasks` when a release also requires
an exact run-wide row count. A diversity or inventory shortfall then becomes an
explicit unmet target governed by `unmet_target_policy`, and the report names the
bound that produced it — candidate inventory, category cap, exact-surface
diversity, the declared mix, or coverage — so the shortfall is actionable.

Two optional `semantic_deduplication_config` limits constrain repeated wording
independently of embedding similarity: `max_exact_surface_reuse` caps how many
published rows may share one masked surface, and `min_exact_surface_ratio`
requires a minimum share of distinct masked surfaces. Both participate in
selection rather than filtering afterwards. Because a masked surface is what they
count, they bound publication by wording inventory: a category needs at least
`tasks_per_category / max_exact_surface_reuse` masked surfaces of its own, which
is a property of how many templates it declares and how many of them opt into
paraphrasing, not of how many fixture bindings exist.

Generation rejects unsupported task-generation keys instead of recording
targets that no stage applies. Category budgets are consumed by expansion and
publication; mix targets and hard limits are consumed by Stage 11. The same rule
covers the rest of the config: a request no stage can honor — a paraphrase count,
a judge's drop authority, an unrecognized `surface_generation` key — stops the
run instead of being dropped in silence. Config values are type-strict as well: quoted strings do
not stand in for booleans or integers, and an empty list does not stand in for an
empty mapping. Timeout values must be positive finite numbers; YAML `NaN` and
infinities are rejected before they can reach deadline arithmetic.

### Surface Quality Validation

Optional Stage 10 runs between replay and final output under versioned contract
`1.1`. It filters publication rows and records its report and artifact hashes in
`run_manifest.json`. The contract is six checks with fixed ownership: Python owns
`surface_shape`, `semantic_preservation`, and `leakage`, and the optional surface
judge owns `language_locale`, `fluency_naturalness`, and `clarity_coherence`.

The judge contract is surface-only. A judge cannot label tool correctness, change
arguments, inspect oracle results, or rewrite benchmark truth, and its responses
carry only a controlled reason code rather than free text. It sees only the
language, the user-facing turns, the style hints, and the surface rubric. The
deterministic stage may run without a judge; `drop_authority: true` requires one.

A complete quality result carries exactly one verdict per check. Python checks are
`passed` or `failed`. A judged check may additionally be `not_applicable` when the
task's turn policy intentionally permits the observed condition, `not_run` when the
judge is skipped, or `error` when the judge call fails. None of those three is a
quality failure and none is counted as a pass. Turn-policy applicability is applied
when the six results are assembled, so intentional ambiguity in a `clarify_only`
task is recorded as `not_applicable` rather than as a fluency failure.

Python owns its three checks by mapping the render and paraphrase guards already
defined above — `must_preserve`, `must_omit`, `must_not_mention`, `novel_literal`,
`expected_result_leakage`, and `semantic_shape` — onto the contract. A canonical
template surface never fails `unchanged_surface`. Under `semantic_shape` a model
variant is dropped when it returns the canonical wording
(`unchanged_surface`) or repeats another variant of the same binding
(`duplicate_variant_surface`), because neither adds a surface. Expected-result and
novel-literal values remain private guard diagnostics and are never copied into
quality-record evidence.

Drop authority is deliberately asymmetric. A Python failure always drops the row,
because those three checks are what protect semantics and leakage. A judge failure
drops the row only under `drop_authority: true`, and is otherwise recorded as an
advisory observation that changes nothing. A judge *error* never decides anything:
an advisory run records it and continues, while an authoritative run refuses to
publish, since a gate that could not answer was never enforced. If the policy drops
every replay survivor, final output refuses to stamp an empty benchmark as gold.

Stage 10 writes `stage_cache/surface_validated_tasks.parquet` with one row per
task: identity, contract version, keep/drop authority, the six queryable statuses,
and canonical JSON check detail. Nested detail stays JSON text so that
pack-specific evidence cannot mutate the Arrow schema.

The judge input/output cache is shared and append-only, so the manifest does not
hash that changing file as though it belonged to one run. Instead
`surface_judge_cache_usage.json` records only the request, input, and observed
response hashes this run used — including an empty request list when Python
rejected every surface first — and the manifest hashes that per-run usage file.

### Semantic Deduplication And Balancing

Optional Stage 11 runs under versioned contract `1.0`. It requires exactly one
decision for every Stage-10 survivor, preserves input order, and restricts its
balancing reports to eight generic dimensions: `intent`, `category`,
`required_tools`, `tools_present`, `difficulty`, `turn_class`, `tool_call_count`,
and `turn_policy`. Controlled drop reasons distinguish semantic duplicates,
balance quotas, and hard turn or call limits. A selected row cannot carry drop
detail, and every selected row carries `selection_rank` `0..k-1` exactly once, so
publication order is total.

Deduplication may only collapse tasks that share one coverage bucket, so every
cluster holds exactly one representative and never mixes `language`,
`turn_policy`, or pack-defined `edge_signatures`. A representative may itself be
dropped on quota only when every member of its cluster is dropped. Validation
preserves at least one selected row for every complete input coverage bucket, not
merely each language, policy, and edge value independently. Bucket keys and task
identifiers are normalized, because bucket identity is string equality and an
untrimmed variant would otherwise become a phantom bucket.

Stage 11 embeds a text projection rather than the published row. The projection
keeps only user-authored turns, in conversation order, each prefixed with a
`[user]` marker; assistant milestones, tool-call payloads, and oracle results
never reach it. Turn whitespace is collapsed, and every literal the pack bound
into the task, corrections included, is masked to `<slot_name>`, so two tasks that
differ only in a bound id project the same text. Masking matches whole tokens and
longest literals first, so a short value cannot corrupt a word that merely
contains it, and a literal two slots share is masked under one deterministic slot
name. Correction aliases are masked as well as canonical slot keys.

Embedding runs through the shared Curator semantic-deduplication workflow, with
`task_id` as the id and the projected text as the embedded field. `n_clusters` is
capped so that non-trivial inputs retain an average of at least two rows per
partition, because setting k equal to the row count would turn pairwise
deduplication into a collection of singletons; the effective value is reported. A
set of fewer than two rows never reaches the embedding backend: one surface cannot
duplicate anything and zero surfaces have nothing to embed. `eps` is constrained
to `(0, 1)`, matching Curator's cosine-similarity threshold and keeping singleton
sentinels below it.

Both duplicate flags and cluster membership come from one Curator run. Stage 11
consumes Curator's K-means-partitioned pairwise artifact: each row crossing
`1 - eps` is linked to the predecessor Curator ranked for it, and those links form
the duplicate clusters. It also requires that the official Curator duplicate
artifact names exactly the same ids. This avoids a second similarity
implementation, comparisons across Curator's candidate partitions, and an extra
global quadratic pass. The run records the settings hash, the projected-input
hash, and a signature over the embeddings themselves, so a deduplication decision
can be traced to the exact model, settings, and vectors that produced it. A
backend result is rejected before it can decide anything if it fails to cover the
embedded ids, if Curator's pairwise and duplicate artifacts disagree, or if a
cluster does not carry exactly one non-duplicate representative. Embeddings decide
duplication only; nothing a model produces reaches a task's text, calls,
arguments, or assertions.

Curator clusters are then partitioned by the complete
`(language, turn_policy, edge_signatures)` coverage bucket and by a hash of the
task's generic executable capabilities: `required_tools`, `tools_present`, the
assertion contract, the mutation flag, and the call-order policy. A Curator
representative in one partition therefore cannot erase the final survivor of
another. Each resulting multi-row partition selects exactly one representative
using, in order: eligibility under the run's `max_turns` and `max_tool_calls`
limits, no judge error or advisory failure, applicable-check status, coverage
rarity, the configured surface-source preference, a seeded hash, and `task_id`.
The default source preference is `template` before `model`; a run may reverse it
explicitly. Metadata retains the original Curator cluster, duplicate flag,
predecessor, similarity score, final cluster, capability signature, coverage
bucket, and every deterministic rank component.

Balancing then projects every candidate onto the eight locked dimensions. Here
“candidate” means a replay-valid Stage-10 survivor, never a model response.
`max_turns` and `max_tool_calls` are hard filters, and removing the final survivor
of a coverage bucket aborts rather than weakening the lock. One representative per
complete coverage bucket is selected before quotas. Category caps and the
configured `difficulty_mix`, `turn_mix`, `tool_call_count_mix`, and `policy_mix`
targets are then applied without cloning rows, with fractional targets allocated
by deterministic largest remainder.

Selection is a deterministic binary optimization. Coverage, representative
lineage, and the repetition caps below are hard constraints, while category-cap
overflow, then cross-dimension target deviation, then stable rank order are
minimized in that order. The priority is exact rather than weighted: each
objective is minimized, pinned at its optimum, and the next is minimized against
it, so no solver tolerance can trade a real category overflow for a cheaper mix,
or a real deviation for a cheaper row order. This avoids both greedy local optima
and a cubic exchange pass, so a feasible mix is not reported unmet merely because
of the order rows were picked in. Minimal environments use an exact bounded
fallback; production BYOB environments use PuLP with CBC. A target that inventory
or a locked coverage survivor genuinely prevents is returned with explicit
inventory, target, actual, and reason metadata rather than silently claimed as
met.

Three optional caps — `max_exact_surface_reuse`, `max_execution_case_reuse`, and
`max_rows_per_intent` — are one mechanism applied to three different projections
of a row: its masked wording, its executable case, and its intent. Each is a hard
cap on how many published rows may share that value, each shrinks the feasible
publication budget by `sum(min(group_size, cap))`, and each carries its own
shortfall reason so a report says which kind of repetition ran out.
`max_execution_case_reuse: 1` is the strongest statement a release can make: no
two published rows call the same tools with the same arguments against the same
state, so a candidate cannot earn credit twice for one behavior.
`max_rows_per_intent` is what stops one broad intent — typically a refusal or
out-of-scope intent with cheap inventory — from owning a disproportionate share of
the benchmark.

Stage 11 therefore reports three separate signals: exact masked-surface diversity,
embedding-based surface similarity, and executable-case diversity. A model
paraphrase intentionally keeps the executable-case hash of its canonical task,
while different fixture bindings, call policies, assertions, or distractor sets do
not. Keeping the three apart is what stops repeated wording and repeated
executable meaning from being read as the same failure mode.

Model surface calls are batched, and a failed call is an infrastructure event: it
stays out of the immutable cache so that a repaired endpoint can be retried, and
the rejection report records what produced nothing. A failed batch is retried one
request at a time, so a single refused request costs its own variant rather than
the variants of everything batched with it.

Stage 11 writes `stage_cache/balanced_tasks.parquet` with one row for every
Stage-10 survivor, carrying Curator lineage, final cluster and representative ids,
the duplicate and selection verdict, drop detail, the total publication rank, the
locked coverage bucket, and all eight balancing dimensions. It also writes
`stage_cache/dedup_balancing_report.json` with pre- and post-counts, grouped
selection and drop statistics, target and actual mixes, unmet targets, hard-limit
drops, rare-edge preservation, and the semantic, config, and artifact hashes
needed to audit the decision. Both files are replaced atomically.

`remove_duplicates` controls whether duplicate rows must be dropped or may be
retained as annotations. Under `false`, semantic similarity alone never drops a
row; the hard limits and balancing quotas still apply and keep their own drop
reasons.

Stage 11 is fail-closed. An embedding or Curator error propagates, and no final
parquet or manifest is written. Missing Stage-11 artifacts stop final output, and
an empty selected set cannot become an empty gold benchmark. Infeasible soft
targets are always recorded without inventing rows. `unmet_target_policy: abort`
leaves the diagnostic Stage-11 artifacts but stops before publication;
`publish_non_gold` permits publication only after setting both the manifest and
every row's `gold_eligible` to false and recording `stage_eleven_unmet_targets` as
the reason. Enabling Stage 11 requires Stage 10 and every Stage-11 model and
clustering setting to be present, so deduplication never admits an unvalidated or
under-specified run. The manifest embeds the Stage-11 report, model identifier,
settings hash, embedding signature, stage counts, and hashes of both Stage-11
artifacts.

### Held-Out Enforcement

A pack that references `held_out.yaml` from its manifest is enforced under
versioned contract `1.0` at two points.

Stage 4 never binds a reserved template or fixture row. The reservation is applied
while slot candidates are collected, so a reserved row cannot enter a task at all.
A slot whose every matching row is reserved, a category that falls short of
`tasks_per_category` once reservations are honored, and a policy that reserves
every template each stop generation with an explicit error rather than quietly
publishing a smaller or differently mixed set. Stage 4 records what it examined
and withheld in `stage_cache/held_out_bindings.json`.

Stage 12 re-scans every executable row against the same policy before anything is
written, comparing the canonical JSON `[collection, primary_id]` references
expansion recorded and the template each row came from. The scan writes
`stage_cache/held_out_scan.json` and stamps `held_out_hit` on published rows, so
that column reports a checked result rather than the null a policy-free run
publishes.

Enforcement is fail-closed and abort-only. A single hit stops the run before
`benchmark_raw.parquet`, `benchmark.parquet`, or the manifest exists, because
Stage 11 has already fixed the publication set and silently dropping a row would
break the balance the manifest reports. Missing or mismatched Stage-4 evidence
stops publication for the same reason. The manifest carries the policy lineage,
the scan counters, the Stage-4 counters, and hashes of both artifacts, and marks
bias-audit dimension `B7` `na` when a declared policy reserves nothing.

### Slot Sources And Filters

A slot's `source` is `<kind>:<rest>`, and a bare value means `fixture`:

| Source | Binds |
| --- | --- |
| `fixture:<collection>.<field>` | `field` of every row of `collection` the `filter` matches. |
| `enum:<tool>.<parameter>` | the parameter's declared `enum` values. |
| `literal:[a, b]` / `literal:200000` | the listed values, or one value keeping its Python type. |
| `range:{min: 1, max: 5, step: 1}` | the inclusive integer range; step is non-zero and points from min toward max, including descending ranges. |
| `absent:<collection>` | the manifest's `absent_ids` for that collection. |

`filter` is a conjunction of comparisons over one fixture row, parsed as a Python
expression: a field name on the left, a literal on the right, `and` between clauses,
and the operators `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not in`. Because it is
parsed rather than split on operator text, a literal may contain anything —
`title == 'Pride and Prejudice'` compares the whole title. A field the row does not
carry makes its comparison false; `or`, chained comparisons, and a non-literal
right-hand side are rejected.

Calls that share a `call_group` must be declared consecutively; a template that
interleaves another milestone between them is rejected, because the group can then
no longer map to a single assistant completion. A call that declares no `call_group`
is a batch of its own and can never be absorbed into a declared group. The published
`call_group` numbers batches in the order their assistant turns issue them, starting
at 0, so declaring 0 and 5 publishes 0 and 1 while keeping the declared order.

A `dependent_call` argument names a value that only exists after an earlier call
has run, so the template points at that call's result instead of a slot:

```yaml
assistant_milestones:
  - {id: recent_list, type: tool_call, tool: list_recent_transactions, call_group: 0}
  - type: tool_call
    tool: get_transaction_status
    call_group: 1
    args:
      transaction_id: {from_result: {call: recent_list, path: "transactions.0.transaction_id"}}
  - {type: final_answer}
```

`expected_trace` opens one worker episode, executes the calls incrementally, reads
`path` out of each producing result (dotted fields, integer list indices), and locks
the extracted value into the next call; it does not restart and replay the whole
prefix for every dependency. `executable_replay` then re-derives the trace from fresh
resets, so a wrong path shows up as a failing task rather than a plausible row. Rules
the stage enforces:

- `call` must be a declared milestone `id` issued earlier, in a **strictly lower**
  `call_group` — a dependency cannot share a group with its producer.
- `path` must resolve to a scalar, and the producing call must not have returned
  an `error`. A path that misses for one instance drops that instance, with the
  reason recorded in `expected_traces.parquet`; a path that cannot work for any
  instance — indexing a list with a name, for example — fails the run. If more
  instances drop than you expect, narrow the template's slot filter.
- A `from_result` argument shadows a same-named slot, and the two directions must
  agree — a marker requires `turn_policy: dependent_call`, and that policy
  requires at least one marker, so the edge can never quietly degrade into
  slot-bound arguments.
- Pair the template with an assertion that the consumed id appeared in the earlier
  result; that is what keeps the anti-hallucinated-id meaning of this edge.

During executable evaluation, the published locked value is not replayed into
the expected downstream call. The runner projects the marker coordinates,
extracts the value again from the candidate episode's paired live producer
result, and compares the candidate's next call against that live-derived
expectation. The runner never edits candidate arguments. If the live result
cannot satisfy the declared producer, path, scalar type, and consumer schema,
the episode ends with dependency infrastructure evidence rather than falling
back to generation-time oracle bytes.

A `correction` template has a later user turn replace a value the user already
gave, through `user_simulator_turns[].slot_updates`:

```yaml
slots:
  amount_vnd: {source: "literal:[200000]", visible_in_first_turn: true}
assistant_milestones:
  - {id: confirm_original, type: ask_confirm}
  - {id: confirm_corrected, type: ask_confirm}
  - {type: tool_call, tool: create_transfer, args: {confirm: true}}
  - {type: final_answer}
user_simulator_turns:
  - after: confirm_original
    content_template: {vi: "Khoan đã, sửa số tiền thành {amount_vnd_corrected}đ nhé."}
    slot_updates:
      amount_vnd: {source: "literal:[1500000]", bind_as: amount_vnd_corrected}
  - after: confirm_corrected
    content_template: {vi: "Đúng rồi, xác nhận số tiền mới."}
```

Expansion binds both values, so each instance carries the original bindings, the
replacements **in the order the conversation delivers them** — milestone order, not
declaration order — and the values in force at the end; rendering and the expected
trace then walk the conversation and use whichever value is current at that point. `bind_as` is a surface-only alias for the replacement, so the turn can
name the new value while calls keep binding from the canonical slot key. Rules:

- A replacement resolves through a source of its own, which faces the same gold
  gate as the value it replaces and must be the same source kind.
- Only a slot the user already stated (`visible_in_first_turn: true`) can be
  corrected, and a replacement equal to the value in force when the turn lands is
  skipped as a no-op — the pair is dropped while the instance budget is being filled,
  so a template drawing both values from one collection still binds instances.
- `slot_updates` requires `turn_policy: correction` and that policy requires at
  least one replacement, in both directions.
- No call may consume the superseded value. The check follows the **slot**, not the
  value: a call is rejected when a corrected slot it read still held the old value at
  that point, so an unrelated argument that merely happens to equal the replaced value
  is left alone. `expected_trace` refuses it while deriving, and `schema_validation`
  re-checks the published arguments by slot name.
- A correction withdraws whatever the user confirmed before it. A confirmed
  mutation therefore needs a fresh `ask_confirm` and a fresh user reply after the
  replacement, otherwise the row fails validation the same way an unconfirmed
  mutation does.

`task_id` and the per-task `seed` are SHA-256 derivations over pack id/version,
`template_id`, sorted `fixture_refs`, canonical slot bindings (a correction's
replacements included), and `variant_index`, so ids are stable across machines.

A task is dropped rather than published when its surface breaks a guard, its
trace fails schema validation, the two replays diverge (`nondeterministic_replay`),
or an assertion fails. `run_manifest.json` reports `stage_counts` with one entry per
stage that can drop a task (`expanded`, `surface_passed`, `trace_derived`,
`trace_dropped`, `schema_passed`, `replay_passed`, `published`). Each count reports its
own stage only — a guard-rejected row that replayed still counts in `replay_passed` —
and `published` is where the stages meet, so a shortfall names the stage that caused it.
The manifest also summarizes dependent binding failures under `trace_drop_rejections`,
and counts guard rejections per template under `surface_guard_rejections`.
`benchmark_raw.parquet` holds every row that passed schema and replay;
`benchmark.parquet` is that set minus surface-guard rejections, so the two files match
only when no guard fired. A declared held-out policy is validated and included in the
pack fingerprint during prepare, then enforced twice during generation: expansion
never binds a reserved template or fixture row, and publication re-scans every row
against the same policy and stamps `held_out_hit`. Any hit aborts the run before a
parquet or manifest exists, and the offending task ids are listed in
`stage_cache/held_out_scan.json`. A reservation that leaves a slot with no bindable
row, starves a category's `tasks_per_category`, or withholds every template is a pack
error and stops generation with the shortfall named. The manifest records the policy
lineage, the rows scanned, and the templates and fixture rows Stage 4 withheld. Packs
without a held-out source keep `held_out_hit` null and record `evaluated: false`.
When `held_out.policy.fixtures_in_backend_state` is `false`, the same reserved fixture
rows are also removed from every generation replay and executable-evaluation oracle
reset. The full inventory remains available only to the trusted binding and publication
checks that prove those rows were withheld. Local Python episodes run from a temporary
pack mirror whose fixture file contains that same projection, so backend or assertion
code resolving `fixtures.json` relative to `__file__` cannot reopen the full inventory.
This mirror protects the pack-relative contract; it is not an OS sandbox for hostile
allowlisted code that deliberately names unrelated host paths. A value of `true`
retains the reserved rows in backend state for packs whose oracle behavior intentionally
depends on them.
Validation does not reject a probe merely because one argument equals a reserved id:
tool argument names do not prove which fixture, if any, the backend dereferences. If
the executed probe instead returns `not_found`, a mismatch records the matching
argument, collection, and canonical held-out fixture reference in the validation
report.
Each row exposes exactly the tool definitions named by its `tools_present`. A
template that omits `tools_present` exposes the pack's entire tool catalog, which
is the realistic default: a deployed assistant sees every tool it owns, not a
pre-narrowed shortlist. Declaring the field narrows that exposure and is checked
to name known tools and to cover the template's `required_tools`, so a narrowed
row can never hide a tool its own gold calls need.
Rows from `lineage.policy: smoke_no_publication` retain the pack's validation
`tier` but set `gold_eligible: false`.

Two columns deviate from a naive Arrow struct, because a struct unions keys across
rows and would pad every call with nulls — turning "argument absent" into
"argument is null" and advertising parameters a tool does not accept:

- `expected_tool_calls[].arguments` is a `map<string, string>` whose values are
  canonical JSON. Decode with `row_schema.decode_arguments`.
- `tools` is canonical JSON text (a JSON Schema has per-tool keys). Decode with
  `row_schema.decode_tools`.

`expected_tool_calls[].turn_index` is the 0-based ordinal of the assistant message
that issues the call, counted across assistant messages — so the confirming call
of a `confirmation` task has `turn_index: 1`, after the `ask_confirm` message.

## Run It

```bash
# Create a runnable local-Python starter
python -m nemotron.steps.byob.scripts.scaffold_oracle_pack \
  --domain "inventory service" \
  --target /tmp/inventory_oracle_pack \
  --transport python

# Validate a pack and print its report
python -m nemotron.steps.byob.scripts.validate_oracle_pack \
  --config src/nemotron/steps/byob/bfcl/config/tiny.yaml \
  --output-dir /tmp/bfcl-tiny-validation

# Normal family lifecycle
python -m nemotron.steps.byob.scripts.run \
  --config src/nemotron/steps/byob/bfcl/config/tiny.yaml \
  --stage prepare
python -m nemotron.steps.byob.scripts.run \
  --config src/nemotron/steps/byob/bfcl/config/tiny.yaml \
  --stage all
```

`config/tiny.yaml` (`tiny_library`) and `config/smoke.example.yaml` pin
`lineage.policy: smoke_no_publication`; neither is publication-eligible.
`banking_vn_oracle_pack` is the reference domain pack, and it is worth reading
before authoring your own: it declares a template for every policy edge the
pipeline supports (`single_turn`, `missing_slot`, `confirmation`, `correction`,
`multi_tool`, `dependent_call`, `negative_path`, `clarify_only`, `irrelevant`),
and no template narrows `tools_present`, so every row must pick its calls out of
the full nine-tool catalog. `config/publication.example.yaml` runs it at
publication scale.
`config/default.yaml` is the publication-oriented template. Its
`oracle_pack.manifest_path` is a `REPLACE_ME_*` placeholder so the template can never
publish an example domain by omission; point it at your own pack and the config runs
as-is, because its optional profile, paraphrase, and surface-judge roles are disabled.
Disabled roles are recorded with null identity and set
`generation_mode: template_only`, which does not affect gold eligibility. The
remaining `REPLACE_ME_*` entries only matter once the corresponding role is enabled.

The generated pack README is the shortest transport-specific quick start.
The complete manual lifecycle, including endpoint identity pins and evaluation,
is in
[`create-bfcl-from-oracle-pack.md`](../patterns/create-bfcl-from-oracle-pack.md).

Generation supports reference profiling, controlled paraphrasing, Stage 10
surface-quality validation, and optional Stage 11 semantic deduplication and
balancing. Paraphrasing asks each binding for a different structural style, so a
pack whose domain or language needs other registers declares its own list in
`surface_generation.surface_style_axes`; `paraphrases_per_template` may not exceed
that axis count, because one binding cannot be asked for the same style twice. Stage 11 fails closed on backend/artifact errors and requires an
explicit abort-or-non-gold policy for unmet targets. Stage 12 supports
`exports.bfcl_json` and `exports.nemo_evaluator_bundle`; unknown export names and
settings owned by later stages are refused rather than ignored.

## Compatibility Exports and Recovery

Both compatibility writers receive the same canonical projection of
`benchmark.parquet`. Stage 12 reads enabled outputs back, checks row count, task
order, tool definitions, expected calls, ordering policy, and content hashes, and
writes `exports/export_validation_report.json`. `run_manifest.json` records every
format as enabled or disabled and pins the report and export hashes.

Exports are specified by versioned export contract `1.0`, and `bfcl_json` and
`nemo_evaluator_bundle` each carry their own `schema_version`. The BFCL adapter
pins the upstream `BFCL_v4_multi_turn` question/function and separate
ground-truth JSONL envelopes, while Nemotron metadata retains the assertions,
parallel groups, ordering policy, and provenance that upstream BFCL does not
represent. This is data-format compatibility, not a claim that an arbitrary
oracle pack provides BFCL's own domain-specific executable classes.

`export_projection` (`1.0`) is the single decode path both writers read.
`benchmark.parquet` is read once, its schema is checked against the published
schema, and every row becomes a canonical export object. Tool definitions decode
from canonical JSON text and each argument decodes from its own canonical JSON
value in the Arrow map, then re-encodes byte for byte, so the contract
distinguishes `"1"`, `1`, `true`, and `1.0`. A writer receives the projection and
never a path, so no format can decode the parquet its own way, and the projection
is deeply immutable. `project_published_benchmark` optionally binds the projection
to the content hash and publication order the manifest reports, which stops an
export built from a parquet that was replaced after Stage 12 verified it. The
contract requires row-count and truth-field equivalence, source benchmark and
validation-report hashes, and complete NeMo bundle references for the dataset
schema, metadata, evaluator config, and system-prompt catalog.

The projection also derives, once, the structure a writer would otherwise
reconstruct. Each assistant message that issues tool calls becomes one call group,
checked against the rendered `messages` by name and argument: a group is parallel
exactly when that message issues more than one call, its `turn_index` is the
ordinal of the assistant message, and `user_turn_index` is the request it answers.
`calls_by_user_turn` keeps an empty slot for a clarifying turn that triggers no
call, so BFCL's per-user-turn ground truth cannot shift answers onto the wrong
request. Projection-level provenance — pack, version, tier, prompts, languages,
turn policies — is derived from every row rather than read off the first, so rows
that disagree stop the export instead of being silently labelled after row zero.

The `bfcl_json` writer turns that projection into two JSONL files under
`exports/bfcl_json/`: the questions, and beside them in `possible_answer/` the
expected calls, joined by `id`. Four format decisions are fixed there. JSONL
rather than a JSON array, so a harness can stream the benchmark and retry a single
task rather than a whole file. Two files rather than one, because a record that
carries its own answer invites a runner to prompt the model with it. Parallel
calls stay grouped in the Nemotron extension, since upstream's per-turn answer
list is flat and cannot tell two simultaneous calls from two sequential ones. And
the expected calls are the only answer exported: the recorded oracle results stay
under `x-nemotron.messages` as provenance and never appear in `question` or
`ground_truth`, so a scorer re-executes tools instead of diffing a model's output
against a snapshot of one backend revision. Provenance likewise lives only in the
extension, so rendering `question` cannot leak a pack version or a seed into the
prompt. Bytes are deterministic — sorted keys, no incidental whitespace, `\n`
endings, UTF-8 left unescaped so a non-Latin surface stays readable — and the
format's `content_hash` covers file names together with bytes, so a renamed file
or a swapped question/answer pair changes the digest.

The NeMo bundle is input for the native function-calling adapter, written as six
files under `exports/nemo_evaluator_bundle/`: `bundle.json` (the descriptor, which
names the other files and pins the dataset's hash and record count),
`dataset.jsonl` in publication order, `dataset.schema.json`, `metadata.json`,
`evaluator.yaml`, and `system_prompts.json`. Its
`evaluator.yaml` describes seed/replay/scoring semantics but is not a standalone
NeMo Evaluator 0.2.x task registration or Launcher config. Native tool calls need
an installed/containerized harness plus a tool resource service.

Three of the bundle's decisions differ from `bfcl_json`. `seed_messages` is the
only model-input field and contains only the leading system messages plus the
first user turn; gold assistant actions remain in `reference_trace`, and
`replay_steps` lets the adapter release a recorded tool result only after the
candidate produced the corresponding expected call, then release the next user
turn. That separation is what prevents a generic chat adapter from forwarding a
full gold trace as the prompt. The dataset schema is generated from the record
model rather than written by hand, so it cannot drift from the records beside it.
And the declared metrics stop at `tool_selection` and `arguments`, plus
`call_ordering` only when some task expects more than one call: `results` and
`task_success` would need the pack's tools re-executed against oracle state, which
no bundle file provides, and an ordering metric over single-call tasks would
report a perfect score for something it never measured. The adapter task id is
derived from `pack_id`; lossy normalization, including an entirely non-ASCII id,
receives a deterministic hash suffix to avoid collisions, while the verbatim
`pack_id` remains in `metadata.json` and `bundle.json`.

Which formats can be written is declared once, as the writer registry Stage 12
dispatches through. Config validation reads the same registry, so a format named
in the contract but never wired to a writer is refused at startup instead of
silently producing no file. The whole bundle is encoded, digested, and validated
in memory before any file is created, so a projection that cannot be
expressed — an unresolvable prompt id, a row no evaluator record represents —
leaves nothing behind to be mistaken for a bundle. The `exports/` tree is removed
before writing, because a file this run did not write would otherwise travel
inside a bundle whose digest never covered it, and the bytes on disk are
re-digested afterwards so a truncated write cannot publish a descriptor nobody
re-checks. `content_hash` and `files` are bundle-relative like the descriptor, so
archiving the directory elsewhere does not invalidate it. A writer that fails
takes the export tree and both parquet files with it, since a reader cannot tell a
partial bundle from a complete one, and any later abort in Stage 12 discards the
tree for the same reason.

Stage 12 writes into `.stage12-*`, validates all payloads, then promotes parquet
and exports before moving `run_manifest.json` last. Treat the manifest as the
commit marker: without it, no adjacent parquet or export belongs to a published
run.

Recovery is fail-closed:

- **Schema mismatch:** do not edit generated rows; remove final Stage 12 payloads
  and regenerate with one code revision. Select consumers using the
  `schema_version` values in the manifest.
- **Unsupported call layout:** use the task id in the exception to fix the pack's
  tool identifiers or conversation shape. The BFCL JSON adapter requires
  Python-compatible function and argument names; the NeMo input contract requires
  an unambiguous seed/replay plan.
- **Hash or equivalence mismatch:** inspect
  `exports/export_validation_report.json` when it exists, then regenerate the
  whole publication. Never repair one exported file in place.
- **Interrupted publication:** if `run_manifest.json` is absent, rerun
  `stage=generate`. Startup removes abandoned `.stage12-*` directories and stale
  final payloads before the next attempt.
