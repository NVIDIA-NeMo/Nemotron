# BFCL Eval Scoring Contract

This document defines what a BFCL score *means*. It is referenced by
`scoring.contract` in `eval_config.yaml` and content-hashed into
`eval_config_hash`, so editing it changes the identity of every evaluation that
points at it: two runs that agree on every flag but disagree on the rules below
are not the same evaluation and must not be compared.

Schema version: `1.1`. Sections marked *pinned* are the values a publishable run
must use; a run that relaxes any of them is debug-only and reports
`publication_allowed: false` with the field that caused it.

## What is being scored

Each benchmark row states a user request, the tool definitions the model may
call, the gold tool calls (`expected_tool_calls`) with their turn, group, and
order, and the pack's success assertions. A candidate model sees the request and
the tools. It never sees the gold calls, the assertions, or the oracle results:
those are the answer key.

Two evaluation modes exist, and a report always says which one produced a number.

- `trace`: compare the calls the model proposed against the gold calls. No oracle
  runs, so a trace score says the model asked for the right thing, not that the
  request would have worked.
- `executable`: additionally replay the proposed calls against the oracle pack and
  evaluate the pack's assertions. Only gold-eligible source runs may be published
  in this mode, because only those rows were validated against a real oracle at
  generation time. The eval config must also pin a resolvable `source_oracle`:
  the matching pack manifest plus the exact Python backend or endpoint config.
  The source manifest's `oracle.kind` is lineage, not an executable resource.

An executable run records a versioned `ExecutableEpisode` before scoring. The
episode binds the verified source, oracle, and complete executable-task identities and preserves every
candidate turn, one outcome for every proposed call (including calls that were
not executable), the final-state hash, and classified assertion outcomes. Each
outcome is bound to the provider call it records by typed JSON equality, so a
coerced argument cannot pass as the value the provider sent.

Structured business rejection is distinct from tool or infrastructure failure.
A non-object JSON return is retained as malformed evidence whose value, type, and
canonical hash agree, rather than stored as if it had conformed. Result shape and
commit state are independent: malformed output from a mutating call can coexist
with an unknown commit verdict, and unknown commit takes terminal precedence
rather than being hidden by the result failure. Terminal status is checked in
both directions, so neither fact can be restamped as a completed episode.

Obtaining a result is also distinct from admitting it to the candidate prompt: a
result the driver obtained but never released is retained without claiming the
candidate read it. Scripted user-message releases are bound to their turns by
content hash, and both user-message and tool-result release counts are derived
from those records rather than accepted as independent counters. A
tool-execution event cites both its outcome and the turn that owns it. An
advanced nonterminal turn must release its live results in the following model
request; a terminal tool-only turn can complete without inventing such a
release.

Executable driving starts from `ExecutableTaskSpec`, not an arbitrary row. Its
builder binds the complete canonical projection to the verified benchmark
content and task order, checks candidate/task authorization in
`EligibleEvalPlan`, requires a gold-eligible executable source, and binds the
verified oracle and source clock. Model requests can read only the answer-free
seed, model-facing tools, candidate output, live tool results, and earned
scripted user turns. Expected calls, recorded results, fixture references,
milestones, dependencies, and assertion metadata remain runner-only.
Each verified `from_result` marker is projected into explicit producer,
consumer, argument-path, and result-path coordinates. Before the consumer is
asked, its expected argument is rebuilt from the paired producer's live result.
The candidate call is never repaired. Missing or ambiguous producers,
unavailable results, missing paths, type mismatches, and schema-invalid
substitutions are deterministic infrastructure outcomes; the recorded gold
result is never consulted as a fallback.
Assertion execution receives verified template metadata plus `slots`,
`slots_initial`, and `slot_updates` reconstructed from the published opening
surface, expected trace, cited fixture rows, and typed source values from the
verified pack's fixture, literal, enum, range, and absent-id declarations. A
final value no channel settles is reported in `unresolved_slots`; unknown
pre-correction and correction values are reported independently in
`unresolved_slots_initial` and `unresolved_slot_updates` instead of being guessed
from another phase. The isolated assertion runner tracks reads of every class of
missing slot; an assertion that reads one is recorded as an infrastructure error
and terminates the episode rather than scoring against the candidate, while
unrelated assertions retain their ordinary pass or fail verdict.

Python and endpoint packs implement one `OracleSession` protocol. Both adapters
use a task-local persistent process worker for reset, ordered calls, state, and
assertions; Python pack modules are never imported by the evaluator process.
Endpoint sessions are cleaned up on every exit path. A mutating call is never
retried automatically, because losing its response does not prove its state
change was rolled back. When a response arrives, canonical pre-call and
post-call state hashes prove whether the mutation committed; a missing snapshot
keeps the commit verdict unknown. Plan, source, publication, scoring policy, and
task identities must agree before reset, and even preflight failure closes the
already-open task-local session.
A call that asserts the pack's confirmation parameter reaches the oracle only on
a call turn covered by the user's scripted confirmation and after the
candidate's complete call batch matches the authorized batch. A call that leaves
that parameter unset is the unconfirmed probe a confirmation template issues
before the user answers, so it executes wherever the trace places it — the gate
follows the same rule generation applies when it admits the template as gold. A
bypass attempt is retained as `not_executed` evidence without invoking pack
code, and ends the episode as `confirmation_not_earned`, a candidate stop.

Diagnostic prose and cache replay do not change the episode identity; canonical
calls, results, reason codes, release verdicts, state, and assertion verdicts do.
Each record excludes only its own diagnostic wording, so a key an oracle happens
to name `detail` stays part of the evidence.

Executable scoring is a pure projection of that evidence.
`score_executable_episode(...)` first requires the episode, task spec,
authorization plan, source, oracle, script, eval config, scoring policy,
candidate identity, and task authorization to agree. Identity drift, missing or
unexpected assertions, and mutation-policy contradictions are typed evidence
errors, not ordinary failed scores.

The scorer reuses the trace scorer's normalized call view and pure comparison
kernel for tool selection, arguments, schema validity, grouping, ordering, text,
and trace completion. That normalized scorer is an internal kernel, not an
authorization-free public scoring API. The executable parser independently
requires the episode and task spec to agree on task, candidate, plan, config,
source, oracle, script, and task-spec identities before it creates the shared
view. It also refuses either trace contract when `completed` evidence omits a
scripted turn. Executable scoring adds five dimensions:

- `oracle_execution`: `completed` and structured `business_rejection` are
  successful exchanges; not-executed calls are candidate failures, while
  malformed results, tool failures, timeouts, and unknown execution outcomes
  are infrastructure failures.
- `dependency_resolution`: applies to reached `dependent_call` arguments.
  Every expected downstream value must be derived from exactly one paired,
  successful prior live execution and satisfy the verified path, JSON type, and
  consumer schema. A candidate must emit that value itself for the ordinary
  argument gate to pass.
- `commit_state_known`: applies to mutating tools only. `committed` and
  `not_committed` are determined evidence, `not_started` is valid for an
  unexecuted call, and `unknown` is an infrastructure stop. A state delta never
  substitutes for a semantic pack assertion.
- `assertions`: observed assertions must form the declared sequence in order.
  `passed` succeeds, `failed` is a candidate failure, `not_applicable` is
  skipped, and `infrastructure_error` is a non-candidate stop. A fatal oracle
  or state failure may leave a declared suffix unrun; the failed infrastructure
  gate remains the stop and assertions are not fabricated. The scorer never
  runs an assertion.
- `executable_completion`: records whether live driving reached its terminal
  boundary and classifies candidate and infrastructure stops separately.

Executable task scores use contract `1.3`. Version `1.3` binds terminal episode
status to the executable-completion gate and infrastructure-stop attribution,
binds `task_success_rate` back to the derived task verdict, and requires every
metric N/A reason to come from the registered metric taxonomy rather than a gate
namespace. Existing scores retain their historical version and identity.

Every executable score reports all twelve trace and executable gates, using
`not_applicable` rather than omission. Under pinned
`task_success: all_applicable_gates`, every applicable gate must pass. Debug-only
`assertions_only` requires at least one declared assertion and all of them to
pass; candidate trace failures do not decide that mode, but evidence and
infrastructure failures remain blockers. `allow_llm_repair` is always refused.

Each score also emits the fixed per-task metric taxonomy
`schema_valid_rate`, `tool_name_accuracy`, `argument_accuracy`,
`call_group_accuracy`, `call_order_accuracy`,
`required_call_subset_accuracy`, `milestone_accuracy`, `turn_success_rate`,
`tool_execution_success_rate`, `assertion_success_rate`, `state_match_rate`,
`path_success_rate`, `final_answer_success_rate`, and `task_success_rate`.
Every metric carries integer numerator and denominator plus their value.
A zero denominator carries `value: null` and a stable N/A reason code; it is
never reported as a vacuous zero or one. A gate-derived metric that does not
apply uses `metric.gate_not_applicable`; all N/A reasons are members of the
error taxonomy's `metric.*` registry. Assertion-derived metrics use the
pack's literal `ASSERTION_CAPABILITIES` category, while `path_success_rate`
also requires the applicable call, milestone, dependency, execution, and
completion gates. A declared assertion that an infrastructure stop left unrun,
and an infrastructure failure in a path gate, make the affected metric N/A
(`metric.assertion_evidence_incomplete`, `metric.path_evidence_incomplete`)
rather than a candidate failure: the gate already records the stop, and a
broken oracle is not a wrong answer. Run-level aggregation sums numerators and
denominators over task scores and preserves a shared zero-denominator reason.
When task contributions have different N/A causes, the aggregate uses the
stable `metric.no_applicable_task` code instead of concatenating diagnostics.
`ExecutableCandidateScore` requires the exact task ids and publication order
authorized by `EligibleEvalPlan`; it refuses partial, duplicate, cross-candidate,
cross-oracle, or cross-contract inputs. Its identity cites every task
`score_hash`, while metric quotients remain derived rather than hashed.

A metric's `value` is its numerator over its denominator, so `score_hash`
carries the counts and the reason code but not the quotient.

`ExecutableTaskScore.score_hash` is path-free and time-free. It includes the
task-spec and episode identities, result and malformed-result hashes, commit
verdicts, assertion verdicts, gate reason codes, and the scoring contract and
policy. It excludes each record's own diagnostic `detail`, so rewording a
diagnosis does not fork a score.

Executable driving may persist a complete episode in
`tool_trace_cache.jsonl`. The cache is append-only, record-hashed, process
locked, and keyed by candidate, task, plan, eval config, source, oracle, script,
and task-spec identities. A request without a completion is crash evidence, not
a cache hit; a second non-identical observation for the same key is a conflict.
On a hit the driver replays the complete `ExecutableEpisode` and sets only its
non-semantic `replayed` flag. It never memoizes one tool call in isolation:
skipping a mutating call would not reproduce state for dependent calls, final
state, or assertions. The cache's byte hash is available for
`eval_manifest.json`.

Final publication uses `write_executable_eval_artifacts(...)`. It refuses task
scores that do not exactly match each aggregate's ordered task ids and score
hashes. It then writes immutable `eval_report.json` and, when enabled,
`eval_task_results.parquet`; `eval_manifest.json` binds their byte hashes to the
source/config/plan identities, candidate aggregate hashes, and the byte hashes
of both `candidate_io_cache.jsonl` and `tool_trace_cache.jsonl`. A required cache
that is absent or lives outside the artifact directory is a publication error,
not an omitted manifest entry. Publication reparses both hash-verified caches,
requires one complete tool-trace episode per task score, and proves every
episode turn's request/status/response hash against the candidate cache; valid
but unrelated cache files are refused. Episode evidence is streamed one record at
a time, because a whole run's episodes do not fit in memory at the point where
the run is otherwise finished. A run that publishes the candidate cache without
tool results keeps no episodes to compare against, so that cache must still prove
that no claimed request was left without a completion.

Immutability of `eval_task_results.parquet` is a property of its rows, not its
bytes: a re-encoding by another writer build is accepted, a changed row is not.
The manifest also records Python, platform, machine, pipeline Git sha and dirty
state, pipeline source, dependency-lock, and worker-image identities. The
pipeline source hash covers the whole `byob` step package, since this family
executes through shared runtime and isolation code. Every provenance field is
best effort and reports null when it cannot be established; a Git sha is refused
outright when the surrounding repository does not contain this pipeline, because
a confident wrong commit is worse than an absent one.

The complete executable run is exposed as `run_bfcl_eval(...)`. Source
verification and contamination gating finish before the first candidate
request. Candidates run sequentially in plan alias order; within one candidate,
authorized tasks run in publication order with at most
`limits.max_parallel_tasks` active tasks. Every task owns one oracle session.
The first raised setup, authorization, cache, or scoring exception cancels
sibling tasks and closes every opened oracle and candidate client; an
infrastructure terminal represented by a completed `ExecutableEpisode` remains
scored evidence and does not abort the batch. A candidate endpoint that rejects
the credential is raised rather than recorded, because every task presents the
same key: a per-task record would spend the whole task set on a configuration
fault and aggregate the refusals into metrics indistinguishable from a score.
No completion is written for the refused call, so a rerun re-contacts the
endpoint instead of replaying the refusal.

A caller-supplied `eval_run_id` is reused exactly. Otherwise a fresh output tree
gets a timestamp-plus-UUID identity; replay in a completed output tree reuses
the immutable report's id. Config identity is never used as run identity because
two fresh provider samples under one config are different runs.

Error attribution is frozen by `error_taxonomy.py` contract `1.0`. It defines
the trace and executable terminal-status mappings, the shared non-candidate-stop
sets, fatal `eval_*` exception codes, metric N/A codes, and accepted reason-code
namespaces. Final report and manifest documents cite its content hash, and a
regression test binds every set in it to the codes the pipeline can actually
emit, so the hash cannot claim a coverage the code no longer has.
`eval_task_results.parquet` retains `failure_codes` and additionally records
`episode_status`, `non_candidate_stop`, and structured `failure_records` with
layer, code, attribution, and subject. Each incomplete episode contributes one
`episode`-layer record before its gate records: every incomplete episode fails
the same completion gate, so only the terminal status separates a spent budget
from a broken oracle. `non_candidate_stop` is true for a terminal the taxonomy
attributes away from the model and for infrastructure that broke inside an
otherwise finished episode. Oracle business error codes are domain results and
never enter the harness taxonomy.

## What the score is taken over

A number is only meaningful if the benchmark behind it is identified. Before any
candidate is contacted, the source named by the eval config is verified against
the committed publication: `run_manifest.json` still hashes to what the
config resolved, both benchmark tables hash to every declaration the manifest
makes about them, the published table is a selection of the raw table with no
truth rewritten and no held-out row shipped, and the published rows decode into a
unique task set. Every score cites the resulting `verification_identity`, which
covers hashes, row counts, task ids, and the oracle pack fingerprint, and no path
or timestamp. A run that cannot name the source verification it executed under is
not publishable, because "the model scored 0.71" without a benchmark identity is
not a comparable result.

An `executable` score additionally requires the oracle pack to still fingerprint
to what generation certified, across every file in its tree, and requires the
resource that runs to be the one the pack's own manifest selects. Replaying
against an oracle that changed since generation cannot confirm the gold trace it
certified, so such a run is refused rather than reported with a caveat.

The source is re-checked immediately before execution and the run aborts if
anything moved. Two halves of a run taken against two different benchmarks would
otherwise be averaged into one number.

A translated benchmark is scored under this same contract only if it preserves
every field the contract compares — tool names and complete parameter schemas,
the gold calls with their turn, group, and order, the assertions, and the gating
columns. Translation may change approved conversation content, intent display
text, localized metadata, and `tools[].function.description`. The evaluator
compares a field-level tool projection that removes only that description; a
translation that changes any other tool field or scorer input is a different
benchmark and its scores are not comparable to the source's.

## How candidate output is observed

Each assistant turn is requested through the native OpenAI-compatible
function-calling protocol. The request contains only the model-facing
conversation and ordered tool definitions, plus the sampling parameters pinned
by the candidate config. Gold calls, assertions, fixtures, and unreleased oracle
results never enter the request.

The evaluator records the provider's `message.tool_calls` in order and keeps each
function argument exactly as returned. It parses the argument string once under
strict JSON. Invalid JSON, non-object JSON, missing arguments, wrong types,
unknown function names, and duplicate calls are candidate observations, not
transport failures: they are not retried, coerced, or repaired by another LLM.
Only transient endpoint failures may be retried, under the pinned retry budget
and deadline. Attempts are buffered and enter the hash-verified candidate I/O
cache only with a semantically completed response. Replay uses such a committed
completion without contacting the endpoint; transport, provider, authentication,
malformed-envelope, deadline, and cancellation outcomes remain uncached. A
cancelled call is an interruption of the run, never an observation of the model,
so it is never scored as one.

Transient means exactly one set: request timeouts, connection failures, `408`,
`429`, and selected `5xx` responses, attempted within `limits.max_retries` and
the logical candidate deadline. Backoff jitter is derived from the request hash
rather than from a clock, and a `Retry-After` header is honored only inside that
deadline. Response bodies are streamed under a fixed size bound. An envelope
failure under HTTP 200 is model output rather than a transport error, so it is
not retried. Provider-only request fields may enter only through the namespace
matching the candidate's `provider` and `provider_api_version`, and may never
replace a standard field. Credentials are read from `api.api_key_env` only after
the cache lookup misses, which is what lets a replay run without any secret, and
no credential value enters a request hash, a diagnostic, or an artifact.

Each cache record is verified once, when it first appears, and a completion cites
its attempts by hash, so one response body is stored once however many records
refer to it. No failure-only request or attempt is persisted, so a resumed run
contacts the endpoint instead of reading infrastructure failure as the model's
immutable answer.

## How the conversation advances

A multi-turn task is replayed rather than simulated. The prompt opens as the
system prompts and the first user request; nothing else from the published row is
in it. From there exactly three things may be added: an assistant turn the
candidate itself produced, a tool result the benchmark recorded, and the next
scripted user request.

A recorded tool result is released only after the assistant turn that asked for it
matches the trace, and it is addressed to the id the candidate's own call carried.
Every replayed call must declare type `function`, and ids within an assistant turn
must be non-empty and unique; a missing type is preserved as a mismatch rather
than repaired. Duplicate ids are ambiguous and receive no result.
This is what keeps a wrong call from seeing the right answer: a model that
transfers 999 instead of 100 is not handed the result of the correct transfer. The
next user request is released only after the turn that would have prompted it, so
a model that calls a tool where the trace asked a clarifying question never
receives the answer to the question it did not ask.

For an intermediate text-only turn, `intermediate_text_matching: structural`
(*pinned*) asks what the turn did rather than how it read: the candidate answered
in words instead of calling a tool, and those words say something. Both halves
are load-bearing. A turn that called a tool where the trace asked a question has
taken material it did not earn, and a turn that returned nothing asked the user
nothing, so neither releases the next request.

The recorded sentence is one translation of one phrasing of that behavior, and a
pack states its intermediate turns as milestone classes — ask for a slot, ask to
confirm — rather than as sentences. Requiring the sentence back would score
wording rather than tool use, and would fail a model that asked the right
question in its own words. What holds the turn to its domain meaning is the
pack's own success assertions, which read the trace and the final state.

`verbatim` restores the character-for-character demand. It measures a different
thing, so a run using it is not comparable to a publication run; it is kept for
reproducing runs that were scored that way.

The tool-result release decision uses the same call comparison this document
defines for scoring. That agreement matters in one direction in particular: a
release gate stricter than the scorer would end an episode the scorer would have
credited, so a correct model would fail a task on transport grounds.

Trace replay does constrain one thing the scoring rules do not. Because a recorded
result is only meaningful at the point the trace reached it, a candidate that
defers a call group to a later assistant turn ends the episode there. Within a
single turn, ordering follows the row's `call_order`.

An episode records what happened rather than a verdict: which turns were asked,
what came back, whether each sent turn advanced, and why it stopped — reaching the
end of the trace, a mismatch, malformed output, an unreachable endpoint, an
ambiguous tool-call id, or a spent turn or episode budget. A turn the episode
budget prevented from being sent is a terminal event, not a fabricated candidate
observation. The record retains `finish_reason`; an explicit truncation, content
filter, or max-token finish cannot be converted into a complete answer by
re-scoring. Every number is derived from that record, so re-scoring never re-asks
the model.

The driver applies that same rule while the episode is still running. A provider
finish that explicitly means truncation or filtering — `length`,
`content_filter`, or a provider's max-token variant — never advances a turn, even
when the partial payload otherwise matches the trace, because the turn the model
meant to take is not the turn that arrived. A terminal turn must likewise carry
non-empty plain or structured text.

The episode is built from a conversation script, and the script selects its row
and conversation plan from a canonical projection only after the projection's
content hash, row count, and complete task sequence match the verified source, so
no caller-supplied identity can stamp a stale row as current and every candidate
replays the identical source-bound conversation. Immediately before a turn is
sent, the candidate's canonical weights identity must still equal the identity the
contamination plan authorized; reusing an alias for different weights is refused
rather than scored.

Driving returns rather than raises for everything the model can cause. A wrong
tool, wrong arguments, unparseable arguments, a call with no id, an unreachable
endpoint, a malformed envelope, and an exhausted turn or episode budget each
become a distinct terminal status on the recorded episode, alongside the ordered
events and every observed turn. An unauthorized task, or a row that is not a
replayable conversation, raises instead: those are defects in the run rather than
facts about the model, and recording them as episodes would put them into a
metric. Cancellation propagates without producing an episode at all.

## Argument matching

`argument_matching: schema_then_canonical` (*pinned*)

1. **Schema step.** The candidate argument object must satisfy the tool's
   declared parameter schema from the row's `tools`. Defaults are annotations,
   not supplied arguments: a parameter that is both required and defaulted is
   still missing when the candidate omits it. A schema-invalid call neither
   matches the trace nor earns a recorded result.
2. **Default-equivalence step.** After schema validation passes, when
   `insert_declared_defaults: true`, defaults are inserted recursively on both
   sides before canonical comparison. A candidate may therefore omit a default
   stated by gold, or explicitly supply the exact declared default that gold
   omits. A different candidate-supplied value remains a mismatch; for example,
   `confirm: true` differs when the omitted gold value defaults to `false`.
   Insertion recurses through nested objects and arrays and follows validated
   local `$ref` and `allOf` schemas.
   Pack validation rejects external, missing, or cyclic references and rejects a
   declared default that does not satisfy the schema it would be inserted under.

   A required argument is still missing when the candidate omits it; defaults
   never run before schema validation.
3. **Canonical step.** Both sides are then compared as canonical JSON, by type as
   well as by value. `1`, `1.0`, `"1"`, and `true` are four distinct values: a
   scorer that treated them as equal would accept a limit of `"1"` where the gold
   call passed the integer `1`.

Extra arguments the schema does not declare are a mismatch. Missing required
arguments are a mismatch. Argument order inside an object is irrelevant;
canonical JSON sorts object keys.

`canonical_only` skips the schema step. It exists for debugging packs whose
schemas are incomplete and is not publishable.

## Call selection, grouping, and order

- **Selection** is scored per call: the function name must be one the row's
  `tools` declares, and it must be the name the gold call used.
- **Grouping** (`respect_call_group: true`, *pinned*) requires the calls a model
  emits in one assistant turn to match the gold call group. Parallel calls the
  benchmark asked for in one turn may not be spread across turns, and sequential
  calls may not be collapsed into one.
- **Ordering** (`respect_call_order: true`, *pinned*) requires the gold order
  within a group and across groups. A row that carries `call_order_prefix` scores
  the configured number of `required_tools` first appearances as ordered and the
  remainder as unordered, which is how a pack declares "these two may happen in
  either order, but both come after login". Repeated calls to one tool do not
  consume additional prefix positions.

Relaxing `respect_call_group` or `respect_call_order` changes what is *scored*,
not what replay can hand back. Replay holds exactly one recorded result per gold
call, so a turn of a different size still has no faithful reply and still ends
the episode; and a strict row answered out of order still earns no result at the
point the trace reached. A debug run with either flag relaxed therefore reports
the gate as not applicable while the completion gate below still fails. This is
deliberate: the flags are for asking "would this model have been right under
looser rules", not for turning an unreplayable episode into a success.

## What a trace score reports

A trace task score uses contract `1.1`. Version `1.1` adds mandatory per-gate
failure attribution to the score identity and intentionally does not load a
persisted `1.0` score as if it had been produced under the new semantics. Existing
`1.0` evidence must be re-scored; its historical `score_hash` remains an identity
under the old contract and is never restamped. Omitting `failure_class` from a
failed `1.1` gate is invalid rather than defaulting to either party.

A number is only comparable if it says what it measured, so a trace score is not
a verdict with a label. It names every gate this contract defines, says whether
that gate applied to the row, and — when a gate failed — which assistant turn a
reader should look at. A gate that does not apply is reported as such rather than
omitted or counted as a pass: a report that silently dropped the ordering gate on
single-call rows could not be told apart from one where ordering was checked.

Two kinds of gate are reported, and the difference decides what a gate rate over
a benchmark means.

*Coverage* gates ask whether the whole gold trace was requested. Gold calls in
turns the episode never reached count against them, because a model that stopped
early did not ask for them.

- `tool_selection` — every gold call was requested, and nothing the trace does
  not ask for was called.
- `arguments` — every gold call's arguments matched, under the three steps above.

*Consistency* gates ask whether what the model did do was well formed. They are
measured only over the turns that were actually asked, so an episode that stopped
early is not additionally penalized for turns it never saw.

- `schema_valid` — every call the model made satisfies its own declared parameter
  schema. This is reported separately from `arguments` because "the amount was
  wrong" and "the schema forbids this argument at all" are different failures.
  Declared defaults are deliberately *not* filled in before this check: a
  parameter that is both defaulted and required is one the caller has to pass,
  and inserting it first would let default insertion launder a missing required
  argument into a pass. Under `canonical_only` there is no schema step, so this
  gate does not apply.
- `call_grouping` — each asked turn issued its gold call group.
- `call_ordering` — the calls that were made respect the row's `call_order`.
  Ordering is judged independently of selection: a turn that called the trace's
  tools in the wrong order fails ordering only, and a turn that called something
  else entirely fails selection only. Attributing a permutation to both gates
  would report that the model got the order wrong when it never got as far as
  having an order to get wrong.
- `text_turn` — each asked turn the trace answers in words was answered in words
  rather than with a call, and contained non-empty plain or structured textual
  content. `null`, whitespace, empty containers, and metadata-only content are
  neither final answers nor intermediate turns. Under debug-only
  `intermediate_text_matching: verbatim`, an intermediate turn must additionally
  reproduce the recorded sentence.

One gate stands apart and always applies:

- `trace_completion` — the conversation reached the end of the trace. Because
  this gate is never skipped, no unfinished episode can be a success, whatever
  the flags say. An explicit incomplete provider `finish_reason` fails this gate
  even if imported evidence incorrectly claims that the episode completed. An
  episode that ended for a reason that is not the model's
  answer — an unreachable endpoint, a spent episode budget, a turn budget below
  what the trace needs — is additionally marked as such, so a report can separate
  a broken run from a wrong model without softening either one's score.

Every failed gate also says who is answerable for it, and never who earns the
task. A failure is the run's when the episode stopped for a reason the model did
not choose *and* the turn the failure names is a turn the model did not answer:
a turn the episode never sent, or one whose request never came back. Coverage
failures on such a turn are the clearest case — a gold call that was never
requested because the endpoint was unreachable is not weak tool use. Everything
else is the model's own, including a truncated or filtered answer inside an
episode that did reach the end of its trace, and every failure in an episode the
model itself ended. A score whose gates blame the run without a non-candidate
terminal, or one whose non-candidate terminal blames nothing, is refused rather
than published. Because the classification is part of a score's identity, two
reports that disagree about whose failure it was are two different scores.

The trace layer classifies these gates once and executable evaluation carries the
result unchanged. Re-deriving it from the terminal status in the executable scorer
would let the two layers disagree about one piece of evidence.

A trace score projects onto the same error taxonomy an executable one does:
`trace_failure_records` emits one `episode`-layer record for a terminal that is
not `completed` — every incomplete episode fails the same completion gate, so only
the status separates a spent budget from an endpoint that never answered — and one
`gate`-layer record per failed gate, attributed as the scorer attributed it.

These gates roll up into the metric names an exported bundle declares:
`tool_selection` and `arguments` map to themselves, `schema_valid` also reports
under `arguments`, `call_grouping` and `call_ordering` both report under
`call_ordering`, and `text_turn` and `trace_completion` report under
`task_success`. The bundle's remaining metric, `results`, has no trace gate: it
is what oracle replay measures, and a trace score never stands in for one.

A score requires the `EligibleEvalPlan` that authorized the episode. The
episode's plan identity, candidate weights, task, script source, and the scoring
policy's content hash must all match that plan. The score derives the complete
`eval_config_hash` from the plan rather than accepting a caller-provided value,
so evidence cannot be restamped as coming from another benchmark or evaluation
configuration. It cites the script and episode it was derived from, the pinned
policy, and the content hash of this document, and it reads no clock and re-parses
no provider bytes. Re-scoring the same recorded episode under the same authorized
rules therefore reproduces the same score identity, which is what makes a
published number auditable after the endpoint it came from is gone.

Every gate carries a stable `reason_code`. Human-readable `detail` explains the
verdict but is excluded from `score_hash`, as are call- and turn-level diagnostic
strings. Rewording a diagnosis without changing evidence, reason codes, or
structural verdicts does not create a different score. Each record excludes only
its own diagnostic wording; a declared constraint or argument value is not
diagnostic prose because a schema author named it `detail`, so reported evidence
stays inside the score identity.

## What a trace-only run publishes

A trace score is a set of per-task gate verdicts, so a run-level number over them
is a rate of tasks, not a rate of calls. `aggregate_trace_scores(...)` therefore
reports its own taxonomy under trace aggregation contract `1.0`: one
`<gate>_pass_rate` for every gate this document defines, in that order, plus
`task_success_rate`. The names are deliberately not the executable ones —
`arguments_pass_rate` counts tasks whose argument gate passed while
`argument_accuracy` counts calls whose arguments matched, and publishing one
under the other's name would make two incomparable numbers look like the same
measurement. The taxonomy is derived from the gate list rather than restated, so
a new gate cannot be scored without a published metric.

A gate that did not apply to a task is left out of that metric's denominator
rather than counted as a pass, exactly as the per-task contract records it. A
metric no task could apply is `value: null` with the stable
`metric.no_applicable_task` code instead of a vacuous zero or one. As with an
executable aggregate, the counts and reason code carry identity while the
quotient stays derived, and the aggregate requires the exact task ids and
publication order `EligibleEvalPlan` authorized: partial, duplicated, reordered,
or cross-candidate, cross-plan, cross-policy, and cross-scoring-contract inputs
are refused rather than averaged.

`run_bfcl_trace_eval(...)` runs the trace-only batch, and
`run_declared_eval_sync(...)` dispatches on the pinned `eval.mode` so an operator
does not restate the mode by choosing a function. Trace batching shares the
executable run's scaffolding: the same run-identity rules, source verification,
contamination gate, plan recheck, sequential candidates in plan alias order,
tasks in publication order with at most `limits.max_parallel_tasks` in flight,
and sibling cancellation on the first raised error. It opens no oracle session
and persists no tool-trace cache, because a released tool result is benchmark
bytes that source verification already hashed by content. The candidate I/O cache
is therefore the whole of a trace run's replay evidence and must prove that no
claimed request was left without a completion.

`write_trace_eval_artifacts(...)` publishes the same immutable `eval_report.json`,
`eval_task_results.parquet`, and `eval_manifest.json` under artifact contract
`1.5`, with oracle-, assertion-, milestone-, and final-answer-only columns null.
Every candidate aggregate declares the scope it measured — `trace` or
`trace_and_executable` — the report stamps it both at run and candidate level and
the manifest stamps `eval_scope`. Public report and writer entry points revalidate
candidate coverage, scope, model, source, plan, policy, scoring-contract, task-score
hashes, and executable oracle identity; deserialized or restamped evidence cannot
bypass the aggregator merely because the batch runner normally supplies it. A
report whose aggregates mix scopes is refused. A trace-only artifact set never
stands in for an executable one, and an executable config handed to the trace
runner is refused rather than measured with fewer gates.

## Private held-out generalization

`eval.mode: [held_out_eval]` enables held-out contract `1.0`. It is executable
evaluation, so it requires the frozen, hash-verified source publication and the
exact oracle pack. Its `held_out_eval` section pins the normalized `held_out.yaml`
policy hash, the canonical fixture references, the template ids, the policy seed,
the pack version, and a deterministic per-template cap.

The three selection cases are explicit rather than inferred. A fixture-only policy
uses ordinary templates and requires at least one declared held-out fixture
binding. A template-only policy uses the declared templates while excluding
held-out fixture rows. A policy declaring both evaluates the set union with
duplicates removed. Private tasks receive full SHA-256 ids, and the ordered slice
receives its own content hash, so a private run can be cited without naming its
tasks.

For a private fixture slice the evaluator deliberately opens the full verified
fixture inventory, even when the public runtime policy used
`fixtures_in_backend_state: false`. Without that override the run would measure
`not_found` behavior instead of held-out generalization. The override exists only
in the ephemeral private process and never alters the pack or the publication.

Held-out mode may publish only aggregate diagnostics. `outputs.write_task_results`,
`outputs.cache_candidate_responses`, and `outputs.cache_tool_results` must all be
false, and the loader refuses any configuration that could persist private tasks,
prompts, tool traces, or candidate responses. The private slice always uses its
own cache under the ephemeral private directory; it never reuses a seen or
durable cache, even if an internal caller bypasses config validation. The
aggregate report carries the seen
and private success rates, 95% Wilson intervals for each, matched
applicable-tool and turn-policy strata, and
`held_out_generalization_gap = seen_success_rate - held_out_success_rate` with a
conservative Newcombe 95% interval. Reporting the gap with an interval rather than
as a point estimate is what keeps a small private slice from reading as a
measured regression.

## NeMo Evaluator native adapter

Native adapter contract `1.2` targets `nemo-evaluator==0.2.8` and
`nemo-evaluator-launcher==0.2.6`. One `NemoNativeAdapterConfig` binds the
six-file `nemo_evaluator_bundle` tree hash, resolved BFCL eval config, candidate
alias, native output location, and whether the run probes the oracle pack.
Version `1.2` adds that `probe_oracle` field: a Launcher task is submitted once,
so an orchestrator that could not state the probe choice would silently probe
against an operator who declared otherwise. It is the sole source of truth for
both CLI and programmatic runs; the legacy `--no-probe-oracle` flag is accepted
only when it agrees with the immutable config. A Launcher
task represents exactly one candidate; candidate sets are not silently collapsed
into one deployment result. Before inference, the adapter verifies the exact file
set, tree and dataset hashes, every dataset row, generated JSON schema,
descriptor, metadata, prompt catalog, evaluator YAML, package versions, Launcher
endpoint/model, verified BFCL source hash, and authorized task order.

`read_native_bundle_tree(...)` and `native_bundle_tree_hash(...)` are the single
definition of what the bundle tree is and how it is digested, so an orchestrator
that must digest a bundle before it can name that digest in a config cannot
accept a tree this verifier would reject. `native_framework_distribution(...)`
likewise names the distribution `install_native_framework(...)` builds, so a
submitter verifies that exact name rather than reconstructing it.

The framework command calls `run_nemo_native_adapter(...)`, which pre-authorizes
the source and then selects `run_bfcl_eval(...)` or `run_bfcl_trace_eval(...)`
with that exact authorization. The generic NeMo BYOB single-turn strategy and
scalar mean reducer do not define BFCL conversation or metric semantics. This
keeps dependent calls, scripted
multiturns, native tool payloads, released tool results, executable resource
isolation, trace/executable scoring, error attribution, bounded concurrency, and
immutable BFCL artifacts on their established contracts.

After BFCL aggregation, every metric with a positive denominator is projected
into NeMo `EvaluationResult` with its exact binary count, sum, squared sum, mean,
population variance, standard deviation, and standard error. A zero-denominator
metric cannot be represented by NeMo's required floating score value, so it is
omitted from that score map and recorded by metric name and stable N/A reason in
`nemo_native_adapter_manifest.json`. The manifest also binds the adapter config,
bundle, package versions, candidate, served model id and URL, canonical model
identity, run id, scope, aggregate hash, native result, and BFCL report. Existing
output is accepted only when byte-identical.

`native_framework_definition(...)` and `install_native_framework(...)` build an
immutable NeMo namespace package without mutating global site-packages;
`launcher_task_entry(...)` validates the task entry against the pinned Launcher
model in an isolated process. Building writes no `.pth` and changes nothing
already installed, so the generated package must be installed explicitly in the
Launcher environment or baked into its image; a build is not an install. The two
dependencies sit on opposite sides of that boundary: the evaluation container
needs `nemo-evaluator`, while `nemo-evaluator-launcher` remains a host and
control-plane dependency. The evaluation image and mounts
must provide the Nemotron package and every absolute path named by the adapter
and BFCL configs. Missing mounts or a moving package/API version are setup
failures, not candidate failures.

## Nemotron CLI orchestration

CLI orchestration contract `1.0` is an operational envelope around the resolved
eval config; `execution_backend`, output rendering, dry-run, Launcher paths, and
submission state never enter `eval_config_hash`. `stage=eval` dispatches through
the optional family `evaluate` hook, so unsupported benchmark families fail
explicitly instead of inheriting BFCL behavior. `all` remains generation-only.

The `direct` backend delegates to `run_declared_eval_sync(...)`, which performs
the one authoritative load of the pinned eval config; the CLI does not pre-load
it for the executing path, so the file cannot change between the CLI's view of it
and the authorized run. Its dry-run verifies source and contamination and reports
authorized task counts without contacting a candidate. The `nemo_launcher`
backend requires one candidate, verifies the exact exported bundle tree,
materializes an immutable native adapter config, framework package, Launcher
task, and merged `eval/model_eval` config, then optionally submits it through the
same `launch_model_eval_config(...)` API used by the generic step. Submission
requires the exact generated framework distribution, named by
`native_framework_distribution(...)` rather than reconstructed from the build
directory, to have been installed explicitly in the Launcher environment.

Every orchestration artifact — adapter config, framework build, Launcher task,
merged Launcher config, and both output trees — must lie outside the bundle,
because the bundle is verified by exact file set and one extra file would fail
the next verification of that publication. If the merged Launcher config names an
evaluation container anywhere, whether through the CLI override, a global base
config, or the task itself, the CLI requires explicit `evaluation_mounts` and
merges them into Launcher `execution.mounts.evaluation`. They must be identity
mounts and must cover every adapter/eval config, source publication, executable
oracle resource, and output path, because those immutable contracts contain
absolute paths; the bundle's dataset mount remains separate. A base config that
deploys its own endpoint keeps its `deployment` block and receives no pinned
candidate URL or model id, as Launcher 0.2.6 rejects both for a managed
deployment. The adapter's `launcher` target binding revalidates the URL and
requires the served model id Launcher supplies to equal the candidate's declared
served model. A differing id is refused before authorization rather than scored
under the pinned weight and contamination identity. The runtime URL may move to
the managed endpoint, and the native manifest records both effective route fields
alongside the canonical model identity.

CLI failures retain registered BFCL taxonomy codes, and every code in the error
taxonomy is assigned one published process exit status: `2` for a declaration the
operator has to edit, `3` for setup, source, scoring, or aggregation refusal, `4`
for contamination and answer-key exposure, `5` for candidate-endpoint failure,
`6` for live oracle or assertion infrastructure, and `7` for an immutable
artifact that already holds different evidence. The assignment is explicit per
code and checked against the taxonomy at import, so registering a new code is a
build failure rather than a silent reclassification. Projecting a finished run
into CLI output happens inside the same guard, so no failure reaches an operator
as a bare traceback. Human output is line-oriented with JSON-rendered values
while JSON output is stable and contains run identity and artifact paths, never
credentials.

## Task success

`task_success: all_applicable_gates` (*pinned*)

A task counts as a success only when every gate that applies to it passes: the
gates above and — in `executable` mode — oracle replay plus every success
assertion the pack declared for the row. A gate that does not apply to a row (a
single-call row has no ordering gate) is not counted for or against it.

`assertions_only` scores just the pack assertions. It answers a different
question ("did the end state come out right, however the model got there") and is
not publishable as a function-calling score. A trace score has no oracle to
evaluate assertions with, so asking for that mode there is refused rather than
approximated by the trace gates.

## Repair

`allow_llm_repair: false` (*pinned*)

No model — including a judge — may rewrite, complete, or reinterpret a candidate's
tool call before it is scored. Repair turns a wrong answer into a right one and
makes the score a property of the repairer rather than the candidate.

## Determinism

Every decoding parameter is pinned by the config (`temperature`, `top_p`,
`max_tokens`, `seed`, `tool_choice`), and every runtime limit is pinned by
`limits`. A truncated or timed-out response is a failed task, not a skipped one:
silently dropping it would let a slow model score higher than a fast wrong one.

Candidate identity is weight identity. `provider` and `model` name the route a
request took; a 40–64 hexadecimal commit in `model_identity.revision` or a
`weights_digest` names what answered. Branches and tags are not accepted as
immutable merely because their current names are absent from a denylist. Model
and revision case is preserved for registries where case is meaningful. Scores
are only comparable across runs that pin the same weights.

Two things are then possible, and the config says which one it is rather than
requiring the stronger one everywhere. A candidate that pins a commit or a digest
is `weights_pinned`. A candidate that leaves both null is `provider_managed`: it
names a route the provider may re-point, its `canonical_id` ends in
`@provider_managed` so no consumer can read it as a pin, and the run appears in
`non_publication_reasons` as `candidates[<alias>].model_identity`. Requesting
publication for such a run is refused when the config loads, before anything is
scored. This is the honest shape for a hosted frontier model that publishes
neither value: the alternative — a digest derived from the model name — would
identify the string rather than the weights, and would go on claiming
reproducibility after the provider replaced them. A *stated* reference that can
move is still refused outright, because it claims a pin it does not have.

An unpinned identity must restate the candidate's own `provider` and `model`.
With no pin, `source` and `model` are the whole canonical identity and they are
free text, so two candidates on one endpoint could otherwise be spelled two ways
and slip past the duplicate check that exists to stop exactly that. Requiring the
route makes the collision surface where it belongs.

A `weights_digest` names its scheme, because the two schemes measure different
things. `sha256:<64 hex>` covers weight bytes, so two unequal ones are two sets of
weights. `bfcl-weight-manifest-v1:<64 hex>` covers a manifest of every file served
in a weights directory, down to the names: two equal ones are the same directory,
but two unequal ones may be the same weights with a note added beside them. So the
gate settles a manifest comparison only when it is equal, and reports every other
manifest comparison — unequal, or against a `sha256:` digest — as unresolved. That
costs an operator a pinned identity; the alternative would clear a candidate that
answered from the weights that wrote the rows, on the strength of a README.
Schema 1.1 reads only `sha256:` and requires every candidate to pin; both
widenings arrived in 1.2, and a config that declares 1.1 is still held to 1.1.

`python -m nemotron.steps.byob.scripts.resolve_bfcl_model_identity` produces the
block rather than leaving an operator to find it: `registry` resolves a branch or
tag to the commit it currently names (only for registries this build has a client
for, so no source is answered with another registry's commit), `local` digests
weights on disk through a file manifest, and `provider-managed` records the
unpinnable route together with what it costs. Its `identity_publication_gate`
field reports that one gate only; scoring, contamination, artifact, and source
gates are checked when the config loads.

## Contamination

`contamination.enforce: true`, `on_violation: fail_run`,
`comparison_set: common_intersection` (*pinned*)

Held-out material is not a scoring dimension; it is a validity precondition. When
a candidate is found to have seen a task, the run fails rather than quietly
dropping the row, and all candidates are scored on the same task set so a
difference between two numbers cannot come from a difference in which rows each
one answered.

**What counts as having seen a task.** A benchmark records every model that read
a published row while it was being built: the model that profiled the reference
style, the one that paraphrased the surface, the one that judged surface quality,
and the one that translated the rows when a translation is scored. Each is
recorded with the rows it actually read, because scope matters — a paraphraser
that wrote three rows of fifty exposed three rows, not the benchmark. A
publication that cannot state this inventory is not scorable: a missing role
would read as "no contamination found".

**What counts as being the same model.** A candidate is compared against each
exposed model on the axes the two configs carry, strongest evidence first: two
weights digests settle it in either direction, then an equal operator canonical
id, then an equal serving route, then a normalized model name together with a
revision. Digests settle it only when they measure the same thing — identical
weights hash differently under two algorithms, so digests that disagree about
their algorithm are set aside instead of being read as two different models,
while the same digest recorded with and without its `sha256:` prefix is still one
digest. Model names are compared case-insensitively and with the registry
prefix and punctuation removed, so `meta/Llama-3.3-70B` and
`meta-llama/llama_3.3_70b` are not treated as different models. This
over-matching is deliberate: it turns a comparison that would have read
"different" into "unresolved", which asks the operator for a pinned identity
instead of clearing a candidate that may have written the rows.

**What happens when the comparison cannot settle it.** An unresolved comparison
is neither a violation nor a clearance. It never shrinks a task set on suspicion,
and it never aborts a debug run, but it always blocks publication: a published
score asserts that the candidate had not seen these rows, and an unresolved
comparison cannot support that assertion. Pinning `weights_digest` on both sides
resolves it.

Every candidate is compared before any candidate is refused, so one run names all
of the collisions an operator has to resolve rather than surfacing them one rerun
at a time.

**What the score is then taken over.** Under the pinned settings, a publishable
run has no collision at all — the alternative is refusal, not adjustment. The
debug paths are `exclude_row`, which drops exactly the exposed rows and records
the reduced set as `contamination.excluded_rows:<alias>`, and `per_candidate`,
which lets each candidate keep its own eligible rows. Both are reported with the
task-id hash of what was actually scored, and neither is publishable: a number
over a task set chosen per candidate is not comparable to a number over the
benchmark.
