# BFCL authoring structured events

BFCL guided authoring writes an append-only event stream to
`<workspace>/.events/authoring_events.jsonl`. The stream is operational telemetry; it
does not replace digest-bound workflow and release artifacts.

Every line is a `bfcl-authoring-event-v1` envelope with a verified event digest,
tenant/run namespace, optional session digest, and exactly one strict payload from
`runtime/authoring_workflow/events.py`. The supported event types are:

- `adapter_identity_bound`
- `certification_verified`
- `refusal_recorded`
- `revision_authorized`
- `validation_verdict`
- `release_frozen`

Payloads are constructed from an allowlist. They contain adapter names, certification
tiers, stable reason codes, verdicts, and artifact digests only. Source subjects,
domain-brief prose, fixture values, credentials, validation details, model prompts,
and model responses are not accepted payload fields. Operator identity and free-form
error text are also excluded.

The JSONL sink creates files with mode `0600`, serializes writes under an advisory
lock, and fsyncs each complete event. Consumers must use `load_authoring_events`;
unknown fields, duplicate JSON keys, malformed records, and digest drift fail closed.

Failure events use stable codes only. Detailed recovery guidance remains in the CLI
error response and the authoritative refusal/session artifacts. If failure telemetry
cannot be written, it must not replace or obscure the original workflow error.
