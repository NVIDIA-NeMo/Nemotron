# BFCL authoring cache retention

The retention tool applies only to `authoring_io_cache.jsonl`. Generation and evaluation
caches have separate checkpoint and replay contracts and are deliberately rejected.
Behavior is verified by
[`test_bfcl_authoring_retention.py`](../../../../../tests/steps/byob/test_bfcl_authoring_retention.py).

<!-- doc-smoke: bfcl-author-purge-help -->
```shell
python -m nemotron.steps.byob.scripts.bfcl_author purge-cache --help
```

Use the guided command in dry-run mode first:

The following is an operator template, not an executable smoke command:

```text
python -m nemotron.steps.byob.scripts.bfcl_author --ci purge-cache \
  --workspace WORKSPACE --tenant-id TENANT --run-id RUN \
  --actor RETENTION_BOT --reason-code retention_expired
```

Dry-run writes a digest-bound audit record but does not modify the cache. To prevent a
time-of-check/time-of-use purge, execute the exact plan by supplying its digest:

```text
python -m nemotron.steps.byob.scripts.bfcl_author --ci purge-cache \
  --workspace WORKSPACE --tenant-id TENANT --run-id RUN \
  --actor RETENTION_BOT --reason-code retention_expired \
  --execute --expected-plan-digest sha256:...
```

The tool acquires the same tenant/run workspace lock as guided authoring. It verifies all
immutable session records and bound draft provenance, then retains every referenced
`request_hash`. If an active head has not committed draft provenance, the entire cache is
protected. A changed cache or session set makes the supplied plan stale.

Execution rewrites retained validated records to a mode-`0600` temporary file, fsyncs it,
atomically replaces the cache, and fsyncs the parent directory. Cache paths outside the
workspace, symlinks, non-private files, and cache kinds other than authoring are refused.

Audit records are appended to `.events/cache_purge_audit.jsonl`. They contain only
request hashes, counts, file/plan digests, actor, and a stable reason code. Model inputs
and responses are never copied into an audit record.
