# Grouped query-disjoint retrieval over a fixed collection

Stacked on the multimodal recipe implementation. This consumer extension lets
a producer hold out query families while keeping the searchable collection
fixed. Documents may appear in training positives and evaluation positives;
that is intentional for in-domain adaptation, not an unseen-document claim.

## Recipe configuration

For Stage 1 (`nemotron embed prep`) and Stage 3 (`nemotron embed eval`), set:

```yaml
sdg_input_path: /data/run/generation_result.json
retrieval_view: image_and_text
retrieval_split_protocol: grouped_query_disjoint
```

The optional `retrieval_split_protocol` is an assertion against the producer's
manifest, not permission to reinterpret an existing document split. If omitted,
the manifest decides. Existing schema-v2 document-disjoint bundles remain
supported unchanged. A protocol override without a portable input is rejected.
Stage 1 mines only exported training queries. Stage 3 uses the exported full
collection, including pages that have no held-out positive label. Base and FT
must each encode both queries and corpus with their own checkpoint.

## Producer contract (schema version 2)

`run_manifest.json` declares `split_protocol: grouped_query_disjoint` and
`corpus_scope: full_collection`, alongside the existing checksummed artifacts.
`retrieval_units.jsonl` retains authoritative text, assets and document IDs,
but assigns each unit `split: shared`. All existing inventory, containment,
source fidelity, positive-only and partial-judgment checks still apply.

Each view writes one `views/<view>/corpus/shared/` Parquet dataset and loader
metadata. Both `train.json` and `validation.json` reference `corpus/shared`.
No validation is required: use `data: []`. The synthetic BEIR corpus contains
exactly the same full view-eligible unit set; it must not be limited to positive
pages. Text admits all units; image views admit units with exactly one image.
Every training, validation and evaluation query records `query_group_id`.

`split_manifest.json` declares:

```json
{
  "split_protocol": "grouped_query_disjoint",
  "corpus_scope": "full_collection",
  "document_disjoint": false,
  "query_assignments": {
    "image_and_text": {
      "q1": {"split": "train", "query_group_id": "family-1"},
      "q2": {"split": "evaluation", "query_group_id": "family-2"}
    }
  }
}
```

Assignments must exactly match each exported view's queries. No family or
normalized duplicate query may cross partitions. The producer must form families
before splitting and positive unrolling, linking known translations, paraphrases
and rewrites. Validation cannot infer unknown semantic relationships from opaque
family IDs. Sharing a document alone does not imply sharing a query family.
Graded multi-page qrels are preserved, never collapsed to one positive.

This PR adds the **consumer contract**, not a new SDG generator or a default
VLM. Producers must explicitly implement this export. It does not retrofit
previous preconverted-input experiments, and it does not assert that synthetic
qrels are independent gold labels. Keep a separate independent-query evaluation
when measuring generalization beyond synthetic-query style.
