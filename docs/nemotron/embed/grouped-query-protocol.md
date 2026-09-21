# Grouped query-disjoint retrieval over a fixed collection

The multimodal recipe and its retrieval-SDG producer hold out query groups
while keeping the searchable collection fixed. Documents may appear in training
positives and evaluation positives;
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
the manifest decides. This feature supports only grouped query-disjoint bundles.
Document-partition bundles must be re-exported, never silently reinterpreted.
A protocol override without a portable input is rejected.
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
pages. Text admits units with nonblank text; image views admit units with exactly one image.
Every training, validation and evaluation query records `query_group_id`.

`split_manifest.json` declares:

```json
{
  "split_protocol": "grouped_query_disjoint",
  "corpus_scope": "full_collection",
  "document_disjoint": false,
  "query_assignments": {
    "image_and_text": {
      "q1": {"split": "train", "query_group_id": "group-1"},
      "q2": {"split": "evaluation", "query_group_id": "group-2"}
    }
  }
}
```

Assignments must exactly match each exported view's queries. No query group or
normalized duplicate query may cross partitions. The producer must form groups
before splitting and positive unrolling, linking known translations, paraphrases
and rewrites. Validation cannot infer unknown semantic relationships from opaque
group IDs. Sharing a document alone does not imply sharing a query group.
Multi-page qrels are preserved, never collapsed to one positive. Binary labels
are sufficient; graded evaluation labels are optional and stay in qrels,
not in optimizer inputs. HNM uses AutoModel unchanged and does not need query
groups after the split has been validated.

Producer input group IDs are optional. Without provenance, start each unique
query in its own group and join normalized duplicates before partitioning.
Preserve caller-provided parent/seed IDs across translations and rewrites.
Lexical near-duplicate detection is optional; unknown semantic equivalents
cannot be guaranteed. Sharing a document, page, summary or generation context
does not automatically join queries. Dataset-specific grouping belongs in
experiment adapters, not this consumer or the general producer.

The public DataDesignerPlugins retrieval-first producer emits this contract;
see the [multimodal embedding EA guide](multimodal-ea.md) for the integrated
source-to-evaluation workflow. Other producers must implement the same export
and validation rules. Synthetic qrels are not independent gold labels. Keep a
separate independent-query evaluation when measuring generalization beyond
synthetic-query style.

## Precision and mining

This protocol does not change the training optimizer or checkpoint dtype.
The default VL configuration uses FP32 resident parameters and FlashAdamW
optimizer states, with BF16 forward/backward computation under FSDP2.
FlashAdamW stores unquantized states in the parameter dtype, so FP32 parameter
storage is required to keep the moments in FP32. Resumable training checkpoints
retain FP32 parameters and optimizer states; consolidated model exports remain
BF16 for BF16 source models.

Mining continues to use AutoModel's own positive exclusion and negative selection.
No post-mining annotation-restoration layer is required: retain the source bundle
for provenance and evaluation labels. Query groups govern the split, not mining.
