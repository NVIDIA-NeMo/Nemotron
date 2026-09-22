---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Measure an approved policy's false-rejection and noise-removal rates against a labelled document set."
topics: ["Curation", "How-To", "Evaluation"]
tags: ["How-To", "Curation", "Policy"]
content:
  type: "How-To"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Evaluate a Policy

Use this guide to measure what an approved policy does to documents a person has labelled.
Retention curves from profiling state how much of a corpus a threshold keeps; they do not state whether the removed documents were the ones that should have been removed.
The evaluation script answers that second question.

The script runs without NeMo Curator, so it works in CI and on a workstation.

## Prerequisites

- An `approved_policy.yaml` that passes the approval schema checks, including the required approval fields and valid thresholds.
- The language pack the policy was calibrated against. The policy records the pack identity but not the pack location, so `--langpack-dir` is required unless the policy's `langpack` block names one.
- A labelled set in JSONL.

Evaluation checks the policy's structure, not whether it is compatible with a particular run.
When `curate/nemo_curator` applies the policy, it compares `signals_impl_version` with the installed scorer implementation.
It also compares `corpus.fingerprint` with the input corpus.
A policy can pass the schema check and still fail either compatibility check during application.

## Write the Labelled Set

Each line is a JSON object with four required fields:

| Field | Values |
| --- | --- |
| `id` | Unique within the set |
| `text` | The document text |
| `label` | `keep` or `drop` |
| `phenomenon` | One of `clean`, `nfd`, `local_digits`, `ocr_noise`, `code_mixing`, `langid_error` |

```json
{"id": "vi-0001", "text": "Hà Nội là thủ đô của Việt Nam.", "label": "keep", "phenomenon": "clean"}
{"id": "vi-0002", "text": "Ha Noi la thu do cua Viet Nam.", "label": "keep", "phenomenon": "nfd"}
{"id": "vi-0003", "text": "H4 N0i l4 thu d0 cu4 V!et N4m", "label": "drop", "phenomenon": "ocr_noise"}
```

The script refuses a set with a missing field, an unknown label or phenomenon, a duplicate `id`, or no records.
A missing label would shrink the denominator silently, and an unknown phenomenon would hide a failure mode from the per-phenomenon table.

## Run the Evaluation

```bash
uv run --extra curate \
  python -m nemotron.steps.curate.nemo_curator.scripts.run_evaluate \
  --policy ./output/vi/policy/approved_policy.yaml \
  --labelled ./eval/vi.jsonl \
  --langpack-dir ./langpacks \
  --report ./output/vi/policy/evaluation.json
```

| Option | Meaning |
| --- | --- |
| `--policy` | Approved policy whose thresholds are evaluated. Required. |
| `--labelled` | One or more labelled JSONL files. Required. |
| `--language` | BCP-47 tag; defaults to the policy's `langpack` block |
| `--langpack-dir` | Pack root; defaults to the policy's `langpack` block |
| `--report` | Write the full report as JSON to this path |

Exit code `0` means the rates were measured; exit code `2` means a problem the user can fix, such as a refused policy or an invalid labelled set.

## Read the Report

Two aggregate rates are printed, and neither is meaningful alone:

- *False rejection*: of the documents labelled `keep`, the fraction the thresholds dropped. A gate that removes nothing scores zero.
- *Noise removal*: of the documents labelled `drop`, the fraction the thresholds dropped. A gate that removes everything scores one.

The per-phenomenon table is where a defect appears.
An aggregate over a mostly clean set can report a good figure while rejecting every OCR-noised document in it.

Two rows carry caveats:

- `langid_error` is marked as not exercised. The harness loads no models, so the row shows how the policy's own signals treat those documents, not whether language-identification errors are handled.
- Signals that wrap a NeMo Curator filter are listed under "not scored" because the harness does not load Curator.

The JSON report also records the policy digest, a digest of each labelled file, and the pack identity, so that a rate can be traced to the exact inputs that produced it.

## Interpret the Result

A rate is worth what its labelled set is worth.
A set constructed to exercise the named phenomena demonstrates that the pipeline handles those phenomena; it does not establish a rate for the production corpus.
Record the evaluation report path in the policy's `approval.evidence` text when the evaluation informed the approval.

## Next Steps

- Policy schema: {doc}`../reference/policy-file`.
- Command reference: {doc}`../reference/cli-curate`.
