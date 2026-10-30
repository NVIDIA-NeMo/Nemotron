---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Use backend=nmt with nemotron steps run translate/nemo_curator and a local HTTP translation service."
topics: ["Translation", "NMT"]
tags: ["How-To", "Translation"]
content:
  type: "How-To"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Run NMT Translation

Use this guide when translation should run against `backend: nmt` and your own HTTP microservice that performs *neural machine translation (NMT)*.

## Prerequisites

- A running NMT service reachable at the URL you pass as `nmt.server_url`, implementing the contract in the next section.
- If `faith_eval.enabled` is `true`, LLM credentials for FAITH remain required; see {doc}`run-faith-evaluation`.

## Service Contract

Your service must implement one endpoint.

`POST /translate` — request body:

```json
{
  "texts": ["Hello, world.", "How are you?"],
  "src_lang": "en",
  "tgt_lang": "hi"
}
```

`texts` is a list of strings (one per segment). `src_lang` and `tgt_lang` are BCP 47 / ISO 639-1 codes; Nemotron lowercases them before the call.

Success response body:

```json
{
  "translations": ["नमस्ते, दुनिया।", "आप कैसे हैं?"]
}
```

`translations` must contain exactly as many strings as `texts`. Curator raises a `RuntimeError` on a count mismatch.

On error, return a non-2xx HTTP status with a JSON body containing an `"error"` field. Curator retries with exponential backoff and surfaces unrecoverable failures as an `aiohttp.ClientResponseError`. Streaming is not supported — return the full batch in a single response.

Curator also sends `GET /health` at startup as an optional liveness check. Return any 2xx to pass; if the endpoint is absent, Curator logs a warning and continues.

Tune `nmt.batch_size`, `nmt.timeout`, and `nmt.max_concurrent_requests` once you understand your server's throughput.

## Procedure

```bash
uv run nemotron steps run translate/nemo_curator -c default \
  backend=nmt \
  nmt.server_url=http://localhost:5000 \
  input_path=/path/to/chat.jsonl \
  output_dir=/path/to/out \
  source_language=en \
  target_language=hi
```

## FAITH Considerations

If `faith_eval.enabled` stays `true` as in the default `default.yaml`, you must still supply large language model (LLM) credentials through `server` because FAITH scoring uses that client even though translation runs on NMT.

## Related Pages

- Field wiring: {doc}`configure-fields-and-output`
- YAML reference for the `nmt` block: {doc}`../reference/translate-config`
