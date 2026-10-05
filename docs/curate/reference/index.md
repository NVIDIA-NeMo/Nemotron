---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Reference pages for the curation steps, the curate flow, and the filter policy schema."
topics: ["Curation", "Reference"]
tags: ["Reference", "Curation", "NeMo Curator"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Data Scientist"]
---

# Curation Reference

Use these pages to look up command syntax, configuration fields, artifact formats, and error identifiers.

## Commands and Formats

```{toctree}
:maxdepth: 1

cli-curate
io-format
troubleshooting
```

## Steps

One page per registered step, in the order the flow runs them.

```{toctree}
:maxdepth: 1

ingest
profile
curate/nemo_curator <curate-config>
audit
decontamination
subset
```

## Flow and Policy

```{toctree}
:maxdepth: 1

flow-config
policy-file
```
