<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Configure Model Endpoints for BYOB

BYOB does not ship model weights.
Instead, YAML blocks describe OpenAI-compatible clients for each stage that calls a language model.

## Generation and judgement

`generation_model_config` and `judge_model_config` are required mappings.
They follow the same structure you see in `src/nemotron/steps/byob/mcq/config/default.yaml`: `alias`, `model`, `provider`, and nested `inference_parameters` such as `max_tokens`, `max_parallel_requests`, `temperature`, and `top_p`.

`setup_model_config` in `runtime/data_designer_utils.py` reads these blocks when Data Designer runs batched stages.

## Distractor and validity stages

When `do_distractor_expansion` is true, `distractor_expansion_model_config` must be present.
`distractor_validity_model_config` is always required in the current schema validation.

## Filtering swarm

`filtering_model_configs` contains two lists, `hallucination` and `easiness`.
Each list entry needs a unique `alias` across both lists.
The structure mirrors other model blocks so you can aim filters at different endpoints than the generator.

## Translation models

Translation uses `translation_model_config` in `translate.yaml`.
`backend_type`, `params` (including `api_key_env` and `base_url`), and nested `stage` / `segment_stage` keys are passed into Curator’s experimental translation pipeline.

## API keys

Model blocks may set `api_key_env` to read secrets from the environment (for example `NGC_API_KEY` in the sample translation config).
Set those variables in the shell that launches `nemotron steps run`.
