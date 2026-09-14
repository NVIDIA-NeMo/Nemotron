# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validated, content-addressed structured authoring calls."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from nemotron.steps.byob.runtime.authoring_workflow.quota import (
    RunQuota,
    estimate_structured_token_units,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.model_io_cache import (
    ImmutableModelIOCache,
    request_hash,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json
from nemotron.steps.byob.runtime.pack_authoring.prompts import prompt_hash

StructuredCaller = Callable[..., dict[str, dict[str, Any]]]


class AuthoringModelError(Exception):
    """Raised when the authoring model cannot be used or did not answer."""


@dataclass(frozen=True)
class AuthoringModel:
    """The model identity and settings an authoring run is pinned to."""

    alias: str
    provider: str
    model: str
    canonical_id: str
    seed: int = 0
    inference_parameters: Mapping[str, Any] = field(default_factory=dict)
    # How long one call may take, which is not a decoding setting: it cannot change the
    # answer, only whether one arrives. It is kept out of `inference_parameters` for that
    # reason, because those feed the request hash and the published provenance, and a
    # deadline raised for a slow route would otherwise orphan every approved draft.
    request_timeout_s: int | None = None

    def as_model_config(self) -> dict[str, Any]:
        parameters = dict(self.inference_parameters)
        if self.request_timeout_s is not None:
            parameters["timeout"] = self.request_timeout_s
        return {
            "alias": self.alias,
            "provider": self.provider,
            "model": self.model,
            "canonical_id": self.canonical_id,
            "inference_parameters": parameters,
        }

    def as_provenance(self) -> dict[str, Any]:
        # No credentials, no endpoint: identity and settings only, so the record can be
        # published next to the pack.
        return {
            "alias": self.alias,
            "provider": self.provider,
            "model": self.model,
            "canonical_id": self.canonical_id.strip().lower(),
            "seed": self.seed,
            "inference_parameters": dict(self.inference_parameters),
        }


@dataclass(frozen=True)
class ModelCallRecord:
    """What was asked, of which model, and whether the answer was already on disk."""

    stage: str
    prompt_version: str
    prompt_hash: str
    request_hash: str
    input_hash: str
    output_schema_hash: str
    model_canonical: str
    served_from_cache: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash,
            "request_hash": self.request_hash,
            "input_hash": self.input_hash,
            "output_schema_hash": self.output_schema_hash,
            "model_canonical": self.model_canonical,
            "served_from_cache": self.served_from_cache,
        }


def _default_caller(
    run_dir: Path,
    *,
    stage_name: str,
    model_config: dict[str, Any],
    requests: Sequence[dict[str, str]],
    system_prompt: str,
    prompt: str,
    output_format: type[BaseModel],
) -> dict[str, dict[str, Any]]:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.model_runner import (
        run_structured_model,
    )

    # The adapter only needs somewhere to put its seed file and artifacts, so an authoring
    # run supplies that directly rather than inventing a pack-shaped pipeline config.
    class _RunDir:
        output_dir = str(run_dir.parent)
        expt_name = run_dir.name

    return run_structured_model(
        _RunDir(),  # type: ignore[arg-type]
        stage_name=stage_name,
        model_config=model_config,
        requests=requests,
        system_prompt=system_prompt,
        prompt=prompt,
        output_format=output_format,
    )


def call_structured(
    model: AuthoringModel,
    *,
    stage_name: str,
    prompt_version: str,
    system_prompt: str,
    prompt: str,
    columns: Mapping[str, str],
    output_format: type[BaseModel],
    cache: ImmutableModelIOCache,
    run_dir: Path,
    caller: StructuredCaller | None = None,
    quota: RunQuota | None = None,
) -> tuple[dict[str, Any], ModelCallRecord]:
    """Return one validated structured response, from cache when it is already known."""
    schema = output_format.model_json_schema()
    model_input = dict(sorted(columns.items()))
    hashed_prompt = prompt_hash(prompt_version, system_prompt, prompt)
    key = request_hash(
        model_canonical=model.canonical_id,
        prompt_hash=hashed_prompt,
        model_input=model_input,
        inference_parameters=dict(model.inference_parameters),
        output_schema=schema,
        seed=model.seed,
    )
    input_hash = sha256_json(model_input)
    record = ModelCallRecord(
        stage=stage_name,
        prompt_version=prompt_version,
        prompt_hash=hashed_prompt,
        request_hash=key,
        input_hash=input_hash,
        output_schema_hash=sha256_json(schema),
        model_canonical=model.canonical_id.strip().lower(),
        served_from_cache=True,
    )

    cached = cache.get(key)
    if cached is not None:
        try:
            validated = output_format.model_validate(cached).model_dump(mode="json")
        except ValidationError:
            # Old clients could persist a schema-invalid provider draft. Remove only the
            # exact invalid observation so the same request can recover on this run.
            cache.invalidate(key, expected_response=cached)
        else:
            if quota is not None:
                quota.record_cache_hit(batch_size=1)
            return validated, record

    if quota is not None:
        quota.reserve_provider_call(
            token_units=estimate_structured_token_units(
                model_canonical=model.canonical_id,
                system_prompt=system_prompt,
                prompt=prompt,
                model_input=model_input,
                inference_parameters=model.inference_parameters,
                output_schema=schema,
            ),
            batch_size=1,
        )
    invoke = caller if caller is not None else _default_caller
    responses = invoke(
        run_dir,
        stage_name=stage_name,
        model_config=model.as_model_config(),
        requests=[{"request_id": stage_name, **model_input}],
        system_prompt=system_prompt,
        prompt=prompt,
        output_format=output_format,
    )
    response = responses.get(stage_name)
    if not isinstance(response, dict):
        raise AuthoringModelError(f"{stage_name} returned no structured response for request {stage_name!r}")
    if quota is not None:
        quota.check_after_provider_call()
    try:
        validated = output_format.model_validate(response).model_dump(mode="json")
    except ValidationError as exc:
        raise AuthoringModelError(f"{stage_name} returned a response outside the requested output schema") from exc
    cache.put(
        key,
        validated,
        model_canonical=model.canonical_id,
        input_hash=input_hash,
    )
    return validated, replace(record, served_from_cache=False)
