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

"""Truth-preserving localization of immutable BFCL publications."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

import pandas as pd  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BYOB_ROOT
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.source_contract import (
    BFCL_TRANSLATION_CONTRACT_VERSION,
    SOURCE_VERIFICATION_CONTRACT_VERSION,
    TRANSLATION_PRESERVED_FIELDS,
    translation_preserved_projection,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    CanonicalExportRow,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_projection import (
    CanonicalExportProjection,
    ExportProjectionError,
    project_published_benchmark,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.publication_contract import (
    PublicationContractError,
    PublicationPlan,
    verify_written_benchmarks,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import (
    benchmark_schema,
    canonical_json,
)
from nemotron.steps.byob.runtime.config import (
    AVAILABLE_QUALITY_METRICS,
    ByobTranslationConfig,
)
from nemotron.steps.byob.runtime.translation.locales import (
    SUPPORTED_SCRIPTS,
    LocaleTag,
    contains_script,
    expected_script,
    normalize_locale,
)
from nemotron.steps.byob.runtime.translation.quality_metrics import (
    evaluate_text_quality_metrics,
)
from nemotron.steps.byob.runtime.translation.translation import TranslationPipeline

TRANSLATION_CONTRACT_VERSION: Final = BFCL_TRANSLATION_CONTRACT_VERSION
FLATTENING_CONTRACT_VERSION: Final = "bfcl-translation-fields/1.0"
PROTECTION_CONTRACT_VERSION: Final = "bfcl-protected-tokens/1.0"
TRANSLATION_MANIFEST_FILE: Final = "translation_manifest.json"
_PLACEHOLDER = re.compile(r"__BFCL_PROTECTED_[0-9]{6}__")
_PLACEHOLDER_PREFIX = "__BFCL_PROTECTED_"


class BFCLTranslationError(RuntimeError):
    """The source, translation, or localized output failed its contract."""


@dataclass(frozen=True)
class _TranslationConfig:
    shared: ByobTranslationConfig
    source_manifest: Path
    translate_tool_descriptions: bool
    output_dir: Path
    model: dict[str, Any]
    model_config_hash: str
    source_locale: LocaleTag
    target_locale: LocaleTag
    localization: _LocalizationPolicy


@dataclass(frozen=True)
class _LocalizationPolicy:
    normalize_unicode: bool
    minimum_changed_fraction: float
    forbidden_patterns: tuple[str, ...]
    required_script: str | None


@dataclass(frozen=True)
class _Source:
    manifest_path: Path
    manifest_hash: str
    manifest: dict[str, Any]
    benchmark_path: Path
    benchmark_hash: str
    raw_path: Path
    raw_hash: str
    projection: CanonicalExportProjection
    rows: list[dict[str, Any]]


@dataclass(frozen=True)
class _ProtectedValue:
    placeholder: str
    value: str


@dataclass(frozen=True)
class _TranslationUnit:
    translation_id: str
    task_id: str
    path: str
    source_text: str
    protected_text: str
    protected: tuple[_ProtectedValue, ...]
    protected_tokens: tuple[str, ...]

    @property
    def placeholder_order(self) -> tuple[str, ...]:
        return tuple(_PLACEHOLDER.findall(self.protected_text))


def _hash_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _hash_json(value: Any) -> str:
    return _hash_bytes(canonical_json(value).encode("utf-8"))


def _plain_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BFCLTranslationError(f"{field} must be a non-empty string")
    return value.strip()


def _contains_template_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return "REPLACE_ME_" in value
    if isinstance(value, Mapping):
        return any(_contains_template_placeholder(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return any(_contains_template_placeholder(child) for child in value)
    return False


def _resolve_config_path(value: Any, field: str) -> Path:
    path = Path(_plain_text(value, field)).expanduser()
    return (path if path.is_absolute() else BYOB_ROOT / path).resolve()


def _localization_policy(
    document: Mapping[str, Any],
    target_locale: LocaleTag,
) -> _LocalizationPolicy:
    raw = document.get("localization", {})
    if not isinstance(raw, Mapping):
        raise BFCLTranslationError("localization must be a mapping")
    deterministic = raw.get("deterministic_fixes", {})
    if not isinstance(deterministic, Mapping):
        raise BFCLTranslationError("localization.deterministic_fixes must be a mapping")
    normalize_unicode = deterministic.get("normalize_unicode", True)
    if type(normalize_unicode) is not bool:
        raise BFCLTranslationError("localization.deterministic_fixes.normalize_unicode must be true or false")
    validation = raw.get("validation", {})
    if not isinstance(validation, Mapping):
        raise BFCLTranslationError("localization.validation must be a mapping")
    minimum_changed_fraction = validation.get("minimum_changed_fraction", 0.01)
    if (
        not isinstance(minimum_changed_fraction, int | float)
        or isinstance(minimum_changed_fraction, bool)
        or not 0 < float(minimum_changed_fraction) <= 1
    ):
        raise BFCLTranslationError("localization.validation.minimum_changed_fraction must be in (0, 1]")
    guards = raw.get("response_guards", {})
    if not isinstance(guards, Mapping):
        raise BFCLTranslationError("localization.response_guards must be a mapping")
    patterns = guards.get("forbidden_patterns", guards.get("english_patterns", []))
    if not isinstance(patterns, list) or any(not isinstance(pattern, str) or not pattern for pattern in patterns):
        raise BFCLTranslationError("localization.response_guards.forbidden_patterns must be a list of regex strings")
    for index, pattern in enumerate(patterns):
        try:
            re.compile(pattern)
        except re.error as exc:
            raise BFCLTranslationError(f"localization response guard {index} is not valid regex: {exc}") from exc
    required_script = validation.get("required_script", expected_script(target_locale.primary))
    if required_script is not None and required_script not in SUPPORTED_SCRIPTS:
        raise BFCLTranslationError(
            f"localization.validation.required_script must be one of {sorted(SUPPORTED_SCRIPTS)} or null"
        )
    return _LocalizationPolicy(
        normalize_unicode=normalize_unicode,
        minimum_changed_fraction=float(minimum_changed_fraction),
        forbidden_patterns=tuple(patterns),
        required_script=cast(str | None, required_script),
    )


def _load_config(path: str | Path) -> _TranslationConfig:
    config_path = Path(path).resolve()
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise BFCLTranslationError(f"cannot read BFCL translation config: {exc}") from exc
    if not isinstance(document, dict):
        raise BFCLTranslationError("BFCL translation config must be a mapping")
    if document.get("family") != "bfcl" or document.get("stage") != "translate":
        raise BFCLTranslationError("BFCL translation config must set family: bfcl and stage: translate")
    if document.get("config_status") != "resolved":
        raise BFCLTranslationError("BFCL translation config must be resolved before it can contact a model")
    if _contains_template_placeholder(document):
        raise BFCLTranslationError("resolved BFCL translation config still contains REPLACE_ME placeholders")
    if "dataset_path" in document:
        raise BFCLTranslationError(
            "BFCL translation refuses dataset_path; name the published source_run_manifest instead"
        )
    source_manifest = _resolve_config_path(document.get("source_run_manifest"), "source_run_manifest")
    source_language = _plain_text(document.get("source_language"), "source_language")
    target_language = _plain_text(document.get("target_language"), "target_language")
    try:
        source_locale = normalize_locale(source_language)
        target_locale = normalize_locale(target_language)
    except ValueError as exc:
        raise BFCLTranslationError(str(exc)) from exc
    if source_locale.primary == target_locale.primary:
        raise BFCLTranslationError("source_language and target_language must differ")
    model_config = document.get("translation_model_config")
    if not isinstance(model_config, dict):
        raise BFCLTranslationError("translation_model_config must be a mapping")
    params = model_config.get("params")
    if not isinstance(params, dict):
        raise BFCLTranslationError("translation_model_config.params must be a mapping")
    provider = _plain_text(
        params.get("provider") or model_config.get("provider"),
        "translation_model_config.params.provider",
    )
    model_name = _plain_text(
        params.get("model") or params.get("model_name") or model_config.get("model"),
        "translation_model_config.params.model",
    )
    canonical_id = _plain_text(
        params.get("canonical_id") or params.get("alias") or model_name,
        "translation_model_config.params.canonical_id",
    )
    weight_source = _plain_text(params.get("source"), "translation_model_config.params.source")
    revision = params.get("revision")
    weights_digest = params.get("weights_digest")
    if weights_digest is not None and (
        not isinstance(weights_digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", weights_digest)
    ):
        raise BFCLTranslationError("translation_model_config.params.weights_digest must be sha256:<64 hex>")
    if weights_digest is None and (not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision)):
        raise BFCLTranslationError("translator provenance requires weights_digest or a 40-64 hex immutable revision")
    metrics = document.get("backtranslation_quality_metrics")
    if not isinstance(metrics, list) or not metrics:
        raise BFCLTranslationError("backtranslation_quality_metrics must be a non-empty list")
    for index, metric in enumerate(metrics):
        if (
            not isinstance(metric, dict)
            or metric.get("type") not in AVAILABLE_QUALITY_METRICS
            or not isinstance(metric.get("threshold"), int | float)
            or isinstance(metric.get("threshold"), bool)
            or metric["threshold"] < 0
        ):
            raise BFCLTranslationError(f"backtranslation_quality_metrics[{index}] is invalid")
    metric_types = [str(metric["type"]) for metric in metrics]
    if len(metric_types) != len(set(metric_types)):
        raise BFCLTranslationError("backtranslation_quality_metrics cannot repeat a metric type")
    remove_low_quality = document.get("remove_low_quality", False)
    if type(remove_low_quality) is not bool:
        raise BFCLTranslationError("remove_low_quality must be true or false")
    if remove_low_quality:
        raise BFCLTranslationError(
            "BFCL localization cannot remove low-quality rows because task identity "
            "and publication order are immutable"
        )
    translate_tool_descriptions = document.get("translate_tool_descriptions", False)
    if type(translate_tool_descriptions) is not bool:
        raise BFCLTranslationError("translate_tool_descriptions must be true or false")
    output_root = _resolve_config_path(document.get("output_dir"), "output_dir")
    expt_name = _plain_text(document.get("expt_name"), "expt_name")
    if Path(expt_name).name != expt_name:
        raise BFCLTranslationError("expt_name must be a single path component")
    shared = ByobTranslationConfig(
        expt_name=expt_name,
        dataset_path=str(source_manifest),
        output_dir=str(output_root),
        source_language=source_locale.canonical,
        target_language=target_locale.canonical,
        translation_model_config=model_config,
        backtranslation_quality_metrics=metrics,
        remove_low_quality=False,
    )
    model = {
        "provider": provider,
        "model": model_name,
        "canonical_id": canonical_id,
        "source": weight_source,
        "revision": revision,
        "weights_digest": weights_digest,
    }
    return _TranslationConfig(
        shared=shared,
        source_manifest=source_manifest,
        translate_tool_descriptions=translate_tool_descriptions,
        output_dir=output_root / expt_name,
        model=model,
        model_config_hash=_hash_json(model_config),
        source_locale=source_locale,
        target_locale=target_locale,
        localization=_localization_policy(document, target_locale),
    )


def _artifact_path(root: Path, name: Any, field: str) -> Path:
    file_name = _plain_text(name, field)
    if Path(file_name).name != file_name:
        raise BFCLTranslationError(f"{field} must name a file beside run_manifest.json")
    path = root / file_name
    if not path.is_file() or path.is_symlink() or path.resolve() != path:
        raise BFCLTranslationError(f"{field} is missing or unsafe: {path}")
    return path


def _declared_hash(value: Any, field: str) -> str:
    text = _plain_text(value, field)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", text):
        raise BFCLTranslationError(f"{field} must be a sha256 content hash")
    return text


def _load_source(path: Path) -> _Source:
    if path.name != "run_manifest.json" or not path.is_file() or path.is_symlink():
        raise BFCLTranslationError("source_run_manifest must name an immutable published run_manifest.json")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BFCLTranslationError("source_run_manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise BFCLTranslationError("source_run_manifest must be a JSON object")
    _plain_text(manifest.get("run_id"), "source_run_manifest.run_id")
    pack = manifest.get("pack")
    if not isinstance(pack, dict):
        raise BFCLTranslationError("source_run_manifest does not identify its Oracle pack")
    _plain_text(pack.get("pack_id"), "source_run_manifest.pack.pack_id")
    _plain_text(pack.get("version"), "source_run_manifest.pack.version")
    _declared_hash(pack.get("content_hash"), "source_run_manifest.pack.content_hash")
    publication = manifest.get("publication")
    artifacts = manifest.get("artifacts")
    if not isinstance(publication, dict) or not isinstance(artifacts, dict):
        raise BFCLTranslationError("source_run_manifest does not carry publication and artifact contracts")
    if publication.get("verified") is not True:
        raise BFCLTranslationError("source publication is not verified")
    root = path.parent.resolve()
    published = publication.get("published")
    raw = publication.get("raw")
    if not isinstance(published, dict) or not isinstance(raw, dict):
        raise BFCLTranslationError("source publication does not declare both tables")
    benchmark_path = _artifact_path(root, published.get("file"), "publication.published.file")
    raw_path = _artifact_path(root, raw.get("file"), "publication.raw.file")
    benchmark_hash = _hash_file(benchmark_path)
    raw_hash = _hash_file(raw_path)
    declarations = (
        (
            benchmark_hash,
            _declared_hash(
                published.get("content_hash"),
                "publication.published.content_hash",
            ),
            "published benchmark",
        ),
        (
            raw_hash,
            _declared_hash(raw.get("content_hash"), "publication.raw.content_hash"),
            "raw benchmark",
        ),
    )
    for actual, expected, label in declarations:
        if actual != expected:
            raise BFCLTranslationError(f"{label} does not match the hash declared by run_manifest.json")
    artifact_declarations = {
        "benchmark_parquet": benchmark_hash,
        "benchmark_raw_parquet": raw_hash,
    }
    for name, expected in artifact_declarations.items():
        entry = artifacts.get(name)
        if (
            not isinstance(entry, dict)
            or _declared_hash(entry.get("content_hash"), f"artifacts.{name}.content_hash") != expected
        ):
            raise BFCLTranslationError(f"artifacts.{name} does not identify the published bytes")
    try:
        projection = project_published_benchmark(benchmark_path, expected_content_hash=benchmark_hash)
    except ExportProjectionError as exc:
        raise BFCLTranslationError(f"published BFCL benchmark does not satisfy its row contract: {exc}") from exc
    declared_rows = published.get("rows")
    if type(declared_rows) is not int or declared_rows != len(projection.rows):
        raise BFCLTranslationError("publication.published.rows does not match benchmark.parquet")
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    if not pq.read_schema(benchmark_path).equals(benchmark_schema()):
        raise BFCLTranslationError("published benchmark does not use the current BFCL schema")
    rows = pq.read_table(benchmark_path).to_pylist()
    raw_ids = pq.read_table(raw_path, columns=["task_id"]).column("task_id").to_pylist()
    published_ids = [str(row["task_id"]) for row in rows]
    surface_gate = published.get("surface_gate")
    ordering = published.get("ordering")
    dedup_applied = published.get("dedup_balancing_applied")
    held_out_evaluated = published.get("held_out_evaluated")
    if (
        surface_gate not in {"deterministic_guards", "surface_quality"}
        or ordering not in {"raw_order", "selection_rank"}
        or type(dedup_applied) is not bool
        or type(held_out_evaluated) is not bool
    ):
        raise BFCLTranslationError("source publication has invalid gate, ordering, or policy declarations")
    try:
        plan = PublicationPlan(
            raw_task_ids=tuple(raw_ids),
            published_task_ids=tuple(published_ids),
            surface_gate=cast(Literal["deterministic_guards", "surface_quality"], surface_gate),
            dedup_balancing_applied=cast(bool, dedup_applied),
            held_out_evaluated=cast(bool, held_out_evaluated),
            ordering=cast(Literal["raw_order", "selection_rank"], ordering),
        )
        verify_written_benchmarks(
            raw_path=raw_path,
            publication_path=benchmark_path,
            plan=plan,
        )
    except (PublicationContractError, ValueError) as exc:
        raise BFCLTranslationError(
            f"source tables do not form the publication declared by run_manifest.json: {exc}"
        ) from exc
    return _Source(
        manifest_path=path,
        manifest_hash=_hash_file(path),
        manifest=manifest,
        benchmark_path=benchmark_path,
        benchmark_hash=benchmark_hash,
        raw_path=raw_path,
        raw_hash=raw_hash,
        projection=projection,
        rows=rows,
    )


def _scalar_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            tokens.add(str(key))
            tokens.update(_scalar_tokens(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            tokens.update(_scalar_tokens(child))
    else:
        text = value if isinstance(value, str) else canonical_json(value)
        if text:
            tokens.add(text)
    return tokens


def _schema_tokens(schema: Any) -> set[str]:
    """Collect parameter keys and enum literals from a nested JSON Schema."""
    if not isinstance(schema, Mapping):
        return set()
    tokens = _scalar_tokens(schema.get("enum") or [])
    for keyword in ("const", "default"):
        if keyword in schema:
            tokens.update(_scalar_tokens(schema[keyword]))
    required = schema.get("required")
    if isinstance(required, Sequence) and not isinstance(required, str | bytes):
        tokens.update(str(name) for name in required)
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        for key, child in properties.items():
            tokens.add(str(key))
            tokens.update(_schema_tokens(child))
    items = schema.get("items")
    if isinstance(items, Mapping):
        tokens.update(_schema_tokens(items))
    for keyword in ("allOf", "anyOf", "oneOf"):
        branches = schema.get(keyword)
        if isinstance(branches, Sequence) and not isinstance(branches, str | bytes):
            for branch in branches:
                tokens.update(_schema_tokens(branch))
    for keyword in ("$defs", "definitions"):
        definitions = schema.get(keyword)
        if isinstance(definitions, Mapping):
            for key, child in definitions.items():
                tokens.add(str(key))
                tokens.update(_schema_tokens(child))
    return tokens


def _protected_tokens(row: CanonicalExportRow) -> set[str]:
    tokens = {
        row.task_id,
        row.template_id,
        row.pack_id,
        row.pack_version,
        row.system_prompt_id,
        *row.fixture_refs,
        *row.required_tools,
        *row.tools_present,
    }
    for tool in row.tools:
        function = tool.get("function") if isinstance(tool, Mapping) else None
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        if isinstance(name, str):
            tokens.add(name)
        parameters = function.get("parameters")
        if isinstance(parameters, Mapping):
            tokens.update(_schema_tokens(parameters))
    for call in row.expected_tool_calls:
        tokens.add(call.function_name)
        tokens.update(_scalar_tokens(call.arguments))
    for message in row.messages:
        if message.tool_call_id:
            tokens.add(message.tool_call_id)
        for wire_call in message.tool_calls:
            tokens.add(wire_call.id)
            tokens.add(wire_call.function.name)
            try:
                tokens.update(_scalar_tokens(json.loads(wire_call.function.arguments)))
            except json.JSONDecodeError:  # CanonicalExportRow already rejects this.
                pass
    return {token for token in tokens if token}


def protected_translation_field(
    row: CanonicalExportRow,
    text: str,
    *,
    path: str,
) -> str:
    """Return the producer's exact placeholder projection for one field."""
    return _protect(text, _protected_tokens(row), path=path)[0]


def _token_pattern(token: str) -> re.Pattern[str]:
    escaped = re.escape(token)
    left = r"(?<![\w])" if token[0].isalnum() or token[0] == "_" else ""
    right = r"(?![\w])" if token[-1].isalnum() or token[-1] == "_" else ""
    return re.compile(f"{left}{escaped}{right}")


def _protect(text: str, tokens: set[str], *, path: str) -> tuple[str, tuple[_ProtectedValue, ...]]:
    if _PLACEHOLDER_PREFIX in text:
        raise BFCLTranslationError(f"{path} contains the reserved BFCL placeholder prefix")
    protected: list[_ProtectedValue] = []
    output = text
    counter = 0
    for token in sorted(tokens, key=lambda value: (-len(value), value)):
        pattern = _token_pattern(token)

        def replace(_match: re.Match[str]) -> str:
            nonlocal counter
            placeholder = f"{_PLACEHOLDER_PREFIX}{counter:06d}__"
            counter += 1
            protected.append(_ProtectedValue(placeholder, token))
            return placeholder

        output = pattern.sub(replace, output)
    return output, tuple(protected)


def _escape_path(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _units(
    source: _Source,
    *,
    translate_tool_descriptions: bool,
) -> list[_TranslationUnit]:
    units: list[_TranslationUnit] = []
    for row in source.projection.rows:
        tokens = _protected_tokens(row)
        fields: list[tuple[str, str]] = []
        for index, message in enumerate(row.messages):
            if message.content is None:
                continue
            if message.role in {"system", "user"} or (message.role == "assistant" and not message.tool_calls):
                fields.append((f"messages/{index}/content", message.content))
        if row.intent:
            fields.append(("intent", row.intent))
        if translate_tool_descriptions:
            for index, tool in enumerate(row.tools):
                function = tool.get("function") if isinstance(tool, Mapping) else None
                description = function.get("description") if isinstance(function, Mapping) else None
                if isinstance(description, str) and description:
                    fields.append((f"tools/{index}/function/description", description))
        for suffix, text in fields:
            path = f"tasks/{_escape_path(row.task_id)}/{suffix}"
            protected_text, protected = _protect(text, tokens, path=path)
            units.append(
                _TranslationUnit(
                    translation_id=_hash_json(
                        {
                            "contract": FLATTENING_CONTRACT_VERSION,
                            "source_manifest": source.manifest_hash,
                            "path": path,
                            "text": text,
                        }
                    ),
                    task_id=row.task_id,
                    path=path,
                    source_text=text,
                    protected_text=protected_text,
                    protected=protected,
                    protected_tokens=tuple(sorted(tokens, key=lambda value: (-len(value), value))),
                )
            )
    return units


def _translation_frame(
    units: Sequence[_TranslationUnit],
    *,
    source_language: str,
    target_language: str,
    text_by_id: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "translation_id": unit.translation_id,
                "field_path": unit.path,
                "text": (text_by_id[unit.translation_id] if text_by_id is not None else unit.protected_text),
                "source_language_code": source_language,
                "target_language_code": target_language,
            }
            for unit in units
        ]
    )


def _translated_text(
    translated: pd.DataFrame,
    units: Sequence[_TranslationUnit],
) -> dict[str, str]:
    required = {"translation_id", "translation"}
    if not required <= set(translated):
        raise BFCLTranslationError("shared translation backend did not return translation_id and translation")
    records: dict[str, str] = {}
    for row in translated.to_dict(orient="records"):
        identifier = row.get("translation_id")
        text = row.get("translation")
        if not isinstance(identifier, str) or identifier in records:
            raise BFCLTranslationError("shared translation backend returned duplicate or invalid field identity")
        if not isinstance(text, str) or not text.strip():
            raise BFCLTranslationError(f"shared translation backend returned empty text for {identifier}")
        records[identifier] = text
    expected = {unit.translation_id for unit in units}
    if set(records) != expected:
        raise BFCLTranslationError("shared translation backend added, dropped, or replaced translation fields")
    return records


def _require_complete_evidence(
    dataframe: pd.DataFrame,
    units: Sequence[_TranslationUnit],
    *,
    label: str,
) -> None:
    if "translation_id" not in dataframe:
        raise BFCLTranslationError(f"{label} has no stable translation_id")
    identifiers = dataframe["translation_id"].tolist()
    expected = {unit.translation_id for unit in units}
    if (
        len(identifiers) != len(expected)
        or any(not isinstance(identifier, str) for identifier in identifiers)
        or len(set(identifiers)) != len(identifiers)
        or set(identifiers) != expected
    ):
        raise BFCLTranslationError(f"{label} does not cover every stable translation field exactly once")


def _validate_protected_output(
    unit: _TranslationUnit,
    translated: str,
    *,
    label: str,
) -> None:
    actual_order = tuple(_PLACEHOLDER.findall(translated))
    if actual_order != unit.placeholder_order:
        raise BFCLTranslationError(
            f"{label} for {unit.path} lost, duplicated, mutated, or reordered a protected token"
        )
    without_placeholders = _PLACEHOLDER.sub("", translated)
    if _PLACEHOLDER_PREFIX in without_placeholders:
        raise BFCLTranslationError(f"{label} for {unit.path} contains a malformed protected placeholder")
    for value in unit.protected_tokens:
        if _token_pattern(value).search(without_placeholders):
            raise BFCLTranslationError(
                f"{label} for {unit.path} introduced or duplicated protected value {value!r} outside its placeholder"
            )


def _restore(unit: _TranslationUnit, translated: str) -> str:
    _validate_protected_output(unit, translated, label="translation")
    result = translated
    for item in unit.protected:
        if result.count(item.placeholder) != 1:
            raise BFCLTranslationError(f"{unit.path} does not contain protected token {item.placeholder} exactly once")
        result = result.replace(item.placeholder, item.value)
    if _PLACEHOLDER_PREFIX in result:
        raise BFCLTranslationError(f"{unit.path} contains an unknown protected-token placeholder")
    return result


def _deterministic_localization_fix(text: str, policy: _LocalizationPolicy) -> str:
    fixed = text.replace("\r\n", "\n").replace("\r", "\n")
    fixed = "\n".join(line.rstrip() for line in fixed.split("\n"))
    return unicodedata.normalize("NFC", fixed) if policy.normalize_unicode else fixed


def _validate_localized_text(
    units: Sequence[_TranslationUnit],
    translated_by_id: Mapping[str, str],
    policy: _LocalizationPolicy,
) -> dict[str, Any]:
    changed = 0
    combined: list[str] = []
    for unit in units:
        translated = translated_by_id[unit.translation_id]
        _validate_protected_output(unit, translated, label="translation")
        if translated != unit.protected_text:
            changed += 1
        unprotected = _PLACEHOLDER.sub("", translated)
        for pattern in policy.forbidden_patterns:
            if re.search(pattern, unprotected, flags=re.IGNORECASE):
                raise BFCLTranslationError(f"localized output for {unit.path} matches forbidden pattern {pattern!r}")
        combined.append(unprotected)
    changed_fraction = changed / len(units)
    if changed_fraction < policy.minimum_changed_fraction:
        raise BFCLTranslationError(
            "localized output failed the language-change gate: "
            f"{changed_fraction:.3f} fields changed, expected at least "
            f"{policy.minimum_changed_fraction:.3f}"
        )
    if policy.required_script is not None:
        text = "\n".join(combined)
        if not contains_script(text, policy.required_script):
            raise BFCLTranslationError(
                f"localized output contains no characters from required {policy.required_script} script"
            )
    return {
        "contract": "bfcl-localization-validation/1.0",
        "model_role": "translator",
        "deterministic_fixes": {
            "line_endings": "lf",
            "trailing_whitespace": "removed",
            "unicode": "NFC" if policy.normalize_unicode else "unchanged",
        },
        "minimum_changed_fraction": policy.minimum_changed_fraction,
        "changed_fraction": changed_fraction,
        "forbidden_patterns": list(policy.forbidden_patterns),
        "required_script": policy.required_script,
        "executable_replay": "truth_projection_and_parquet_readback",
    }


def _validate_quality_verdicts(
    quality: pd.DataFrame,
    metrics: Sequence[Mapping[str, Any]],
) -> None:
    for row_number, row in enumerate(quality.to_dict(orient="records")):
        expected_passes: list[bool] = []
        for metric in metrics:
            metric_type = str(metric["type"])
            score = row.get(f"score_{metric_type}")
            passed = row.get(f"score_{metric_type}_passed")
            if (
                not isinstance(score, int | float)
                or isinstance(score, bool)
                or not math.isfinite(float(score))
                or type(passed).__name__ not in {"bool", "bool_"}
            ):
                raise BFCLTranslationError(f"quality metrics row {row_number} has invalid {metric_type} verdict")
            threshold = float(metric["threshold"])
            expected = float(score) <= threshold if metric_type == "ter" else float(score) >= threshold
            if bool(passed) != expected:
                raise BFCLTranslationError(
                    f"quality metrics row {row_number} has inconsistent {metric_type} pass/fail"
                )
            expected_passes.append(expected)
        aggregate = row.get("is_quality_metric_passed")
        if type(aggregate).__name__ not in {"bool", "bool_"} or bool(aggregate) != all(expected_passes):
            raise BFCLTranslationError(f"quality metrics row {row_number} has inconsistent aggregate verdict")


def _apply_translations(
    source: _Source,
    units: Sequence[_TranslationUnit],
    translated_by_id: Mapping[str, str],
    *,
    target_language: str,
) -> tuple[list[dict[str, Any]], list[CanonicalExportRow]]:
    rows = [dict(row) for row in source.rows]
    by_task = {str(row["task_id"]): row for row in rows}
    for unit in units:
        value = _restore(unit, translated_by_id[unit.translation_id])
        row = by_task[unit.task_id]
        suffix = unit.path.split(f"tasks/{_escape_path(unit.task_id)}/", 1)[1]
        parts = suffix.split("/")
        if parts[0] == "messages":
            messages = [dict(message) for message in row["messages"]]
            index = int(parts[1])
            messages[index]["content"] = value
            row["messages"] = messages
        elif parts[0] == "intent":
            row["intent"] = value
        elif parts[0] == "tools":
            tools = json.loads(str(row["tools"]))
            tools[int(parts[1])]["function"]["description"] = value
            row["tools"] = canonical_json(tools)
        else:  # pragma: no cover - flattening owns the closed path vocabulary.
            raise BFCLTranslationError(f"unsupported translation field {unit.path}")
    for row in rows:
        metadata = json.loads(str(row["metadata"]))
        metadata["language"] = target_language
        row["metadata"] = canonical_json(metadata)
    validated: list[CanonicalExportRow] = []
    for original, row in zip(source.projection.rows, rows, strict=True):
        try:
            localized = CanonicalExportRow.from_benchmark_row(row)
        except (TypeError, ValueError) as exc:
            raise BFCLTranslationError(f"localized row {original.task_id!r} violates the BFCL schema: {exc}") from exc
        original_truth = translation_preserved_projection(original)
        localized_truth = translation_preserved_projection(localized)
        changed = sorted(
            field
            for field in TRANSLATION_PRESERVED_FIELDS
            if canonical_json(original_truth[field]) != canonical_json(localized_truth[field])
        )
        if changed:
            raise BFCLTranslationError(f"localized row {original.task_id!r} changed Oracle truth fields: {changed}")
        validated.append(localized)
    return rows, validated


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        pq.write_table(
            pa.Table.from_pylist(rows, schema=benchmark_schema()),
            temporary,
            compression="zstd",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_dataframe(path: Path, dataframe: pd.DataFrame) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        dataframe.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_unchanged(source: _Source) -> None:
    expected = (
        (source.manifest_path, source.manifest_hash),
        (source.benchmark_path, source.benchmark_hash),
        (source.raw_path, source.raw_hash),
    )
    for path, content_hash in expected:
        if not path.is_file() or _hash_file(path) != content_hash:
            raise BFCLTranslationError(f"source publication changed during translation: {path}")


class BFCLTranslationAdapter:
    """Translate approved BFCL surfaces and prove all executable truth survived."""

    def __init__(self, config: _TranslationConfig):
        self.config = config

    @classmethod
    def from_yaml(cls, path: str | Path) -> BFCLTranslationAdapter:
        return cls(_load_config(path))

    def run(self) -> Path:
        config = self.config
        source = _load_source(config.source_manifest)
        try:
            source_languages = {
                normalize_locale(str(row.metadata.get("language"))).primary for row in source.projection.rows
            }
        except ValueError as exc:
            raise BFCLTranslationError(f"source benchmark carries an invalid metadata.language: {exc}") from exc
        if source_languages != {config.source_locale.primary}:
            raise BFCLTranslationError("source_language does not match every published row's metadata.language")
        publication = source.manifest_path.parent
        destination = config.output_dir
        if destination == publication or destination in publication.parents or publication in destination.parents:
            raise BFCLTranslationError("translation output must not overlap the immutable source publication")
        if destination.exists():
            raise BFCLTranslationError(f"translation output already exists: {destination}")
        units = _units(
            source,
            translate_tool_descriptions=config.translate_tool_descriptions,
        )
        if not units:
            raise BFCLTranslationError("published benchmark exposes no approved natural-language fields")
        forward_input = _translation_frame(
            units,
            source_language=config.shared.source_language,
            target_language=config.shared.target_language,
        )
        forward = TranslationPipeline(config.shared).translate(forward_input)
        forward_text = {
            identifier: _deterministic_localization_fix(text, config.localization)
            for identifier, text in _translated_text(forward, units).items()
        }
        validation = _validate_localized_text(
            units,
            forward_text,
            config.localization,
        )
        forward = forward_input.copy()
        forward["translation"] = forward["translation_id"].map(forward_text)

        back_input = _translation_frame(
            units,
            source_language=config.shared.target_language,
            target_language=config.shared.source_language,
            text_by_id=forward_text,
        )
        back = TranslationPipeline(config.shared).translate(back_input)
        back_text = {
            identifier: _deterministic_localization_fix(text, config.localization)
            for identifier, text in _translated_text(back, units).items()
        }
        back = back_input.copy()
        back["translation"] = back["translation_id"].map(back_text)
        for unit in units:
            _validate_protected_output(
                unit,
                back_text[unit.translation_id],
                label="backtranslation",
            )

        localized_rows, validated = _apply_translations(
            source,
            units,
            forward_text,
            target_language=config.shared.target_language,
        )
        task_ids = [row.task_id for row in validated]
        if task_ids != list(source.projection.task_ids):
            raise BFCLTranslationError("localized benchmark changed task identity or publication order")

        quality_input = pd.DataFrame(
            [
                {
                    "translation_id": unit.translation_id,
                    "field_path": unit.path,
                    "source_text": unit.protected_text,
                    "translation": forward_text[unit.translation_id],
                    "backtranslation": back_text[unit.translation_id],
                }
                for unit in units
            ]
        )
        quality = (
            evaluate_text_quality_metrics(
                quality_input,
                config.shared,
                reference_text_field="source_text",
                hypothesis_text_field="backtranslation",
            )
            if config.shared.backtranslation_quality_metrics
            else quality_input
        )
        _require_complete_evidence(quality, units, label="quality metrics")
        _validate_quality_verdicts(
            quality,
            config.shared.backtranslation_quality_metrics,
        )
        unit_order = [unit.translation_id for unit in units]
        quality = quality.set_index("translation_id", drop=False).loc[unit_order].reset_index(drop=True)

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.parent / f".{destination.name}-{uuid.uuid4().hex}.tmp"
        stage_cache = temporary / "stage_cache"
        stage_cache.mkdir(parents=True)
        benchmark_name = f"benchmark.{config.target_locale.file_slug}.parquet"
        benchmark_path = temporary / benchmark_name
        try:
            _write_parquet(benchmark_path, localized_rows)
            try:
                localized_projection = project_published_benchmark(
                    benchmark_path,
                    expected_content_hash=_hash_file(benchmark_path),
                )
            except ExportProjectionError as exc:
                raise BFCLTranslationError(f"localized benchmark failed read-back validation: {exc}") from exc
            if localized_projection.task_ids != source.projection.task_ids:
                raise BFCLTranslationError("localized benchmark changed task identity during Parquet encoding")

            forward_path = stage_cache / "translation_units.parquet"
            back_path = stage_cache / "backtranslation_units.parquet"
            quality_path = stage_cache / "quality_metrics.parquet"
            _write_dataframe(forward_path, forward)
            _write_dataframe(back_path, back)
            _write_dataframe(quality_path, quality)

            pack = source.manifest.get("pack")
            if not isinstance(pack, dict):
                raise BFCLTranslationError("source manifest does not identify the Oracle pack")
            artifacts = {
                "translation_units": {
                    "file": "stage_cache/translation_units.parquet",
                    "content_hash": _hash_file(forward_path),
                    "rows": len(forward),
                },
                "backtranslation_units": {
                    "file": "stage_cache/backtranslation_units.parquet",
                    "content_hash": _hash_file(back_path),
                    "rows": len(back),
                },
                "quality_metrics": {
                    "file": "stage_cache/quality_metrics.parquet",
                    "content_hash": _hash_file(quality_path),
                    "rows": len(quality),
                },
            }
            body: dict[str, Any] = {
                "schema_version": SOURCE_VERIFICATION_CONTRACT_VERSION,
                "translation_contract": TRANSLATION_CONTRACT_VERSION,
                "source_run_id": _plain_text(source.manifest.get("run_id"), "source run_id"),
                "source_run_manifest_content_hash": source.manifest_hash,
                "source_benchmark_content_hash": source.benchmark_hash,
                "source_oracle_pack": {
                    "pack_id": pack.get("pack_id"),
                    "version": pack.get("version"),
                    "content_hash": pack.get("content_hash"),
                },
                "source_language": config.shared.source_language,
                "language": config.shared.target_language,
                "benchmark": {
                    "file": benchmark_name,
                    "rows": len(localized_rows),
                    "content_hash": _hash_file(benchmark_path),
                    "schema_fingerprint": _hash_bytes(benchmark_schema().serialize().to_pybytes()),
                },
                "task_ids_hash": _hash_json(task_ids),
                "field_policy": {
                    "flattening_contract": FLATTENING_CONTRACT_VERSION,
                    "preserved_fields": list(TRANSLATION_PRESERVED_FIELDS),
                    "localized_fields": [
                        "messages.system.content",
                        "messages.user.content",
                        "messages.assistant_without_tool_calls.content",
                        "intent",
                        "metadata.language",
                        *(["tools.function.description"] if config.translate_tool_descriptions else []),
                    ],
                    "field_paths_hash": _hash_json([unit.path for unit in units]),
                    "unit_count": len(units),
                },
                "protected_tokens": {
                    "contract": PROTECTION_CONTRACT_VERSION,
                    "occurrences": sum(len(unit.protected) for unit in units),
                    "fields_with_tokens": sum(bool(unit.protected) for unit in units),
                },
                "model": {
                    **config.model,
                    "config_hash": config.model_config_hash,
                },
                "contamination": {
                    "role": "translator",
                    "scope": "all_translated_rows",
                    "task_ids_hash": _hash_json(task_ids),
                    "task_count": len(task_ids),
                    "model_canonical_id": config.model["canonical_id"],
                },
                "quality": {
                    "backtranslation": True,
                    "metrics": config.shared.backtranslation_quality_metrics,
                    "row_filtering": False,
                },
                "localization_validation": validation,
                "artifacts": artifacts,
            }
            manifest = {
                **body,
                "translation_id": _hash_json(body),
            }
            manifest_path = temporary / TRANSLATION_MANIFEST_FILE
            manifest_path.write_text(
                json.dumps(
                    manifest,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            _source_unchanged(source)
            temporary.replace(destination)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        return destination / benchmark_name


def translate_bfcl(
    config_path: str | Path,
    *,
    skip_until: str | None = None,
) -> Path:
    """Run BFCL localization from one committed source publication."""
    if skip_until is not None:
        raise BFCLTranslationError(
            "BFCL translation does not support skip_until; outputs are one content-addressed transaction"
        )
    return BFCLTranslationAdapter.from_yaml(config_path).run()


__all__ = [
    "BFCLTranslationAdapter",
    "BFCLTranslationError",
    "FLATTENING_CONTRACT_VERSION",
    "PROTECTION_CONTRACT_VERSION",
    "TRANSLATION_CONTRACT_VERSION",
    "TRANSLATION_MANIFEST_FILE",
    "protected_translation_field",
    "translate_bfcl",
]
