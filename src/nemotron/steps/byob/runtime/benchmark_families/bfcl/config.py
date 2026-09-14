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

"""BFCL family-local configuration (not ByobConfig / MCQ)."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.dedup_balancing_contract import (
    DEDUP_BALANCING_CONTRACT_VERSION,
    TURN_POLICIES,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.surface_quality_contract import (
    SURFACE_QUALITY_CONTRACT_VERSION,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.surface_styles import (
    SURFACE_STYLE_AXES,
)

# runtime/benchmark_families/bfcl/config.py → …/steps/byob
BYOB_ROOT = Path(__file__).resolve().parents[3]

# Benchmark row schemas this build can write. A config naming anything else would
# publish a manifest that promises a shape the parquet does not have.
DEFAULT_BENCHMARK_SCHEMA_VERSION = "1.1"
BENCHMARK_SCHEMA_VERSIONS = frozenset({DEFAULT_BENCHMARK_SCHEMA_VERSION})
LINEAGE_POLICIES = frozenset({"strict_separation", "smoke_no_publication"})

_TOP_LEVEL_KEYS = frozenset(
    {
        "family",
        "expt_name",
        "output_dir",
        "oracle_pack",
        "oracle_runtime",
        "lineage",
        "stage",
        "random_seed",
        "ndd_batch_size",
        "input_dir",
        "schema_version",
        "config_status",
        "surface_generation",
        "surface_quality_validation",
        "task_generation",
        "semantic_deduplication_config",
        "exports",
        "reference_benchmark",
        "generation_model_config",
        "judge_model_config",
        "eval_config_path",
        "eval",
        "translation_config_path",
    }
)
# Eval inputs a generation config may carry for the eval entry point to read.
# They are excluded from every generation lineage hash: swapping a candidate model
# or a decoding temperature changes what was *evaluated*, never what was
# generated, and a benchmark whose identity moved because of an eval edit could
# not be compared against its own earlier scores.
EVAL_REFERENCE_KEYS = frozenset({"eval_config_path", "eval"})
_ORACLE_PACK_KEYS = frozenset(
    {
        "manifest_path",
        "backend_path",
        "endpoint_config_path",
        "fixtures_path",
        "task_templates_path",
        "assertions_path",
        "validation_cases_path",
    }
)
_ORACLE_RUNTIME_KEYS = frozenset(
    {
        "clock",
        "tool_timeout_s",
        "assertion_timeout_s",
        "import_timeout_s",
        "reset_timeout_s",
        "episode_timeout_s",
        "worker",
        "allowed_roots",
    }
)
_LINEAGE_KEYS = frozenset({"policy", "profile_influenced_surface", "judge_advisory", "roles"})
_LINEAGE_ROLES = frozenset({"profile", "paraphrase", "surface_judge"})
_LINEAGE_ROLE_KEYS = frozenset({"enabled", "model_config"})
_MODEL_CONFIG_SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "bearer_token",
        "password",
        "secret",
        "token",
        "access_token",
        "auth_token",
    }
)
SURFACE_GENERATION_KEYS = frozenset(
    {
        "language",
        "model_paraphrase_enabled",
        "paraphrases_per_template",
        "preserve_slot_values",
        "prevent_tool_name_leakage",
        "surface_style_axes",
    }
)
_SURFACE_QUALITY_KEYS = frozenset({"contract_version", "enabled", "drop_authority"})
_TASK_GENERATION_KEYS = frozenset(
    {
        "tasks_per_category",
        "candidate_tasks_per_category",
        "target_published_tasks",
        "max_turns",
        "max_tool_calls",
        "difficulty_mix",
        "turn_mix",
        "tool_call_count_mix",
        "max_intent_share",
        "policy_mix",
    }
)
_DEDUP_KEYS = frozenset(
    {
        "contract_version",
        "enabled",
        "model_identifier",
        "n_clusters",
        "eps",
        "remove_duplicates",
        "max_exact_surface_reuse",
        "min_exact_surface_ratio",
        "max_rows_per_intent",
        "max_execution_case_reuse",
        "representative_source_preference",
        "unmet_target_policy",
    }
)
_EXPORT_KEYS = frozenset({"bfcl_json", "nemo_evaluator_bundle"})
_REFERENCE_BENCHMARK_KEYS = frozenset({"name", "samples_path", "content_hash"})


def _reject_unknown(mapping: dict[str, Any], allowed: frozenset[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"{path} has unknown keys: {', '.join(unknown)}")


def _require_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean, got {type(value).__name__}")
    return value


def _require_int(value: Any, path: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path} must be an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be at least {minimum}, got {value}")
    return value


def _require_number(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{path} must be a number, got {type(value).__name__}")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{path} must be finite, got {value!r}")
    return converted


def _require_probability_mix(value: Any, path: str) -> dict[str, float]:
    """Validate and normalize one deterministic balancing target."""
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{path} must be a non-empty mapping")
    normalized: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{path} keys must be non-empty strings")
        probability = _require_number(raw, f"{path}.{key}")
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{path}.{key} must be between 0 and 1, got {raw!r}")
        normalized[key] = probability
    if not math.isclose(sum(normalized.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{path} probabilities must sum to 1, got {sum(normalized.values())!r}")
    return normalized


def _resolve_declared_style_axes(surface: dict[str, Any]) -> tuple[str, ...]:
    """Validate an optional axis override and return the axes this config will use."""
    declared = surface.get("surface_style_axes")
    if declared is None:
        return SURFACE_STYLE_AXES
    if not isinstance(declared, list) or not declared:
        raise ValueError("surface_generation.surface_style_axes must be a non-empty list")
    axes: list[str] = []
    for index, axis in enumerate(declared):
        if not isinstance(axis, str) or not axis.strip():
            raise ValueError(f"surface_generation.surface_style_axes[{index}] must be a non-empty string")
        axes.append(axis.strip())
    duplicates = sorted({axis for axis in axes if axes.count(axis) > 1})
    if duplicates:
        raise ValueError(
            "surface_generation.surface_style_axes must be unique, found duplicates: "
            + ", ".join(repr(axis) for axis in duplicates)
        )
    surface["surface_style_axes"] = axes
    return tuple(axes)


def _reject_model_secrets(value: Any, path: str) -> None:
    """Keep credentials out of resolved configs, hashes, and public manifests."""
    if isinstance(value, list):
        for index, child in enumerate(value):
            _reject_model_secrets(child, f"{path}[{index}]")
        return
    if not isinstance(value, dict):
        return
    for key, child in value.items():
        lowered = str(key).lower().replace("-", "_")
        if lowered in _MODEL_CONFIG_SECRET_KEYS or lowered.endswith(
            ("_api_key", "_password", "_secret", "_access_token", "_auth_token")
        ):
            raise ValueError(f"{path}.{key} looks like a secret; provide credentials through the provider environment")
        _reject_model_secrets(child, f"{path}.{key}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _validate_reference_samples(path: Path) -> None:
    """Reject malformed samples and any oracle-truth fields before model use."""
    import json

    forbidden = {
        "assertion",
        "assertions",
        "assertion_verdict",
        "backend_state",
        "expected_result",
        "expected_tool_calls",
        "oracle_state",
        "success_assertions",
        "tool_calls",
        "tools",
    }

    def forbidden_paths(value: Any, prefix: str = "") -> list[str]:
        if isinstance(value, dict):
            paths: list[str] = []
            for key, child in value.items():
                current = f"{prefix}.{key}" if prefix else str(key)
                if str(key) in forbidden:
                    paths.append(current)
                paths.extend(forbidden_paths(child, current))
            return paths
        if isinstance(value, list):
            return [item for index, child in enumerate(value) for item in forbidden_paths(child, f"{prefix}[{index}]")]
        return []

    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"reference sample line {line_number} is not valid JSON") from exc
        if not isinstance(sample, dict):
            raise ValueError(f"reference sample line {line_number} must be an object")
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"reference sample line {line_number} must declare a non-empty sample_id")
        if sample_id in seen:
            raise ValueError(f"reference benchmark contains duplicate sample_id {sample_id!r}")
        seen.add(sample_id)
        if not isinstance(sample.get("language"), str) or not sample["language"].strip():
            raise ValueError(f"reference sample {sample_id!r} needs a language")
        tags = sample.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag.strip() for tag in tags):
            raise ValueError(f"reference sample {sample_id!r} tags must be a list of non-empty strings")
        messages = sample.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"reference sample {sample_id!r} messages must be a non-empty list")
        for index, message in enumerate(messages):
            if (
                not isinstance(message, dict)
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
            ):
                raise ValueError(
                    f"reference sample {sample_id!r} messages[{index}] must contain "
                    "a supported role and string content"
                )
        leaked = forbidden_paths(sample)
        if leaked:
            raise ValueError(
                f"reference sample {sample_id!r} contains oracle-truth fields: " + ", ".join(sorted(leaked))
            )
    if not seen:
        raise ValueError("reference benchmark must contain at least one sample")


def _placeholder_paths(value: Any, path: str = "") -> list[str]:
    """Return config paths whose values still carry replacement sentinels."""
    if isinstance(value, dict):
        return [
            item
            for key, child in value.items()
            for item in _placeholder_paths(child, f"{path}.{key}" if path else str(key))
        ]
    if isinstance(value, list):
        return [item for index, child in enumerate(value) for item in _placeholder_paths(child, f"{path}[{index}]")]
    if isinstance(value, str) and "REPLACE_ME_" in value:
        return [path]
    return []


def _validate_clock(raw: str) -> None:
    """Reject clocks the oracle worker cannot parse or that lack an offset."""
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"oracle_runtime.clock must be an ISO-8601 timestamp, got {raw!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"oracle_runtime.clock must carry a UTC offset so runs stay reproducible, got {raw!r}")


def _resolve_path(raw: str | Path | None, *, base: Path = BYOB_ROOT) -> Path | None:
    if raw is None:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


@dataclass(frozen=True)
class OraclePackRef:
    """Paths into an executable oracle pack."""

    manifest_path: Path
    backend_path: Path | None = None
    endpoint_config_path: Path | None = None
    fixtures_path: Path | None = None
    task_templates_path: Path | None = None
    assertions_path: Path | None = None
    validation_cases_path: Path | None = None


@dataclass(frozen=True)
class OracleRuntimeConfig:
    """Process-worker timeouts, frozen clock, and pack trust roots."""

    clock: str
    tool_timeout_s: float = 5.0
    assertion_timeout_s: float = 5.0
    import_timeout_s: float = 10.0
    reset_timeout_s: float = 5.0
    episode_timeout_s: float = 60.0
    worker: str = "process"
    allowed_roots: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if self.worker not in {"process", "thread"}:
            raise ValueError(f"oracle_runtime.worker must be 'process' or 'thread', got {self.worker!r}")
        _validate_clock(self.clock)
        for name in (
            "tool_timeout_s",
            "assertion_timeout_s",
            "import_timeout_s",
            "reset_timeout_s",
            "episode_timeout_s",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"oracle_runtime.{name} must be positive, got {value!r}")


@dataclass(frozen=True)
class LineageRole:
    enabled: bool = False
    model_config: dict[str, Any] | None = None


@dataclass(frozen=True)
class LineageConfig:
    policy: str
    profile_influenced_surface: bool | None = False
    judge_advisory: bool | None = None
    roles: dict[str, LineageRole] = field(default_factory=dict)


@dataclass(frozen=True)
class ReferenceBenchmarkConfig:
    """Allowlisted style-only reference samples with a pinned identity."""

    name: str
    samples_path: Path
    content_hash: str


@dataclass
class BfclConfig:
    """Validated BFCL generation config.

    Kept family-local (conscious deviation from shared ``runtime/config.py``)
    so MCQ fields such as ``hf_dataset`` / ``target_source_mapping`` cannot leak in.
    """

    family: str
    expt_name: str
    output_dir: Path
    oracle_pack: OraclePackRef
    oracle_runtime: OracleRuntimeConfig
    lineage: LineageConfig
    stage: str = "all"
    random_seed: int | None = None
    ndd_batch_size: int = 32
    input_dir: Path | None = None
    schema_version: str | None = None
    config_status: str | None = None
    surface_generation: dict[str, Any] = field(default_factory=dict)
    surface_quality_validation: dict[str, Any] = field(default_factory=dict)
    task_generation: dict[str, Any] = field(default_factory=dict)
    semantic_deduplication_config: dict[str, Any] = field(default_factory=dict)
    exports: dict[str, Any] = field(default_factory=dict)
    reference_benchmark: ReferenceBenchmarkConfig | None = None
    generation_model_config: dict[str, Any] | None = None
    judge_model_config: dict[str, Any] | None = None
    eval_config_path: str | None = None
    # Legacy inline eval block, kept raw. It is normalized by
    # ``bfcl.eval.config.load_eval_config_for_generation`` into the same
    # ``BfclEvalConfig`` a standalone file produces, so there is one validator
    # rather than a second dialect. Never read by a generation stage.
    inline_eval: dict[str, Any] | None = field(default=None, repr=False)
    translation_config_path: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> BfclConfig:
        config_path = Path(path).resolve()
        with config_path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

        if not isinstance(data, dict):
            raise ValueError(f"BFCL config must be a YAML mapping: {config_path}")
        _reject_unknown(data, _TOP_LEVEL_KEYS, "BFCL config")

        family = data.get("family")
        if family != "bfcl":
            raise ValueError(f"BfclConfig requires family='bfcl', got {family!r}")

        for key in ("expt_name", "output_dir", "oracle_pack", "oracle_runtime", "lineage"):
            if key not in data:
                raise ValueError(f"BFCL config missing required field {key!r}: {config_path}")
        if not isinstance(data["expt_name"], str) or not data["expt_name"].strip():
            raise ValueError("expt_name must be a non-empty string")
        # The run directory is named after the experiment, and every artifact hash and
        # cache path is stated relative to it. A separator or a dot segment would move
        # the run somewhere the config does not name, including back out of output_dir.
        if (
            data["expt_name"] != data["expt_name"].strip()
            or data["expt_name"] in {".", ".."}
            or Path(data["expt_name"]).parts != (data["expt_name"],)
        ):
            raise ValueError(
                f"expt_name must be a single directory name, got {data['expt_name']!r}; "
                "point output_dir at the parent instead"
            )
        if not isinstance(data["output_dir"], str) or not data["output_dir"].strip():
            raise ValueError("output_dir must be a non-empty path string")

        schema_version = data.get("schema_version")
        if schema_version is not None and str(schema_version) not in BENCHMARK_SCHEMA_VERSIONS:
            raise ValueError(
                f"schema_version {schema_version!r} is not a schema this build writes; "
                f"use one of {', '.join(sorted(BENCHMARK_SCHEMA_VERSIONS))}"
            )

        config_status = data.get("config_status")
        if config_status not in {None, "template", "resolved"}:
            raise ValueError(f"config_status must be 'template' or 'resolved' when present, got {config_status!r}")
        if config_status == "resolved":
            placeholders = _placeholder_paths(data)
            if placeholders:
                raise ValueError(
                    "resolved BFCL config still contains REPLACE_ME_* values at: " + ", ".join(placeholders)
                )

        pack_raw = data["oracle_pack"]
        if not isinstance(pack_raw, dict) or "manifest_path" not in pack_raw:
            raise ValueError("oracle_pack.manifest_path is required")
        _reject_unknown(pack_raw, _ORACLE_PACK_KEYS, "oracle_pack")

        runtime_raw = data["oracle_runtime"]
        if not isinstance(runtime_raw, dict):
            raise ValueError("oracle_runtime must be a mapping")
        _reject_unknown(runtime_raw, _ORACLE_RUNTIME_KEYS, "oracle_runtime")
        if "clock" not in runtime_raw:
            raise ValueError("oracle_runtime.clock is required")
        if not isinstance(runtime_raw["clock"], str):
            raise ValueError(
                f"oracle_runtime.clock must be a quoted ISO-8601 string; got {type(runtime_raw['clock']).__name__}"
            )

        allowed_raw = runtime_raw.get("allowed_roots")
        if allowed_raw is None:
            allowed_raw = []
        if not isinstance(allowed_raw, list):
            raise ValueError("oracle_runtime.allowed_roots must be a list when present")
        if any(not isinstance(item, str) or not item.strip() for item in allowed_raw):
            raise ValueError("oracle_runtime.allowed_roots entries must be non-empty path strings")
        allowed_roots = tuple(p for p in (_resolve_path(item) for item in allowed_raw) if p is not None)
        if not allowed_roots:
            # Default trust root: checked-in BYOB data directory.
            allowed_roots = ((BYOB_ROOT / "data").resolve(),)

        oracle_runtime = OracleRuntimeConfig(
            clock=runtime_raw["clock"],
            tool_timeout_s=_require_number(runtime_raw.get("tool_timeout_s", 5.0), "oracle_runtime.tool_timeout_s"),
            assertion_timeout_s=_require_number(
                runtime_raw.get("assertion_timeout_s", 5.0),
                "oracle_runtime.assertion_timeout_s",
            ),
            import_timeout_s=_require_number(
                runtime_raw.get("import_timeout_s", 10.0),
                "oracle_runtime.import_timeout_s",
            ),
            reset_timeout_s=_require_number(runtime_raw.get("reset_timeout_s", 5.0), "oracle_runtime.reset_timeout_s"),
            episode_timeout_s=_require_number(
                runtime_raw.get("episode_timeout_s", 60.0),
                "oracle_runtime.episode_timeout_s",
            ),
            worker=str(runtime_raw.get("worker", "process")),
            allowed_roots=allowed_roots,
        )

        lineage_raw = data["lineage"]
        if not isinstance(lineage_raw, dict) or "policy" not in lineage_raw:
            raise ValueError("lineage.policy is required")
        _reject_unknown(lineage_raw, _LINEAGE_KEYS, "lineage")
        policy = str(lineage_raw["policy"])
        if policy not in LINEAGE_POLICIES:
            raise ValueError(f"lineage.policy must be one of {', '.join(sorted(LINEAGE_POLICIES))}, got {policy!r}")
        roles_raw = lineage_raw.get("roles")
        if roles_raw is None:
            roles_raw = {}
        roles: dict[str, LineageRole] = {}
        if not isinstance(roles_raw, dict):
            raise ValueError("lineage.roles must be a mapping")
        _reject_unknown(roles_raw, _LINEAGE_ROLES, "lineage.roles")
        for name, role in roles_raw.items():
            if not isinstance(role, dict):
                raise ValueError(f"lineage.roles.{name} must be a mapping")
            _reject_unknown(role, _LINEAGE_ROLE_KEYS, f"lineage.roles.{name}")
            enabled = _require_bool(role.get("enabled", False), f"lineage.roles.{name}.enabled")
            model_config = role.get("model_config")
            if model_config is not None and not isinstance(model_config, dict):
                raise ValueError(f"lineage.roles.{name}.model_config must be a mapping or null")
            if model_config is not None:
                model_config = dict(model_config)
                _reject_model_secrets(model_config, f"lineage.roles.{name}.model_config")
                inference_parameters = model_config.get("inference_parameters", {})
                if not isinstance(inference_parameters, dict):
                    raise ValueError(f"lineage.roles.{name}.model_config.inference_parameters must be a mapping")
                model_config["inference_parameters"] = dict(inference_parameters)
            if enabled:
                if not model_config:
                    raise ValueError(f"lineage.roles.{name}.model_config is required when the role is enabled")
                missing_identity = [
                    key
                    for key in ("alias", "provider", "model", "canonical_id")
                    if not isinstance(model_config.get(key), str) or not str(model_config[key]).strip()
                ]
                if missing_identity:
                    raise ValueError(
                        f"lineage.roles.{name}.model_config requires non-empty strings for: "
                        + ", ".join(missing_identity)
                    )
            roles[name] = LineageRole(enabled=enabled, model_config=model_config)

        if policy == "strict_separation":
            canonical_roles: dict[str, str] = {
                name: str(role.model_config["canonical_id"]).strip().lower()
                for name, role in roles.items()
                if role.enabled and role.model_config is not None
            }
            duplicates = sorted(
                canonical_id
                for canonical_id in set(canonical_roles.values())
                if list(canonical_roles.values()).count(canonical_id) > 1
            )
            if duplicates:
                colliding = sorted(
                    name for name, canonical_id in canonical_roles.items() if canonical_id in duplicates
                )
                raise ValueError(
                    "lineage.policy strict_separation requires distinct canonical model "
                    f"identities; roles {', '.join(colliding)} collide"
                )

        profile_influenced = lineage_raw.get("profile_influenced_surface", False)
        if profile_influenced is not None:
            profile_influenced = _require_bool(profile_influenced, "lineage.profile_influenced_surface")
        judge_advisory = lineage_raw.get("judge_advisory")
        if judge_advisory is not None:
            judge_advisory = _require_bool(judge_advisory, "lineage.judge_advisory")
        if profile_influenced and not bool(roles.get("profile") and roles["profile"].enabled):
            raise ValueError("lineage.profile_influenced_surface can only be true when the profile role is enabled")

        lineage = LineageConfig(
            policy=policy,
            profile_influenced_surface=profile_influenced,
            judge_advisory=judge_advisory,
            roles=roles,
        )

        oracle_pack = OraclePackRef(
            manifest_path=_resolve_path(pack_raw["manifest_path"]),  # type: ignore[arg-type]
            backend_path=_resolve_path(pack_raw.get("backend_path")),
            endpoint_config_path=_resolve_path(pack_raw.get("endpoint_config_path")),
            fixtures_path=_resolve_path(pack_raw.get("fixtures_path")),
            task_templates_path=_resolve_path(pack_raw.get("task_templates_path")),
            assertions_path=_resolve_path(pack_raw.get("assertions_path")),
            validation_cases_path=_resolve_path(pack_raw.get("validation_cases_path")),
        )

        output_dir = _resolve_path(data["output_dir"])
        assert output_dir is not None
        run_output = (output_dir / data["expt_name"]).resolve()
        pack_root = oracle_pack.manifest_path.parent
        try:
            run_output.relative_to(pack_root)
        except ValueError:
            pass
        else:
            raise ValueError(
                "output_dir/expt_name must be outside the oracle pack root so generated "
                "artifacts cannot change the pack fingerprint"
            )

        strict_sections = (
            ("surface_generation", SURFACE_GENERATION_KEYS),
            ("surface_quality_validation", _SURFACE_QUALITY_KEYS),
            ("task_generation", _TASK_GENERATION_KEYS),
            ("semantic_deduplication_config", _DEDUP_KEYS),
            ("exports", _EXPORT_KEYS),
        )
        sections: dict[str, dict[str, Any]] = {}
        for section, allowed in strict_sections:
            raw_section = data.get(section)
            if raw_section is None:
                raw_section = {}
            if not isinstance(raw_section, dict):
                raise ValueError(f"{section} must be a mapping")
            _reject_unknown(raw_section, allowed, section)
            sections[section] = raw_section

        surface = sections["surface_generation"]
        for key in (
            "model_paraphrase_enabled",
            "preserve_slot_values",
            "prevent_tool_name_leakage",
        ):
            if key in surface:
                _require_bool(surface[key], f"surface_generation.{key}")
        if "paraphrases_per_template" in surface:
            _require_int(
                surface["paraphrases_per_template"],
                "surface_generation.paraphrases_per_template",
                minimum=0,
            )
        if "language" in surface and (not isinstance(surface["language"], str) or not surface["language"].strip()):
            raise ValueError("surface_generation.language must be a non-empty string")
        style_axes = _resolve_declared_style_axes(surface)
        paraphrase_role_enabled = bool(roles.get("paraphrase") and roles["paraphrase"].enabled)
        paraphrase_enabled = bool(surface.get("model_paraphrase_enabled", False))
        paraphrase_count = int(surface.get("paraphrases_per_template", 0))
        # Beyond one variant per axis the assignment can only repeat an axis inside a
        # single binding, which asks the same model for the same rewrite twice.
        if paraphrase_count > len(style_axes):
            raise ValueError(
                "surface_generation.paraphrases_per_template cannot exceed the "
                f"{len(style_axes)} declared surface style axes"
            )
        if paraphrase_enabled != paraphrase_role_enabled:
            raise ValueError("surface_generation.model_paraphrase_enabled must match lineage.roles.paraphrase.enabled")
        if paraphrase_enabled and paraphrase_count < 1:
            raise ValueError(
                "surface_generation.paraphrases_per_template must be positive when model paraphrasing is enabled"
            )
        if not paraphrase_enabled and paraphrase_count:
            raise ValueError(
                "surface_generation.paraphrases_per_template must be zero when model paraphrasing is disabled"
            )
        if profile_influenced and not paraphrase_enabled:
            raise ValueError("lineage.profile_influenced_surface can only be true when model paraphrasing is enabled")

        quality = sections["surface_quality_validation"]
        contract_version = quality.get(
            "contract_version",
            SURFACE_QUALITY_CONTRACT_VERSION,
        )
        if contract_version != SURFACE_QUALITY_CONTRACT_VERSION:
            raise ValueError(
                "surface_quality_validation.contract_version must be "
                f"{SURFACE_QUALITY_CONTRACT_VERSION!r}, got {contract_version!r}"
            )
        quality["contract_version"] = contract_version
        for key in ("enabled", "drop_authority"):
            if key in quality:
                _require_bool(quality[key], f"surface_quality_validation.{key}")
        quality_enabled = bool(quality.get("enabled", False))
        judge_drop_authority = bool(quality.get("drop_authority", False))
        judge_enabled = bool(roles.get("surface_judge") and roles["surface_judge"].enabled)
        if judge_enabled and not quality_enabled:
            raise ValueError(
                "surface_quality_validation.enabled must be true when lineage.roles.surface_judge is enabled"
            )
        if judge_drop_authority and not judge_enabled:
            raise ValueError(
                "surface_quality_validation.drop_authority requires an enabled lineage.roles.surface_judge"
            )
        if judge_enabled:
            expected_advisory = not judge_drop_authority
            if judge_advisory is not expected_advisory:
                raise ValueError(
                    "lineage.judge_advisory must equal the inverse of "
                    "surface_quality_validation.drop_authority when the surface judge is enabled"
                )
        elif judge_advisory is not None:
            raise ValueError("lineage.judge_advisory must be null when the surface judge is disabled")
        task_generation = sections["task_generation"]
        for key in (
            "tasks_per_category",
            "candidate_tasks_per_category",
            "target_published_tasks",
            "max_turns",
            "max_tool_calls",
        ):
            if key not in task_generation:
                continue
            _require_int(
                task_generation[key],
                f"task_generation.{key}",
                minimum=1,
            )
        selected_category_budget = int(
            task_generation.get("tasks_per_category", 1) or 1
        )
        candidate_category_budget = int(
            task_generation.get(
                "candidate_tasks_per_category",
                selected_category_budget,
            )
            or selected_category_budget
        )
        if candidate_category_budget < selected_category_budget:
            raise ValueError(
                "task_generation.candidate_tasks_per_category must be greater "
                "than or equal to task_generation.tasks_per_category"
            )
        for key in ("difficulty_mix", "turn_mix", "tool_call_count_mix", "policy_mix"):
            if key in task_generation:
                task_generation[key] = _require_probability_mix(task_generation[key], f"task_generation.{key}")
        if "max_intent_share" in task_generation:
            max_intent_share = _require_number(
                task_generation["max_intent_share"],
                "task_generation.max_intent_share",
            )
            if not 0.0 < max_intent_share <= 1.0:
                raise ValueError(
                    "task_generation.max_intent_share must be greater than 0 "
                    f"and at most 1, got {task_generation['max_intent_share']!r}"
                )
            task_generation["max_intent_share"] = max_intent_share
        turn_keys = set(task_generation.get("turn_mix") or {})
        if unknown := sorted(turn_keys - {"single_turn", "multi_turn"}):
            raise ValueError("task_generation.turn_mix has unknown keys: " + ", ".join(unknown))
        call_count_keys = set(task_generation.get("tool_call_count_mix") or {})
        if unknown := sorted(call_count_keys - {"1", "2", "3+"}):
            raise ValueError("task_generation.tool_call_count_mix has unknown keys: " + ", ".join(unknown))
        policy_keys = set(task_generation.get("policy_mix") or {})
        if unknown := sorted(policy_keys - TURN_POLICIES):
            raise ValueError("task_generation.policy_mix has unknown keys: " + ", ".join(unknown))
        dedup = sections["semantic_deduplication_config"]
        dedup_contract_version = dedup.get(
            "contract_version",
            DEDUP_BALANCING_CONTRACT_VERSION,
        )
        if dedup_contract_version != DEDUP_BALANCING_CONTRACT_VERSION:
            raise ValueError(
                "semantic_deduplication_config.contract_version must be "
                f"{DEDUP_BALANCING_CONTRACT_VERSION!r}, got "
                f"{dedup_contract_version!r}"
            )
        dedup["contract_version"] = dedup_contract_version
        if "enabled" in dedup:
            _require_bool(dedup["enabled"], "semantic_deduplication_config.enabled")
        if dedup.get("enabled") and not quality_enabled:
            raise ValueError(
                "surface_quality_validation.enabled must be true when semantic_deduplication_config.enabled is true"
            )
        if dedup.get("enabled"):
            required_dedup_keys = {
                "model_identifier",
                "n_clusters",
                "eps",
                "remove_duplicates",
            }
            missing_dedup_keys = sorted(required_dedup_keys - set(dedup))
            if missing_dedup_keys:
                raise ValueError(
                    "semantic_deduplication_config is missing required keys when enabled: "
                    + ", ".join(missing_dedup_keys)
                )
        if "model_identifier" in dedup and (
            not isinstance(dedup["model_identifier"], str) or not dedup["model_identifier"].strip()
        ):
            raise ValueError("semantic_deduplication_config.model_identifier must be a non-empty string")
        if "model_identifier" in dedup:
            dedup["model_identifier"] = dedup["model_identifier"].strip()
        if "n_clusters" in dedup:
            _require_int(
                dedup["n_clusters"],
                "semantic_deduplication_config.n_clusters",
                minimum=1,
            )
        if "eps" in dedup:
            eps = _require_number(dedup["eps"], "semantic_deduplication_config.eps")
            if not 0.0 < eps < 1.0:
                raise ValueError("semantic_deduplication_config.eps must be between 0 and 1")
            dedup["eps"] = eps
        if "remove_duplicates" in dedup:
            _require_bool(
                dedup["remove_duplicates"],
                "semantic_deduplication_config.remove_duplicates",
            )
        if "max_exact_surface_reuse" in dedup:
            _require_int(
                dedup["max_exact_surface_reuse"],
                "semantic_deduplication_config.max_exact_surface_reuse",
                minimum=1,
            )
        for key in ("max_rows_per_intent", "max_execution_case_reuse"):
            if key in dedup:
                _require_int(
                    dedup[key],
                    f"semantic_deduplication_config.{key}",
                    minimum=1,
                )
        if "min_exact_surface_ratio" in dedup:
            ratio = _require_number(
                dedup["min_exact_surface_ratio"],
                "semantic_deduplication_config.min_exact_surface_ratio",
            )
            if not 0.0 < ratio <= 1.0:
                raise ValueError(
                    "semantic_deduplication_config.min_exact_surface_ratio "
                    "must be greater than 0 and at most 1"
                )
            dedup["min_exact_surface_ratio"] = ratio
        if "representative_source_preference" in dedup:
            preference = dedup["representative_source_preference"]
            if (
                not isinstance(preference, list)
                or not preference
                or any(not isinstance(source, str) or not source.strip() for source in preference)
            ):
                raise ValueError(
                    "semantic_deduplication_config.representative_source_preference "
                    "must be a non-empty list of source names"
                )
            normalized_preference = [source.strip() for source in preference]
            if len(set(normalized_preference)) != len(normalized_preference):
                raise ValueError(
                    "semantic_deduplication_config.representative_source_preference must not repeat a source"
                )
            unknown_sources = sorted(set(normalized_preference) - {"template", "model"})
            if unknown_sources:
                raise ValueError(
                    "semantic_deduplication_config.representative_source_preference "
                    "has unknown sources: " + ", ".join(unknown_sources)
                )
            dedup["representative_source_preference"] = normalized_preference
        unmet_target_policy = dedup.get("unmet_target_policy", "abort")
        if (
            not isinstance(unmet_target_policy, str)
            or unmet_target_policy not in {"abort", "publish_non_gold"}
        ):
            raise ValueError(
                "semantic_deduplication_config.unmet_target_policy must be "
                "'abort' or 'publish_non_gold'"
            )
        dedup["unmet_target_policy"] = unmet_target_policy
        exports = sections["exports"]
        for key, value in exports.items():
            _require_bool(value, f"exports.{key}")
        if data.get("random_seed") is not None:
            _require_int(data["random_seed"], "random_seed")
        if "ndd_batch_size" in data:
            _require_int(data["ndd_batch_size"], "ndd_batch_size", minimum=1)

        reference_benchmark = None
        reference_raw = data.get("reference_benchmark")
        if reference_raw is not None:
            if not isinstance(reference_raw, dict):
                raise ValueError("reference_benchmark must be a mapping or null")
            _reject_unknown(reference_raw, _REFERENCE_BENCHMARK_KEYS, "reference_benchmark")
            missing = sorted(
                key
                for key in _REFERENCE_BENCHMARK_KEYS
                if not isinstance(reference_raw.get(key), str) or not str(reference_raw[key]).strip()
            )
            if missing:
                raise ValueError("reference_benchmark requires non-empty strings for: " + ", ".join(missing))
            samples_path = _resolve_path(reference_raw["samples_path"])
            assert samples_path is not None
            from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import (
                assert_pack_allowed,
            )

            samples_path = assert_pack_allowed(samples_path, allowed_roots)
            if not samples_path.is_file():
                raise FileNotFoundError(f"reference_benchmark.samples_path does not exist: {samples_path}")
            content_hash = str(reference_raw["content_hash"]).lower()
            if not content_hash.startswith("sha256:") or len(content_hash) != 71:
                raise ValueError("reference_benchmark.content_hash must be sha256:<64 lowercase hex characters>")
            try:
                int(content_hash.removeprefix("sha256:"), 16)
            except ValueError as exc:
                raise ValueError(
                    "reference_benchmark.content_hash must be sha256:<64 lowercase hex characters>"
                ) from exc
            actual_hash = _sha256_file(samples_path)
            if actual_hash != content_hash:
                raise ValueError(
                    "reference_benchmark.content_hash does not match samples_path "
                    f"(expected {content_hash}, got {actual_hash})"
                )
            _validate_reference_samples(samples_path)
            reference_benchmark = ReferenceBenchmarkConfig(
                name=str(reference_raw["name"]).strip(),
                samples_path=samples_path,
                content_hash=content_hash,
            )
        profile_role_enabled = bool(roles.get("profile") and roles["profile"].enabled)
        if profile_role_enabled and reference_benchmark is None:
            raise ValueError("reference_benchmark must be configured when lineage.roles.profile is enabled")

        # Eval inputs are carried, not interpreted: the eval entry point owns their
        # contract. Two of them in one file is refused here rather than resolved by
        # precedence, because only one of the two could have been the config that ran.
        eval_config_path = data.get("eval_config_path")
        if eval_config_path is not None and (not isinstance(eval_config_path, str) or not eval_config_path.strip()):
            raise ValueError("eval_config_path must be a non-empty path string when present")
        inline_eval = data.get("eval")
        if inline_eval is not None and not isinstance(inline_eval, dict):
            raise ValueError("eval must be a mapping when present; it is a legacy inline eval config")
        if inline_eval is not None and eval_config_path is not None:
            raise ValueError(
                "eval_config_path and an inline eval block cannot both be set; keep eval_config_path so "
                "candidate edits stay out of generation lineage"
            )

        return cls(
            family="bfcl",
            expt_name=str(data["expt_name"]),
            output_dir=output_dir,
            oracle_pack=oracle_pack,
            oracle_runtime=oracle_runtime,
            lineage=lineage,
            stage=str(data.get("stage", "all")),
            random_seed=data.get("random_seed"),
            ndd_batch_size=int(data.get("ndd_batch_size", 32)),
            input_dir=_resolve_path(data.get("input_dir")),
            schema_version=data.get("schema_version"),
            config_status=config_status,
            surface_generation=dict(sections["surface_generation"]),
            surface_quality_validation=dict(sections["surface_quality_validation"]),
            task_generation=dict(sections["task_generation"]),
            semantic_deduplication_config=dict(sections["semantic_deduplication_config"]),
            exports=dict(sections["exports"]),
            reference_benchmark=reference_benchmark,
            generation_model_config=data.get("generation_model_config"),
            judge_model_config=data.get("judge_model_config"),
            eval_config_path=eval_config_path,
            inline_eval=inline_eval,
            translation_config_path=data.get("translation_config_path"),
            raw=data,
        )
