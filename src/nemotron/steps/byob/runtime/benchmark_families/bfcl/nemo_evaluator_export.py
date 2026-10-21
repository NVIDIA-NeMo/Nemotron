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

"""The ``nemo_evaluator_bundle`` writer: canonical projection to a launcher bundle.

The bundle prepares evaluation input and nothing else. It does not run a candidate
model and it publishes no metric, so every file here describes what to ask and how
to compare an answer, never what a model scored.

Six files, all under one directory so the bundle can be moved or archived whole:

``bundle.json``
    The descriptor a launcher reads first. It names the other files, pins the
    dataset's hash and record count, and cites the benchmark it came from.
``dataset.jsonl``
    One record per line, in publication order.
``dataset.schema.json``
    Derived from the record model rather than written by hand, so the schema
    cannot drift away from the records beside it.
``metadata.json``
    Pack, run, and shape descriptors a report can slice on.
``evaluator.yaml``
    The native adapter input contract and scoring policy. It is intentionally not a
    standalone Launcher run config: endpoint, registered environment, and tool
    resource service belong to model evaluation rather than dataset publication.
``system_prompts.json``
    The prompt catalog, keyed by ``system_prompt_id``.

Unlike the BFCL question record, a bundle record keeps the whole rendered trace
including the recorded tool results. A multi-turn function-calling evaluation has
to hand the model the result of the call it just made in order to reach the next
turn, so the results are input here, not an answer key. What the bundle does not
carry is any way to re-execute the pack's tools, which is why the declared metrics
stop at what a scorer can check from the records alone.

Bytes are deterministic for the same reasons as the BFCL writer: sorted keys, no
incidental whitespace, ``\\n`` endings, and UTF-8 left unescaped. The whole bundle
is encoded, digested, and validated before any file is created, so a projection
that cannot be expressed leaves nothing behind to be mistaken for a bundle.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictStr, field_validator, model_validator

from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    EXPORT_DIRECTORY,
    NEMO_EVALUATOR_SCHEMA_VERSION,
    ContentHash,
    ExportScoringMetric,
    NemoEvaluatorBundle,
    NemoEvaluatorRecord,
    NemoEvaluatorScoring,
    NemoEvaluatorSource,
    NonNegativeInt,
    export_content_hash,
    export_tree_hash,
    relative_export_path,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_projection import (
    CanonicalExportProjection,
    ProjectionSource,
    projection_lineage,
)

NEMO_EVALUATOR_ROOT = f"{EXPORT_DIRECTORY}/nemo_evaluator_bundle"
NEMO_BUNDLE_FILE = "bundle.json"
NEMO_DATASET_FILE = "dataset.jsonl"
NEMO_DATASET_SCHEMA_FILE = "dataset.schema.json"
NEMO_METADATA_FILE = "metadata.json"
NEMO_EVALUATOR_CONFIG_FILE = "evaluator.yaml"
NEMO_SYSTEM_PROMPT_FILE = "system_prompts.json"
NEMO_BUNDLE_FILES = (
    NEMO_BUNDLE_FILE,
    NEMO_DATASET_FILE,
    NEMO_DATASET_SCHEMA_FILE,
    NEMO_METADATA_FILE,
    NEMO_EVALUATOR_CONFIG_FILE,
    NEMO_SYSTEM_PROMPT_FILE,
)

# What a scorer can check from the bundle alone. ``results`` and ``task_success``
# are deliberately absent: both need the pack's tools re-executed against oracle
# state, and a bundle that declared them would invite a harness to compare a live
# model against the recorded results of one backend revision instead.
BUNDLE_SCORING_METRICS: tuple[ExportScoringMetric, ...] = ("tool_selection", "arguments")
BUNDLE_ORDERING_METRIC: ExportScoringMetric = "call_ordering"


class NemoEvaluatorWriteError(RuntimeError):
    """The NeMo Evaluator bundle could not be written from the canonical projection."""


class NemoEvaluatorArtifact(BaseModel):
    """What the writer put on disk, for validation and the manifest to cite."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = NEMO_EVALUATOR_SCHEMA_VERSION
    format: Literal["nemo_evaluator_bundle"] = "nemo_evaluator_bundle"
    root: StrictStr = NEMO_EVALUATOR_ROOT
    files: tuple[StrictStr, ...] = NEMO_BUNDLE_FILES
    rows: NonNegativeInt
    content_hash: ContentHash
    source: ProjectionSource

    @field_validator("root")
    @classmethod
    def validate_root(cls, value: str) -> str:
        return relative_export_path(value, label="root")

    @field_validator("files")
    @classmethod
    def validate_files(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != NEMO_BUNDLE_FILES:
            missing = sorted(set(NEMO_BUNDLE_FILES) - set(value))
            extra = sorted(set(value) - set(NEMO_BUNDLE_FILES))
            duplicates = sorted({name for name in value if value.count(name) > 1})
            raise ValueError(
                "a bundle must list its canonical files exactly once, in canonical order "
                f"(missing={missing}, extra={extra}, duplicates={duplicates})"
            )
        return tuple(relative_export_path(name, label="bundle file") for name in value)

    @model_validator(mode="after")
    def validate_artifact(self) -> NemoEvaluatorArtifact:
        if not self.rows:
            raise ValueError("an evaluator bundle with no record asks nothing")
        if self.rows != self.source.rows:
            raise ValueError("the bundle must carry every row of the benchmark it projected")
        return self

    @property
    def bundle_file(self) -> str:
        """The descriptor's path, relative to the run directory.

        ``files`` and ``content_hash`` are bundle-relative, matching the descriptor,
        so that moving the bundle does not change its digest. Anything that needs a
        run-relative path composes it from ``root`` deliberately, as here.
        """
        return f"{self.root}/{NEMO_BUNDLE_FILE}"


def bundle_task_name(pack_id: str) -> str:
    """Derive the launcher task id a bundle registers under.

    A launcher task id travels through shells, filenames, and config keys, so it
    is narrower than a pack id. Normalizing is recorded in ``metadata.json``
    beside the verbatim ``pack_id``, which keeps the rename auditable rather than
    silent; a pack id with nothing usable in it is refused instead of guessed at.
    """
    source = pack_id.strip()
    normalized = re.sub(r"[^a-z0-9_.-]+", "_", source.lower()).strip("_")
    normalized = re.sub(r"^[^a-z0-9]+", "", normalized)
    if normalized != source.lower():
        # Normalization is many-to-one (``bank/vn`` and ``bank:vn`` otherwise
        # collide), and a valid pack id may be entirely non-ASCII. The digest
        # makes both cases deterministic and globally distinct without narrowing
        # the oracle-pack contract just for one exporter.
        prefix = normalized or "pack"
        normalized = f"{prefix}-{hashlib.sha256(source.encode('utf-8')).hexdigest()[:12]}"
    return normalized


def bundle_scoring(projection: CanonicalExportProjection) -> NemoEvaluatorScoring:
    """Declare only what the projected rows let a scorer measure.

    ``call_ordering`` is claimed when some task expects more than one call. A
    benchmark of single-call tasks has no order to grade, and declaring the metric
    anyway would report a perfect ordering score that measured nothing.
    """
    metrics: list[ExportScoringMetric] = list(BUNDLE_SCORING_METRICS)
    if any(len(row.expected_tool_calls) > 1 for row in projection.rows):
        metrics.append(BUNDLE_ORDERING_METRIC)
    return NemoEvaluatorScoring(
        metrics=tuple(metrics),
        call_order_policies=tuple(sorted({row.call_order for row in projection.rows})),
    )


def system_prompt_catalog(projection: CanonicalExportProjection) -> dict[str, str]:
    """Collect the prompt each ``system_prompt_id`` names, or refuse the ambiguity.

    One id mapping to two texts would make the catalog unusable: a launcher
    resolving the id would prompt some rows with text they were not built from.
    """
    catalog: dict[str, str] = {}
    # A pack that renders no system prompt is not an error: the export contract
    # allows it and the record carries its own conversation either way. Such an id
    # stays out of the catalog rather than mapping to empty text, but it may not
    # also name a prompt somewhere else in the same benchmark.
    promptless: set[str] = set()
    for row in projection.rows:
        prompts = {message.content for message in row.messages if message.role == "system" and message.content}
        if len(prompts) > 1:
            raise NemoEvaluatorWriteError(
                f"task {row.task_id!r} carries {len(prompts)} system prompts; a bundle resolves one prompt per record"
            )
        if not prompts:
            promptless.add(row.system_prompt_id)
            continue
        prompt = prompts.pop()
        existing = catalog.setdefault(row.system_prompt_id, prompt)
        if existing != prompt:
            raise NemoEvaluatorWriteError(
                f"system_prompt_id {row.system_prompt_id!r} names two different prompts, "
                "so the bundle catalog cannot resolve it"
            )
    both = sorted(promptless & set(catalog))
    if both:
        raise NemoEvaluatorWriteError(
            f"system_prompt_id(s) {both} name a prompt for some tasks and none for others, "
            "so the bundle catalog cannot resolve them"
        )
    return catalog


def bundle_metadata(projection: CanonicalExportProjection, *, task_name: str) -> dict[str, Any]:
    """Describe the benchmark's shape and lineage for a report to slice on."""
    return {
        "schema_version": NEMO_EVALUATOR_SCHEMA_VERSION,
        "task_name": task_name,
        "records": len(projection.rows),
        # Nothing here is scored. It is what a reader needs to say which pack,
        # which run, and which conversation shapes a later metric came from.
        "projection": projection_lineage(projection),
        "counts": {
            "expected_tool_calls": sum(len(row.expected_tool_calls) for row in projection.rows),
            "tasks_without_expected_calls": sum(1 for row in projection.rows if not row.expected_tool_calls),
            "tasks_with_assertions": sum(1 for row in projection.rows if row.success_assertions),
        },
        "categories": sorted({row.category for row in projection.rows if row.category}),
        "difficulties": sorted({row.difficulty for row in projection.rows if row.difficulty}),
    }


def evaluator_config(
    projection: CanonicalExportProjection,
    *,
    task_name: str,
    scoring: NemoEvaluatorScoring,
) -> dict[str, Any]:
    """Build the input contract a native NeMo Evaluator adapter must implement.

    ``prompt.source`` names the answer-free seed explicitly. The complete trace is
    retained for audit and scoring under ``reference_trace`` but is forbidden as
    model input; later user turns and recorded tool results are released only by
    an adapter following ``replay_steps``.
    """
    return {
        "schema_version": NEMO_EVALUATOR_SCHEMA_VERSION,
        "kind": "nemotron_byob_function_calling_input",
        "execution": {
            "direct_launcher_config": False,
            "requires_registered_environment": True,
            "requires_tool_resource_service": True,
        },
        "task": {
            "name": task_name,
            "type": "function_calling",
            "dataset": NEMO_DATASET_FILE,
            "dataset_schema": NEMO_DATASET_SCHEMA_FILE,
        },
        "prompt": {
            "source": "seed_messages",
            "system_prompts": NEMO_SYSTEM_PROMPT_FILE,
            "reference_trace": "reference_trace",
            "reference_trace_is_model_input": False,
        },
        "interaction": {
            "type": "incremental_tool_replay",
            "steps_field": "replay_steps",
            "release_next_user_after_current_step": True,
            "release_tool_results_after_expected_call_match": True,
        },
        "scoring": {
            "metrics": list(scoring.metrics),
            "argument_match": scoring.argument_match,
            "call_order_policies": list(scoring.call_order_policies),
            "call_order_field": "call_order",
            "call_order_prefix_field": "call_order_prefix",
        },
        "reporting": {
            "group_by": ["turn_policy", "category", "difficulty"],
            "languages": list(projection.provenance.languages),
        },
    }


def bundle_contents(projection: CanonicalExportProjection) -> dict[str, bytes]:
    """Encode every bundle file, before any of them exists.

    Building the whole bundle in memory is what keeps a partial one off disk: a
    projection that cannot be encoded — an unresolvable prompt id, a row no
    evaluator record can represent — fails with nothing written, so a reader never
    finds five of six files and no way to tell which is missing.
    """
    task_name = bundle_task_name(projection.provenance.pack_id)
    scoring = bundle_scoring(projection)
    try:
        records = [NemoEvaluatorRecord.from_canonical(row) for row in projection.rows]
    except ValueError as exc:
        raise NemoEvaluatorWriteError(f"a published row cannot be written as an evaluator record: {exc}") from exc

    contents = {
        NEMO_DATASET_FILE: "".join(f"{_json_line(record.model_dump(mode='json'))}\n" for record in records),
        NEMO_DATASET_SCHEMA_FILE: _json_document(NemoEvaluatorRecord.model_json_schema()),
        NEMO_METADATA_FILE: _json_document(bundle_metadata(projection, task_name=task_name)),
        NEMO_EVALUATOR_CONFIG_FILE: _yaml_document(evaluator_config(projection, task_name=task_name, scoring=scoring)),
        NEMO_SYSTEM_PROMPT_FILE: _json_document(system_prompt_catalog(projection)),
    }
    encoded = {name: text.encode("utf-8") for name, text in contents.items()}
    try:
        bundle = NemoEvaluatorBundle(
            task_name=task_name,
            dataset_file=NEMO_DATASET_FILE,
            dataset_schema_file=NEMO_DATASET_SCHEMA_FILE,
            metadata_file=NEMO_METADATA_FILE,
            evaluator_config_file=NEMO_EVALUATOR_CONFIG_FILE,
            system_prompt_file=NEMO_SYSTEM_PROMPT_FILE,
            record_count=len(records),
            # Digested from the bytes about to be written rather than from a file,
            # so the descriptor is validated before the bundle exists.
            dataset_content_hash=export_content_hash({NEMO_DATASET_FILE: encoded[NEMO_DATASET_FILE]}),
            scoring=scoring,
            source=NemoEvaluatorSource(
                benchmark_file=projection.source.file,
                benchmark_content_hash=projection.source.content_hash,
                pack_id=projection.provenance.pack_id,
                pack_version=projection.provenance.pack_version,
                expt_name=projection.provenance.expt_name,
            ),
        )
    except ValueError as exc:
        raise NemoEvaluatorWriteError(f"the bundle descriptor is not publishable: {exc}") from exc
    encoded[NEMO_BUNDLE_FILE] = _json_document(bundle.model_dump(mode="json")).encode("utf-8")
    if set(encoded) != set(NEMO_BUNDLE_FILES):
        raise NemoEvaluatorWriteError(f"the bundle encoder produced {sorted(encoded)}, not the declared bundle")
    return encoded


def write_nemo_evaluator_bundle(
    projection: CanonicalExportProjection,
    run_directory: Path,
) -> NemoEvaluatorArtifact:
    """Write the bundle and describe what was written.

    The artifact reports paths, counts, and a tree hash. It does not claim the
    bundle is equivalent to the benchmark; reading the files back and proving that
    is a separate step, so a writer cannot certify its own output.
    """
    contents = bundle_contents(projection)
    content_hash = export_content_hash(contents)
    root = run_directory / NEMO_EVALUATOR_ROOT
    # The whole directory, not just the six names: a file this run does not write
    # would otherwise ship inside a bundle whose digest never covered it.
    if root.exists():
        shutil.rmtree(root)
    for name, payload in sorted(contents.items()):
        _write_bytes(root / name, payload)
    written = export_tree_hash(root, NEMO_BUNDLE_FILES)
    if written != content_hash:
        raise NemoEvaluatorWriteError(
            "the bundle on disk does not match the bytes that were encoded, "
            "so the write was truncated or something else wrote beside it"
        )
    return NemoEvaluatorArtifact(
        rows=len(projection.rows),
        content_hash=content_hash,
        source=projection.source,
    )


def read_nemo_evaluator_bundle(
    run_directory: Path,
    artifact: NemoEvaluatorArtifact,
) -> tuple[NemoEvaluatorBundle, list[dict[str, Any]]]:
    """Read the descriptor and dataset back, so the bundle decodes as it was written."""
    root = run_directory / artifact.root
    for name in artifact.files:
        if not (root / name).is_file():
            raise NemoEvaluatorWriteError(f"the evaluator bundle is missing {name}")
    try:
        bundle = NemoEvaluatorBundle.model_validate(_read_json(root / NEMO_BUNDLE_FILE))
    except ValueError as exc:
        raise NemoEvaluatorWriteError(f"{NEMO_BUNDLE_FILE} is not a valid bundle descriptor: {exc}") from exc
    records = _read_dataset(root / bundle.dataset_file)
    if len(records) != bundle.record_count:
        raise NemoEvaluatorWriteError(
            f"{bundle.dataset_file} holds {len(records)} record(s) but the descriptor claims {bundle.record_count}"
        )
    dataset_hash = export_tree_hash(root, (bundle.dataset_file,))
    if dataset_hash != bundle.dataset_content_hash:
        raise NemoEvaluatorWriteError(f"{bundle.dataset_file} changed after the descriptor pinned its hash")
    # The descriptor pins only the dataset, so without this an edited evaluator
    # config or metadata file would read back as though the writer had produced it.
    if export_tree_hash(root, artifact.files) != artifact.content_hash:
        raise NemoEvaluatorWriteError("the evaluator bundle no longer matches the tree hash the writer reported")
    return bundle, records


def _json_line(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _json_document(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _yaml_document(value: Any) -> str:
    import yaml

    # width is pinned because line wrapping would otherwise depend on a default
    # that can differ between PyYAML releases, and the bundle's digest covers it.
    return yaml.safe_dump(value, sort_keys=True, allow_unicode=True, default_flow_style=False, width=1_000_000)


def _write_bytes(path: Path, payload: bytes) -> None:
    # Bytes rather than text: the encoding and the "\n" endings are decided once,
    # where the digest is taken, instead of by a host's newline translation.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NemoEvaluatorWriteError(f"{path.name} is not valid JSON: {exc}") from exc


def _read_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        # Reachable when the descriptor names a dataset the artifact does not list.
        raise NemoEvaluatorWriteError(f"the evaluator bundle is missing {path.name}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                raise NemoEvaluatorWriteError(f"{path.name} line {number} is blank; JSONL carries one record per line")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise NemoEvaluatorWriteError(f"{path.name} line {number} is not valid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise NemoEvaluatorWriteError(f"{path.name} line {number} is not a JSON object")
            records.append(record)
    return records
