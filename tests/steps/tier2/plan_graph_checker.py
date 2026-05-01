"""Helpers for validating Mermaid plan graphs against step manifests and types."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
import tomllib


DEFAULT_STEPS_ROOT = Path(__file__).resolve().parents[3] / "src" / "nemotron" / "steps"
STEP_ID_PATTERN = re.compile(
    r"(benchmark|convert|curate|eval|prep|pretrain|rl|sft|synth|translate)/[a-z0-9_]+"
)


@dataclass(frozen=True)
class StepContract:
    """Minimal step contract data needed for plan validation."""

    step_id: str
    consumes: tuple[str, ...]
    produces: tuple[str, ...]


@dataclass(frozen=True)
class MermaidNode:
    """A Mermaid node with a stable identifier and display label."""

    node_id: str
    label: str

    @property
    def step_id(self) -> str:
        match = STEP_ID_PATTERN.search(self.label)
        return match.group(0) if match else self.label.strip()


@dataclass(frozen=True)
class MermaidEdge:
    """A directed Mermaid edge between two nodes."""

    source: str
    target: str
    artifact_type: str | None = None


@dataclass(frozen=True)
class MermaidGraph:
    """Parsed Mermaid flowchart content."""

    nodes: dict[str, MermaidNode]
    edges: tuple[MermaidEdge, ...]


def extract_mermaid_block(plan_md: str) -> str | None:
    """Return the first Mermaid fenced code block from the plan markdown."""
    match = re.search(r"```mermaid\s*(.*?)```", plan_md, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def parse_mermaid_flowchart(plan_md: str) -> MermaidGraph:
    """Parse a Mermaid flowchart and return nodes plus labeled edges."""
    mermaid = extract_mermaid_block(plan_md)
    if not mermaid:
        return MermaidGraph(nodes={}, edges=())

    nodes: dict[str, MermaidNode] = {}
    edges: list[MermaidEdge] = []

    for raw_line in mermaid.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%%"):
            continue
        if _is_mermaid_directive(line):
            continue

        if "-->" in line:
            edge = _parse_edge_line(line)
            if edge is None:
                continue
            source, target, artifact_type, source_label, target_label = edge
            nodes.setdefault(source, MermaidNode(node_id=source, label=source_label or source))
            nodes.setdefault(target, MermaidNode(node_id=target, label=target_label or target))
            edges.append(MermaidEdge(source=source, target=target, artifact_type=artifact_type))
            continue

        node = _parse_endpoint(line)
        if node is not None:
            node_id, label = node
            nodes.setdefault(node_id, MermaidNode(node_id=node_id, label=label or node_id))

    return MermaidGraph(nodes=nodes, edges=tuple(edges))


def check_plan_graph(
    plan_md: str,
    *,
    steps_root: Path | None = None,
    types_path: Path | None = None,
) -> list[str]:
    """Validate a Mermaid plan graph against the step catalog and type graph."""
    graph = parse_mermaid_flowchart(plan_md)
    if not graph.nodes and not graph.edges:
        return []

    root = (steps_root or DEFAULT_STEPS_ROOT).resolve()
    contracts = load_step_contracts(root)
    type_defs = load_types(types_path or root / "types.toml")

    errors: list[str] = []

    for node in graph.nodes.values():
        if node.step_id not in contracts:
            errors.append(f"Unknown step in plan graph: {node.label!r}")

    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = {node_id: 0 for node_id in graph.nodes}
    outdegree: dict[str, int] = {node_id: 0 for node_id in graph.nodes}

    for edge in graph.edges:
        adjacency[edge.source].append(edge.target)
        indegree[edge.target] = indegree.get(edge.target, 0) + 1
        outdegree[edge.source] = outdegree.get(edge.source, 0) + 1
        indegree.setdefault(edge.source, 0)
        outdegree.setdefault(edge.target, 0)

        source = graph.nodes.get(edge.source)
        target = graph.nodes.get(edge.target)
        if source is None or target is None:
            continue

        source_contract = contracts.get(source.step_id)
        target_contract = contracts.get(target.step_id)
        artifact_type = (edge.artifact_type or "").strip()

        if not artifact_type:
            errors.append(f"Missing artifact type label on edge {source.step_id} -> {target.step_id}")
            continue

        if artifact_type not in type_defs:
            errors.append(f"Unknown artifact type {artifact_type!r} on edge {source.step_id} -> {target.step_id}")
            continue

        if source_contract is not None and source_contract.produces:
            if not any(types_related(artifact_type, produced, type_defs) for produced in source_contract.produces):
                errors.append(
                    f"{source.step_id} does not produce {artifact_type!r} for edge to {target.step_id}"
                )

        if target_contract is not None and target_contract.consumes:
            if not any(types_related(artifact_type, consumed, type_defs) for consumed in target_contract.consumes):
                errors.append(
                    f"{source.step_id} outputs {artifact_type!r} but {target.step_id} does not consume a compatible type"
                )

    for node_id, node in graph.nodes.items():
        if indegree.get(node_id, 0) == 0 and outdegree.get(node_id, 0) == 0:
            errors.append(f"Orphan node: {node.step_id}")

    if _has_cycle(graph.nodes.keys(), adjacency):
        errors.append("Plan graph contains a cycle")

    return errors


def count_plan_stages(plan_md: str) -> int:
    """Return the number of unique nodes in the plan Mermaid graph."""
    return len(parse_mermaid_flowchart(plan_md).nodes)


def extract_plan_steps(plan_md: str) -> set[str]:
    """Return the set of step IDs present in the Mermaid graph."""
    graph = parse_mermaid_flowchart(plan_md)
    return {node.step_id for node in graph.nodes.values()}


def load_types(types_path: Path) -> dict[str, dict]:
    """Load types.toml and normalize both supported schema shapes."""
    with types_path.open("rb") as handle:
        raw = tomllib.load(handle)

    if isinstance(raw.get("types"), dict):
        return {
            name: value
            for name, value in raw["types"].items()
            if isinstance(value, dict)
        }

    return {
        name: value
        for name, value in raw.items()
        if isinstance(value, dict) and name != "convert_to"
    }


def load_step_contracts(steps_root: Path) -> dict[str, StepContract]:
    """Load step IDs plus consume/produce type lists from step manifests."""
    contracts: dict[str, StepContract] = {}

    for manifest_path in sorted(steps_root.rglob("step.toml")):
        with manifest_path.open("rb") as handle:
            data = tomllib.load(handle)

        step = data.get("step", {})
        category = str(step.get("category") or manifest_path.parent.parent.name)
        step_id = str(step.get("id") or f"{category}/{manifest_path.parent.name}")

        consumes = tuple(
            str(entry.get("type"))
            for entry in data.get("consumes", [])
            if isinstance(entry, dict) and entry.get("type")
        )
        produces = tuple(
            str(entry.get("type"))
            for entry in data.get("produces", [])
            if isinstance(entry, dict) and entry.get("type")
        )

        contracts[step_id] = StepContract(step_id=step_id, consumes=consumes, produces=produces)

    return contracts


def is_compatible(actual_type: str, expected_type: str, type_defs: dict[str, dict]) -> bool:
    """Return True when actual_type can be used where expected_type is required."""
    if actual_type == expected_type:
        return True
    for ancestor in _ancestor_types(actual_type, type_defs):
        if ancestor == expected_type:
            return True
    return False


def types_related(left: str, right: str, type_defs: dict[str, dict]) -> bool:
    """Return True when two type names are equal or compatible in either direction."""
    return is_compatible(left, right, type_defs) or is_compatible(right, left, type_defs)


def _ancestor_types(type_name: str, type_defs: dict[str, dict]) -> set[str]:
    seen: set[str] = set()
    stack = [type_name]

    while stack:
        current = stack.pop()
        definition = type_defs.get(current, {})
        parents = definition.get("is_a", [])
        if isinstance(parents, str):
            parents = [parents]

        for parent in parents:
            if not isinstance(parent, str) or parent in seen:
                continue
            seen.add(parent)
            stack.append(parent)

    return seen


def _has_cycle(node_ids: object, adjacency: dict[str, list[str]]) -> bool:
    state: dict[str, str] = {}

    def visit(node_id: str) -> bool:
        current = state.get(node_id)
        if current == "visiting":
            return True
        if current == "visited":
            return False

        state[node_id] = "visiting"
        for neighbor in adjacency.get(node_id, []):
            if visit(neighbor):
                return True
        state[node_id] = "visited"
        return False

    return any(visit(str(node_id)) for node_id in node_ids)


def _is_mermaid_directive(line: str) -> bool:
    lowered = line.lower()
    return lowered.startswith(
        ("flowchart", "graph", "classdef", "class ", "style ", "linkstyle", "subgraph", "end")
    )


def _parse_edge_line(
    line: str,
) -> tuple[str, str, str | None, str | None, str | None] | None:
    left, right = line.split("-->", 1)
    source = _parse_endpoint(left)
    if source is None:
        return None

    right = right.strip()
    artifact_type: str | None = None
    if right.startswith("|"):
        closing = right.find("|", 1)
        if closing != -1:
            artifact_type = right[1:closing].strip() or None
            right = right[closing + 1 :].strip()

    target = _parse_endpoint(right)
    if target is None:
        return None

    return source[0], target[0], artifact_type, source[1], target[1]


def _parse_endpoint(text: str) -> tuple[str, str | None] | None:
    match = re.match(
        r'^\s*([A-Za-z0-9_]+)\s*(\[[^\]]*\]|\([^\)]*\)|\{[^\}]*\})?\s*$',
        text,
    )
    if not match:
        return None

    node_id = match.group(1)
    raw_label = match.group(2)
    if raw_label is None:
        return node_id, None

    label = raw_label[1:-1].strip()
    if label.startswith('"') and label.endswith('"'):
        label = label[1:-1]
    return node_id, label.strip() or None
