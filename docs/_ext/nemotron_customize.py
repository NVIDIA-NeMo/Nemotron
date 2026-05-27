from __future__ import annotations

from pathlib import Path
import re
import tomllib

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import ViewList
from sphinx.application import Sphinx


REPO_ROOT = Path(__file__).resolve().parents[2]
GITHUB_BLOB_BASE = "https://github.com/NVIDIA-NeMo/Nemotron/blob/main"
GITHUB_TREE_BASE = "https://github.com/NVIDIA-NeMo/Nemotron/tree/main"


class StepTomlDirective(Directive):
    required_arguments = 1
    has_content = False

    def run(self) -> list[nodes.Node]:
        path = _resolve_repo_path(self.arguments[0])
        if not path.exists():
            return [_error(self, f"step-toml could not find file: {path}")]

        try:
            data = _load_toml(path)
        except Exception as exc:  # pragma: no cover - surfaced in docs build
            return [_error(self, f"step-toml failed to parse {path}: {exc}")]

        step = data.get("step", {})
        container = nodes.container(classes=["nemotron-step-toml"])

        title = str(step.get("name") or path.parent.name)
        container += nodes.rubric(text=title)

        description = step.get("description") or step.get("summary")
        if description:
            container += nodes.paragraph(text=str(description).strip())

        metadata = nodes.field_list()
        _append_field(metadata, "Category", [str(step.get("category", ""))])
        _append_field(metadata, "Tags", _normalize_sequence(step.get("tags")))
        if metadata.children:
            container += metadata

        container.extend(_table_section("Consumes", [
            [
                str(entry.get("type", "")),
                str(entry.get("description", "")),
                _yes_no(entry.get("required", True)),
            ]
            for entry in _table_list(data.get("consumes"))
        ], ["Type", "Description", "Required"]))

        container.extend(_table_section("Produces", [
            [
                str(entry.get("type", "")),
                str(entry.get("description", "")),
            ]
            for entry in _table_list(data.get("produces"))
        ], ["Type", "Description"]))

        container.extend(_table_section("Parameters", [
            [
                str(entry.get("name", "")),
                str(entry.get("description", "")),
                _format_scalar(entry.get("default")),
                _format_sequence(entry.get("choices") or entry.get("values")),
            ]
            for entry in _table_list(data.get("parameters"))
        ], ["Name", "Description", "Default", "Choices"]))

        container.extend(_table_section("Models", [
            [
                str(entry.get("name", "")),
                str(entry.get("description") or entry.get("notes") or ""),
                _format_scalar(entry.get("default")),
                _format_scalar(entry.get("min_gpus")),
            ]
            for entry in _table_list(data.get("models"))
        ], ["Name", "Description", "Default", "Min GPUs"]))

        strategies = _table_list(data.get("strategies"))
        if strategies:
            container += _list_admonition(
                "Strategies",
                "step-strategies",
                [
                    _join_parts(
                        [
                            ("When", entry.get("when")),
                            ("Then", entry.get("then") or entry.get("recommendation")),
                            ("Skill", entry.get("skill")),
                        ]
                    )
                    for entry in strategies
                ],
            )

        errors = _table_list(data.get("errors"))
        if errors:
            container += _list_admonition(
                "Errors",
                "step-errors",
                [
                    _join_parts(
                        [
                            ("Name", entry.get("name") or entry.get("symptom")),
                            ("Cause", entry.get("cause")),
                            ("Recovery", entry.get("recovery") or entry.get("fix")),
                            ("Skill", entry.get("skill")),
                        ]
                    )
                    for entry in errors
                ],
            )

        reference = data.get("reference", {})
        if isinstance(reference, dict) and reference:
            container += nodes.rubric(text="Reference")
            refs = nodes.bullet_list()
            for key, value in reference.items():
                values = _normalize_sequence(value)
                if not values:
                    continue
                item = nodes.list_item()
                item += nodes.paragraph(text=f"{key}:")
                nested = nodes.bullet_list()
                for raw in values:
                    nested_item = nodes.list_item()
                    nested_item += _link_paragraph(raw)
                    nested += nested_item
                item += nested
                refs += item
            if refs.children:
                container += refs

        return [container]


class TypesTomlDirective(Directive):
    required_arguments = 1
    has_content = False

    def run(self) -> list[nodes.Node]:
        path = _resolve_repo_path(self.arguments[0])
        if not path.exists():
            return [_error(self, f"types-toml could not find file: {path}")]

        try:
            data = _load_toml(path)
        except Exception as exc:  # pragma: no cover - surfaced in docs build
            return [_error(self, f"types-toml failed to parse {path}: {exc}")]

        type_entries = _extract_types(data)
        rows = []
        for name in sorted(type_entries):
            entry = type_entries[name]
            converts = []
            for target, step in _normalize_mapping(entry.get("convert_to")).items():
                converts.append(f"{target} ({step})" if step else str(target))
            rows.append([
                name,
                str(entry.get("description", "")),
                _format_sequence(entry.get("is_a")),
                ", ".join(converts),
            ])

        container = nodes.container(classes=["nemotron-types-toml"])
        container += _make_table(
            ["Type", "Description", "Compatible With (is_a)", "Converts To"],
            rows,
        )

        if _mermaid_enabled(self):
            mermaid = _build_type_graph(type_entries)
            if mermaid:
                container += nodes.rubric(text="Type Graph")
                mermaid_container = nodes.container(classes=["nemotron-type-graph"])
                view = ViewList()
                view.append(".. mermaid::", "nemotron_customize")
                view.append("", "nemotron_customize")
                for line in mermaid.splitlines():
                    view.append(f"   {line}", "nemotron_customize")
                self.state.nested_parse(view, self.content_offset, mermaid_container)
                container += mermaid_container

        return [container]


def _resolve_repo_path(argument: str) -> Path:
    path = Path(argument.strip())
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _extract_types(data: dict) -> dict[str, dict]:
    if isinstance(data.get("types"), dict):
        types = {str(name): value for name, value in data["types"].items() if isinstance(value, dict)}
        global_convert = _normalize_mapping(data.get("convert_to"))
        for mapping in global_convert.values():
            if isinstance(mapping, dict):
                source = str(mapping.get("from", ""))
                target = str(mapping.get("to", ""))
                step = str(mapping.get("step", ""))
                if source and target and source in types:
                    convert_map = dict(_normalize_mapping(types[source].get("convert_to")))
                    convert_map[target] = step
                    types[source]["convert_to"] = convert_map
        return types
    return {str(name): value for name, value in data.items() if isinstance(value, dict)}


def _make_table(headers: list[str], rows: list[list[str]]) -> nodes.table:
    table = nodes.table()
    tgroup = nodes.tgroup(cols=len(headers))
    table += tgroup

    for _ in headers:
        tgroup += nodes.colspec(colwidth=1)

    thead = nodes.thead()
    tgroup += thead
    header_row = nodes.row()
    thead += header_row
    for header in headers:
        entry = nodes.entry()
        entry += nodes.paragraph(text=header)
        header_row += entry

    tbody = nodes.tbody()
    tgroup += tbody
    for row_values in rows:
        row = nodes.row()
        tbody += row
        for value in row_values:
            entry = nodes.entry()
            entry += nodes.paragraph(text=value)
            row += entry

    return table


def _table_section(title: str, rows: list[list[str]], headers: list[str]) -> list[nodes.Node]:
    if not rows:
        return []
    return [nodes.rubric(text=title), _make_table(headers, rows)]


def _list_admonition(title: str, css_class: str, items: list[str]) -> nodes.admonition:
    admonition = nodes.admonition(classes=["admonition", css_class])
    admonition += nodes.title(text=title)
    bullet_list = nodes.bullet_list()
    for item_text in items:
        if not item_text:
            continue
        item = nodes.list_item()
        item += _paragraph_with_links(item_text)
        bullet_list += item
    admonition += bullet_list
    return admonition


def _append_field(field_list: nodes.field_list, name: str, values: list[str]) -> None:
    cleaned = [value for value in values if value]
    if not cleaned:
        return

    field = nodes.field()
    field += nodes.field_name(text=name)
    body = nodes.field_body()
    if len(cleaned) == 1:
        body += nodes.paragraph(text=cleaned[0])
    else:
        bullet_list = nodes.bullet_list()
        for value in cleaned:
            item = nodes.list_item()
            item += nodes.paragraph(text=value)
            bullet_list += item
        body += bullet_list
    field += body
    field_list += field


def _normalize_sequence(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _normalize_mapping(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return {}


def _table_list(value: object) -> list[dict]:
    if isinstance(value, list):
        return [entry for entry in value if isinstance(entry, dict)]
    return []


def _format_scalar(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _format_sequence(value: object) -> str:
    return ", ".join(_normalize_sequence(value))


def _yes_no(value: object) -> str:
    return "yes" if bool(value) else "no"


def _join_parts(parts: list[tuple[str, object]]) -> str:
    rendered = []
    for label, value in parts:
        if value is None:
            continue
        text = _format_sequence(value) if isinstance(value, (list, tuple, set)) else str(value).strip()
        if text:
            rendered.append(f"{label}: {text}")
    return "; ".join(rendered)


def _paragraph_with_links(text: str) -> nodes.paragraph:
    paragraph = nodes.paragraph()
    parts = re.split(r"(https?://\S+|(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+/?(?:[A-Za-z0-9_.-]+)?)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("http://") or part.startswith("https://"):
            paragraph += nodes.reference(text=part, refuri=part)
        elif "/" in part and not part.startswith(" ") and not part.endswith(" "):
            paragraph += nodes.reference(text=part, refuri=_repo_link(part))
        else:
            paragraph += nodes.Text(part)
    return paragraph


def _link_paragraph(value: str) -> nodes.paragraph:
    paragraph = nodes.paragraph()
    paragraph += nodes.reference(text=value, refuri=_repo_link(value))
    return paragraph


def _repo_link(value: str) -> str:
    if value.startswith("http://") or value.startswith("https://"):
        return value
    candidate = value.strip()
    if not candidate:
        return value
    if candidate.endswith("/") or Path(candidate).suffix == "":
        return f"{GITHUB_TREE_BASE}/{candidate.rstrip('/')}"
    return f"{GITHUB_BLOB_BASE}/{candidate}"


def _build_type_graph(type_entries: dict[str, dict]) -> str:
    lines = ["flowchart LR"]
    seen_edges: set[str] = set()

    for name in sorted(type_entries):
        entry = type_entries[name]
        for parent in _normalize_sequence(entry.get("is_a")):
            edge = f'    { _node_id(name) }["{name}"] -->|is_a| { _node_id(parent) }["{parent}"]'
            if edge not in seen_edges:
                lines.append(edge)
                seen_edges.add(edge)

        for target, step in _normalize_mapping(entry.get("convert_to")).items():
            label = str(step) if step else "convert"
            edge = (
                f'    { _node_id(name) }["{name}"] -. "{label}" .-> '
                f'{ _node_id(str(target)) }["{target}"]'
            )
            if edge not in seen_edges:
                lines.append(edge)
                seen_edges.add(edge)

    return "\n".join(lines) if len(lines) > 1 else ""


def _node_id(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return f"type_{sanitized}"


def _error(directive: Directive, message: str) -> nodes.error:
    error = nodes.error()
    error += nodes.paragraph(text=message)
    return error


def _insert_after_title(doctree: nodes.document, node: nodes.Node) -> None:
    if doctree.children and isinstance(doctree.children[0], nodes.section):
        section = doctree.children[0]
        if section.children and isinstance(section.children[0], nodes.title):
            section.insert(1, node)
            return
        section.insert(0, node)
        return

    if doctree.children and isinstance(doctree.children[0], nodes.title):
        doctree.insert(1, node)
    else:
        doctree.insert(0, node)


def _frontmatter_admonition(title: str, css_class: str) -> nodes.admonition:
    admonition = nodes.admonition(classes=["admonition", css_class])
    admonition += nodes.title(text=title)
    return admonition


def _inject_frontmatter_admonitions(app: Sphinx, doctree: nodes.document, docname: str) -> None:
    metadata = app.env.metadata.get(docname, {}) or {}

    triggers = _normalize_sequence(metadata.get("triggers"))
    if triggers:
        admonition = _frontmatter_admonition("Pattern Metadata", "pattern-metadata")
        fields = nodes.field_list()
        _append_field(fields, "Triggers", triggers)
        _append_field(fields, "Steps", _normalize_sequence(metadata.get("steps")))
        _append_field(fields, "Confidence", _normalize_sequence(metadata.get("confidence")))
        _append_field(fields, "Tags", _normalize_sequence(metadata.get("tags")))
        admonition += fields
        _insert_after_title(doctree, admonition)

    paper = metadata.get("paper")
    if paper:
        admonition = _frontmatter_admonition("Paper Reference", "paper-reference")
        fields = nodes.field_list()
        paper_value = str(paper)
        if paper_value.startswith("arxiv:"):
            paper_id = paper_value.split(":", 1)[1]
            field = nodes.field()
            field += nodes.field_name(text="Paper")
            body = nodes.field_body()
            paragraph = nodes.paragraph()
            paragraph += nodes.reference(text=paper_value, refuri=f"https://arxiv.org/abs/{paper_id}")
            body += paragraph
            field += body
            fields += field
        else:
            _append_field(fields, "Paper", [paper_value])
        _append_field(fields, "Section", _normalize_sequence(metadata.get("section")))
        _append_field(fields, "Summary", _normalize_sequence(metadata.get("summary")))
        admonition += fields
        _insert_after_title(doctree, admonition)


def _mermaid_enabled(directive: Directive) -> bool:
    env = getattr(directive.state.document.settings, "env", None)
    app = getattr(env, "app", None)
    if app is None:
        return False
    return "sphinxcontrib.mermaid" in app.extensions


def setup(app: Sphinx) -> dict[str, bool]:
    app.add_directive("step-toml", StepTomlDirective)
    app.add_directive("types-toml", TypesTomlDirective)
    app.connect("doctree-resolved", _inject_frontmatter_admonitions)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
