"""Parser for canonical Parse chart targets."""
from __future__ import annotations

import re
from dataclasses import dataclass


_CHART_OPEN_RE = re.compile(r'^<chart type="([a-z0-9_]+)">$')


_CHANNEL_RE = r"[a-z0-9_.-]+"


_RECORD_RE = re.compile(
    rf"^(title|subtitle|note|source|annotation|panel|layer|bind|range|axis\[({_CHANNEL_RE})\]|legend\[({_CHANNEL_RE})\]):(?: (.*))?$"
)


_PANEL_TABLE_RE = re.compile(rf"^table\[({_CHANNEL_RE})\]:$")


_LAYER_TABLE_RE = re.compile(rf"^layer_table\[({_CHANNEL_RE})\]:$")


_SCOPED_TEXT_RE = re.compile(rf"^(note|source)\[({_CHANNEL_RE})\]:(?: (.*))?$")


_LAYER_TOPOLOGY_ATTRIBUTE_KEYS = frozenset({
    "arrangement", "normalize", "orient", "sign", "category", "sparsity",
    "segment_color", "connection", "align",
})


@dataclass(frozen=True)
class LegendGuide:
    channel: str
    title: str
    items: tuple[str, ...]


@dataclass(frozen=True)
class PanelGuide:
    panel_id: str
    chart_type: str
    title: str = ""


@dataclass(frozen=True)
class SeriesBinding:
    series: str
    panel_id: str = ""
    axis: str = "y"


@dataclass(frozen=True)
class LayerGuide:
    """Semantic identity and scope of one independently rendered chart layer."""

    layer_id: str
    mark: str
    role: str
    axis: str = "y"
    panel_id: str = ""
    table_id: str = ""
    attributes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class RangeTopology:
    """Visible connected-range topology, with compact serialization defaults."""

    connector: str = "line"
    orientation: str = "horizontal"
    summary: str = "none"


@dataclass(frozen=True)
class PanelTable:
    panel_id: str
    table_markdown: str


@dataclass(frozen=True)
class LayerTable:
    table_id: str
    table_markdown: str


@dataclass(frozen=True)
class ChartTarget:
    chart_type: str
    title: str = ""
    subtitle: str = ""
    axes: tuple[tuple[str, str], ...] = ()
    legends: tuple[LegendGuide, ...] = ()
    table_markdown: str = ""
    panels: tuple[PanelGuide, ...] = ()
    bindings: tuple[SeriesBinding, ...] = ()
    layers: tuple[LayerGuide, ...] = ()
    range_topology: RangeTopology | None = None
    notes: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    annotations: tuple[tuple[str, ...], ...] = ()
    panel_tables: tuple[PanelTable, ...] = ()
    layer_tables: tuple[LayerTable, ...] = ()
    panel_notes: tuple[tuple[str, str], ...] = ()
    panel_sources: tuple[tuple[str, str], ...] = ()


def _contains_unescaped(value: str, needle: str) -> bool:
    escaped = False
    for char in value:
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == needle:
            return True
    return False


def _split_record_values(payload: str) -> list[str]:
    # Read historical pipe-delimited V3/V4 targets, while new serialization
    # uses semicolons so the ParseBench Markdown parser sees only the actual
    # data table. Old records always have an unescaped separator pipe.
    delimiter = "|" if _contains_unescaped(payload, "|") else ";"
    values: list[str] = []
    current: list[str] = []
    escaped = False
    for char in payload:
        if escaped:
            current.append({"n": "\n", "r": "\r", "p": "|"}.get(char, char))
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == delimiter:
            values.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if escaped:
        current.append("\\")
    values.append("".join(current).strip())
    return values


def parse_chart_target(text: str, *, strict: bool = True) -> ChartTarget:
    """Parse V3/V4/V5 output; non-strict mode salvages known records and tables."""
    lines = text.strip().splitlines()
    opening_index = next((idx for idx, line in enumerate(lines) if _CHART_OPEN_RE.fullmatch(line.strip())), None)
    if opening_index is None:
        if strict:
            raise ValueError("missing canonical <chart type=...> opener")
        chart_type = "other"
        opening_index = -1
    else:
        chart_type = _CHART_OPEN_RE.fullmatch(lines[opening_index].strip()).group(1)  # type: ignore[union-attr]
    closing_index = next((idx for idx in range(len(lines) - 1, opening_index, -1) if lines[idx].strip() == "</chart>"), None)
    if closing_index is None:
        if strict:
            raise ValueError("missing </chart> closer")
        closing_index = len(lines)
    if strict and (opening_index != 0 or closing_index != len(lines) - 1):
        raise ValueError("chart wrapper must enclose the complete target")

    title = subtitle = ""
    axes: list[tuple[str, str]] = []
    legends: list[LegendGuide] = []
    panels: list[PanelGuide] = []
    bindings: list[SeriesBinding] = []
    layers: list[LayerGuide] = []
    range_topology: RangeTopology | None = None
    notes: list[str] = []
    sources: list[str] = []
    annotations: list[tuple[str, ...]] = []
    table_lines: list[str] = []
    panel_table_lines: dict[str, list[str]] = {}
    panel_table_order: list[str] = []
    layer_table_lines: dict[str, list[str]] = {}
    layer_table_order: list[str] = []
    panel_notes: list[tuple[str, str]] = []
    panel_sources: list[tuple[str, str]] = []
    active_panel_table = ""
    active_layer_table = ""
    for raw in lines[opening_index + 1:closing_index]:
        line = raw.strip()
        if not line:
            continue
        panel_table_match = _PANEL_TABLE_RE.fullmatch(line)
        if panel_table_match:
            active_panel_table = panel_table_match.group(1)
            if active_panel_table in panel_table_lines:
                if strict:
                    raise ValueError(f"duplicate panel table id: {active_panel_table!r}")
                continue
            panel_table_lines[active_panel_table] = []
            panel_table_order.append(active_panel_table)
            active_layer_table = ""
            continue
        layer_table_match = _LAYER_TABLE_RE.fullmatch(line)
        if layer_table_match:
            active_layer_table = layer_table_match.group(1)
            if active_layer_table in layer_table_lines:
                if strict:
                    raise ValueError(f"duplicate layer table id: {active_layer_table!r}")
                continue
            layer_table_lines[active_layer_table] = []
            layer_table_order.append(active_layer_table)
            active_panel_table = ""
            continue
        if line.startswith("|") and line.endswith("|"):
            if active_layer_table:
                layer_table_lines[active_layer_table].append(line)
            elif active_panel_table:
                panel_table_lines[active_panel_table].append(line)
            else:
                table_lines.append(line)
            continue
        active_panel_table = ""
        active_layer_table = ""
        scoped_text_match = _SCOPED_TEXT_RE.fullmatch(line)
        if scoped_text_match:
            kind, panel_id, payload = scoped_text_match.groups()
            values = _split_record_values(payload or "")
            if not values or not values[0]:
                if strict:
                    raise ValueError(f"{kind}[{panel_id}] requires text")
                continue
            (panel_notes if kind == "note" else panel_sources).append((panel_id, values[0]))
            continue
        match = _RECORD_RE.fullmatch(line)
        if not match:
            if strict:
                raise ValueError(f"unknown chart record: {raw!r}")
            continue
        kind, axis_channel, legend_channel, payload = match.groups()
        values = _split_record_values(payload or "")
        if kind == "title":
            if title and strict:
                raise ValueError("duplicate title record")
            title = values[0]
        elif kind == "subtitle":
            if subtitle and strict:
                raise ValueError("duplicate subtitle record")
            subtitle = values[0]
        elif kind == "panel":
            if len(values) < 2 or not values[0] or not values[1]:
                if strict:
                    raise ValueError("panel requires id and chart type")
                continue
            panels.append(PanelGuide(values[0], values[1], values[2] if len(values) > 2 else ""))
        elif axis_channel:
            axes.append((axis_channel, values[0]))
        elif legend_channel:
            if values and values[0].startswith("items="):
                item_values = [values[0][len("items="):], *values[1:]]
                legend_title = ""
            else:
                # Backward compatibility for V3/V4 records, including the
                # historical empty title slot (`; A ; B`).
                item_values = values[1:]
                legend_title = values[0] if values else ""
            if not any(item_values):
                if strict:
                    raise ValueError(f"legend[{legend_channel}] requires at least one item")
                continue
            legends.append(LegendGuide(legend_channel, legend_title, tuple(value for value in item_values if value)))
        elif kind == "bind":
            if len(values) < 3 or not values[0] or not values[2]:
                if strict:
                    raise ValueError("bind requires series, panel id slot, and axis")
                continue
            bindings.append(SeriesBinding(values[0], values[1], values[2]))
        elif kind == "range":
            if range_topology is not None:
                if strict:
                    raise ValueError("duplicate range record")
                continue
            attributes: dict[str, str] = {}
            for value in values:
                key, separator, attribute_value = value.partition("=")
                if not separator or key not in {"connector", "orient", "summary"} or not attribute_value:
                    if strict:
                        raise ValueError(f"invalid range topology attribute: {value!r}")
                    continue
                if key in attributes:
                    if strict:
                        raise ValueError(f"duplicate range topology attribute: {key!r}")
                    continue
                attributes[key] = attribute_value
            range_topology = RangeTopology(
                connector=attributes.get("connector", "line"),
                orientation=attributes.get("orient", "horizontal"),
                summary=attributes.get("summary", "none"),
            )
        elif kind == "layer":
            if len(values) < 6 or not all(values[:4]):
                if strict:
                    raise ValueError(
                        "layer requires id, mark, role, axis, panel id slot, and table id slot"
                    )
                continue
            attributes: list[tuple[str, str]] = []
            seen_attribute_keys: set[str] = set()
            for value in values[6:]:
                key, separator, attribute_value = value.partition("=")
                if (
                    not separator
                    or key not in _LAYER_TOPOLOGY_ATTRIBUTE_KEYS
                    or not attribute_value
                    or not re.fullmatch(_CHANNEL_RE, attribute_value)
                    or key in seen_attribute_keys
                ):
                    if strict:
                        raise ValueError(f"invalid layer topology attribute: {value!r}")
                    continue
                seen_attribute_keys.add(key)
                attributes.append((key, attribute_value))
            layers.append(LayerGuide(*values[:6], attributes=tuple(attributes)))
        elif kind == "note":
            notes.append(values[0])
        elif kind == "source":
            sources.append(values[0])
        elif kind == "annotation":
            if not any(values):
                if strict:
                    raise ValueError("annotation requires at least one value")
                continue
            annotations.append(tuple(values))

    table_markdown = "\n".join(table_lines)
    panel_tables = tuple(
        PanelTable(panel_id, "\n".join(panel_table_lines[panel_id]))
        for panel_id in panel_table_order
    )
    layer_tables = tuple(
        LayerTable(table_id, "\n".join(layer_table_lines[table_id]))
        for table_id in layer_table_order
    )
    if strict and table_markdown and panel_tables:
        raise ValueError("chart target cannot mix a shared table with panel tables")
    if table_markdown:
        parse_markdown_table(table_markdown)
    elif panel_tables:
        for panel_table in panel_tables:
            parse_markdown_table(panel_table.table_markdown)
    elif strict:
        raise ValueError("chart target has no Markdown table")
    for layer_table in layer_tables:
        parse_markdown_table(layer_table.table_markdown)
    if strict and len({panel.panel_id for panel in panels}) != len(panels):
        raise ValueError("duplicate panel id")
    if strict:
        if range_topology is not None:
            if chart_type != "connected_range":
                raise ValueError("range topology is valid only for connected_range targets")
            if range_topology.connector not in {"line", "arrow", "double_arrow", "range"}:
                raise ValueError(f"invalid range connector: {range_topology.connector!r}")
            if range_topology.orientation not in {"horizontal", "vertical"}:
                raise ValueError(f"invalid range orientation: {range_topology.orientation!r}")
            if range_topology.summary not in {"none", "mean"}:
                raise ValueError(f"invalid range summary: {range_topology.summary!r}")
        declared_panel_ids = {panel.panel_id for panel in panels}
        layer_ids = [layer.layer_id for layer in layers]
        if len(layer_ids) != len(set(layer_ids)):
            raise ValueError("duplicate layer id")
        axis_channels = {channel for channel, _label in axes}
        declared_layer_table_ids = {table.table_id for table in layer_tables}
        for layer in layers:
            if not all(re.fullmatch(_CHANNEL_RE, value) for value in (layer.layer_id, layer.mark, layer.role, layer.axis)):
                raise ValueError(f"invalid layer record: {layer!r}")
            if layer.panel_id and layer.panel_id not in declared_panel_ids:
                raise ValueError(f"layer references undeclared panel: {layer.panel_id!r}")
            if layer.table_id and not re.fullmatch(_CHANNEL_RE, layer.table_id):
                raise ValueError(f"invalid layer table id: {layer.table_id!r}")
            if layer.table_id and layer.table_id not in declared_layer_table_ids:
                raise ValueError(f"layer references undeclared layer table: {layer.table_id!r}")
            scoped_axis = f"{layer.panel_id}.{layer.axis}" if layer.panel_id else ""
            if (
                layer.axis not in {"x", "y"}
                and layer.axis not in axis_channels
                and scoped_axis not in axis_channels
            ):
                raise ValueError(f"layer references undeclared axis: {layer.axis!r}")
        referenced_layer_table_ids = {layer.table_id for layer in layers if layer.table_id}
        if declared_layer_table_ids - referenced_layer_table_ids:
            raise ValueError(
                "layer tables must be referenced by a declared layer: "
                f"unreferenced={sorted(declared_layer_table_ids - referenced_layer_table_ids)!r}"
            )
        for panel_id, _value in (*panel_notes, *panel_sources):
            if panel_id not in declared_panel_ids:
                raise ValueError(f"scoped text references undeclared panel: {panel_id!r}")
    if strict and panel_tables:
        table_panel_ids = {table.panel_id for table in panel_tables}
        if table_panel_ids != declared_panel_ids:
            raise ValueError(
                "panel table ids must exactly match declared panels: "
                f"tables={sorted(table_panel_ids)!r}, panels={sorted(declared_panel_ids)!r}"
            )
    if chart_type == "connected_range" and range_topology is None:
        # Legacy connected-range targets omitted topology. Interpret them with
        # the compact record's defaults so old text remains valid and a
        # default-only new target round-trips without an otherwise empty line.
        range_topology = RangeTopology()
    return ChartTarget(
        chart_type=chart_type,
        title=title,
        subtitle=subtitle,
        axes=tuple(axes),
        legends=tuple(legends),
        table_markdown=table_markdown,
        panels=tuple(panels),
        bindings=tuple(bindings),
        layers=tuple(layers),
        range_topology=range_topology,
        notes=tuple(notes),
        sources=tuple(sources),
        annotations=tuple(annotations),
        panel_tables=panel_tables,
        layer_tables=layer_tables,
        panel_notes=tuple(panel_notes),
        panel_sources=tuple(panel_sources),
    )


def chart_target_tables(target: ChartTarget) -> tuple[tuple[str, str], ...]:
    """Expose shared and panel-scoped tables through one consumer API.

    The empty scope identifies a legacy/shared table.  V5 consumers should
    preserve non-empty panel ids through scoring or export instead of
    concatenating tables whose axes may have different semantics.
    """
    if target.panel_tables:
        return tuple((table.panel_id, table.table_markdown) for table in target.panel_tables)
    if target.table_markdown.strip():
        return (("", target.table_markdown),)
    return ()


def chart_target_layer_tables(target: ChartTarget) -> tuple[tuple[str, str], ...]:
    """Expose supplementary data tables keyed by compositional layer table ID."""

    return tuple((table.table_id, table.table_markdown) for table in target.layer_tables)


# Upstream targets.py SHA256: b0473fc1427ce09528cd983b5d2121fe59754f28223863ea4be137c90d3144f3

def parse_markdown_table(markdown: str) -> list[list[str]]:
    # Chart targets may have a small XML-like context preamble ahead of the
    # table.  It carries visible title/axis/legend text but is intentionally
    # not treated as a data row.
    source_lines = markdown.strip().splitlines()
    table_source = [line for line in source_lines if line.strip().startswith("|") and line.strip().endswith("|")]
    if table_source:
        source_lines = table_source
    rows: list[list[str]] = []
    for raw in source_lines:
        line = raw.strip()
        if not line:
            continue
        if not (line.startswith("|") and line.endswith("|")):
            raise ValueError(f"not a markdown table row: {raw!r}")
        cells = []
        current = []
        escaped = False
        for char in line[1:-1]:
            if escaped:
                current.append(char)
                escaped = False
            elif char == "\\":
                escaped = True
                current.append(char)
            elif char == "|":
                cells.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        cells.append("".join(current).strip())
        rows.append(cells)
    if len(rows) < 2:
        raise ValueError("markdown table requires header and separator")
    width = len(rows[0])
    if width == 0:
        raise ValueError("markdown table has zero columns")
    if any(len(row) != width for row in rows):
        raise ValueError("markdown table rows have inconsistent widths")
    if any(cell.replace("-", "").strip() for cell in rows[1]):
        raise ValueError("markdown separator row is malformed")
    return rows
