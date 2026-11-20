"""GriTS and optional TEDS scoring for a single HTML/Markdown table or grid."""

from dataclasses import dataclass
import html
from html.parser import HTMLParser

from ._normalization import decoded_cell_text, canonical_unicode_scripts, normalize_visible_text


@dataclass
class _Table:
    rows: int
    columns: int
    cells: list  # row, column, rowspan, colspan, text

    def grids(self):
        import numpy as np
        content = np.empty((self.rows, self.columns), dtype=object)
        topology = np.empty_like(content)
        for row, column, rowspan, colspan, text in self.cells:
            for r in range(row, row + rowspan):
                for c in range(column, column + colspan):
                    content[r, c] = text
                    topology[r, c] = (column - c, row - r,
                                      column - c + colspan, row - r + rowspan)
        return content, topology

    def html(self):
        rows = [[] for _ in range(self.rows)]
        for row, _, rowspan, colspan, text in self.cells:
            rows[row].append(f'<td rowspan="{rowspan}" colspan="{colspan}">{html.escape(text)}</td>')
        return "<html><body><table>" + "".join("<tr>" + "".join(r) + "</tr>" for r in rows) + "</table></body></html>"


class _Reader(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.rows, self.cell = [], [], None
        self.tables = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"table", "caption", "thead", "tbody", "tfoot", "tr", "td", "th"}:
            if tag == "table":
                self.tables += 1
                if self.tables != 1 or self.stack:
                    raise ValueError("Expected one non-nested table")
            elif "table" not in self.stack:
                raise ValueError("Table element outside table")
            if tag in {"caption", "thead", "tbody", "tfoot"} and self.stack[-1] != "table":
                raise ValueError("Table section must be directly inside the table")
            if tag == "tr":
                if self.stack[-1] not in {"table", "thead", "tbody", "tfoot"}:
                    raise ValueError("Nested table rows")
                self.rows.append([])
            if tag in {"td", "th"}:
                if not self.stack or self.stack[-1] != "tr" or self.cell is not None:
                    raise ValueError("Cell must be inside a row")
                values = dict(attrs)
                rowspan, colspan = int(values.get("rowspan", 1)), int(values.get("colspan", 1))
                if min(rowspan, colspan) < 1:
                    raise ValueError("Spans must be positive integers")
                self.cell = [rowspan, colspan, []]
            self.stack.append(tag)
        elif self.cell is not None:
            if tag in {"sup", "sub"}:
                self.cell[2].append("^{" if tag == "sup" else "_{")
            elif tag == "br":
                self.cell[2].append(" ")

    def handle_endtag(self, tag):
        if tag in {"table", "caption", "thead", "tbody", "tfoot", "tr", "td", "th"}:
            if not self.stack or self.stack.pop() != tag:
                raise ValueError("Unbalanced table markup")
            if tag in {"td", "th"}:
                rowspan, colspan, parts = self.cell
                self.rows[-1].append((rowspan, colspan, decoded_cell_text(canonical_unicode_scripts("".join(parts)))))
                self.cell = None
        elif self.cell is not None and tag in {"sup", "sub"}:
            self.cell[2].append("}")

    def handle_data(self, data):
        if self.cell is not None:
            self.cell[2].append(data)
        elif data.strip() and "caption" not in self.stack:
            raise ValueError("Text outside table cells")


def _parse(value, max_cells):
    if isinstance(value, str):
        if "<table" in value.lower():
            reader = _Reader()
            reader.feed(value)
            reader.close()
            if reader.stack or reader.tables != 1:
                raise ValueError("Incomplete table")
            rows = reader.rows
        else:
            from ._chart_schema import parse_markdown_table
            parsed = parse_markdown_table(value)
            rows = [[(1, 1, normalize_visible_text(cell, output_mode="markdown").replace(r"\|", "|"))
                     for cell in row] for row in [parsed[0], *parsed[2:]]]
    else:
        if value is None or isinstance(value, (dict, bytes)):
            raise ValueError("Expected a rectangular grid")
        rows = []
        for row in value:
            if isinstance(row, (str, bytes, dict)):
                raise ValueError("Grid rows must be sequences of cells")
            rows.append([(1, 1, decoded_cell_text(canonical_unicode_scripts(str(cell)))) for cell in row])
    if not rows or not any(rows):
        raise ValueError("Table has no cells")
    cells, occupied = [], set()
    for r, row in enumerate(rows):
        c = 0
        for rowspan, colspan, text in row:
            while (r, c) in occupied:
                c += 1
            if r + rowspan > len(rows) or rowspan * colspan + len(occupied) > max_cells:
                raise ValueError("Invalid span or table exceeds max_cells")
            slots = {(rr, cc) for rr in range(r, r + rowspan) for cc in range(c, c + colspan)}
            if slots & occupied:
                raise ValueError("Overlapping table cells")
            occupied.update(slots)
            cells.append((r, c, rowspan, colspan, text))
            c += colspan
    width = max(c for _, c in occupied) + 1
    if len(rows) * width > max_cells or len(occupied) != len(rows) * width:
        raise ValueError("Ragged or oversized table")
    return _Table(len(rows), width, cells)


def _inputs(prediction, reference, max_cells):
    if not isinstance(max_cells, int) or max_cells < 1:
        raise ValueError("max_cells must be a positive integer")
    try:
        ref = _parse(reference, max_cells)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"Unscorable reference table: {error}") from error
    try:
        pred = _parse(prediction, max_cells)
    except (ValueError, TypeError, OverflowError):
        pred = None
    return pred, ref


def table_quality(prediction, reference, *, content_weight=0.8, max_cells=10000):
    """Return reference GriTS-Con and GriTS-Top with an explicit weighted score.

    Accept one HTML table (with spans), Markdown table, or rectangular string
    grid. Convert LaTeX before calling. Invalid predictions score zero; invalid
    references raise. max_cells bounds grid expansion, not a metric fallback.
    """
    if reference is None:
        return {"score": None}
    if not 0 <= content_weight <= 1:
        raise ValueError("content_weight must be in [0, 1]")
    pred, ref = _inputs(prediction, reference, max_cells)
    if pred is None:
        return dict(score=0.0, grits_con=0.0, grits_top=0.0, prediction_valid=False)
    if pred == ref:
        con = top = 1.0
    else:
        from ._grits import grits_con, grits_top
        rc, rt = ref.grids()
        pc, pt = pred.grids()
        con, top = float(grits_con(rc, pc)[0]), float(grits_top(rt, pt)[0])
    return dict(score=content_weight * con + (1 - content_weight) * top,
                grits_con=con, grits_top=top, prediction_valid=True)


def table_teds(prediction, reference, *, structure_only=False, max_cells=10000):
    """Optional PubTabNet TEDS on normalized table HTML; same input forms."""
    if reference is None:
        return {"score": None}
    pred, ref = _inputs(prediction, reference, max_cells)
    if pred is None:
        return dict(score=0.0, teds=0.0, prediction_valid=False)
    from ._teds import TEDS
    score = float(TEDS(structure_only=structure_only).evaluate(pred.html(), ref.html()))
    return dict(score=score, teds=score, prediction_valid=True)
