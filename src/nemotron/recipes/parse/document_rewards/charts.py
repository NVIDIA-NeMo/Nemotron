"""Schema-aware plot/chart baselines using the training parser and fact keys."""

from collections import Counter
import math

from ._chart_facts import contract, number, categorical_field
from ._numeric import score_facts
from ._normalization import decoded_cell_text
from ._chart_schema import parse_markdown_table


def _f1(pred, ref):
    p, r = Counter(pred), Counter(ref)
    total = sum(p.values()) + sum(r.values())
    return 2 * sum((p & r).values()) / total if total else 1.0


def _categorical(chart):
    """Keep table headers and values associated even for nonnumeric charts."""
    rows = []
    for scope, table in chart["tables"].items():
        cells = parse_markdown_table(table)
        headers = tuple(decoded_cell_text(h) for h in cells[0])
        for row in cells[2:]:
            if not any(not categorical_field(cells[0][i]) and number(value, context=cells[0][i]) is not None
                       for i, value in enumerate(row[1:], 1)):
                rows.append((scope, headers, tuple(decoded_cell_text(c) for c in row)))
    return rows


def chart_quality(prediction, reference, *, relative_tolerance=0.05,
                  absolute_tolerance=1e-8, metadata_weight=0.1):
    """Score canonical <chart type="..."> targets, not dental odontograms.

    Reuses the training schema parser and series/category/axis/unit fact keys.
    Numeric matches use the training one-to-one tolerance matcher. Metadata
    scales factual credit and cannot make an incorrect fact set perfect.
    This is a baseline, not the full production reward's aggregation policy.
    """
    if reference is None:
        return {"score": None}
    if not math.isfinite(metadata_weight) or not 0 <= metadata_weight <= 1:
        raise ValueError("metadata_weight must be in [0, 1]")
    ref = contract(reference)
    if not ref["parseable"] or not ref["tables"]:
        raise ValueError("Reference must contain a valid canonical chart and data table")
    pred = contract(prediction or "")
    if not pred["parseable"]:
        return dict(score=0.0, prediction_valid=False, factual_f1=0.0)
    def numeric_facts(chart):
        return [(key, (value, key[-1])) for key, value in chart["facts"]]
    numeric = score_facts(numeric_facts(pred), numeric_facts(ref),
                          relative_tolerance=relative_tolerance,
                          absolute_tolerance=absolute_tolerance)
    categorical_pred, categorical_ref = _categorical(pred), _categorical(ref)
    categorical = _f1(categorical_pred, categorical_ref)
    numeric_mass = len(pred["facts"]) + len(ref["facts"])
    categorical_mass = len(categorical_pred) + len(categorical_ref)
    def headers(chart):
        return [(scope, tuple(decoded_cell_text(h) for h in parse_markdown_table(table)[0]))
                for scope, table in chart["tables"].items()]
    factual = ((numeric["primary_f1"] * numeric_mass + categorical * categorical_mass)
               / (numeric_mass + categorical_mass)) if numeric_mass + categorical_mass else _f1(headers(pred), headers(ref))
    plot_type = float(pred["chart_type"] == ref["chart_type"])
    def fields(chart):
        return Counter({(kind, key): count for kind in ("structure", "legends")
                        for key, count in chart[kind].items()})
    metadata = _f1(fields(pred), fields(ref))
    scopes = _f1(pred["tables"].keys(), ref["tables"].keys())
    residual = float(pred["unconsumed_text"] == ref["unconsumed_text"])
    score = factual * plot_type * scopes * residual * (1 - metadata_weight + metadata_weight * metadata)
    return dict(score=score, factual_f1=factual, numeric_f1=numeric["primary_f1"],
                numeric_precision=numeric["precision"], numeric_recall=numeric["recall"],
                categorical_f1=categorical,
                plot_type=plot_type, metadata_f1=metadata, table_presence_f1=scopes,
                residual_exact=residual, prediction_valid=True)
