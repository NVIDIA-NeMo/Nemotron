"""Associated chart facts with explicit series, axes, categories and units."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict
from functools import lru_cache
from decimal import Decimal, DecimalException

from ._chart_schema import parse_chart_target, parse_markdown_table, _split_record_values
from ._normalization import strip_markdown_emphasis, normalize_unicode, preserve_script_markup


def visible_text(value: str) -> str:
    value = preserve_script_markup(normalize_unicode(value))
    value = re.sub(r'</?(?:b|strong|i|em|u|s|del|code)\b[^>]*>', '', value, flags=re.I)
    value = strip_markdown_emphasis(value)
    value = value.replace(r'\_', '_').replace(r'\|', '|')
    return ' '.join(value.split())


def label(value: str) -> str:
    return visible_text(value).casefold()


_CURRENCIES = ('USD', 'EUR', 'CHF', 'GBP', 'JPY', 'CNY', 'INR', 'CAD', 'AUD',
               'NZD', 'SGD', 'HKD', 'BRL', 'IDR', 'KRW', 'RUB', 'ZAR', 'Rp',
               'R$', 'US$', 'C$', 'A$', '$', '€', '£', '¥', '₹', '₩', '₽')
_CURRENCY_RE = re.compile('|'.join(re.escape(x) for x in sorted(_CURRENCIES, key=len, reverse=True)), re.I)
_UNIT = r'(?:%|[a-zA-Zµμ°][a-zA-Z0-9µμ°/^.-]*)'
_AMOUNT = r'(?:(?:\d{1,3}(?:,\d{3})+|\d{1,3}(?: \d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
_SUPERSCRIPTS = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻', '0123456789+-')


def _numeric_text(value: str) -> str:
    value = visible_text(value).replace('−', '-').replace('µ', 'μ')
    # Interpret powers explicitly instead of compatibility-folding 10² to 102.
    value = re.sub(r'[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+', lambda m: '^' + m.group().translate(_SUPERSCRIPTS), value)
    value = re.sub(r'\^\{([+-]?\d+)\}|\^\(([+-]?\d+)\)', lambda m: '^' + next(x for x in m.groups() if x is not None), value)
    return value.replace(r'\times', '×').replace(r'\cdot', '×')


def _field_unit(context: str) -> str:
    context = _numeric_text(context)
    for payload in re.findall(r'\(([^()]*)\)', context):
        if _CURRENCY_RE.fullmatch(payload):
            return _currency(payload)
        if re.fullmatch(_UNIT, payload):
            return payload
    return ''


def _currency(value: str) -> str:
    # Currency codes are aliases only where identity is unambiguous. A bare
    # dollar sign stays unspecified rather than being silently called USD.
    aliases = {'us$': 'USD', 'c$': 'CAD', 'a$': 'AUD', 'r$': 'BRL', 'rp': 'IDR'}
    return aliases.get(value.casefold(), value.upper() if value.isalpha() else value)


def number(value: str, *, context: str = '') -> tuple[float, str] | None:
    """Parse explicit numeric notation, units and currencies without guessing.

    Thousands separators must form groups of three; malformed comma strings
    are rejected. SI unit case is retained. Spaced m means metres, attached M
    may mean million, and explicit field units resolve ambiguous suffixes.
    Compatible-unit conversions and locale-dependent decimal commas are not
    inferred. Superscripts are powers, never concatenated decimal digits.
    """
    value = _numeric_text(value).lstrip('~≈').strip()
    for opener, closer in (('$$', '$$'), (r'\(', r'\)'), (r'\[', r'\]'), ('$', '$')):
        if value.startswith(opener) and value.endswith(closer) and len(value) > len(opener) + len(closer):
            value = value[len(opener):-len(closer)].strip()
            break
    negative = value.startswith('(') and value.endswith(')')
    if negative:
        value = value[1:-1].strip()
    sign = ''
    if value[:1] in {'+', '-'}:
        sign, value = value[0], value[1:].lstrip()
    currency_match = _CURRENCY_RE.match(value)
    currency = ''
    if currency_match:
        currency = _currency(currency_match.group())
        value = value[currency_match.end():].lstrip()
        if value[:1] in {'+', '-'}:
            if sign:
                return None
            sign, value = value[0], value[1:].lstrip()
    # Accounting parentheses are already a sign, so a nested sign is invalid.
    if negative and sign:
        return None
    currency_suffix = '|'.join(re.escape(x) for x in sorted(_CURRENCIES, key=len, reverse=True))
    match = re.fullmatch(rf'({_AMOUNT})(\s*[×*]\s*10\^[+-]?\d+|\s*/\s*{_AMOUNT}|\^[+-]?\d+)?(\s*)({_UNIT}|{currency_suffix})?', value)
    if not match:
        return None
    digits, expression, gap, suffix = match.groups()
    try:
        decimal = Decimal(digits.replace(',', '').replace(' ', ''))
        if expression:
            expression = expression.replace(' ', '')
            if expression.startswith('/'):
                denominator = Decimal(expression[1:].replace(',', ''))
                if not denominator:
                    return None
                decimal /= denominator
            else:
                exponent = int(expression.split('^')[1])
                if abs(exponent) > 1000:
                    return None
                decimal = decimal ** exponent if expression.startswith('^') else decimal * (Decimal(10) ** exponent)
        numeric = float(decimal)
    except (DecimalException, ValueError, OverflowError, ZeroDivisionError):
        return None
    if decimal and not numeric:  # Underflow is not an exact zero.
        return None
    if sign == '-' or negative:
        numeric = -numeric
    field_unit = _field_unit(context)
    if suffix and _CURRENCY_RE.fullmatch(suffix):
        suffix_currency = _currency(suffix)
        if currency and currency != suffix_currency:
            return None
        currency = suffix_currency
        suffix = None
    currency_context = currency or (field_unit if _CURRENCY_RE.fullmatch(field_unit) else '')
    scale = 1.0
    unit = suffix or field_unit
    if suffix == '%':
        numeric /= 100
    elif suffix and (
        (not gap and suffix in {'k', 'K', 'M', 'B', 'bn'})
        or (currency_context and suffix in {'k', 'K', 'm', 'M', 'b', 'B', 'bn'})
    ) and suffix != field_unit:
        scale = {'k': 1e3, 'K': 1e3, 'm': 1e6, 'M': 1e6, 'b': 1e9, 'B': 1e9, 'bn': 1e9}[suffix]
        unit = field_unit
    elif not suffix and field_unit == '%':
        numeric /= 100
    numeric *= scale
    if currency_context:
        unit = currency_context + (':' + unit if unit and unit != currency_context else '')
    return (numeric, unit) if math.isfinite(numeric) else None


def _unconsumed_record_text(text: str) -> str:
    """Recover fields discarded by the pinned production parser.

    This is content accounting, not a parser fork: historical references remain
    scorable and their residual text is compared to predicted residual text.
    """
    residual = []
    seen_singletons = set()
    for raw in text.splitlines():
        match = re.match(r'^(title|subtitle|note(?:\[[^]]+\])?|source(?:\[[^]]+\])?|axis\[[^]]+\]|panel|bind):(?: (.*))?$', raw.strip())
        if not match:
            continue
        kind, payload = match.groups()
        values = _split_record_values(payload or '')
        consumed = 3 if kind in {'panel', 'bind'} else 1
        if kind in {'title', 'subtitle'}:
            if kind in seen_singletons:
                residual.append(kind + ':')
            seen_singletons.add(kind)
            if not values[0]:
                residual.append(kind + ':')
        if len(values) > consumed:
            # Keep delimiters too: an extra empty field is still extra syntax.
            residual.append(kind + ': ' + ' ; '.join(['', *values[consumed:]]))
    return '\n'.join(residual)


_CONTINUOUS_X_TYPES = frozenset({
    'scatter', 'bubble', 'line', 'multi_line', 'step_line', 'stem', 'area',
    'stacked_area', 'normalized_stacked_area', 'histogram', 'density_plot',
    'interval_band', 'dual_axis_line', 'faceted_line',
})


def categorical_field(header: str) -> bool:
    """Conservative explicit categorical roles, including numeric identifiers.

    Free-form column semantics are not guessed: "group revenue" is a measure,
    while "Group", "Region ID", and "product_code" declare identities.
    """
    name = label(re.sub(r'\([^()]*\)', '', header)).replace('_', ' ').strip()
    return name in {'id', 'identifier', 'code', 'category', 'label', 'item', 'name',
                    'stage', 'sector', 'region', 'group', 'series'} or bool(
        re.search(r'(?:^|\s)(?:id|identifier|code)$', name))


@lru_cache(maxsize=1024)
def contract(text: str) -> dict:
    try:
        target = parse_chart_target(text, strict=True)
        # The production parser validates layers; add the same referential
        # integrity checks for series bindings, which it currently only parses.
        panels = {panel.panel_id: f'panel:{i}' for i, panel in enumerate(target.panels)}
        axes = {channel for channel, _ in target.axes}
        seen_bindings = set()
        for binding in target.bindings:
            if binding.panel_id and binding.panel_id not in panels:
                raise ValueError('binding references undeclared panel')
            if binding.axis not in {'x', 'y'} and binding.axis not in axes and f'{binding.panel_id}.{binding.axis}' not in axes:
                raise ValueError('binding references undeclared axis')
            key = (binding.series, binding.panel_id)
            if key in seen_bindings:
                raise ValueError('duplicate series binding')
            seen_bindings.add(key)
        if len(axes) != len(target.axes):
            raise ValueError('duplicate axis channel')
    except (ValueError, IndexError) as error:
        return {'parseable': False, 'error': str(error), 'chart_type': '',
                'legends': Counter(), 'structure': Counter(), 'tables': {}, 'facts': [],
                'unconsumed_text': _unconsumed_record_text(text)}

    def panel_id(value):
        return panels.get(value, value)

    def axis_id(value):
        for original, canonical in panels.items():
            if value.startswith(original + '.'):
                return canonical + value[len(original):]
        return value

    axis_labels = dict(target.axes)
    raw_panels = {canonical: original for original, canonical in panels.items()}
    panel_types = {panel.panel_id: panel.chart_type for panel in target.panels}

    def resolve_axis(value, panel=''):
        scoped = f'{panel}.{value}' if panel and not value.startswith(panel + '.') else value
        raw = scoped if scoped in axis_labels else value
        return axis_id(raw), axis_labels.get(raw, '')

    structure = Counter()
    for name in ('title', 'subtitle'):
        if getattr(target, name):
            structure[(name, label(getattr(target, name)))] += 1
    for channel, value in target.axes:
        structure[('axis:' + axis_id(channel), label(value))] += 1
    for i, panel in enumerate(target.panels):
        structure[(f'panel:{i}', label(panel.chart_type + ' ' + panel.title))] += 1
    for binding in target.bindings:
        structure[('bind:' + label(binding.series), panel_id(binding.panel_id) + ':' + resolve_axis(binding.axis, binding.panel_id)[0])] += 1
    for name in ('notes', 'sources', 'annotations', 'panel_notes', 'panel_sources'):
        for value in getattr(target, name):
            if name.startswith('panel_'):
                value = (panel_id(value[0]), value[1])
            structure[(name, label(str(value)))] += 1
    if target.range_topology:
        for key, value in asdict(target.range_topology).items():
            structure[('range:' + key, value)] += 1
    tables = {('shared', ''): target.table_markdown} if target.table_markdown else {}
    tables.update({('panel', panel_id(table.panel_id)): table.table_markdown for table in target.panel_tables})
    layer_scopes = {}
    for i, layer in enumerate(target.layers):
        scope = (f'layer:{i}', panel_id(layer.panel_id), resolve_axis(layer.axis, layer.panel_id)[0])
        structure[(scope[0], str((layer.mark, layer.role, scope[1:], tuple(sorted(layer.attributes)))))] += 1
        if layer.table_id:
            layer_scopes.setdefault(layer.table_id, []).append(scope)
    for table in target.layer_tables:
        # One table can serve several layers. Count cells once while preserving
        # all declared owners in the fact key.
        tables[('layer', tuple(layer_scopes[table.table_id]))] = table.table_markdown
    legends = Counter((axis_id(g.channel), label(item)) for g in target.legends for item in g.items)
    for guide in target.legends:
        if guide.title:
            structure[('legend_title:' + axis_id(guide.channel), label(guide.title))] += 1

    facts = []
    for scope, markdown in tables.items():
        rows = parse_markdown_table(markdown)
        headers = rows[0]
        raw_panel = raw_panels.get(scope[1], '') if scope[0] == 'panel' else ''
        plot_type = panel_types.get(raw_panel, target.chart_type)
        horizontal = 'horizontal' in plot_type or plot_type in {'funnel', 'tornado'}
        value_channel, dimension_channel = ('x', 'y') if horizontal else ('y', 'x')
        dimension_axis = resolve_axis(dimension_channel, raw_panel)
        dimension_context = headers[0] + ' ' + dimension_axis[1]
        explicit_category = categorical_field(headers[0]) or categorical_field(dimension_axis[1])
        categorical_columns = {i for i, header in enumerate(headers) if categorical_field(header)}
        coordinate_columns = {0}
        if plot_type in _CONTINUOUS_X_TYPES:
            dimension_name = label(re.sub(r'\([^()]*\)', '', dimension_axis[1]).strip())
            coordinate_columns.update(i for i, header in enumerate(headers)
                                      if label(re.sub(r'\([^()]*\)', '', header).strip())
                                      in {dimension_channel, dimension_name} - {''})

        def dimension(i, value):
            # Leading-zero identifiers and explicitly categorical fields remain
            # exact labels even when drawn by a generally continuous mark.
            identifier_like = bool(re.match(r'^[+-]?0\d', visible_text(value)))
            if i in coordinate_columns and plot_type in _CONTINUOUS_X_TYPES and i not in categorical_columns and not (i == 0 and explicit_category) and not identifier_like:
                numeric = number(value, context=headers[i] + ' ' + dimension_axis[1])
                if numeric is not None:
                    return (label(headers[i]), ('numeric_coordinate', numeric[0], numeric[1]))
            return (label(headers[i]), label(value))

        for row in rows[2:]:
            # First column is the dimension (including numeric x coordinates).
            # Other nonnumeric fields carry long-form series/category identity.
            dimensions = tuple(dimension(i, value) for i, value in enumerate(row)
                               if i in coordinate_columns or i in categorical_columns or number(value, context=headers[i]) is None)
            for col, value in enumerate(row[1:], start=1):
                if col in categorical_columns or col in coordinate_columns:
                    continue
                # Bindings refer to series/category labels, never incidental
                # numeric values in another cell of the same row.
                series_labels = {label(headers[col]), *(label(v) for i, v in enumerate(row)
                                  if i != col and number(v, context=headers[i]) is None)}
                matching_bindings = [b for b in target.bindings if label(b.series) in series_labels]
                bindings = tuple(sorted((panel_id(b.panel_id), resolve_axis(b.axis, b.panel_id)[0])
                                        for b in matching_bindings))
                if matching_bindings:
                    resolved_axes = [resolve_axis(b.axis, b.panel_id) for b in matching_bindings]
                elif scope[0] == 'layer':
                    resolved_axes = [(owner[2], next((v for k, v in target.axes if axis_id(k) == owner[2]), '')) for owner in scope[1]]
                else:
                    # Explicitly matching axis labels take precedence over the
                    # conventional value axis. Wide tables represent series;
                    # scatter/bubble auxiliary Size columns need their own unit.
                    matching_axes = [(axis_id(k), v) for k, v in target.axes
                                     if (not raw_panel or '.' not in k or k.startswith(raw_panel + '.'))
                                     if label(re.sub(r'\([^()]*\)', '', v).strip()) == label(re.sub(r'\([^()]*\)', '', headers[col]).strip())]
                    if matching_axes:
                        resolved_axes = matching_axes
                    elif plot_type in {'scatter', 'bubble'} and col > 1:
                        resolved_axes = []
                    else:
                        resolved_axes = [resolve_axis(value_channel, raw_panel)]
                axis_units = tuple(sorted({(axis, _field_unit(axis_label)) for axis, axis_label in resolved_axes}))
                inherited_units = {unit for _, unit in axis_units if unit}
                context = headers[col]
                if not _field_unit(context) and len(inherited_units) == 1:
                    context += ' (' + next(iter(inherited_units)) + ')'
                parsed = number(value, context=context)
                if parsed is None:
                    continue
                key = (scope, dimensions, label(headers[col]), bindings, axis_units, parsed[1])
                facts.append((key, parsed[0]))
    return {'parseable': True, 'error': None, 'chart_type': target.chart_type,
            'legends': legends, 'structure': structure, 'tables': tables, 'facts': facts,
            # Keep unit-bearing text and owner/axis fields losslessly. The
            # normalized visible-text counters are unsuitable for unit lookup.
            'raw_axes': {axis_id(channel): value for channel, value in target.axes},
            'typed_bindings': tuple((label(b.series), panel_id(b.panel_id),
                                     resolve_axis(b.axis, b.panel_id)[0])
                                    for b in target.bindings),
            'unconsumed_text': _unconsumed_record_text(text)}
