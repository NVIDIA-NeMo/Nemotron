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

"""The single fixture filter dialect shared by validation and expansion.

Both stages must agree on what a slot filter means: if validation accepts an
expression that expansion rejects (or the two match different rows), a pack can
pass its gold gate and still fail to expand.

The expression is parsed with Python's own parser and then walked over a closed set
of nodes, rather than split on operator text. Splitting cannot tell an operator from
the same characters inside a string, so a perfectly ordinary value — a title holding
" and ", a code holding "==" — either raised or silently matched nothing.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

_COMPARATORS: dict[type[ast.cmpop], Any] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

MISSING = object()


class FilterError(ValueError):
    """Raised for an expression outside the supported dialect."""


def evaluate_filter(row: dict[str, Any], expr: str | None) -> bool:
    """Evaluate a conjunction of comparisons over one fixture row.

    Supported: a field name on the left, a literal on the right, joined by ``and``,
    with ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``, ``in``, and ``not in``. A field
    the row does not carry makes its comparison false.
    """
    if not expr:
        return True
    return _evaluate(_parse(expr), row, expr)


def _parse(expr: str) -> ast.expr:
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise FilterError(f"filter {expr!r} is not a parsable expression") from exc
    return tree.body


def _evaluate(node: ast.expr, row: dict[str, Any], expr: str) -> bool:
    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, ast.And):
            raise FilterError(f"filter {expr!r} uses 'or'; only 'and' is supported")
        return all(_evaluate(value, row, expr) for value in node.values)
    if isinstance(node, ast.Compare):
        return _compare(node, row, expr)
    raise FilterError(f"filter {expr!r} must be comparisons joined by 'and'")


def _compare(node: ast.Compare, row: dict[str, Any], expr: str) -> bool:
    if len(node.ops) != 1 or len(node.comparators) != 1:
        raise FilterError(f"filter {expr!r} chains comparisons; write them as separate 'and' clauses")
    if not isinstance(node.left, ast.Name):
        raise FilterError(f"filter {expr!r} must compare a field name on the left")
    field = node.left.id
    right = _literal(node.comparators[0], expr)
    value = row.get(field, MISSING)
    op = node.ops[0]

    # A filter only selects rows that actually carry the field it predicates on.
    # Treating a missing field as "not in" a collection would admit incomplete rows
    # even though every other comparator rejects them.
    if value is MISSING:
        return False

    if isinstance(op, (ast.In, ast.NotIn)):
        if not isinstance(right, (list, tuple, set, frozenset, str)):
            raise FilterError(f"filter {expr!r} uses 'in' against a value that is not a collection")
        contained = value in right
        return contained if isinstance(op, ast.In) else not contained

    apply = _COMPARATORS.get(type(op))
    if apply is None:
        raise FilterError(f"filter {expr!r} uses an unsupported operator")
    try:
        return bool(apply(value, right))
    except TypeError as exc:
        raise FilterError(
            f"filter {expr!r} compares {field!r} of type {type(value).__name__} with "
            f"{type(right).__name__}"
        ) from exc


def _literal(node: ast.expr, expr: str) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError) as exc:
        raise FilterError(f"filter {expr!r} has a non-literal right-hand side") from exc
