#!/usr/bin/env python3
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

"""Extract the error-code vocabulary a local Python source can raise, for review.

A probe plan has to name the code each error case expects, and a case naming a code the
source never raises fails the A1 error probe outright. Reading the codes off the source is
therefore worth more than guessing them, and it is static analysis rather than judgement:
the codes are string literals reaching the position the plan's `error_path` points at.

What this costs is worth stating plainly. A vocabulary taken from the implementation cannot
catch an implementation whose code set is itself wrong, because there is no longer an
independent statement to compare against. What the probe still establishes is the pairing:
which situation raises which code, for arguments chosen from the declared surface alone.
Read the emitted messages before accepting the file; they are the source's own words about
when each code applies, and they are what the drafting model reasons from.

The scan is bounded by what `backend.py` imports, transitively. A code sitting in a test,
a dead module or a vendored sample is not a code the source under probe can raise, and
offering one to the drafting model buys an error case that fails for naming it.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.source_adapters.local_python_reach import walk_source

# Names for the entry beside the code that holds a sentence for a person to read. Tried in
# order and only where the source actually fills it, so a helper that declares `message`
# and never passes one does not win the position on the strength of its name alone. Where
# no name matches, `_choose_message_key` decides by shape instead, which is what keeps a
# source that calls the field something else from going undescribed.
_MESSAGE_KEYS = ("message", "msg", "detail", "details", "reason", "description", "explanation", "text")

_Function = ast.FunctionDef | ast.AsyncFunctionDef


def _template(node: ast.AST) -> str | None:
    """A message literal as a template, with its interpolations left as placeholders."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for piece in node.values:
            if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                parts.append(piece.value)
            elif isinstance(piece, ast.FormattedValue):
                parts.append("{" + ast.unparse(piece.value) + "}")
        return "".join(parts)
    return None


def _dict_value(node: ast.Dict, key: str) -> ast.expr | None:
    for candidate, value in zip(node.keys, node.values, strict=True):
        if isinstance(candidate, ast.Constant) and candidate.value == key:
            return value
    return None


def _resolve(node: ast.expr | None, bound: Mapping[str, ast.Dict]) -> ast.expr | None:
    """A name standing for a dict literal built earlier in the same function."""
    if isinstance(node, ast.Name):
        return bound.get(node.id, node)
    return node


def _envelope_at(
    node: ast.expr | None,
    error_path: Sequence[str],
    bound: Mapping[str, ast.Dict],
) -> ast.Dict | None:
    """The dict literal holding the last segment of the error path, at whatever depth.

    Every segment is followed rather than just the first and the last, so a path of three
    or more parts lands where it says it does instead of reading the third key off the
    dict that holds the second.
    """
    current = _resolve(node, bound)
    for key in error_path[:-1]:
        if not isinstance(current, ast.Dict):
            return None
        current = _resolve(_dict_value(current, key), bound)
    return current if isinstance(current, ast.Dict) else None


def _choose_message_key(offered: Sequence[Mapping[str, str]]) -> str | None:
    """Which entry beside the code carries the sentence a reviewer wants to read.

    Decided once for the whole source, from what the source was seen to write rather than
    from what its helper declares. A helper that takes a code, an entity, a field and a
    message fills the first three at nearly every call and the fourth at most of them, so
    reading the envelope's shape alone picked `entity` and the vocabulary then explained
    `not_found` as the word "accounts".

    A name from the list settles it where one is filled. Where none is, the shape does: a
    message is a sentence and a sentence has a space in it, while an entity or a field name
    is a single token. Two keys that both look like sentences are left undecided, because
    guessing between them is how a vocabulary comes to describe the wrong thing, and the
    placeholder tells the reviewer to write the description instead.
    """
    seen: dict[str, list[str]] = defaultdict(list)
    for candidates in offered:
        for key, template in candidates.items():
            seen[key].append(template)
    for preferred in _MESSAGE_KEYS:
        if seen.get(preferred):
            return preferred
    sentences = [
        key
        for key, templates in seen.items()
        if all(any(character.isspace() for character in template) for template in templates)
    ]
    return sentences[0] if len(sentences) == 1 else None


def _parameters(node: _Function) -> tuple[list[str], set[str]]:
    """The names a call can fill positionally, and every name it can fill at all."""
    positional = [argument.arg for argument in (*node.args.posonlyargs, *node.args.args)]
    return positional, set(positional) | {argument.arg for argument in node.args.kwonlyargs}


def _builders(tree: ast.Module, error_path: Sequence[str]) -> dict[str, tuple[str, dict[str, str]]]:
    """Functions that construct the error envelope, and which parameters they read it from.

    A source rarely writes the envelope at each call site; it has one helper that takes the
    code and wraps it. Finding that helper turns every call to it into a place a literal
    code appears. Reported alongside the code parameter is every other envelope entry the
    helper fills from a parameter of its own, keyed by the entry's name, which is the set
    `_choose_message_key` picks the description out of.
    """
    found: dict[str, tuple[str, dict[str, str]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, _Function):
            continue
        _, every = _parameters(node)
        bound: dict[str, ast.Dict] = {}
        # An entry the helper adds conditionally is not a key of any literal, and a
        # description is exactly the entry a helper adds conditionally.
        grafted: dict[str, str] = {}
        for statement in ast.walk(node):
            if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Dict):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        bound[target.id] = statement.value
            elif isinstance(statement, ast.AnnAssign) and isinstance(statement.value, ast.Dict):
                if isinstance(statement.target, ast.Name):
                    bound[statement.target.id] = statement.value
            elif isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Name):
                if statement.value.id not in every:
                    continue
                for target in statement.targets:
                    if not isinstance(target, ast.Subscript):
                        continue
                    key = target.slice
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        grafted[key.value] = statement.value.id
        for statement in ast.walk(node):
            # The returned expression may be the envelope or a name for one built above it,
            # which `_envelope_at` settles. Insisting on a literal here missed the common
            # shape where a helper assembles the dict, adjusts it, and returns the variable.
            if not isinstance(statement, ast.Return):
                continue
            envelope = _envelope_at(statement.value, error_path, bound)
            if envelope is None:
                continue
            code = _dict_value(envelope, error_path[-1])
            if not isinstance(code, ast.Name) or code.id not in every:
                continue
            candidates = dict(grafted)
            for key, value in zip(envelope.keys, envelope.values, strict=True):
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    continue
                if key.value == error_path[-1]:
                    continue
                if isinstance(value, ast.Name) and value.id in every:
                    candidates[key.value] = value.id
            found[node.name] = (code.id, candidates)
    return found


def _argument(call: ast.Call, function: _Function | None, parameter: str) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == parameter:
            return keyword.value
    if function is not None:
        positional, _ = _parameters(function)
        if parameter in positional:
            index = positional.index(parameter)
            # A bound method drops its receiver from the call's own arguments, so the
            # parameter list is one longer than the argument list it is indexed against.
            if isinstance(call.func, ast.Attribute) and positional and positional[0] in {"self", "cls"}:
                index -= 1
            if 0 <= index < len(call.args):
                return call.args[index]
    return None


def _called_name(call: ast.Call) -> str | None:
    """The bare name of whatever is being called, receiver discarded.

    Discarding it is what lets `self._error(...)` and an `_error` imported from a sibling
    module be recognised as the same helper. Two helpers of the same name in one source
    would be conflated, which is why the caller only trusts a positional argument when the
    name resolves to exactly one definition.
    """
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def extract(
    root: Path,
    backend: Path,
    error_path: tuple[str, ...],
) -> tuple[str | None, dict[str, dict[str, Any]]]:
    """Every literal code reaching the error path, with where and how it is described.

    Only the files the backend actually reaches are read. Sweeping the whole directory was
    the first attempt and it put codes from tests, dead modules and vendored samples into a
    vocabulary the drafting model then treats as raisable, so an error case names a code
    the source under probe will never produce and the case fails for that alone.

    Helpers are gathered across the whole closure before any call is scanned, so a helper
    defined in one module and called from another is recognised.
    """
    root = root.resolve()
    trees = {
        relative: ast.parse((root / relative).read_text(encoding="utf-8"), filename=relative)
        for relative in walk_source(root, backend).modules
    }
    builders: dict[str, tuple[str, dict[str, str]]] = {}
    definitions: dict[str, list[_Function]] = defaultdict(list)
    for tree in trees.values():
        builders.update(_builders(tree, error_path))
        for node in ast.walk(tree):
            if isinstance(node, _Function):
                definitions[node.name].append(node)

    # Gathered before anything is described, because which entry holds the description is
    # settled by looking at all of them at once rather than at one call.
    sightings: list[tuple[str, str, dict[str, str]]] = []
    for relative, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _called_name(node) or ""
                spec = builders.get(name)
                if spec is None:
                    continue
                code_parameter, candidates = spec
                known = definitions[name]
                declaration = known[0] if len(known) == 1 else None
                supplied = _argument(node, declaration, code_parameter)
                if not isinstance(supplied, ast.Constant) or not isinstance(supplied.value, str):
                    continue
                offered: dict[str, str] = {}
                for key, parameter in candidates.items():
                    passed = _argument(node, declaration, parameter)
                    template = _template(passed) if passed is not None else None
                    if template:
                        offered[key] = template
                sightings.append((supplied.value, f"{relative}:{node.lineno}", offered))
            elif isinstance(node, ast.Dict):
                # A source that inlines the envelope instead of routing it through a helper.
                envelope = _envelope_at(node, error_path, {})
                if envelope is None:
                    continue
                code = _dict_value(envelope, error_path[-1])
                if not isinstance(code, ast.Constant) or not isinstance(code.value, str):
                    continue
                offered = {}
                for key, value in zip(envelope.keys, envelope.values, strict=True):
                    if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                        continue
                    if key.value == error_path[-1]:
                        continue
                    template = _template(value)
                    if template:
                        offered[key.value] = template
                sightings.append((code.value, f"{relative}:{node.lineno}", offered))

    message_key = _choose_message_key([offered for _, _, offered in sightings])
    collected: dict[str, dict[str, Any]] = defaultdict(lambda: {"messages": set(), "sites": []})
    for code, site, offered in sightings:
        entry = collected[code]
        entry["sites"].append(site)
        described = offered.get(message_key) if message_key else None
        if described:
            entry["messages"].add(described)

    return message_key, {
        code: {
            "messages": sorted(entry["messages"]),
            "sites": entry["sites"],
            "occurrences": len(entry["sites"]),
        }
        for code, entry in sorted(collected.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Local Python source")
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to write the code-to-meaning file draft_probe_plan consumes",
    )
    parser.add_argument(
        "--error-path",
        default="error.code",
        help="Dotted path the probe plan reads a code from; must match the plan",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    error_path = tuple(part for part in args.error_path.split(".") if part)

    try:
        if not error_path:
            raise ValueError("--error-path must name at least one field")
        backend = source / "backend.py"
        if not backend.is_file():
            raise ValueError(f"no backend.py under {source} to walk the source from")
        message_key, evidence = extract(source, backend, error_path)
        if not evidence:
            raise ValueError(
                f"no literal error code reaches {args.error_path!r} in anything backend.py "
                f"under {source} imports; check the path against the probe plan, or write "
                "the vocabulary by hand"
            )
    except (OSError, SyntaxError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": "fail",
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    # The meaning is the source's own messages, joined. Where a code carries none, the file
    # still needs a line for it, and a placeholder is more honest than a description this
    # script would have to invent.
    vocabulary = {
        code: "; ".join(entry["messages"])
        or "DESCRIBE WHEN THIS APPLIES: the source raises this code without a message"
        for code, entry in evidence.items()
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(vocabulary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    undescribed = sorted(code for code, entry in evidence.items() if not entry["messages"])
    print(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "pass",
                "error_path": list(error_path),
                # Named so a reviewer can tell a vocabulary read off the wrong entry from
                # one the source simply does not describe.
                "message_key": message_key,
                "vocabulary": vocabulary,
                "evidence": evidence,
                "undescribed_codes": undescribed,
                "output": str(args.output) if args.output else None,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if undescribed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
