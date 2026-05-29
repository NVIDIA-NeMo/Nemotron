"""Disclosure-level assertions for Tier 2 golden cases."""

from __future__ import annotations


def check_disclosure(read_log: list[str] | list[dict], expected: dict) -> list[str]:
    """Validate that required files were read and forbidden files were skipped."""
    if not expected:
        return []

    normalized_log = [_normalize_entry(entry) for entry in read_log]
    errors: list[str] = []

    for required in expected.get("must_read", []):
        if not any(required in entry for entry in normalized_log):
            errors.append(f"Expected to read {required!r} but it was missing from the read log")

    for forbidden in expected.get("must_not_read", []):
        if any(forbidden in entry for entry in normalized_log):
            errors.append(f"Read {forbidden!r} unexpectedly")

    return errors


def _normalize_entry(entry: str | dict) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        path = entry.get("path")
        if isinstance(path, str):
            return path
    return str(entry)
