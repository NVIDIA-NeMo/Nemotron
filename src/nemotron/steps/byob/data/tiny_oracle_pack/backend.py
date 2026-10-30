"""Deterministic tiny_library backend.

Clock and seed arrive only via RunContext. Pipeline owns reset().
"""

from __future__ import annotations

import copy
from typing import Any

_FIXTURE_SNAPSHOT: dict[str, Any] | None = None
_STATE: dict[str, Any] = {}
_CTX: Any = None
_ID_COUNTER = 0


def list_tools() -> list[str]:
    return ["get_book_status", "checkout_book"]


def reset(*, ctx: Any, fixtures: dict | None = None) -> None:
    global _FIXTURE_SNAPSHOT, _STATE, _CTX, _ID_COUNTER
    if fixtures is not None:
        _FIXTURE_SNAPSHOT = copy.deepcopy(fixtures)
    if _FIXTURE_SNAPSHOT is None:
        raise RuntimeError("tiny_library reset requires fixtures on first call")
    _STATE = copy.deepcopy(_FIXTURE_SNAPSHOT)
    _CTX = ctx
    _ID_COUNTER = 0


def get_state() -> dict:
    return copy.deepcopy(_STATE)


def call_tool(name: str, arguments: dict, *, ctx: Any) -> dict:
    global _CTX, _ID_COUNTER
    _CTX = ctx
    if name == "get_book_status":
        return _get_book_status(arguments)
    if name == "checkout_book":
        return _checkout_book(arguments)
    return {
        "error": {
            "code": "invalid_argument",
            "entity": None,
            "id": None,
            "field": "name",
            "message": f"unknown tool {name!r}",
        }
    }


def _find_book(book_id: str) -> dict | None:
    for book in _STATE.get("books", []):
        if book["book_id"] == book_id:
            return book
    return None


def _find_patron(patron_id: str) -> dict | None:
    for patron in _STATE.get("patrons", []):
        if patron["patron_id"] == patron_id:
            return patron
    return None


def _get_book_status(arguments: dict) -> dict:
    book_id = arguments.get("book_id")
    if not isinstance(book_id, str):
        return {
            "error": {
                "code": "invalid_argument",
                "entity": "books",
                "id": None,
                "field": "book_id",
                "message": "book_id must be a string",
            }
        }
    book = _find_book(book_id)
    if book is None:
        return {
            "error": {
                "code": "not_found",
                "entity": "books",
                "id": book_id,
                "field": "book_id",
                "message": f"book {book_id} not found",
            }
        }
    return {
        "book_id": book["book_id"],
        "title": book["title"],
        "status": book["status"],
        "copies": book["copies"],
    }


def _next_loan_id() -> str:
    global _ID_COUNTER
    _ID_COUNTER += 1
    return f"LN-{_ID_COUNTER:06d}"


def _checkout_book(arguments: dict) -> dict:
    book_id = arguments.get("book_id")
    patron_id = arguments.get("patron_id")
    confirm = arguments.get("confirm", False)

    if not isinstance(book_id, str) or not isinstance(patron_id, str):
        return {
            "error": {
                "code": "invalid_argument",
                "entity": None,
                "id": None,
                "field": "book_id" if not isinstance(book_id, str) else "patron_id",
                "message": "book_id and patron_id must be strings",
            }
        }
    if not isinstance(confirm, bool):
        return {
            "error": {
                "code": "invalid_argument",
                "entity": None,
                "id": None,
                "field": "confirm",
                "message": "confirm must be a boolean",
            }
        }

    book = _find_book(book_id)
    if book is None:
        return {
            "error": {
                "code": "not_found",
                "entity": "books",
                "id": book_id,
                "field": "book_id",
                "message": f"book {book_id} not found",
            }
        }
    if _find_patron(patron_id) is None:
        return {
            "error": {
                "code": "not_found",
                "entity": "patrons",
                "id": patron_id,
                "field": "patron_id",
                "message": f"patron {patron_id} not found",
            }
        }

    if not confirm:
        return {
            "status": "awaiting_confirmation",
            "loan_id": None,
            "book_id": book_id,
            "patron_id": patron_id,
        }

    if book["status"] != "available" or book["copies"] <= 0:
        return {
            "status": "rejected_unavailable",
            "loan_id": None,
            "book_id": book_id,
            "patron_id": patron_id,
        }

    loan_id = _next_loan_id()
    book["status"] = "on_loan"
    book["copies"] = max(0, int(book["copies"]) - 1)
    _STATE.setdefault("loans", []).append(
        {
            "loan_id": loan_id,
            "book_id": book_id,
            "patron_id": patron_id,
            "status": "active",
        }
    )
    return {
        "status": "succeeded",
        "loan_id": loan_id,
        "book_id": book_id,
        "patron_id": patron_id,
    }
