from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Union, cast, get_args, get_origin, get_type_hints


def unwrap_optional(tp: Any) -> tuple[Any, bool]:
    origin = get_origin(tp)
    if origin not in (Union, UnionType):
        return tp, False

    args = get_args(tp)
    if len(args) == 2 and type(None) in args:
        inner = args[0] if args[1] is type(None) else args[1]
        return inner, True

    return tp, False


def flatten_dataclass(instance: Any, prefix: str = "") -> dict[str, tuple[Any, Any]]:
    """Flatten a (nested) dataclass instance into dotted paths.

    Returns:
        Mapping path -> (value, annotation)
    """
    if not is_dataclass(instance):
        raise TypeError("flatten_dataclass requires a dataclass instance")

    type_hints = get_type_hints(type(instance))

    result: dict[str, tuple[Any, Any]] = {}
    for f in instance.__dataclass_fields__.values():
        name = f.name
        value = getattr(instance, name)
        ann = type_hints.get(name, f.type)
        path = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"

        if is_dataclass(value):
            result.update(flatten_dataclass(value, prefix=path))
        else:
            result[path] = (value, ann)
    return result


def set_dotted_attr(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    target = obj
    for p in parts[:-1]:
        target = getattr(target, p)
    setattr(target, parts[-1], value)


def dataclass_to_primitive_dict(instance: Any) -> dict[str, Any]:
    if isinstance(instance, type) or not is_dataclass(instance):
        raise TypeError("dataclass_to_primitive_dict requires a dataclass instance")
    data = asdict(instance)
    normalized = _normalize_paths(data)
    return cast(dict[str, Any], normalized)


def _normalize_paths(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _normalize_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_paths(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def coerce_value(raw: str, tp: Any) -> Any:
    inner, is_opt = unwrap_optional(tp)
    if is_opt and raw.strip() == "":
        return None

    if inner is bool:
        s = raw.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid bool: {raw}")
    if inner is int:
        return int(raw)
    if inner is float:
        return float(raw)
    if inner is Path:
        return Path(raw)
    if inner in (str, Any):
        return raw

    origin = get_origin(inner)
    if origin in (list, tuple):
        args = get_args(inner)
        elem_type = args[0] if args else str
        items = [s.strip() for s in raw.split(",") if s.strip()]
        coerced = [coerce_value(i, elem_type) for i in items]
        return coerced if origin is list else tuple(coerced)

    return raw
