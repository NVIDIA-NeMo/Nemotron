"""Size parsing and formatting utilities."""

import math
import re
from typing import Union

_SIZE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*([KMGTPEZY]i?)?B?", re.IGNORECASE)
_BYTE_SIZES = {
    "": 1,
    "K": 1024,
    "M": 1024**2,
    "G": 1024**3,
    "T": 1024**4,
    "P": 1024**5,
    "E": 1024**6,
}


def parse_byte_size(size: Union[int, float, str]) -> int:
    """Parse human-readable size to bytes.

    Accepts integers, floats, or strings with optional unit suffixes.
    Supports both SI-style (MB) and IEC-style (MiB) notations.

    Args:
        size: Size specification. Examples:
            - 1024 (integer bytes)
            - "256MB" or "256M" (256 megabytes)
            - "1G" or "1GB" (1 gigabyte)
            - "500MiB" (500 mebibytes, same as 500M)

    Returns:
        Size in bytes as integer.

    Raises:
        ValueError: If the size format is invalid.

    Examples:
        >>> parse_byte_size("256MB")
        268435456
        >>> parse_byte_size("1G")
        1073741824
        >>> parse_byte_size(1024)
        1024
    """
    if isinstance(size, (int, float)):
        return int(size)

    match = _SIZE_PATTERN.fullmatch(size.strip())
    if not match:
        raise ValueError(f"Invalid size format: {size!r}")

    num = float(match.group(1))
    unit_str = match.group(2) or ""
    # Extract first character for unit, handle 'i' suffix (MiB = M)
    unit = unit_str[0].upper() if unit_str else ""

    multiplier = _BYTE_SIZES.get(unit)
    if multiplier is None:
        raise ValueError(f"Unknown size unit: {unit_str!r}")

    return int(num * multiplier)


def format_byte_size(size: int) -> str:
    """Format bytes as human-readable string.

    Uses the largest unit that results in a value >= 1.

    Args:
        size: Size in bytes.

    Returns:
        Human-readable size string (e.g., "256MB", "1.5GB").

    Examples:
        >>> format_byte_size(268435456)
        '256MB'
        >>> format_byte_size(1073741824)
        '1GB'
        >>> format_byte_size(1500000000)
        '1.4GB'
    """
    if size < 0:
        return f"-{format_byte_size(-size)}"

    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if abs(size) < 1024:
            if size == int(size):
                return f"{int(size)}{unit}"
            return f"{size:.1f}{unit}"
        size = size / 1024

    return f"{size:.1f}EB"


def compute_num_shards(total_bytes: int, shard_size: Union[str, int]) -> int:
    """Compute number of shards from target shard size.

    Args:
        total_bytes: Total data size in bytes.
        shard_size: Target shard size (string like "256MB" or integer bytes).

    Returns:
        Number of shards (minimum 1).

    Examples:
        >>> compute_num_shards(1073741824, "256MB")  # 1GB / 256MB
        4
        >>> compute_num_shards(100, "256MB")  # Small data still gets 1 shard
        1
    """
    target_bytes = parse_byte_size(shard_size)
    if target_bytes <= 0:
        raise ValueError(f"shard_size must be positive, got {shard_size}")
    return max(1, math.ceil(total_bytes / target_bytes))
