"""Canonical serialization for the deterministic local GTO oracle."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from decimal import Decimal
from enum import Enum
import hashlib
import json
from typing import Any

from .models import OracleValidationError


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        raise OracleValidationError("canonical JSON cannot contain non-finite decimals")
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def _canonical_data(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _canonical_data(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return _decimal_text(value)
    if isinstance(value, tuple) or isinstance(value, list):
        return [_canonical_data(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise OracleValidationError("canonical JSON dictionary keys must be strings")
        return {key: _canonical_data(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise OracleValidationError(
        f"canonical JSON does not support {type(value).__name__}"
    )


def canonical_json(value: Any) -> str:
    """Serialize a supported immutable value to deterministic sorted JSON."""

    return json.dumps(
        _canonical_data(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_key(value: Any) -> str:
    """Hash the canonical JSON representation with SHA-256."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
