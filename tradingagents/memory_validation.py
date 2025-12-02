from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import jsonschema
except ImportError:  # pragma: no cover - defensive
    jsonschema = None


def _schema_path() -> Path:
    return Path(__file__).resolve().parent / "memory_schema.json"


def load_schema() -> Dict[str, Any]:
    path = _schema_path()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle) or {}


def validate_memory(entry: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate a memory entry against the unified schema. Returns (ok, error_msg)."""
    if jsonschema is None:
        return True, None  # validation unavailable but not fatal
    schema = load_schema()
    if not schema:
        return True, None
    try:
        jsonschema.validate(instance=entry, schema=schema)
        return True, None
    except Exception as exc:  # pragma: no cover - validation errors
        return False, str(exc)
