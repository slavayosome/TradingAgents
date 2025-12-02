from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tradingagents.memory_validation import validate_memory


class TickerMemoryStore:
    """Filesystem-backed short-term memory for per-ticker decisions."""

    def __init__(
        self,
        base_dir: str,
        *,
        max_entries: int = 5,
        enabled: bool = True,
        schema_version: str = "v1",
        validation_mode: str = "warn",
    ) -> None:
        self.enabled = enabled
        self.base_path = Path(base_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.max_entries = max(1, int(max_entries))
        self.schema_version = str(schema_version or "v1").lower()
        self.validation_mode = str(validation_mode or "warn").lower()

    def is_enabled(self) -> bool:
        return self.enabled

    def _path(self, ticker: str) -> Path:
        return self.base_path / f"{ticker.upper()}.json"

    def load(self, ticker: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        path = self._path(ticker)
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        limit_value = limit if limit is not None else self.max_entries
        if limit_value <= 0:
            return data[-self.max_entries :]
        return data[-limit_value:]

    def load_structured(self, ticker: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load entries and filter by schema validity (no conversion)."""
        entries = self.load(ticker, limit)
        if not entries:
            return []
        valid: List[Dict[str, Any]] = []
        for entry in entries:
            ok, _ = validate_memory(entry)
            if ok:
                valid.append(entry)
        return valid

    def append(self, ticker: str, entry: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._path(ticker)
        try:
            history = []
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    history = json.load(handle) or []
            if not isinstance(history, list):
                history = []
        except Exception:
            history = []
        history.append(entry)
        history = history[-self.max_entries :]
        with path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2, default=str)

    def append_structured(self, ticker: str, entry: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Append entry after schema validation. Returns (ok, error)."""
        ok, error = validate_memory(entry)
        if not ok:
            if self.validation_mode == "strict":
                return False, error
        self.append(ticker, entry)
        return True, None

    def record_decisions(self, decisions: List[Dict[str, Any]]) -> None:
        if not self.enabled:
            return
        timestamp = datetime.utcnow().isoformat()
        for decision in decisions:
            ticker = str(decision.get("ticker") or "").upper()
            if not ticker:
                continue
            payload = {
                "timestamp": timestamp,
                "action": decision.get("action") or decision.get("final_decision") or "",
                "priority": decision.get("priority"),
                "notes": decision.get("notes") or decision.get("final_notes") or "",
                "plan_actions": decision.get("plan_actions") or decision.get("sequential_plan", {}).get("actions"),
                "raw": decision,
            }
            self.append(ticker, payload)
