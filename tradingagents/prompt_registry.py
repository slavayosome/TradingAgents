from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _project_root() -> Path:
    # tradingagents/ -> project root is parent of this directory
    return Path(__file__).resolve().parent.parent


def load_prompt_config(name: str) -> Dict[str, Any]:
    """Load a prompt config (prompt + default request params) from prompts/<name>.json."""
    path = _project_root() / "prompts" / f"{name}.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle) or {}
    except Exception:
        return {}


def prompt_text(name: str, default: str = "") -> str:
    """Convenience to fetch just the system prompt text with a default fallback."""
    cfg = load_prompt_config(name)
    text = cfg.get("system_prompt")
    return str(text) if text else default


def prompt_defaults(name: str) -> Dict[str, Any]:
    """Return default request parameters (model/temperature/reasoning/etc.) for a prompt."""
    cfg = load_prompt_config(name)
    defaults = cfg.get("request_defaults")
    return defaults if isinstance(defaults, dict) else {}


def required_prompt_files() -> List[str]:
    """List required prompt config basenames (without extension). Extend as new prompts are centralized."""
    return [
        "responses_auto_trade",
        "market_analyst",
        "reflection",
    ]


def check_prompts() -> List[Tuple[str, bool]]:
    """Return list of (name, exists) for required prompts."""
    base = _project_root() / "prompts"
    results: List[Tuple[str, bool]] = []
    for name in required_prompt_files():
        exists = (base / f"{name}.json").exists()
        results.append((name, exists))
    return results
