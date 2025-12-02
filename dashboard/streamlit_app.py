import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# Ensure project root on path for local dashboard runs
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tradingagents.integrations.alpaca_mcp import AlpacaMCPClient, AlpacaMCPConfig, AlpacaMCPError

# Ensure project root on path for local dashboard runs
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = Path(os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"))
HYP_DIR = RESULTS_DIR / "hypotheses"
AUTO_RUN_DIR = RESULTS_DIR
MCP_URL = os.getenv("TRADINGAGENTS_MCP_URL", "http://127.0.0.1:8000/mcp")
CHECK_SERVICES = os.getenv("DASHBOARD_SERVICE_CHECK", "true").lower() not in ("0", "false", "no")
SSH_CMD = os.getenv("DASHBOARD_SERVICE_SSH", "").strip()
MODEL_BADGE = os.getenv("AUTO_TRADE_RESPONSES_MODEL") or os.getenv("TRADINGAGENTS_RESPONSES_MODEL") or ""


def _color_from_status(status: str) -> str:
    if status.startswith("active") or status == "ok":
        return "green"
    if status in {"activating", "reloading", "skipped"}:
        return "yellow"
    return "red"


def _color_from_mcp(status: str) -> str:
    if status == "ok":
        return "green"
    if status.startswith("error"):
        return "red"
    return "yellow"


def status_badge(text: str, color: str) -> str:
    dot = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(color, "⚪")
    return f"{dot} {text}"


def status_badge(text: str, color: str) -> str:
    dot = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(color, "⚪")
    return f"{dot} {text}"


def read_json_files(path: Path, prefix: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not path.exists():
        return items
    for file in sorted(path.glob(f"{prefix}*.json")):
        try:
            with file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            data["_file"] = str(file)
            items.append(data)
        except Exception:
            continue
    return items


def load_hypotheses() -> List[Dict[str, Any]]:
    records = []
    if not HYP_DIR.exists():
        return records
    for file in sorted(HYP_DIR.glob("*.json")):
        try:
            with file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            data["_file"] = str(file)
            records.append(data)
        except Exception:
            continue
    return sorted(records, key=lambda x: x.get("created_at", ""), reverse=True)


def latest_auto_trade() -> Optional[Dict[str, Any]]:
    runs = read_json_files(AUTO_RUN_DIR, "auto_trade_")
    if not runs:
        return None
    return runs[-1]


def service_status(name: str) -> str:
    if not CHECK_SERVICES:
        return "skipped"
    cmd = ["systemctl", "is-active", name]
    if SSH_CMD:
        cmd = shlex.split(SSH_CMD) + cmd
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        return out
    except FileNotFoundError:
        return "n/a (no systemd?)"
    except Exception:
        return "unknown"


def mcp_health(url: str = MCP_URL) -> str:
    try:
        headers = {"Accept": "application/json, text/event-stream"}
        payload = {"jsonrpc": "2.0", "id": 1, "method": "list_tools", "params": {}}
        resp = requests.post(url, json=payload, headers=headers, timeout=5)
        if resp.status_code in (200, 400, 406):
            return "ok"
        return f"{resp.status_code}"
    except Exception as exc:
        return f"error: {exc}"


def mcp_clock(url: str = MCP_URL) -> Dict[str, Any]:
    try:
        headers = {"Accept": "application/json, text/event-stream"}
        payload = {"jsonrpc": "2.0", "id": 1, "method": "call_tool", "params": {"name": "get_market_clock", "arguments": {}}}
        resp = requests.post(url, json=payload, headers=headers, timeout=5)
        if resp.status_code not in (200, 400, 406):
            return {"status": "red", "message": f"http {resp.status_code}"}
        data = resp.json()
        result = data.get("result") or {}
        content = result.get("content")
        parsed = _parse_clock_text(content)
        return parsed
    except Exception as exc:
        return {"status": "red", "message": str(exc)}


def mcp_account(url: str = MCP_URL) -> Dict[str, Any]:
    try:
        headers = {"Accept": "application/json, text/event-stream"}
        payload = {"jsonrpc": "2.0", "id": 1, "method": "call_tool", "params": {"name": "get_account_info", "arguments": {}}}
        resp = requests.post(url, json=payload, headers=headers, timeout=5)
        if resp.status_code not in (200, 400, 406):
            return {}
        data = resp.json()
        result = data.get("result") or {}
        content = result.get("content")
        return _parse_account_text(content)
    except Exception:
        return {}


def mcp_account_live(url: str = MCP_URL) -> Dict[str, Any]:
    """Use the MCP client (session-based) to fetch live account info."""
    try:
        cfg = AlpacaMCPConfig.from_dict(
            {
                "enabled": True,
                "transport": "streamable-http",
                "base_url": url,
                "required_tools": [],
            }
        )
        client = AlpacaMCPClient(cfg)
        text = client.fetch_account_info()
        return _parse_account_text(text)
    except AlpacaMCPError:
        return {}
    except Exception:
        return {}


def _parse_account_text(content: Any) -> Dict[str, float]:
    vals: Dict[str, float] = {}
    if isinstance(content, dict):
        vals = {k.lower(): v for k, v in content.items()}
    elif isinstance(content, str):
        for line in content.splitlines():
            if ":" not in line:
                continue
            label, val = line.split(":", 1)
            key = label.strip().lower().replace(" ", "_")
            try:
                num = float(str(val).replace("$", "").replace(",", "").strip())
            except ValueError:
                continue
            vals[key] = num
    return {
        "cash": float(vals.get("cash", 0.0) or 0.0),
        "buying_power": float(vals.get("buying_power", vals.get("buying_power_usd", 0.0) or 0.0)),
        "portfolio_value": float(vals.get("portfolio_value", vals.get("equity", 0.0) or 0.0)),
    }


def _parse_clock_text(text: str) -> Dict[str, Any]:
    if not text:
        return {"status": "red", "message": "no clock"}
    lines = [line.strip() for line in str(text).splitlines() if ":" in line]
    fields: Dict[str, str] = {}
    for line in lines:
        label, val = line.split(":", 1)
        fields[label.strip().lower()] = val.strip()
    is_open = "yes" in (fields.get("is open", "").lower())
    current_time = fields.get("current time")
    next_open = fields.get("next open")
    next_close = fields.get("next close")
    status = "green" if is_open else "yellow"
    eta = None
    if not is_open and next_open:
        try:
            from datetime import datetime, timezone

            nxt = next_open.replace(" ", "T", 1) if "T" not in next_open and " " in next_open else next_open
            dt = datetime.fromisoformat(nxt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            eta = max(int((dt - now).total_seconds() // 60), 0)
        except Exception:
            eta = None
    return {
        "status": status,
        "is_open": is_open,
        "current_time": current_time,
        "next_open": next_open,
        "next_close": next_close,
        "eta_open_minutes": eta,
    }


def render_status():
    st.title("TradingAgents Dashboard")
    cols = st.columns(3)
    svc = service_status("tradingagents")
    mcp = mcp_health()
    cols[0].metric("TradingAgents", status_badge(svc, _color_from_status(svc)))
    cols[1].metric("Alpaca MCP", status_badge(mcp, _color_from_mcp(mcp)))
    market = mcp_clock()
    market_text = "open" if market.get("is_open") else "closed"
    if market.get("eta_open_minutes") is not None:
        market_text += f" (opens in {market['eta_open_minutes']}m)"
    cols[2].metric("Market", status_badge(market_text, market.get("status", "red")))
    if MODEL_BADGE:
        st.caption(f"Model: {MODEL_BADGE}")


def render_account(latest_run: Optional[Dict[str, Any]], live: Dict[str, Any]):
    st.subheader("Account Snapshot")
    cash = bp = pv = 0.0
    source = "snapshot"
    fetched = None
    if latest_run:
        fetched = latest_run.get("fetched_at")
        acct = latest_run.get("account_snapshot", {}) or {}
        cash = acct.get("cash", latest_run.get("cash", 0))
        bp = acct.get("buying_power", latest_run.get("buying_power", 0))
        pv = acct.get("portfolio_value", latest_run.get("portfolio_value", 0))
    if live:
        source = "live MCP"
        cash = live.get("cash", cash)
        bp = live.get("buying_power", bp)
        pv = live.get("portfolio_value", pv)
    cols = st.columns(3)
    cols[0].metric("Cash", f"${cash:,}")
    cols[1].metric("Buying Power", f"${bp:,}")
    cols[2].metric("Portfolio", f"${pv:,}")
    if source == "live MCP":
        st.caption("Source: live MCP")
    elif fetched:
        st.caption(f"Source: latest run ({fetched})")


def render_decisions(run: Dict[str, Any]):
    st.subheader("Decisions")
    decisions = run.get("decisions", [])
    if not decisions:
        st.info("No decisions yet.")
        return
    rows = []
    for d in decisions:
        rows.append(
            {
                "Ticker": d.get("ticker"),
                "Action": (d.get("final_decision") or d.get("immediate_action") or "hold").upper(),
                "Priority": d.get("priority"),
                "Strategy": (d.get("strategy") or {}).get("name"),
                "Target %": (d.get("strategy") or {}).get("target_pct"),
                "Stop %": (d.get("strategy") or {}).get("stop_pct"),
            }
        )
    st.dataframe(rows, use_container_width=True)

    st.subheader("Decision Details")
    for d in decisions:
        ticker = d.get("ticker") or "?"
        action = (d.get("final_decision") or d.get("immediate_action") or "hold").upper()
        with st.expander(f"{ticker} – {action}"):
            st.write("Strategy:", d.get("strategy"))
            st.write("Priority:", d.get("priority"))
            st.write("Trader Plan:", d.get("trader_plan"))
            st.write("Final Notes:", d.get("final_notes") or "<none>")
            sp = d.get("sequential_plan") or {}
            st.write("Plan Actions:", sp.get("actions"))
            st.write("Plan Next:", sp.get("next_decision"))
            st.write("Reasoning:", sp.get("reasoning"))
            st.write("Action Queue:", d.get("action_queue"))


def render_hypotheses(records: List[Dict[str, Any]]):
    st.subheader("Hypotheses")
    if not records:
        st.info("No hypotheses yet.")
        return
    rows = []
    for rec in records:
        rows.append(
            {
                "ID": rec.get("id", "")[-6:],
                "Ticker": rec.get("ticker"),
                "Status": rec.get("status"),
                "Action": rec.get("action"),
                "Priority": rec.get("priority"),
                "Next Step": (rec.get("steps") or [{}])[-1].get("description", "") if rec.get("steps") else "",
                "Created": rec.get("created_at", "").split("T")[0],
            }
        )
    st.dataframe(rows, use_container_width=True)


def main():
    render_status()
    refresh_live = st.button("Refresh Live MCP Account")
    latest = latest_auto_trade()
    live_acc = mcp_account_live() if refresh_live else mcp_account_live()
    render_account(latest, live_acc)
    if latest:
        render_decisions(latest)
    else:
        st.info("No auto-trade runs found yet.")
    render_hypotheses(load_hypotheses())


if __name__ == "__main__":
    main()


# Helpers for status coloring
def _color_from_status(status: str) -> str:
    if status.startswith("active") or status == "ok":
        return "green"
    if status in {"activating", "reloading", "skipped"}:
        return "yellow"
    return "red"


def _color_from_mcp(status: str) -> str:
    if status == "ok":
        return "green"
    if status.startswith("error"):
        return "red"
    return "yellow"
