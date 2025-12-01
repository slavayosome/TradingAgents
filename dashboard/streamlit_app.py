import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

RESULTS_DIR = Path(os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"))
HYP_DIR = RESULTS_DIR / "hypotheses"
AUTO_RUN_DIR = RESULTS_DIR
MCP_URL = os.getenv("TRADINGAGENTS_MCP_URL", "http://127.0.0.1:8000/mcp")


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
    try:
        out = subprocess.check_output(["systemctl", "is-active", name], text=True).strip()
        return out
    except Exception:
        return "unknown"


def mcp_health(url: str = MCP_URL) -> str:
    try:
        headers = {"Accept": "application/json, text/event-stream"}
        payload = {"jsonrpc": "2.0", "id": 1, "method": "list_tools", "params": {}}
        resp = requests.post(url, json=payload, headers=headers, timeout=5)
        if resp.status_code == 200:
            return "ok"
        return f"{resp.status_code}"
    except Exception as exc:
        return f"error: {exc}"


def render_status():
    st.title("TradingAgents Dashboard")
    cols = st.columns(4)
    cols[0].metric("TradingAgents", service_status("tradingagents"))
    cols[1].metric("Alpaca MCP", service_status("alpaca-mcp"))
    cols[2].metric("MCP Health", mcp_health())
    cols[3].metric("Results Dir", str(RESULTS_DIR))


def render_account(run: Dict[str, Any]):
    st.subheader("Latest Auto-Trade Snapshot")
    acct = run.get("account_snapshot", {})
    cols = st.columns(3)
    cols[0].metric("Cash", f"${acct.get('cash', 0):,}")
    cols[1].metric("Buying Power", f"${acct.get('buying_power', 0):,}")
    cols[2].metric("Portfolio", f"${acct.get('portfolio_value', 0):,}")


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
    latest = latest_auto_trade()
    if latest:
        render_account(latest)
        render_decisions(latest)
    else:
        st.info("No auto-trade runs found yet.")

    render_hypotheses(load_hypotheses())


if __name__ == "__main__":
    main()
