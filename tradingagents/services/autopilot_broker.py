from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Dict, Iterable, Optional, Any

from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.services.autopilot_worker import AutopilotWorker
from tradingagents.services.hypothesis_store import HypothesisRecord, HypothesisStore


@dataclass
class PriceThreshold:
    symbol: str
    operator: str
    value: float


class AutopilotBroker:
    """Simple polling broker that watches price thresholds and enqueues events."""

    def __init__(
        self,
        store: HypothesisStore,
        worker: AutopilotWorker,
        price_fetcher: Optional[Callable[[str], Optional[float]]] = None,
    ) -> None:
        self.store = store
        self.worker = worker
        self.price_fetcher = price_fetcher or default_price_fetcher
        self.poll_interval = 60  # seconds

    def parse_triggers(self, record: HypothesisRecord) -> Iterable[PriceThreshold]:
        for trigger in record.triggers:
            parsed = self._parse_trigger(record, trigger)
            if parsed:
                yield parsed

    def poll_once(self) -> Dict[str, str]:
        outcomes: Dict[str, str] = {}
        records = self.store.list()
        for record in records:
            for trigger in self.parse_triggers(record):
                latest_price = self.price_fetcher(trigger.symbol)
                if latest_price is None:
                    continue
                if self._evaluate(trigger, latest_price):
                    event = self.worker.enqueue_event(
                        record.id,
                        event_type="price_threshold",
                        payload={"symbol": trigger.symbol, "price": latest_price, "operator": trigger.operator, "value": trigger.value},
                    )
                    outcomes[event.id] = f"Triggered price alert for {trigger.symbol}"
        return outcomes

    def _evaluate(self, trigger: PriceThreshold, price: float) -> bool:
        if trigger.operator == ">=":
            return price >= trigger.value
        if trigger.operator == "<=":
            return price <= trigger.value
        return False

    def _parse_simple_trigger(self, default_symbol: str, trigger_str: str, operator: str) -> (str, float):
        parts = trigger_str.replace("price", "").strip().split(operator)
        if len(parts) != 2:
            raise ValueError("invalid trigger format")
        left = parts[0].strip().upper()
        symbol = left if left else default_symbol.upper()
        value = float(parts[1].strip())
        return symbol, value

    def _parse_trigger(self, record: HypothesisRecord, raw: Any) -> Optional[PriceThreshold]:
        # structured
        if isinstance(raw, dict):
            cond = raw.get("condition") or {}
            source = str(cond.get("source") or "").lower()
            if source and source != "price":
                return None
            operator = str(cond.get("operator") or "").strip()
            value = cond.get("value")
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                return None
            if operator not in {">=", "<="}:
                return None
            symbol = str(raw.get("symbol") or record.ticker).upper()
            return PriceThreshold(symbol=symbol, operator=operator, value=value_f)
        # text
        trigger_str = str(raw).strip().lower()
        if trigger_str.startswith("price >="):
            symbol, value = self._parse_simple_trigger(record.ticker, trigger_str, ">=")
            return PriceThreshold(symbol=symbol, operator=">=", value=value)
        if trigger_str.startswith("price <="):
            symbol, value = self._parse_simple_trigger(record.ticker, trigger_str, "<=")
            return PriceThreshold(symbol=symbol, operator="<=", value=value)
        return None


def default_price_fetcher(symbol: str) -> Optional[float]:
    symbol = symbol.upper()
    today = date.today()
    start = today - timedelta(days=2)
    try:
        csv_text = route_to_vendor("get_stock_data", symbol, start.isoformat(), today.isoformat())
    except Exception:
        return None
    lines = [line for line in str(csv_text).splitlines() if line and not line.startswith("#")]
    if not lines:
        return None
    last_line = lines[-1]
    parts = last_line.split(",")
    if len(parts) < 5:
        return None
    try:
        return float(parts[4])  # close price
    except ValueError:
        return None
