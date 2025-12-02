from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Iterable

try:  # optional dependency
    from alpaca.data.live import StockDataStream
    from alpaca.data.enums import DataFeed
except ImportError:  # pragma: no cover - only when dependency missing
    StockDataStream = None  # type: ignore
    DataFeed = None  # type: ignore

from tradingagents.services.autopilot_worker import AutopilotWorker
from tradingagents.services.hypothesis_store import HypothesisStore, HypothesisRecord


@dataclass
class PriceTrigger:
    hypothesis_id: str
    symbol: str
    operator: str
    value: float


class RealtimeBroker:
    """Subscribe to Alpaca stock data stream and enqueue autopilot events."""

    def __init__(
        self,
        store: HypothesisStore,
        worker: AutopilotWorker,
        api_key: str,
        secret_key: str,
        *,
        feed: str = "iex",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.store = store
        self.worker = worker
        self.logger = logger or logging.getLogger(__name__)
        if StockDataStream is None or DataFeed is None:
            raise RuntimeError(
                "alpaca-py is required for realtime broker. Install with `pip install alpaca-py`."
            )
        data_feed = DataFeed.IEX if feed.lower() == "iex" else DataFeed.SIP
        self.stream = StockDataStream(api_key, secret_key, feed=data_feed)
        self.triggers: Dict[str, List[PriceTrigger]] = {}
        self.macro_triggers: Dict[str, List[Any]] = {}
        self._fired_count = 0
        self._subscribed: set[str] = set()
        self._registered_keys: set[str] = set()
        self._lock = threading.Lock()

    def bootstrap_triggers(self) -> None:
        self.refresh_triggers(reset=True)

    def refresh_triggers(
        self,
        records: Optional[List[HypothesisRecord]] = None,
        *,
        reset: bool = False,
    ) -> int:
        """Register triggers for stored hypotheses.

        When ``reset`` is True the in-memory trigger cache is rebuilt so updates to
        hypothesis triggers take effect immediately.
        Returns the number of triggers registered during this call (useful for logging).
        """

        dataset = records or self.store.list()
        with self._lock:
            if reset:
                self.triggers.clear()
                self.macro_triggers.clear()
                self._registered_keys.clear()

            registered = 0
            for record in dataset:
                for trigger in self._parse_triggers(record):
                    if self._register_trigger_locked(trigger):
                        registered += 1
                # Collect macro/custom triggers for surfacing
                macros = [raw for raw in record.triggers if self._is_macro_trigger(raw)]
                if macros:
                    self.macro_triggers[record.ticker.upper()] = macros
        return registered

    def _parse_triggers(self, record: HypothesisRecord) -> List[PriceTrigger]:
        triggers: List[PriceTrigger] = []
        for raw in record.triggers:
            parsed = self._parse_single_trigger(record, raw)
            if parsed:
                triggers.append(parsed)
        return triggers

    def _parse_single_trigger(self, record: HypothesisRecord, raw: Any) -> Optional[PriceTrigger]:
        """
        Parse a trigger that may be:
        - Structured dict with fields {type, condition:{operator, value, source}, action,...}
        - Simple text like "price >= 195"
        """
        # Structured trigger
        if isinstance(raw, dict):
            trig_type = (raw.get("type") or "").lower()
            trigger_price = raw.get("trigger_price") or raw.get("target_price") or raw.get("stop_price")
            operator = None
            if trigger_price is not None:
                try:
                    val = float(trigger_price)
                except (TypeError, ValueError):
                    val = None
                if val is not None:
                    if trig_type in {"price_take_profit", "price_breakout_watch"}:
                        operator = ">="
                    elif trig_type in {"price_stop_loss", "price_breakdown_watch"}:
                        operator = "<="
                    if operator:
                        return PriceTrigger(
                            hypothesis_id=raw.get("hypothesis_id") or record.id,
                            symbol=record.ticker.upper(),
                            operator=operator,
                            value=val,
                        )
            trig_type = (raw.get("type") or "").lower()
            condition = raw.get("condition") or {}
            parsed = self._parse_condition(condition)
            if not parsed:
                # maybe condition is a string
                parsed = self._parse_condition(condition.get("text") if isinstance(condition, dict) else condition)
            if not parsed:
                return None
            operator, value_f = parsed
            symbol = record.ticker.upper()
            return PriceTrigger(
                hypothesis_id=raw.get("hypothesis_id") or record.id,
                symbol=symbol,
                operator=operator,
                value=value_f,
            )

        # Text trigger
        text = str(raw).strip().lower()
        parsed = self._parse_condition(text)
        if parsed:
            operator, value = parsed
            return PriceTrigger(
                hypothesis_id=record.id,
                symbol=record.ticker.upper(),
                operator=operator,
                value=value,
            )
        return None

    def _parse_condition(self, condition: Any) -> Optional[tuple[str, float]]:
        """Parse condition dict or text into (operator, value) for price triggers."""
        if isinstance(condition, dict):
            source = str(condition.get("source") or "").lower()
            if source and source != "price":
                return None
            operator = str(condition.get("operator") or "").strip()
            value = condition.get("value")
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                return None
            if operator not in {">=", "<="}:
                return None
            return operator, value_f
        text = str(condition or "").strip().lower()
        if not text:
            return None
        for op in (">=", "<="):
            if op in text:
                try:
                    parts = text.split(op, 1)
                    right = parts[1].strip()
                    value_str = right.split()[0]
                    value_f = float(value_str)
                    return op, value_f
                except Exception:
                    continue
        if text.startswith("price >="):
            try:
                _, value = self._extract_symbol_value("", text, ">=")
                return ">=", value
            except Exception:
                return None
        if text.startswith("price <="):
            try:
                _, value = self._extract_symbol_value("", text, "<=")
                return "<=", value
            except Exception:
                return None
        return None

    def _is_macro_trigger(self, raw: Any) -> bool:
        if isinstance(raw, dict):
            trig_type = (raw.get("type") or "").lower()
            return trig_type in {"macro", "custom"} or bool(raw.get("description"))
        text = str(raw).strip().lower()
        return bool(text) and not text.startswith("price >=") and not text.startswith("price <=")

    def _extract_symbol_value(self, default_symbol: str, text: str, operator: str) -> (str, float):
        left, right = text.replace("price", "").split(operator, 1)
        symbol = left.strip().upper() or default_symbol.upper()
        value = float(right.strip())
        return symbol, value

    def _register_trigger_locked(self, trigger: PriceTrigger) -> bool:
        key = self._trigger_key(trigger)
        if key in self._registered_keys:
            return False

        self._registered_keys.add(key)
        symbol = trigger.symbol
        bucket = self.triggers.setdefault(symbol, [])
        bucket.append(trigger)
        if symbol not in self._subscribed:
            self.stream.subscribe_trades(self._trade_handler, symbol)
            self._subscribed.add(symbol)
        return True

    def register_manual_triggers(self, triggers: List[PriceTrigger]) -> int:
        """Register triggers supplied at runtime (e.g., from decisions/memory)."""
        with self._lock:
            registered = 0
            for trigger in triggers:
                if self._register_trigger_locked(trigger):
                    registered += 1
            return registered

    def stats(self) -> Dict[str, int]:
        with self._lock:
            symbols = len(self.triggers)
            total = sum(len(bucket) for bucket in self.triggers.values())
            macros = sum(len(v) for v in self.macro_triggers.values())
            fired = self._fired_count
        return {"symbols": symbols, "triggers": total, "macro_triggers": macros, "fired": fired}

    def _trigger_key(self, trigger: PriceTrigger) -> str:
        return f"{trigger.hypothesis_id}:{trigger.symbol}:{trigger.operator}:{trigger.value}"

    async def _trade_handler(self, data) -> None:  # pragma: no cover - network callback
        symbol = getattr(data, "symbol", "")
        price = getattr(data, "price", None)
        if not symbol or price is None:
            return
        with self._lock:
            triggers = list(self.triggers.get(symbol, []))
        for trigger in triggers:
            if self._evaluate(trigger, price):
                self.worker.enqueue_event(
                    trigger.hypothesis_id,
                    event_type="price_threshold",
                    payload={"symbol": symbol, "price": price, "operator": trigger.operator, "value": trigger.value},
                )
                self._fired_count += 1
                if self.logger:
                    self.logger.info("[Realtime Trigger] %s %s %s fired at %s", symbol, trigger.operator, trigger.value, price)

    def _evaluate(self, trigger: PriceTrigger, price: float) -> bool:
        if trigger.operator == ">=":
            return price >= trigger.value
        if trigger.operator == "<=":
            return price <= trigger.value
        return False

    def run_forever(self) -> None:
        self.bootstrap_triggers()
        self.logger.info("Starting Alpaca stock data stream …")
        try:
            self.stream.run()
        except KeyboardInterrupt:  # pragma: no cover - manual stop
            self.logger.info("Realtime broker stopped")
