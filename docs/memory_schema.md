## Unified Memory Schema (Draft-07)

Single-source schema for ticker memory entries. Goal: eliminate duplication, separate state from procedure, and keep machine-parsable triggers.

Top-level shape (per entry):
```json
{
  "timestamp": "ISO8601",
  "ticker": "TSLA",
  "position": {},
  "market_snapshot": {},
  "strategy": {},
  "thesis": {},
  "triggers": [],
  "current_decision": {},
  "next_plan": {}
}
```

Key principles:
- One copy of each concept (no duplicate triggers/notes/targets/stops).
- Memory = state + configuration; procedures live in `next_plan`.
- Triggers are first-class structured objects.
- Derived levels are stored once or derived on read; strategy holds only configuration.

See `tradingagents/memory_schema.json` for authoritative validation rules.
