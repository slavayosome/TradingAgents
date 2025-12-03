from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError, root_validator, validator, model_validator


class OrchestratorHypothesis(BaseModel):
    ticker: str = Field("", description="Uppercase symbol")
    rationale: str = ""
    priority: float = 0.0
    required_analysts: List[str] = Field(default_factory=list)
    immediate_action: str = Field("monitor", description="monitor|escalate|trade|execute|buy|sell")
    suggested_size: Optional[float] = None
    tif: Optional[str] = Field(None, description="Time in force if proposing execution")

    @validator("ticker")
    def _upper(cls, v: str) -> str:  # noqa: N805
        return (v or "").upper()

    @validator("required_analysts", each_item=True)
    def _lower_list(cls, v: str) -> str:  # noqa: N805
        return (v or "").lower()

    @validator("immediate_action")
    def _lower_action(cls, v: str) -> str:  # noqa: N805
        return (v or "").lower()


class OrchestratorOutput(BaseModel):
    hypotheses: List[OrchestratorHypothesis] = Field(default_factory=list)
    summary: str = ""
    status: str = "ok"


class SizeHint(BaseModel):
    qty: Optional[float] = None
    notional: Optional[float] = None

    @model_validator(mode="after")
    def _at_least_one(self):  # noqa: N805
        if self.qty is None and self.notional is None:
            raise ValueError("Expected qty or notional")
        return self


class PlannerOutput(BaseModel):
    actions: List[str] = Field(default_factory=list)
    next_decision: str = Field("monitor", description="monitor|escalate|trade|execute")
    notes: str = ""
    reasoning: List[str] = Field(default_factory=list)
    time_in_force: Optional[str] = None
    size_hint: Optional[SizeHint] = None

    @validator("actions", each_item=True)
    def _lower_actions(cls, v: str) -> str:  # noqa: N805
        return (v or "").lower()

    @validator("next_decision")
    def _lower_next(cls, v: str) -> str:  # noqa: N805
        return (v or "").lower()

    @validator("time_in_force")
    def _upper_tif(cls, v: Optional[str]) -> Optional[str]:  # noqa: N805
        return v.upper() if v else v


def validate_orchestrator_payload(raw: object) -> OrchestratorOutput:
    if isinstance(raw, OrchestratorOutput):
        return raw
    if not isinstance(raw, dict):
        raise ValidationError([("payload", "not a dict", raw)], OrchestratorOutput)  # type: ignore[arg-type]
    return OrchestratorOutput(**raw)


def validate_planner_payload(raw: object) -> PlannerOutput:
    if isinstance(raw, PlannerOutput):
        return raw
    if not isinstance(raw, dict):
        raise ValidationError([("payload", "not a dict", raw)], PlannerOutput)  # type: ignore[arg-type]
    return PlannerOutput(**raw)
