"""
models.py — Pydantic typed models for strict OpenEnv API compliance.
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal, List, Dict, Any
from enum import Enum


class TaskLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ActionType(str, Enum):
    CLASSIFY = "classify_ticket"
    RESPOND = "respond"
    REQUEST_INFO = "request_info"
    ESCALATE = "escalate"
    INVESTIGATE = "investigate"
    VERIFY = "verify"
    RESOLVE = "resolve"


class TicketLabel(str, Enum):
    DELIVERY = "delivery_issue"
    BILLING = "billing"
    DEFECT = "product_defect"
    ACCOUNT = "account_access"
    CANCEL = "cancellation"
    REFUND = "refund"
    TECH = "technical_issue"
    FRAUD = "fraud"
    OTHER = "other"


class ResolutionType(str, Enum):
    PROCESS_REFUND = "process_refund"
    EXPEDITE_REFUND = "expedite_refund"
    FREEZE_AND_REFUND = "freeze_account_and_refund"
    DENY_REFUND = "deny_refund_politely"
    TROUBLESHOOT = "guide_troubleshoot_or_replace"


class Sentiment(str, Enum):
    ANGRY = "angry"
    FRUSTRATED = "frustrated"
    CONFUSED = "confused"
    DISAPPOINTED = "disappointed"
    PANICKED = "panicked"
    NEUTRAL = "neutral"


# ── Actions ────────────────────────────────────────────────────────────

class ClassifyAction(BaseModel):
    action_type: Literal["classify_ticket"] = "classify_ticket"
    label: TicketLabel


class RespondAction(BaseModel):
    action_type: Literal["respond"] = "respond"
    message: str = Field(..., min_length=5, max_length=1000)


class RequestInfoAction(BaseModel):
    action_type: Literal["request_info"] = "request_info"
    info_field: str
    message: str


class EscalateAction(BaseModel):
    action_type: Literal["escalate"] = "escalate"
    reason: str


class InvestigateAction(BaseModel):
    action_type: Literal["investigate"] = "investigate"
    target: str


class VerifyAction(BaseModel):
    action_type: Literal["verify"] = "verify"
    field: str
    value: str


class ResolveAction(BaseModel):
    action_type: Literal["resolve"] = "resolve"
    resolution_type: ResolutionType
    message: str = Field(..., min_length=5)


# ── Observation ────────────────────────────────────────────────────────

class Observation(BaseModel):
    customer_message: str
    sentiment: Sentiment
    ticket_type: str = ""
    history: List[Dict[str, Any]] = Field(default_factory=list)
    step_count: int = 0
    task_level: TaskLevel = TaskLevel.EASY
    info_gathered: Dict[str, Any] = Field(default_factory=dict)
    ticket_id: str = ""
    done: bool = False
    sla_seconds_remaining: Optional[float] = None
    customer_satisfaction: Optional[float] = None
    cost_incurred: float = 0.0


# ── Step Result ────────────────────────────────────────────────────────

class StepResult(BaseModel):
    reward: float
    breakdown: Dict[str, float] = Field(default_factory=dict)
    feedback: List[str] = Field(default_factory=list)
    sla_penalty: float = 0.0
    satisfaction_delta: float = 0.0
    cost: float = 0.0
    episode_summary: Optional[Dict[str, Any]] = None


# ── Episode Summary ────────────────────────────────────────────────────

class EpisodeSummary(BaseModel):
    ticket_id: str
    task_level: TaskLevel
    total_reward: float
    steps: int
    success: bool
    customer_satisfaction: float
    total_cost: float
    sla_met: bool
    resolution_type: Optional[str] = None