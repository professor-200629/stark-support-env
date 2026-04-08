"""
business.py — Business realism layer.
SLA tracking, customer satisfaction score, and cost simulation.
"""
import time
from typing import Optional


SLA_BUDGETS = {
    "easy": 60.0,
    "medium": 120.0,
    "hard": 300.0,
}

SENTIMENT_SLA_MULTIPLIER = {
    "angry": 0.7,
    "panicked": 0.6,
    "frustrated": 0.8,
    "confused": 0.9,
    "disappointed": 0.9,
    "neutral": 1.0,
}

ACTION_COSTS = {
    "escalate": 15.0,
    "investigate": 5.0,
    "verify": 2.0,
    "request_info": 1.0,
    "respond": 0.5,
    "classify_ticket": 0.0,
    "resolve": 0.0,
}

RESOLUTION_COSTS = {
    "process_refund": 49.99,
    "expedite_refund": 89.99,
    "freeze_account_and_refund": 249.97,
    "deny_refund_politely": 0.0,
    "guide_troubleshoot_or_replace": 25.0,
}


class CustomerSatisfaction:
    """Tracks CSAT score (0–10) throughout episode."""

    SENTIMENT_BASE = {
        "angry": 2.0,
        "panicked": 2.5,
        "frustrated": 3.0,
        "confused": 4.0,
        "disappointed": 3.5,
        "neutral": 6.0,
    }

    def __init__(self, sentiment: str):
        self.score = self.SENTIMENT_BASE.get(sentiment, 5.0)
        self.sentiment = sentiment

    def update(self, action_type: str, reward: float, tone_good: bool = False) -> float:
        delta = 0.0
        if action_type == "respond":
            delta = +1.5 if tone_good else (-2.0 if reward < 0 else +0.5)
        elif action_type == "request_info":
            delta = +0.3
        elif action_type == "escalate":
            delta = +1.0 if self.sentiment in ("angry", "panicked") else -0.5
        elif action_type == "resolve":
            delta = +3.0 if reward > 0.5 else (+1.5 if reward > 0 else -2.0)
        self.score = max(0.0, min(10.0, self.score + delta))
        return delta

    @property
    def label(self) -> str:
        if self.score >= 9: return "Delighted"
        elif self.score >= 7: return "Satisfied"
        elif self.score >= 5: return "Neutral"
        elif self.score >= 3: return "Dissatisfied"
        return "Very Unhappy"


class SLATracker:
    """Time-based SLA compliance tracker."""

    def __init__(self, task_level: str, sentiment: str):
        budget = SLA_BUDGETS.get(task_level, 120.0)
        multiplier = SENTIMENT_SLA_MULTIPLIER.get(sentiment, 1.0)
        self.budget_seconds = budget * multiplier
        self.start_time = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def remaining(self) -> float:
        return max(0.0, self.budget_seconds - self.elapsed)

    @property
    def met(self) -> bool:
        return self.elapsed <= self.budget_seconds

    def penalty(self) -> float:
        if self.met:
            return 0.0
        ratio = (self.elapsed - self.budget_seconds) / self.budget_seconds
        return round(-min(0.5, ratio * 0.5), 3)

    def bonus(self) -> float:
        ratio = self.remaining / self.budget_seconds
        if ratio > 0.7: return +0.15
        elif ratio > 0.4: return +0.05
        return 0.0


class CostTracker:
    """Operational cost tracker."""

    def __init__(self):
        self.total = 0.0
        self.log = []

    def record(self, action_type: str, resolution_type: str = None) -> float:
        cost = ACTION_COSTS.get(action_type, 0.0)
        if resolution_type:
            cost += RESOLUTION_COSTS.get(resolution_type, 0.0)
        self.total += cost
        self.log.append({"action": action_type, "cost": cost})
        return cost

    def reward_adjustment(self) -> float:
        if self.total == 0: return 0.0
        elif self.total < 10: return +0.1
        elif self.total < 50: return 0.0
        elif self.total < 150: return -0.1
        return -0.2