"""
Stark Support Trainer — RL Environment v2
With SLA, CSAT, and cost simulation built in.
"""
import random
from typing import Optional

from tasks.easy import CLASSIFICATION_TASKS
from tasks.medium import RESPONSE_TASKS
from tasks.hard import HARD_SCENARIOS
from graders.easy_grader import grade as easy_grade
from graders.medium_grader import grade as medium_grade
from graders.hard_grader import grade_step, grade_episode
from business import CustomerSatisfaction, SLATracker, CostTracker


class SupportEnv:
    TASK_LEVELS = ["easy", "medium", "hard"]
    MAX_STEPS = {"easy": 1, "medium": 1, "hard": 8}

    def __init__(self, task_level: str = "easy", seed: Optional[int] = None):
        assert task_level in self.TASK_LEVELS
        self.task_level = task_level
        self.rng = random.Random(seed)
        self._reset_state()

    def _reset_state(self):
        self._task = None
        self._step_count = 0
        self._history = []
        self._trajectory = []
        self._info_gathered = {}
        self._classified = False
        self._done = False
        self._sla = None
        self._csat = None
        self._cost = None

    def reset(self) -> dict:
        self._reset_state()
        if self.task_level == "easy":
            self._task = self.rng.choice(CLASSIFICATION_TASKS)
        elif self.task_level == "medium":
            self._task = self.rng.choice(RESPONSE_TASKS)
        else:
            self._task = self.rng.choice(HARD_SCENARIOS)

        sentiment = self._task.get("sentiment", "neutral")
        if self.task_level == "hard":
            self._sla = SLATracker(self.task_level, sentiment)
            self._csat = CustomerSatisfaction(sentiment)
            self._cost = CostTracker()

        return self._get_obs()

    def step(self, action: dict) -> tuple:
        assert not self._done, "Episode done. Call reset()."
        assert isinstance(action, dict) and "action_type" in action

        self._step_count += 1
        reward = 0.0
        info = {}

        if self.task_level == "easy":
            result = easy_grade(action, self._task)
            reward, info = result["reward"], result
            self._done = True
        elif self.task_level == "medium":
            result = medium_grade(action, self._task)
            reward, info = result["reward"], result
            self._done = True
        else:
            reward, info = self._hard_step(action)

        if self._step_count >= self.MAX_STEPS[self.task_level] and not self._done:
            info.setdefault("feedback", []).append("Timed out.")
            info["timeout"] = True
            reward -= 0.3
            self._done = True

        self._history.append({
            "step": self._step_count,
            "action": action,
            "reward": round(reward, 3),
            "info": {k: v for k, v in info.items() if k != "episode_summary"}
        })

        return self._get_obs(), reward, self._done, info

    def render(self):
        print("=" * 60)
        print(f"STARK SUPPORT | {self.task_level.upper()} | Step {self._step_count}")
        msg = self._task.get("customer_message", self._task.get("initial_message", ""))
        print(f"Message  : {msg}")
        print(f"Sentiment: {self._task.get('sentiment', '?')}")
        if self._csat:
            print(f"CSAT     : {self._csat.score:.1f}/10 ({self._csat.label})")
        if self._sla:
            print(f"SLA      : {self._sla.remaining:.1f}s remaining | met={self._sla.met}")
        if self._cost:
            print(f"Cost     : ${self._cost.total:.2f}")
        for h in self._history[-3:]:
            print(f"  [{h['step']}] {h['action']['action_type']} → {h['reward']:.3f}")
        print("=" * 60)

    def _hard_step(self, action: dict) -> tuple:
        steps = self._task["resolution_steps"]
        current = steps[min(self._step_count - 1, len(steps) - 1)]

        if action["action_type"] == "classify_ticket":
            self._classified = True
        if action["action_type"] == "request_info":
            field = action.get("info_field", "")
            if field and field in self._task.get("hidden_info", {}):
                self._info_gathered[field] = self._task["hidden_info"][field]

        result = grade_step(action, current, self._task)
        reward = result["reward"]

        cost = self._cost.record(
            action["action_type"],
            action.get("resolution_type") if action["action_type"] == "resolve" else None
        )
        tone_good = reward > 0 and action["action_type"] == "respond"
        csat_delta = self._csat.update(action["action_type"], reward, tone_good)

        result.update({
            "csat": round(self._csat.score, 2),
            "csat_delta": round(csat_delta, 2),
            "cost": cost,
            "sla_remaining": round(self._sla.remaining, 1),
        })

        self._trajectory.append({"action": action, "reward": reward, "step": current})

        if self._step_count >= len(steps) or action["action_type"] == "resolve":
            ep = grade_episode(self._trajectory, self._task)

            sla_adj = self._sla.penalty() + self._sla.bonus()
            ep["total_reward"] = round(ep["total_reward"] + sla_adj, 3)
            ep["breakdown"]["sla_adjustment"] = round(sla_adj, 3)
            ep["feedback"].append(
                f"SLA {'✓ met' if self._sla.met else '✗ breached'} "
                f"({self._sla.elapsed:.1f}s / {self._sla.budget_seconds:.0f}s)"
            )

            cost_adj = self._cost.reward_adjustment()
            ep["total_reward"] = round(ep["total_reward"] + cost_adj, 3)
            ep["feedback"].append(f"Total cost: ${self._cost.total:.2f}")

            if self._csat.score >= 8:
                ep["total_reward"] = round(ep["total_reward"] + 0.2, 3)
                ep["breakdown"]["csat_bonus"] = 0.2
                ep["feedback"].append(f"CSAT bonus: {self._csat.score:.1f}/10 ({self._csat.label})")
            elif self._csat.score < 4:
                ep["total_reward"] = round(ep["total_reward"] - 0.2, 3)
                ep["feedback"].append(f"CSAT penalty: {self._csat.score:.1f}/10")

            ep.update({
                "csat_final": round(self._csat.score, 2),
                "csat_label": self._csat.label,
                "total_cost": round(self._cost.total, 2),
                "sla_met": self._sla.met,
            })

            reward += ep["total_reward"]
            result["episode_summary"] = ep
            self._done = True

        return reward, result

    def _get_obs(self) -> dict:
        task = self._task or {}
        obs = {
            "customer_message": task.get("customer_message", task.get("initial_message", "")),
            "sentiment": task.get("sentiment", "neutral"),
            "ticket_type": task.get("ticket_type", "") if self._classified else "",
            "history": list(self._history),
            "step_count": self._step_count,
            "task_level": self.task_level,
            "info_gathered": dict(self._info_gathered),
            "ticket_id": task.get("ticket_id", ""),
            "done": self._done,
        }
        if self._sla:
            obs["sla_seconds_remaining"] = round(self._sla.remaining, 1)
        if self._csat:
            obs["customer_satisfaction"] = round(self._csat.score, 2)
        if self._cost:
            obs["cost_incurred"] = round(self._cost.total, 2)
        return obs