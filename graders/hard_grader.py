"""
Grader for Task 3 (Hard): Full Multi-Step Resolution
"""
from tasks.hard import SENTIMENT_MULTIPLIERS


STEP_REWARDS = {
    "classify": 0.15,
    "request_info": 0.10,
    "investigate": 0.10,
    "verify": 0.10,
    "respond": 0.10,
    "escalate": 0.10,
    "resolve": 0.30,
}

RESOLUTION_REWARDS = {
    "process_refund": 0.5,
    "expedite_refund": 0.5,
    "freeze_account_and_refund": 0.6,
    "deny_refund_politely": 0.5,
    "guide_troubleshoot_or_replace": 0.5,
}


def grade_step(action: dict, step_name: str, task: dict) -> dict:
    reward = 0.0
    breakdown = {}
    feedback = []

    expected_action = _step_to_action(step_name)
    actual_action = action.get("action_type", "")

    if actual_action == expected_action:
        r = STEP_REWARDS.get(step_name, 0.1)
        breakdown[f"step_{step_name}"] = r
        reward += r
        feedback.append(f"✓ Correct step: {step_name}")
    elif actual_action in [_step_to_action(s) for s in STEP_REWARDS]:
        breakdown["wrong_step_but_valid"] = -0.05
        reward -= 0.05
        feedback.append(f"Wrong step order. Expected {step_name}, got {actual_action}.")
    else:
        breakdown["invalid_step"] = -0.1
        reward -= 0.1
        feedback.append(f"Invalid action: {actual_action}")

    if actual_action == "escalate" and task.get("ticket_type") not in ("fraud", "billing"):
        breakdown["unnecessary_escalation"] = -0.2
        reward -= 0.2
        feedback.append("Unnecessary escalation penalized.")

    return {"reward": round(reward, 3), "breakdown": breakdown, "feedback": feedback}


def grade_episode(trajectory: list, task: dict) -> dict:
    total_reward = sum(s["reward"] for s in trajectory)
    breakdown = {}
    feedback = []

    gathered_info = set()
    for step in trajectory:
        if step.get("action", {}).get("action_type") == "request_info":
            field = step.get("action", {}).get("info_field", "")
            if field:
                gathered_info.add(field)

    required_fields = set(task.get("required_info_fields", []))
    missing = required_fields - gathered_info
    if not missing:
        breakdown["all_info_gathered"] = 0.2
        total_reward += 0.2
        feedback.append("✓ All required information collected.")
    else:
        breakdown["missing_info_penalty"] = -0.15 * len(missing)
        total_reward -= 0.15 * len(missing)
        feedback.append(f"Missing info fields: {missing}")

    final_action = trajectory[-1].get("action", {}) if trajectory else {}
    resolution = final_action.get("resolution_type", "")
    expected_resolution = task.get("correct_resolution", "")

    if resolution == expected_resolution:
        r = RESOLUTION_REWARDS.get(resolution, 0.4)
        breakdown["correct_resolution"] = r
        total_reward += r
        feedback.append(f"✓ Correct resolution: {resolution}")
    elif resolution:
        breakdown["wrong_resolution"] = -0.2
        total_reward -= 0.2
        feedback.append(f"Wrong resolution. Got {resolution}, expected {expected_resolution}.")
    else:
        breakdown["no_resolution"] = -0.3
        total_reward -= 0.3
        feedback.append("Episode ended without resolution.")

    sentiment = task.get("sentiment", "neutral")
    multiplier = SENTIMENT_MULTIPLIERS.get(sentiment, 1.0)
    scaled = total_reward * multiplier
    breakdown["sentiment_multiplier"] = multiplier
    feedback.append(f"Sentiment multiplier applied: {multiplier}x")

    steps_taken = len(trajectory)
    optimal_steps = len(task.get("resolution_steps", []))
    if steps_taken <= optimal_steps:
        breakdown["efficiency_bonus"] = 0.1
        scaled += 0.1
        feedback.append(f"✓ Efficiency bonus: resolved in {steps_taken} steps.")
    elif steps_taken > optimal_steps + 2:
        penalty = -0.05 * (steps_taken - optimal_steps)
        breakdown["unnecessary_steps_penalty"] = penalty
        scaled += penalty
        feedback.append(f"Too many steps ({steps_taken} vs optimal {optimal_steps}).")

    return {
        "total_reward": round(scaled, 3),
        "raw_reward": round(total_reward, 3),
        "breakdown": breakdown,
        "feedback": feedback
    }


def _step_to_action(step: str) -> str:
    mapping = {
        "classify": "classify_ticket",
        "request_info": "request_info",
        "investigate": "investigate",
        "verify": "verify",
        "respond": "respond",
        "escalate": "escalate",
        "resolve": "resolve",
    }
    return mapping.get(step, step)