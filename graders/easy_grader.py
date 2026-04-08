"""
Grader for Task 1 (Easy): Ticket Classification
"""
from tasks.easy import VALID_LABELS


def grade(action: dict, task: dict) -> dict:
    reward = 0.0
    breakdown = {}
    feedback = []

    if action.get("action_type") != "classify_ticket":
        return {
            "reward": -0.2,
            "breakdown": {"wrong_action_type": -0.2},
            "feedback": ["Expected action_type=classify_ticket"]
        }

    predicted = action.get("label", "").strip().lower()
    expected = task["expected_label"]

    if predicted not in VALID_LABELS:
        breakdown["invalid_label"] = -0.1
        reward -= 0.1
        feedback.append(f"Label '{predicted}' is not in VALID_LABELS.")
    elif predicted == expected:
        breakdown["correct_classification"] = +1.0
        reward += 1.0
        feedback.append(f"Correct! '{predicted}' matches expected '{expected}'.")
    else:
        domain_map = {
            "billing": ["billing", "refund"],
            "refund": ["billing", "refund"],
            "account_access": ["account_access", "fraud"],
            "fraud": ["account_access", "fraud"],
        }
        related = domain_map.get(expected, [expected])
        if predicted in related:
            breakdown["partial_classification"] = +0.3
            reward += 0.3
            feedback.append(f"Partially correct. Got '{predicted}', expected '{expected}'.")
        else:
            breakdown["wrong_classification"] = 0.0
            feedback.append(f"Wrong. Got '{predicted}', expected '{expected}'.")

    return {"reward": round(reward, 3), "breakdown": breakdown, "feedback": feedback}