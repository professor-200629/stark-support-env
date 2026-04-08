"""
Grader for Task 2 (Medium): Response Quality
"""

POLITE_MARKERS = [
    "sorry", "apologize", "apologies", "understand", "help you",
    "assist", "please", "thank you", "appreciate", "concern"
]

RUDE_MARKERS = [
    "not our problem", "your fault", "impossible", "can't help",
    "not sure what you want", "just wait", "obviously"
]


def _tone_score(message: str) -> tuple:
    msg = message.lower()
    polite_hits = sum(1 for m in POLITE_MARKERS if m in msg)
    rude_hits = sum(1 for m in RUDE_MARKERS if m in msg)

    if rude_hits > 0:
        return -0.5, f"Rude language detected ({rude_hits} markers)."
    if polite_hits >= 2:
        return 0.3, f"Polite tone ({polite_hits} markers)."
    if polite_hits == 1:
        return 0.15, "Somewhat polite tone."
    return 0.0, "Neutral tone — add empathy."


def _relevance_score(message: str, task: dict) -> tuple:
    msg = message.lower()
    required = task.get("keywords_required", [])
    forbidden = task.get("keywords_forbidden", [])

    hits = sum(1 for k in required if k.lower() in msg)
    bad_hits = sum(1 for k in forbidden if k.lower() in msg)

    if bad_hits > 0:
        return -0.5, f"Forbidden phrases used ({bad_hits})."

    ratio = hits / max(len(required), 1)
    if ratio >= 0.75:
        return 0.5, f"Highly relevant ({hits}/{len(required)} keywords)."
    elif ratio >= 0.5:
        return 0.3, f"Relevant ({hits}/{len(required)} keywords)."
    elif ratio >= 0.25:
        return 0.1, f"Partially relevant ({hits}/{len(required)} keywords)."
    return 0.0, "Response seems off-topic."


def _length_score(message: str) -> tuple:
    words = len(message.split())
    if words < 10:
        return -0.2, "Response too short (under 10 words)."
    if words > 200:
        return -0.1, "Response too long — be concise."
    return 0.1, f"Good length ({words} words)."


def grade(action: dict, task: dict) -> dict:
    reward = 0.0
    breakdown = {}
    feedback = []

    if action.get("action_type") != "respond":
        return {
            "reward": -0.2,
            "breakdown": {"wrong_action_type": -0.2},
            "feedback": ["Expected action_type=respond"]
        }

    message = action.get("message", "")

    ts, tf = _tone_score(message)
    rs, rf = _relevance_score(message, task)
    ls, lf = _length_score(message)

    breakdown["tone"] = ts
    breakdown["relevance"] = rs
    breakdown["length"] = ls
    reward = ts + rs + ls
    feedback += [tf, rf, lf]

    sentiment = task.get("sentiment", "neutral")
    if sentiment in ("angry", "panicked") and ts >= 0.3:
        breakdown["sentiment_handling_bonus"] = 0.2
        reward += 0.2
        feedback.append("Bonus: Handled difficult sentiment well.")

    return {"reward": round(reward, 3), "breakdown": breakdown, "feedback": feedback}