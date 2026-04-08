from env import SupportEnv


def classify(text):
    text = text.lower()

    if any(x in text for x in ["refund", "money back", "return"]):
        return "refund"
    elif any(x in text for x in ["delay", "late", "shipping", "not delivered"]):
        return "shipping"
    elif any(x in text for x in ["cancel", "cancellation"]):
        return "cancel"
    elif any(x in text for x in ["broken", "damaged", "defective"]):
        return "complaint"
    else:
        return "general"


def agent(obs):
    step = obs.get("step_count", 0)
    text = obs.get("ticket_text", "").lower()
    label = classify(text)

    # Step 0: classify
    if step == 0:
        return {
            "action_type": "classify_ticket",
            "label": label
        }

    # Step 1: request info (only if needed)
    if step == 1:
        return {
            "action_type": "request_info",
            "info_field": "order_id",
            "message": "Could you please share your order ID to help us proceed?"
        }

    # Step 2: verify
    if step == 2:
        return {
            "action_type": "verify",
            "field": "order_id",
            "value": obs.get("order_id", "12345")
        }

    # Step 3: smart response
    if step == 3:
        if label == "refund":
            msg = "We’ve initiated your refund process. You’ll receive confirmation shortly."
        elif label == "shipping":
            msg = "We’re checking the shipping status and will update you soon."
        elif label == "cancel":
            msg = "Your cancellation request is being processed."
        elif label == "complaint":
            msg = "We’re sorry for the inconvenience. We’re reviewing your issue carefully."
        else:
            msg = "Thanks for reaching out. We’re looking into your request."

        return {
            "action_type": "respond",
            "message": msg
        }

    # Step 4: resolve based on type
    if label == "refund":
        resolution = "process_refund"
    elif label == "shipping":
        resolution = "check_shipping_status"
    elif label == "cancel":
        resolution = "process_cancellation"
    elif label == "complaint":
        resolution = "escalate_issue"
    else:
        resolution = "general_support"

    return {
        "action_type": "resolve",
        "resolution_type": resolution,
        "message": "Your issue has been successfully resolved. Thank you for your patience!"
    }


def run():
    env = SupportEnv(task_level="hard")
    obs = env.reset()
    total_reward = 0

    while True:
        action = agent(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    print("Final score:", total_reward)


if __name__ == "__main__":
    run()