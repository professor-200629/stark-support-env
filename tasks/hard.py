"""
Task 3 (Hard): Full Multi-Step Issue Resolution
"""

HARD_SCENARIOS = [
    {
        "ticket_id": "hard_001",
        "initial_message": "I was charged twice this month and I want my money back NOW!",
        "sentiment": "angry",
        "ticket_type": "billing",
        "hidden_info": {
            "order_id": "ORD-7721",
            "charge_dates": ["2026-03-01", "2026-03-01"],
            "amount": 49.99,
            "payment_method": "Visa ending 4242"
        },
        "resolution_steps": ["classify", "request_info", "verify", "respond", "resolve"],
        "required_info_fields": ["order_id", "payment_method"],
        "correct_resolution": "process_refund",
        "edge_case": None
    },
    {
        "ticket_id": "hard_002",
        "initial_message": "I returned my item 3 weeks ago and still no refund!",
        "sentiment": "frustrated",
        "ticket_type": "refund",
        "hidden_info": {
            "return_tracking": "RET-4491",
            "return_received_date": "2026-03-14",
            "refund_status": "pending_review",
            "amount": 89.99
        },
        "resolution_steps": ["classify", "request_info", "investigate", "respond", "resolve"],
        "required_info_fields": ["return_tracking", "order_id"],
        "correct_resolution": "expedite_refund",
        "edge_case": None
    },
    {
        "ticket_id": "hard_003",
        "initial_message": "Someone made purchases on my account that weren't me!",
        "sentiment": "panicked",
        "ticket_type": "fraud",
        "hidden_info": {
            "suspicious_transactions": 3,
            "total_unauthorized": 249.97,
            "account_age_days": 730
        },
        "resolution_steps": ["classify", "escalate", "request_info", "respond", "resolve"],
        "required_info_fields": ["account_email", "last_login_date"],
        "correct_resolution": "freeze_account_and_refund",
        "edge_case": "fraud"
    },
    {
        "ticket_id": "hard_004",
        "initial_message": "I want a refund. I just don't like it.",
        "sentiment": "neutral",
        "ticket_type": "refund",
        "hidden_info": {
            "purchase_date": "2026-01-01",
            "return_window_days": 30,
            "days_since_purchase": 95,
            "previous_refund_requests": 3
        },
        "resolution_steps": ["classify", "request_info", "verify", "respond"],
        "required_info_fields": ["order_id", "purchase_date"],
        "correct_resolution": "deny_refund_politely",
        "edge_case": "refund_abuse"
    },
    {
        "ticket_id": "hard_005",
        "initial_message": "asdfgh my thing broken help",
        "sentiment": "confused",
        "ticket_type": "unknown",
        "hidden_info": {
            "actual_issue": "product_defect",
            "product": "Bluetooth Speaker Z"
        },
        "resolution_steps": ["classify", "request_info", "request_info", "respond", "resolve"],
        "required_info_fields": ["product_name", "issue_description"],
        "correct_resolution": "guide_troubleshoot_or_replace",
        "edge_case": "unclear_message"
    },
]

SENTIMENT_MULTIPLIERS = {
    "angry": 0.9,
    "panicked": 0.85,
    "frustrated": 0.95,
    "confused": 0.9,
    "neutral": 1.0,
    "disappointed": 0.95
}