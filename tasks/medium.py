"""
Task 2 (Medium): Proper Response Generation
The agent must respond to a customer message with the correct tone and relevance.
"""

RESPONSE_TASKS = [
    {
        "ticket_id": "med_001",
        "customer_message": "I was charged twice for my order #4521!",
        "sentiment": "angry",
        "ticket_type": "billing",
        "context": {"order_id": "4521", "charge_count": 2},
        "keywords_required": ["sorry", "check", "refund", "billing"],
        "keywords_forbidden": ["not our problem", "your fault", "impossible"],
    },
    {
        "ticket_id": "med_002",
        "customer_message": "My delivery is 5 days late. Where is my package?",
        "sentiment": "frustrated",
        "ticket_type": "delivery_issue",
        "context": {"order_id": "9988", "days_late": 5},
        "keywords_required": ["apologize", "track", "investigate", "update"],
        "keywords_forbidden": ["not sure", "can't help", "wait longer"],
    },
    {
        "ticket_id": "med_003",
        "customer_message": "Your product stopped working after 2 days.",
        "sentiment": "disappointed",
        "ticket_type": "product_defect",
        "context": {"product": "SmartWatch X2", "days_used": 2},
        "keywords_required": ["sorry", "replace", "warranty", "defect"],
        "keywords_forbidden": ["user error", "not covered", "buy new"],
    },
    {
        "ticket_id": "med_004",
        "customer_message": "I've been trying to reset my password for 3 hours!",
        "sentiment": "frustrated",
        "ticket_type": "account_access",
        "context": {"attempts": 3},
        "keywords_required": ["help", "reset", "account", "send"],
        "keywords_forbidden": ["simple", "just", "obviously"],
    },
]