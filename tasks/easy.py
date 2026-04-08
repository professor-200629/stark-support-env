"""
Task 1 (Easy): Ticket Classification
The agent must classify a customer message into the correct ticket type.
"""

CLASSIFICATION_TASKS = [
    {"customer_message": "My order didn't arrive and it's been 2 weeks!", "sentiment": "angry", "expected_label": "delivery_issue", "ticket_id": "easy_001"},
    {"customer_message": "I was charged twice for the same item.", "sentiment": "frustrated", "expected_label": "billing", "ticket_id": "easy_002"},
    {"customer_message": "The product I received is broken.", "sentiment": "disappointed", "expected_label": "product_defect", "ticket_id": "easy_003"},
    {"customer_message": "I can't log into my account, password reset doesn't work.", "sentiment": "confused", "expected_label": "account_access", "ticket_id": "easy_004"},
    {"customer_message": "I want to cancel my subscription immediately.", "sentiment": "neutral", "expected_label": "cancellation", "ticket_id": "easy_005"},
    {"customer_message": "I never received a refund for my returned item.", "sentiment": "angry", "expected_label": "refund", "ticket_id": "easy_006"},
    {"customer_message": "Your app keeps crashing on my phone.", "sentiment": "frustrated", "expected_label": "technical_issue", "ticket_id": "easy_007"},
    {"customer_message": "I think someone hacked my account.", "sentiment": "panicked", "expected_label": "fraud", "ticket_id": "easy_008"},
]

VALID_LABELS = [
    "delivery_issue", "billing", "product_defect", "account_access",
    "cancellation", "refund", "technical_issue", "fraud", "other"
]