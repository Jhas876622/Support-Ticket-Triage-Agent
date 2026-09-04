import re
from typing import Tuple, Dict, Any

class TicketClassifier:
    """
    Classifies support tickets into categories ('billing', 'technical', 'general')
    and assigns priority ('high', 'medium', 'low').
    """

    BILLING_KEYWORDS = [
        "refund", "payment", "credit card", "invoice", "charge", "subscription",
        "billing", "tax", "vat", "gst", "receipt", "downgrade", "cancel plan",
        "overcharged", "money", "cost", "pricing", "pay"
    ]
    
    TECHNICAL_KEYWORDS = [
        "error", "bug", "crash", "password", "lockout", "login", "429", "rate limit",
        "api", "webhook", "2fa", "cors", "sdk", "token", "auth", "authentication",
        "failed", "connection", "http", "timeout", "exception", "server error"
    ]
    
    HIGH_PRIORITY_KEYWORDS = [
        "urgent", "emergency", "lockout", "locked out", "system down", "outage",
        "payment failed", "declined", "unauthorized", "security breach", "production broken",
        "critical", "asap", "immediately"
    ]
    
    MEDIUM_PRIORITY_KEYWORDS = [
        "error", "429", "rate limit", "refund", "invoice", "webhook failure",
        "cannot login", "reset password", "cors"
    ]

    @classmethod
    def classify(cls, ticket_text: str) -> Tuple[str, str, Dict[str, Any]]:
        text_lower = ticket_text.lower()
        
        # 1. Determine Category
        billing_matches = sum(1 for kw in cls.BILLING_KEYWORDS if kw in text_lower)
        technical_matches = sum(1 for kw in cls.TECHNICAL_KEYWORDS if kw in text_lower)
        
        if billing_matches > technical_matches and billing_matches > 0:
            category = "billing"
        elif technical_matches > billing_matches and technical_matches > 0:
            category = "technical"
        else:
            category = "general"

        # 2. Determine Priority
        high_matches = sum(1 for kw in cls.HIGH_PRIORITY_KEYWORDS if kw in text_lower)
        medium_matches = sum(1 for kw in cls.MEDIUM_PRIORITY_KEYWORDS if kw in text_lower)
        
        if high_matches > 0:
            priority = "high"
        elif medium_matches > 0 or len(ticket_text) > 200:
            priority = "medium"
        else:
            priority = "low"
            
        metadata = {
            "billing_keyword_hits": billing_matches,
            "technical_keyword_hits": technical_matches,
            "high_priority_hits": high_matches,
            "medium_priority_hits": medium_matches
        }
        
        return category, priority, metadata
