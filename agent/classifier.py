import os
import re
import json
from typing import Tuple, Dict, Any

try:
    from google import genai
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False

class TicketClassifier:
    """
    Production-grade Intent & Priority Classifier.
    Features:
    1. LLM Structured Output (Gemini 2.5 Flash) when API key is configured.
    2. Semantic Negation-Aware Rule Engine with word boundaries & context windows.
    3. Intent Confidence & Urgency Scoring.
    """

    NEGATION_PATTERNS = [
        r"\b(?:not|don't|dont|do not|never|no|without|rather than|instead of|stop)\b"
    ]

    BILLING_KEYWORDS = [
        r"\brefund\b", r"\bpayments?\b", r"\bcredit cards?\b", r"\binvoices?\b",
        r"\bcharges?\b", r"\bsubscriptions?\b", r"\bbilling\b", r"\btax\b",
        r"\bvat\b", r"\bgst\b", r"\breceipts?\b", r"\bdowngrade\b", r"\bcancel(?:led|ing)? plan\b",
        r"\bovercharged?\b", r"\bcosts?\b", r"\bpric(?:ing|e)\b", r"\bpay\b"
    ]
    
    TECHNICAL_KEYWORDS = [
        r"\berrors?\b", r"\bbugs?\b", r"\bcrashes?\b", r"\bpasswords?\b", r"\blockouts?\b",
        r"\blogins?\b", r"\b429\b", r"\brate limits?\b", r"\bapis?\b", r"\bwebhooks?\b",
        r"\b2fa\b", r"\bcors\b", r"\bsdks?\b", r"\btokens?\b", r"\bauth\b", r"\bauthenticat(?:ion|e)\b",
        r"\bfailed\b", r"\bconnections?\b", r"\bhttps?\b", r"\btimeouts?\b", r"\bexceptions?\b",
        r"\bservers?\b", r"\bdeployments?\b", r"\bdatabase\b"
    ]
    
    HIGH_PRIORITY_KEYWORDS = [
        r"\burgent\b", r"\bemergency\b", r"\block(?:ed)? out\b", r"\bsystem down\b",
        r"\boutages?\b", r"\bpayment failed\b", r"\bdeclined\b", r"\bunauthorized\b",
        r"\bsecurity breach\b", r"\bproduction broken\b", r"\bcritical\b", r"\basap\b",
        r"\bimmediately\b", r"\bhacked\b"
    ]
    
    MEDIUM_PRIORITY_KEYWORDS = [
        r"\berrors?\b", r"\b429\b", r"\brate limits?\b", r"\brefund\b", r"\binvoices?\b",
        r"\bwebhook failure\b", r"\bcannot login\b", r"\breset password\b", r"\bcors\b"
    ]

    @classmethod
    def classify(cls, ticket_text: str) -> Tuple[str, str, Dict[str, Any]]:
        api_key = os.environ.get("GEMINI_API_KEY")
        
        # 1. Attempt LLM Structured Classification if API key is present
        if HAS_GOOGLE_GENAI and api_key:
            llm_result = cls._classify_with_llm(ticket_text, api_key)
            if llm_result:
                return llm_result

        # 2. Semantic Negation-Aware Rule Engine
        return cls._classify_with_semantic_rules(ticket_text)

    @classmethod
    def _is_negated(cls, text: str, match_start: int, window: int = 35) -> bool:
        """Checks if a keyword match is preceded by a negation within a context window."""
        start = max(0, match_start - window)
        prefix = text[start:match_start].lower()
        for neg in cls.NEGATION_PATTERNS:
            if re.search(neg, prefix):
                return True
        return False

    @classmethod
    def _count_valid_matches(cls, text: str, patterns: list) -> Tuple[int, int]:
        """Counts positive matches while discounting negated matches."""
        positive_count = 0
        negated_count = 0
        for pat in patterns:
            for match in re.finditer(pat, text, re.IGNORECASE):
                if cls._is_negated(text, match.start()):
                    negated_count += 1
                else:
                    positive_count += 1
        return positive_count, negated_count

    @classmethod
    def _classify_with_semantic_rules(cls, ticket_text: str) -> Tuple[str, str, Dict[str, Any]]:
        text_lower = ticket_text.lower()

        # Check Category Matches with Negation Awareness
        bill_pos, bill_neg = cls._count_valid_matches(text_lower, cls.BILLING_KEYWORDS)
        tech_pos, tech_neg = cls._count_valid_matches(text_lower, cls.TECHNICAL_KEYWORDS)

        # Net effective score
        net_billing = max(0, bill_pos - (bill_neg * 2))
        net_technical = max(0, tech_pos - (tech_neg * 2))

        total_signals = net_billing + net_technical
        if net_billing > net_technical and net_billing > 0:
            category = "billing"
            confidence = round(net_billing / max(1, total_signals), 2)
        elif net_technical > net_billing and net_technical > 0:
            category = "technical"
            confidence = round(net_technical / max(1, total_signals), 2)
        else:
            category = "general"
            confidence = 0.50

        # Check Priority Matches
        high_pos, _ = cls._count_valid_matches(text_lower, cls.HIGH_PRIORITY_KEYWORDS)
        med_pos, _ = cls._count_valid_matches(text_lower, cls.MEDIUM_PRIORITY_KEYWORDS)

        if high_pos > 0:
            priority = "high"
        elif med_pos > 0 or len(ticket_text) > 220:
            priority = "medium"
        else:
            priority = "low"

        metadata = {
            "engine": "semantic_negation_rules",
            "net_billing_signals": net_billing,
            "net_technical_signals": net_technical,
            "negated_billing_signals": bill_neg,
            "negated_technical_signals": tech_neg,
            "intent_confidence": confidence
        }

        return category, priority, metadata

    @classmethod
    def _classify_with_llm(cls, ticket_text: str, api_key: str) -> Tuple[str, str, Dict[str, Any]] | None:
        try:
            client = genai.Client(api_key=api_key)
            prompt = f"""Analyze this customer support ticket and classify it into exact JSON:
Ticket: "{ticket_text}"

Requirements:
- category: one of ["billing", "technical", "general"]
- priority: one of ["high", "medium", "low"]
- reason: brief 1-line reason

Respond ONLY with valid JSON in this format:
{{"category": "...", "priority": "...", "reason": "..."}}
"""
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            if response and response.text:
                clean = response.text.strip().replace("```json", "").replace("```", "").strip()
                data = json.loads(clean)
                cat = data.get("category", "general").lower()
                pri = data.get("priority", "medium").lower()
                if cat not in ["billing", "technical", "general"]:
                    cat = "general"
                if pri not in ["high", "medium", "low"]:
                    pri = "medium"
                return cat, pri, {"engine": "gemini-2.5-flash", "reason": data.get("reason", "")}
        except Exception as e:
            print(f"[WARN] LLM Classifier fallback: {e}")
        return None
