from typing import TypedDict, List, Dict, Any, Optional

class TicketState(TypedDict):
    ticket_id: str
    customer_id: str
    ticket_text: str
    category: Optional[str]            # 'billing' | 'technical' | 'general'
    priority: Optional[str]            # 'low' | 'medium' | 'high'
    confidence_score: float            # 0.0 to 1.0 (retrieval confidence score)
    confidence_threshold: float        # e.g., 0.60
    retrieved_docs: List[Dict[str, Any]] # List of top FAQ matches with similarity scores
    draft_reply: Optional[str]         # Grounded response generated for high-confidence tickets
    escalation_reason: Optional[str]   # Explanation if ticket is sent to human agent
    status: str                        # 'classified' | 'retrieved' | 'resolved' | 'escalated'
    execution_trace: List[Dict[str, Any]] # List of graph node steps executed with timestamps
