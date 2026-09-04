from typing import List, Dict, Any, Optional

class TicketEscalator:
    """
    Formats human handoff tickets for low-confidence or high-complexity customer cases.
    """

    @classmethod
    def create_handoff(
        cls,
        ticket_id: str,
        customer_id: str,
        ticket_text: str,
        category: str,
        priority: str,
        confidence_score: float,
        confidence_threshold: float,
        retrieved_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        
        reason = (
            f"Low retrieval confidence ({confidence_score:.2f} < threshold {confidence_threshold:.2f}). "
            "Automated system could not guarantee grounded answer accuracy with sufficient certainty."
        )
        
        suggested_notes = []
        if retrieved_docs:
            suggested_notes.append("Top partial matches evaluated:")
            for doc in retrieved_docs:
                suggested_notes.append(f" - [{doc.get('id')}] {doc.get('title')} (score: {doc.get('similarity_score', 0):.2f})")
        else:
            suggested_notes.append("No relevant knowledge base articles matched the customer's query.")

        handoff_ticket = {
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "category": category,
            "priority": priority,
            "confidence_score": confidence_score,
            "escalation_reason": reason,
            "status": "escalated_to_human",
            "original_query": ticket_text,
            "agent_handoff_summary": (
                f"[HUMAN ESCALATION MANDATED]\n"
                f"Reason: {reason}\n"
                f"Suggested Investigation: Customer issue marked as '{category.upper()}' with '{priority.upper()}' priority.\n"
                f"{chr(10).join(suggested_notes)}\n"
                f"Action Required: Assign human tier-2 support representative for manual triage."
            )
        }
        
        return handoff_ticket
