import time
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END

from agent.state import TicketState
from agent.classifier import TicketClassifier
from agent.kb_retriever import KnowledgeBaseRetriever
from agent.reply_generator import ReplyGenerator
from agent.escalator import TicketEscalator

# Global Retriever Instance
retriever = KnowledgeBaseRetriever()

def classify_node(state: TicketState) -> Dict[str, Any]:
    """Node 1: Classify ticket category and assign priority."""
    ticket_text = state.get("ticket_text", "")
    category, priority, metadata = TicketClassifier.classify(ticket_text)
    
    trace = state.get("execution_trace", [])
    trace.append({
        "node": "classify",
        "timestamp": time.strftime("%H:%M:%S"),
        "details": f"Category: {category.upper()}, Priority: {priority.upper()}",
        "metadata": metadata
    })
    
    return {
        "category": category,
        "priority": priority,
        "execution_trace": trace,
        "status": "classified"
    }

def retrieve_node(state: TicketState) -> Dict[str, Any]:
    """Node 2: Search knowledge base & calculate RAG retrieval confidence score."""
    ticket_text = state.get("ticket_text", "")
    top_docs, confidence = retriever.search(ticket_text, top_k=3)
    
    trace = state.get("execution_trace", [])
    trace.append({
        "node": "retrieve",
        "timestamp": time.strftime("%H:%M:%S"),
        "details": f"Retrieved {len(top_docs)} docs. Top similarity score: {top_docs[0]['similarity_score'] if top_docs else 0:.2f}. Confidence: {confidence:.2f}",
        "retrieved_count": len(top_docs),
        "confidence_score": confidence
    })
    
    return {
        "retrieved_docs": top_docs,
        "confidence_score": confidence,
        "execution_trace": trace,
        "status": "retrieved"
    }

def decide_routing(state: TicketState) -> Literal["draft_reply", "escalate"]:
    """Conditional Edge: Decide whether to automate draft reply or escalate to human agent."""
    confidence = state.get("confidence_score", 0.0)
    threshold = state.get("confidence_threshold", 0.60)
    
    if confidence >= threshold:
        return "draft_reply"
    else:
        return "escalate"

def draft_reply_node(state: TicketState) -> Dict[str, Any]:
    """Node 3a: Generate grounded draft response for high-confidence tickets."""
    ticket_text = state.get("ticket_text", "")
    category = state.get("category", "general")
    priority = state.get("priority", "medium")
    retrieved_docs = state.get("retrieved_docs", [])
    
    draft = ReplyGenerator.generate(ticket_text, category, priority, retrieved_docs)
    
    trace = state.get("execution_trace", [])
    trace.append({
        "node": "draft_reply",
        "timestamp": time.strftime("%H:%M:%S"),
        "details": f"Generated grounded resolution ({len(draft)} chars). Ticket auto-triaged."
    })
    
    return {
        "draft_reply": draft,
        "status": "resolved",
        "execution_trace": trace
    }

def escalate_node(state: TicketState) -> Dict[str, Any]:
    """Node 3b: Escalate ticket to human tier-2 support agent for low-confidence queries."""
    handoff = TicketEscalator.create_handoff(
        ticket_id=state.get("ticket_id", "T-000"),
        customer_id=state.get("customer_id", "C-000"),
        ticket_text=state.get("ticket_text", ""),
        category=state.get("category", "general"),
        priority=state.get("priority", "medium"),
        confidence_score=state.get("confidence_score", 0.0),
        confidence_threshold=state.get("confidence_threshold", 0.60),
        retrieved_docs=state.get("retrieved_docs", [])
    )
    
    trace = state.get("execution_trace", [])
    trace.append({
        "node": "escalate",
        "timestamp": time.strftime("%H:%M:%S"),
        "details": f"Escalated to human support. Reason: {handoff['escalation_reason']}"
    })
    
    return {
        "escalation_reason": handoff["escalation_reason"],
        "draft_reply": handoff["agent_handoff_summary"],
        "status": "escalated",
        "execution_trace": trace
    }

def build_triage_graph():
    """Construct and compile the LangGraph StateGraph pipeline."""
    workflow = StateGraph(TicketState)
    
    # Add Nodes
    workflow.add_node("classify", classify_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("draft_reply", draft_reply_node)
    workflow.add_node("escalate", escalate_node)
    
    # Add Edges
    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "retrieve")
    
    # Conditional Edge from retrieve -> decide routing
    workflow.add_conditional_edges(
        "retrieve",
        decide_routing,
        {
            "draft_reply": "draft_reply",
            "escalate": "escalate"
        }
    )
    
    workflow.add_edge("draft_reply", END)
    workflow.add_edge("escalate", END)
    
    return workflow.compile()

# Global Compiled Agent Pipeline
triage_agent = build_triage_graph()
