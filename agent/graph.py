import time
from typing import Dict, Any, Literal, Generator
from langgraph.graph import StateGraph, END

from agent.state import TicketState
from agent.classifier import TicketClassifier
from agent.kb_retriever import KnowledgeBaseRetriever
from agent.reply_generator import ReplyGenerator
from agent.escalator import TicketEscalator

# Global Thread-Safe Retriever Instance
retriever = KnowledgeBaseRetriever()

def classify_node(state: TicketState) -> Dict[str, Any]:
    """Node 1: Classify ticket category and assign priority with negation awareness."""
    ticket_text = state.get("ticket_text", "")
    start_t = time.time()
    category, priority, metadata = TicketClassifier.classify(ticket_text)
    latency_ms = round((time.time() - start_t) * 1000, 2)
    
    trace = state.get("execution_trace", [])
    trace.append({
        "node": "classify",
        "timestamp": time.strftime("%H:%M:%S"),
        "latency_ms": latency_ms,
        "details": f"Category: {category.upper()}, Priority: {priority.upper()} (Engine: {metadata.get('engine', 'rules')})",
        "metadata": metadata
    })
    
    return {
        "category": category,
        "priority": priority,
        "execution_trace": trace,
        "status": "classified"
    }

def retrieve_node(state: TicketState) -> Dict[str, Any]:
    """Node 2: Search knowledge base with Category Metadata Filtering & calculate confidence."""
    ticket_text = state.get("ticket_text", "")
    category = state.get("category")
    start_t = time.time()
    
    top_docs, confidence = retriever.search(ticket_text, top_k=3, category=category)
    latency_ms = round((time.time() - start_t) * 1000, 2)
    
    boosted_count = sum(1 for d in top_docs if d.get("category_boosted"))
    trace = state.get("execution_trace", [])
    trace.append({
        "node": "retrieve",
        "timestamp": time.strftime("%H:%M:%S"),
        "latency_ms": latency_ms,
        "details": f"Retrieved {len(top_docs)} docs ({boosted_count} category-boosted). Top score: {top_docs[0]['similarity_score'] if top_docs else 0:.2f}. Confidence: {confidence:.2f}",
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
    start_t = time.time()
    
    draft = ReplyGenerator.generate(ticket_text, category, priority, retrieved_docs)
    latency_ms = round((time.time() - start_t) * 1000, 2)
    
    trace = state.get("execution_trace", [])
    trace.append({
        "node": "draft_reply",
        "timestamp": time.strftime("%H:%M:%S"),
        "latency_ms": latency_ms,
        "details": f"Generated grounded resolution ({len(draft)} chars). Ticket auto-triaged."
    })
    
    return {
        "draft_reply": draft,
        "status": "resolved",
        "execution_trace": trace
    }

def escalate_node(state: TicketState) -> Dict[str, Any]:
    """Node 3b: Escalate ticket to human tier-2 support agent for low-confidence queries."""
    start_t = time.time()
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
    latency_ms = round((time.time() - start_t) * 1000, 2)
    
    trace = state.get("execution_trace", [])
    trace.append({
        "node": "escalate",
        "timestamp": time.strftime("%H:%M:%S"),
        "latency_ms": latency_ms,
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

def stream_triage_pipeline(initial_state: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    Step-by-step generator for real-time Server-Sent Events (SSE) streaming.
    Yields event dictionaries as each node begins and finishes.
    """
    current_state = dict(initial_state)
    current_state["execution_trace"] = []
    
    # 1. Classify Node
    yield {"event": "node_start", "node": "classify", "message": "Analyzing ticket intent & priority..."}
    classify_update = classify_node(current_state)
    current_state.update(classify_update)
    yield {
        "event": "node_complete",
        "node": "classify",
        "category": current_state["category"],
        "priority": current_state["priority"],
        "trace": current_state["execution_trace"][-1]
    }
    
    # 2. Retrieve Node
    yield {"event": "node_start", "node": "retrieve", "message": f"Querying FAQ vector index with {current_state['category']} domain boost..."}
    retrieve_update = retrieve_node(current_state)
    current_state.update(retrieve_update)
    yield {
        "event": "node_complete",
        "node": "retrieve",
        "confidence_score": current_state["confidence_score"],
        "retrieved_count": len(current_state.get("retrieved_docs", [])),
        "trace": current_state["execution_trace"][-1]
    }
    
    # 3. Decide Edge
    decision = decide_routing(current_state)
    yield {
        "event": "decide",
        "decision": decision,
        "confidence": current_state["confidence_score"],
        "threshold": current_state["confidence_threshold"],
        "message": f"Confidence {current_state['confidence_score']:.2f} {'meets' if decision == 'draft_reply' else 'below'} threshold {current_state['confidence_threshold']:.2f}"
    }
    
    # 4. Final Node
    if decision == "draft_reply":
        yield {"event": "node_start", "node": "draft_reply", "message": "Formulating grounded automated response..."}
        final_update = draft_reply_node(current_state)
        current_state.update(final_update)
        yield {
            "event": "node_complete",
            "node": "draft_reply",
            "status": "resolved",
            "trace": current_state["execution_trace"][-1]
        }
    else:
        yield {"event": "node_start", "node": "escalate", "message": "Formatting safe human handoff payload..."}
        final_update = escalate_node(current_state)
        current_state.update(final_update)
        yield {
            "event": "node_complete",
            "node": "escalate",
            "status": "escalated",
            "trace": current_state["execution_trace"][-1]
        }
        
    # Final Result
    yield {"event": "pipeline_complete", "final_state": current_state}
