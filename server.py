import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from agent.graph import triage_agent, retriever

app = FastAPI(
    title="Support Ticket Triage Agent API",
    description="LangGraph Support Ticket Triage Pipeline with RAG and Reliable AI Routing",
    version="1.0.0"
)

# CORS middleware - allow all origins for public deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request & Response Models
class TicketRequest(BaseModel):
    ticket_text: str = Field(..., description="Content of customer support ticket")
    customer_id: Optional[str] = Field("CUST-101", description="Optional customer ID")
    confidence_threshold: Optional[float] = Field(0.60, ge=0.0, le=1.0, description="Routing threshold")

class FAQAddRequest(BaseModel):
    title: str
    category: str
    content: str
    tags: List[str] = []

@app.post("/api/triage")
def process_ticket(req: TicketRequest):
    if not req.ticket_text.strip():
        raise HTTPException(status_code=400, detail="Ticket text cannot be empty.")

    ticket_id = f"TCK-{uuid.uuid4().hex[:6].upper()}"
    
    initial_state = {
        "ticket_id": ticket_id,
        "customer_id": req.customer_id,
        "ticket_text": req.ticket_text,
        "confidence_threshold": req.confidence_threshold,
        "execution_trace": []
    }
    
    try:
        final_state = triage_agent.invoke(initial_state)
        return {
            "success": True,
            "ticket_id": final_state.get("ticket_id"),
            "customer_id": final_state.get("customer_id"),
            "ticket_text": final_state.get("ticket_text"),
            "category": final_state.get("category"),
            "priority": final_state.get("priority"),
            "confidence_score": final_state.get("confidence_score"),
            "confidence_threshold": final_state.get("confidence_threshold"),
            "retrieved_docs": final_state.get("retrieved_docs", []),
            "draft_reply": final_state.get("draft_reply"),
            "escalation_reason": final_state.get("escalation_reason"),
            "status": final_state.get("status"),
            "execution_trace": final_state.get("execution_trace", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing triage graph: {str(e)}")

@app.get("/api/faqs")
def get_faqs():
    return {
        "count": len(retriever.docs),
        "docs": retriever.docs
    }

@app.post("/api/faqs")
def add_faq(req: FAQAddRequest):
    new_doc = retriever.add_doc(
        title=req.title,
        category=req.category,
        content=req.content,
        tags=req.tags
    )
    return {"success": True, "doc": new_doc}

@app.get("/api/graph")
def get_graph_schema():
    return {
        "nodes": [
            {"id": "classify", "label": "1. Classify Node", "type": "process", "desc": "Categorizes ticket & assigns priority"},
            {"id": "retrieve", "label": "2. Retrieve Node", "type": "process", "desc": "SentenceTransformer RAG embedding search"},
            {"id": "decide", "label": "3. Decide Node", "type": "decision", "desc": "Checks confidence >= threshold"},
            {"id": "draft_reply", "label": "4a. Draft Reply Node", "type": "terminal", "desc": "Generates grounded auto-reply"},
            {"id": "escalate", "label": "4b. Escalate Node", "type": "terminal", "desc": "Human handoff fallback"}
        ],
        "edges": [
            {"from": "START", "to": "classify"},
            {"from": "classify", "to": "retrieve"},
            {"from": "retrieve", "to": "decide"},
            {"from": "decide", "to": "draft_reply", "condition": "confidence >= threshold"},
            {"from": "decide", "to": "escalate", "condition": "confidence < threshold"}
        ]
    }

# Static path
static_path = os.path.join(os.path.dirname(__file__), "static")

# --- Non-static routes MUST come BEFORE app.mount() ---

@app.get("/health")
@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "support-ticket-triage-agent"}

@app.get("/", response_class=HTMLResponse)
def index():
    index_file = os.path.join(static_path, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return "<h1>Support Ticket Triage Agent API is running!</h1>"

# Mount static files LAST so it doesn't shadow API routes
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"[*] Starting server on 0.0.0.0:{port}...")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)

