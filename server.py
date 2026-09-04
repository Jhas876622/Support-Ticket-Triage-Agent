import os
import time
import json
import uuid
import threading
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from agent.graph import triage_agent, retriever, stream_triage_pipeline

app = FastAPI(
    title="Support Ticket Triage Agent API",
    description="Enterprise LangGraph Support Ticket Triage Pipeline with RAG, SSE Streaming & Live Analytics",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Thread-Safe Analytics Metric Store ---
class AnalyticsTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self.total_processed = 0
        self.auto_resolved = 0
        self.escalated = 0
        self.confidence_sum = 0.0
        self.categories = {"billing": 0, "technical": 0, "general": 0}
        self.recent_tickets = []

    def record_triage(self, category: str, status: str, confidence: float, ticket_id: str, ticket_text: str):
        with self._lock:
            self.total_processed += 1
            if status == "resolved":
                self.auto_resolved += 1
            else:
                self.escalated += 1
            self.confidence_sum += confidence
            cat = category.lower() if category else "general"
            self.categories[cat] = self.categories.get(cat, 0) + 1
            
            # Keep last 10 tickets in memory for audit
            self.recent_tickets.insert(0, {
                "id": ticket_id,
                "text": ticket_text[:80] + ("..." if len(ticket_text) > 80 else ""),
                "category": category,
                "status": status,
                "confidence": round(confidence, 2),
                "timestamp": time.strftime("%H:%M:%S")
            })
            if len(self.recent_tickets) > 10:
                self.recent_tickets.pop()

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            avg_conf = (self.confidence_sum / self.total_processed) if self.total_processed > 0 else 0.0
            auto_rate = round((self.auto_resolved / self.total_processed * 100), 1) if self.total_processed > 0 else 0.0
            esc_rate = round((self.escalated / self.total_processed * 100), 1) if self.total_processed > 0 else 0.0
            return {
                "total_processed": self.total_processed,
                "auto_resolved": self.auto_resolved,
                "escalated": self.escalated,
                "auto_resolution_rate_pct": auto_rate,
                "escalation_rate_pct": esc_rate,
                "avg_confidence_score": round(avg_conf, 2),
                "category_breakdown": dict(self.categories),
                "recent_tickets": list(self.recent_tickets)
            }

analytics = AnalyticsTracker()

# --- Request Tracing Middleware ---
@app.middleware("http")
async def add_process_time_and_trace_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", f"req-{uuid.uuid4().hex[:8]}")
    start_time = time.time()
    response: Response = await call_next(request)
    process_time = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = str(process_time)
    return response

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

# --- Real-Time Server-Sent Events (SSE) Streaming Endpoint ---
@app.post("/api/triage/stream")
def process_ticket_stream(req: TicketRequest):
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

    def event_generator():
        final_state = {}
        try:
            for step in stream_triage_pipeline(initial_state):
                if step.get("event") == "pipeline_complete":
                    final_state = step.get("final_state", {})
                    # Record metrics
                    analytics.record_triage(
                        category=final_state.get("category", "general"),
                        status=final_state.get("status", "resolved"),
                        confidence=final_state.get("confidence_score", 0.0),
                        ticket_id=final_state.get("ticket_id", ticket_id),
                        ticket_text=req.ticket_text
                    )
                payload = json.dumps(step)
                yield f"data: {payload}\n\n"
        except Exception as e:
            err_payload = json.dumps({"event": "error", "error": str(e)})
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# --- Standard Batch Triage Endpoint ---
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
        # Record analytics
        analytics.record_triage(
            category=final_state.get("category", "general"),
            status=final_state.get("status", "resolved"),
            confidence=final_state.get("confidence_score", 0.0),
            ticket_id=ticket_id,
            ticket_text=req.ticket_text
        )
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

@app.get("/api/analytics")
def get_analytics():
    return analytics.get_metrics()

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
            {"id": "classify", "label": "1. Classify Node", "type": "process", "desc": "Categorizes ticket & assigns priority (Negation-aware)"},
            {"id": "retrieve", "label": "2. Retrieve Node", "type": "process", "desc": "Category-Boosted Vector/TF-IDF RAG Search"},
            {"id": "decide", "label": "3. Decide Node", "type": "decision", "desc": "Checks confidence >= threshold"},
            {"id": "draft_reply", "label": "4a. Draft Reply Node", "type": "terminal", "desc": "Generates grounded auto-reply"},
            {"id": "escalate", "label": "4b. Escalate Node", "type": "terminal", "desc": "Human handoff fallback with HITL Console"}
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

# Health endpoints
@app.get("/health")
@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "support-ticket-triage-agent", "version": "2.0.0"}

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
