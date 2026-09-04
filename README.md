# Support Ticket Triage Agent (LangGraph + RAG)

An autonomous Customer Support Ticket Triage Agent built with **LangGraph**, **Sentence Transformers RAG Retrieval**, and **Reliable AI Conditional Routing**.

## Key Features

- **`classify` Node**: Automatically categorizes tickets into `billing`, `technical`, or `general` and assigns priority.
- **`retrieve` Node**: Performs vector similarity search over a FAQ knowledge base using `SentenceTransformer` embeddings.
- **`decide` Node**: Routes to auto-reply if confidence is high, or escalates to human if confidence is low.
- **`draft_reply` Node**: Generates grounded resolutions citing FAQ documents.
- **`escalate` Node**: Formats human handoff summary for Tier-2 agents.
- **Interactive Dashboard**: Dark glassmorphic web UI with live node step visualizer.

## Quick Start (Local)

```bash
pip install -r requirements.txt
python server.py
# Open http://127.0.0.1:8000
```
