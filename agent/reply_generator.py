import os
from typing import List, Dict, Any

try:
    from google import genai
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False

class ReplyGenerator:
    """
    Generates grounded support responses using retrieved documentation chunks
    and optional LLM generation.
    """

    @classmethod
    def generate(cls, ticket_text: str, category: str, priority: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        api_key = os.environ.get("GEMINI_API_KEY")
        
        if HAS_GOOGLE_GENAI and api_key:
            try:
                client = genai.Client(api_key=api_key)
                
                context_str = "\n\n".join([
                    f"--- FAQ Doc: {doc.get('title')} (ID: {doc.get('id')}) ---\nCategory: {doc.get('category')}\nContent: {doc.get('content')}"
                    for doc in retrieved_docs
                ])
                
                prompt = f"""You are an empathetic, highly professional Customer Support Agent.
A customer submitted the following support ticket:

Customer Ticket: "{ticket_text}"
Ticket Category: {category}
Assigned Priority: {priority}

Use the following official Grounding Knowledge Base articles to formulate a helpful, precise resolution:

{context_str}

Instructions:
1. Greet the user warmly and acknowledge their issue.
2. Provide a clear step-by-step resolution based strictly on the grounded FAQ articles above.
3. Cite the relevant article title or policy (e.g. "[Ref: {retrieved_docs[0].get('title') if retrieved_docs else 'Help Center'}]").
4. Keep the tone helpful, clear, and reassuring.
5. End with a polite sign-off.
"""
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                print(f"[WARN] LLM API call failed ({e}), falling back to template-grounded response.")

        # High-quality template-grounded fallback response generator
        return cls._generate_template_reply(ticket_text, category, priority, retrieved_docs)

    @classmethod
    def _generate_template_reply(cls, ticket_text: str, category: str, priority: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        top_doc = retrieved_docs[0] if retrieved_docs else None
        
        greeting = "Hello,"
        intro = f"Thank you for contacting our support team regarding your {category} query."
        
        if not top_doc:
            return f"{greeting}\n\n{intro}\n\nWe have received your ticket and are currently reviewing standard procedures. An agent will follow up shortly.\n\nBest regards,\nCustomer Support Team"
            
        doc_title = top_doc.get("title", "Help Article")
        doc_content = top_doc.get("content", "")
        doc_id = top_doc.get("id", "")
        
        body = (
            f"Based on your request, here is the official resolution regarding **{doc_title}**:\n\n"
            f"> {doc_content}\n\n"
            f"### Recommended Steps:\n"
            f"1. Please follow the instructions detailed above in your account settings.\n"
            f"2. If you experience any further issues, respond directly to this email with screenshot details.\n\n"
            f"_Reference Document: [{doc_id}] {doc_title}_"
        )
        
        if len(retrieved_docs) > 1:
            related = "\n".join([f"- **{d.get('title')}**: {d.get('content')[:120]}..." for d in retrieved_docs[1:]])
            body += f"\n\n### Related Help Resources:\n{related}"
            
        sign_off = "\n\nHope this helps resolve your issue! Please let us know if you need any additional assistance.\n\nBest regards,\nCustomer Support Automation Team"
        
        return f"{greeting}\n\n{intro}\n\n{body}{sign_off}"
