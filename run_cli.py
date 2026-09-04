import uuid
from agent.graph import triage_agent

TEST_TICKETS = [
    {
        "name": "Password Reset (Technical - High Confidence)",
        "text": "I forgot my account password and got locked out after 5 login attempts. Can you help reset it?",
        "threshold": 0.60
    },
    {
        "name": "Payment Failed (Billing - High Confidence)",
        "text": "My credit card payment failed when attempting to renew my subscription today.",
        "threshold": 0.60
    },
    {
        "name": "Ambiguous/Out-of-Scope (Low Confidence Escalation)",
        "text": "Can your team build a custom satellite communication protocol plugin for my smartwatch?",
        "threshold": 0.65
    }
]

def run_test():
    print("=" * 70)
    print("      SUPPORT TICKET TRIAGE AGENT - LANGGRAPH WORKFLOW CLI      ")
    print("=" * 70)
    
    for idx, test in enumerate(TEST_TICKETS, 1):
        print(f"\n--- [Test #{idx}] {test['name']} ---")
        print(f"Ticket Query: \"{test['text']}\"")
        
        initial_state = {
            "ticket_id": f"TCK-{uuid.uuid4().hex[:6].upper()}",
            "customer_id": f"CUST-{idx:03d}",
            "ticket_text": test["text"],
            "confidence_threshold": test["threshold"],
            "execution_trace": []
        }
        
        final_state = triage_agent.invoke(initial_state)
        
        print(f"Assigned Category   : {final_state.get('category', '').upper()}")
        print(f"Assigned Priority   : {final_state.get('priority', '').upper()}")
        print(f"Confidence Score    : {final_state.get('confidence_score', 0):.4f} (Threshold: {final_state.get('confidence_threshold'):.2f})")
        print(f"Final Status        : {final_state.get('status', '').upper()}")
        
        print("\nExecution Path:")
        for step in final_state.get("execution_trace", []):
            print(f"  |- [{step['timestamp']}] Node '{step['node']}': {step['details']}")
            
        print("\nGenerated Output:")
        print("-" * 50)
        print(final_state.get("draft_reply") or final_state.get("escalation_reason"))
        print("-" * 50)

if __name__ == "__main__":
    run_test()
