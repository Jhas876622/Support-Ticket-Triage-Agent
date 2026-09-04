// Global State
let allFaqs = [];

const PRESETS = {
    password: {
        text: "I forgot my account password and got locked out after 5 login attempts. Can you help reset it?",
        threshold: 0.60,
        cust: "CUST-8821"
    },
    billing: {
        text: "My credit card payment failed when attempting to renew my subscription today.",
        threshold: 0.60,
        cust: "CUST-3049"
    },
    api: {
        text: "My application is getting HTTP 429 Rate Limit Exceeded errors when hitting the search API endpoint.",
        threshold: 0.60,
        cust: "CUST-5510"
    },
    escalate: {
        text: "Can your team build a custom satellite communication protocol plugin for my smartwatch hardware?",
        threshold: 0.65,
        cust: "CUST-9901"
    }
};

document.addEventListener("DOMContentLoaded", () => {
    setupTabNavigation();
    loadFaqs();
});

// Tab Switcher
function setupTabNavigation() {
    const tabs = document.querySelectorAll(".tab-btn");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

            tab.classList.add("active");
            const targetId = tab.getAttribute("data-tab");
            document.getElementById(targetId).classList.add("active");
        });
    });
}

function updateThresholdDisplay(val) {
    document.getElementById("threshold-val").innerText = parseFloat(val).toFixed(2);
}

function loadPreset(key) {
    const preset = PRESETS[key];
    if (!preset) return;
    document.getElementById("ticket_text").value = preset.text;
    document.getElementById("customer_id").value = preset.cust;
    document.getElementById("threshold_slider").value = preset.threshold;
    updateThresholdDisplay(preset.threshold);
}

// Submit Ticket to API
async function submitTicket(event) {
    event.preventDefault();

    const ticket_text = document.getElementById("ticket_text").value.trim();
    const customer_id = document.getElementById("customer_id").value.trim();
    const confidence_threshold = parseFloat(document.getElementById("threshold_slider").value);

    if (!ticket_text) return;

    // Reset UI
    setRunStatus("running", "Executing Graph...");
    resetNodesVisual();
    document.getElementById("results-dashboard").style.display = "block";

    // Animate Node 1: Classify
    setNodeActive("node-classify", "Classifying category & priority...");

    try {
        const response = await fetch("/api/triage", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ticket_text, customer_id, confidence_threshold })
        });

        const data = await response.json();
        if (!data.success) {
            alert("Error: " + (data.detail || "Failed to process ticket"));
            setRunStatus("ready", "Ready");
            return;
        }

        renderResults(data);

    } catch (err) {
        console.error("API error:", err);
        alert("Server error connecting to triage backend.");
        setRunStatus("ready", "Ready");
    }
}

function resetNodesVisual() {
    ["node-classify", "node-retrieve", "node-decide", "node-outcome"].forEach(id => {
        const el = document.getElementById(id);
        el.className = "flow-node";
    });
    document.getElementById("sub-classify").innerText = "Pending...";
    document.getElementById("sub-retrieve").innerText = "Pending...";
    document.getElementById("sub-decide").innerText = "Pending...";
    document.getElementById("sub-outcome").innerText = "Pending...";
}

function setNodeActive(nodeId, subText) {
    const el = document.getElementById(nodeId);
    el.classList.add("active");
    const sub = el.querySelector(".node-sub");
    if (sub && subText) sub.innerText = subText;
}

function setNodeCompleted(nodeId, subText) {
    const el = document.getElementById(nodeId);
    el.classList.remove("active");
    el.classList.add("completed");
    const sub = el.querySelector(".node-sub");
    if (sub && subText) sub.innerText = subText;
}

function renderResults(data) {
    // 1. Update Classify Node visual
    setNodeCompleted("node-classify", `${data.category.toUpperCase()} | ${data.priority.toUpperCase()}`);

    // 2. Update Retrieve Node visual
    setNodeCompleted("node-retrieve", `Confidence: ${(data.confidence_score).toFixed(2)}`);

    // 3. Update Decide Node visual
    const isResolved = data.status === "resolved";
    setNodeCompleted("node-decide", isResolved ? "Auto-Draft Approved" : "Human Escalation");

    // 4. Update Outcome Node visual
    const outcomeNode = document.getElementById("node-outcome");
    outcomeNode.className = `flow-node ${isResolved ? 'completed' : 'escalated'}`;
    document.getElementById("title-outcome").innerText = isResolved ? "4a. Draft Reply" : "4b. Escalate";
    document.getElementById("sub-outcome").innerText = isResolved ? "Resolved" : "Human Handoff";

    // Set overall status indicator
    setRunStatus(isResolved ? "resolved" : "escalated", isResolved ? "Resolved (Auto)" : "Escalated (Human)");

    // Render Stats
    document.getElementById("res-category").innerText = data.category.toUpperCase();
    document.getElementById("res-priority").innerText = data.priority.toUpperCase();
    document.getElementById("res-confidence").innerText = `${(data.confidence_score).toFixed(4)} (Min: ${data.confidence_threshold.toFixed(2)})`;
    
    const statusVal = document.getElementById("res-status");
    statusVal.innerText = isResolved ? "AUTO-DRAFTED" : "ESCALATED";
    statusVal.style.color = isResolved ? "var(--accent-emerald)" : "var(--accent-rose)";

    // Render Response Box
    const boxTitle = document.getElementById("response-box-title");
    const boxBody = document.getElementById("response-body");

    if (isResolved) {
        boxTitle.innerHTML = `<i class="fa-solid fa-robot" style="color: var(--accent-emerald);"></i> Grounded Auto-Draft Reply`;
        boxBody.innerText = data.draft_reply || "No reply generated.";
    } else {
        boxTitle.innerHTML = `<i class="fa-solid fa-triangle-exclamation" style="color: var(--accent-rose);"></i> Tier-2 Human Escalation Handoff Ticket`;
        boxBody.innerText = data.draft_reply || data.escalation_reason || "Escalated to human support.";
    }

    // Render RAG Docs
    const ragList = document.getElementById("rag-docs-list");
    ragList.innerHTML = "";
    if (data.retrieved_docs && data.retrieved_docs.length > 0) {
        data.retrieved_docs.forEach(doc => {
            const card = document.createElement("div");
            card.className = "rag-doc-card";
            card.innerHTML = `
                <div class="rag-doc-header">
                    <span class="rag-doc-title">[${doc.id}] ${doc.title}</span>
                    <span class="similarity-badge">Score: ${(doc.similarity_score || 0).toFixed(4)}</span>
                </div>
                <div class="rag-doc-snippet">${doc.content}</div>
            `;
            ragList.appendChild(card);
        });
    } else {
        ragList.innerHTML = `<div class="rag-doc-card">No grounding documents retrieved.</div>`;
    }

    // Render Execution Trace
    const traceList = document.getElementById("trace-list");
    traceList.innerHTML = "";
    (data.execution_trace || []).forEach(step => {
        const li = document.createElement("li");
        li.className = "trace-item";
        li.innerHTML = `
            <span class="trace-time">[${step.timestamp}]</span>
            <span class="trace-node">${step.node}</span>
            <span>${step.details}</span>
        `;
        traceList.appendChild(li);
    });
}

function setRunStatus(type, label) {
    const el = document.getElementById("run-status");
    el.className = `status-indicator ${type}`;
    el.innerText = label;
}

// Load FAQ Database
async function loadFaqs() {
    try {
        const res = await fetch("/api/faqs");
        const data = await res.json();
        allFaqs = data.docs || [];
        document.getElementById("faq-count").innerText = allFaqs.length;
        renderFaqsGrid(allFaqs);
    } catch (err) {
        console.error("Failed to load FAQs:", err);
    }
}

function renderFaqsGrid(faqs) {
    const grid = document.getElementById("kb-grid");
    grid.innerHTML = "";
    faqs.forEach(doc => {
        const card = document.createElement("div");
        card.className = "faq-card";
        const catClass = `cat-${doc.category.toLowerCase()}`;
        const tagsHtml = (doc.tags || []).map(t => `<span class="tag">#${t}</span>`).join(" ");

        card.innerHTML = `
            <div class="faq-card-header">
                <div class="faq-card-title">${doc.title}</div>
                <span class="cat-pill ${catClass}">${doc.category}</span>
            </div>
            <div class="faq-card-body">${doc.content}</div>
            <div class="faq-tags">${tagsHtml}</div>
        `;
        grid.appendChild(card);
    });
}

function filterFaqs() {
    const query = document.getElementById("kb-search").value.toLowerCase();
    const cat = document.getElementById("kb-category-filter").value;

    const filtered = allFaqs.filter(doc => {
        const matchCat = (cat === "all") || (doc.category.toLowerCase() === cat);
        const matchText = doc.title.toLowerCase().includes(query) ||
                          doc.content.toLowerCase().includes(query) ||
                          (doc.tags && doc.tags.some(t => t.toLowerCase().includes(query)));
        return matchCat && matchText;
    });

    renderFaqsGrid(filtered);
}

// Modal Handlers
function openAddFaqModal() {
    document.getElementById("add-faq-modal").classList.add("open");
}

function closeAddFaqModal() {
    document.getElementById("add-faq-modal").classList.remove("open");
}

async function saveCustomFaq(event) {
    event.preventDefault();

    const title = document.getElementById("new-faq-title").value.trim();
    const category = document.getElementById("new-faq-category").value;
    const content = document.getElementById("new-faq-content").value.trim();
    const tagsStr = document.getElementById("new-faq-tags").value.trim();

    const tags = tagsStr ? tagsStr.split(",").map(t => t.trim()) : [];

    try {
        const res = await fetch("/api/faqs", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title, category, content, tags })
        });
        const data = await res.json();
        if (data.success) {
            closeAddFaqModal();
            loadFaqs();
        }
    } catch (err) {
        alert("Failed to save FAQ.");
    }
}
