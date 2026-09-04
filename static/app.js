// Global State
let allFaqs = [];
let currentEscalatedTicket = null;

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
    negation: {
        text: "I do NOT want a refund for my billing, please fix the critical webhook 429 timeout bug!",
        threshold: 0.60,
        cust: "CUST-7744"
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
    loadAnalytics();
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

// Fetch and Update Analytics KPI Ribbon
async function loadAnalytics() {
    try {
        const res = await fetch("/api/analytics");
        if (!res.ok) return;
        const data = await res.json();
        document.getElementById("kpi-total").innerText = data.total_processed || 0;
        document.getElementById("kpi-auto-rate").innerText = `${data.auto_resolution_rate_pct || 0}%`;
        document.getElementById("kpi-esc-rate").innerText = `${data.escalation_rate_pct || 0}%`;
        document.getElementById("kpi-avg-conf").innerText = (data.avg_confidence_score || 0).toFixed(2);
    } catch (e) {
        console.warn("Analytics fetch failed:", e);
    }
}

// Reset Node Visual State
function resetNodesVisual() {
    ["node-classify", "node-retrieve", "node-decide", "node-outcome"].forEach(id => {
        const el = document.getElementById(id);
        el.className = "flow-node";
    });
    document.getElementById("sub-classify").innerText = "Pending...";
    document.getElementById("sub-retrieve").innerText = "Pending...";
    document.getElementById("sub-decide").innerText = "Pending...";
    document.getElementById("sub-outcome").innerText = "Pending...";
    document.getElementById("hitl-panel").style.display = "none";
}

function setNodePulsing(nodeId, subText) {
    const el = document.getElementById(nodeId);
    el.className = "flow-node streaming-active";
    const sub = el.querySelector(".node-sub");
    if (sub && subText) sub.innerText = subText;
}

function setNodeCompleted(nodeId, subText) {
    const el = document.getElementById(nodeId);
    el.className = "flow-node completed";
    const sub = el.querySelector(".node-sub");
    if (sub && subText) sub.innerText = subText;
}

function setRunStatus(status, text) {
    const el = document.getElementById("run-status");
    el.className = `status-indicator ${status}`;
    el.innerText = text;
}

// Real-Time Server-Sent Events (SSE) Streaming Ticket Submission
async function submitTicket(event) {
    event.preventDefault();

    const ticket_text = document.getElementById("ticket_text").value.trim();
    const customer_id = document.getElementById("customer_id").value.trim();
    const confidence_threshold = parseFloat(document.getElementById("threshold_slider").value);

    if (!ticket_text) return;

    // Reset UI for stream
    setRunStatus("running", "Streaming Pipeline Execution...");
    resetNodesVisual();
    document.getElementById("results-dashboard").style.display = "block";
    const submitBtn = document.getElementById("submit-btn");
    submitBtn.disabled = true;

    // Clear trace list
    document.getElementById("trace-list").innerHTML = "";

    try {
        const response = await fetch("/api/triage/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ticket_text, customer_id, confidence_threshold })
        });

        if (!response.ok) {
            throw new Error(`Server returned HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n\n");
            buffer = lines.pop(); // keep partial

            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    try {
                        const eventData = JSON.parse(line.substring(6));
                        handleStreamEvent(eventData);
                    } catch (err) {
                        console.error("Error parsing stream chunk:", err);
                    }
                }
            }
        }

        // Refresh analytics after completion
        loadAnalytics();

    } catch (err) {
        console.error("Streaming error:", err);
        alert("Error executing streaming pipeline: " + err.message);
        setRunStatus("ready", "Error");
    } finally {
        submitBtn.disabled = false;
    }
}

// Handle Real-Time SSE Event
function handleStreamEvent(data) {
    const traceList = document.getElementById("trace-list");

    switch (data.event) {
        case "node_start":
            if (data.node === "classify") {
                setNodePulsing("node-classify", "Classifying intent...");
            } else if (data.node === "retrieve") {
                setNodePulsing("node-retrieve", "Searching vectors...");
            } else if (data.node === "draft_reply" || data.node === "escalate") {
                setNodePulsing("node-outcome", "Drafting output...");
            }
            break;

        case "node_complete":
            if (data.node === "classify") {
                setNodeCompleted("node-classify", `${data.category.toUpperCase()} | ${data.priority.toUpperCase()}`);
                appendTraceItem(data.trace);
            } else if (data.node === "retrieve") {
                setNodeCompleted("node-retrieve", `Confidence: ${data.confidence_score.toFixed(2)} (${data.retrieved_count} docs)`);
                appendTraceItem(data.trace);
            } else if (data.node === "draft_reply") {
                setNodeCompleted("node-outcome", "Auto-Resolved");
                appendTraceItem(data.trace);
            } else if (data.node === "escalate") {
                const el = document.getElementById("node-outcome");
                el.className = "flow-node escalated";
                document.getElementById("title-outcome").innerText = "4b. Escalate";
                document.getElementById("sub-outcome").innerText = "Human Handoff";
                appendTraceItem(data.trace);
            }
            break;

        case "decide":
            const isResolved = data.decision === "draft_reply";
            setNodeCompleted("node-decide", isResolved ? "Approved for Reply" : "Routed to Escalate");
            break;

        case "pipeline_complete":
            renderFinalDashboard(data.final_state);
            break;
    }
}

function appendTraceItem(step) {
    if (!step) return;
    const traceList = document.getElementById("trace-list");
    const li = document.createElement("li");
    li.className = "trace-item";
    const latencyTag = step.latency_ms ? ` <small style="color:var(--accent-cyan)">(${step.latency_ms}ms)</small>` : "";
    li.innerHTML = `
        <span class="trace-time">[${step.timestamp}]</span>
        <span class="trace-node">${step.node}</span>
        <span>${step.details}${latencyTag}</span>
    `;
    traceList.appendChild(li);
}

// Render Final Response & HITL Controls
function renderFinalDashboard(data) {
    const isResolved = data.status === "resolved";
    setRunStatus(isResolved ? "resolved" : "escalated", isResolved ? "Resolved (Auto)" : "Escalated (Human)");

    // Stats
    document.getElementById("res-category").innerText = (data.category || "GENERAL").toUpperCase();
    document.getElementById("res-priority").innerText = (data.priority || "MEDIUM").toUpperCase();
    document.getElementById("res-confidence").innerText = `${(data.confidence_score || 0).toFixed(4)} (Threshold: ${(data.confidence_threshold || 0.60).toFixed(2)})`;
    
    const statusVal = document.getElementById("res-status");
    statusVal.innerText = isResolved ? "AUTO-DRAFTED" : "ESCALATED";
    statusVal.style.color = isResolved ? "var(--accent-emerald)" : "var(--accent-rose)";

    // Response Box
    const boxTitle = document.getElementById("response-box-title");
    const boxBody = document.getElementById("response-body");

    if (isResolved) {
        boxTitle.innerHTML = `<i class="fa-solid fa-robot" style="color: var(--accent-emerald);"></i> Grounded Auto-Draft Reply`;
        boxBody.innerText = data.draft_reply || "No reply generated.";
        document.getElementById("hitl-panel").style.display = "none";
    } else {
        boxTitle.innerHTML = `<i class="fa-solid fa-triangle-exclamation" style="color: var(--accent-amber);"></i> Tier-2 Human Escalation Summary`;
        boxBody.innerText = data.draft_reply || data.escalation_reason || "Escalated to human support.";

        // Activate HITL Panel
        currentEscalatedTicket = data;
        const hitlPanel = document.getElementById("hitl-panel");
        hitlPanel.style.display = "block";
        document.getElementById("hitl-editor").value = `Hello, regarding your inquiry ("${data.ticket_text.substring(0, 60)}..."): Our senior support team has reviewed your request. Here are the customized steps...`;
    }

    // Render RAG Docs
    const ragList = document.getElementById("rag-docs-list");
    ragList.innerHTML = "";
    if (data.retrieved_docs && data.retrieved_docs.length > 0) {
        data.retrieved_docs.forEach(doc => {
            const card = document.createElement("div");
            card.className = "rag-doc-card";
            const boostTag = doc.category_boosted ? `<span style="background: rgba(16,185,129,0.2); color:#34d399; font-size:10px; padding:2px 6px; border-radius:4px; margin-left:6px;"><i class="fa-solid fa-arrow-trend-up"></i> Domain Boosted</span>` : "";
            card.innerHTML = `
                <div class="rag-doc-header">
                    <span class="rag-doc-title">[${doc.id}] ${doc.title} ${boostTag}</span>
                    <span class="similarity-badge">Score: ${(doc.similarity_score || 0).toFixed(4)}</span>
                </div>
                <div class="rag-doc-snippet">${doc.content}</div>
            `;
            ragList.appendChild(card);
        });
    } else {
        ragList.innerHTML = `<div class="rag-doc-card">No grounding documents retrieved.</div>`;
    }
}

// HITL: Approve & Dispatch Manual Response
function approveHitlResponse() {
    const editedReply = document.getElementById("hitl-editor").value.trim();
    if (!editedReply) return;

    document.getElementById("response-body").innerText = editedReply;
    document.getElementById("response-box-title").innerHTML = `<i class="fa-solid fa-check-circle" style="color: var(--accent-emerald);"></i> Human-Approved & Dispatched Resolution`;
    document.getElementById("res-status").innerText = "MANUALLY RESOLVED";
    document.getElementById("res-status").style.color = "var(--accent-emerald)";
    document.getElementById("hitl-panel").style.display = "none";
    alert("Resolution successfully approved and dispatched to customer! Ticket marked as resolved.");
}

// HITL: Convert Escalated Query to Knowledge Base FAQ
function convertQueryToFaq() {
    if (!currentEscalatedTicket) return;
    openAddFaqModal();
    document.getElementById("new-faq-title").value = `Resolution for: ${currentEscalatedTicket.ticket_text.substring(0, 45)}...`;
    document.getElementById("new-faq-category").value = currentEscalatedTicket.category || "general";
    document.getElementById("new-faq-content").value = document.getElementById("hitl-editor").value || "";
    document.getElementById("new-faq-tags").value = `${currentEscalatedTicket.category}, tier2, escalated`;
}

// Knowledge Base Management
async function loadFaqs() {
    try {
        const res = await fetch("/api/faqs");
        const data = await res.json();
        allFaqs = data.docs || [];
        document.getElementById("faq-count").innerText = allFaqs.length;
        renderFaqGrid(allFaqs);
    } catch (err) {
        console.error("Error loading FAQs:", err);
    }
}

function renderFaqGrid(docs) {
    const grid = document.getElementById("kb-grid");
    grid.innerHTML = "";
    docs.forEach(doc => {
        const card = document.createElement("div");
        card.className = "faq-card";
        const tags = (doc.tags || []).map(t => `<span class="tag-chip">${t}</span>`).join(" ");
        card.innerHTML = `
            <div class="faq-card-header">
                <span class="faq-id">${doc.id}</span>
                <span class="faq-cat cat-${doc.category}">${doc.category.toUpperCase()}</span>
            </div>
            <div class="faq-title">${doc.title}</div>
            <div class="faq-content">${doc.content}</div>
            <div class="faq-tags">${tags}</div>
        `;
        grid.appendChild(card);
    });
}

function filterFaqs() {
    const search = document.getElementById("kb-search").value.toLowerCase();
    const cat = document.getElementById("kb-category-filter").value;

    const filtered = allFaqs.filter(doc => {
        const matchCat = (cat === "all" || doc.category === cat);
        const matchSearch = doc.title.toLowerCase().includes(search) || 
                            doc.content.toLowerCase().includes(search) ||
                            (doc.tags && doc.tags.some(t => t.toLowerCase().includes(search)));
        return matchCat && matchSearch;
    });

    renderFaqGrid(filtered);
}

function openAddFaqModal() {
    document.getElementById("add-faq-modal").classList.add("open");
}

function closeAddFaqModal() {
    document.getElementById("add-faq-modal").classList.remove("open");
}

async function saveCustomFaq(e) {
    e.preventDefault();
    const title = document.getElementById("new-faq-title").value.trim();
    const category = document.getElementById("new-faq-category").value;
    const content = document.getElementById("new-faq-content").value.trim();
    const tags = document.getElementById("new-faq-tags").value.split(",").map(t => t.trim()).filter(Boolean);

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
            alert("New FAQ successfully indexed into the Knowledge Base!");
        }
    } catch (err) {
        alert("Failed to save FAQ.");
    }
}
