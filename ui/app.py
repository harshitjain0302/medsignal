"""
MedSignal Streamlit UI.

Usage:
    streamlit run ui/app.py

Connects to FastAPI backend at MEDSIGNAL_API_URL (default: http://localhost:8000).
"""

import os
import time

import requests
import streamlit as st

API_URL = os.getenv("MEDSIGNAL_API_URL", "http://localhost:8000")


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MedSignal",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 MedSignal")
    st.caption("Multi-agent clinical trial RAG system")
    st.divider()

    st.subheader("Settings")
    top_k = st.slider("Chunks to retrieve (top_k)", min_value=3, max_value=20, value=10)

    st.divider()

    # Health check
    st.subheader("System Status")
    if st.button("Check health", use_container_width=True):
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            h = r.json()
            status_icon = "✅" if h["status"] == "ok" else "⚠️"
            st.write(f"{status_icon} **{h['status'].upper()}**")
            st.write(f"DB: {'✅' if h['db_connected'] else '❌'}")
            st.write(f"BM25: {'✅' if h['bm25_loaded'] else '❌'}")
        except Exception as e:
            st.error(f"API unreachable: {e}")

    st.divider()

    # Eval summary
    st.subheader("Eval Metrics")
    if st.button("Load latest eval", use_container_width=True):
        try:
            r = requests.get(f"{API_URL}/eval/latest", timeout=5)
            evals = r.json()
            if evals:
                for e in evals:
                    with st.expander(f"📊 {e['baseline']}", expanded=False):
                        st.metric("Faithfulness", f"{e['faithfulness_mean']:.2f}" if e.get('faithfulness_mean') else "N/A")
                        st.metric("Hit rate", f"{e['retrieval_hit_rate']:.1%}" if e.get('retrieval_hit_rate') else "N/A")
                        st.metric("Router acc", f"{e['router_accuracy']:.1%}" if e.get('router_accuracy') else "N/A")
                        st.metric("Avg latency", f"{e['avg_latency_ms']:.0f}ms" if e.get('avg_latency_ms') else "N/A")
            else:
                st.info("No eval results yet. Run harness first.")
        except Exception as e:
            st.error(f"Could not load eval: {e}")

    st.divider()
    st.caption("Data: ClinicalTrials.gov v2 API · 1,500 trials")
    st.caption("LLM: Llama 3.3 70B via Groq")
    st.caption("Embeddings: BAAI/bge-m3")


# ── Main ────────────────────────────────────────────────────────────────────────

st.title("MedSignal — Clinical Trial Intelligence")
st.caption("Query Phase 2/3 clinical trials using natural language. Answers grounded in ClinicalTrials.gov data.")

# Example queries
with st.expander("💡 Example queries", expanded=False):
    examples = [
        "Which Phase 3 NSCLC trials used pembrolizumab and what were the primary outcomes?",
        "What is the primary endpoint of NCT03737981?",
        "Which Phase 3 leukemia trials measured progression-free survival?",
        "What chemotherapy regimens were tested in Phase 3 lung cancer trials?",
        "Which trials compared ibrutinib and what was the primary outcome?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:30]}", use_container_width=True):
            st.session_state["query_text"] = ex  # set widget key directly
            st.rerun()

st.divider()

# Query input — key "query_text" shared with example buttons above
query = st.text_area(
    "Enter your clinical trial query:",
    height=80,
    placeholder="e.g. Which Phase 3 trials for NSCLC measured progression-free survival?",
    key="query_text",
)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("🔍 Search", type="primary", use_container_width=True)
with col2:
    if st.button("Clear", use_container_width=True):
        st.session_state["query_text"] = ""
        st.rerun()

# ── Query execution ────────────────────────────────────────────────────────────

if submit and query.strip():
    with st.spinner("Running agent pipeline… (router → NER → retrieval → synthesis → eval)"):
        t0 = time.time()
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"query": query.strip(), "top_k": top_k},
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            wall_ms = (time.time() - t0) * 1000

        except requests.exceptions.Timeout:
            st.error("⏱ Request timed out (>120s). Try a simpler query.")
            st.stop()
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Cannot connect to API at {API_URL}. Is the server running?\n\n`uvicorn api.main:app --port 8000`")
            st.stop()
        except Exception as e:
            st.error(f"❌ API error: {e}")
            st.stop()

    # ── Answer ─────────────────────────────────────────────────────────────────

    if data.get("low_confidence_warning"):
        st.warning(data["low_confidence_warning"])

    st.subheader("Answer")
    st.markdown(data["answer"])

    # ── Metrics row ────────────────────────────────────────────────────────────

    st.divider()
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        conf = data.get("confidence_score")
        color = "normal" if conf and conf >= 0.5 else "inverse"
        st.metric("Confidence", f"{conf:.2f}" if conf is not None else "N/A", delta=None)
    with m2:
        faith = data.get("faithfulness_score")
        st.metric("Faithfulness", f"{faith:.2f}" if faith is not None else "—")
    with m3:
        st.metric("Intent", data.get("query_intent") or "—")
    with m4:
        lat = data.get("latency_ms") or wall_ms
        st.metric("Latency", f"{lat/1000:.1f}s")
    with m5:
        st.metric("Citations", len(data.get("citations", [])))

    # ── Citations ──────────────────────────────────────────────────────────────

    citations = data.get("citations", [])
    if citations:
        st.divider()
        st.subheader(f"Citations ({len(citations)})")
        for i, c in enumerate(citations[:5], 1):
            score = c.get("relevance_score", 0)
            score_bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            with st.expander(f"[{i}] {c['source_id']}  ·  score={score:.3f}  {score_bar}", expanded=(i == 1)):
                st.caption(c.get("excerpt", "No excerpt available"))
                st.markdown(
                    f"[View on ClinicalTrials.gov](https://clinicaltrials.gov/study/{c['source_id']})",
                    unsafe_allow_html=False,
                )

    # ── Debug / trace ──────────────────────────────────────────────────────────

    with st.expander("🔍 Debug info", expanded=False):
        st.json({
            "trace_id": data.get("trace_id"),
            "query_intent": data.get("query_intent"),
            "confidence_score": data.get("confidence_score"),
            "faithfulness_score": data.get("faithfulness_score"),
            "latency_ms": data.get("latency_ms"),
            "error": data.get("error"),
        })

elif submit and not query.strip():
    st.warning("Enter a query first.")
