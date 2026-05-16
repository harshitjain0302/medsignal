"""
LangGraph StateGraph — wires all agent nodes with conditional routing.

Graph topology (from master prompt Section 5):
  START → router_node → ner_node → retrieval_node
  retrieval_node → table_node     [if intent: statistical | safety]
  retrieval_node → synthesis_node [if intent: general | trial_design]
  table_node     → synthesis_node
  synthesis_node → eval_node      [if answer present and confidence >= 0.3]
  synthesis_node → error_node     [if answer None or confidence < 0.3]
  eval_node      → END
  error_node     → END
"""

import logging
import time
import uuid

from langgraph.graph import StateGraph, START, END

from agents.state import AgentState
from agents.router import router_node
from agents.ner import ner_node
from agents.retrieval import retrieval_node
from agents.table_qa import table_node
from agents.synthesis import synthesis_node

logger = logging.getLogger(__name__)


# ── Eval node (inline RAGAS — non-blocking) ──────────────────────────────────

def eval_node(state: AgentState) -> dict:
    """
    Inline faithfulness + answer_relevance scoring via RAGAS.
    Non-blocking: if RAGAS fails, answer still returns (master prompt decision #3).
    Logs scores to state for LangSmith trace.
    """
    try:
        from ragas import evaluate
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.llms import llm_factory
        from openai import OpenAI
        from datasets import Dataset
        import os

        answer = state.get("synthesized_answer", "") or ""
        query = state["query"]
        chunks = state.get("retrieved_chunks", [])
        contexts = [c["text"] for c in chunks[:5]]

        if not answer or not contexts:
            return {"faithfulness_score": None, "answer_relevance_score": None}

        groq_client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        # 8B model for RAGAS judge: 500K TPD vs 100K for 70B on Groq free tier
        judge_llm = llm_factory(
            "llama-3.1-8b-instant",
            provider="openai",
            client=groq_client,
        )
        f_metric = Faithfulness()
        f_metric.llm = judge_llm

        ds = Dataset.from_dict({
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        })

        result = evaluate(
            ds,
            metrics=[f_metric],
            raise_exceptions=False,
        )

        df = result.to_pandas()
        f_score = float(df["faithfulness"].iloc[0]) if "faithfulness" in df.columns else None

        logger.info(f"Eval: faithfulness={f_score}")
        return {"faithfulness_score": f_score, "answer_relevance_score": None}

    except Exception as e:
        logger.warning(f"Inline eval failed (non-fatal): {e}")
        return {"faithfulness_score": None, "answer_relevance_score": None}


# ── Error node ────────────────────────────────────────────────────────────────

def error_node(state: AgentState) -> dict:
    """Handle synthesis failures. Returns partial state with error flag."""
    logger.warning(f"Error node reached. error={state.get('error')} confidence={state.get('confidence_score')}")
    return {
        "synthesized_answer": state.get("synthesized_answer") or "Unable to generate a confident answer from the available trial data.",
        "error": state.get("error") or "Low confidence answer",
    }


# ── Conditional routing functions ─────────────────────────────────────────────

def route_after_retrieval(state: AgentState) -> str:
    """Route to table_node for quantitative queries, synthesis_node otherwise."""
    intent = state.get("query_intent", "general")
    if intent in ("statistical", "safety"):
        return "table_node"
    return "synthesis_node"


def route_after_synthesis(state: AgentState) -> str:
    """Route to error_node if answer missing or low-confidence."""
    answer = state.get("synthesized_answer")
    confidence = state.get("confidence_score", 0.0) or 0.0
    if answer is None or confidence < 0.3:
        return "error_node"
    return "eval_node"


# ── Timing wrapper ────────────────────────────────────────────────────────────

def _timed(node_fn):
    """Wrap a node function to track cumulative latency in state."""
    def wrapper(state: AgentState) -> dict:
        t0 = time.time()
        result = node_fn(state)
        elapsed_ms = (time.time() - t0) * 1000
        prev_latency = state.get("latency_ms") or 0.0
        result["latency_ms"] = prev_latency + elapsed_ms
        return result
    wrapper.__name__ = node_fn.__name__
    return wrapper


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("router_node",    _timed(router_node))
    graph.add_node("ner_node",       _timed(ner_node))
    graph.add_node("retrieval_node", _timed(retrieval_node))
    graph.add_node("table_node",     _timed(table_node))
    graph.add_node("synthesis_node", _timed(synthesis_node))
    graph.add_node("eval_node",      _timed(eval_node))
    graph.add_node("error_node",     error_node)

    graph.add_edge(START, "router_node")
    graph.add_edge("router_node", "ner_node")
    graph.add_edge("ner_node", "retrieval_node")

    graph.add_conditional_edges(
        "retrieval_node",
        route_after_retrieval,
        {"table_node": "table_node", "synthesis_node": "synthesis_node"},
    )

    graph.add_edge("table_node", "synthesis_node")

    graph.add_conditional_edges(
        "synthesis_node",
        route_after_synthesis,
        {"eval_node": "eval_node", "error_node": "error_node"},
    )

    graph.add_edge("eval_node", END)
    graph.add_edge("error_node", END)

    return graph


def run_query(query: str) -> AgentState:
    """
    Run a query through the full agent graph.
    Returns final AgentState with answer, citations, confidence, eval scores.
    """
    graph = build_graph()
    app = graph.compile()

    initial_state: AgentState = {
        "query": query,
        "query_intent": None,
        "retrieved_chunks": [],
        "retrieved_tables": [],
        "ner_entities": {},
        "bm25_results": [],
        "semantic_results": [],
        "synthesized_answer": None,
        "citations": [],
        "confidence_score": None,
        "faithfulness_score": None,
        "answer_relevance_score": None,
        "trace_id": str(uuid.uuid4()),
        "latency_ms": None,
        "error": None,
    }

    final_state = app.invoke(initial_state)
    return final_state
