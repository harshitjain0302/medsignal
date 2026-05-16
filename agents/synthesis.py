"""
Synthesis agent — generates grounded answer with citations using Groq LLM.

Model: Llama 3.1 70B via Groq API (free tier, ~800ms latency)
No local model load — pure API call.

Builds a citation list from retrieved chunks and tables.
Computes a heuristic confidence score based on:
  - number of chunks retrieved
  - average retrieval score
  - presence of TAPAS answer (for quantitative queries)
"""

import logging
import os
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import AgentState

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_CONTEXT_CHUNKS = 8
CITATION_EXCERPT_LEN = 200


SYSTEM_PROMPT = """You are a clinical trial research assistant. Answer the user's question
based ONLY on the provided clinical trial context.

Rules:
- Cite specific NCT IDs when referencing trial data (e.g. [NCT04507841])
- If the context doesn't contain enough information, say so explicitly
- For quantitative questions, quote specific numbers from the context
- Be concise and precise — this is for biomedical researchers, not lay audiences
- Never invent statistics or trial outcomes not present in the context"""


def _build_context(chunks: list[dict], tables: list[dict]) -> str:
    """Build context string from retrieved chunks and table QA results."""
    parts = []

    for i, chunk in enumerate(chunks[:MAX_CONTEXT_CHUNKS], 1):
        src = chunk.get("source_id", "unknown")
        text = chunk.get("text", "")[:400]
        parts.append(f"[{i}] Source: {src}\n{text}")

    if tables:
        for t in tables:
            answer = t.get("tapas_answer", "")
            nct = t.get("source_nct_id", "")
            if answer:
                parts.append(f"[TABLE QA] From {nct}: {answer}")

    return "\n\n---\n\n".join(parts)


def _build_citations(chunks: list[dict]) -> list[dict]:
    return [
        {
            "source_id": c.get("source_id", ""),
            "excerpt": c.get("text", "")[:CITATION_EXCERPT_LEN],
            "relevance_score": float(c.get("score", 0.0)),
        }
        for c in chunks[:MAX_CONTEXT_CHUNKS]
    ]


def _compute_confidence(
    chunks: list[dict],
    tables: list[dict],
    answer: Optional[str],
) -> float:
    """
    Heuristic confidence score [0, 1].
      - 0 chunks → 0.0
      - avg score of top chunks (already 0–1 for cosine, scaled for RRF)
      - bonus +0.1 if TAPAS returned an answer (quantitative grounding)
      - penalty if answer contains "I don't know" / "not enough"
    """
    if not chunks or not answer:
        return 0.0

    scores = [min(c.get("score", 0.0), 1.0) for c in chunks[:5]]
    base = sum(scores) / len(scores)

    # RRF scores are small (0.01–0.05) — normalize them up
    if base < 0.1:
        base = min(base * 10, 0.7)

    bonus = 0.1 if any(t.get("tapas_answer") for t in tables) else 0.0
    uncertainty_phrases = ["don't have", "not enough", "cannot determine", "unclear"]
    penalty = 0.15 if any(p in answer.lower() for p in uncertainty_phrases) else 0.0

    return round(min(max(base + bonus - penalty, 0.0), 1.0), 3)


def synthesis_node(state: AgentState) -> dict:
    """Generate grounded answer with citations. Sets synthesized_answer, citations, confidence_score."""
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    tables = state.get("retrieved_tables", [])

    if not chunks:
        return {
            "synthesized_answer": None,
            "citations": [],
            "confidence_score": 0.0,
            "error": "No retrieved chunks — cannot synthesize answer.",
        }

    context = _build_context(chunks, tables)
    user_message = f"Question: {query}\n\nContext:\n{context}"

    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
        max_tokens=1024,
    )

    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ])
        answer = response.content
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return {
            "synthesized_answer": None,
            "citations": _build_citations(chunks),
            "confidence_score": 0.0,
            "error": str(e),
        }

    citations = _build_citations(chunks)
    confidence = _compute_confidence(chunks, tables, answer)

    logger.info(f"Synthesis: confidence={confidence} answer_len={len(answer)}")

    return {
        "synthesized_answer": answer,
        "citations": citations,
        "confidence_score": confidence,
        "error": None,
    }
