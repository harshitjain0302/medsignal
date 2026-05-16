"""
FastAPI route handlers.

POST /query  — run full agent graph, return answer + citations + scores
GET  /health — liveness + dependency checks
GET  /eval/latest — most recent eval harness result from eval/results/
"""

import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from api.models import Citation, EvalSummary, HealthResponse, QueryRequest, QueryResponse
from eval.inline_eval import format_warning_flag

logger = logging.getLogger(__name__)

router = APIRouter()

RESULTS_DIR = Path(__file__).parent.parent / "eval" / "results"


# ── POST /query ───────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Run query through the full MedSignal agent graph.

    Pipeline: Router → NER → Hybrid Retrieval → (Table QA) → Synthesis → Eval

    Returns grounded answer with citations, confidence, and faithfulness score.
    Typical latency: 5-15s (cold model load), 3-8s (warm).
    """
    try:
        # Run LangGraph pipeline in thread pool (blocking I/O + model inference)
        from agents.graph import run_query
        state = await run_in_threadpool(run_query, request.query)
    except Exception as e:
        logger.error(f"Pipeline error for query={request.query!r}: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    answer = state.get("synthesized_answer") or "Unable to generate a confident answer from the available trial data."
    citations = [
        Citation(
            source_id=c.get("source_id", ""),
            excerpt=c.get("excerpt", c.get("text", ""))[:200],
            relevance_score=float(c.get("relevance_score", c.get("score", 0.0))),
        )
        for c in state.get("citations", [])
    ]

    warning = format_warning_flag(state)

    return QueryResponse(
        query=request.query,
        answer=answer,
        citations=citations,
        confidence_score=state.get("confidence_score"),
        faithfulness_score=state.get("faithfulness_score"),
        query_intent=state.get("query_intent"),
        latency_ms=state.get("latency_ms"),
        trace_id=state.get("trace_id", ""),
        low_confidence_warning=warning,
        error=state.get("error"),
    )


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness check — verifies DB connection and BM25 index."""
    db_ok = False
    bm25_ok = False

    # DB check
    try:
        import psycopg2
        from dotenv import load_dotenv
        load_dotenv()
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", 5433)),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            dbname=os.getenv("POSTGRES_DB"),
            connect_timeout=3,
        )
        conn.close()
        db_ok = True
    except Exception as e:
        logger.warning(f"DB health check failed: {e}")

    # BM25 index check
    try:
        bm25_path = Path(__file__).parent.parent / "data" / "bm25_index.pkl"
        bm25_ok = bm25_path.exists()
    except Exception:
        pass

    status = "ok" if (db_ok and bm25_ok) else "degraded"
    return HealthResponse(status=status, db_connected=db_ok, bm25_loaded=bm25_ok)


# ── GET /eval/latest ──────────────────────────────────────────────────────────

@router.get("/eval/latest", response_model=list[EvalSummary])
async def eval_latest() -> list[EvalSummary]:
    """
    Return summary of the most recent eval run per baseline.
    Reads from eval/results/*.json — one entry per baseline type.
    """
    if not RESULTS_DIR.exists():
        return []

    # Group by baseline, take latest per baseline
    latest: dict[str, dict] = {}
    for path in sorted(RESULTS_DIR.glob("*_eval_run.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
            baseline = data.get("baseline", "unknown")
            if baseline not in latest:
                latest[baseline] = data
        except Exception as e:
            logger.warning(f"Could not parse {path}: {e}")

    summaries = []
    for data in latest.values():
        summaries.append(
            EvalSummary(
                baseline=data.get("baseline", ""),
                n_questions=data.get("n_questions", 0),
                faithfulness_mean=data.get("faithfulness_mean"),
                retrieval_hit_rate=data.get("retrieval_hit_rate"),
                router_accuracy=data.get("router_accuracy"),
                avg_latency_ms=data.get("avg_latency_ms"),
                timestamp=data.get("timestamp", ""),
                run_id=data.get("run_id", ""),
            )
        )

    return summaries
