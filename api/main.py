"""
MedSignal FastAPI application entry point.

Usage:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    POST /query        — run agent graph, return grounded answer
    GET  /health       — liveness check
    GET  /eval/latest  — latest eval harness results
    GET  /docs         — Swagger UI (auto-generated)
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan — pre-warm BM25 index on startup ─────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load BM25 index into memory (fast, ~50ms).
    BGE-M3 and HF models load lazily on first request to avoid slow startup.
    """
    logger.info("MedSignal API starting up...")

    try:
        from retrieval.bm25_index import get_index
        idx = get_index()
        app.state.bm25_loaded = True
        logger.info(f"BM25 index pre-loaded: {len(idx.corpus_ids)} docs")
    except Exception as e:
        app.state.bm25_loaded = False
        logger.warning(f"BM25 pre-load failed (queries will load lazily): {e}")

    logger.info("MedSignal API ready.")
    yield

    logger.info("MedSignal API shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MedSignal",
    description=(
        "Multi-agent RAG system for querying ClinicalTrials.gov data. "
        "Combines hybrid retrieval (BGE-M3 + BM25), TAPAS table QA, "
        "Groq LLM synthesis, and RAGAS evaluation."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


# ── Root ───────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "MedSignal",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
