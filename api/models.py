"""
Pydantic schemas for MedSignal API.

QueryRequest  → POST /query input
QueryResponse → POST /query output (with citations, confidence, eval scores)
HealthResponse → GET /health
EvalSummary   → GET /eval/latest
"""

from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="Natural language clinical trial query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of chunks to retrieve")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Which Phase 3 NSCLC trials used pembrolizumab and what were the primary outcomes?",
                "top_k": 10,
            }
        }
    }


class Citation(BaseModel):
    source_id: str = Field(..., description="NCT ID or PubMed ID")
    excerpt: str = Field(..., description="Relevant text excerpt (first 200 chars)")
    relevance_score: float = Field(..., description="Retrieval relevance score [0, 1]")


class QueryResponse(BaseModel):
    query: str
    answer: str = Field(..., description="Synthesized answer grounded in retrieved trial data")
    citations: list[Citation] = Field(default_factory=list)
    confidence_score: Optional[float] = Field(None, description="Heuristic confidence [0, 1]")
    faithfulness_score: Optional[float] = Field(None, description="RAGAS faithfulness [0, 1]; None if eval skipped")
    query_intent: Optional[str] = Field(None, description="Router classification: statistical|safety|trial_design|general")
    latency_ms: Optional[float] = Field(None, description="Total pipeline latency in ms")
    trace_id: str = Field(..., description="Unique trace ID for LangSmith")
    low_confidence_warning: Optional[str] = Field(None, description="Warning message if answer quality is low")
    error: Optional[str] = Field(None, description="Error message if pipeline partially failed")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Which Phase 3 NSCLC trials used pembrolizumab?",
                "answer": "NCT03867175 used pembrolizumab combined with SBRT in metastatic lung cancer [NCT03867175].",
                "citations": [
                    {
                        "source_id": "NCT03867175",
                        "excerpt": "Testing of pembrolizumab in combination with stereotactic body radiation...",
                        "relevance_score": 0.82,
                    }
                ],
                "confidence_score": 0.71,
                "faithfulness_score": 0.85,
                "query_intent": "statistical",
                "latency_ms": 4200.0,
                "trace_id": "a1b2c3d4-...",
                "low_confidence_warning": None,
                "error": None,
            }
        }
    }


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' or 'degraded'")
    db_connected: bool
    bm25_loaded: bool
    version: str = "0.1.0"


class EvalSummary(BaseModel):
    baseline: str
    n_questions: int
    faithfulness_mean: Optional[float]
    retrieval_hit_rate: Optional[float]
    router_accuracy: Optional[float]
    avg_latency_ms: Optional[float]
    timestamp: str
    run_id: str
