"""
LangGraph AgentState — exact spec from master prompt Section 5.
Do not modify field names; agents read/write these keys by name.
"""

from typing import Optional, Literal
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # Input
    query: str
    query_intent: Optional[Literal["statistical", "safety", "trial_design", "general"]]

    # Retrieval
    retrieved_chunks: list[dict]    # {text, source_type, source_id, score}
    retrieved_tables: list[dict]    # {table_data, source_nct_id, outcome_type}

    # Processing
    ner_entities: dict              # {drugs: [], conditions: [], endpoints: []}
    bm25_results: list[dict]
    semantic_results: list[dict]

    # Output
    synthesized_answer: Optional[str]
    citations: list[dict]           # {source_id, excerpt, relevance_score}
    confidence_score: Optional[float]

    # Eval (written inline, every response)
    faithfulness_score: Optional[float]
    answer_relevance_score: Optional[float]

    # Meta
    trace_id: str
    latency_ms: Optional[float]
    error: Optional[str]
