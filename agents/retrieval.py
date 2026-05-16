"""
Retrieval agent — hybrid search + MCP tool calls.

Design rules (from master prompt):
  - MCP calls happen ONLY in this node — agents never call external APIs directly
  - Hybrid search (pgvector + BM25 + RRF) runs against local DB
  - NER entities from ner_node are appended to query for enriched retrieval
  - MCP calls supplement local DB with live ClinicalTrials.gov results

MCP pattern: each call spawns a subprocess via stdio_client.
For production FastAPI startup, MCP servers are kept alive (see pitfalls).
Here we use per-call connections (acceptable for Phase 4 dev).
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.embedder import embed_query
from retrieval.hybrid import hybrid_search
from agents.state import AgentState

logger = logging.getLogger(__name__)

TOP_K = 10
MCP_MAX_RESULTS = 5


def _enrich_query(query: str, ner_entities: dict) -> str:
    """Append extracted drug/condition names to query for better retrieval."""
    extras = []
    extras.extend(ner_entities.get("drugs", [])[:3])
    extras.extend(ner_entities.get("conditions", [])[:2])
    if extras:
        return f"{query} {' '.join(extras)}"
    return query


async def _call_mcp_search(query: str, intent: str) -> list[dict]:
    """
    Call ClinicalTrials MCP server to supplement local DB results.
    Returns list of trial dicts from live API.
    """
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_script = str(Path(__file__).parent.parent / "mcp_servers" / "clinical_trials_server.py")
        server_params = StdioServerParameters(
            command=sys.executable, args=[server_script]
        )

        phase = "2 3"
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "search_trials",
                    {"query": query, "phase": phase, "max_results": MCP_MAX_RESULTS},
                )
                # MCP returns TextContent — parse the JSON content
                import json
                if result.content:
                    raw = result.content[0].text
                    return json.loads(raw) if isinstance(raw, str) else []
    except Exception as e:
        logger.warning(f"MCP call failed (non-fatal): {e}")
    return []


def retrieval_node(state: AgentState) -> dict:
    """
    Run hybrid search + MCP live search. Merge and deduplicate results.
    """
    query = state["query"]
    ner_entities = state.get("ner_entities") or {}
    enriched_query = _enrich_query(query, ner_entities)

    # Embed query (loads BGE-M3 lazily — router already unloaded)
    q_embedding = embed_query(enriched_query).tolist()

    # Hybrid search against local DB
    hybrid_results = hybrid_search(enriched_query, q_embedding, top_k=TOP_K)
    semantic_results = [r for r in hybrid_results]  # hybrid includes semantic scores
    bm25_from_hybrid = []  # BM25 is fused inside hybrid — expose separately if needed

    # MCP live search (async → run in sync context)
    mcp_trials = asyncio.run(_call_mcp_search(query, state.get("query_intent", "general")))

    # Convert MCP trials to chunk-like dicts for uniform handling
    mcp_chunks = [
        {
            "source_type": "trial",
            "source_id": t.get("nct_id", ""),
            "text": f"{t.get('title', '')} {' '.join(t.get('conditions', []))} {' '.join(t.get('interventions', []))}",
            "score": 0.5,  # neutral score for live results
            "chunk_id": f"mcp_{t.get('nct_id', '')}",
        }
        for t in mcp_trials
    ]

    # Merge: local hybrid results first, then MCP supplements
    seen_ids = {r["source_id"] for r in hybrid_results}
    for chunk in mcp_chunks:
        if chunk["source_id"] not in seen_ids:
            hybrid_results.append(chunk)
            seen_ids.add(chunk["source_id"])

    retrieved_chunks = [
        {
            "text": r["text"],
            "source_type": r.get("source_type", "trial"),
            "source_id": r.get("source_id", ""),
            "score": float(r.get("rrf_score", r.get("score", 0.0))),
        }
        for r in hybrid_results[:TOP_K]
    ]

    logger.info(f"Retrieval: {len(retrieved_chunks)} chunks (local={len(hybrid_results[:TOP_K])}, mcp={len(mcp_chunks)})")

    return {
        "retrieved_chunks": retrieved_chunks,
        "semantic_results": semantic_results,
        "bm25_results": bm25_from_hybrid,
    }
