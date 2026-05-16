"""
Hybrid retrieval: Reciprocal Rank Fusion (RRF) of semantic + BM25 results.

RRF formula: score(d) = Σ 1 / (k + rank(d))
  k=60 is standard (Cormack et al. 2009) — dampens impact of very top ranks,
  making fusion robust to one retriever dominating.

Why RRF over score normalization:
  - BM25 scores are unbounded and query-length dependent
  - Cosine similarity scores cluster in a narrow range (0.45–0.65 for biomedical)
  - RRF only uses rank position — no score normalization needed
  - Empirically matches or beats learned fusion at this scale
"""

from retrieval.vector_store import semantic_search
from retrieval.bm25_index import bm25_search

RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    id_key: str = "chunk_id",
    k: int = RRF_K,
) -> list[dict]:
    """
    Fuse multiple ranked lists using RRF.

    Args:
        ranked_lists: list of result lists, each sorted by score DESC
        id_key: field to use as unique document identifier
        k: RRF constant (default 60)

    Returns:
        Fused list sorted by RRF score DESC, each item has rrf_score added.
    """
    rrf_scores: dict = {}
    doc_store: dict = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = doc[id_key]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            doc_store[doc_id] = doc  # last writer wins for metadata

    fused = []
    for doc_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        doc = dict(doc_store[doc_id])
        doc["rrf_score"] = rrf_score
        fused.append(doc)

    return fused


def hybrid_search(
    query: str,
    query_embedding: list[float],
    top_k: int = 10,
    semantic_top_k: int = 20,
    bm25_top_k: int = 20,
) -> list[dict]:
    """
    Hybrid search: semantic + BM25 fused with RRF.

    Fetches semantic_top_k and bm25_top_k candidates from each retriever,
    fuses with RRF, returns top_k final results.

    Over-fetching (20 candidates for 10 final) is intentional — RRF reranking
    benefits from seeing more candidates, especially when the two retrievers
    disagree on top results.

    Args:
        query: raw query string (for BM25 tokenization)
        query_embedding: embedded query vector (for semantic search)
        top_k: final number of results to return
        semantic_top_k: candidates from semantic search
        bm25_top_k: candidates from BM25

    Returns:
        List of result dicts with rrf_score, sorted DESC.
        Each dict: {chunk_id, source_type, source_id, text, score, rrf_score}
    """
    semantic_results = semantic_search(query_embedding, top_k=semantic_top_k)
    bm25_results = bm25_search(query, top_k=bm25_top_k)

    # Normalize chunk_id field: semantic returns int from DB, BM25 returns int
    # Both use chunk_id as the fusion key — verify consistency
    fused = reciprocal_rank_fusion(
        [semantic_results, bm25_results],
        id_key="chunk_id",
    )

    return fused[:top_k]
