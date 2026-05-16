"""
pgvector semantic search using cosine similarity.

Queries the `chunks` table using the <=> operator (cosine distance).
Embeddings are L2-normalized at insert time (embedder.py), so cosine
similarity = 1 - cosine_distance = dot product. We sort by distance ASC.
"""

import os
from typing import Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()


def _get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        dbname=os.getenv("POSTGRES_DB", "medsignal"),
        user=os.getenv("POSTGRES_USER", "medsignal"),
        password=os.getenv("POSTGRES_PASSWORD", "medsignal"),
    )


def semantic_search(
    query_embedding: list[float],
    top_k: int = 10,
    source_type: Optional[str] = None,
) -> list[dict]:
    """
    Search chunks by cosine similarity to query_embedding.

    Args:
        query_embedding: 1024-dim float list from embed_query()
        top_k: number of results to return
        source_type: optional filter — 'trial' or 'abstract'

    Returns:
        List of dicts: {chunk_id, source_type, source_id, text, score, metadata}
        Sorted by score DESC (higher = more similar).
    """
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if source_type:
                cur.execute(
                    """
                    SELECT
                        id AS chunk_id,
                        source_type,
                        source_id,
                        text,
                        1 - (embedding <=> %s::vector) AS score,
                        metadata
                    FROM chunks
                    WHERE source_type = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, source_type, query_embedding, top_k),
                )
            else:
                cur.execute(
                    """
                    SELECT
                        id AS chunk_id,
                        source_type,
                        source_id,
                        text,
                        1 - (embedding <=> %s::vector) AS score,
                        metadata
                    FROM chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, query_embedding, top_k),
                )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [dict(r) for r in rows]
