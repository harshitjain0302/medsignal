"""
Semantic search smoke test — runs 5 oncology queries against pgvector.
Usage: python scripts/smoke_search.py
"""

import os
import sys
from pathlib import Path

# Add project root to path so ingestion package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from dotenv import load_dotenv

load_dotenv()

from ingestion.embedder import embed_query  # noqa: E402

QUERIES = [
    "pembrolizumab NSCLC overall survival Phase 3",
    "BRCA mutation breast cancer PARP inhibitor",
    "grade 3 toxicity adverse events immunotherapy",
    "colorectal cancer FOLFOX chemotherapy response rate",
    "glioblastoma temozolomide progression free survival",
]

TOP_K = 3


def get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        dbname=os.getenv("POSTGRES_DB", "medsignal"),
        user=os.getenv("POSTGRES_USER", "medsignal"),
        password=os.getenv("POSTGRES_PASSWORD", "medsignal"),
    )


def search(cur, embedding: list[float], top_k: int = TOP_K):
    cur.execute(
        """
        SELECT
            c.source_type,
            c.source_id,
            1 - (c.embedding <=> %s::vector) AS cosine_sim,
            LEFT(c.text, 120) AS snippet
        FROM chunks c
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
        """,
        (embedding, embedding, top_k),
    )
    return cur.fetchall()


def main():
    conn = get_conn()
    cur = conn.cursor()

    for query in QUERIES:
        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        print("-" * 70)

        vec = embed_query(query).tolist()
        results = search(cur, vec)

        for rank, (src_type, src_id, sim, snippet) in enumerate(results, 1):
            print(f"  [{rank}] {src_type}:{src_id}  sim={sim:.4f}")
            print(f"      {snippet!r}")

    cur.close()
    conn.close()
    print(f"\n{'='*70}")
    print("Smoke test complete.")


if __name__ == "__main__":
    main()
