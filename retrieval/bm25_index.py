"""
BM25 index over chunk text using rank_bm25.

Build once from DB, persist to disk with pickle.
Reload from disk on subsequent runs — fast (~50ms for 3K chunks).

Index stores parallel arrays:
  _corpus_ids:  list of chunk_id (int) — maps rank → DB row
  _source_ids:  list of source_id (str) — for result assembly
  _source_types: list of source_type (str)
  _texts:       list of raw text — for snippet return
  _bm25:        BM25Okapi instance

Tokenization: simple whitespace + lowercase. Biomedical text benefits from
keeping compound terms (e.g. "non-small-cell" stays intact vs splitting on "-").
"""

import os
import pickle
import re
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

INDEX_PATH = Path("data/bm25_index.pkl")


def _get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        dbname=os.getenv("POSTGRES_DB", "medsignal"),
        user=os.getenv("POSTGRES_USER", "medsignal"),
        password=os.getenv("POSTGRES_PASSWORD", "medsignal"),
    )


def _tokenize(text: str) -> list[str]:
    """Lowercase + split on whitespace and punctuation except hyphens."""
    text = text.lower()
    # Split on whitespace and common punctuation, keep hyphens inside words
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)
    return tokens


def build_index(index_path: Path = INDEX_PATH) -> "BM25Index":
    """
    Build BM25 index from all chunks in DB. Persist to disk.
    Run once after ingestion pipeline completes.
    """
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, source_type, source_id, text FROM chunks ORDER BY id"
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    corpus_ids = [r["id"] for r in rows]
    source_types = [r["source_type"] for r in rows]
    source_ids = [r["source_id"] for r in rows]
    texts = [r["text"] for r in rows]

    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)

    index_data = {
        "corpus_ids": corpus_ids,
        "source_types": source_types,
        "source_ids": source_ids,
        "texts": texts,
        "bm25": bm25,
    }

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump(index_data, f)

    print(f"BM25 index built: {len(corpus_ids)} chunks → {index_path}")
    return BM25Index(index_data)


class BM25Index:
    def __init__(self, data: dict):
        self.corpus_ids: list[int] = data["corpus_ids"]
        self.source_types: list[str] = data["source_types"]
        self.source_ids: list[str] = data["source_ids"]
        self.texts: list[str] = data["texts"]
        self._bm25: BM25Okapi = data["bm25"]

    @classmethod
    def load(cls, index_path: Path = INDEX_PATH) -> "BM25Index":
        """Load index from disk. Build first if not found."""
        if not index_path.exists():
            return build_index(index_path)
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        return cls(data)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        BM25 search. Returns top_k results sorted by score DESC.

        Returns:
            List of dicts: {chunk_id, source_type, source_id, text, score}
        """
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Get top_k indices by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] == 0.0:
                break  # no more matches
            results.append({
                "chunk_id": self.corpus_ids[idx],
                "source_type": self.source_types[idx],
                "source_id": self.source_ids[idx],
                "text": self.texts[idx],
                "score": float(scores[idx]),
            })

        return results


# Module-level singleton — load once, reuse across calls
_index: Optional[BM25Index] = None


def get_index() -> BM25Index:
    global _index
    if _index is None:
        _index = BM25Index.load()
    return _index


def bm25_search(query: str, top_k: int = 10) -> list[dict]:
    """Convenience wrapper — loads index lazily."""
    return get_index().search(query, top_k=top_k)
