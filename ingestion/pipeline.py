"""
Full ingestion pipeline: fetch → chunk → embed → store.

Run: python ingestion/pipeline.py

Steps:
  1. Fetch trials from ClinicalTrials.gov (cached to data/raw/)
  2. Insert parsed trials into DB
  3. Fetch PubMed abstracts for each NCT ID
  4. Insert abstracts into DB
  5. Chunk all trial full_text + abstract text
  6. Embed chunks with BGE-M3
  7. Insert chunks + embeddings into pgvector
  8. Build HNSW index after all insertions
"""

import logging
import os
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

from ingestion.clinical_trials import ClinicalTrialsClient
from ingestion.pubmed import PubMedClient
from ingestion.chunker import TextChunker
from ingestion.embedder import embed_texts

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAX_TRIALS = int(os.getenv("MAX_TRIALS", "10000"))
MAX_PUBMED_PER_NCT = int(os.getenv("MAX_PUBMED_PER_NCT", "3"))


def get_db_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "medsignal"),
        user=os.getenv("POSTGRES_USER", "medsignal"),
        password=os.getenv("POSTGRES_PASSWORD", "medsignal"),
    )


def upsert_trials(conn, trials: list[dict]) -> int:
    inserted = 0
    with conn.cursor() as cur:
        for t in trials:
            cur.execute(
                """
                INSERT INTO trials
                    (nct_id, title, phase, conditions, interventions, status,
                     start_date, completion_date, primary_outcomes, secondary_outcomes, raw_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (nct_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    phase = EXCLUDED.phase,
                    status = EXCLUDED.status,
                    raw_json = EXCLUDED.raw_json
                """,
                (
                    t["nct_id"],
                    t["title"],
                    t["phase"],
                    t["conditions"],
                    t["interventions"],
                    t["status"],
                    t.get("start_date"),
                    t.get("completion_date"),
                    psycopg2.extras.Json(t["primary_outcomes"]),
                    psycopg2.extras.Json(t["secondary_outcomes"]),
                    psycopg2.extras.Json(t["raw_json"]),
                ),
            )
            inserted += 1
    conn.commit()
    return inserted


def upsert_abstracts(conn, abstracts: list[dict]) -> int:
    inserted = 0
    with conn.cursor() as cur:
        for a in abstracts:
            cur.execute(
                """
                INSERT INTO abstracts (pmid, nct_id, title, abstract_text, authors, journal, pub_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (pmid) DO NOTHING
                """,
                (
                    a["pmid"],
                    a.get("nct_id"),
                    a["title"],
                    a["abstract_text"],
                    a["authors"],
                    a["journal"],
                    a.get("pub_date"),
                ),
            )
            inserted += 1
    conn.commit()
    return inserted


def insert_chunks_with_embeddings(conn, chunks, embeddings) -> int:
    with conn.cursor() as cur:
        records = [
            (
                c.source_type,
                c.source_id,
                c.chunk_index,
                c.text,
                embeddings[i].tolist(),
                psycopg2.extras.Json(c.metadata or {}),
            )
            for i, c in enumerate(chunks)
        ]
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO chunks (source_type, source_id, chunk_index, text, embedding, metadata)
            VALUES %s
            ON CONFLICT DO NOTHING
            """,
            records,
        )
    conn.commit()
    return len(records)


def build_hnsw_index(conn):
    """Build HNSW index after all chunks loaded — not during insertion (see pitfalls)."""
    logger.info("Building HNSW index on chunks.embedding ...")
    with conn.cursor() as cur:
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_embedding "
            "ON chunks USING hnsw (embedding vector_cosine_ops)"
        )
    conn.commit()
    logger.info("HNSW index built.")


def run():
    conn = get_db_conn()
    chunker = TextChunker()

    # Step 1-2: Trials
    logger.info("=== Step 1: Fetching trials ===")
    ct_client = ClinicalTrialsClient()
    trials = ct_client.fetch_and_parse(max_trials=MAX_TRIALS)
    logger.info(f"Fetched {len(trials)} trials. Inserting ...")
    n_trials = upsert_trials(conn, trials)
    logger.info(f"Inserted/updated {n_trials} trials.")

    # Step 3-4: Abstracts (skippable — NCBI throttles heavily at scale)
    skip_pubmed = os.getenv("SKIP_PUBMED", "false").lower() in ("1", "true", "yes")
    abstracts: list[dict] = []
    if skip_pubmed:
        logger.info("=== Step 2: Skipping PubMed (SKIP_PUBMED=true) ===")
    else:
        logger.info("=== Step 2: Fetching PubMed abstracts ===")
        nct_ids = [t["nct_id"] for t in trials if t["nct_id"]]
        pm_client = PubMedClient()
        abstracts = pm_client.fetch_for_nct_ids(nct_ids, max_per_nct=MAX_PUBMED_PER_NCT)
        logger.info(f"Fetched {len(abstracts)} abstracts. Inserting ...")
        n_abstracts = upsert_abstracts(conn, abstracts)
        logger.info(f"Inserted {n_abstracts} abstracts.")

    # Step 5-7: Chunk + embed + store
    logger.info("=== Step 3: Chunking ===")
    all_chunks = []
    for t in trials:
        all_chunks.extend(chunker.chunk_trial(t))
    for a in abstracts:
        all_chunks.extend(chunker.chunk_abstract(a))
    logger.info(f"Total chunks: {len(all_chunks)}")

    logger.info("=== Step 4: Embedding ===")
    texts = [c.text for c in all_chunks]
    embeddings = embed_texts(texts)
    logger.info(f"Embedded {len(embeddings)} chunks.")

    logger.info("=== Step 5: Storing chunks ===")
    n_chunks = insert_chunks_with_embeddings(conn, all_chunks, embeddings)
    logger.info(f"Stored {n_chunks} chunks.")

    # Step 8: Index
    build_hnsw_index(conn)

    conn.close()
    logger.info("=== Ingestion complete. ===")


if __name__ == "__main__":
    run()
