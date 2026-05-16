"""
Retrieval precision@5 and precision@10 evaluation.

20 hand-labeled queries with relevant NCT IDs.
Tests semantic-only vs hybrid retrieval, records delta.

Usage: python scripts/eval_retrieval.py

A result is "relevant" if its source_id matches any NCT ID in the
ground truth set for that query.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.embedder import embed_query
from retrieval.vector_store import semantic_search
from retrieval.hybrid import hybrid_search

# 20 hand-labeled queries with relevant NCT IDs verified against DB contents.
# Derived by ILIKE text search on chunks table — ground truth reflects what is
# actually in the 1500-trial corpus, not assumed from external knowledge.
LABELED_QUERIES = [
    {
        "query": "pembrolizumab NSCLC overall survival immunotherapy",
        "relevant_ids": {"NCT02434081", "NCT07163507"},
    },
    {
        "query": "PARP inhibitor BRCA breast cancer olaparib",
        "relevant_ids": {"NCT05232006", "NCT05498155", "NCT07005583"},
    },
    {
        "query": "colorectal cancer FOLFOX FOLFIRI chemotherapy",
        "relevant_ids": {"NCT01134666", "NCT02376452", "NCT05954078", "NCT06405139"},
    },
    {
        "query": "PD-1 checkpoint inhibitor melanoma immunotherapy",
        "relevant_ids": {"NCT03025256", "NCT04949113", "NCT07005583", "NCT07156019", "NCT07163507"},
    },
    {
        "query": "bevacizumab VEGF angiogenesis lung cancer",
        "relevant_ids": {"NCT00945139", "NCT01004250", "NCT01445509", "NCT03470350", "NCT06383884"},
    },
    {
        "query": "neoadjuvant chemotherapy breast cancer pathologic complete response",
        "relevant_ids": {"NCT00193115", "NCT00349076", "NCT00775645", "NCT01050322", "NCT01170143"},
    },
    {
        "query": "EGFR mutation targeted therapy non-small cell lung cancer",
        "relevant_ids": {"NCT01579630", "NCT02163733", "NCT02387216", "NCT02407366", "NCT02824458"},
    },
    {
        "query": "small cell lung cancer platinum-based chemotherapy",
        "relevant_ids": {"NCT00034164", "NCT00312975", "NCT00453115", "NCT00801736", "NCT01579630"},
    },
    {
        "query": "cetuximab EGFR metastatic colorectal cancer",
        "relevant_ids": {"NCT00362102", "NCT01134666", "NCT03470350"},
    },
    {
        "query": "trastuzumab HER2 positive breast cancer",
        "relevant_ids": {"NCT01050322", "NCT01170143", "NCT02005484", "NCT02041338", "NCT03273595"},
    },
    {
        "query": "multiple myeloma bortezomib proteasome inhibitor",
        "relevant_ids": {"NCT00511238", "NCT00523848", "NCT01286077", "NCT01556347", "NCT01677858"},
    },
    {
        "query": "durvalumab anti-PD-L1 clinical trial",
        "relevant_ids": {"NCT02434081", "NCT02882308", "NCT02937818", "NCT03258554", "NCT04062708"},
    },
    {
        "query": "relapsed refractory hematologic malignancy salvage therapy",
        "relevant_ids": {"NCT00002501", "NCT00003737", "NCT00014209", "NCT00047281", "NCT00206726"},
    },
    {
        "query": "autoimmune disease exclusion criteria eligibility",
        "relevant_ids": {"NCT00089063", "NCT00310804", "NCT00417430", "NCT00434954", "NCT00578812"},
    },
    {
        "query": "carboplatin paclitaxel combination chemotherapy solid tumor",
        "relevant_ids": {"NCT00193115", "NCT00453115", "NCT01170143"},
    },
    {
        "query": "phase 2 overall response rate ORR primary endpoint",
        "relevant_ids": {"NCT02376452", "NCT03025256", "NCT05353439"},
    },
    {
        "query": "biomarker tumor mutation burden TMB predictive",
        "relevant_ids": {"NCT07163507", "NCT04949113"},
    },
    {
        "query": "lymphoma autologous stem cell transplant ASCT",
        "relevant_ids": {"NCT00206726", "NCT00047281"},
    },
    {
        "query": "prostate cancer androgen deprivation hormone therapy",
        "relevant_ids": {"NCT00310804", "NCT00434954"},
    },
    {
        "query": "pancreatic cancer gemcitabine first-line treatment",
        "relevant_ids": {"NCT00578812", "NCT00003737"},
    },
]


def precision_at_k(results: list[dict], relevant_ids: set, k: int) -> float:
    top_k_ids = {r["source_id"] for r in results[:k]}
    hits = top_k_ids & relevant_ids
    return len(hits) / k


def evaluate(method: str, results_fn) -> dict:
    p5_scores, p10_scores = [], []

    for item in LABELED_QUERIES:
        results = results_fn(item["query"])
        p5 = precision_at_k(results, item["relevant_ids"], k=5)
        p10 = precision_at_k(results, item["relevant_ids"], k=10)
        p5_scores.append(p5)
        p10_scores.append(p10)

    mean_p5 = sum(p5_scores) / len(p5_scores)
    mean_p10 = sum(p10_scores) / len(p10_scores)

    print(f"\n{method}")
    print(f"  precision@5:  {mean_p5:.4f}  ({mean_p5*100:.1f}%)")
    print(f"  precision@10: {mean_p10:.4f}  ({mean_p10*100:.1f}%)")

    return {"method": method, "p5": mean_p5, "p10": mean_p10}


def main():
    print("Loading BGE-M3 and BM25 index...")

    # Pre-embed all queries (load model once)
    embeddings = {item["query"]: embed_query(item["query"]).tolist() for item in LABELED_QUERIES}

    def semantic_fn(query):
        return semantic_search(embeddings[query], top_k=10)

    def hybrid_fn(query):
        return hybrid_search(query, embeddings[query], top_k=10)

    print(f"\nEvaluating {len(LABELED_QUERIES)} queries...")
    print("=" * 50)

    sem_scores = evaluate("Semantic-only (BGE-M3 + pgvector)", semantic_fn)
    hyb_scores = evaluate("Hybrid (BGE-M3 + BM25 + RRF)", hybrid_fn)

    print("\n" + "=" * 50)
    print("DELTA (hybrid - semantic):")
    print(f"  precision@5:  {(hyb_scores['p5'] - sem_scores['p5'])*100:+.1f}pp")
    print(f"  precision@10: {(hyb_scores['p10'] - sem_scores['p10'])*100:+.1f}pp")
    print("\n→ Record these numbers in PROGRESS.md")


if __name__ == "__main__":
    main()
