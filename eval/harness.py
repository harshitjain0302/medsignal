"""
RAGAS Evaluation Harness — runs 3 baselines over golden_set.json.

Baselines:
  1. naive_rag    — semantic search only (no BM25, no TAPAS)
  2. hybrid_rag   — semantic + BM25 RRF (no TAPAS)
  3. full_system  — full MedSignal pipeline (TAPAS + synthesis + eval)

Usage:
    python eval/harness.py --baseline all          # run all 3
    python eval/harness.py --baseline full_system  # run one
    python eval/harness.py --baseline all --limit 10  # quick smoke run

Outputs:
    eval/results/{timestamp}_{baseline}_eval_run.json
    MLflow experiment: medsignal_eval
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Baseline runners ──────────────────────────────────────────────────────────

def run_naive_rag(query: str) -> dict:
    """Semantic search only — no BM25, no TAPAS, Groq synthesis."""
    from retrieval.vector_store import semantic_search
    from ingestion.embedder import embed_query
    from agents.synthesis import synthesis_node

    qvec = embed_query(query).tolist()
    chunks = semantic_search(qvec, top_k=10)

    state = {
        "query": query,
        "retrieved_chunks": chunks,
        "retrieved_tables": [],
        "ner_entities": {},
        "bm25_results": [],
        "semantic_results": chunks,
        "synthesized_answer": None,
        "citations": [],
        "confidence_score": None,
        "error": None,
    }
    result = synthesis_node(state)
    state.update(result)
    return state


def run_hybrid_rag(query: str) -> dict:
    """Hybrid search (semantic + BM25 RRF) — no TAPAS, Groq synthesis."""
    from retrieval.hybrid import hybrid_search
    from ingestion.embedder import embed_query
    from agents.synthesis import synthesis_node

    qvec = embed_query(query).tolist()
    chunks = hybrid_search(query, qvec, top_k=10)

    state = {
        "query": query,
        "retrieved_chunks": chunks,
        "retrieved_tables": [],
        "ner_entities": {},
        "bm25_results": [],
        "semantic_results": [],
        "synthesized_answer": None,
        "citations": [],
        "confidence_score": None,
        "error": None,
    }
    result = synthesis_node(state)
    state.update(result)
    return state


def run_full_system(query: str) -> dict:
    """Full MedSignal pipeline via run_query."""
    from agents.graph import run_query
    return run_query(query)


BASELINE_FNS = {
    "naive_rag": run_naive_rag,
    "hybrid_rag": run_hybrid_rag,
    "full_system": run_full_system,
}


# ── RAGAS evaluation ──────────────────────────────────────────────────────────

def run_ragas(questions: list[str], answers: list[str], contexts: list[list[str]], sample: int = 10) -> dict:
    """
    Run RAGAS faithfulness on a random sample (default 10) to avoid TPM limits.
    Returns dict of metric_name → float score.
    Non-blocking: returns empty dict on failure.
    """
    import random

    try:
        from ragas import evaluate
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.llms import llm_factory
        from openai import OpenAI
        from datasets import Dataset

        # Subsample to avoid Groq TPM wall — faithfulness still representative
        if len(questions) > sample:
            indices = random.sample(range(len(questions)), sample)
            questions  = [questions[i]  for i in indices]
            answers    = [answers[i]    for i in indices]
            contexts   = [contexts[i]   for i in indices]
            logger.info(f"RAGAS: sampled {sample}/{len(questions) + len(questions) - sample} questions for faithfulness eval")

        # Use 8B model for RAGAS judge — 500K TPD vs 100K for 70B on Groq free tier
        groq_client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        judge_llm = llm_factory(
            "llama-3.1-8b-instant",
            provider="openai",
            client=groq_client,
        )
        f_metric = Faithfulness()
        f_metric.llm = judge_llm

        ds = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        })

        result = evaluate(
            ds,
            metrics=[f_metric],
            raise_exceptions=False,
        )

        scores = {}
        df = result.to_pandas()
        if "faithfulness" in df.columns:
            scores["faithfulness_mean"] = float(df["faithfulness"].mean())

        logger.info(f"RAGAS scores: {scores}")
        return scores

    except Exception as e:
        logger.warning(f"RAGAS eval failed (non-fatal): {e}")
        return {}


# ── Router accuracy ───────────────────────────────────────────────────────────

EXPECTED_INTENTS = {
    "statistical": "statistical",
    "safety": "safety",
    "trial_design": "trial_design",
    "general": "general",
}


def compute_router_accuracy(items: list[dict]) -> float:
    """Fraction of questions where router intent matched expected category."""
    correct = 0
    total = 0
    for item in items:
        expected = EXPECTED_INTENTS.get(item["category"])
        got = item.get("query_intent")
        if expected and got:
            total += 1
            if got == expected:
                correct += 1
    return round(correct / total, 3) if total > 0 else 0.0


# ── Main harness ──────────────────────────────────────────────────────────────

def run_baseline(baseline: str, golden_set: list[dict], limit: int | None = None, ragas_sample: int = 10) -> dict:
    """
    Run one baseline over golden set. Returns results dict.
    """
    run_fn = BASELINE_FNS[baseline]
    items = golden_set[:limit] if limit else golden_set

    questions, answers, contexts, results = [], [], [], []
    start = time.time()

    for i, item in enumerate(items):
        query = item["query"]
        logger.info(f"[{baseline}] {i+1}/{len(items)}: {query[:60]}")

        try:
            state = run_fn(query)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            state = {
                "synthesized_answer": None,
                "retrieved_chunks": [],
                "query_intent": None,
                "error": str(e),
            }

        answer = state.get("synthesized_answer") or ""
        chunks = state.get("retrieved_chunks", [])
        ctx = [c["text"] for c in chunks[:5]] if chunks else [""]

        questions.append(query)
        answers.append(answer)
        contexts.append(ctx)

        results.append({
            "id": item["id"],
            "category": item["category"],
            "query": query,
            "ground_truth_answer": item["ground_truth_answer"],
            "relevant_nct_ids": item["relevant_nct_ids"],
            "generated_answer": answer,
            "query_intent": state.get("query_intent"),
            "confidence_score": state.get("confidence_score"),
            "latency_ms": state.get("latency_ms"),
            "retrieved_nct_ids": list({c.get("source_id","") for c in chunks}),
            "error": state.get("error"),
        })

        # Throttle to avoid Groq rate limits
        time.sleep(1.5)

    total_latency = (time.time() - start) * 1000

    # RAGAS eval (skip if no answers)
    valid_answers = [a for a in answers if a]
    ragas_scores = {}
    if valid_answers:
        ragas_scores = run_ragas(questions, answers, contexts, sample=ragas_sample)

    # Router accuracy
    router_acc = compute_router_accuracy(results)

    # Retrieval hit@k (fraction where ≥1 relevant NCT ID retrieved)
    hit_at_5 = 0
    for item, res in zip(items, results):
        relevant = set(item["relevant_nct_ids"])
        retrieved = set(res["retrieved_nct_ids"])
        if relevant & retrieved:
            hit_at_5 += 1
    retrieval_hit_rate = round(hit_at_5 / len(items), 3) if items else 0.0

    summary = {
        "baseline": baseline,
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "n_questions": len(items),
        "total_latency_ms": round(total_latency),
        "avg_latency_ms": round(total_latency / len(items)) if items else 0,
        "router_accuracy": router_acc,
        "retrieval_hit_rate": retrieval_hit_rate,
        **ragas_scores,
        "results": results,
    }

    return summary


def save_results(summary: dict) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{ts}_{summary['baseline']}_eval_run.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Results saved: {path}")
    return path


def log_to_mlflow(summary: dict):
    """Log metrics to MLflow. Non-blocking."""
    try:
        from monitoring.mlflow_logger import log_eval_run
        log_eval_run(summary)
    except Exception as e:
        logger.warning(f"MLflow logging failed (non-fatal): {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MedSignal RAGAS eval harness")
    parser.add_argument("--baseline", default="full_system",
                        choices=["naive_rag", "hybrid_rag", "full_system", "all"],
                        help="Which baseline to run")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of questions (for smoke testing)")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Skip MLflow logging")
    parser.add_argument("--ragas-sample", type=int, default=10,
                        help="Number of questions to sample for RAGAS faithfulness (default: 10)")
    args = parser.parse_args()

    golden_set = json.loads(GOLDEN_SET_PATH.read_text())
    logger.info(f"Loaded {len(golden_set)} questions from golden set")

    baselines = list(BASELINE_FNS.keys()) if args.baseline == "all" else [args.baseline]

    all_summaries = []
    for baseline in baselines:
        logger.info(f"\n{'='*60}\nRunning baseline: {baseline}\n{'='*60}")
        summary = run_baseline(baseline, golden_set, limit=args.limit, ragas_sample=args.ragas_sample)

        path = save_results(summary)
        if not args.no_mlflow:
            log_to_mlflow(summary)

        print(f"\n{'='*50}")
        print(f"Baseline:          {summary['baseline']}")
        print(f"Questions:         {summary['n_questions']}")
        print(f"Router accuracy:   {summary['router_accuracy']:.1%}")
        print(f"Retrieval hit rate:{summary['retrieval_hit_rate']:.1%}")
        print(f"Faithfulness:      {summary.get('faithfulness_mean', 'N/A')}")
        print(f"Answer relevancy:  {summary.get('answer_relevancy_mean', 'N/A')}")
        print(f"Avg latency:       {summary['avg_latency_ms']}ms")
        print(f"Results:           {path}")

        all_summaries.append(summary)

    if len(all_summaries) > 1:
        print(f"\n{'='*50}\nCOMPARISON TABLE\n{'='*50}")
        print(f"{'Baseline':<15} {'Faithfulness':>14} {'Ans.Relevancy':>14} {'Hit Rate':>10} {'Router Acc':>11}")
        for s in all_summaries:
            print(
                f"{s['baseline']:<15}"
                f" {s.get('faithfulness_mean', 'N/A'):>14}"
                f" {s.get('answer_relevancy_mean', 'N/A'):>14}"
                f" {s['retrieval_hit_rate']:>10.1%}"
                f" {s['router_accuracy']:>11.1%}"
            )


if __name__ == "__main__":
    main()
