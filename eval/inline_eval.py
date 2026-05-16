"""
Inline eval — lightweight per-response faithfulness + answer_relevance.

Runs on every production query (not just eval harness).
Result written to AgentState and logged to LangSmith trace.

Design decisions:
- Uses Groq Llama 3.3-70B as judge (same as eval_node in graph)
- Runs in eval_node of the LangGraph graph (already wired in agents/graph.py)
- This module is a standalone helper; graph.py calls eval_node directly
- If faithfulness < 0.5: state carries low_confidence_warning flag

Usage from graph eval_node (agents/graph.py already uses this):
    from eval.inline_eval import score_response
    scores = score_response(query, answer, contexts)
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def score_response(
    query: str,
    answer: str,
    contexts: list[str],
    *,
    model: str = "llama-3.1-8b-instant",  # 8B for RAGAS judge: 500K TPD vs 100K for 70B
) -> dict:
    """
    Score a single response with RAGAS faithfulness + answer_relevancy.

    Returns:
        {
            "faithfulness": float | None,
            "answer_relevancy": float | None,
            "low_confidence_warning": bool,  # True if faithfulness < 0.5
        }

    Non-blocking: returns Nones on any failure.
    """
    if not answer or not contexts:
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "low_confidence_warning": False,
        }

    try:
        from ragas import evaluate
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.llms import llm_factory
        from openai import OpenAI
        from datasets import Dataset

        groq_client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        judge_llm = llm_factory(
            model,
            provider="openai",
            client=groq_client,
        )
        f_metric = Faithfulness()
        f_metric.llm = judge_llm

        ds = Dataset.from_dict({
            "question": [query],
            "answer": [answer],
            "contexts": [contexts[:5]],  # RAGAS context cap
        })

        result = evaluate(
            ds,
            metrics=[f_metric],
            raise_exceptions=False,
        )

        df = result.to_pandas()
        f_score = float(df["faithfulness"].iloc[0]) if "faithfulness" in df.columns else None
        low_conf = (f_score is not None and f_score < 0.5)

        logger.info(f"Inline eval: faithfulness={f_score} low_conf={low_conf}")
        return {
            "faithfulness": f_score,
            "answer_relevancy": None,  # requires separate embedding model
            "low_confidence_warning": low_conf,
        }

    except Exception as e:
        logger.warning(f"Inline eval failed (non-fatal): {e}")
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "low_confidence_warning": False,
        }


def format_warning_flag(state: dict) -> Optional[str]:
    """
    Return a user-facing warning string if the answer is low confidence.
    Used by API layer to annotate responses.
    """
    f_score = state.get("faithfulness_score")
    if f_score is not None and f_score < 0.5:
        return f"⚠ Low faithfulness score ({f_score:.2f}). Answer may not be fully grounded in retrieved trial data."
    conf = state.get("confidence_score")
    if conf is not None and conf < 0.3:
        return f"⚠ Low confidence ({conf:.2f}). Insufficient trial data found for this query."
    return None
