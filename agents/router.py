"""
Router agent — classifies query intent using Zero-Shot Classification.

Model: facebook/bart-large-mnli (~1.6GB)
Labels: statistical | safety | trial_design | general

Memory strategy (8GB RAM):
  Load model → classify → explicitly unload before retrieval node loads BGE-M3.
  Peak: ~1.6GB (router) → freed → ~2.3GB (BGE-M3). Never both loaded simultaneously.
"""

import gc
import logging

import torch
from transformers import pipeline

from agents.state import AgentState

logger = logging.getLogger(__name__)

ROUTER_MODEL = "facebook/bart-large-mnli"
INTENT_LABELS = ["statistical", "safety", "trial_design", "general"]
HYPOTHESIS_TEMPLATE = "This query is about {} information from clinical trials."

_classifier = None


def _load_classifier():
    global _classifier
    if _classifier is None:
        logger.info(f"Loading router model: {ROUTER_MODEL}")
        _classifier = pipeline(
            "zero-shot-classification",
            model=ROUTER_MODEL,
            device="mps" if torch.backends.mps.is_available() else "cpu",
        )
    return _classifier


def _unload_classifier():
    """Free router model memory before retrieval node loads BGE-M3."""
    global _classifier
    if _classifier is not None:
        del _classifier
        _classifier = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        logger.info("Router model unloaded.")


def router_node(state: AgentState) -> dict:
    """Classify query intent. Unloads model after classification."""
    query = state["query"]
    classifier = _load_classifier()

    result = classifier(
        query,
        candidate_labels=INTENT_LABELS,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        multi_label=False,
    )

    intent = result["labels"][0]
    score = result["scores"][0]
    logger.info(f"Router: intent={intent} score={score:.3f} query={query[:60]!r}")

    _unload_classifier()
    return {"query_intent": intent}
