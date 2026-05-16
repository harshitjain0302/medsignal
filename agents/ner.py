"""
NER agent — extracts drugs, conditions, endpoints from query.

Model: d4data/biomedical-ner-all (~440MB)
Always runs (enriches retrieval regardless of intent — master prompt design decision #1).
Unloaded after use before BGE-M3 loads in retrieval_node.
"""

import gc
import logging
from collections import defaultdict

import torch
from transformers import pipeline

from agents.state import AgentState

logger = logging.getLogger(__name__)

NER_MODEL = "d4data/biomedical-ner-all"

# Entity type mappings from d4data model labels → our schema
DRUG_LABELS = {"Chemical", "Drug", "Simple_chemical"}
CONDITION_LABELS = {"Disease_or_Syndrome", "Neoplastic_Process", "Disease", "Sign_or_Symptom"}
ENDPOINT_LABELS = {"Clinical_Attribute", "Quantitative_Concept", "Finding"}

_ner_pipeline = None


def _load_ner():
    global _ner_pipeline
    if _ner_pipeline is None:
        logger.info(f"Loading NER model: {NER_MODEL}")
        _ner_pipeline = pipeline(
            "ner",
            model=NER_MODEL,
            aggregation_strategy="simple",
            device="cpu",  # NER is small — CPU avoids MPS setup overhead
        )
    return _ner_pipeline


def _unload_ner():
    global _ner_pipeline
    if _ner_pipeline is not None:
        del _ner_pipeline
        _ner_pipeline = None
        gc.collect()
        logger.info("NER model unloaded.")


def ner_node(state: AgentState) -> dict:
    """
    Extract biomedical entities from query.
    Sets state['ner_entities'] = {drugs, conditions, endpoints}.
    Unloads model after extraction.
    """
    query = state["query"]
    ner = _load_ner()

    try:
        entities = ner(query)
    except Exception as e:
        logger.warning(f"NER failed (non-fatal): {e}")
        _unload_ner()
        return {"ner_entities": {"drugs": [], "conditions": [], "endpoints": []}}

    grouped: dict = defaultdict(list)
    for ent in entities:
        label = ent.get("entity_group", "")
        word = ent.get("word", "").strip()
        if not word:
            continue
        if label in DRUG_LABELS:
            grouped["drugs"].append(word)
        elif label in CONDITION_LABELS:
            grouped["conditions"].append(word)
        elif label in ENDPOINT_LABELS:
            grouped["endpoints"].append(word)

    result = {
        "drugs": list(set(grouped["drugs"])),
        "conditions": list(set(grouped["conditions"])),
        "endpoints": list(set(grouped["endpoints"])),
    }
    logger.info(f"NER: drugs={result['drugs']} conditions={result['conditions']}")

    _unload_ner()
    return {"ner_entities": result}
