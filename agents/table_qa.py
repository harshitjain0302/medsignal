"""
Table QA agent — extracts quantitative answers from outcome tables using TAPAS.

Model: google/tapas-base-finetuned-wtq (~440MB)
CONDITIONAL — only fires on statistical/safety queries (master prompt design decision #2).
Unloaded after use.

Input: retrieved_chunks that contain structured outcome data from trial protocolSection.
TAPAS expects a pandas DataFrame (table) + natural language question.
"""

import gc
import logging

import torch
import pandas as pd
from transformers import pipeline

from agents.state import AgentState

logger = logging.getLogger(__name__)

TAPAS_MODEL = "google/tapas-base-finetuned-wtq"

_tapas_pipeline = None


def _load_tapas():
    global _tapas_pipeline
    if _tapas_pipeline is None:
        logger.info(f"Loading TAPAS: {TAPAS_MODEL}")
        _tapas_pipeline = pipeline(
            "table-question-answering",
            model=TAPAS_MODEL,
            device="cpu",  # TAPAS has MPS compatibility issues; CPU is reliable
        )
    return _tapas_pipeline


def _unload_tapas():
    global _tapas_pipeline
    if _tapas_pipeline is not None:
        del _tapas_pipeline
        _tapas_pipeline = None
        gc.collect()
        logger.info("TAPAS unloaded.")


def _build_outcome_table(chunks: list[dict]) -> pd.DataFrame:
    """
    Convert retrieved chunks into a table TAPAS can query.
    Extracts NCT ID, title snippet, and text as columns.
    """
    rows = []
    for chunk in chunks[:10]:  # TAPAS has token limits — cap at 10 rows
        rows.append({
            "nct_id": chunk.get("source_id", ""),
            "source": chunk.get("source_type", ""),
            "text": chunk.get("text", "")[:200],  # truncate for TAPAS input limit
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def table_node(state: AgentState) -> dict:
    """
    Run TAPAS table QA on retrieved chunks.
    Only called for statistical/safety intents (conditional edge in graph).
    """
    query = state["query"]
    retrieved_chunks = state.get("retrieved_chunks", [])

    if not retrieved_chunks:
        logger.info("Table node: no chunks to query.")
        return {"retrieved_tables": []}

    tapas = _load_tapas()
    table = _build_outcome_table(retrieved_chunks)

    if table.empty:
        _unload_tapas()
        return {"retrieved_tables": []}

    try:
        result = tapas(table=table, query=query)
        answer = result.get("answer", "")
        cells = result.get("cells", [])
        coords = result.get("coordinates", [])

        # Map coords back to source NCT IDs
        nct_ids = []
        for row_idx, _ in coords:
            if row_idx < len(table):
                nct_ids.append(table.iloc[row_idx]["nct_id"])

        retrieved_tables = [
            {
                "table_data": table.to_dict(orient="records"),
                "source_nct_id": nct_ids[0] if nct_ids else "",
                "outcome_type": "extracted",
                "tapas_answer": answer,
                "tapas_cells": cells,
            }
        ]
        logger.info(f"TAPAS answer: {answer!r} from cells: {cells}")

    except Exception as e:
        logger.warning(f"TAPAS failed (non-fatal): {e}")
        retrieved_tables = []

    _unload_tapas()
    return {"retrieved_tables": retrieved_tables}
