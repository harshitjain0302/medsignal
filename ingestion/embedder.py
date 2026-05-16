"""
BGE-M3 embedding pipeline via sentence-transformers.

BGE-M3 outputs 1024-dim dense vectors. Loaded once at module level
to avoid reload overhead on repeated calls (see pitfalls in master prompt).
Batch encoding used throughout — never encode one chunk at a time.

Device / batch-size notes:
  MPS (Apple Silicon): BGE-M3 at 512-token sequences OOMs at batch_size=64.
    Safe default is 8. Increase via EMBED_BATCH_SIZE env var if you have more VRAM.
  CUDA: 64 is fine on 16GB+ GPU. Tune up for A100/H100.
  CPU: 32 is a reasonable default (no VRAM limit, just RAM + time).
  Override for any device: set EMBED_BATCH_SIZE env var.
"""

import logging
import os
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-m3"
EMBED_DIM = 1024

# Batch size defaults by device — MPS needs small batches for BGE-M3
_BATCH_DEFAULTS = {"mps": 8, "cuda": 64, "cpu": 32}


def _default_batch_size() -> int:
    """Return env-override or device-appropriate default batch size."""
    if env := os.getenv("EMBED_BATCH_SIZE"):
        return int(env)
    if torch.backends.mps.is_available():
        return _BATCH_DEFAULTS["mps"]
    if torch.cuda.is_available():
        return _BATCH_DEFAULTS["cuda"]
    return _BATCH_DEFAULTS["cpu"]


_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading {MODEL_NAME} — first load takes ~30s")
        _model = SentenceTransformer(MODEL_NAME)
        device = _model.device
        logger.info(f"BGE-M3 loaded on device={device}, batch_size={_default_batch_size()}")
    return _model


def embed_texts(texts: list[str], batch_size: Optional[int] = None) -> np.ndarray:
    """
    Embed a list of texts. Returns float32 array of shape (len(texts), 1024).
    BGE-M3 recommend prepending 'Represent this sentence: ' for passage encoding,
    but for retrieval passage encoding no prefix is needed (differs from query encoding).
    batch_size: defaults to device-appropriate value (8 MPS / 64 CUDA / 32 CPU).
    """
    if batch_size is None:
        batch_size = _default_batch_size()
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,   # cosine similarity = dot product after normalization
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.
    BGE-M3 uses 'Represent this sentence for searching relevant passages: ' prefix for queries.
    """
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    return embed_texts([prefixed])[0]
