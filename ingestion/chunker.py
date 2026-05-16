"""
Text chunking strategy for trial descriptions and PubMed abstracts.

Decisions (required by master prompt):
  chunk_size = 512 tokens
    Why: BGE-M3 max input is 8192 tokens, but shorter chunks improve retrieval precision.
    512 is the empirical sweet spot for biomedical text — long enough to preserve
    clinical context (outcome descriptions, eligibility criteria), short enough to
    avoid conflating unrelated concepts within a single embedding.

  overlap = 64 tokens
    Why: Sliding window overlap preserves sentence continuity across chunk boundaries.
    64 tokens (~3-4 sentences) is sufficient for most biomedical sentence lengths
    without bloating chunk count.

  Tokenizer: tiktoken cl100k_base (GPT-4 tokenizer)
    Why: BGE-M3 uses a WordPiece tokenizer internally, but tiktoken cl100k_base gives
    consistent, fast token counting without loading the full BGE-M3 model here.
    Actual BGE-M3 token counts will differ slightly, but cl100k_base is a good proxy.
"""

from dataclasses import dataclass
from typing import Optional

import tiktoken

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
ENCODING = "cl100k_base"


@dataclass
class Chunk:
    text: str
    chunk_index: int
    source_type: str          # 'trial' or 'abstract'
    source_id: str            # nct_id or pmid
    token_count: int
    metadata: Optional[dict] = None


class TextChunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.enc = tiktoken.get_encoding(ENCODING)

    def _split_tokens(self, text: str) -> list[list[int]]:
        """Split token list into overlapping windows."""
        tokens = self.enc.encode(text)
        windows = []
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            windows.append(tokens[start:end])
            if end >= len(tokens):
                break
            start += self.chunk_size - self.overlap
        return windows

    def chunk_trial(self, trial: dict) -> list[Chunk]:
        """
        Chunk a parsed trial dict.
        Uses full_text (title + brief summary + detailed description + eligibility).
        Preserves nct_id and phase in chunk metadata for filtering.
        """
        text = trial.get("full_text", "").strip()
        if not text:
            return []

        nct_id = trial["nct_id"]
        windows = self._split_tokens(text)

        return [
            Chunk(
                text=self.enc.decode(w),
                chunk_index=i,
                source_type="trial",
                source_id=nct_id,
                token_count=len(w),
                metadata={
                    "phase": trial.get("phase"),
                    "status": trial.get("status"),
                    "conditions": trial.get("conditions", []),
                },
            )
            for i, w in enumerate(windows)
        ]

    def chunk_abstract(self, abstract: dict) -> list[Chunk]:
        """
        Chunk a parsed PubMed abstract.
        Abstracts are typically 200-400 tokens — usually a single chunk.
        Title prepended to preserve semantic context when the abstract alone is ambiguous.
        """
        title = abstract.get("title", "")
        body = abstract.get("abstract_text", "")
        text = f"{title}\n\n{body}".strip()
        if not text:
            return []

        pmid = abstract["pmid"]
        windows = self._split_tokens(text)

        return [
            Chunk(
                text=self.enc.decode(w),
                chunk_index=i,
                source_type="abstract",
                source_id=pmid,
                token_count=len(w),
                metadata={
                    "nct_id": abstract.get("nct_id"),
                    "journal": abstract.get("journal"),
                    "pub_date": abstract.get("pub_date"),
                },
            )
            for i, w in enumerate(windows)
        ]
