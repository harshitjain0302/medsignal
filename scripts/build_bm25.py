"""Build BM25 index from DB chunks. Run once after ingestion."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.bm25_index import build_index

build_index()
