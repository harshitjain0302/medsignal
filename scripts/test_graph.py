"""
End-to-end agent graph smoke test.
Usage: python scripts/test_graph.py
"""
import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from agents.graph import run_query

QUERY = "Which Phase 3 NSCLC trials used pembrolizumab and what were the primary outcomes?"

print(f"\nQuery: {QUERY}\n{'='*70}")
state = run_query(QUERY)

print(f"\nIntent:     {state['query_intent']}")
print(f"NER drugs:  {state['ner_entities'].get('drugs', [])}")
print(f"Chunks:     {len(state['retrieved_chunks'])}")
print(f"Confidence: {state['confidence_score']}")
print(f"Faithfulness: {state['faithfulness_score']}")
print(f"Latency:    {state['latency_ms']:.0f}ms")
print(f"\nAnswer:\n{state['synthesized_answer']}")
print(f"\nCitations ({len(state['citations'])}):")
for c in state['citations'][:3]:
    print(f"  {c['source_id']}  score={c['relevance_score']:.3f}  {c['excerpt'][:80]!r}")
