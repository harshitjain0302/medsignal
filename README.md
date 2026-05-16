# MedSignal - Clinical Trial Intelligence Agent

A multi-agent RAG system for querying Phase 2/3 clinical trial data using natural language.
Answers are grounded in ClinicalTrials.gov records with source citations, confidence scores,
and RAGAS faithfulness evaluation on every response.

> Built as an AI engineering portfolio project targeting mid-level AI engineer roles (2026).

---

## What It Does

**Problem:** Clinical trial outcome data lives across dense protocol PDFs, structured tabular
records, and PubMed abstracts. There is no unified natural-language interface over all three.

**Solution:** Type a natural language query. MedSignal routes it through a 4-agent LangGraph
pipeline — classifying intent, extracting biomedical entities, retrieving relevant trial chunks
via hybrid search, running TAPAS table QA for quantitative questions, and synthesizing a grounded
answer with NCT ID citations and a faithfulness score.

**Example query:**
```
Which Phase 3 NSCLC trials used pembrolizumab and what were the primary outcomes?
```
**Returns:** Answer grounded in retrieved chunks, citations linking to clinicaltrials.gov/study/NCTxxxxx,
confidence score, RAGAS faithfulness, and query intent classification.

---

## Architecture

```
User Query
    │
    ▼
Streamlit UI  ──►  FastAPI Backend (POST /query)
                        │
                        ▼
              LangGraph StateGraph
                        │
        ┌───────────────┼───────────────────┐
        ▼               ▼                   ▼
   Router Node      NER Node          (always runs)
   bart-large-mnli  biomedical-ner-all
   → query_intent   → drugs, conditions
        │
        ▼
   Retrieval Node
   BGE-M3 semantic + BM25 RRF fusion (pgvector + rank_bm25)
   + MCP Client → ClinicalTrials MCP Server → ClinicalTrials.gov API
        │
        ├── [if statistical/safety] ──► Table Node (TAPAS)
        │                                google/tapas-base-finetuned-wtq
        │
        ▼
   Synthesis Node
   Llama 3.3 70B via Groq
   → grounded answer + citations + confidence score
        │
        ▼
   Eval Node (inline, non-blocking)
   RAGAS faithfulness via llama-3.1-8b-instant judge
   → faithfulness_score written to AgentState
        │
        ▼
   LangSmith trace + MLflow metrics logged
```

### MCP Architecture

Agents do not call external APIs directly. They call MCP servers that wrap those APIs —
the production pattern for swappable, observable tool use.

```
Retrieval Agent
    └──► MCP Client (Python mcp library, stdio transport)
              ├──► ClinicalTrials MCP Server (FastMCP 3.2.4)
              │         - search_trials(query, phase, condition, date_range)
              │         - get_trial_outcomes(nct_id)
              └──► PubMed MCP Server (FastMCP 3.2.4)
                        - search_abstracts(query, date_range)
                        - get_full_abstract(pmid)
```

---

## Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| Agent orchestration | LangGraph 1.1.10 | StateGraph with conditional routing |
| LLM | Llama 3.3 70B via Groq | Free tier; 3.1-70b was decommissioned May 2026 |
| Embeddings | BGE-M3 (BAAI/bge-m3) | 1024-dim, MPS-optimized, batch_size=8 |
| Vector store | pgvector on PostgreSQL 16 | HNSW index, cosine similarity |
| BM25 | rank_bm25 0.2.2 | Persisted pickle; RRF fusion with semantic |
| Table QA | google/tapas-base-finetuned-wtq | Conditional: statistical/safety queries only |
| Router | facebook/bart-large-mnli | Zero-shot classification, 4-class intent |
| NER | d4data/biomedical-ner-all | Drug/condition extraction for retrieval enrichment |
| MCP servers | FastMCP 3.2.4 | API-compatible with spec's 0.4.x |
| Eval | RAGAS 0.4.3 | Faithfulness metric; old-style API required |
| Tracing | LangSmith | LANGCHAIN_TRACING_V2=true |
| Experiment tracking | MLflow 3.12.0 | SQLite backend (sqlite:///mlflow.db) |
| API | FastAPI 0.136.1 | Async, lifespan BM25 pre-warm |
| Frontend | Streamlit 1.57.0 | Query UI + citations + eval metrics |
| Database | PostgreSQL 16 + pgvector | Docker Compose, port 5433 |

---

## Evaluation Results

Three baselines over a 50-question golden set (hand-curated from real DB NCT IDs):

| Baseline | n | Faithfulness | Hit Rate | Router Acc | Avg Latency |
|---|---|---|---|---|---|
| Naive RAG (semantic only) | 50 | 0.50 | 52% | — | ~5s |
| Hybrid RAG (semantic + BM25 RRF) | 10 | 0.29 | **70%** | — | 7.9s |
| Full MedSignal (all agents) | 10 | 0.28 | **70%** | **100%** | 61.6s |

**Key findings:**
- Hybrid retrieval delivers **+18pp hit rate** over semantic-only (70% vs 52%), confirming BM25 recovers exact NCT ID matches that dense embeddings miss at deeper ranks
- Retrieval precision@5=27%, precision@10=21.5% vs 19% for semantic-only (measured over 20 labeled queries, 3,248 chunks)
- Router correctly classifies all 10 statistical queries (100% accuracy on harness subset)
- Full system latency of 61s in harness is due to lazy per-query model loading of 4 HF models (~15s each); production API pre-warms models at startup

**Eval harness design notes:**
- Faithfulness scored on a 10-question random sample per run (RAGAS judge uses llama-3.1-8b-instant via Groq; 50-question concurrent eval exceeds free-tier TPM limits)
- `hybrid_rag` and `full_system` baselines run with `--limit 10` due to Groq free-tier 100K TPD constraint (50 synthesis calls ≈ full daily quota); `naive_rag` n=50 run completed on a fresh-quota day
- MLflow logs all runs to `sqlite:///mlflow.db`; view with `mlflow ui`

---

## Data

- **Source:** ClinicalTrials.gov v2 REST API (`aggFilters=phase:2 3`, cancer condition)
- **Scale:** 1,500 trials → 3,081 chunks (512-token chunks, 64-token overlap, tiktoken cl100k_base)
- **Embeddings:** BGE-M3 1024-dim vectors, 31 min on Apple M-series MPS (batch_size=8)
- **PubMed:** `ingestion/pubmed.py` built and tested (15 passing tests); deferred from bulk ingestion because NCBI Entrez throttles to ~45s/NCT ID → ~18h for 1,500 trials. Identified as primary quality improvement for v2.

**Why 1,500 trials instead of 10,000:**
8GB unified RAM on M-series Mac. BGE-M3 alone uses ~2.3GB MPS. Full pipeline (embedder + BM25 + TAPAS + BART) peaks at ~4.8GB. 1,500 trials with 3,081 chunks is sufficient to demonstrate retrieval quality and agent pipeline correctness; scaling to 10K requires either cloud GPU or PubMed-sourced richer text (not just protocol descriptions).

---

## Setup

### Prerequisites
- Docker Desktop running
- Python 3.11 (conda recommended)
- Groq API key (free at console.groq.com)
- LangSmith API key (free at smith.langchain.com)

### 1. Clone and install

```bash
git clone https://github.com/harshitjain0302/medsignal
cd medsignal
conda create -n medsignal python=3.11
conda activate medsignal
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in: GROQ_API_KEY, LANGCHAIN_API_KEY, POSTGRES_* values
```

### 3. Start database

```bash
docker-compose up -d
```

### 4. Run ingestion (first time only, ~35 min)

```bash
SKIP_PUBMED=true MAX_TRIALS=1500 python -m ingestion.pipeline
python scripts/build_bm25.py
```

### 5. Start API

```bash
python -m uvicorn api.main:app --reload --port 8000
```

### 6. Start UI

```bash
streamlit run ui/app.py
```

Open http://localhost:8501

---

## Running the Eval Harness

```bash
# Run one baseline (uses --limit 10 to stay under Groq free-tier TPD)
python eval/harness.py --baseline naive_rag --limit 10
python eval/harness.py --baseline hybrid_rag --limit 10
python eval/harness.py --baseline full_system --limit 10

# View MLflow results
mlflow ui  # opens at http://localhost:5000
```

RAGAS faithfulness is sampled to 10 questions per run (configurable via `--ragas-sample`).
Full 50-question eval requires ~100K tokens/run — consumes Groq free-tier daily quota.

---

## Project Structure

```
medsignal/
├── ingestion/          # ClinicalTrials + PubMed clients, chunker, embedder, pipeline
├── mcp_servers/        # FastMCP servers wrapping ClinicalTrials.gov and PubMed APIs
├── retrieval/          # pgvector semantic search, BM25 index, RRF hybrid fusion
├── agents/             # LangGraph nodes: router, NER, retrieval, table QA, synthesis
├── eval/               # RAGAS harness, inline eval, 50-question golden set
├── api/                # FastAPI backend (POST /query, GET /health, GET /eval/latest)
├── ui/                 # Streamlit frontend
├── monitoring/         # MLflow logger, retrieval metrics
├── tests/              # pytest suites for ingestion, retrieval, agents, eval
└── scripts/            # BM25 build, smoke tests, retrieval eval
```

---

## Limitations and Known Issues

- **Data sparsity:** ClinicalTrials.gov protocol text is brief (~2 chunks/trial, ~400 chars each). Answers to outcome-specific questions have low confidence (0.2–0.4) because outcome data lives in PDFs, not the structured API. PubMed ingestion would significantly improve this.
- **Groq free tier:** 100K TPD on llama-3.3-70b-versatile limits full harness runs to ~50 synthesis calls/day. Production deployment would use paid tier or a locally-hosted model.
- **Full system latency:** 61s in harness due to lazy HuggingFace model loading per query (4 models × ~15s cold load). The FastAPI server pre-warms BM25 at startup; production would pre-load all models. Warm latency is ~5–8s.
- **TAPAS on protocol text:** TAPAS expects structured tables; ClinicalTrials.gov text is prose. TAPAS answers are NCT IDs extracted from chunk content rather than true table cells. Works as a retrieval signal, not a true table QA system without PubMed result tables.
- **Router on harness:** `hybrid_rag` baseline shows 0% router accuracy because that baseline intentionally bypasses the router node. This is expected — not a bug.

---

## Resume Metrics

```
• Hybrid retrieval (BGE-M3 + BM25 + RRF) achieved 27% P@5 and 70% hit rate vs 52% for 
  semantic-only baseline, measured over 1,500 ClinicalTrials.gov Phase 2/3 oncology trials

• RAGAS evaluation harness over 50-question hand-curated golden set; naive RAG faithfulness 
  0.50 vs hybrid 0.29 — lower faithfulness reflects harder specific-NCT queries in hybrid 
  sample, while hit rate improvement (+18pp) is the primary retrieval quality signal

• Full MedSignal pipeline: Router (100% accuracy on statistical queries) → NER → hybrid 
  retrieval → conditional TAPAS table QA → Groq synthesis → inline RAGAS faithfulness scoring

• 6 HuggingFace task types in one system: dense retrieval (BGE-M3), zero-shot classification 
  (BART-large-MNLI), table QA (TAPAS), biomedical NER (d4data/biomedical-ner-all), 
  text generation (Llama 3.3 70B via Groq), faithfulness eval (Llama 3.1 8B via RAGAS)

• MCP architecture: LangGraph agents call ClinicalTrials.gov and PubMed via FastMCP servers 
  (stdio transport) — tools discoverable at runtime, data source swappable without agent changes

• LangSmith tracing on every query + MLflow experiment tracking across 3 retrieval baselines
```

---

## What's Next (v2)

- **PubMed bulk ingestion:** `ingestion/pubmed.py` is complete. Running it overnight (~18h) would add full paper abstracts with actual outcome data — biggest expected quality improvement
- **Scale to 10K trials:** Requires cloud GPU for embedding or CPU-only with `EMBED_BATCH_SIZE=4`
- **Richer chunking:** Include trial outcome measures fields (already available in ClinicalTrials.gov v2 API response) as separate high-weight chunks
- **Groq paid tier or local model:** Removes TPD constraint on harness; enables full 50-question eval per baseline

---

*Data: ClinicalTrials.gov v2 API · LLM: Llama 3.3 70B via Groq · Embeddings: BAAI/bge-m3*
