# MedSignal — Master Project Prompt
### Clinical Trial Intelligence Agent | AI Engineering Portfolio Project
**Version:** 1.0 | **Owner:** Harshit Jain | **Target:** Mid-level AI Engineer roles (2026)

---

## HOW TO USE THIS DOCUMENT

This file is the single source of truth for the entire project. At the start of every Claude Code session:

```
"Read medsignal_master_prompt.md and PROGRESS.md, then help me work on Phase [X]: [phase name]."
```

At the end of every session, ask Claude Code:
```
"Update PROGRESS.md with what we completed, what decisions were made, and what's next."
```

The repo is your memory across sessions. This file + PROGRESS.md + the code itself = full continuity.

---

## 1. PROJECT OVERVIEW

### What it is
MedSignal is a multi-agent AI system that lets biomedical researchers and analysts query across clinical trial data using natural language. A user types: *"Which Phase 2 oncology trials from 2023–2025 showed overall survival benefit in NSCLC with less than 20% grade 3 toxicity?"* and gets a grounded, cited answer synthesized from real ClinicalTrials.gov records and PubMed abstracts — with a structured confidence score and source lineage.

### Why it exists (the real-world problem)
Clinical trial results are published across dense PDFs, structured tabular outcome data, and PubMed abstracts. Researchers manually read papers, extract tables by hand, and grep through supplementary materials. There is no unified natural-language interface over all three. MedSignal solves this.

### Why this project, why now
- Directly extends prior biomedical imaging research at Indiana University — authentic, not fabricated
- Covers every major gap in a mid-level AI engineer portfolio: multi-agent orchestration, MCP integration, hybrid RAG, structured eval harness, LLMOps observability
- Uses 6 distinct HuggingFace task types in one coherent system
- MCP integration directly addresses the fastest-growing requirement in 2026 AI engineer JDs

### Portfolio narrative
**Project 1 (existing):** RAG + LoRA fine-tuning + single-modality text search → proves retrieval fundamentals and fine-tuning
**Project 2 (this):** Multi-agent orchestration + MCP + multimodal HF tasks + eval harness + LLMOps → proves production system design

Together: "I understand retrieval deeply, can adapt models, and can build observable production-grade agentic systems."

---

## 2. TECHNICAL ARCHITECTURE

### High-Level System Diagram

```
User Query
    │
    ▼
FastAPI Async Backend
    │
    ▼
LangGraph Orchestrator (StateGraph)
    │
    ├──► Router Agent          (Zero-Shot Classification via HF)
    │         │
    │    [routes by intent: statistical / safety / trial design / general]
    │
    ├──► Retrieval Agent       (Hybrid BM25 + Semantic Search → pgvector)
    │         │
    │    MCP Client ──► ClinicalTrials MCP Server ──► ClinicalTrials.gov REST API
    │    MCP Client ──► PubMed MCP Server          ──► NCBI Entrez API
    │
    ├──► Table Agent           (TAPAS via HuggingFace for structured outcome data)
    │
    └──► Synthesis Agent       (Llama 3.1 70B via Groq — grounded answer + citations)
              │
              ▼
         Inline RAGAS Eval (faithfulness + answer relevance scored per response)
              │
              ▼
    LangSmith trace logged + MLflow metrics written
              │
              ▼
    Response returned to user with citations, confidence score, source lineage
```

### MCP Architecture (Key 2026 Differentiator)

Agents do NOT call external APIs directly. They call MCP servers that wrap those APIs. This is the production pattern.

```
LangGraph Agent
    │
    └──► MCP Client (Python: mcp library)
              │
              ├──► ClinicalTrials MCP Server (local FastMCP server)
              │         Tools exposed:
              │         - search_trials(query, phase, condition, date_range) → list[Trial]
              │         - get_trial_results(nct_id) → TrialResults
              │         - get_trial_outcomes(nct_id) → OutcomeTable
              │
              └──► PubMed MCP Server (local FastMCP server)
                        Tools exposed:
                        - search_abstracts(query, date_range, max_results) → list[Abstract]
                        - get_full_abstract(pmid) → Abstract
```

This means agents discover tools via MCP protocol, not hardcoded function calls. Swapping the data source requires only changing the MCP server — the agent graph is unchanged.

---

## 3. TECH STACK (exact versions — do not deviate)

| Layer | Tool | Version | Notes |
|---|---|---|---|
| Language | Python | 3.11 | Use pyenv if needed |
| Agent orchestration | LangGraph | 0.2.x | Core framework |
| LLM | Llama 3.1 70B | via Groq API | Free tier, fast |
| Embeddings | BGE-M3 (HF) | BAAI/bge-m3 | Better than MiniLM for biomedical |
| Vector store | pgvector | 0.7.x | On PostgreSQL 16 |
| BM25 | rank_bm25 | 0.2.2 | Hybrid retrieval |
| Table QA | TAPAS | google/tapas-base-finetuned-wtq | HuggingFace |
| Zero-Shot Classification | facebook/bart-large-mnli | via HF pipeline | Query routing |
| NER (Biomedical) | d4data/biomedical-ner-all | HF pipeline | Drug/condition extraction |
| MCP servers | FastMCP | 0.4.x | Python MCP server framework |
| Eval | RAGAS | 0.1.x | faithfulness, answer_relevance, context_precision |
| Tracing | LangSmith | latest | Set LANGCHAIN_TRACING_V2=true |
| Experiment tracking | MLflow | 2.x | Local tracking server |
| API layer | FastAPI | 0.111.x | Async, with background tasks |
| Database | PostgreSQL 16 | + pgvector ext | Via Docker |
| Frontend | Streamlit | 1.35.x | Keep minimal — not the point |
| Container | Docker Compose | — | Postgres + pgvector in container |
| Linting | ruff | — | Not optional |
| Testing | pytest | — | At least eval harness tests |

### API Keys needed (get these before starting)
- `GROQ_API_KEY` — free at console.groq.com
- `LANGCHAIN_API_KEY` — free at smith.langchain.com (for LangSmith tracing)
- No key needed for ClinicalTrials.gov or PubMed (public APIs)

---

## 4. REPOSITORY STRUCTURE

```
medsignal/
├── medsignal_master_prompt.md    ← this file (committed to repo)
├── PROGRESS.md                   ← updated at end of every Claude Code session
├── README.md                     ← public-facing, written last
├── .env.example                  ← template for all env vars, never commit .env
├── docker-compose.yml            ← PostgreSQL + pgvector
├── pyproject.toml                ← dependencies, ruff config
│
├── ingestion/
│   ├── __init__.py
│   ├── clinical_trials.py        ← ClinicalTrials.gov REST API client
│   ├── pubmed.py                 ← NCBI Entrez API client
│   ├── chunker.py                ← text chunking strategy (documented decisions)
│   ├── embedder.py               ← BGE-M3 embedding pipeline
│   └── pipeline.py               ← orchestrates full ingestion run
│
├── mcp_servers/
│   ├── __init__.py
│   ├── clinical_trials_server.py ← FastMCP server wrapping ClinicalTrials.gov
│   └── pubmed_server.py          ← FastMCP server wrapping PubMed
│
├── agents/
│   ├── __init__.py
│   ├── state.py                  ← LangGraph AgentState TypedDict definition
│   ├── router.py                 ← Router agent (Zero-Shot Classification)
│   ├── retrieval.py              ← Retrieval agent (hybrid search via MCP)
│   ├── table_qa.py               ← Table agent (TAPAS)
│   ├── synthesis.py              ← Synthesis agent (Groq LLM + citation builder)
│   └── graph.py                  ← StateGraph definition: nodes + edges + entry point
│
├── retrieval/
│   ├── __init__.py
│   ├── vector_store.py           ← pgvector CRUD operations
│   ├── bm25_index.py             ← BM25 index build + query
│   └── hybrid.py                 ← RRF fusion of semantic + BM25 results
│
├── eval/
│   ├── __init__.py
│   ├── golden_set.json           ← 200-question evaluation dataset (hand-curated)
│   ├── harness.py                ← runs RAGAS metrics over golden set
│   ├── inline_eval.py            ← per-response faithfulness + relevance scoring
│   └── results/                  ← logged eval run outputs (gitignored large files)
│
├── api/
│   ├── __init__.py
│   ├── main.py                   ← FastAPI app entry point
│   ├── routes.py                 ← /query, /eval, /health endpoints
│   └── models.py                 ← Pydantic request/response schemas
│
├── ui/
│   └── app.py                    ← Streamlit frontend
│
├── monitoring/
│   ├── mlflow_logger.py          ← MLflow run logging helper
│   └── metrics.py                ← retrieval precision@k, latency, hallucination rate
│
└── tests/
    ├── test_ingestion.py
    ├── test_agents.py
    ├── test_retrieval.py
    └── test_eval_harness.py
```

---

## 5. LANGGRAPH STATE MACHINE (Full Spec)

### AgentState (state.py)

```python
from typing import TypedDict, Optional, Literal
from langgraph.graph import MessagesState

class AgentState(TypedDict):
    # Input
    query: str
    query_intent: Optional[Literal["statistical", "safety", "trial_design", "general"]]
    
    # Retrieval
    retrieved_chunks: list[dict]       # {text, source, nct_id_or_pmid, score}
    retrieved_tables: list[dict]       # {table_data, source_nct_id, outcome_type}
    
    # Processing
    ner_entities: dict                 # {drugs: [], conditions: [], endpoints: []}
    bm25_results: list[dict]
    semantic_results: list[dict]
    
    # Output
    synthesized_answer: Optional[str]
    citations: list[dict]              # {source_id, excerpt, relevance_score}
    confidence_score: Optional[float]
    
    # Eval (written inline, every response)
    faithfulness_score: Optional[float]
    answer_relevance_score: Optional[float]
    
    # Meta
    trace_id: str
    latency_ms: Optional[float]
    error: Optional[str]
```

### Graph Definition (graph.py)

```
Nodes:
  - router_node       → calls Router agent → sets query_intent
  - ner_node          → runs BioNER → sets ner_entities (used to improve retrieval)
  - retrieval_node    → calls Retrieval agent via MCP → sets retrieved_chunks
  - table_node        → calls Table agent (TAPAS) → sets retrieved_tables
  - synthesis_node    → calls Synthesis agent → sets synthesized_answer + citations
  - eval_node         → runs inline RAGAS → sets faithfulness + relevance scores
  - error_node        → handles failures gracefully, returns partial answer if possible

Edges:
  START → router_node → ner_node → retrieval_node
  retrieval_node → table_node [conditional: if intent is "statistical" or "safety"]
  retrieval_node → synthesis_node [if intent is "general" or "trial_design"]
  table_node → synthesis_node
  synthesis_node → eval_node → END

Conditional logic:
  After retrieval_node:
    if query_intent in ["statistical", "safety"]: → table_node
    else: → synthesis_node
  
  After synthesis_node:
    if synthesized_answer is None or confidence < 0.3: → error_node
    else: → eval_node
```

### Key design decisions to preserve across sessions:
1. **NER always runs** — it enriches retrieval regardless of intent, using extracted drug/condition names as additional search terms
2. **Table agent is conditional, not always-on** — TAPAS is slow; only invoke it when the query is clearly quantitative
3. **Eval node is non-blocking** — if inline eval fails, the answer still returns; eval failure is logged but never surfaces to user
4. **MCP calls are inside retrieval_node only** — agents never call external APIs directly, ever

---

## 6. MCP SERVER DESIGN (Full Spec)

### Why MCP matters for this project
JDs in 2026 explicitly require MCP experience. This implementation demonstrates: building MCP servers (not just consuming them), wrapping real external APIs as MCP tools, and connecting LangGraph agents to MCP clients. Almost no portfolio project does this yet.

### ClinicalTrials MCP Server (clinical_trials_server.py)

```python
# Run with: fastmcp run mcp_servers/clinical_trials_server.py

@mcp.tool()
def search_trials(
    query: str,
    phase: Optional[str] = None,          # "PHASE2", "PHASE3", etc.
    condition: Optional[str] = None,       # "NSCLC", "breast cancer", etc.
    date_range_start: Optional[str] = None, # "2023-01-01"
    date_range_end: Optional[str] = None,
    max_results: int = 20
) -> list[dict]:
    """Search ClinicalTrials.gov for trials matching query and filters."""
    # Calls: https://clinicaltrials.gov/api/v2/studies
    ...

@mcp.tool()
def get_trial_outcomes(nct_id: str) -> dict:
    """Get structured outcome measures table for a specific trial."""
    # Calls: https://clinicaltrials.gov/api/v2/studies/{nct_id}
    # Returns: primary_outcomes, secondary_outcomes as structured dict
    ...
```

### PubMed MCP Server (pubmed_server.py)

```python
@mcp.tool()
def search_abstracts(
    query: str,
    date_range_start: Optional[str] = None,
    max_results: int = 20
) -> list[dict]:
    """Search PubMed abstracts via NCBI Entrez API."""
    # Uses: Bio.Entrez (biopython) — free, no auth needed for low volume
    ...

@mcp.tool()
def get_full_abstract(pmid: str) -> dict:
    """Fetch full abstract + metadata for a PubMed ID."""
    ...
```

### How agents call MCP (in retrieval.py)

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def call_mcp_tool(server_script: str, tool_name: str, args: dict) -> dict:
    server_params = StdioServerParameters(command="python", args=[server_script])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, args)
            return result
```

---

## 7. EVAL HARNESS DESIGN (Full Spec)

This is the most important section. The eval harness is what separates this from a demo.

### Golden Set (golden_set.json) — 200 questions across 4 categories

Structure of each entry:
```json
{
  "id": "stat_001",
  "category": "statistical",
  "query": "Which Phase 3 trials in NSCLC from 2022-2024 reported median OS above 18 months?",
  "ground_truth_answer": "...",
  "relevant_nct_ids": ["NCT04...", "NCT05..."],
  "requires_table": true
}
```

Categories (50 questions each):
- `statistical`: OS, PFS, response rates, hazard ratios — requires TAPAS table agent
- `safety`: Adverse events, grade 3+ toxicity rates — also requires table agent
- `trial_design`: Phase, enrollment criteria, arms, endpoints — retrieval only
- `general`: Background, mechanism, comparison questions — synthesis only

**Build the first 50 manually before running any eval.** Do not use LLM to generate ground truth — that creates circular evaluation. Use real ClinicalTrials.gov records.

### RAGAS Metrics (harness.py)

Run against the full golden set after each major change:

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision

# Per-run output written to eval/results/{timestamp}_eval_run.json
# MLflow logs: faithfulness_mean, answer_relevance_mean, context_precision_mean
# Also log: % questions routed correctly by Router agent
```

### Inline Eval (inline_eval.py)

Runs on every single query in production (not just eval runs):

```python
# Lightweight version — only faithfulness + answer_relevance
# Result written to AgentState, logged to LangSmith trace
# If faithfulness < 0.5: flag response with low-confidence warning to user
```

### Baseline comparisons to track and report on resume:
1. **Naive RAG** (semantic-only, no hybrid, no table agent, no MCP)
2. **Hybrid RAG** (semantic + BM25, no table agent)
3. **Full MedSignal** (hybrid + table agent + MCP + all agents)

Track faithfulness, answer_relevance, and context_precision across all three. Report the delta. This is your resume metric.

---

## 8. PHASE-BY-PHASE IMPLEMENTATION PLAN

Each phase has explicit **done criteria**. Do not start the next phase until all criteria are met.

---

### PHASE 1: Infrastructure + Data Ingestion
**Goal:** Working database with indexed trial data. No agents yet.
**Timeline:** ~2 weeks

**Steps:**
1. Set up repo structure exactly as specified in Section 4
2. Docker Compose with PostgreSQL 16 + pgvector extension running locally
3. `ingestion/clinical_trials.py` — fetch 10,000 trials from ClinicalTrials.gov v2 API (oncology focus: `cond=cancer&aggFilters=phase:2,3`)
4. `ingestion/pubmed.py` — fetch abstracts for the trial NCT IDs found above via Entrez
5. `ingestion/chunker.py` — chunk trial descriptions + abstracts. Document decisions: chunk size (512 tokens), overlap (64 tokens), why. These decisions must be written as comments in the file.
6. `ingestion/embedder.py` — embed chunks with BGE-M3, store in pgvector
7. `ingestion/pipeline.py` — single script that runs the full ingestion end to end

**Done criteria:**
- [ ] `docker-compose up` brings up Postgres with pgvector extension enabled
- [ ] `python ingestion/pipeline.py` runs without errors and populates the DB
- [ ] Can run `SELECT COUNT(*) FROM chunks` and see > 50,000 rows
- [ ] Can run a test semantic search query and get sensible results back
- [ ] `PROGRESS.md` updated with record counts and any API quirks discovered

---

### PHASE 2: MCP Servers
**Goal:** Two working MCP servers that agents can call. Test them standalone before wiring to agents.
**Timeline:** ~1 week

**Steps:**
1. Install FastMCP: `pip install fastmcp`
2. Build `mcp_servers/clinical_trials_server.py` with the three tools specified in Section 6
3. Build `mcp_servers/pubmed_server.py` with the two tools specified in Section 6
4. Test each server standalone with `fastmcp dev mcp_servers/clinical_trials_server.py`
5. Write `tests/test_mcp_servers.py` — at least 3 tests per server

**Done criteria:**
- [ ] Both MCP servers start without errors
- [ ] `search_trials("NSCLC phase 3 immunotherapy", phase="PHASE3")` returns real results
- [ ] `search_abstracts("pembrolizumab NSCLC overall survival")` returns real PubMed results
- [ ] All tests pass
- [ ] `PROGRESS.md` updated

---

### PHASE 3: Retrieval Layer
**Goal:** Hybrid retrieval working. No agents yet — test retrieval in isolation.
**Timeline:** ~1 week

**Steps:**
1. `retrieval/vector_store.py` — pgvector semantic search with cosine similarity
2. `retrieval/bm25_index.py` — build BM25 index over chunk text, persist to disk
3. `retrieval/hybrid.py` — Reciprocal Rank Fusion (RRF) combining semantic + BM25 scores
4. Write precision@5 and precision@10 test against 20 hand-labeled queries

**Done criteria:**
- [ ] Hybrid search outperforms semantic-only on the 20 test queries (measure and record)
- [ ] Retrieval latency p95 under 2 seconds
- [ ] `PROGRESS.md` updated with precision numbers (this becomes a resume metric)

---

### PHASE 4: LangGraph Agent Graph
**Goal:** Full 4-agent graph working end to end. This is the core of the project.
**Timeline:** ~2–3 weeks

**Steps:**
1. `agents/state.py` — implement AgentState exactly as specified in Section 5
2. `agents/router.py` — Zero-Shot Classification with `facebook/bart-large-mnli` to classify intent into 4 categories
3. `agents/retrieval.py` — calls hybrid retrieval + calls MCP servers via the pattern in Section 6
4. `agents/table_qa.py` — loads TAPAS, takes retrieved_tables from state, runs table QA
5. `agents/synthesis.py` — calls Groq (Llama 3.1 70B), assembles answer with citations
6. `agents/graph.py` — StateGraph with all nodes, edges, and conditional routing as specified in Section 5
7. Wire LangSmith tracing: set `LANGCHAIN_TRACING_V2=true`, verify traces appear in LangSmith dashboard

**Done criteria:**
- [ ] Full query runs end to end: router → NER → retrieval → [table] → synthesis → eval
- [ ] LangSmith trace visible for every run
- [ ] Router correctly classifies at least 80% of 20 test queries (manual check)
- [ ] Table agent only fires on statistical/safety queries (verify from traces)
- [ ] `PROGRESS.md` updated

---

### PHASE 5: Inline Eval + RAGAS Harness
**Goal:** Measurable, logged quality metrics. The most important phase for resume credibility.
**Timeline:** ~1 week

**Steps:**
1. `eval/inline_eval.py` — per-response faithfulness + answer_relevance, writes to AgentState
2. Build first 50 golden set questions manually (`eval/golden_set.json`, statistical category)
3. `eval/harness.py` — runs full RAGAS eval over golden set, logs to MLflow
4. Run baseline 1 (naive RAG), record scores
5. Run baseline 2 (hybrid RAG), record scores
6. Run full MedSignal, record scores
7. `monitoring/mlflow_logger.py` — helper to log all three runs as MLflow experiments

**Done criteria:**
- [ ] RAGAS faithfulness, answer_relevance, context_precision scores computed for all 3 baselines
- [ ] Full MedSignal outperforms naive RAG on at least 2 of 3 metrics (if not, debug before proceeding)
- [ ] MLflow UI shows all 3 experiment runs with metrics
- [ ] `PROGRESS.md` updated with exact numbers — these are your resume metrics

---

### PHASE 6: FastAPI Backend + Streamlit UI
**Goal:** Deployable system with an API and a minimal UI.
**Timeline:** ~1 week

**Steps:**
1. `api/models.py` — Pydantic schemas: QueryRequest, QueryResponse (with citations, confidence, eval scores)
2. `api/routes.py` — `POST /query` (async, runs agent graph), `GET /health`, `GET /eval/latest`
3. `api/main.py` — FastAPI app, CORS, startup/shutdown lifespan
4. `ui/app.py` — Streamlit: query input, response display with citations, confidence score, eval scores shown
5. End-to-end test: query from Streamlit → FastAPI → LangGraph → response

**Done criteria:**
- [ ] `uvicorn api.main:app` starts without errors
- [ ] POST /query returns a response with citations and confidence score in under 10 seconds
- [ ] Streamlit UI renders cleanly with no errors
- [ ] `PROGRESS.md` updated

---

### PHASE 7: Polish + README + Resume Bullets
**Goal:** Portfolio-ready. Public-facing.
**Timeline:** ~3–4 days

**Steps:**
1. Write `README.md` — architecture diagram (ASCII is fine), what it does, how to run it, what the eval numbers are
2. Fill in resume bullets template (Section 9 below) with real numbers from PROGRESS.md
3. Push to GitHub with clean commit history (squash if needed)
4. Record a 2–3 minute Loom walkthrough: show a real query, show the LangSmith trace, show the MLflow eval dashboard
5. Add Loom link to README

**Done criteria:**
- [ ] README is clear enough that someone unfamiliar can understand the system in 5 minutes
- [ ] All real eval numbers are in the README
- [ ] Loom recorded and linked

---

## 9. RESUME BULLETS TEMPLATE

Fill in [X] values from PROGRESS.md after Phase 5 completes.

```
MedSignal — Clinical Trial Intelligence Agent                    [Month Year]
github.com/harshitjain0302/medsignal

• Architected a 4-agent LangGraph system (Router, Retrieval, Table QA, Synthesis) over 10K+ 
  ClinicalTrials.gov records and PubMed abstracts, with MCP servers wrapping both external APIs

• Built hybrid retrieval (BGE-M3 semantic + BM25 with RRF fusion) achieving [X]% context 
  precision@5 vs [Y]% for semantic-only baseline, measured over 50 hand-labeled queries

• Implemented a RAGAS evaluation harness over a 200-question golden set; full pipeline achieved 
  faithfulness of [X] vs [Y] for naive RAG baseline across statistical and safety query categories

• Integrated inline per-response eval scoring with LangSmith tracing and MLflow experiment 
  tracking across 3 retrieval configurations, enabling measurable quality monitoring in production

• Routed structured outcome queries to a TAPAS table QA model (google/tapas-base-finetuned-wtq) 
  and biomedical NER (d4data/biomedical-ner-all) for query enrichment — 6 distinct HuggingFace 
  task types across one production system

• Deployed async FastAPI backend with p95 query latency under [X]ms; Streamlit UI with 
  citation display, source lineage, and confidence scoring per response
```

---

## 10. SESSION HANDOFF PROTOCOL

Do not create a git worktree. Work directly in the current directory.
Always use `python -m` to run scripts, never `python path/to/script.py`.
The conda environment is `medsignal`. All commands assume it is already activated.

### Starting a new Claude Code session
Paste this at the start:
```
Read medsignal_master_prompt.md and PROGRESS.md carefully.

Current phase: [X] — [phase name]
What we're doing today: [1-2 sentences on today's goal]

The project structure and all design decisions are in medsignal_master_prompt.md.
Do not deviate from the tech stack, repo structure, AgentState schema, or graph design 
without flagging it to me first.
```

### Ending a Claude Code session
Ask Claude Code:
```
Update PROGRESS.md with:
1. What was completed in this session (specific files written/modified)
2. Done criteria checked off for the current phase
3. Any design decisions made that aren't in master_prompt.md
4. Exact numbers recorded (counts, latencies, eval scores)
5. What to do at the start of the next session
```

### When something doesn't work
Do not skip ahead. Debug the current phase completely before moving on. If a dependency is broken or an API behaves unexpectedly, document it in PROGRESS.md under "Known issues / deviations."

---

## 11. COMMON PITFALLS (pre-empted)

| Risk | Mitigation |
|---|---|
| ClinicalTrials.gov API rate limits | Add `time.sleep(0.5)` between requests; cache raw responses to disk before inserting to DB |
| TAPAS slow on CPU | Only invoke on statistical/safety queries (conditional edge handles this); run on Colab if local is too slow |
| BGE-M3 memory usage | Load once at startup, do not reload per query; use sentence-transformers batch encoding |
| Groq rate limits on free tier | Add retry with exponential backoff; cache responses for identical queries during testing |
| RAGAS requires an LLM judge | Use Groq (Llama 3.1 70B) as the judge model for RAGAS — configure via `ragas.llms` |
| pgvector index build time | Build HNSW index after all chunks loaded, not during insertion |
| MCP server subprocess overhead | Start MCP servers once at FastAPI startup, reuse connections; don't spawn per-request |

---

*This document is version controlled. Any architectural change must be reflected here before implementation.*
