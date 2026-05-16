CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS trials (
    id SERIAL PRIMARY KEY,
    nct_id TEXT UNIQUE NOT NULL,
    title TEXT,
    phase TEXT,
    conditions TEXT[],
    interventions TEXT[],
    status TEXT,
    start_date DATE,
    completion_date DATE,
    primary_outcomes JSONB,
    secondary_outcomes JSONB,
    raw_json JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS abstracts (
    id SERIAL PRIMARY KEY,
    pmid TEXT UNIQUE NOT NULL,
    nct_id TEXT REFERENCES trials(nct_id) ON DELETE SET NULL,
    title TEXT,
    abstract_text TEXT,
    authors TEXT[],
    journal TEXT,
    pub_date DATE,
    raw_xml TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    source_type TEXT NOT NULL,       -- 'trial' or 'abstract'
    source_id TEXT NOT NULL,         -- nct_id or pmid
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding vector(1024),          -- BGE-M3 outputs 1024-dim
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Built after all chunks loaded (see pitfalls section in master prompt)
-- CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_trials_phase ON trials(phase);
CREATE INDEX IF NOT EXISTS idx_trials_status ON trials(status);
