-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a table to store machine information (only if it doesn't exist)
CREATE TABLE IF NOT EXISTS machines (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    model TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create the documents table if it doesn't exist
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT,
    description TEXT,
    machine_id TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create the chunks table if it doesn't exist
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT REFERENCES documents(id),
    page_numbers INTEGER[],
    page_display TEXT,
    text TEXT,
    metadata JSONB,
    embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS chunks_embedding_idx 
ON chunks 
USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);

-- Create an index on document_id for faster lookups
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks(document_id);

-- Create the standard match_chunks function 
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(1536),
    match_threshold float,
    match_count int
)
RETURNS TABLE (
    id text,
    document_id text,
    page_numbers integer[],
    page_display text,
    text text,
    metadata jsonb,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        chunks.id,
        chunks.document_id,
        chunks.page_numbers,
        chunks.page_display,
        chunks.text,
        chunks.metadata,
        1 - (chunks.embedding <-> query_embedding) AS similarity
    FROM chunks
    WHERE 1 - (chunks.embedding <-> query_embedding) > match_threshold
    ORDER BY chunks.embedding <-> query_embedding
    LIMIT match_count;
END;
$$;

-- Create a document-filtered match function
CREATE OR REPLACE FUNCTION match_chunks_by_document(
    query_embedding vector(1536),
    input_document_id text,
    match_threshold float,
    match_count int
)
RETURNS TABLE (
    id text,
    document_id text,
    page_numbers integer[],
    page_display text,
    text text,
    metadata jsonb,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        chunks.id,
        chunks.document_id,
        chunks.page_numbers,
        chunks.page_display,
        chunks.text,
        chunks.metadata,
        1 - (chunks.embedding <-> query_embedding) AS similarity
    FROM chunks
    WHERE chunks.document_id = input_document_id
      AND 1 - (chunks.embedding <-> query_embedding) > match_threshold
    ORDER BY chunks.embedding <-> query_embedding
    LIMIT match_count;
END;
$$;

-- Create the machine-specific match function
CREATE OR REPLACE FUNCTION match_chunks_by_machine(
    query_embedding vector(1536),
    input_machine_id text,
    match_threshold float,
    match_count int
)
RETURNS TABLE (
    id text,
    document_id text,
    page_numbers integer[],
    page_display text,
    text text,
    metadata jsonb,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        chunks.id,
        chunks.document_id,
        chunks.page_numbers,
        chunks.page_display,
        chunks.text,
        chunks.metadata,
        1 - (chunks.embedding <-> query_embedding) AS similarity
    FROM chunks
    JOIN documents ON chunks.document_id = documents.id
    WHERE documents.machine_id = input_machine_id
      AND 1 - (chunks.embedding <-> query_embedding) > match_threshold
    ORDER BY chunks.embedding <-> query_embedding
    LIMIT match_count;
END;
$$;

-- Hybrid search function that uses both text and vector similarity
CREATE OR REPLACE FUNCTION match_chunks_hybrid(
    query_text text,
    query_embedding vector(1536),
    input_document_id text,
    match_threshold float,
    match_count int
)
RETURNS TABLE (
    id text,
    document_id text,
    page_numbers integer[],
    page_display text,
    text text,
    metadata jsonb,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    -- For hybrid search, give a boost to chunks that contain the query text
    RETURN QUERY
    SELECT
        chunks.id,
        chunks.document_id,
        chunks.page_numbers,
        chunks.page_display,
        chunks.text,
        chunks.metadata,
        CASE
            WHEN chunks.text ILIKE '%' || query_text || '%' THEN 
                (1 - (chunks.embedding <-> query_embedding)) * 1.5  -- 50% boost for text match
            ELSE
                1 - (chunks.embedding <-> query_embedding)
        END AS similarity
    FROM chunks
    WHERE chunks.document_id = input_document_id
      AND 1 - (chunks.embedding <-> query_embedding) > match_threshold
    ORDER BY similarity DESC  -- Order by our boosted similarity score
    LIMIT match_count;
END;
$$;
