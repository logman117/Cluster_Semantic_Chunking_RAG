-- Drop the existing function first
DROP FUNCTION IF EXISTS match_chunks(vector(1536), float, int);

-- Drop existing table if needed (WARNING: This will delete all your existing chunks!)
DROP TABLE IF EXISTS chunks;

-- Create the updated chunks table with vector support
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT REFERENCES documents(id),
    page_numbers INTEGER[],
    page_display TEXT,
    text TEXT,
    metadata JSONB,
    embedding VECTOR(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for vector similarity search
CREATE INDEX chunks_embedding_idx 
ON chunks 
USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);

-- Create the updated match_chunks function
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
    WHERE 1 - (chunks.embedding <-> query_embedding) > 1 - match_threshold
    ORDER BY chunks.embedding <-> query_embedding
    LIMIT match_count;
END;
$$;