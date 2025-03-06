from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import tempfile
import shutil
import asyncio
import uuid
from typing import List, Dict, Any, Optional
import logging
from contextlib import asynccontextmanager
import uvicorn

# Import our RAG system components
from cluster_semantic_chunker import Chunk, process_documents
from embedding_generation import process_and_store, EmbeddingGenerator
from rag_system import RAGSystem, initialize_rag_system

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a lifespan context manager for our RAG system
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize RAG system
    app.state.rag_system = await initialize_rag_system()
    yield
    # Clean up resources
    await app.state.rag_system.close()

# Create FastAPI app
app = FastAPI(
    title="Nilfisk Service Manual RAG API",
    description="API for the Nilfisk Service Manual Retrieval-Augmented Generation System",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify allowed origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    response: str
    sources: Optional[List[Dict[str, Any]]] = None

class DocumentUploadResponse(BaseModel):
    document_id: str
    message: str
    chunk_count: int

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Process a user query and return a response with page citations.
    """
    try:
        rag_system = app.state.rag_system
        response = await rag_system.process_query(request.query, request.top_k)
        
        # Sources could be extended to include more details about the retrieved chunks
        return {
            "response": response,
            "sources": []  # We could populate this with source information if needed
        }
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_id: Optional[str] = None
):
    """
    Upload a PDF document to the system.
    
    The document will be processed, chunked, and stored in the vector database.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Generate a document ID if not provided
    if not document_id:
        document_id = str(uuid.uuid4())
    
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the uploaded file
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Initialize embedding generator for use in the chunker
            embedding_gen = EmbeddingGenerator()
            
            # Process the document using Cluster Semantic Chunking
            chunks = process_documents(temp_dir, embedding_gen.generate_embedding_sync)
            
            # Schedule background task to generate embeddings and store them
            background_tasks.add_task(process_and_store_async, chunks)
            
            return {
                "document_id": document_id,
                "message": f"Document uploaded and processing started. Document ID: {document_id}",
                "chunk_count": len(chunks)
            }
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_and_store_async(chunks: List[Chunk]):
    """Async wrapper for process_and_store to use in background tasks."""
    try:
        await process_and_store(chunks)
        logger.info(f"Completed processing {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Error in background processing: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    # Run the server
    uvicorn.run("api_endpoints:app", host="0.0.0.0", port=8000, reload=True)
