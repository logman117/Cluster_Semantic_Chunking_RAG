from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, Body, Query
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
    machine_id: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    response: str
    sources: Optional[List[Dict[str, Any]]] = None

class DocumentUploadResponse(BaseModel):
    document_id: str
    message: str
    chunk_count: int

class Machine(BaseModel):
    id: str
    name: str
    model: str
    description: Optional[str] = None

class MachineList(BaseModel):
    machines: List[Machine]

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Process a user query for a specific machine and return a response with page citations.
    """
    try:
        rag_system = app.state.rag_system
        # Pass the machine_id to the process_query method, which will be updated in rag_system.py
        response = await rag_system.process_query(request.query, request.machine_id, request.top_k)
        
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
    machine_id: str = Query(..., description="The ID of the machine this document is for"),
    document_id: Optional[str] = None
):
    """
    Upload a PDF document for a specific machine.
    
    The document will be processed, chunked, and stored in the vector database
    with the associated machine_id.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Generate a document ID if not provided
    if not document_id:
        document_id = str(uuid.uuid4())
    
    try:
        # Verify the machine exists
        machine = await get_machine_by_id(machine_id)
        if not machine:
            raise HTTPException(status_code=404, detail=f"Machine with ID {machine_id} not found")
        
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
            
            # Set the document's machine_id
            for chunk in chunks:
                if not chunk.metadata:
                    chunk.metadata = {}
                chunk.metadata["machine_id"] = machine_id
            
            # Schedule background task to generate embeddings and store them
            background_tasks.add_task(process_and_store_async, chunks, machine_id)
            
            return {
                "document_id": document_id,
                "message": f"Document uploaded and processing started for machine {machine_id}. Document ID: {document_id}",
                "chunk_count": len(chunks)
            }
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_and_store_async(chunks: List[Chunk], machine_id: str):
    """Async wrapper for process_and_store to use in background tasks."""
    try:
        await process_and_store(chunks, machine_id)
        logger.info(f"Completed processing {len(chunks)} chunks for machine {machine_id}")
    except Exception as e:
        logger.error(f"Error in background processing: {e}")

async def get_machine_by_id(machine_id: str) -> Optional[Dict[str, Any]]:
    """
    Get machine information from the database.
    
    Args:
        machine_id: The ID of the machine to retrieve
        
    Returns:
        Machine information or None if not found
    """
    try:
        from supabase import create_client
        
        # Get Supabase credentials from environment variables
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not supabase_url or not supabase_key:
            logger.error("Missing Supabase credentials")
            return None
        
        # Initialize Supabase client
        supabase = create_client(supabase_url, supabase_key)
        
        # Query the machines table
        response = supabase.table("machines").select("*").eq("id", machine_id).execute()
        
        if hasattr(response, "data") and response.data:
            return response.data[0]
        
        return None
    except Exception as e:
        logger.error(f"Error retrieving machine: {e}")
        return None

@app.get("/machines", response_model=MachineList)
async def list_machines():
    """
    Get a list of all available machines.
    """
    try:
        from supabase import create_client
        
        # Get Supabase credentials from environment variables
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not supabase_url or not supabase_key:
            raise HTTPException(status_code=500, detail="Missing Supabase credentials")
        
        # Initialize Supabase client
        supabase = create_client(supabase_url, supabase_key)
        
        # Query the machines table
        response = supabase.table("machines").select("*").execute()
        
        if hasattr(response, "data"):
            return {"machines": response.data}
        else:
            return {"machines": []}
    except Exception as e:
        logger.error(f"Error listing machines: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    # Run the server
    uvicorn.run("api_endpoints:app", host="0.0.0.0", port=8000, reload=True)
