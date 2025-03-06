import os
import json
import httpx
import asyncio
import numpy as np
import requests
from typing import List, Dict, Any, Union, Callable
from dotenv import load_dotenv
from supabase import create_client, Client
import logging
from tqdm import tqdm
from cluster_semantic_chunker import Chunk, process_documents

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generates embeddings using Azure OpenAI API."""
    
    def __init__(self):
        """Initialize with API credentials from environment variables."""
        self.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = "2023-05-15"
        self.model_name = "text-embedding-3-large"
        
        if not self.api_base or not self.api_key:
            raise ValueError("Missing Azure OpenAI API credentials in environment variables")
        
        self.embedding_endpoint = f"{self.api_base}/openai/deployments/{self.model_name}/embeddings?api-version={self.api_version}"
        self.client = httpx.AsyncClient(timeout=60.0)  # Set a longer timeout for embedding calls
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text asynchronously.
        
        Args:
            text: The text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        # Truncate text if necessary (text-embedding-3-large has an 8191 token limit)
        if len(text) > 32000:  # Approximate character limit
            logger.warning(f"Text too long ({len(text)} chars), truncating to 32000 chars")
            text = text[:32000]
        
        payload = {
            "input": text,
            "dimensions": 1536  # Default dimension for text-embedding-3-large
        }
        
        try:
            response = await self.client.post(
                self.embedding_endpoint,
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"Error generating embedding: {response.status_code} {response.text}")
                raise Exception(f"Error generating embedding: {response.status_code} {response.text}")
            
            result = response.json()
            embedding = result["data"][0]["embedding"]
            return embedding
            
        except Exception as e:
            logger.error(f"Error calling Azure OpenAI API: {e}")
            raise
    
    # Fully synchronous version of generate_embedding for use in the chunker
    def generate_embedding_sync(self, text: str) -> List[float]:
        """
        Generate embedding for a single text synchronously.
        This is a completely separate implementation to avoid nested event loops.
        
        Args:
            text: The text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        # Truncate text if necessary
        if len(text) > 32000:
            logger.warning(f"Text too long ({len(text)} chars), truncating to 32000 chars")
            text = text[:32000]
        
        payload = {
            "input": text,
            "dimensions": 1536
        }
        
        try:
            # Use the synchronous requests library instead of httpx
            response = requests.post(
                self.embedding_endpoint,
                headers=headers,
                json=payload,
                timeout=60.0
            )
            
            if response.status_code != 200:
                logger.error(f"Error generating embedding: {response.status_code} {response.text}")
                raise Exception(f"Error generating embedding: {response.status_code} {response.text}")
            
            result = response.json()
            embedding = result["data"][0]["embedding"]
            return embedding
            
        except Exception as e:
            logger.error(f"Error calling Azure OpenAI API: {e}")
            raise
    
    async def generate_embeddings_batch(self, 
                                       chunks: List[Chunk], 
                                       batch_size: int = 3) -> Dict[str, List[float]]:
        """
        Generate embeddings for multiple chunks in batches with rate limit handling.
        
        Args:
            chunks: List of Chunk objects
            batch_size: Number of embeddings to generate in parallel
            
        Returns:
            Dictionary mapping chunk_id to embedding vector
        """
        embeddings = {}
        
        # Process in batches
        for i in tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings"):
            batch = chunks[i:i+batch_size]
            batch_embeddings = []
            
            # Generate embeddings for the batch with retries
            for chunk in batch:
                # Retry logic for each individual embedding
                max_retries = 7
                retry_count = 0
                retry_delay = 15  # Initial delay in seconds
                
                while retry_count < max_retries:
                    try:
                        # Add a small delay between each request to avoid rate limits
                        if retry_count > 0:
                            logger.info(f"Retrying embedding generation for chunk {chunk.chunk_id} (attempt {retry_count+1})")
                        
                        embedding = await self.generate_embedding(chunk.text)
                        batch_embeddings.append((chunk, embedding))
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        retry_count += 1
                        if "429" in str(e) and retry_count < max_retries:
                            # Rate limit hit, implement exponential backoff
                            wait_time = retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
                            logger.warning(f"Rate limit hit. Waiting {wait_time} seconds before retry {retry_count+1}/{max_retries}")
                            await asyncio.sleep(wait_time)
                        elif retry_count < max_retries:
                            # Other error, still retry but with shorter delay
                            logger.warning(f"Error generating embedding: {e}. Retrying in {retry_delay} seconds.")
                            await asyncio.sleep(retry_delay)
                        else:
                            # Max retries reached
                            logger.error(f"Failed to generate embedding after {max_retries} retries: {e}")
                            raise
                
            # Map embeddings to chunk IDs
            for chunk, embedding in batch_embeddings:
                embeddings[chunk.chunk_id] = embedding
            
            # Add a short delay between batches to avoid hitting rate limits
            if i + batch_size < len(chunks):
                await asyncio.sleep(1)
        
        return embeddings
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

class SupabaseVectorStore:
    """Stores and retrieves embeddings and chunks using Supabase."""
    
    def __init__(self):
        """Initialize Supabase client with credentials from environment variables."""
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Missing Supabase credentials in environment variables")
        
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
    
    def store_document(self, document_id: str, title: str, description: str = "") -> None:
        """
        Store document metadata.
        
        Args:
            document_id: Unique identifier for the document
            title: Document title
            description: Optional document description
        """
        response = self.supabase.table("documents").upsert({
            "id": document_id,
            "title": title,
            "description": description
        }).execute()
        
        # Check for errors
        if hasattr(response, "error") and response.error:
            logger.error(f"Error storing document: {response.error}")
            raise Exception(f"Error storing document: {response.error}")
    
    def store_chunks(self, 
                    chunks: List[Chunk], 
                    embeddings: Dict[str, List[float]]) -> None:
        """
        Store chunks and their embeddings in Supabase.
        
        Args:
            chunks: List of Chunk objects
            embeddings: Dictionary mapping chunk_id to embedding vector
        """
        # Prepare data for insertion
        data = []
        for chunk in chunks:
            if chunk.chunk_id not in embeddings:
                logger.warning(f"No embedding found for chunk {chunk.chunk_id}")
                continue
                
            # Format page numbers as a string range if needed
            if len(chunk.page_numbers) == 1:
                page_display = str(chunk.page_numbers[0])
            else:
                page_display = f"{min(chunk.page_numbers)}-{max(chunk.page_numbers)}"
                
            data.append({
                "id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "page_numbers": chunk.page_numbers,
                "page_display": page_display,
                "text": chunk.text,
                "metadata": json.dumps(chunk.metadata or {}),
                "embedding": embeddings[chunk.chunk_id]
            })
        
        # Insert in batches of 100 to avoid request size limits
        batch_size = 100
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            logger.info(f"Storing batch of {len(batch)} chunks")
            
            self.supabase.table("chunks").upsert(batch).execute()
    
    def similarity_search(self, 
                         query_embedding: List[float], 
                         limit: int = 5, 
                         similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Perform similarity search using vector embeddings.
        
        Args:
            query_embedding: Embedding vector of the query
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of matching chunks with metadata and similarity scores
        """
        # First try with standard threshold
        response = self.supabase.rpc(
            "match_chunks",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.0,  # Use a very low threshold to ensure we get results
                "match_count": limit
            }
        ).execute()
        
        if hasattr(response, "data"):
            return response.data
        else:
            logger.warning("No data returned from similarity search")
            return []

async def process_and_store(chunks: List[Chunk]) -> None:
    """
    Generate embeddings for chunks and store them in Supabase.
    
    Args:
        chunks: List of Chunk objects to process
    """
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator()
    
    try:
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = await embedding_generator.generate_embeddings_batch(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Store in Supabase
        vector_store = SupabaseVectorStore()
        
        # Store document metadata (assuming all chunks are from the same document)
        if chunks:
            document_id = chunks[0].document_id
            vector_store.store_document(
                document_id=document_id,
                title=f"Document {document_id}",
                description="Processed from PDF"
            )
        
        # Store chunks with embeddings
        vector_store.store_chunks(chunks, embeddings)
        logger.info("Stored chunks and embeddings in Supabase")
        
    finally:
        # Close the embedding generator client
        await embedding_generator.close()

if __name__ == "__main__":
    # Use the Cluster Semantic Chunker
    
    async def main():
        # Get the current working directory
        current_dir = os.getcwd()
        
        # Create data directories if they don't exist
        pdf_dir = os.path.join(current_dir, "data", "pdfs")
        os.makedirs(pdf_dir, exist_ok=True)
        
        # Print directory information for debugging
        logger.info(f"Current working directory: {current_dir}")
        logger.info(f"PDF directory: {pdf_dir}")
        logger.info(f"PDF directory exists: {os.path.exists(pdf_dir)}")
        
        # List files in PDF directory
        if os.path.exists(pdf_dir):
            files = os.listdir(pdf_dir)
            pdf_files = [f for f in files if f.lower().endswith('.pdf')]
            logger.info(f"Found {len(pdf_files)} PDF files in directory: {pdf_files}")
        
        # Initialize embedding generator for use in the chunker
        embedding_gen = EmbeddingGenerator()
        
        # Process documents using the Cluster Semantic Chunker
        chunks = process_documents(pdf_dir, embedding_gen.generate_embedding_sync)
        
        if not chunks:
            logger.error("No chunks were generated. Check if there are PDF files in the data/pdfs directory.")
            return
            
        logger.info(f"Generated {len(chunks)} chunks from PDFs")
        
        # Generate embeddings and store in Supabase
        await process_and_store(chunks)
    
    asyncio.run(main())