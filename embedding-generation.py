import os
import json
import httpx
import asyncio
import numpy as np
import requests
from typing import List, Dict, Any, Union, Callable, Optional
from dotenv import load_dotenv
from supabase import create_client, Client
import logging
from tqdm import tqdm
from cluster_semantic_chunker import Chunk, process_documents, openai_token_count

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SupabaseVectorStore:
    """Stores and retrieves embeddings and chunks using Supabase."""
    
    def __init__(self):
        """Initialize Supabase client with credentials from environment variables."""
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Missing Supabase credentials in environment variables")
        
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
    
    def store_document(self, 
                      document_id: str, 
                      title: str, 
                      description: str = "", 
                      machine_id: Optional[str] = None) -> None:
        """
        Store document metadata.
        
        Args:
            document_id: Unique identifier for the document
            title: Document title
            description: Optional document description
            machine_id: ID of the machine this document is for
        """
        document_data = {
            "id": document_id,
            "title": title,
            "description": description
        }
        
        # Add machine_id if provided
        if machine_id:
            document_data["machine_id"] = machine_id
        
        response = self.supabase.table("documents").upsert(document_data).execute()
        
        # Check for errors
        if hasattr(response, "error") and response.error:
            logger.error(f"Error storing document: {response.error}")
            raise Exception(f"Error storing document: {response.error}")
    
    def store_chunks(self, 
                    chunks: List[Any], 
                    embeddings: Dict[str, List[float]],
                    machine_id: Optional[str] = None) -> None:
        """
        Store chunks and their embeddings in Supabase.
        
        Args:
            chunks: List of Chunk objects
            embeddings: Dictionary mapping chunk_id to embedding vector
            machine_id: Optional machine ID to associate with chunks
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
            
            # Initialize metadata or use existing
            metadata = chunk.metadata or {}
            
            # Add machine_id to metadata if provided
            if machine_id and "machine_id" not in metadata:
                metadata["machine_id"] = machine_id
                
            data.append({
                "id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "page_numbers": chunk.page_numbers,
                "page_display": page_display,
                "text": chunk.text,
                "metadata": json.dumps(metadata),
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
                "match_threshold": similarity_threshold,
                "match_count": limit
            }
        ).execute()
        
        if hasattr(response, "data"):
            return response.data
        else:
            logger.warning("No data returned from similarity search")
            return []
    
    def similarity_search_by_machine(self,
                                   query_embedding: List[float],
                                   machine_id: str,
                                   limit: int = 5,
                                   similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Perform similarity search filtered by machine ID.
        
        Args:
            query_embedding: Embedding vector of the query
            machine_id: ID of the machine to filter by
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of matching chunks with metadata and similarity scores
        """
        try:
            # Debug check for chunks
            count = self._check_chunks_for_machine(machine_id)
            logger.info(f"Found {count} chunks for machine {machine_id}")
            
            # Use the machine-specific match function with parameter name input_machine_id
            response = self.supabase.rpc(
                "match_chunks_by_machine",
                {
                    "query_embedding": query_embedding,
                    "input_machine_id": machine_id,  # Use input_machine_id to match SQL function
                    "match_threshold": similarity_threshold,
                    "match_count": limit
                }
            ).execute()
            
            if hasattr(response, "data"):
                return response.data
        except Exception as e:
            logger.error(f"Error in machine-specific similarity search: {e}")
            # Fallback to filtering in application code if the RPC fails
            logger.info("Falling back to application-level filtering")
            return self._fallback_machine_search(query_embedding, machine_id, limit, similarity_threshold)
        
        logger.warning(f"No data returned from machine-specific similarity search for machine {machine_id}")
        return []
    
    def _check_chunks_for_machine(self, machine_id: str) -> int:
        """
        Check how many chunks exist for a specific machine's documents.
        
        Args:
            machine_id: ID of the machine
            
        Returns:
            Count of chunks
        """
        try:
            # First, get document IDs for this machine
            doc_response = self.supabase.table("documents").select("id").eq("machine_id", machine_id).execute()
            
            if not hasattr(doc_response, "data") or not doc_response.data:
                logger.warning(f"No documents found for machine {machine_id}")
                return 0
                
            document_ids = [doc["id"] for doc in doc_response.data]
            logger.info(f"Found documents for machine {machine_id}: {document_ids}")
            
            # Now count chunks for these documents
            if not document_ids:
                return 0
                
            # For simplicity, just count chunks for the first document
            first_doc_id = document_ids[0]
            chunks_response = self.supabase.table("chunks").select("id").eq("document_id", first_doc_id).execute()
            
            if hasattr(chunks_response, "data"):
                return len(chunks_response.data)
                
            return 0
        except Exception as e:
            logger.error(f"Error counting chunks for machine {machine_id}: {e}")
            return 0
    
    def _fallback_machine_search(self,
                               query_embedding: List[float],
                               machine_id: str,
                               limit: int = 5,
                               similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Fallback method that filters by machine ID in application code.
        Used if the database-level filtering fails.
        """
        # Get documents for this machine
        document_response = self.supabase.table("documents").select("id").eq("machine_id", machine_id).execute()
        
        if not hasattr(document_response, "data") or not document_response.data:
            logger.warning(f"No documents found for machine {machine_id}")
            return []
        
        # Extract document IDs
        document_ids = [doc["id"] for doc in document_response.data]
        
        # Get chunks for these documents with similarity search
        all_results = self.similarity_search(query_embedding, limit=limit*2, similarity_threshold=similarity_threshold)
        
        # Filter to only include chunks from the machine's documents
        filtered_results = [
            result for result in all_results 
            if result.get("document_id") in document_ids
        ]
        
        # Limit to requested number
        return filtered_results[:limit]
    
    async def get_machine_info(self, machine_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a machine.
        
        Args:
            machine_id: ID of the machine
            
        Returns:
            Machine information or None if not found
        """
        try:
            response = self.supabase.table("machines").select("*").eq("id", machine_id).execute()
            
            if hasattr(response, "data") and response.data:
                return response.data[0]
            
            return None
        except Exception as e:
            logger.error(f"Error retrieving machine info: {e}")
            return None
    
    def list_machines(self) -> List[Dict[str, Any]]:
        """
        Get a list of all machines.
        
        Returns:
            List of machine information dictionaries
        """
        try:
            response = self.supabase.table("machines").select("*").execute()
            
            if hasattr(response, "data"):
                return response.data
            
            return []
        except Exception as e:
            logger.error(f"Error listing machines: {e}")
            return []
    
    def store_machine(self, 
                     machine_id: str, 
                     name: str, 
                     model: str, 
                     description: str = "") -> None:
        """
        Store machine information.
        
        Args:
            machine_id: Unique identifier for the machine
            name: Machine name
            model: Machine model number
            description: Optional machine description
        """
        response = self.supabase.table("machines").upsert({
            "id": machine_id,
            "name": name,
            "model": model,
            "description": description
        }).execute()
        
        # Check for errors
        if hasattr(response, "error") and response.error:
            logger.error(f"Error storing machine: {response.error}")
            raise Exception(f"Error storing machine: {response.error}")

class EmbeddingGenerator:
    """Generates embeddings using Azure OpenAI API."""
    
    def __init__(self):
        """Initialize with API credentials from environment variables."""
        self.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = "2023-05-15"
        self.model_name = "text-embedding-3-large"
        self.max_tokens = 8000  # Set max tokens below the model limit of 8192 to be safe
        
        if not self.api_base or not self.api_key:
            raise ValueError("Missing Azure OpenAI API credentials in environment variables")
        
        self.embedding_endpoint = f"{self.api_base}/openai/deployments/{self.model_name}/embeddings?api-version={self.api_version}"
        self.client = httpx.AsyncClient(timeout=60.0)  # Set a longer timeout for embedding calls
    
    def _truncate_text_by_tokens(self, text: str) -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            
        Returns:
            Truncated text that fits within the token limit
        """
        token_count = openai_token_count(text)
        
        if token_count <= self.max_tokens:
            return text
        
        # If too long, truncate by calculating an approximate character-to-token ratio
        # and cutting down to fit within the token limit
        ratio = len(text) / token_count
        estimated_chars = int(self.max_tokens * ratio * 0.9)  # Apply a safety factor
        
        truncated_text = text[:estimated_chars]
        
        # Verify the truncation worked
        if openai_token_count(truncated_text) > self.max_tokens:
            # If still too long, do a more aggressive truncation
            logger.warning(f"First truncation attempt still has too many tokens. Applying more aggressive truncation.")
            estimated_chars = int(estimated_chars * 0.8)
            truncated_text = text[:estimated_chars]
            
        logger.warning(f"Truncated text from {token_count} tokens to {openai_token_count(truncated_text)} tokens (limit: {self.max_tokens})")
        return truncated_text
    
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
        
        # Truncate text to fit within token limit
        text = self._truncate_text_by_tokens(text)
        
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
        
        # Truncate text to fit within token limit
        text = self._truncate_text_by_tokens(text)
        
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
                await asyncio.sleep(0.2)
        
        return embeddings
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

async def process_and_store(chunks: List[Chunk], machine_id: Optional[str] = None) -> None:
    """
    Generate embeddings for chunks and store them in Supabase.
    
    Args:
        chunks: List of Chunk objects to process
        machine_id: Optional ID of the machine these chunks are for
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
                description="Processed from PDF",
                machine_id=machine_id
            )
        
        # Store chunks with embeddings
        vector_store.store_chunks(chunks, embeddings, machine_id)
        logger.info(f"Stored chunks and embeddings in Supabase for machine ID: {machine_id or 'None'}")
        
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
        
        # Ask for machine ID if processing for a specific machine
        machine_id = input("Enter machine ID (or leave blank if not machine-specific): ").strip()
        
        if machine_id:
            # Verify machine exists or create it
            vector_store = SupabaseVectorStore()
            machine_info = await vector_store.get_machine_info(machine_id)
            
            if not machine_info:
                logger.info(f"Machine {machine_id} not found. Creating a new machine entry.")
                name = input(f"Enter name for machine {machine_id}: ").strip()
                model = input(f"Enter model for machine {machine_id}: ").strip()
                description = input(f"Enter description for machine {machine_id} (optional): ").strip()
                
                vector_store.store_machine(
                    machine_id=machine_id,
                    name=name or machine_id.upper(),
                    model=model or machine_id.upper(),
                    description=description
                )
        
        # Generate embeddings and store in Supabase
        await process_and_store(chunks, machine_id if machine_id else None)
    
    asyncio.run(main())
