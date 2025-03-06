import os
import asyncio
import re
from dotenv import load_dotenv
import logging
import json
import httpx
from typing import List, Dict, Any
from supabase import create_client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGQuery:
    def __init__(self):
        # Azure OpenAI settings
        self.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = "2024-08-01-preview"
        self.model_name = "gpt-4o"
        
        if not self.api_base or not self.api_key:
            raise ValueError("Missing Azure OpenAI API credentials in environment variables")
        
        self.chat_endpoint = f"{self.api_base}/openai/deployments/{self.model_name}/chat/completions?api-version={self.api_version}"
        self.client = httpx.AsyncClient(timeout=60.0)

        # Embedding settings
        self.embedding_api_version = "2023-05-15"
        self.embedding_model = "text-embedding-3-large"
        self.embedding_endpoint = f"{self.api_base}/openai/deployments/{self.embedding_model}/embeddings?api-version={self.embedding_api_version}"
        
        # Supabase settings
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Missing Supabase credentials in environment variables")
        
        self.supabase = create_client(self.supabase_url, self.supabase_key)

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for query text."""
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        payload = {
            "input": text,
            "dimensions": 1536
        }
        
        response = await self.client.post(
            self.embedding_endpoint,
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Error generating embedding: {response.status_code} {response.text}")
        
        result = response.json()
        return result["data"][0]["embedding"]

    async def retrieve_context(self, query_embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks from Supabase using vector search."""
        # Try with a very low threshold to ensure we get results
        response = self.supabase.rpc(
            "match_chunks",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.0,
                "match_count": top_k
            }
        ).execute()
        
        results = response.data if hasattr(response, "data") else []
        
        # Print some debug information about the results
        for i, result in enumerate(results):
            similarity = result.get('similarity', 'unknown')
            
            # Handle different page number formats
            if "page_numbers" in result:
                page_numbers = result["page_numbers"]
                if len(page_numbers) == 1:
                    page_info = f"Page {page_numbers[0]}"
                else:
                    page_info = f"Pages {min(page_numbers)}-{max(page_numbers)}"
            elif "page_display" in result:
                page_info = f"Page(s) {result['page_display']}"
            else:
                page_info = f"Page {result.get('page_number', 'unknown')}"
                
            document = result.get('document_id', 'unknown')
            logger.info(f"Result {i+1}: Document {document}, {page_info}, Similarity: {similarity}")
        
        # Diagnostic code to inspect the chunks table
        if not results:
            try:
                # Check if there are any chunks in the database
                count_query = self.supabase.table("chunks").select("count", count="exact").execute()
                total_chunks = count_query.count if hasattr(count_query, "count") else 0
                logger.info(f"Total chunks in database: {total_chunks}")
                
                if total_chunks > 0:
                    # Check a sample chunk
                    sample = self.supabase.table("chunks").select("*").limit(1).execute()
                    if hasattr(sample, "data") and sample.data:
                        logger.info(f"Sample chunk: {sample.data[0]}")
                    else:
                        logger.warning("No sample chunk available")
            except Exception as e:
                logger.error(f"Error checking database: {e}")
        
        return results

    async def retrieve_context_hybrid(self, query_text: str, query_embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks using both text and vector similarity."""
        try:
            # First try text + vector hybrid search
            response = self.supabase.rpc(
                "match_chunks_with_text",
                {
                    "query_text": query_text,
                    "query_embedding": query_embedding,
                    "match_threshold": 0.0,
                    "match_count": top_k
                }
            ).execute()
            
            results = response.data if hasattr(response, "data") else []
            
            # If hybrid search returned results, use them
            if results:
                # Print debug information
                for i, result in enumerate(results):
                    similarity = result.get('similarity', 'unknown')
                    
                    # Handle different page number formats
                    if "page_numbers" in result:
                        page_numbers = result["page_numbers"]
                        if len(page_numbers) == 1:
                            page_info = f"Page {page_numbers[0]}"
                        else:
                            page_info = f"Pages {min(page_numbers)}-{max(page_numbers)}"
                    elif "page_display" in result:
                        page_info = f"Page(s) {result['page_display']}"
                    else:
                        page_info = f"Page {result.get('page_number', 'unknown')}"
                        
                    document = result.get('document_id', 'unknown')
                    logger.info(f"Result {i+1}: Document {document}, {page_info}, Similarity: {similarity}")
                
                return results
            
            # If hybrid search returned no results, fall back to standard vector search
            logger.info("No results from hybrid search, falling back to vector search")
            return await self.retrieve_context(query_embedding, top_k)
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fall back to regular vector search if hybrid fails
            logger.info("Hybrid search failed, falling back to vector search")
            return await self.retrieve_context(query_embedding, top_k)

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks as context with page numbers."""
        context_parts = []
        
        for chunk in chunks:
            document_id = chunk.get("document_id", "unknown")
            
            # Handle both legacy single page and new multi-page format
            if "page_numbers" in chunk:
                # New format with multiple pages
                page_numbers = chunk["page_numbers"]
                if len(page_numbers) == 1:
                    page_display = f"Page {page_numbers[0]}"
                else:
                    page_display = f"Pages {min(page_numbers)}-{max(page_numbers)}"
            elif "page_display" in chunk:
                # Use pre-formatted page display if available
                page_display = f"Page(s) {chunk['page_display']}"
            else:
                # Legacy format with single page
                page_display = f"Page {chunk.get('page_number', 'unknown')}"
                
            context_parts.append(f"[Document: {document_id}, {page_display}]\n{chunk['text']}\n")
        
        return "\n".join(context_parts)

    async def generate_response(self, query: str, context: str) -> str:
        """Generate a response using GPT-4o."""
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        system_prompt = """
You are an assistant for technical service manuals. Your task is to provide accurate and helpful information based on the provided context.

Important guidelines:
1. Base your responses ONLY on the provided context. Do not use any prior knowledge.
2. ALWAYS cite the specific document and page number where you found the information using the format "According to [Document: X, Pages Y-Z]" or "[Document: X, Pages Y-Z]".
3. Include page citations for EVERY piece of information you provide, not just once at the beginning.
4. If the information spans multiple pages, cite the page range (e.g., "Pages 15-16").
5. If the information isn't in the provided context, say "I couldn't find information about this in the service manual" rather than making up an answer.
6. If different pages contain conflicting information, present both and note the discrepancy.
7. Use clear, concise language appropriate for a technical manual.
8. When providing instructions, present them in step-by-step format for clarity.
9. When mentioning specific parts or components, include their part numbers if available.
10. END your response with a list of all pages consulted, formatted as "Pages referenced: X, Y-Z"

Remember: Page citations are CRITICAL. Every fact must be linked to a specific page or page range.
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}\n\nContext:\n{context}"}
        ]
        
        payload = {
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        response = await self.client.post(
            self.chat_endpoint,
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Error generating response: {response.status_code} {response.text}")
        
        result = response.json()
        return result["choices"][0]["message"]["content"]

    async def query(self, user_query: str) -> str:
        """Process a query through the improved RAG pipeline with hybrid search."""
        try:
            logger.info(f"Processing query: {user_query}")
            
            # Extract error code if present (using regex for common formats)
            error_code_match = re.search(r'(\d+-\d+|\d+[A-Za-z]\d+|[A-Za-z]\d+)', user_query)
            search_text = error_code_match.group(0) if error_code_match else user_query
            
            # Generate embedding for query
            query_embedding = await self.generate_embedding(user_query)
            logger.info("Generated query embedding")
            
            # Try hybrid search first with extracted code or full query
            chunks = await self.retrieve_context_hybrid(search_text, query_embedding)
            logger.info(f"Retrieved {len(chunks)} chunks")
            
            if not chunks:
                return "I couldn't find any relevant information in the service manual."
            
            # Format context
            context = self._format_context(chunks)
            
            # Generate response
            response = await self.generate_response(user_query, context)
            
            return response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"An error occurred while processing your query: {str(e)}"
    
    async def close(self):
        await self.client.aclose()

async def interactive_mode():
    print("\n=== Nilfisk Service Manual RAG System ===")
    print("Type 'exit' to quit\n")
    
    rag = RAGQuery()
    
    try:
        while True:
            query = input("\nEnter your question: ")
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
                
            print("-" * 80)
            response = await rag.query(query)
            print(f"Response:\n{response}")
            print("-" * 80)
            
    finally:
        await rag.close()

if __name__ == "__main__":
    asyncio.run(interactive_mode())