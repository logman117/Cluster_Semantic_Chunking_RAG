import os
import asyncio
import re
from dotenv import load_dotenv
import logging
import json
import httpx
from typing import List, Dict, Any, Optional
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
        
        # Current selected machine
        self.current_machine = None

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

    async def retrieve_context_by_machine(self, 
                                        query_embedding: List[float], 
                                        machine_id: str,
                                        top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks from Supabase using vector search for a specific machine."""
        try:
            # Get machine information
            machine_response = self.supabase.table("machines").select("*").eq("id", machine_id).execute()
            if hasattr(machine_response, "data") and machine_response.data:
                logger.info(f"Found machine: {machine_response.data[0]}")
            else:
                logger.warning(f"Machine {machine_id} not found in the database")
            
            # Map machine ID to document ID (we assume they follow the naming convention)
            document_id = machine_id.upper()  # Convert cs7010 to CS7010
            
            # Verify document exists
            doc_check = self.supabase.table("documents").select("*").eq("id", document_id).execute()
            if hasattr(doc_check, "data") and doc_check.data:
                logger.info(f"Found document with ID {document_id}: {doc_check.data[0]}")
                # Get the actual machine_id from the document (just for logging)
                doc_machine_id = doc_check.data[0].get('machine_id')
                logger.info(f"Document {document_id} is linked to machine_id: {doc_machine_id}")
                
                # Use the match_chunks_by_document function to get relevant chunks
                # This gets chunks ONLY for this document, ordered by vector similarity
                response = self.supabase.rpc(
                    "match_chunks_by_document",
                    {
                        "query_embedding": query_embedding,
                        "input_document_id": document_id,
                        "match_threshold": 0.0,
                        "match_count": top_k
                    }
                ).execute()
                
                if hasattr(response, "data") and response.data:
                    results = response.data
                    logger.info(f"Retrieved {len(results)} relevant chunks for document {document_id}")
                    
                    # Print debug info about the results
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
                            
                        logger.info(f"Result {i+1}: Document {document_id}, {page_info}, Similarity: {similarity}")
                    
                    return results
                else:
                    logger.warning(f"No chunks found with vector similarity for document {document_id}")
                    return []
            else:
                logger.warning(f"No document found with ID {document_id}")
                return []
                
        except Exception as e:
            logger.error(f"Error in machine-specific retrieval: {e}")
            return []

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

    async def generate_response(self, query: str, context: str, machine_info: Dict[str, Any]) -> str:
        """Generate a response using GPT-4o."""
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        system_prompt = f"""
You are an assistant for the {machine_info['name']} ({machine_info['model']}) technical service manual. Your task is to provide accurate and helpful information based on the provided context.

Important guidelines:
1. Base your responses ONLY on the provided context. Do not use any prior knowledge.
2. ALWAYS cite the specific document and page number where you found the information using the format "According to [Document: X, Pages Y-Z]" or "[Document: X, Pages Y-Z]".
3. Include page citations for EVERY piece of information you provide, not just once at the beginning.
4. If the information spans multiple pages, cite the page range (e.g., "Pages 15-16").
5. If the information isn't in the provided context, say "I couldn't find information about this in the {machine_info['name']} service manual" rather than making up an answer.
6. If different pages contain conflicting information, present both and note the discrepancy.
7. Use clear, concise language appropriate for a technical manual.
8. When providing instructions, present them in step-by-step format for clarity.
9. When mentioning specific parts or components, include their part numbers if available.
10. END your response with a list of all pages consulted, formatted as "Pages referenced: X, Y-Z"

Remember: Page citations are CRITICAL. Every fact must be linked to a specific page or page range.
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query about {machine_info['name']} ({machine_info['model']}): {query}\n\nContext:\n{context}"}
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

    async def list_machines(self) -> List[Dict[str, Any]]:
        """Get a list of all available machines."""
        try:
            response = self.supabase.table("machines").select("*").execute()
            
            if hasattr(response, "data"):
                return response.data
            
            return []
        except Exception as e:
            logger.error(f"Error listing machines: {e}")
            return []

    async def get_machine_info(self, machine_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific machine."""
        try:
            response = self.supabase.table("machines").select("*").eq("id", machine_id).execute()
            
            if hasattr(response, "data") and response.data:
                return response.data[0]
            
            return None
        except Exception as e:
            logger.error(f"Error getting machine info: {e}")
            return None

    async def query(self, user_query: str, machine_id: str) -> str:
        """
        Process a query through the RAG pipeline for a specific machine.
        
        Args:
            user_query: The user's query text
            machine_id: ID of the machine to query about
            
        Returns:
            Generated response with page citations
        """
        try:
            logger.info(f"Processing query for machine {machine_id}: {user_query}")
            
            # Get machine information
            machine_info = await self.get_machine_info(machine_id)
            
            if not machine_info:
                return f"Error: Machine with ID '{machine_id}' not found. Please select a valid machine."
            
            # Extract error code if present (using regex for common formats)
            error_code_match = re.search(r'(\d+-\d+|\d+[A-Za-z]\d+|[A-Za-z]\d+)', user_query)
            search_text = error_code_match.group(0) if error_code_match else user_query
            
            # Generate embedding for query
            query_embedding = await self.generate_embedding(user_query)
            logger.info("Generated query embedding")
            
            # If error code is detected, emphasize it in the query
            if error_code_match:
                # Create a focused query for error codes
                expanded_query = f"error code {search_text} problem issue troubleshooting"
                error_embedding = await self.generate_embedding(expanded_query)
                # Average the original query embedding with the error-focused embedding
                query_embedding = [(a + b) / 2 for a, b in zip(query_embedding, error_embedding)]
                logger.info(f"Enhanced query embedding for error code: {search_text}")
            
            # Retrieve relevant chunks for the specified machine using vector search
            chunks = await self.retrieve_context_by_machine(query_embedding, machine_id)
            logger.info(f"Retrieved {len(chunks)} chunks")
            
            if not chunks:
                return f"I couldn't find any information about this in the {machine_info['name']} service manual. Please try a different query."
            
            # Format context
            context = self._format_context(chunks)
            
            # Generate response
            response = await self.generate_response(user_query, context, machine_info)
            
            return response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"An error occurred while processing your query: {str(e)}"
    
    async def select_machine(self) -> Optional[str]:
        """
        Interactive machine selection.
        
        Returns:
            Selected machine ID or None if canceled
        """
        machines = await self.list_machines()
        
        if not machines:
            print("No machines found in the database. Please add machines first.")
            return None
        
        print("\n== Available Machines ==")
        for i, machine in enumerate(machines):
            print(f"{i+1}. {machine['name']} ({machine['model']})")
        
        while True:
            try:
                selection = input("\nSelect a machine (number) or type 'exit' to quit: ").strip()
                
                if selection.lower() in ['exit', 'quit', 'q']:
                    return None
                
                idx = int(selection) - 1
                if 0 <= idx < len(machines):
                    selected_machine = machines[idx]
                    print(f"\nSelected machine: {selected_machine['name']} ({selected_machine['model']})")
                    return selected_machine['id']
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
    async def close(self):
        await self.client.aclose()

async def interactive_mode():
    print("\n=== Nilfisk Service Manual RAG System ===")
    print("Type 'exit' to quit, 'switch' to change machine\n")
    
    rag = RAGQuery()
    
    try:
        # First, select a machine
        machine_id = await rag.select_machine()
        
        if not machine_id:
            print("No machine selected. Exiting.")
            return
        
        machine_info = await rag.get_machine_info(machine_id)
        print(f"\nNow querying manual for {machine_info['name']} ({machine_info['model']})")
        
        while True:
            query = input("\nEnter your question: ")
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            if query.lower() == 'switch':
                machine_id = await rag.select_machine()
                if not machine_id:
                    print("No machine selected. Exiting.")
                    break
                
                machine_info = await rag.get_machine_info(machine_id)
                print(f"\nNow querying manual for {machine_info['name']} ({machine_info['model']})")
                continue
                
            print("-" * 80)
            response = await rag.query(query, machine_id)
            print(f"Response:\n{response}")
            print("-" * 80)
            
    finally:
        await rag.close()

if __name__ == "__main__":
    asyncio.run(interactive_mode())
