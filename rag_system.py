import os
import json
import asyncio
import httpx
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import logging
from embedding_generation import EmbeddingGenerator, SupabaseVectorStore

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Generates responses using Azure OpenAI GPT-4o."""
    
    def __init__(self):
        """Initialize with API credentials from environment variables."""
        self.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = "2024-08-01-preview"
        self.model_name = "gpt-4o"
        
        if not self.api_base or not self.api_key:
            raise ValueError("Missing Azure OpenAI API credentials in environment variables")
        
        self.chat_endpoint = f"{self.api_base}/openai/deployments/{self.model_name}/chat/completions?api-version={self.api_version}"
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def generate_response(self, 
                              system_prompt: str,
                              user_query: str, 
                              context: str,
                              machine_info: Dict[str, Any]) -> str:
        """
        Generate a response using GPT-4o.
        
        Args:
            system_prompt: System prompt for the model
            user_query: User's query
            context: Retrieved context with page numbers
            machine_info: Information about the selected machine
            
        Returns:
            Generated response
        """
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        # Update system prompt with machine-specific information
        machine_specific_prompt = system_prompt.replace(
            "You are an assistant for technical service manuals.",
            f"You are an assistant for the {machine_info['name']} ({machine_info['model']}) technical service manual."
        )
        
        messages = [
            {"role": "system", "content": machine_specific_prompt},
            {"role": "user", "content": f"Query about {machine_info['name']} ({machine_info['model']}): {user_query}\n\nContext:\n{context}"}
        ]
        
        payload = {
            "messages": messages,
            "temperature": 0.1,  # Lower temperature for more consistent responses
            "max_tokens": 1000
        }
        
        try:
            response = await self.client.post(
                self.chat_endpoint,
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"Error generating response: {response.status_code} {response.text}")
                raise Exception(f"Error generating response: {response.status_code} {response.text}")
            
            result = response.json()
            response_text = result["choices"][0]["message"]["content"]
            return response_text
            
        except Exception as e:
            logger.error(f"Error calling Azure OpenAI API: {e}")
            raise
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

class RAGSystem:
    """Retrieval-Augmented Generation system for service manuals."""
    
    def __init__(self):
        """Initialize RAG system components."""
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = SupabaseVectorStore()
        self.response_generator = ResponseGenerator()
    
    async def process_query(self, 
                          query: str,
                          machine_id: str,
                          top_k: int = 5) -> str:
        """
        Process a user query and generate a response with page citations.
        
        Args:
            query: User's query
            machine_id: ID of the machine to query about
            top_k: Number of chunks to retrieve
            
        Returns:
            Generated response with page citations
        """
        try:
            # Get machine information
            machine_info = await self.vector_store.get_machine_info(machine_id)
            
            if not machine_info:
                return f"Error: Machine with ID '{machine_id}' not found. Please select a valid machine."
            
            # Generate embedding for the query
            query_embedding = await self.embedding_generator.generate_embedding(query)
            
            # Retrieve relevant chunks for the specified machine
            chunks = await self._retrieve_chunks(query_embedding, machine_id, top_k)
            
            if not chunks:
                return f"I couldn't find any information about this in the {machine_info['name']} service manual. Please try a different query or select a different machine."
            
            # Format context with page numbers
            context = self._format_context(chunks)
            
            # Generate system prompt
            system_prompt = self._create_system_prompt()
            
            # Generate response
            response = await self.response_generator.generate_response(
                system_prompt=system_prompt,
                user_query=query,
                context=context,
                machine_info=machine_info
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"An error occurred while processing your query: {e}"
    
    async def _retrieve_chunks(self, 
                             query_embedding: List[float],
                             machine_id: str,
                             top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from the vector store for a specific machine.
        
        Args:
            query_embedding: Embedding vector of the query
            machine_id: ID of the machine to query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of chunks with metadata and similarity scores
        """
        # First get machine info to have access to the correct machine_id in the database
        try:
            machine_info = await self.vector_store.get_machine_info(machine_id)
            if machine_info:
                logger.info(f"Found machine info: {machine_info}")
                db_machine_id = machine_info.get('id')  # The ID in our database
            else:
                logger.warning(f"No machine info found for machine_id {machine_id}")
                db_machine_id = machine_id
        except Exception as e:
            logger.warning(f"Error getting machine info: {e}")
            db_machine_id = machine_id

        # DIRECT APPROACH: Try to find the document with an ID matching the machine ID (uppercase)
        document_id = machine_id.upper()  # Convert sc500 to SC500
        logger.info(f"Looking for document with ID: {document_id}")
        
        try:
            # Check if the document exists
            doc_check = self.vector_store.supabase.table("documents").select("*").eq("id", document_id).execute()
            
            if hasattr(doc_check, "data") and doc_check.data:
                logger.info(f"Found document with ID {document_id}: {doc_check.data[0]}")
                
                # Use match_chunks_by_document function
                response = self.vector_store.supabase.rpc(
                    "match_chunks_by_document",
                    {
                        "query_embedding": query_embedding,
                        "input_document_id": document_id,
                        "match_threshold": 0.0,
                        "match_count": top_k
                    }
                ).execute()
                
                if hasattr(response, "data") and response.data and response.data:
                    chunks = response.data
                    logger.info(f"Retrieved {len(chunks)} chunks for document {document_id}")
                    
                    # Print debug info
                    for i, result in enumerate(chunks):
                        similarity = result.get('similarity', 'unknown')
                        page_display = result.get('page_display', result.get('page_numbers', 'unknown'))
                        logger.info(f"Result {i+1}: Pages {page_display}, Similarity: {similarity}")
                    
                    return chunks
                else:
                    logger.warning(f"No chunks found for document {document_id}")
            else:
                logger.warning(f"No document found with ID {document_id}")
        except Exception as e:
            logger.error(f"Error in direct document lookup: {e}")
            
        # Attempt to find document by querying documents table directly using known values
        try:
            # First try to get all documents to see what's available
            all_docs = self.vector_store.supabase.table("documents").select("*").execute()
            if hasattr(all_docs, "data") and all_docs.data:
                logger.info(f"Available documents: {[doc['id'] for doc in all_docs.data]}")
                logger.info(f"Machine IDs in docs: {[doc.get('machine_id', 'unknown') for doc in all_docs.data]}")
                
                # Try to find a document that matches our machine in some way
                document_id = None
                
                # First, try exact match on document id with machine_id uppercased (most likely case)
                matching_docs = [doc for doc in all_docs.data if doc['id'] == machine_id.upper()]
                if matching_docs:
                    document_id = matching_docs[0]['id']
                    logger.info(f"Found document by direct ID match: {document_id}")
                
                # If that didn't work, check if any document has our machine_id in the machine_id column
                if not document_id:
                    matching_docs = [doc for doc in all_docs.data if str(doc.get('machine_id', '')) == machine_id]
                    if matching_docs:
                        document_id = matching_docs[0]['id']
                        logger.info(f"Found document with matching machine_id column: {document_id}")
                
                # If we found a document, try to get chunks for it
                if document_id:
                    response = self.vector_store.supabase.rpc(
                        "match_chunks_by_document",
                        {
                            "query_embedding": query_embedding,
                            "input_document_id": document_id,
                            "match_threshold": 0.0,
                            "match_count": top_k
                        }
                    ).execute()
                    
                    if hasattr(response, "data") and response.data:
                        chunks = response.data
                        if chunks:
                            logger.info(f"Retrieved {len(chunks)} chunks for document {document_id}")
                            return chunks
        except Exception as e:
            logger.error(f"Error in document search: {e}")
        
        # If original document_id failed and no document was found by machine_id, try querying by the numeric machine_id
        # from the machine_info if it exists
        try:
            if machine_info and 'id' not in machine_info:
                # Check for any other fields that might be the real database machine_id
                possible_ids = []
                for k, v in machine_info.items():
                    if isinstance(v, (str, int)) and str(v).isdigit():
                        possible_ids.append(str(v))
                
                logger.info(f"Trying possible numeric machine IDs from machine_info: {possible_ids}")
                
                for possible_id in possible_ids:
                    doc_response = self.vector_store.supabase.table("documents").select("*").eq("machine_id", possible_id).execute()
                    
                    if hasattr(doc_response, "data") and doc_response.data:
                        logger.info(f"Found document with machine_id {possible_id}: {doc_response.data[0]}")
                        
                        document_id = doc_response.data[0]["id"]
                        response = self.vector_store.supabase.rpc(
                            "match_chunks_by_document",
                            {
                                "query_embedding": query_embedding,
                                "input_document_id": document_id,
                                "match_threshold": 0.0,
                                "match_count": top_k
                            }
                        ).execute()
                        
                        if hasattr(response, "data") and response.data:
                            chunks = response.data
                            if chunks:
                                logger.info(f"Retrieved {len(chunks)} chunks using numeric machine_id {possible_id}")
                                return chunks
        except Exception as e:
            logger.error(f"Error looking up documents by numeric machine_id: {e}")

        # BRUTE FORCE: Try all available documents one by one as a last resort
        try:
            all_docs = self.vector_store.supabase.table("documents").select("*").execute()
            if hasattr(all_docs, "data") and all_docs.data:
                logger.info(f"Trying brute force search through all {len(all_docs.data)} documents")
                
                for doc in all_docs.data:
                    document_id = doc["id"]
                    logger.info(f"Trying document {document_id}")
                    
                    # Use match_chunks_by_document function
                    response = self.vector_store.supabase.rpc(
                        "match_chunks_by_document",
                        {
                            "query_embedding": query_embedding,
                            "input_document_id": document_id,
                            "match_threshold": 0.0,
                            "match_count": top_k
                        }
                    ).execute()
                    
                    if hasattr(response, "data") and response.data and len(response.data) > 0:
                        # For each document, check if any chunks have good similarity
                        chunks = response.data
                        logger.info(f"Found {len(chunks)} chunks for document {document_id}")
                        
                        # Get best similarity score
                        best_similarity = 0
                        for chunk in chunks:
                            if 'similarity' in chunk and chunk['similarity'] > best_similarity:
                                best_similarity = chunk['similarity']
                        
                        # If we have any decent matches, return them
                        if best_similarity > 0.5:  # Threshold for "good enough" match
                            logger.info(f"Found good matches (similarity {best_similarity}) in document {document_id}")
                            return chunks
                        else:
                            logger.info(f"Matches in {document_id} not good enough (max similarity: {best_similarity})")
        except Exception as e:
            logger.error(f"Error in brute force document search: {e}")
        
        # Last resort: Try general similarity search
        try:
            logger.info("Trying general similarity search as final fallback")
            chunks = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                limit=top_k,
                similarity_threshold=0.0
            )
            
            if chunks:
                logger.info(f"General search found {len(chunks)} chunks")
                return chunks
        except Exception as e:
            logger.error(f"Final fallback failed: {e}")
        
        logger.error(f"All retrieval methods failed for machine {machine_id}")
        return []
    
    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks as context with page numbers.
        
        Args:
            chunks: List of chunks with metadata
            
        Returns:
            Formatted context string with page numbers
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks):
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
    
    def _create_system_prompt(self) -> str:
        """
        Create the system prompt for GPT-4o with enhanced page citation requirements.
        
        Returns:
            System prompt instructing the model to emphasize page citations
        """
        return """
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
    
    async def close(self):
        """Close all clients."""
        await self.embedding_generator.close()
        await self.response_generator.close()

async def initialize_rag_system():
    """Initialize and return a RAG system instance."""
    return RAGSystem()

if __name__ == "__main__":
    # Example usage
    
    async def test_rag():
        rag_system = await initialize_rag_system()
        
        try:
            # Test with a sample query for a specific machine
            machine_id = "sc50"  # Example machine ID
            query = "What does error code E001 mean?"
            response = await rag_system.process_query(query, machine_id)
            
            print(f"Query for machine {machine_id}: {query}")
            print("-" * 50)
            print(f"Response:\n{response}")
            
        finally:
            await rag_system.close()
    
    asyncio.run(test_rag())
