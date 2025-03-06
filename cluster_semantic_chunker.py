import os
import re
import fitz  # PyMuPDF
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any, Set
from dataclasses import dataclass
import logging
from tqdm import tqdm
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    page_numbers: List[int]  # Now a list of page numbers that this chunk spans
    document_id: str
    chunk_id: str
    metadata: Dict = None

def openai_token_count(text: str) -> int:
    """Count tokens using tiktoken for OpenAI models."""
    encoder = tiktoken.encoding_for_model("gpt-4")
    return len(encoder.encode(text))

class ClusterSemanticChunker:
    """
    Implements a semantic chunking approach that clusters text pieces based on
    semantic similarity while maintaining size constraints.
    """
    
    def __init__(self, 
                embedding_function: Callable[[str], List[float]],
                max_chunk_size: int = 200,    # Reduced from 400 to 200
                min_chunk_size: int = 50,     # Reduced from 100 to 50
                chunk_overlap: int = 50,      # Added overlap parameter
                length_function: Callable[[str], int] = len):
        """
        Initialize the cluster semantic chunker.
        
        Args:
            embedding_function: Function to convert text to embeddings
            max_chunk_size: Maximum size of a chunk
            min_chunk_size: Minimum size of a chunk
            chunk_overlap: Number of tokens/chars to overlap between chunks
            length_function: Function to calculate text length (tokens, chars, etc.)
        """
        self.embedding_function = embedding_function
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
    
    def split_text(self, text: str, page_boundaries: Dict[int, int] = None) -> List[Tuple[str, List[int]]]:
        """
        Split text into semantically coherent chunks.
        
        Args:
            text: Input text to split
            page_boundaries: Dictionary mapping character positions to page numbers
            
        Returns:
            List of tuples (chunk_text, page_numbers)
        """
        # Step 1: Split text into small fixed-size pieces
        pieces, piece_boundaries = self._split_into_small_pieces(text)
        
        if not pieces:
            return []
        
        # If text is short enough, return as a single chunk
        if self.length_function(text) <= self.max_chunk_size:
            if page_boundaries:
                pages = self._get_pages_for_span(0, len(text), page_boundaries)
                return [(text, pages)]
            else:
                return [(text, [1])]  # Default to page 1 if no boundaries provided
        
        # Step 2: Generate embeddings for all pieces
        embeddings = []
        for i, piece in enumerate(tqdm(pieces, desc="Generating embeddings for pieces")):
            embeddings.append(self.embedding_function(piece))
        
        # Step 3: Calculate similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(embeddings)
        
        # Step 4: Use dynamic programming to find optimal clusters
        chunk_indices = self._find_optimal_clusters(pieces, similarity_matrix)
        
        # If no chunks were formed, use a fallback strategy
        if not chunk_indices:
            logger.warning("No optimal clusters found. Using fallback chunking strategy.")
            chunk_indices = self._fallback_chunking(pieces)
        
        # Step 5: Add overlapping chunks
        if self.chunk_overlap > 0 and len(chunk_indices) > 1:
            overlapped_indices = self._add_chunk_overlap(pieces, chunk_indices)
            chunk_indices = overlapped_indices
        
        # Step 6: Combine pieces into final chunks and determine page ranges
        chunks_with_pages = []
        for indices in chunk_indices:
            # Combine text from pieces
            chunk_text = " ".join([pieces[i] for i in indices])
            
            # Determine page range if boundaries are provided
            if page_boundaries:
                # Find the start and end character positions for this chunk
                start_pos = piece_boundaries[indices[0]][0]
                end_pos = piece_boundaries[indices[-1]][1]
                
                # Get the pages this chunk spans
                pages = self._get_pages_for_span(start_pos, end_pos, page_boundaries)
                chunks_with_pages.append((chunk_text, pages))
            else:
                chunks_with_pages.append((chunk_text, [1]))  # Default to page 1
        
        logger.info(f"Created {len(chunks_with_pages)} chunks from {len(pieces)} pieces")
        return chunks_with_pages
    
    def _add_chunk_overlap(self, pieces: List[str], chunk_indices: List[List[int]]) -> List[List[int]]:
        """
        Add overlap between chunks to ensure context isn't lost at boundaries.
        
        Args:
            pieces: List of text pieces
            chunk_indices: List of lists, where each inner list contains indices of pieces in a chunk
            
        Returns:
            Modified list of chunk indices with overlap
        """
        piece_sizes = [self.length_function(p) for p in pieces]
        overlapped_indices = []
        
        for i, indices in enumerate(chunk_indices):
            # Add the current chunk
            overlapped_indices.append(indices.copy())
            
            # For all except the last chunk, create an overlapping chunk
            if i < len(chunk_indices) - 1:
                # Calculate the size of the current chunk
                current_size = sum(piece_sizes[idx] for idx in indices)
                
                # Add pieces from the next chunk until we reach the overlap size
                overlap_size = 0
                overlap_indices = []
                next_indices = chunk_indices[i+1]
                
                # Add last pieces from current chunk to create context
                num_context_pieces = min(3, len(indices))  # Take at most 3 pieces for context
                overlap_indices.extend(indices[-num_context_pieces:])
                
                # Add initial pieces from next chunk until overlap target is reached
                for next_idx in next_indices:
                    if overlap_size >= self.chunk_overlap:
                        break
                    
                    overlap_indices.append(next_idx)
                    overlap_size += piece_sizes[next_idx]
                
                # If we've created a meaningful overlap, add it to the result
                if len(overlap_indices) > num_context_pieces:
                    overlapped_indices.append(overlap_indices)
        
        return overlapped_indices
    
    def _fallback_chunking(self, pieces: List[str]) -> List[List[int]]:
        """
        Fallback chunking strategy when optimal clustering fails.
        Simply groups consecutive pieces until reaching max size.
        """
        piece_sizes = [self.length_function(p) for p in pieces]
        chunk_indices = []
        current_chunk = []
        current_size = 0
        
        for i, size in enumerate(piece_sizes):
            # If adding this piece would exceed max size, start a new chunk
            if current_size + size > self.max_chunk_size and current_chunk:
                if len(current_chunk) > 0:  # Only add non-empty chunks
                    chunk_indices.append(current_chunk)
                current_chunk = [i]
                current_size = size
            else:
                current_chunk.append(i)
                current_size += size
                
            # If we've accumulated enough for a min chunk, check if we should split
            if current_size >= self.min_chunk_size and i < len(pieces) - 1:
                # Look ahead to next piece
                if current_size + piece_sizes[i+1] > self.max_chunk_size:
                    chunk_indices.append(current_chunk)
                    current_chunk = []
                    current_size = 0
        
        # Add the last chunk if it's not empty
        if current_chunk and current_size >= self.min_chunk_size:
            chunk_indices.append(current_chunk)
        elif current_chunk:  # If it's smaller than min_chunk_size but we have pieces
            if chunk_indices:  # Add to the previous chunk if possible
                chunk_indices[-1].extend(current_chunk)
            else:  # Otherwise create a chunk anyway
                chunk_indices.append(current_chunk)
        
        logger.info(f"Fallback chunking created {len(chunk_indices)} chunks")
        return chunk_indices
    
    def _get_pages_for_span(self, start_pos: int, end_pos: int, page_boundaries: Dict[int, int]) -> List[int]:
        """
        Determine which pages a text span covers based on character positions.
        
        Args:
            start_pos: Starting character position in the document
            end_pos: Ending character position in the document
            page_boundaries: Dictionary mapping character positions to page numbers
            
        Returns:
            List of page numbers that this span covers
        """
        pages = set()
        # Find the page for start position
        for pos, page_num in sorted(page_boundaries.items()):
            if pos > start_pos:
                # Add the previous page (where the span starts)
                pages.add(page_boundaries.get(max(k for k in page_boundaries.keys() if k <= start_pos), 1))
                break
                
        # Find pages between start and end
        for pos, page_num in sorted(page_boundaries.items()):
            if start_pos <= pos <= end_pos:
                pages.add(page_num)
            if pos > end_pos:
                break
        
        # If no pages found, default to page 1
        if not pages:
            pages.add(1)
            
        return sorted(list(pages))
    
    def _split_into_small_pieces(self, text: str, target_size: int = 50) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Split text into small, fixed-size pieces.
        
        Args:
            text: Input text
            target_size: Target size of each piece (in tokens or chars)
            
        Returns:
            Tuple of (list of text pieces, list of character position tuples (start, end))
        """
        # Split by sentences first
        sentence_spans = []
        for match in re.finditer(r'[^.!?]+[.!?](?:\s|$)', text):
            sentence_spans.append((match.start(), match.end()))
        
        if not sentence_spans:
            # If no sentences found, just use the whole text
            return [text], [(0, len(text))]
        
        pieces = []
        piece_boundaries = []
        current_piece = ""
        current_start = sentence_spans[0][0]
        
        for start, end in sentence_spans:
            sentence = text[start:end]
            
            # If adding this sentence would exceed target size, store current piece and start new one
            if self.length_function(current_piece + sentence) > target_size and current_piece:
                pieces.append(current_piece.strip())
                piece_boundaries.append((current_start, start - 1))
                current_piece = sentence
                current_start = start
            else:
                if current_piece:
                    current_piece += " " + sentence
                else:
                    current_piece = sentence
        
        # Add the last piece if it's not empty
        if current_piece.strip():
            pieces.append(current_piece.strip())
            piece_boundaries.append((current_start, sentence_spans[-1][1]))
        
        return pieces, piece_boundaries
    
    def _calculate_similarity_matrix(self, embeddings: List[List[float]]) -> np.ndarray:
        """
        Calculate cosine similarity matrix between all embeddings.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Similarity matrix as numpy array
        """
        # Convert to numpy array for faster calculations
        embeddings_array = np.array(embeddings)
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1e-10
        normalized_embeddings = embeddings_array / norms
        
        # Calculate cosine similarity matrix
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        return similarity_matrix
    
    def _find_optimal_clusters(self, 
                              pieces: List[str], 
                              similarity_matrix: np.ndarray) -> List[List[int]]:
        """
        Use dynamic programming to find optimal clusters.
        
        Args:
            pieces: List of text pieces
            similarity_matrix: Matrix of similarities between pieces
            
        Returns:
            List of lists, where each inner list contains indices of pieces that form a chunk
        """
        n = len(pieces)
        
        # Calculate size of each piece
        piece_sizes = [self.length_function(p) for p in pieces]
        
        # Initialize dynamic programming table - FIXED: create independent lists
        # dp[i] = (best score ending at i, previous split point, cluster indices)
        dp = [(0, -1, []) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            best_score = -float('inf')
            best_prev = -1
            best_cluster = []
            
            # Try different possible previous split points
            for j in range(max(0, i - self.max_chunk_size), i):
                # Calculate total size of this potential chunk
                chunk_size = sum(piece_sizes[j:i])
                
                # Skip if chunk is too large or too small
                if chunk_size > self.max_chunk_size or chunk_size < self.min_chunk_size:
                    continue
                
                # Calculate similarity score for this potential chunk
                chunk_indices = list(range(j, i))
                if len(chunk_indices) <= 1:
                    similarity_score = 0  # No similarity for single-piece chunks
                else:
                    # Calculate average similarity between all pairs in the chunk
                    total_sim = 0
                    pair_count = 0
                    for idx1 in range(len(chunk_indices)):
                        for idx2 in range(idx1 + 1, len(chunk_indices)):
                            total_sim += similarity_matrix[chunk_indices[idx1], chunk_indices[idx2]]
                            pair_count += 1
                    
                    similarity_score = total_sim / max(1, pair_count)
                
                # Calculate total score (previous best score + current chunk score)
                score = dp[j][0] + similarity_score
                
                if score > best_score:
                    best_score = score
                    best_prev = j
                    best_cluster = chunk_indices.copy()  # Make a copy to avoid reference issues
            
            # Update dynamic programming table
            dp[i] = (best_score, best_prev, best_cluster)
        
        # Reconstruct solution
        clusters = []
        i = n
        while i > 0:
            _, prev, cluster = dp[i]
            if not cluster:  # No valid cluster found
                break
            clusters.append(cluster)
            i = prev
        
        # Reverse to get correct order
        clusters.reverse()
        
        logger.info(f"Optimal clustering created {len(clusters)} clusters")
        return clusters

class DocumentProcessor:
    """Processes PDF documents for a RAG system with multi-page semantic chunking."""
    
    def __init__(self, 
                 embedding_function: Callable[[str], List[float]],
                 max_chunk_size: int = 200,    # Reduced from 400 to 200
                 min_chunk_size: int = 50,     # Reduced from 100 to 50
                 chunk_overlap: int = 50):     # Added overlap parameter
        """
        Initialize the document processor.
        
        Args:
            embedding_function: Function to convert text to embeddings
            max_chunk_size: Maximum number of tokens per chunk
            min_chunk_size: Minimum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunker = ClusterSemanticChunker(
            embedding_function=embedding_function,
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=openai_token_count
        )
    
    def process_pdf(self, pdf_path: str, document_id: str) -> List[Chunk]:
        """
        Process a PDF document and return a list of chunks with metadata.
        
        Args:
            pdf_path: Path to the PDF file
            document_id: Unique identifier for the document
            
        Returns:
            List of Chunk objects
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Open the PDF
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Error opening PDF {pdf_path}: {e}")
            raise
        
        if len(doc) == 0:
            logger.warning(f"PDF {pdf_path} has no pages.")
            return []
            
        # Process the entire document, tracking page boundaries
        full_text = ""
        page_boundaries = {}  # Maps character positions to page numbers
        page_texts = []
        
        # First pass: collect all text and page boundaries
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_number = page_idx + 1
            
            # Extract text from the page
            page_text = page.get_text()
            
            # Clean and normalize text
            page_text = self._clean_text(page_text)
            
            if not page_text.strip():
                logger.debug(f"Page {page_number} has no text content.")
                continue
                
            # Record page boundary
            page_start_pos = len(full_text)
            full_text += page_text + " "  # Add a space between pages
            page_end_pos = len(full_text) - 1  # Exclude the added space
            
            # Store the page boundary
            page_boundaries[page_start_pos] = page_number
            
            # Store page text for metadata extraction
            page_texts.append((page_number, page_text))
        
        if not full_text.strip():
            logger.warning(f"PDF {pdf_path} contains no readable text.")
            return []
            
        logger.info(f"Extracted {len(full_text)} characters and {len(page_texts)} text pages from PDF")
        
        # Extract section metadata from each page
        page_metadata = {}
        for page_number, text in page_texts:
            page_metadata[page_number] = self._extract_page_metadata(text, page_number)
        
        # Apply semantic chunking to the entire document
        chunks_with_pages = self.chunker.split_text(full_text, page_boundaries)
        
        # If no chunks were created by the chunker, use a simple page-based chunking
        if not chunks_with_pages:
            logger.warning("Semantic chunking produced no chunks. Falling back to page-based chunking.")
            chunks_with_pages = self._fallback_page_chunking(page_texts)
        
        # Create Chunk objects with page ranges
        all_chunks = []
        for i, (chunk_text, pages) in enumerate(chunks_with_pages):
            # Combine metadata from all pages in the range
            combined_metadata = {"page_range": pages}
            for page in pages:
                if page in page_metadata:
                    # Add page-specific metadata (like section headers)
                    for key, value in page_metadata[page].items():
                        if key != "page_number":
                            combined_metadata[f"page_{page}_{key}"] = value
            
            chunk = Chunk(
                text=chunk_text,
                page_numbers=pages,
                document_id=document_id,
                chunk_id=f"{document_id}_p{min(pages)}-{max(pages)}_c{i+1}",
                metadata=combined_metadata
            )
            all_chunks.append(chunk)
            
        logger.info(f"Completed processing {pdf_path}: created {len(all_chunks)} total chunks")
        return all_chunks
    
    def _fallback_page_chunking(self, page_texts: List[Tuple[int, str]]) -> List[Tuple[str, List[int]]]:
        """
        Simple fallback chunking strategy that treats each page as a chunk.
        """
        chunks = []
        for page_number, text in page_texts:
            if len(text.strip()) > 0:
                if openai_token_count(text) <= self.max_chunk_size:
                    chunks.append((text, [page_number]))
                else:
                    # If a page is too large, split it into parts
                    sentences = re.findall(r'[^.!?]+[.!?](?:\s|$)', text)
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if openai_token_count(current_chunk + sentence) > self.max_chunk_size and current_chunk:
                            chunks.append((current_chunk.strip(), [page_number]))
                            current_chunk = sentence
                        else:
                            current_chunk += " " + sentence if current_chunk else sentence
                    
                    if current_chunk.strip():
                        chunks.append((current_chunk.strip(), [page_number]))
        
        logger.info(f"Fallback page chunking created {len(chunks)} chunks")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and headers/footers (customize as needed)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        return text.strip()
    
    def _extract_page_metadata(self, text: str, page_number: int) -> Dict:
        """
        Extract metadata from page text.
        
        This can be customized to extract headers, section titles, etc.
        """
        metadata = {
            "page_number": page_number,
        }
        
        # Extract section headers (customize regex as needed)
        section_match = re.search(r'^(Chapter|Section)\s+\d+[.:]\s+(.+?)$', 
                                 text, re.MULTILINE)
        if section_match:
            metadata["section"] = section_match.group(2).strip()
        
        return metadata

# Main function to process documents
def process_documents(pdf_dir: str, embedding_function: Callable[[str], List[float]]) -> List[Chunk]:
    """
    Process all PDF documents in a directory.
    
    Args:
        pdf_dir: Directory containing PDF files
        embedding_function: Function to convert text to embeddings
        
    Returns:
        List of all chunks from all documents
    """
    processor = DocumentProcessor(
        embedding_function=embedding_function,
        max_chunk_size=200,  # Reduced from 400 to 200
        min_chunk_size=50,   # Reduced from 100 to 50 
        chunk_overlap=50     # Added 50 token overlap
    )
    all_chunks = []
    
    # Make sure the directory exists
    if not os.path.exists(pdf_dir):
        logger.error(f"Directory not found: {pdf_dir}")
        return []
        
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logger.error(f"No PDF files found in directory: {pdf_dir}")
        return []
        
    logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
    
    # Process each PDF file in the directory
    for filename in pdf_files:
        pdf_path = os.path.join(pdf_dir, filename)
        document_id = os.path.splitext(filename)[0]
        
        # Process the PDF
        chunks = processor.process_pdf(pdf_path, document_id)
        all_chunks.extend(chunks)
    
    return all_chunks

# This makes the functions and classes accessible when importing
__all__ = ['Chunk', 'process_documents', 'ClusterSemanticChunker', 'DocumentProcessor', 'openai_token_count']