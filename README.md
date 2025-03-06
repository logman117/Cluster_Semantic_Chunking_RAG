# Semantic PDF RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system designed to semantically process, index, and query PDF documents using state-of-the-art language models. This system leverages advanced semantic chunking techniques and vector similarity search to provide precise answers from your document collection with accurate page citations.

![RAG System Overview](https://via.placeholder.com/800x400?text=RAG+System+Overview)

## Features

- **Advanced Semantic Chunking**: Uses a clustering-based approach to create semantically coherent chunks that span multiple pages when conceptually related
- **Vector Search**: High-performance vector similarity search using Supabase Vector Store
- **Accurate Page Citations**: Automatically tracks and cites the specific pages where information is found
- **Azure OpenAI Integration**: Leverages Azure OpenAI API for embeddings and response generation
- **PDF Processing**: Reliable extraction of text content from PDF files with page boundary tracking
- **API Interface**: FastAPI-based REST API for easy integration
- **Interactive CLI**: Command-line interface for testing and direct interaction

## Architecture

The system consists of several key components:

1. **PDF Processing Pipeline**
   - Extracts text while preserving page information
   - Processes document structure and metadata

2. **Semantic Chunking Engine**
   - Splits documents intelligently using semantic clustering
   - Maintains multi-page chunks when concepts span page boundaries
   - Preserves original page citations for each chunk

3. **Vector Database Integration**
   - Generates and stores embeddings using Azure OpenAI text-embedding-3-large
   - Uses Supabase vector store for efficient similarity search

4. **Query Processing**
   - Generates embeddings for user queries
   - Retrieves relevant document chunks with page citations
   - Formats context for the language model

5. **Response Generation**
   - Uses GPT-4o to create accurate, helpful responses
   - Includes precise page citations in responses
   - Summarizes and presents information clearly

## System Components

### 1. Cluster Semantic Chunker

The `ClusterSemanticChunker` implements an innovative approach to document splitting:

- Breaks documents into semantically coherent chunks
- Uses embeddings to determine content similarity
- Maintains optimal chunk sizes for the RAG system
- Tracks page boundaries to provide accurate citations
- Implements chunk overlap to preserve context

### 2. Embedding Generation

The `EmbeddingGenerator` provides:

- Asynchronous embedding generation for optimal performance
- Rate limiting and retry logic for API stability
- Batch processing to handle large document collections
- Configurable embedding dimensions and models

### 3. Supabase Vector Store

The `SupabaseVectorStore` manages:

- Storage of document metadata
- Vector embeddings for similarity search
- Custom SQL functions for efficient queries
- Metadata management for documents and chunks

### 4. RAG System

The core `RAGSystem` orchestrates:

- Query processing and embedding generation
- Context retrieval and formatting
- Response generation with citations
- Integration of all system components

### 5. API Endpoints

The `FastAPI` application provides:

- Document upload and processing endpoints
- Query processing endpoint
- Health check and system status

## Installation

### Prerequisites

- Python 3.9+
- Azure OpenAI API access
- Supabase account with Vector extension enabled

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/semantic-pdf-rag.git
cd semantic-pdf-rag
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your credentials:
```
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_SERVICE_KEY=your-service-key
```

5. Set up Supabase database:
   - Run the SQL script in `supabase query.sql` to create the necessary tables and functions
   - Enable the Vector extension in your Supabase project

## Usage

### Processing Documents

1. Place PDF files in the `data/pdfs` directory.

2. Run the embedding generation script:
```bash
python embedding_generation.py
```

This will:
- Process all PDFs in the directory
- Generate semantic chunks
- Create and store embeddings
- Update the vector database

### Querying the System

#### Using the API

1. Start the API server:
```bash
python api_endpoints.py
```

2. Upload a document:
```bash
curl -X POST -F "file=@your_document.pdf" http://localhost:8000/documents/upload
```

3. Query the system:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"query": "What does error code E001 mean?", "top_k": 5}' \
  http://localhost:8000/query
```

#### Using the Interactive CLI

For testing and direct interaction:

```bash
python test_query_improved.py
```

This will start an interactive session where you can type queries and view responses.

## Configuration Options

The system can be configured by modifying parameters in the initialization of various components:

- **Chunking parameters**: Adjust chunk sizes in `DocumentProcessor` initialization
- **Embedding dimensions**: Configure in `EmbeddingGenerator`
- **Similarity thresholds**: Modify in `similarity_search` methods
- **API rate limits**: Adjust retry and batch parameters in `generate_embeddings_batch`

## Contributing

Contributions to improve the system are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Azure OpenAI for providing the embedding and language models
- Supabase for the vector database capabilities
- PyMuPDF for PDF processing
- FastAPI for the API server framework
