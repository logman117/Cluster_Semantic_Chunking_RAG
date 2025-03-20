# Semantic RAG System

A specialized Retrieval-Augmented Generation (RAG) system for industrial service manuals that provides precise answers with page-specific citations. This system combines advanced semantic chunking with vector similarity search to help technicians quickly find relevant information in large technical documents.

![Manual Assistant](https://via.placeholder.com/800x400?text=Service+Manual+Assistant)

## 🌟 Features

- **Machine-Specific Retrieval**: Target searches to specific equipment models
- **Page-Accurate Citations**: Every answer includes exact page numbers from the source manuals
- **Multi-Page Semantic Chunking**: Intelligently handles content that spans across page boundaries
- **Error Code Lookup**: Enhanced query handling for error codes and troubleshooting
- **Vector Search Optimization**: Precision-tuned embedding similarity for technical content
- **Multi-Interface Access**: CLI, API, and React web interface options

## 🏗️ System Architecture

The system has five main components:

1. **Document Processing Pipeline**
   - Extracts text while preserving page boundaries
   - Creates semantically coherent chunks using clustering algorithms
   - Handles cross-page concepts intelligently

2. **Vector Database**
   - Supabase PostgreSQL with pgvector extension
   - Custom SQL functions for optimized document-specific search
   - Efficient machine-to-document mapping

3. **Embedding Generation**
   - Uses Azure OpenAI text-embedding-3-large model
   - Implements retry logic and rate limiting
   - Batch processing for large documents

4. **Query Processing**
   - Error code detection and query enhancement
   - Hybrid vector + text search capabilities
   - Smart chunking to provide concise context

5. **Response Generation**
   - GPT-4o-powered answers with strict citation requirements
   - Page-specific references for every fact provided
   - Machine-specific context awareness

## 💻 Installation

### Prerequisites

- Python 3.9+
- Azure OpenAI API access
- Supabase account with Vector extension enabled

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/service-rag.git
cd service-rag
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
   - Run the SQL script in `supabase_query.sql` to create the necessary tables and functions
   - Ensure the Vector extension is enabled in your Supabase project

## 🚀 Usage

### Processing Service Manuals

1. Place PDF service manuals in the `data/pdfs` directory
2. If you have an Excel mapping file, place it in the root directory as `Service_manual_L5_relations.xlsx`
3. Run the document processing script:

```bash
python process_manuals.py
```

This will:
- Process all PDFs in the directory
- Create machine entities in the database (if they don't exist)
- Generate semantically coherent chunks
- Create and store embeddings in Supabase

### Interactive Command Line Interface

For direct, machine-specific querying:

```bash
python test_query_improved.py
```

This will:
1. Let you select a machine from the available options
2. Allow you to ask questions about that specific machine
3. Provide answers with page citations from the service manual

### Web Interface

Start the FastAPI backend:
```bash
python api_endpoints.py
```

Then integrate the React component (`service-manual-frontend.tsx`) into your web application.

### API Usage

The system provides a RESTful API:

```bash
# Query about a machine
curl -X POST -H "Content-Type: application/json" \
  -d '{"query": "What does error code 5-120 mean?", "machine_id": "cs7010", "top_k": 5}' \
  http://localhost:8000/query

# Upload a new manual
curl -X POST -F "file=@your_manual.pdf" -F "machine_id=cs7010" \
  http://localhost:8000/documents/upload
```

## ⚙️ Configuration Options

The system can be tuned by modifying these parameters:

- **Chunking Size**: In `cluster_semantic_chunker.py`, modify `max_chunk_size` (default: 200 tokens)
- **Vector Search Threshold**: In DB functions or API calls, adjust `match_threshold` (default: 0.0)
- **Top Results**: Change `top_k` parameter for more or fewer results
- **Error Code Boost**: In `test_query_improved.py`, modify the weighted average value for error code queries

## 🔧 Technologies Used

- **Azure OpenAI**: For embeddings (text-embedding-3-large) and response generation (GPT-4o)
- **Supabase**: Vector database with pgvector for similarity search
- **FastAPI**: Backend API services 
- **React**: Web frontend components
- **PyMuPDF**: PDF text extraction with page awareness
- **Pandas**: For processing service manual mappings

## 📚 Related Papers and Techniques

- This implementation uses an innovative clustering-based semantic chunking approach that extends beyond simple text splitting:
  - Creates embeddings for small text pieces
  - Uses cosine similarity to group semantically related content
  - Maintains optimal chunk sizes while preserving semantic coherence 
  - Implements cross-page tracking for concepts that span multiple pages

## 🤝 Contributing

Contributions to improve the system are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Azure OpenAI for providing embedding and language models
- Supabase for vector database capabilities
- PyMuPDF for PDF processing
- FastAPI for the API framework
- The Retrieval-Augmented Generation research community
