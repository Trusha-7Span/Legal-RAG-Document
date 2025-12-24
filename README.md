# Legal RAG Document System

This project implements a Retrieval-Augmented Generation (RAG) system specialized for legal documents. It features a complete pipeline for processing text data, generating hybrid embeddings (Dense + BM25), and performing accurate context retrieval.

## Project Structure

The project is organized into modular directories representing each stage of the RAG pipeline:

```
Legal-RAG-Document/
├── Chunking/               # Data processing and segmentation
│   ├── chunking.py         # Main script to chunk text files
│   ├── chunks.json         # Generated output chunks
│   └── chunking_strategy.md # Documentation on the chunking logic
├── Embedding/              # Vector generation and indexing
│   ├── hybrid_embedding.py # Generates Dense & Sparse vectors
│   ├── embedding.py        # Standard embedding logic
│   ├── bm25_params.json    # Saved BM25 model parameters
│   └── .env                # API keys and configuration
├── RAG/                    # Retrieval logic
│   └── retrieve_hybrid.py  # Script to query the RAG system
├── Held Section/           # Source input text files
└── PDF/                    # Source PDF documents
```

## Setup & specific requirements

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Trusha-7Span/Legal-RAG-Document.git
    cd Legal-RAG-Document
    ```

2.  **Environment Variables:**
    Create a `.env` file in the `Embedding/` directory (or root, depending on script config) with your API keys:
    ```env
    PINECONE_API_KEY=your_key_here
    OPENAI_API_KEY=your_key_here
    # or GEMINI_API_KEY depending on the model used
    ```

3.  **Dependencies:**
    Ensure you have Python installed. Install necessary libraries (adjust based on your actual imports):
    ```bash
    pip install pinecone-client langchain openai tiktoken rank_bm25
    ```

## Usage Pipeline

### 1. Data Chunking
Process source text files from `Held Section/` into structured JSON chunks.
```bash
cd Chunking
python chunking.py
```
*   **Input:** Text files in `../Held Section/`
*   **Output:** `chunks.json`

### 2. Embedding & Indexing
Generate embeddings (Hybrid: Dense + Sparse) and upload them to the vector database (Pinecone).
```bash
cd Embedding
python hybrid_embedding.py
```
*   **Input:** `../Chunking/chunks.json`
*   **Action:** Creates embeddings and upserts to Pinecone index.

### 3. Retrieval
Query the system to retrieve relevant document contexts.
```bash
cd RAG
python retrieve_hybrid.py
```
*   **Action:** Performs a hybrid search to find the most relevant chunks for your query.

## Features
*   **Metadata-Augmented Chunking:** Preserves context like "PDF Name" and "Paragraph Number" for better traceability.
*   **Hybrid Search:** Combines semantic search (embeddings) with keyword search (BM25) for higher accuracy.
*   **Legal Focus:** Tailored for processing "Held Section" legal texts.
