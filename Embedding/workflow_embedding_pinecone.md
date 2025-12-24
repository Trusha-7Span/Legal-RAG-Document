# Workflow: Embedding and Pinecone Ingestion

This workflow outlines the steps to convert text chunks into vector embeddings and store them in a Pinecone vector database for retrieval.

## 1. Prerequisites
Before running the scripts, ensure you have:
-   **Pinecone API Key**: Sign up at [pinecone.io](https://www.pinecone.io/).
-   **Embedding Provider**:
    -   **Option A (Recommended)**: OpenAI API Key (for `text-embedding-3-small` or `text-embedding-ada-002`).
    -   **Option B**: Google Gemini API Key (for `models/embedding-001`).
    -   **Option C**: Local (HuggingFace `sentence-transformers`) - Free but slower/requires CPU/GPU.

## 2. Environment Setup
Install the necessary Python libraries.
```bash
pip install pinecone-client openai python-dotenv
```

## 3. Implementation Plan

### Step 3.1: Create `embedding.py`
We will create a Python script that performs the following:
1.  **Load Data**: Read the `Chunking/chunks.json` file.
2.  **Initialize Pinecone**: Connect to your Pinecone index.
3.  **Generate Embeddings**: Loop through the chunks and generate vector embeddings for each using the chosen provider.
4.  **Upsert**: Upload the vectors (ID, Embedding, Metadata) to Pinecone in batches to handle large datasets efficiently.

### Step 3.2: Pinecone Index Configuration
*   **Dimensions**: Must match the embedding model (e.g., 1536 for OpenAI `text-embedding-3-small`).
*   **Metric**: Cosine Similarity.
*   **Metadata Config**: Ensure `source` and `para_id` are indexed for filtering.

## 4. Script Logic (Draft)

```python
import json
import os
from pinecone import Pinecone
from openai import OpenAI

# Config
PINECONE_API_KEY = "your-key"
OPENAI_API_KEY = "your-key"
INDEX_NAME = "legal-rag"

def main():
    # 1. Load Chunks
    with open("Chunking/chunks.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Init Clients
    pc = Pinecone(api_key=PINECONE_API_KEY)
    client = OpenAI(api_key=OPENAI_API_KEY)
    index = pc.Index(INDEX_NAME)

    # 3. Process & Upsert
    batch_size = 100
    vectors = []
    
    for i, item in enumerate(data):
        # Generate ID
        chunk_id = f"{item['metadata']['source']}_{i}"
        
        # Embed
        response = client.embeddings.create(input=item['content'], model="text-embedding-3-small")
        embedding = response.data[0].embedding
        
        # Prepare Metadata
        metadata = {
            "text": item['content'],
            "source": item['metadata']['source'],
            "para_id": item['metadata']['para_id']
        }
        
        vectors.append((chunk_id, embedding, metadata))
        
        # Batch Upsert
        if len(vectors) >= batch_size:
            index.upsert(vectors)
            vectors = []
            
    # Final Upsert
    if vectors:
        index.upsert(vectors)

if __name__ == "__main__":
    main()
```

## 5. Verification
After running the script, we will verify:
1.  **Vector Count**: Matches the number of chunks in the JSON.
2.  **Query Test**: Perform a simple similarity search to ensure relevant results are returned.
