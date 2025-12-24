import os
import json
import google.generativeai as genai
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv

# Load env from parent dir if needed, or assume it's set
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(base_dir, 'Embedding', '.env')
load_dotenv(env_path)

PINECONE_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_NAME = "hybrid-legal-index" 

if not PINECONE_KEY or not GOOGLE_KEY:
    print("Error: API Keys not found. Make sure .env is in Embedding/ folder.")

def get_hybrid_results(query, top_k=5, alpha=0.5):
    """
    Executes a hybrid search.
    alpha: 0.0 = Pure Keyword (Sparse), 1.0 = Pure Semantic (Dense)
    """
    # 1. Setup
    genai.configure(api_key=GOOGLE_KEY)
    pc = Pinecone(api_key=PINECONE_KEY)
    index = pc.Index(INDEX_NAME)
    
    # 2. Load BM25 logic
    bm25_path = os.path.join(base_dir, 'Embedding', 'bm25_params.json')
    if not os.path.exists(bm25_path):
        print("Error: BM25 params not found. Run hybrid_embedding.py first.")
        return []
        
    bm25 = BM25Encoder()
    bm25.load(bm25_path)
    
    # 3. Dense Vector (Google)
    dense_res = genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="retrieval_query"
    )
    dense_vec = dense_res['embedding']
    
    # 4. Sparse Vector (BM25)
    sparse_vec = bm25.encode_queries(query)
    
    # 5. Hybrid Query
    scaled_dense = [v * alpha for v in dense_vec]
    scaled_sparse_values = [v * (1 - alpha) for v in sparse_vec['values']]
    scaled_sparse = {
        'indices': sparse_vec['indices'],
        'values': scaled_sparse_values
    }
    
    results = index.query(
        vector=scaled_dense,
        sparse_vector=scaled_sparse,
        top_k=top_k,
        include_metadata=True
    )
    
    return results['matches']

if __name__ == "__main__":
    print("--- Hybrid RAG Search ---")
    while True:
        q = input("\nEnter Query (or 'exit'): ")
        if q.lower() == 'exit':
            break
            
        print(f"Searching for: '{q}'...")
        matches = get_hybrid_results(q, alpha=0.7) 
        
        print(f"\nFound {len(matches)} results:\n")
        
        for m in matches:
            # Extract metadata
            source = m['metadata'].get('source', 'Unknown')
            para = m['metadata'].get('para_id', 'N/A')
            text = m['metadata'].get('text', '')
            score = m['score']
            
            # Requested Format
            print(f"1. PDF Name : {source}")
            print(f"2. Para Number : {para}")
            print(f"   (Score: {score:.4f})")
            print("-" * 30)
