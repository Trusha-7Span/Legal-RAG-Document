import json
import os
import time
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv

# Load params
load_dotenv()
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_NAME = "hybrid-legal-index"

if not PINECONE_KEY or not GOOGLE_KEY:
    raise ValueError("Missing API Keys")

def main():
    print("Initializing Enhanced Hybrid Embedding Process...")
    
    # 1. Setup Clients
    genai.configure(api_key=GOOGLE_KEY)
    pc = Pinecone(api_key=PINECONE_KEY)
    
    # 2. Check Index
    existing_indexes = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
         # Create if not exists (same logic as before)
         pc.create_index(
            name=INDEX_NAME,
            dimension=768,
            metric='dotproduct',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    index = pc.Index(INDEX_NAME)
    
    # 3. Load Data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, 'Chunking', 'chunks.json')
    
    with open(json_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks.")
    
    # 4. Prepare 'Contextualized' Corpus
    # We prepend the source name to the content so BM25 and Dense Embeddings "see" the filename relevance.
    print("Contextualizing chunks with source metadata...")
    contextualized_chunks = []
    for c in chunks:
        # Format: "Source: [Filename] | [Original Content]"
        # This ensures 'Auda' in the query matches 'Auda' in the vector content.
        enhanced_text = f"Source: {c['metadata']['source']} | {c['content']}"
        contextualized_chunks.append(enhanced_text)

    # 5. Train BM25 on Enhanced Text
    print("Training BM25 on contextualized text...")
    bm25 = BM25Encoder()
    bm25.fit(contextualized_chunks)
    
    bm25_path = os.path.join(base_dir, 'Embedding', 'bm25_params.json')
    bm25.dump(bm25_path)
    print(f"BM25 params saved to {bm25_path}")
    
    # 6. Generate & Upsert
    batch_size = 50
    vectors = []
    
    print("Generating vectors...")
    for i, (chunk, enhanced_text) in enumerate(zip(chunks, contextualized_chunks)):
        source = chunk['metadata']['source']
        para_id = chunk['metadata']['para_id']
        vector_id = f"{source.replace(' ', '_')}_{i}"
        
        try:
            # Dense Vector (Google) - Uses Enhanced Text
            dense_res = genai.embed_content(
                model="models/text-embedding-004",
                content=enhanced_text, # Key Change!
                task_type="retrieval_document"
            )
            dense_values = dense_res['embedding']
            
            # Sparse Vector (BM25) - Uses Enhanced Text
            sparse_values = bm25.encode_documents(enhanced_text) 
            
            # Metadata
            metadata = {
                "text": chunk['content'], # Store original text for display (clean)
                "context_text": enhanced_text, # Store enhanced text for debug
                "source": source,
                "para_id": para_id
            }
            
            vectors.append({
                "id": vector_id,
                "values": dense_values,
                "sparse_values": sparse_values,
                "metadata": metadata
            })
            
            if len(vectors) >= batch_size:
                print(f"Upserting batch {i+1}...")
                index.upsert(vectors=vectors)
                vectors = []
                time.sleep(0.5)
                
        except Exception as e:
            print(f"Error at {i}: {e}")
            
    if vectors:
        index.upsert(vectors=vectors)
        
    print("Success: Hybrid Index Updated with Context Awareness.")

if __name__ == "__main__":
    main()
