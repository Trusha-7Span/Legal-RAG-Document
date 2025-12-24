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
    try:
        # Try loading from local file if env var not set (development mode)
        # This matches the previous logic of looking in Embedding/.env
        base_dir = os.path.dirname(os.path.abspath(__file__))
        load_dotenv(os.path.join(base_dir, '.env'))
        PINECONE_KEY = os.getenv("PINECONE_API_KEY")
        GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
    except:
        pass

if not PINECONE_KEY or not GOOGLE_KEY:
    raise ValueError("Missing API Keys")

def load_chunks_generator(filepath):
    """
    Generator that parses the chunks.jsonl file line by line.
    This prevents loading the entire dataset into memory (Improvement 3: Scalability).
    """
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

def get_contextualized_text(chunk):
    """
    Helper to format the text for embedding/BM25.
    """
    return f"Source: {chunk['metadata']['source']} | {chunk['content']}"

def main():
    print("Initializing Enhanced Hybrid Embedding Process (Scalable Mode)...")
    
    # 1. Setup Clients
    genai.configure(api_key=GOOGLE_KEY)
    pc = Pinecone(api_key=PINECONE_KEY)
    
    # 2. Check Index
    existing_indexes = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
         pc.create_index(
            name=INDEX_NAME,
            dimension=768,
            metric='dotproduct',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    index = pc.Index(INDEX_NAME)
    
    # 3. Path Setup
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, 'Chunking', 'chunks.json')
    
    # 4. Train BM25 (Pass 1 over data)
    print("Pass 1: Training BM25 on contextualized text...")
    bm25 = BM25Encoder()
    
    # We pass a generator to fit(), so it streams data instead of loading a list
    bm25_corpus_generator = (get_contextualized_text(c) for c in load_chunks_generator(json_path))
    bm25.fit(bm25_corpus_generator)
    
    bm25_path = os.path.join(base_dir, 'Embedding', 'bm25_params.json')
    bm25.dump(bm25_path)
    print(f"BM25 params saved to {bm25_path}")
    
    # 5. Generate & Upsert (Pass 2 over data)
    print("Pass 2: Generating vectors and upserting...")
    
    batch_size = 50
    vectors = []
    
    # Reuse generator for second pass
    chunk_stream = load_chunks_generator(json_path)
    
    for i, chunk in enumerate(chunk_stream):
        enhanced_text = get_contextualized_text(chunk)
        source = chunk['metadata']['source']
        para_id = chunk['metadata']['para_id']
        vector_id = f"{source.replace(' ', '_')}_{i}"
        
        try:
            # Dense Vector (Google)
            dense_res = genai.embed_content(
                model="models/text-embedding-004",
                content=enhanced_text,
                task_type="retrieval_document"
            )
            dense_values = dense_res['embedding']
            
            # Sparse Vector (BM25)
            sparse_values = bm25.encode_documents(enhanced_text) 
            
            # Metadata
            metadata = {
                "text": chunk['content'],
                "context_text": enhanced_text,
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
                print(f"Upserting batch ending at {i}...")
                index.upsert(vectors=vectors)
                vectors = []
                # Slight delay to respect rate limits if needed
                time.sleep(0.5) 
                
        except Exception as e:
            print(f"Error at chunk {i}: {e}")
            
    # Upsert remaining
    if vectors:
        index.upsert(vectors=vectors)
        
    print("Success: Hybrid Index Updated (Memory Optimized).")

if __name__ == "__main__":
    main()
