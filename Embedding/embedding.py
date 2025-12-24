import json
import os
import time
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Keys from environment
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "embedding")

if not PINECONE_KEY or not GOOGLE_KEY:
    raise ValueError("Missing API Keys in .env file")

def main():
    print("Initializing...")
    
    # 1. Setup Google
    genai.configure(api_key=GOOGLE_KEY)
    
    # 2. Setup Pinecone
    pc = Pinecone(api_key=PINECONE_KEY)
    
    # Connect to Index
    print(f"Connecting to Pinecone index: {INDEX_NAME}")
    
    # Check if index exists and check dimensions
    existing_indexes = [i.name for i in pc.list_indexes()]
    
    if INDEX_NAME in existing_indexes:
        index = pc.Index(INDEX_NAME)
        stats = index.describe_index_stats()
        current_dim = stats.get('dimension')
        print(f"Current Index Dimensions: {current_dim}")
        
        if current_dim != 768:
            print(f"Dimension Mismatch! Index has {current_dim}, but Google Embedding model produces 768.")
            print("Deleting and recreating index with correct dimensions...")
            pc.delete_index(INDEX_NAME)
            # Wait for deletion
            while INDEX_NAME in [i.name for i in pc.list_indexes()]:
                time.sleep(1)
            
            # Recreate with 768
            from pinecone import ServerlessSpec
            pc.create_index(
                name=INDEX_NAME,
                dimension=768, 
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            print("Index recreated successfully.")
            index = pc.Index(INDEX_NAME)
    else:
        print(f"Index {INDEX_NAME} does not exist. Creating...")
        from pinecone import ServerlessSpec
        pc.create_index(
            name=INDEX_NAME,
            dimension=768, 
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        index = pc.Index(INDEX_NAME)
    
    # 3. Load Data
    # Path: ParentDir -> Chunking -> chunks.json
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, 'Chunking', 'chunks.json')
    
    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        
    print(f"Loaded {len(chunks)} chunks from file.")
    
    # 4. Embed and Upsert
    batch_size = 50 
    vectors = []
    
    print("Starting embedding generation using Google 'models/text-embedding-004'...")
    
    for i, chunk in enumerate(chunks):
        text = chunk['content']
        source = chunk['metadata']['source']
        para_id = chunk['metadata']['para_id']
        
        # Create a unique ID for the vector
        # Sanitize source name for ID usage
        safe_source = source.replace(" ", "_").replace(".txt", "")
        vector_id = f"{safe_source}_{i}"
        
        try:
            # Generate Embedding
            # text-embedding-004 is the recommended model for new projects
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document",
                title=None
            )
            vector_values = result['embedding']
            
            # Prepare Metadata
            # Note: Pinecone metadata values must be strings, numbers, booleans, or lists of strings
            metadata = {
                "text": text,
                "source": source,
                "para_id": para_id
            }
            
            vectors.append({
                "id": vector_id, 
                "values": vector_values, 
                "metadata": metadata
            })
            
            # Progress marker
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{len(chunks)} embeddings...")

            # Upsert Batch
            if len(vectors) >= batch_size:
                print(f"Upserting batch of {len(vectors)} vectors...")
                index.upsert(vectors=vectors)
                vectors = []
                # Sleep briefly to handle rate limits politely
                time.sleep(1)

        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
            time.sleep(2) # Backoff on error
            continue
            
    # Final Upsert for remaining vectors
    if vectors:
        print(f"Upserting final batch of {len(vectors)} vectors...")
        index.upsert(vectors=vectors)
        
    print("------------------------------------------------")
    print("SUCCESS: All chunks processed and upserted to Pinecone.")

if __name__ == "__main__":
    main()
