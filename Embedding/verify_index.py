import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Explicitly load from local .env
load_dotenv()

api_key = os.getenv('PINECONE_API_KEY')
index_name = os.getenv('PINECONE_INDEX_NAME')

print(f"Checking index '{index_name}'...")

try:
    pc = Pinecone(api_key=api_key)
    idx = pc.Index(index_name)
    stats = idx.describe_index_stats()
    print("Index Stats:")
    print(stats)
except Exception as e:
    print(f"Error: {e}")
