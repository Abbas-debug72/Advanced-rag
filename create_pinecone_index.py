# create_pinecone_index.py
from dotenv import load_dotenv
load_dotenv()

import os
from pinecone import Pinecone, ServerlessSpec

# --- Configuration (read from .env or set defaults) ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "knowledge-brain-groq")  # Updated default name
DIMENSIONS = 1536  # Updated for Groq embeddings (text-embedding-3-small)
METRIC = "cosine"
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is missing. Add it to your .env file.")

# --- Connect to Pinecone ---
pc = Pinecone(api_key=PINECONE_API_KEY)

# --- Create index if it doesn't exist ---
existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSIONS,
        metric=METRIC,
        spec=ServerlessSpec(
            cloud=CLOUD,
            region=REGION
        )
    )
    print(f"✅ Index '{INDEX_NAME}' created successfully (dim={DIMENSIONS}, metric={METRIC}).")
    print(f"   Cloud: {CLOUD}, Region: {REGION}")
else:
    print(f"⚠️  Index '{INDEX_NAME}' already exists. No action taken.")
    print(f"   Existing index details will be used.")

# --- Display index information ---
try:
    index = pc.Index(host=INDEX_NAME)
    stats = index.describe_index_stats()
    print(f"\n📊 Index Stats:")
    print(f"   Total vectors: {stats.get('total_vector_count', 0)}")
    print(f"   Dimension: {stats.get('dimension', 'N/A')}")
    print(f"   Metric: {stats.get('metric', 'N/A')}")
except Exception as e:
    print(f"⚠️  Could not fetch index stats: {e}")