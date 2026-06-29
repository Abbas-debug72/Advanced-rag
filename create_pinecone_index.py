# create_pinecone_index.py
from dotenv import load_dotenv
load_dotenv()

import os
from pinecone import Pinecone, ServerlessSpec

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "knowledge-brain")
DIMENSIONS = 384  # BGE-small dimension
METRIC = "cosine"
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is missing. Add it to your .env file.")

pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSIONS,
        metric=METRIC,
        spec=ServerlessSpec(cloud=CLOUD, region=REGION)
    )
    print(f"✅ Index '{INDEX_NAME}' created successfully (dim={DIMENSIONS}, {METRIC}).")
    
    # Get the host URL
    index = pc.Index(name=INDEX_NAME)
    print(f"🔗 Index Host: {index.host}")
    print(f"📌 Add this to your .env: PINECONE_INDEX_HOST={index.host}")
else:
    print(f"ℹ️  Index '{INDEX_NAME}' already exists.")