import os
import pinecone
from groq import Groq
from dotenv import load_dotenv
import PyPDF2
import tiktoken

load_dotenv()

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_embedding(text: str):
    """Get embedding using Groq API"""
    response = groq_client.embeddings.create(
        model=os.getenv("GROQ_EMBEDDING_MODEL", "text-embedding-3-small"),
        input=text
    )
    return response.data[0].embedding

def chunk_text(text: str, chunk_size: int = 500):
    """Split text into chunks"""
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def ingest_documents():
    """Ingest all PDFs into Pinecone"""
    # Initialize Pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT")
    )
    
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-chatbot")
    
    # Create index if it doesn't exist
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=1536,  # Groq embedding dimension
            metric="cosine"
        )
    
    index = pinecone.Index(index_name)
    
    # Process all PDFs
    docs_dir = "./documents"
    vectors = []
    
    for filename in os.listdir(docs_dir):
        if filename.endswith('.pdf'):
            print(f"📄 Processing: {filename}")
            
            # Extract text from PDF
            with open(os.path.join(docs_dir, filename), 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            
            # Chunk text
            chunks = chunk_text(text)
            
            # Create embeddings and vectors
            for i, chunk in enumerate(chunks):
                embedding = get_embedding(chunk)
                vectors.append({
                    'id': f"{filename}_{i}",
                    'values': embedding,
                    'metadata': {
                        'text': chunk,
                        'source': filename,
                        'chunk': i
                    }
                })
            
            # Upload in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                index.upsert(vectors=batch)
                print(f"   ✅ Uploaded batch {i//batch_size + 1}")
    
    print(f"✅ Done! Total vectors: {len(vectors)}")

if __name__ == "__main__":
    ingest_documents()