# ingest_local.py - Run this locally to ingest documents
import os
import json
from dotenv import load_dotenv
from brain import KnowledgeBrain

load_dotenv()

def main():
    print("📚 Starting document ingestion...")
    print(f"   PDF Directory: {os.getenv('PDF_DIRECTORY', './pdfs')}")
    print(f"   Pinecone Index: {os.getenv('PINECONE_INDEX_NAME', 'knowledge-brain')}")
    print(f"   Embedding Model: {os.getenv('EMBEDDING_MODEL', 'multilingual-e5-large')}")
    print()
    
    # Optional: Delete metadata to force re-ingestion
    metadata_file = "brain_metadata.json"
    if os.path.exists(metadata_file):
        response = input(f"🗑️  Delete existing metadata file ({metadata_file})? (y/N): ")
        if response.lower() == 'y':
            os.remove(metadata_file)
            print("✅ Metadata deleted")
    
    brain = KnowledgeBrain(pdf_directory=os.getenv("PDF_DIRECTORY", "./pdfs"))
    results = brain.ingest_all_pdfs()
    
    print("\n" + "=" * 50)
    print("📊 INGESTION RESULTS:")
    print(f"   Total PDFs found: {results['total']}")
    print(f"   Processed: {results['processed']}")
    print(f"   Skipped: {results['skipped']}")
    print(f"   Failed: {results['failed']}")
    print("=" * 50)
    
    stats = brain.get_stats()
    print(f"\n📊 Knowledge Base Stats:")
    print(f"   Total Documents: {stats['total_documents']}")
    print(f"   Total Chunks: {stats['total_chunks']}")
    print(f"   Total Pages: {stats['total_pages']}")
    print()

if __name__ == "__main__":
    main()