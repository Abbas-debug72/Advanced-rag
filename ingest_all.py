# ingest_all.py
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import time
from brain import KnowledgeBrain

def main():
    print("🧠 Building Pinecone Knowledge Brain with Free Embeddings\n")
    print("📥 Using BAAI/bge-small-en-v1.5 (384-dim) - Free, local embeddings")
    print()
    
    pdf_dir = os.getenv("PDF_DIRECTORY", "./pdfs")
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        print(f"Created {pdf_dir}. Add PDFs and run again.")
        sys.exit(0)

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDFs found in {pdf_dir}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDFs\n")
    start = time.time()
    brain = KnowledgeBrain(pdf_directory=pdf_dir)
    results = brain.ingest_all_pdfs()
    elapsed = time.time() - start

    print("\n✅ Done!")
    print(f"Processed: {results['processed']}, Skipped: {results['skipped']}, Failed: {results['failed']}")
    print(f"Time: {elapsed:.1f}s")
    stats = brain.get_stats()
    print(f"Brain: {stats['total_documents']} docs, {stats['total_chunks']} chunks")
    print("\n🚀 Run 'python app.py' to start the chatbot")

if __name__ == "__main__":
    main()