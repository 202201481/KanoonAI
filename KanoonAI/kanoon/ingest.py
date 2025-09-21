import os
import argparse
import glob
from typing import List, Dict, Any
import PyPDF2
from pinecone import Pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Initialize the sentence-transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def read_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def read_text_file(file_path: str) -> str:
    """Read text from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using sentence-transformers."""
    embeddings = model.encode(texts)
    return embeddings.tolist()  # Convert numpy arrays to lists

def ingest_folder(folder_path: str) -> None:
    """Process all PDF and text files in a folder and store in Pinecone."""
    # Find all PDF and text files
    pdf_files = glob.glob(os.path.join(folder_path, "**/*.pdf"), recursive=True)
    txt_files = glob.glob(os.path.join(folder_path, "**/*.txt"), recursive=True)
    all_files = pdf_files + txt_files
    
    if not all_files:
        print(f"No PDF or text files found in {folder_path}")
        return
    
    print(f"Found {len(all_files)} files to process")
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Connect to the existing index
    print(f"Connecting to existing Pinecone index: {PINECONE_INDEX_NAME}")
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # Process each file
    for file_path in all_files:
        print(f"Processing {file_path}")
        
        # Read file content
        if file_path.lower().endswith('.pdf'):
            text = read_pdf(file_path)
        else:
            text = read_text_file(file_path)
        
        # Skip empty files
        if not text.strip():
            print(f"  Skipping empty file: {file_path}")
            continue
        
        # Chunk the text
        chunks = chunk_text(text)
        print(f"  Created {len(chunks)} chunks")
        
        # Process in batches of 100 (Pinecone limitation)
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            
            # Generate vectors
            batch_embeddings = get_embeddings(batch_chunks)
            
            # Prepare records for Pinecone
            records = []
            for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                record_id = f"{os.path.basename(file_path)}_chunk_{i+j}"
                records.append({
                    "id": record_id,
                    "values": embedding,
                    "metadata": {
                        "source": file_path,
                        "text": chunk
                    }
                })
            
            # Upsert to Pinecone
            index.upsert(vectors=records)
            print(f"  Uploaded batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
    
    print("Ingestion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Pinecone")
    parser.add_argument("--source", required=True, help="Path to folder with documents")
    args = parser.parse_args()
    
    ingest_folder(args.source)