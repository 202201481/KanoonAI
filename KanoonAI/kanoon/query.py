import os
import argparse
import json
import sys
from typing import List, Dict, Any
import requests
from pinecone import Pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "kanoondb")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_TOP_K = 15  # Increased from 5 to get more comprehensive references

# Initialize the sentence-transformer model that produces 384-dimensional vectors
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions

def get_embedding(text: str) -> List[float]:
    """Generate embedding for a text using sentence-transformers."""
    embedding = model.encode([text])[0]
    return embedding.tolist()  # Convert numpy arrays to lists

def query_pinecone(query_embedding: List[float], top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """Query Pinecone for similar vectors."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    return results["matches"]

def format_context(results: List[Dict[str, Any]]) -> str:
    """Format retrieved documents as context for the LLM."""
    if not results:
        return "No relevant information found."
    
    context = ""
    
    for i, result in enumerate(results):
        metadata = result['metadata']
        source = metadata.get('source', 'Unknown source')
        text = metadata.get('text', 'No text available')
        context += f"Document {i+1} (Source: {source}):\n{text}\n\n"
    
    return context

def generate_with_gemini(query: str, context: str) -> str:
    """Generate a response using Gemini API based on the retrieved context."""
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    prompt = f"""You are a legal assistant specialized in Indian law. Use only the information provided in the following context to answer the query. If the context doesn't contain relevant information to fully answer the query, say so clearly. Never make up legal information.

The context contains multiple references from Indian legal documents. You MUST be COMPREHENSIVE and include ALL relevant references from the provided context, not just one or two.

Context:
{context}

Query: {query}

Your response MUST be structured as follows:
1. Direct answer identifying ALL relevant laws/sections/articles found in the context
2. For EACH relevant provision:
   a. EXACT ORIGINAL WORDING from the legal text (use direct quotes)
   b. Explanation of what this specific provision means in simple, plain language
3. Overall synthesis of how these provisions work together (if multiple provisions are relevant)
4. Brief practical implications

Always include the precise, verbatim wording from the original legal texts before offering explanations. Be thorough and mention ALL relevant provisions from the context."""
    
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }
    
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024
        }
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response_json = response.json()
        
        if response.status_code == 200 and "candidates" in response_json:
            return response_json["candidates"][0]["content"]["parts"][0]["text"]
        else:
            error_message = response_json.get("error", {}).get("message", "Unknown error")
            return f"Error generating response: {error_message}"
    
    except Exception as e:
        return f"Error: {str(e)}"

def search(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """Search for relevant legal information and generate a response."""
    print(f"\nSearching for: {query}")
    print("Retrieving relevant documents...")
    
    # Generate embedding for the query
    query_embedding = get_embedding(query)
    
    # Query Pinecone
    results = query_pinecone(query_embedding, top_k)
    
    # Get context from retrieved documents
    context = format_context(results)
    
    if "No relevant information found" in context:
        return "I couldn't find any relevant information about that in my knowledge base."
    
    print(f"Found {len(results)} relevant documents. Generating response...")
    
    # Generate response
    response = generate_with_gemini(query, context)
    
    return response

def interactive_mode():
    """Run an interactive loop for continuous querying."""
    # Initialize Pinecone connection once at the beginning
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    print(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")
    
    print("\n===== KanoonAI Interactive Legal Assistant =====")
    print("Enter your legal questions below. Press Ctrl+C to exit.")
    print("=================================================\n")
    
    try:
        while True:
            query = input("\n📝 Enter your question: ")
            if not query.strip():
                continue
                
            print("\n⏳ Processing...")
            result = search(query)
            
            # Print a separator line before the response
            print("\n" + "=" * 80 + "\n")
            print(result)
            print("\n" + "=" * 80)
            
    except KeyboardInterrupt:
        print("\n\nExiting KanoonAI. Thank you for using our service!")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for legal information")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Number of results to retrieve")
    parser.add_argument("--raw", action="store_true", help="Display raw retrieval results without LLM generation")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode with continuous querying")
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.query:
        if args.raw:
            # For debugging: show raw retrieved documents
            query_embedding = get_embedding(args.query)
            results = query_pinecone(query_embedding, args.top_k)
            for i, result in enumerate(results):
                print(f"Result {i+1} (Score: {result['score']:.2f}):")
                print(f"Source: {result['metadata'].get('source', 'Unknown')}")
                print(f"Text: {result['metadata'].get('text', 'No text')[:300]}...\n")
        else:
            # Normal RAG operation: search and generate response
            result = search(args.query, args.top_k)
            print(result)
    else:
        print("No query provided. Use --query to specify a query or --interactive for interactive mode.")
        print("Example: python kanoon/query.py --query 'What does Article 21 of the Constitution say?'")
        print("Example: python kanoon/query.py --interactive")