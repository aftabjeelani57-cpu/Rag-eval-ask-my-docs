import json
from typing import List, Dict

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """Splits raw text into overlapping chunks based on character length."""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
        
    return chunks

def process_documents(documents: List[Dict], chunk_size: int = 200, overlap: int = 50) -> List[Dict]:
    """Processes a list of raw documents into chunked records with IDs."""
    chunked_docs = []
    for doc in documents:
        raw_chunks = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for idx, chunk in enumerate(raw_chunks):
            chunked_docs.append({
                "chunk_id": f"{doc['id']}_c{idx}",
                "doc_id": doc["id"],
                "text": chunk
            })
    return chunked_docs

if __name__ == "__main__":
    with open("data/sample_docs.json", "r") as f:
        docs = json.load(f)
    processed = process_documents(docs, chunk_size=150, overlap=30)
    print(f"Generated {len(processed)} chunks from {len(docs)} documents.\n")
    for chunk in processed[:3]:
        print(chunk)