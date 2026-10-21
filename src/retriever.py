import json
from typing import List, Dict
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.chunking import process_documents


class HybridRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name="rag_chunks")
        self.bm25 = None
        self.chunks = []

    def index_chunks(self, chunk_list: List[Dict]):
        """Indexes text chunks into both BM25 and ChromaDB vector store."""
        self.chunks = chunk_list
        texts = [c["text"] for c in chunk_list]
        ids = [c["chunk_id"] for c in chunk_list]

        # 1. Index for BM25 (Sparse)
        tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. Index for ChromaDB (Dense)
        embeddings = self.model.encode(texts).tolist()
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=[{"doc_id": c["doc_id"]} for c in chunk_list]
        )

    def search_hybrid(self, query: str, top_k: int = 2) -> List[Dict]:
        """Performs sparse BM25 search and dense vector search."""
        # BM25 Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]

        # Dense Vector Search
        query_embedding = self.model.encode([query]).tolist()
        dense_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        results = []
        for idx in bm25_top_indices:
            results.append({
                "chunk_id": self.chunks[idx]["chunk_id"],
                "text": self.chunks[idx]["text"],
                "bm25_score": round(float(bm25_scores[idx]), 4)
            })
        return results


if __name__ == "__main__":
    with open("data/sample_docs.json", "r") as f:
        docs = json.load(f)
    
    chunks = process_documents(docs, chunk_size=150, overlap=30)
    
    retriever = HybridRetriever()
    retriever.index_chunks(chunks)
    
    query = "What does BM25 excel at?"
    results = retriever.search_hybrid(query, top_k=1)
    
    print(f"Query: '{query}'")
    print("Retrieved context:", results[0])