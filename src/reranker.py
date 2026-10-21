from typing import List, Dict
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: List[Dict], top_n: int = 2) -> List[Dict]:
        """Re-scores retrieved candidate chunks using a Cross-Encoder."""
        if not chunks:
            return []

        pairs = [[query, chunk["text"]] for chunk in chunks]
        scores = self.model.predict(pairs)

        for idx, score in enumerate(scores):
            chunks[idx]["rerank_score"] = round(float(score), 4)

        ranked_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return ranked_chunks[:top_n]


if __name__ == "__main__":
    from src.retriever import HybridRetriever
    from src.chunking import process_documents
    import json

    with open("data/sample_docs.json", "r") as f:
        docs = json.load(f)

    chunks = process_documents(docs, chunk_size=150, overlap=30)
    retriever = HybridRetriever()
    retriever.index_chunks(chunks)

    query = "How does reranking reduce hallucinations?"
    retrieved_chunks = retriever.search_hybrid(query, top_k=3)

    reranker = Reranker()
    top_ranked = reranker.rerank(query, retrieved_chunks, top_n=1)

    print(f"Query: '{query}'")
    print("Top Reranked Chunk:", top_ranked[0])