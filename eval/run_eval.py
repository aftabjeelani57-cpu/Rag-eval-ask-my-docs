import json
from typing import List, Dict
from src.chunking import process_documents
from src.retriever import HybridRetriever
from src.reranker import Reranker


def evaluate_retrieval(eval_dataset: List[Dict], retriever: HybridRetriever, reranker: Reranker, top_k: int = 2) -> Dict:
    """Evaluates Recall@K metric across the evaluation dataset."""
    total_queries = len(eval_dataset)
    successful_retrievals = 0

    print(f"\n--- Running Evaluation (Total Queries: {total_queries}, Top K: {top_k}) ---")

    for item in eval_dataset:
        query = item["question"]
        expected_doc = item["expected_doc_id"]

        # 1. Retrieve candidates
        retrieved = retriever.search_hybrid(query, top_k=top_k * 2)

        # 2. Rerank candidates
        reranked = reranker.rerank(query, retrieved, top_n=top_k)

        # 3. Check if expected document ID is among top reranked chunks
        retrieved_doc_ids = [chunk["chunk_id"].split("_")[0] for chunk in reranked]
        hit = expected_doc in retrieved_doc_ids

        if hit:
            successful_retrievals += 1

        status = "PASSED" if hit else "FAILED"
        print(f"[{status}] Query: '{query}' -> Expected: {expected_doc} | Got: {retrieved_doc_ids}")

    recall_at_k = successful_retrievals / total_queries if total_queries > 0 else 0.0
    return {
        "total_queries": total_queries,
        "successful_hits": successful_retrievals,
        "recall_at_k": round(recall_at_k, 4)
    }


if __name__ == "__main__":
    # Load sample documents and chunk them
    with open("data/sample_docs.json", "r") as f:
        docs = json.load(f)

    # Load evaluation dataset
    with open("data/eval_dataset.json", "r") as f:
        eval_data = json.load(f)

    # Index corpus
    chunks = process_documents(docs, chunk_size=150, overlap=30)
    retriever = HybridRetriever()
    retriever.index_chunks(chunks)

    # Initialize Reranker
    reranker = Reranker()

    # Run Benchmark
    metrics = evaluate_retrieval(eval_data, retriever, reranker, top_k=1)
    
    print("\n================ EVALUATION SUMMARY ================")
    print(f"Recall@1 Score : {metrics['recall_at_k'] * 100}%")
    print(f"Passed Queries : {metrics['successful_hits']}/{metrics['total_queries']}")
    print("====================================================")