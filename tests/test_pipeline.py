import pytest
from src.chunking import chunk_text, process_documents
from src.retriever import HybridRetriever
from src.reranker import Reranker
from src.generator import LLMGenerator


def test_chunk_text():
    sample_text = "Retrieval-Augmented Generation improves accuracy by providing external context."
    chunks = chunk_text(sample_text, chunk_size=30, overlap=10)
    assert len(chunks) > 0
    assert chunks[0] in sample_text


def test_hybrid_retriever():
    docs = [{"id": "doc1", "text": "BM25 keyword search is fast and efficient for exact terms."}]
    chunks = process_documents(docs, chunk_size=100, overlap=20)
    
    retriever = HybridRetriever()
    retriever.index_chunks(chunks)
    
    results = retriever.search_hybrid("keyword search", top_k=1)
    assert len(results) == 1
    assert "BM25" in results[0]["text"]


def test_reranker():
    chunks = [
        {"chunk_id": "c1", "text": "Python is a high-level programming language."},
        {"chunk_id": "c2", "text": "Cross-encoders evaluate query and document pairs jointly for reranking."}
    ]
    reranker = Reranker()
    ranked = reranker.rerank("What is a cross encoder?", chunks, top_n=1)
    assert ranked[0]["chunk_id"] == "c2"


def test_generator():
    generator = LLMGenerator()
    chunks = [{"text": "Grounding anchors LLM responses in real facts."}]
    answer = generator.generate_answer("What does grounding do?", chunks)
    assert "Grounding" in answer