import os
from typing import List, Dict
from openai import OpenAI

class LLMGenerator:
    def __init__(self, model: str = "gpt-4o-mini"):
        # Uses OPENAI_API_KEY environment variable if available
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Generates a grounded answer using retrieved context chunks."""
        if not context_chunks:
            return "No relevant context found to answer the query."

        context_str = "\n---\n".join([c["text"] for c in context_chunks])
        
        prompt = f"""You are a precise AI assistant. Answer the question using ONLY the context provided below. 
If the answer cannot be found in the context, respond with 'Information not available in context.'

Context:
{context_str}

Question: {query}
Answer:"""

        if self.client:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content
        else:
            # Fallback mock mode if no OpenAI key is set
            return f"[Grounded Context Summary]: {context_chunks[0]['text']}"


if __name__ == "__main__":
    from src.retriever import HybridRetriever
    from src.reranker import Reranker
    from src.chunking import process_documents
    import json

    with open("data/sample_docs.json", "r") as f:
        docs = json.load(f)

    chunks = process_documents(docs, chunk_size=150, overlap=30)
    retriever = HybridRetriever()
    retriever.index_chunks(chunks)
    reranker = Reranker()

    query = "What is RAG and why is grounding important?"
    retrieved = retriever.search_hybrid(query, top_k=3)
    top_chunks = reranker.rerank(query, retrieved, top_n=1)

    generator = LLMGenerator()
    answer = generator.generate_answer(query, top_chunks)

    print(f"Query: '{query}'\n")
    print("Generated Answer:")
    print(answer)