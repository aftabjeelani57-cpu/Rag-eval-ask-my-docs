# RAG Eval

> A production-grade Retrieval-Augmented Generation (RAG) system with a built-in evaluation harness to quantitatively measure and optimize chunking, retrieval recall, and answer faithfulness.

---

## 💡 Why This Project?

Most RAG implementations fail in production because chunking and retrieval parameters are selected arbitrarily. **`Rag-eval-ask-my-docs`** bridges the gap between basic RAG scripts and production-ready systems by pairing hybrid search with an empirical benchmarking harness—allowing data-driven optimization of context quality.

---

## 🏗️ Architecture Overview

> **Documents** ──> **Chunking Strategy** ──> **Vector Store**  
> **Query** ──> **Hybrid Search (BM25 + Dense)**  
> └──> **Cross-Encoder Reranker** ──> **LLM Generation** ──> **Evaluation Harness**

---

## ✨ Core Features

* **Hybrid Retrieval:** Blends sparse keyword search (BM25) with dense vector embeddings to capture both exact terminology and semantic intent.
* **Two-Stage Reranking:** Applies a cross-encoder model to re-score candidates before passing context to the LLM, reducing noise and hallucination risks.
* **Empirical Evaluation Harness:** Tracks key metrics against a labeled golden dataset:
  * **Retrieval Recall@K:** Evaluates whether ground-truth context is successfully retrieved.
  * **Answer Faithfulness:** Verifies that generated answers stay strictly grounded in retrieved documents.
* **Modular Experimentation:** Easily test and compare different chunk sizes, overlap ratios, and embedding models.

---

## 🚀 Quickstart

1. **Clone the Repository:**
   `git clone https://github.com/aftabjeelani57-cpu/Rag-eval-ask-my-docs.git`  
   `cd Rag-eval-ask-my-docs`

2. **Install Dependencies:**
   `pip install -r requirements.txt`

3. **Run Pipeline & Evaluation:**
   `python run_eval.py --chunk-size 512 --top-k 5`

---

## 📊 Benchmarks & Strategy Comparison

| Chunk Strategy | Retrieval Method | Reranker | Recall@5 | Faithfulness |
| :--- | :--- | :--- | :--- | :--- |
| Fixed 512 / 50 Overlap | Dense Vector Only | None | 0.72 | 0.81 |
| Fixed 512 / 50 Overlap | **Hybrid (BM25 + Dense)** | **Cross-Encoder** | **0.91** | **0.94** |

---

## 🛠️ Built With

* **Language:** Python 3.10+
* **Retrieval:** Hybrid Search (BM25 + Dense Embeddings)
* **Evaluation:** Retrieval Recall & Faithfulness Harness