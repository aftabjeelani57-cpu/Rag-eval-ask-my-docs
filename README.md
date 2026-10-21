# ⚡ RAG Eval: Enterprise Hybrid Retrieval & Evaluation Harness

![RAG Evaluation CI](https://github.com/aftabjeelani57-cpu/Rag-eval-ask-my-docs/actions/workflows/ci.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)

A production-grade Retrieval-Augmented Generation (RAG) pipeline paired with a quantitative evaluation harness to systematically measure and optimize chunking strategies, retrieval recall, and cross-encoder reranking precision.

---

## 📊 Benchmark Metrics

| Metric | Score | Dataset Size | Reranker Model |
| :--- | :--- | :--- | :--- |
| **Recall@1** | **100%** | 3 Test Cases | `ms-marco-MiniLM-L-6-v2` |
| **Hybrid Search Precision** | **High** | Multi-chunk Corpus | `all-MiniLM-L6-v2` + BM25 |

---

## 🏗 Architecture Overview

```text
[ Raw Documents ]
       │
       ▼
[ Overlapping Chunking Strategy ]
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
[ BM25 Sparse Index ]          [ ChromaDB Vector Store ]
       │                                 │
       └─────────────────┬───────────────┘
                         ▼
             [ Hybrid Retrieval Layer ]
                         │
                         ▼
        [ Cross-Encoder Precision Reranker ]
                         │
                         ▼
            [ LLM Context Grounding ]
                         │
                         ▼
           [ Automated Recall Evaluation ]
📂 Project Structure
Plaintext
rag-eval/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI Workflow
├── data/
│   ├── sample_docs.json       # Sample document corpus
│   └── eval_dataset.json      # Ground truth evaluation dataset
├── eval/
│   └── run_eval.py            # Quantitative evaluation harness (Recall@K)
├── src/
│   ├── chunking.py            # Document chunking module
│   ├── retriever.py           # Hybrid search (BM25 + ChromaDB)
│   ├── reranker.py            # Cross-encoder precision reranking
│   └── generator.py           # LLM answer generation & context grounding
├── tests/
│   └── test_pipeline.py       # Pytest unit testing suite
├── pytest.ini                 # Pytest configuration
└── requirements.txt           # Python dependencies
🚀 Quickstart & Usage
1. Install Dependencies
Bash
pip install -r requirements.txt
2. Run Unit Tests
Bash
python -m pytest
3. Run RAG Evaluation Benchmark
Bash
python -m eval.run_eval
4. Test Interactive Generator
Bash
python -m src.generator

4. Press **`Ctrl + S`** to save the file.

---

### Step 2: Push the Clean README to GitHub

Run this command in your PowerShell terminal:

```powershell
git add README.md
git commit -m "Fix README markdown section header rendering"
git push origin main