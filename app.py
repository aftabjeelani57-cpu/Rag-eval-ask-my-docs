import streamlit as st
import json
from pypdf import PdfReader
from src.chunking import process_documents
from src.retriever import HybridRetriever
from src.reranker import Reranker
from src.generator import LLMGenerator

st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide", page_icon="⚡")

st.title("⚡ RAG Pipeline Interactive Dashboard")
st.markdown("Query your knowledge base using **Hybrid Search (BM25 + Vector)** and **Cross-Encoder Reranking**.")

# Sidebar Document Source Selection
st.sidebar.header("📁 Document Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload a custom PDF document", type=["pdf"])

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    extracted_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"
    return extracted_text

def build_pipeline(docs):
    chunks = process_documents(docs, chunk_size=150, overlap=30)
    retriever = HybridRetriever()
    retriever.index_chunks(chunks)
    reranker = Reranker()
    generator = LLMGenerator()
    return retriever, reranker, generator

if uploaded_file is not None:
    st.sidebar.success(f"Loaded PDF: {uploaded_file.name}")
    pdf_text = extract_text_from_pdf(uploaded_file)
    docs = [{"doc_id": uploaded_file.name, "text": pdf_text}]
    retriever, reranker, generator = build_pipeline(docs)
else:
    st.sidebar.info("Using default `data/sample_docs.json` database.")
    with open("data/sample_docs.json", "r") as f:
        docs = json.load(f)
    retriever, reranker, generator = build_pipeline(docs)

query = st.text_input("Enter your question:", "What is RAG and why is grounding important?")
top_k = st.slider("Number of hybrid search candidates (Top K)", 1, 5, 3)

if st.button("Run RAG Pipeline", type="primary"):
    with st.spinner("Processing query..."):
        retrieved_chunks = retriever.search_hybrid(query, top_k=top_k)
        reranked_chunks = reranker.rerank(query, retrieved_chunks, top_n=1)
        answer = generator.generate_answer(query, reranked_chunks)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 Grounded LLM Answer")
        st.success(answer)

    with col2:
        st.subheader("🔍 Top Reranked Context Chunk")
        if reranked_chunks:
            st.info(reranked_chunks[0]["text"])
        else:
            st.warning("No relevant chunk found.")

    with st.expander("Show Raw Candidate Chunks from Hybrid Search"):
        st.json(retrieved_chunks)
