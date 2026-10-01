# Ask My PDF — RAG Chatbot (from scratch)

Chat with one or many PDFs. Answers are grounded strictly in the documents, with document + page citations.

## Architecture
PDF (PyMuPDF) → sentence-aware chunking (800 chars, 150 overlap) → BAAI/bge-small-en-v1.5 embeddings
→ FAISS (semantic) + BM25 (keyword) → Reciprocal Rank Fusion → Gemini / local Ollama LLM

## Features
- Hybrid retrieval (semantic + keyword), multi-document support
- Page-level citations for every answer
- Out-of-scope guard: relevance cutoff + strict prompt → refuses instead of hallucinating
- Multi-turn conversational memory
- Fully offline mode via Ollama (llama3.2:3b)

## Run
```bash
pip install -r requirements.txt
streamlit run app.py          # paste a free Gemini key from aistudio.google.com in the sidebar
```
Offline: `ollama pull llama3.2:3b`, then choose "ollama (offline)" in the sidebar.
Docker: `docker build -t askpdf . && docker run -p 8501:8501 askpdf`

## Evaluation (add your numbers)
Write 15-20 Q&A pairs from a sample PDF; report retrieval hit-rate@4 and how many out-of-scope
questions were correctly refused.

## Future work
Cross-encoder reranking (bge-reranker), HyDE, Celery+Redis async ingestion, multimodal tables/images.
