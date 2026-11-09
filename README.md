# DocuMind

A RAG-powered research assistant: upload PDFs, ask questions, get answers that cite the page they came from.

## How it works

1. **Ingestion** (`app/rag/ingestion.py`): PDF → text per page → overlapping chunks → OpenAI embeddings → ChromaDB
2. **Retrieval** (`app/rag/retriever.py`): embed the question, find the most similar chunks
3. **Generation** (`app/rag/generator.py`): send the chunks plus a system prompt (`app/prompts/system_v1.txt`) to the LLM for a cited answer

The same pipeline is exposed two ways: a Streamlit web app (`frontend/`) and a FastAPI REST API (`app/main.py`).

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then put your OpenAI key in .env

streamlit run frontend/streamlit_app.py      # web app
python -m uvicorn app.main:app --reload      # API, docs at http://127.0.0.1:8000/docs
python -m pytest -v                          # tests
```
