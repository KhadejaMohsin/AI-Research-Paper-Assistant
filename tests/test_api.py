"""Run with:  python -m pytest -v
These tests don't call OpenAI, so they're free and work without an API key.
"""
from fastapi.testclient import TestClient

from app.main import app
from app.rag.ingestion import chunk_text

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_ask_rejects_empty_question():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_chunks_overlap_and_cover_text():
    text = "".join(str(i % 10) for i in range(3000))
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    assert all(len(c) <= 1000 for c in chunks)
    assert chunks[0][-200:] == chunks[1][:200]  # overlap is preserved
    assert chunks[-1].endswith(text[-50:])  # nothing lost at the end
