"""DocuMind REST API. Run with:  python -m uvicorn app.main:app --reload"""
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.rag.generator import answer_question
from app.rag.ingestion import ingest_pdf, list_documents

app = FastAPI(
    title="DocuMind API",
    description="AI-powered research paper assistant using RAG",
    version="0.2.0",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    k: int = Field(default=5, ge=1, le=20)


@app.get("/")
def root():
    return {"message": "Welcome to DocuMind!", "status": "running"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "DocuMind API"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    chunks = ingest_pdf(await file.read(), file.filename)
    if chunks == 0:
        raise HTTPException(status_code=422, detail="No text found in this PDF.")
    return {"filename": file.filename, "chunks_indexed": chunks}


@app.get("/documents")
def documents():
    return {"documents": list_documents()}


@app.post("/ask")
def ask(request: AskRequest):
    return answer_question(request.question, request.k)
