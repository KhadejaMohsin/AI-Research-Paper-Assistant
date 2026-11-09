"""Step 1 of RAG: turn a PDF into searchable chunks.

PDF bytes -> text per page -> overlapping chunks -> embeddings -> stored in ChromaDB
"""
import fitz  # this is PyMuPDF

from app.rag.vectorstore import embed_texts, get_collection


def extract_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for every page that has text."""
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                pages.append((page_number, text))
    return pages


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """Split text into pieces of ~chunk_size characters.

    Each chunk repeats the last `overlap` characters of the previous one, so a
    sentence that falls on a boundary still appears whole in at least one chunk.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks


def ingest_pdf(pdf_bytes: bytes, filename: str) -> int:
    """Index one PDF. Returns how many chunks were stored."""
    ids, documents, metadatas = [], [], []
    for page_number, page_text in extract_pages(pdf_bytes):
        for i, chunk in enumerate(chunk_text(page_text)):
            ids.append(f"{filename}::p{page_number}::c{i}")
            documents.append(chunk)
            metadatas.append({"source": filename, "page": page_number})

    if not documents:
        return 0  # e.g. a scanned PDF with no text layer

    get_collection().upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embed_texts(documents),
    )
    return len(documents)


def list_documents() -> list[str]:
    """Names of all PDFs currently in the index."""
    result = get_collection().get(include=["metadatas"])
    return sorted({m["source"] for m in result["metadatas"]})
