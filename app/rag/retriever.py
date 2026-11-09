"""Step 2 of RAG: find the chunks most relevant to a question."""
from app.rag.vectorstore import embed_texts, get_collection


def retrieve(question: str, k: int = 5) -> list[dict]:
    """Embed the question and return the k closest chunks with their sources."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=embed_texts([question]),
        n_results=min(k, collection.count()),
    )
    chunks = []
    for text, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append(
            {
                "text": text,
                "source": meta["source"],
                "page": meta["page"],
                "score": round(1 - distance, 3),  # cosine similarity, higher = closer
            }
        )
    return chunks
