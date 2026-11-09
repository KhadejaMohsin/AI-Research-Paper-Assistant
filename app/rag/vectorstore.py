"""Shared helpers: the OpenAI client, embeddings, and the ChromaDB collection.

Both ingestion (writing) and retrieval (reading) use these, so they live here.
"""
from functools import lru_cache

import chromadb
from openai import OpenAI

from app import config


@lru_cache
def get_openai_client() -> OpenAI:
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=config.OPENAI_API_KEY)


@lru_cache
def get_collection():
    """Open (or create) the on-disk vector database and return our collection."""
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    # cosine distance is the usual choice for OpenAI embeddings
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Turn a list of strings into embedding vectors, a batch at a time."""
    client = get_openai_client()
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors
